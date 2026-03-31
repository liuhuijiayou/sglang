"""
Optimal scalar quantizer codebook construction via Lloyd-Max algorithm.

Implements the continuous 1D k-means problem from TurboQuant paper (Equation 4):

    C(f_X, b) = min_{c_1,...,c_{2^b}} sum_i integral |x - c_i|^2 f_X(x) dx

where f_X is the coordinate density of a uniformly random point on S^{d-1}
after applying a Haar-random rotation.

PAPER FIDELITY:
- Uses the EXACT Beta-derived density from Lemma 1:
      f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
  for x in [-1, 1].  No Gaussian approximation.
- Lloyd-Max iteration on this exact density: faithful to paper Section 3.2.
- Numerical centroids for b=1: match paper's closed-form ±sqrt(2/(pi*d)).
"""

import functools
import math
from typing import Tuple

import torch


def _beta_pdf(x: torch.Tensor, d: int) -> torch.Tensor:
    """Exact coordinate density on S^{d-1} from Lemma 1 of the paper.

    f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)

    for x in [-1, 1].  Outside this range, f_X = 0.

    Uses log-gamma for numerical stability with large d.
    """
    # Normalization constant in log-space:
    #   log C = lgamma(d/2) - 0.5*log(pi) - lgamma((d-1)/2)
    log_C = math.lgamma(d / 2) - 0.5 * math.log(math.pi) - math.lgamma((d - 1) / 2)

    # (1 - x^2)^((d-3)/2) in log-space for stability
    exponent = (d - 3) / 2.0
    # Clamp to avoid log(0) at boundaries x = ±1
    one_minus_x2 = (1.0 - x * x).clamp(min=1e-30)
    log_body = exponent * torch.log(one_minus_x2)

    pdf = torch.exp(log_C + log_body)
    # Zero out values outside [-1, 1]
    pdf = pdf * ((x > -1.0) & (x < 1.0)).float()
    return pdf


def _lloyd_max_step(
    centroids: torch.Tensor,
    grid: torch.Tensor,
    pdf_values: torch.Tensor,
    dx: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One iteration of Lloyd-Max: update boundaries then centroids.

    The Voronoi boundaries are midpoints between adjacent centroids (paper Eq. 4).
    Centroids are updated to the conditional expectation within each Voronoi cell.
    """
    num_levels = centroids.shape[0]

    # Boundaries = midpoints between consecutive centroids
    boundaries = 0.5 * (centroids[:-1] + centroids[1:])

    # Compute new centroids as E[X | X in cell_i]
    new_centroids = torch.zeros_like(centroids)
    for i in range(num_levels):
        lo = boundaries[i - 1] if i > 0 else grid[0]
        hi = boundaries[i] if i < num_levels - 1 else grid[-1]

        mask = (grid >= lo) & (grid < hi)
        weighted = grid * pdf_values * mask.float()
        weight = pdf_values * mask.float()

        total_weight = weight.sum() * dx
        total_weighted = weighted.sum() * dx

        if total_weight > 1e-15:
            new_centroids[i] = total_weighted / total_weight
        else:
            new_centroids[i] = centroids[i]

    return new_centroids, boundaries


def lloyd_max_solve(
    d: int,
    num_levels: int,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Solve the optimal scalar quantizer using the exact coordinate density.

    Uses the exact Beta-derived density f_X from Lemma 1, supported on [-1, 1].

    Args:
        d: ambient dimension (= head_dim)
        num_levels: number of quantization levels (= 2^b)
        max_iter: maximum Lloyd-Max iterations
        tol: convergence tolerance on centroid movement

    Returns:
        (centroids, boundaries, distortion)
    """
    # Integration grid on [-1, 1] (the support of f_X)
    num_points = 8192
    grid = torch.linspace(-1.0 + 1e-7, 1.0 - 1e-7, num_points)
    dx = (2.0 - 2e-7) / (num_points - 1)
    pdf_values = _beta_pdf(grid, d)

    # For large d, the density is concentrated around 0 with std ~ 1/sqrt(d).
    # Initialize centroids using quantiles of the density.
    # Compute CDF numerically
    cdf = torch.cumsum(pdf_values * dx, dim=0)
    cdf = cdf / cdf[-1]  # normalize to [0, 1]

    if num_levels == 2:
        # Paper b=1: optimal centroids are ±sqrt(2/(pi*d))
        c = math.sqrt(2.0 / (math.pi * d))
        centroids = torch.tensor([-c, c])
    else:
        # Initialize at CDF quantiles
        target_probs = torch.linspace(
            0.5 / num_levels, 1.0 - 0.5 / num_levels, num_levels
        )
        centroid_indices = torch.searchsorted(cdf, target_probs)
        centroid_indices = centroid_indices.clamp(0, num_points - 1)
        centroids = grid[centroid_indices]

    # Lloyd-Max iteration
    for _ in range(max_iter):
        new_centroids, boundaries = _lloyd_max_step(centroids, grid, pdf_values, dx)
        movement = (new_centroids - centroids).abs().max().item()
        centroids = new_centroids
        if movement < tol:
            break

    # Final boundaries
    boundaries = 0.5 * (centroids[:-1] + centroids[1:])

    # Compute distortion: E[(X - Q(X))^2]
    distortion = 0.0
    for i in range(num_levels):
        lo = boundaries[i - 1] if i > 0 else grid[0]
        hi = boundaries[i] if i < num_levels - 1 else grid[-1]
        mask = (grid >= lo) & (grid < hi)
        sq_err = (grid - centroids[i]) ** 2 * pdf_values * mask.float()
        distortion += sq_err.sum().item() * dx

    return centroids, boundaries, distortion


@functools.lru_cache(maxsize=32)
def get_codebook(
    d: int,
    bits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get precomputed (centroids, boundaries) for given dimension and bit-width.

    Results are cached so Lloyd-Max only runs once per (d, bits) pair.
    Returned tensors are on CPU; caller should move to target device.

    Args:
        d: dimension (= head_dim, typically 128)
        bits: quantization bit-width (1, 2, 3, or 4)

    Returns:
        (centroids, boundaries) both as float32 CPU tensors
    """
    num_levels = 1 << bits
    centroids, boundaries, _distortion = lloyd_max_solve(d, num_levels)
    return centroids, boundaries
