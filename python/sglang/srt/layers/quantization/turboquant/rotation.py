"""
Random rotation matrix generation and application for TurboQuant.

Implements the random rotation step from TurboQuant paper (Algorithm 1, Setup):
    "Generate random rotation matrix Pi via QR decomposition
     of a random matrix with i.i.d. Normal entries."

The rotation makes each coordinate of Pi*x follow the same marginal distribution
(Lemma 1) and makes distinct coordinates nearly independent in high dimensions,
which justifies applying independent scalar quantizers per coordinate.

PAPER FIDELITY:
- QR decomposition of Gaussian matrix: exact match to paper Algorithm 1
- Deterministic from seed: paper says "shared randomness" — we implement via seed

ENGINEERING ASSUMPTION:
- One rotation matrix per head_dim value, shared across all layers and heads.
  Paper does not specify per-head vs global; sharing is reasonable because the
  rotation's purpose (inducing near-independence) depends only on d, not on
  which head or layer the vector comes from.
"""

from typing import Dict, Tuple

import torch

# Module-level cache: (d, seed) -> rotation matrix on each device
_rotation_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}


def get_rotation_matrix(
    d: int,
    seed: int = 42,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate a Haar-uniform random orthogonal matrix of size d x d.

    Uses QR decomposition of a d x d matrix with i.i.d. N(0,1) entries,
    as specified in the paper. The result is deterministic given the seed.

    To ensure a proper Haar-uniform distribution (not just orthogonal),
    we follow the standard correction: multiply by diag(sign(diag(R)))
    to make the decomposition unique.

    Args:
        d: matrix dimension (= head_dim)
        seed: random seed for reproducibility
        device: target device

    Returns:
        Orthogonal matrix of shape (d, d), dtype float32
    """
    cache_key = (d, seed, str(device))
    if cache_key in _rotation_cache:
        return _rotation_cache[cache_key]

    # Generate on CPU for reproducibility, then move to device
    gen = torch.Generator()
    gen.manual_seed(seed)
    gaussian = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(gaussian)

    # Haar correction: Q * diag(sign(diag(R))) gives a proper Haar-uniform sample
    # Reference: Mezzadri (2007), "How to generate random matrices from the
    # classical compact groups"
    diag_sign = torch.sign(torch.diag(R))
    Q = Q * diag_sign.unsqueeze(0)  # broadcast multiply columns

    Q = Q.to(device=device)
    _rotation_cache[cache_key] = Q
    return Q


def rotate(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Apply rotation: y = x @ R^T (equivalent to Pi * x in the paper).

    On CUDA, operates in the input's native dtype (bf16/fp16) to leverage
    tensor cores. On CPU, uses float32 for correctness (no bf16 hardware).
    The quantization step (bucketize) that follows is far coarser than the
    bf16 rounding error, so the dtype choice does not affect quality.

    Args:
        x: tensor of shape (..., d), any dtype
        R: rotation matrix of shape (d, d), float32

    Returns:
        Rotated tensor of shape (..., d)
    """
    if x.is_cuda and x.dtype in (torch.bfloat16, torch.float16):
        R_cast = R.to(dtype=x.dtype)
        return torch.matmul(x, R_cast.t())
    x_f32 = x.float()
    return torch.matmul(x_f32, R.t())


def unrotate(y: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Apply inverse rotation: x_hat = y @ R (equivalent to Pi^T * y_hat in the paper).

    Since R is orthogonal, R^{-1} = R^T, so unrotate(y) = y @ R.

    Args:
        y: tensor of shape (..., d)
        R: rotation matrix of shape (d, d), float32

    Returns:
        Unrotated tensor of shape (..., d)
    """
    if y.is_cuda and y.dtype in (torch.bfloat16, torch.float16):
        R_cast = R.to(dtype=y.dtype)
        return _safe_matmul(y, R_cast)
    return _safe_matmul(y.float(), R)


def _safe_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matmul with chunking to avoid CUBLAS int32 overflow on large tensors."""
    shape = a.shape
    a_2d = a.reshape(-1, shape[-1])
    total_rows = a_2d.shape[0]
    _CHUNK = 1 << 22  # 4M rows
    if total_rows <= _CHUNK:
        return torch.matmul(a, b)
    out_2d = torch.empty(total_rows, b.shape[-1], dtype=a.dtype, device=a.device)
    for start in range(0, total_rows, _CHUNK):
        end = min(start + _CHUNK, total_rows)
        torch.matmul(a_2d[start:end], b, out=out_2d[start:end])
    return out_2d.reshape(*shape[:-1], b.shape[-1])
