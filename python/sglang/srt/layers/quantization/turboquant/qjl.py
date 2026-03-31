"""
Quantized Johnson-Lindenstrauss (QJL) 1-bit projection.

Implements Definition 1 and Lemma 4 from the TurboQuant paper:

  Quantize:   Q_qjl(x) = sign(S · x)     where S has i.i.d. N(0,1) entries
  Dequantize: Q_qjl^{-1}(z) = sqrt(pi/2) / d · S^T · z

Properties (Lemma 4, for x on S^{d-1}):
  - Unbiased: E[<y, Q_qjl^{-1}(Q_qjl(x))>] = <y, x>
  - Variance:  Var(...) <= (pi / 2d) · ||y||^2

Used as the second stage of TurboQuant_prod (Algorithm 2) to correct the
bias introduced by the MSE quantizer in inner product estimation.
"""

import math
from typing import Dict, Tuple

import torch

# Cache for QJL projection matrices: (d, seed, device_str) -> Tensor
_qjl_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}


def get_qjl_matrix(
    d: int,
    seed: int = 137,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate a d x d random Gaussian projection matrix for QJL.

    The matrix has i.i.d. N(0, 1) entries. Deterministic from seed.
    NOTE: Uses a different default seed (137) from the rotation matrix (42)
    to ensure S and Pi are independent.

    Args:
        d: dimension
        seed: random seed
        device: target device

    Returns:
        S: shape (d, d), dtype float32
    """
    cache_key = (d, seed, str(device))
    if cache_key in _qjl_cache:
        return _qjl_cache[cache_key]

    gen = torch.Generator()
    gen.manual_seed(seed)
    S = torch.randn(d, d, generator=gen).to(device=device)
    _qjl_cache[cache_key] = S
    return S


def qjl_quantize(x: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """QJL quantization: sign(S · x).

    Args:
        x: (..., d) float tensor (does NOT need to be unit-norm;
           sign is scale-invariant)
        S: (d, d) projection matrix

    Returns:
        sign_bits: (..., d) uint8 with values {0, 1}
            0 = negative, 1 = positive (for bit-packing compatibility)
    """
    # y = x @ S^T  (equivalent to S @ x for row vectors)
    y = torch.matmul(x.float(), S.t())
    # Convert sign to {0, 1}: positive -> 1, non-positive -> 0
    return (y > 0).to(torch.uint8)


def qjl_dequantize(
    sign_bits: torch.Tensor,
    S: torch.Tensor,
    residual_norm: torch.Tensor,
    d: int,
) -> torch.Tensor:
    """QJL dequantization: gamma * sqrt(pi/2) / d * S^T * z.

    Args:
        sign_bits: (..., d) uint8 with values {0, 1}
        S: (d, d) projection matrix
        residual_norm: (...,) float tensor (gamma = ||r||)
        d: dimension

    Returns:
        x_hat_qjl: (..., d) float32
    """
    # Convert {0, 1} back to {-1, +1}
    z = (2.0 * sign_bits.float() - 1.0)  # (..., d)

    # sqrt(pi/2) / d * S^T * z
    scale = math.sqrt(math.pi / 2.0) / d
    # Chunk matmul to avoid CUBLAS int32 overflow on large KV pools
    shape = z.shape
    z_2d = z.reshape(-1, shape[-1])
    total_rows = z_2d.shape[0]
    _CHUNK = 1 << 22  # 4M rows
    if total_rows <= _CHUNK:
        x_hat = scale * torch.matmul(z, S)
    else:
        out_2d = torch.empty(total_rows, S.shape[-1], dtype=z.dtype, device=z.device)
        for start in range(0, total_rows, _CHUNK):
            end = min(start + _CHUNK, total_rows)
            torch.matmul(z_2d[start:end], S, out=out_2d[start:end])
        x_hat = scale * out_2d.reshape(*shape[:-1], S.shape[-1])

    # Scale by residual norm
    x_hat = x_hat * residual_norm.float().unsqueeze(-1)

    return x_hat
