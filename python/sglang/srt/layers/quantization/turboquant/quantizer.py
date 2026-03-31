"""
TurboQuant quantizer implementations.

Contains three quantizer classes:

1. TurboQuantMSE — Algorithm 1 from the paper.
   Rotation + optimal scalar quantization. Minimizes MSE.

2. TurboQuantProd — Algorithm 2 from the paper.
   MSE at (b-1) bits + QJL on residual. Provides UNBIASED inner product
   estimation with distortion D_prod <= (sqrt(3)*pi^2/d) * 4^{-b} * ||y||^2.

3. TurboQuantMixed — Non-integer bit-width via channel splitting.
   Splits head_dim into two groups with different bit-widths to achieve
   fractional average bits (e.g., 2.5-bit, 3.5-bit). Paper Section 6.4.

All quantizers expose a uniform interface:
    .packed_dim         — bytes of packed data per vector
    .has_residual_norm  — whether compress() returns a second norm
    .compress(x) -> (packed, norms, [residual_norms])
    .decompress(packed, norms, [residual_norms]) -> x_hat
"""

import math
from typing import Optional, Tuple, Union

import torch

from sglang.srt.layers.quantization.turboquant.codebook import get_codebook
from sglang.srt.layers.quantization.turboquant.packing import (
    pack_indices,
    packed_dim,
    unpack_indices,
)
from sglang.srt.layers.quantization.turboquant.qjl import (
    get_qjl_matrix,
    qjl_dequantize,
    qjl_quantize,
)
from sglang.srt.layers.quantization.turboquant.rotation import (
    get_rotation_matrix,
    rotate,
    unrotate,
)

# Triton ops: available on CUDA with triton installed
_triton_available = False
try:
    from sglang.srt.layers.quantization.turboquant.triton_ops import (
        triton_dequant,
        triton_dequant_no_norm,
        triton_quant_pack,
    )
    _triton_available = True
except (ImportError, ModuleNotFoundError):
    pass


# ============================================================================
# TurboQuantMSE — Algorithm 1
# ============================================================================

class TurboQuantMSE:
    """TurboQuant MSE quantizer (Algorithm 1).

    Compresses d-dimensional vectors to b bits per coordinate plus a scalar norm.
    Achieves MSE <= (sqrt(3)*pi/2) * 4^{-b} on unit-norm vectors (Theorem 1).
    """

    has_residual_norm = False

    def __init__(self, head_dim: int, bits: int, seed: int = 42, device: str = "cpu"):
        if bits < 1 or bits > 8:
            raise ValueError(f"bits must be in [1, 8], got {bits}")
        self.head_dim = head_dim
        self.bits = bits
        self.device = device
        self.packed_dim = packed_dim(head_dim, bits)

        self.R = get_rotation_matrix(head_dim, seed=seed, device=device)
        centroids_cpu, boundaries_cpu = get_codebook(head_dim, bits)
        self.centroids = centroids_cpu.to(device=device)
        self.boundaries = boundaries_cpu.to(device=device)

    def compress(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize and bit-pack."""
        norms = x.float().norm(dim=-1)
        x_unit = x / norms.to(dtype=x.dtype).unsqueeze(-1).clamp(min=1e-12)
        y = rotate(x_unit, self.R)
        # Fused quantize + pack on CUDA via Triton
        if _triton_available and y.is_cuda and self.bits in (2, 3, 4):
            packed = triton_quant_pack(y, self.boundaries, self.bits, self.head_dim)
        else:
            y_f = y.float() if y.dtype != self.boundaries.dtype else y
            indices = torch.bucketize(y_f, self.boundaries).to(torch.uint8)
            packed = pack_indices(indices, self.bits)
        return packed, norms.to(torch.float16)

    def decompress(
        self,
        packed: torch.Tensor,
        norms: torch.Tensor,
        residual_norms: Optional[torch.Tensor] = None,
        out_dtype: torch.dtype = torch.bfloat16,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Unpack, lookup, norm-scale, unrotate.

        On CUDA with Triton: fuses unpack + gather + norm_scale into ONE kernel,
        then does rotation matmul via cuBLAS. Total: 1 Triton + 1 matmul.
        Eliminates two intermediate full-size tensors (indices, y_hat).
        """
        if _triton_available and packed.is_cuda and self.bits in (2, 3, 4):
            # Fused kernel: unpack + centroid_lookup + norm_scale → bf16
            y_hat_scaled = triton_dequant(
                packed, self.centroids, norms, self.bits, self.head_dim
            )
            # Rotation matmul: write directly into output buffer if provided
            if output is not None:
                R_cast = self.R.to(dtype=y_hat_scaled.dtype)
                shape = y_hat_scaled.shape
                src_2d = y_hat_scaled.reshape(-1, shape[-1])
                dst_2d = output.reshape(-1, shape[-1])
                total_rows = src_2d.shape[0]
                # Chunk to avoid CUBLAS int32 overflow on large KV pools
                # (e.g. 10M tokens × 4 heads = 40M rows exceeds CUBLAS limits)
                _CHUNK = 1 << 22  # 4M rows per chunk — safe margin
                if total_rows <= _CHUNK:
                    torch.matmul(src_2d, R_cast, out=dst_2d)
                else:
                    for start in range(0, total_rows, _CHUNK):
                        end = min(start + _CHUNK, total_rows)
                        torch.matmul(
                            src_2d[start:end], R_cast, out=dst_2d[start:end]
                        )
                return output
            x_hat = unrotate(y_hat_scaled, self.R)
        else:
            # CPU fallback: separate ops
            indices = unpack_indices(packed, self.bits, self.head_dim)
            y_hat = self.centroids[indices.long()]
            if y_hat.is_cuda and out_dtype in (torch.bfloat16, torch.float16):
                y_hat = y_hat.to(out_dtype)
            x_hat = unrotate(y_hat, self.R)
            x_hat = x_hat * norms.to(dtype=x_hat.dtype).unsqueeze(-1)

        if output is not None:
            output.copy_(x_hat)
            return output
        return x_hat.to(out_dtype)


# ============================================================================
# TurboQuantProd — Algorithm 2
# ============================================================================

class TurboQuantProd:
    """TurboQuant inner-product quantizer (Algorithm 2).

    Two-stage pipeline:
      1. MSE quantizer at (b-1) bits → coarse reconstruction
      2. QJL (1-bit sign projection) on the residual → bias correction

    Provides UNBIASED inner product estimation (Theorem 2):
      E[<y, x_hat>] = <y, x>

    Total storage: (b-1) bits MSE indices + 1 bit QJL signs + 2 norms = b*d + O(1).

    PAPER FIDELITY: Exact match to Algorithm 2.
    """

    has_residual_norm = True

    def __init__(self, head_dim: int, bits: int, seed: int = 42, device: str = "cpu"):
        if bits < 1 or bits > 8:
            raise ValueError(f"bits must be in [1, 8], got {bits}")
        self.head_dim = head_dim
        self.bits = bits
        self.device = device

        # Stage 1: MSE quantizer at (b-1) bits
        # For b=1, mse_bits=0 means no MSE stage (pure QJL)
        self.mse_bits = bits - 1
        if self.mse_bits > 0:
            self.mse = TurboQuantMSE(
                head_dim=head_dim, bits=self.mse_bits, seed=seed, device=device
            )
            self._mse_packed_dim = packed_dim(head_dim, self.mse_bits)
        else:
            self.mse = None
            self._mse_packed_dim = 0

        # Stage 2: QJL (1-bit)
        self._qjl_packed_dim = packed_dim(head_dim, 1)  # d/8 bytes
        # Use a different seed for S to ensure independence from Pi
        self.S = get_qjl_matrix(head_dim, seed=seed + 1000, device=device)

        # Total packed dim = MSE packed + QJL packed
        self.packed_dim = self._mse_packed_dim + self._qjl_packed_dim

    def compress(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize via two-stage pipeline.

        Returns:
            packed:          (..., packed_dim) uint8 — concat of [mse_packed, qjl_packed]
            norms:           (...,) float16 — input L2 norms
            residual_norms:  (...,) float16 — residual L2 norms
        """
        input_norms = x.float().norm(dim=-1)

        if self.mse is not None:
            # Stage 1: MSE quantize at (b-1) bits
            mse_packed, mse_norms = self.mse.compress(x)
            # Reconstruct for residual computation
            x_hat_mse = self.mse.decompress(mse_packed, mse_norms, out_dtype=torch.float32)
        else:
            # b=1: no MSE stage, reconstruction is zero
            mse_packed = None
            x_hat_mse = torch.zeros_like(x, dtype=torch.float32)

        # Residual (paper Algorithm 2, line 2)
        residual = x.float() - x_hat_mse
        residual_norms = residual.norm(dim=-1)  # (...,)

        # Stage 2: QJL on residual (paper Algorithm 2, line 3)
        # sign(S · r) — sign is scale-invariant, so no need to normalize
        qjl_bits = qjl_quantize(residual, self.S)  # (..., d) uint8 {0,1}
        qjl_packed = pack_indices(qjl_bits, 1)  # (..., d/8)

        # Concatenate packed data
        if mse_packed is not None:
            packed = torch.cat([mse_packed, qjl_packed], dim=-1)
        else:
            packed = qjl_packed

        return packed, input_norms.to(torch.float16), residual_norms.to(torch.float16)

    def decompress(
        self,
        packed: torch.Tensor,
        norms: torch.Tensor,
        residual_norms: Optional[torch.Tensor] = None,
        out_dtype: torch.dtype = torch.bfloat16,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reconstruct: x_hat = x_hat_mse + gamma * sqrt(pi/2)/d * S^T * z."""
        assert residual_norms is not None, "TurboQuantProd requires residual_norms"

        # Split packed data
        if self.mse is not None:
            mse_packed = packed[..., :self._mse_packed_dim]
            qjl_packed = packed[..., self._mse_packed_dim:]
            # MSE reconstruction
            x_hat_mse = self.mse.decompress(mse_packed, norms, out_dtype=torch.float32)
        else:
            qjl_packed = packed
            x_hat_mse = torch.zeros(
                *packed.shape[:-1], self.head_dim,
                dtype=torch.float32, device=packed.device
            )

        # QJL reconstruction (paper Algorithm 2, DeQuant line 2-3)
        qjl_bits = unpack_indices(qjl_packed, 1, self.head_dim)
        x_hat_qjl = qjl_dequantize(qjl_bits, self.S, residual_norms, self.head_dim)

        x_hat = x_hat_mse + x_hat_qjl
        if output is not None:
            output.copy_(x_hat.to(out_dtype))
            return output
        return x_hat.to(out_dtype)


# ============================================================================
# TurboQuantMixed — Non-integer bit-width (Section 6.4)
# ============================================================================

class TurboQuantMixed:
    """Non-integer bit-width quantizer via channel splitting.

    Splits the head_dim coordinates into two groups after rotation,
    quantizing each group at a different integer bit-width to achieve
    a fractional average bit-width.

    Paper Section 6.4: "Channels are split into outlier and non-outlier sets.
    Two independent TurboQuant instances are applied with different bit-widths."

    Example for 2.5-bit, head_dim=128:
        64 channels at 3-bit + 64 channels at 2-bit
        average = (64*3 + 64*2) / 128 = 2.5

    ENGINEERING ASSUMPTION:
    After rotation, all coordinates are (nearly) identically distributed,
    so which channels get higher bits doesn't matter. We assign the first
    n_high channels to the higher bit-width. This is data-oblivious and
    consistent with TurboQuant's design philosophy.
    """

    has_residual_norm = False

    def __init__(
        self, head_dim: int, target_bits: float, seed: int = 42, device: str = "cpu"
    ):
        self.head_dim = head_dim
        self.target_bits = target_bits
        self.device = device

        # Compute channel split
        self.bits_low = int(math.floor(target_bits))
        self.bits_high = self.bits_low + 1
        # n_high * bits_high + n_low * bits_low = target_bits * head_dim
        self.n_high = int(round((target_bits - self.bits_low) * head_dim))
        self.n_low = head_dim - self.n_high

        if self.n_high <= 0 or self.n_low <= 0:
            raise ValueError(
                f"target_bits={target_bits} with head_dim={head_dim} gives "
                f"n_high={self.n_high}, n_low={self.n_low}. Need both > 0."
            )

        # Verify the split achieves the target
        effective_bits = (self.n_high * self.bits_high + self.n_low * self.bits_low) / head_dim
        assert abs(effective_bits - target_bits) < 0.01, (
            f"Split mismatch: effective={effective_bits}, target={target_bits}"
        )

        # Shared rotation for the full vector
        self.R = get_rotation_matrix(head_dim, seed=seed, device=device)

        # Separate codebooks for each group (different bit-widths, but same d
        # since the coordinate density only depends on the ambient dimension d,
        # not the sub-group size — Lemma 1 applies to each coordinate of Pi*x)
        c_high, b_high = get_codebook(head_dim, self.bits_high)
        c_low, b_low = get_codebook(head_dim, self.bits_low)
        self.centroids_high = c_high.to(device=device)
        self.boundaries_high = b_high.to(device=device)
        self.centroids_low = c_low.to(device=device)
        self.boundaries_low = b_low.to(device=device)

        # Packed dimensions for each group
        self._pd_high = packed_dim(self.n_high, self.bits_high)
        self._pd_low = packed_dim(self.n_low, self.bits_low)
        self.packed_dim = self._pd_high + self._pd_low

    def compress(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize with channel-split mixed precision.

        Returns:
            packed: (..., packed_dim) uint8 — concat of [high_packed, low_packed]
            norms:  (...,) float16
        """
        norms = x.float().norm(dim=-1)
        x_unit = x.float() / norms.unsqueeze(-1).clamp(min=1e-12)
        y = rotate(x_unit, self.R)  # (..., head_dim)

        # Split into two groups
        y_high = y[..., :self.n_high]   # (..., n_high)
        y_low = y[..., self.n_high:]    # (..., n_low)

        # Quantize each group with its own codebook
        idx_high = torch.bucketize(y_high, self.boundaries_high).to(torch.uint8)
        idx_low = torch.bucketize(y_low, self.boundaries_low).to(torch.uint8)

        # Pack each group
        packed_high = pack_indices(idx_high, self.bits_high)
        packed_low = pack_indices(idx_low, self.bits_low)

        packed = torch.cat([packed_high, packed_low], dim=-1)
        return packed, norms.to(torch.float16)

    def decompress(
        self,
        packed: torch.Tensor,
        norms: torch.Tensor,
        residual_norms: Optional[torch.Tensor] = None,
        out_dtype: torch.dtype = torch.bfloat16,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reconstruct from mixed-precision packed data."""
        # Split packed data
        packed_high = packed[..., :self._pd_high]
        packed_low = packed[..., self._pd_high:]

        # Unpack and lookup centroids
        idx_high = unpack_indices(packed_high, self.bits_high, self.n_high)
        idx_low = unpack_indices(packed_low, self.bits_low, self.n_low)

        y_hat_high = self.centroids_high[idx_high.long()]
        y_hat_low = self.centroids_low[idx_low.long()]

        # Concatenate and inverse rotate
        y_hat = torch.cat([y_hat_high, y_hat_low], dim=-1)  # (..., head_dim)
        x_hat = unrotate(y_hat, self.R)
        x_hat = x_hat * norms.float().unsqueeze(-1)
        if output is not None:
            output.copy_(x_hat.to(out_dtype))
            return output
        return x_hat.to(out_dtype)


# ============================================================================
# Factory function
# ============================================================================

def create_quantizer(
    head_dim: int,
    bits: float,
    variant: str = "mse",
    seed: int = 42,
    device: str = "cpu",
) -> Union[TurboQuantMSE, TurboQuantProd, TurboQuantMixed]:
    """Create the appropriate quantizer based on bit-width and variant.

    Args:
        head_dim: vector dimension
        bits: bit-width (integer or fractional like 2.5, 3.5)
        variant: "mse" or "prod"
        seed: random seed
        device: target device
    """
    is_integer = bits == int(bits)

    if not is_integer:
        # Non-integer bit-width → MixedPrecision (always MSE internally)
        return TurboQuantMixed(head_dim, target_bits=bits, seed=seed, device=device)
    elif variant == "prod":
        return TurboQuantProd(head_dim, bits=int(bits), seed=seed, device=device)
    else:
        return TurboQuantMSE(head_dim, bits=int(bits), seed=seed, device=device)
