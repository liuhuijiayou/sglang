"""
Triton fused kernels for TurboQuant compress/decompress hot paths.

Two kernel families:

  1. triton_dequant: packed uint8 → centroid_values * norm (bf16)
     Fuses: unpack + centroid gather + per-vector norm scaling.
     Output is PRE-ROTATION centroid values already scaled by norm.
     The caller only needs to do the rotation matmul.

  2. triton_quant_pack: float values → packed uint8
     Fuses: nearest-centroid search + bit packing.

Each bit-width (2, 3, 4) has its own specialized kernel.
Falls back to PyTorch when Triton is unavailable.
"""

import torch
import triton
import triton.language as tl


# =============================================================================
# Dequant kernels: packed uint8 + norms → centroid_values * norm (bf16)
#
# Fuses THREE operations into one kernel pass:
#   1. Unpack b-bit indices from packed bytes
#   2. Gather centroid values from the codebook
#   3. Multiply by per-vector norm
#
# Without fusion, steps 1-2 produce a full-size intermediate tensor, and
# step 3 reads/writes the full output again. The fused kernel reads packed
# bytes + norms, and writes final scaled values in one pass.
#
# Each element = one output coordinate. Grid: ceil(total_elements / BLOCK).
# =============================================================================

@triton.jit
def _dequant_2bit_kernel(
    packed_ptr, output_ptr, centroids_ptr, norms_ptr,
    N_VECS, N_ELEMS,
    D: tl.constexpr,
    PACKED_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < N_ELEMS

    vec_id = off // D
    d_id = off % D

    byte_idx = d_id >> 2
    bit_pos = (d_id & 3) << 1

    raw = tl.load(packed_ptr + vec_id * PACKED_DIM + byte_idx, mask=mask, other=0).to(tl.int32)
    idx = (raw >> bit_pos) & 3

    val = tl.load(centroids_ptr + idx, mask=mask, other=0.0)
    norm = tl.load(norms_ptr + vec_id, mask=(vec_id < N_VECS), other=1.0)
    tl.store(output_ptr + off, (val * norm).to(tl.bfloat16), mask=mask)


@triton.jit
def _dequant_3bit_kernel(
    packed_ptr, output_ptr, centroids_ptr, norms_ptr,
    N_VECS, N_ELEMS,
    D: tl.constexpr,
    PACKED_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < N_ELEMS

    vec_id = off // D
    d_id = off % D

    grp = d_id >> 3
    pos = d_id & 7

    byte_base = vec_id * PACKED_DIM + grp * 3
    b0 = tl.load(packed_ptr + byte_base,     mask=mask, other=0).to(tl.int32)
    b1 = tl.load(packed_ptr + byte_base + 1, mask=mask, other=0).to(tl.int32)
    b2 = tl.load(packed_ptr + byte_base + 2, mask=mask, other=0).to(tl.int32)

    idx = tl.where(pos == 0, b0 & 7,
          tl.where(pos == 1, (b0 >> 3) & 7,
          tl.where(pos == 2, ((b0 >> 6) | (b1 << 2)) & 7,
          tl.where(pos == 3, (b1 >> 1) & 7,
          tl.where(pos == 4, (b1 >> 4) & 7,
          tl.where(pos == 5, ((b1 >> 7) | (b2 << 1)) & 7,
          tl.where(pos == 6, (b2 >> 2) & 7,
                             (b2 >> 5) & 7)))))))

    val = tl.load(centroids_ptr + idx, mask=mask, other=0.0)
    norm = tl.load(norms_ptr + vec_id, mask=(vec_id < N_VECS), other=1.0)
    tl.store(output_ptr + off, (val * norm).to(tl.bfloat16), mask=mask)


@triton.jit
def _dequant_4bit_kernel(
    packed_ptr, output_ptr, centroids_ptr, norms_ptr,
    N_VECS, N_ELEMS,
    D: tl.constexpr,
    PACKED_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < N_ELEMS

    vec_id = off // D
    d_id = off % D

    byte_idx = d_id >> 1
    is_high = d_id & 1

    raw = tl.load(packed_ptr + vec_id * PACKED_DIM + byte_idx, mask=mask, other=0).to(tl.int32)
    idx = tl.where(is_high == 1, (raw >> 4) & 0xF, raw & 0xF)

    val = tl.load(centroids_ptr + idx, mask=mask, other=0.0)
    norm = tl.load(norms_ptr + vec_id, mask=(vec_id < N_VECS), other=1.0)
    tl.store(output_ptr + off, (val * norm).to(tl.bfloat16), mask=mask)


# --- Dequant WITHOUT norm (for intermediate use, e.g., Prod residual) ---

@triton.jit
def _dequant_3bit_no_norm_kernel(
    packed_ptr, output_ptr, centroids_ptr,
    N_ELEMS,
    D: tl.constexpr,
    PACKED_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < N_ELEMS
    vec_id = off // D
    d_id = off % D
    grp = d_id >> 3
    pos = d_id & 7
    byte_base = vec_id * PACKED_DIM + grp * 3
    b0 = tl.load(packed_ptr + byte_base,     mask=mask, other=0).to(tl.int32)
    b1 = tl.load(packed_ptr + byte_base + 1, mask=mask, other=0).to(tl.int32)
    b2 = tl.load(packed_ptr + byte_base + 2, mask=mask, other=0).to(tl.int32)
    idx = tl.where(pos == 0, b0 & 7,
          tl.where(pos == 1, (b0 >> 3) & 7,
          tl.where(pos == 2, ((b0 >> 6) | (b1 << 2)) & 7,
          tl.where(pos == 3, (b1 >> 1) & 7,
          tl.where(pos == 4, (b1 >> 4) & 7,
          tl.where(pos == 5, ((b1 >> 7) | (b2 << 1)) & 7,
          tl.where(pos == 6, (b2 >> 2) & 7,
                             (b2 >> 5) & 7)))))))
    val = tl.load(centroids_ptr + idx, mask=mask, other=0.0)
    tl.store(output_ptr + off, val.to(tl.bfloat16), mask=mask)


# =============================================================================
# Quant+Pack kernels: float values → packed uint8
# =============================================================================

@triton.jit
def _quant_pack_2bit_kernel(
    input_ptr, boundaries_ptr, output_ptr,
    N_GROUPS, D: tl.constexpr, PACKED_DIM: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    goff = pid * BLOCK + tl.arange(0, BLOCK)
    groups_per_vec = D >> 2
    mask = goff < N_GROUPS
    vec_id = goff // groups_per_vec
    grp = goff % groups_per_vec
    in_base = vec_id * D + grp * 4
    bnd0 = tl.load(boundaries_ptr + 0)
    bnd1 = tl.load(boundaries_ptr + 1)
    bnd2 = tl.load(boundaries_ptr + 2)
    packed = tl.zeros([BLOCK], dtype=tl.int32)
    for i in tl.static_range(4):
        v = tl.load(input_ptr + in_base + i, mask=mask, other=0.0)
        idx = (v > bnd0).to(tl.int32) + (v > bnd1).to(tl.int32) + (v > bnd2).to(tl.int32)
        packed = packed | (idx << (i * 2))
    tl.store(output_ptr + vec_id * PACKED_DIM + grp, packed.to(tl.uint8), mask=mask)


@triton.jit
def _quant_pack_3bit_kernel(
    input_ptr, boundaries_ptr, output_ptr,
    N_GROUPS, D: tl.constexpr, PACKED_DIM: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    goff = pid * BLOCK + tl.arange(0, BLOCK)
    groups_per_vec = D >> 3
    mask = goff < N_GROUPS
    vec_id = goff // groups_per_vec
    grp = goff % groups_per_vec
    in_base = vec_id * D + grp * 8
    bnd0 = tl.load(boundaries_ptr + 0)
    bnd1 = tl.load(boundaries_ptr + 1)
    bnd2 = tl.load(boundaries_ptr + 2)
    bnd3 = tl.load(boundaries_ptr + 3)
    bnd4 = tl.load(boundaries_ptr + 4)
    bnd5 = tl.load(boundaries_ptr + 5)
    bnd6 = tl.load(boundaries_ptr + 6)

    # Inline quantize: Triton does not support nested function definitions
    val0 = tl.load(input_ptr + in_base + 0, mask=mask, other=0.0)
    v0 = ((val0 > bnd0).to(tl.int32) + (val0 > bnd1).to(tl.int32) +
          (val0 > bnd2).to(tl.int32) + (val0 > bnd3).to(tl.int32) +
          (val0 > bnd4).to(tl.int32) + (val0 > bnd5).to(tl.int32) +
          (val0 > bnd6).to(tl.int32))
    val1 = tl.load(input_ptr + in_base + 1, mask=mask, other=0.0)
    v1 = ((val1 > bnd0).to(tl.int32) + (val1 > bnd1).to(tl.int32) +
          (val1 > bnd2).to(tl.int32) + (val1 > bnd3).to(tl.int32) +
          (val1 > bnd4).to(tl.int32) + (val1 > bnd5).to(tl.int32) +
          (val1 > bnd6).to(tl.int32))
    val2 = tl.load(input_ptr + in_base + 2, mask=mask, other=0.0)
    v2 = ((val2 > bnd0).to(tl.int32) + (val2 > bnd1).to(tl.int32) +
          (val2 > bnd2).to(tl.int32) + (val2 > bnd3).to(tl.int32) +
          (val2 > bnd4).to(tl.int32) + (val2 > bnd5).to(tl.int32) +
          (val2 > bnd6).to(tl.int32))
    val3 = tl.load(input_ptr + in_base + 3, mask=mask, other=0.0)
    v3 = ((val3 > bnd0).to(tl.int32) + (val3 > bnd1).to(tl.int32) +
          (val3 > bnd2).to(tl.int32) + (val3 > bnd3).to(tl.int32) +
          (val3 > bnd4).to(tl.int32) + (val3 > bnd5).to(tl.int32) +
          (val3 > bnd6).to(tl.int32))
    val4 = tl.load(input_ptr + in_base + 4, mask=mask, other=0.0)
    v4 = ((val4 > bnd0).to(tl.int32) + (val4 > bnd1).to(tl.int32) +
          (val4 > bnd2).to(tl.int32) + (val4 > bnd3).to(tl.int32) +
          (val4 > bnd4).to(tl.int32) + (val4 > bnd5).to(tl.int32) +
          (val4 > bnd6).to(tl.int32))
    val5 = tl.load(input_ptr + in_base + 5, mask=mask, other=0.0)
    v5 = ((val5 > bnd0).to(tl.int32) + (val5 > bnd1).to(tl.int32) +
          (val5 > bnd2).to(tl.int32) + (val5 > bnd3).to(tl.int32) +
          (val5 > bnd4).to(tl.int32) + (val5 > bnd5).to(tl.int32) +
          (val5 > bnd6).to(tl.int32))
    val6 = tl.load(input_ptr + in_base + 6, mask=mask, other=0.0)
    v6 = ((val6 > bnd0).to(tl.int32) + (val6 > bnd1).to(tl.int32) +
          (val6 > bnd2).to(tl.int32) + (val6 > bnd3).to(tl.int32) +
          (val6 > bnd4).to(tl.int32) + (val6 > bnd5).to(tl.int32) +
          (val6 > bnd6).to(tl.int32))
    val7 = tl.load(input_ptr + in_base + 7, mask=mask, other=0.0)
    v7 = ((val7 > bnd0).to(tl.int32) + (val7 > bnd1).to(tl.int32) +
          (val7 > bnd2).to(tl.int32) + (val7 > bnd3).to(tl.int32) +
          (val7 > bnd4).to(tl.int32) + (val7 > bnd5).to(tl.int32) +
          (val7 > bnd6).to(tl.int32))

    b0 = (v0 | (v1 << 3) | (v2 << 6)) & 0xFF
    b1 = ((v2 >> 2) | (v3 << 1) | (v4 << 4) | (v5 << 7)) & 0xFF
    b2 = ((v5 >> 1) | (v6 << 2) | (v7 << 5)) & 0xFF

    out_base = vec_id * PACKED_DIM + grp * 3
    tl.store(output_ptr + out_base,     b0.to(tl.uint8), mask=mask)
    tl.store(output_ptr + out_base + 1, b1.to(tl.uint8), mask=mask)
    tl.store(output_ptr + out_base + 2, b2.to(tl.uint8), mask=mask)


@triton.jit
def _quant_pack_4bit_kernel(
    input_ptr, boundaries_ptr, output_ptr,
    N_GROUPS, D: tl.constexpr, PACKED_DIM: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    goff = pid * BLOCK + tl.arange(0, BLOCK)
    groups_per_vec = D >> 1
    mask = goff < N_GROUPS
    vec_id = goff // groups_per_vec
    grp = goff % groups_per_vec
    in_base = vec_id * D + grp * 2
    packed = tl.zeros([BLOCK], dtype=tl.int32)
    for pos in tl.static_range(2):
        v = tl.load(input_ptr + in_base + pos, mask=mask, other=0.0)
        idx = tl.zeros([BLOCK], dtype=tl.int32)
        for b in tl.static_range(15):
            bnd = tl.load(boundaries_ptr + b)
            idx += (v > bnd).to(tl.int32)
        packed = packed | (idx << (pos * 4))
    tl.store(output_ptr + vec_id * PACKED_DIM + grp, packed.to(tl.uint8), mask=mask)


# =============================================================================
# Python wrappers
# =============================================================================

_DEQUANT_KERNELS = {2: _dequant_2bit_kernel, 3: _dequant_3bit_kernel, 4: _dequant_4bit_kernel}
_QUANT_KERNELS = {2: _quant_pack_2bit_kernel, 3: _quant_pack_3bit_kernel, 4: _quant_pack_4bit_kernel}
_BLOCK = 1024


def triton_dequant(
    packed: torch.Tensor,
    centroids: torch.Tensor,
    norms: torch.Tensor,
    bits: int,
    head_dim: int,
    output: torch.Tensor = None,
) -> torch.Tensor:
    """Fused unpack + centroid lookup + norm scaling via Triton.

    Args:
        packed: (..., packed_dim) uint8
        centroids: (num_levels,) float32
        norms: (...,) float16 — per-vector L2 norms
        bits: 2, 3, or 4
        head_dim: D
        output: optional pre-allocated (..., head_dim) bf16 buffer

    Returns:
        (..., head_dim) bfloat16 — centroid_values * norm (PRE-ROTATION)
    """
    batch_shape = packed.shape[:-1]
    packed_dim = packed.shape[-1]
    N = packed.numel() // packed_dim

    flat_packed = packed.reshape(-1, packed_dim).contiguous()
    flat_norms = norms.reshape(-1).contiguous().float()
    if output is None:
        flat_output = torch.empty(N, head_dim, dtype=torch.bfloat16, device=packed.device)
    else:
        flat_output = output.reshape(-1, head_dim)

    n_elems = N * head_dim
    grid = ((n_elems + _BLOCK - 1) // _BLOCK,)

    kernel = _DEQUANT_KERNELS[bits]
    kernel[grid](
        flat_packed, flat_output, centroids, flat_norms,
        N, n_elems,
        D=head_dim, PACKED_DIM=packed_dim, BLOCK=_BLOCK,
    )
    return flat_output.reshape(*batch_shape, head_dim)


def triton_dequant_no_norm(
    packed: torch.Tensor,
    centroids: torch.Tensor,
    bits: int,
    head_dim: int,
    output: torch.Tensor = None,
) -> torch.Tensor:
    """Fused unpack + centroid lookup WITHOUT norm (for Prod residual path)."""
    batch_shape = packed.shape[:-1]
    packed_dim = packed.shape[-1]
    N = packed.numel() // packed_dim

    flat_packed = packed.reshape(-1, packed_dim).contiguous()
    if output is None:
        flat_output = torch.empty(N, head_dim, dtype=torch.bfloat16, device=packed.device)
    else:
        flat_output = output.reshape(-1, head_dim)

    n_elems = N * head_dim
    grid = ((n_elems + _BLOCK - 1) // _BLOCK,)

    # Use 3-bit no-norm kernel; for other bits, fall back to with-norm and norm=1
    if bits == 3:
        _dequant_3bit_no_norm_kernel[grid](
            flat_packed, flat_output, centroids,
            n_elems, D=head_dim, PACKED_DIM=packed_dim, BLOCK=_BLOCK,
        )
    else:
        ones = torch.ones(N, dtype=torch.float32, device=packed.device)
        kernel = _DEQUANT_KERNELS[bits]
        kernel[grid](
            flat_packed, flat_output, centroids, ones,
            N, n_elems, D=head_dim, PACKED_DIM=packed_dim, BLOCK=_BLOCK,
        )
    return flat_output.reshape(*batch_shape, head_dim)


def triton_quant_pack(
    values: torch.Tensor,
    boundaries: torch.Tensor,
    bits: int,
    head_dim: int,
    output: torch.Tensor = None,
) -> torch.Tensor:
    """Fused nearest-centroid quantization + bit packing via Triton."""
    batch_shape = values.shape[:-1]
    N = values.numel() // head_dim
    packed_dim = (head_dim * bits + 7) // 8

    flat_values = values.reshape(-1, head_dim).contiguous()
    boundaries_f32 = boundaries.to(torch.float32).contiguous()
    if flat_values.dtype != torch.float32:
        flat_values = flat_values.float()

    if output is None:
        flat_output = torch.empty(N, packed_dim, dtype=torch.uint8, device=values.device)
    else:
        flat_output = output.reshape(-1, packed_dim)

    if bits == 3:
        groups_per_vec = head_dim // 8
    elif bits == 2:
        groups_per_vec = head_dim // 4
    elif bits == 4:
        groups_per_vec = head_dim // 2
    else:
        raise ValueError(f"Unsupported bits={bits}")

    n_groups = N * groups_per_vec
    grid = ((n_groups + _BLOCK - 1) // _BLOCK,)

    kernel = _QUANT_KERNELS[bits]
    kernel[grid](
        flat_values, boundaries_f32, flat_output,
        n_groups, D=head_dim, PACKED_DIM=packed_dim, BLOCK=_BLOCK,
    )
    return flat_output.reshape(*batch_shape, packed_dim)
