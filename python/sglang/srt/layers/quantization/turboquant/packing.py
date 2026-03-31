"""
Bit-packing utilities for sub-byte quantization indices.

Packs b-bit indices (b = 1, 2, 3, 4) into uint8 bytes.

Packed sizes for head_dim=128:
    1-bit: 128 / 8 = 16 bytes   (8 indices per byte)
    2-bit: 128 / 4 = 32 bytes   (4 indices per byte)
    3-bit: 128 * 3 / 8 = 48 bytes (8 indices per 3 bytes)
    4-bit: 128 / 2 = 64 bytes   (2 indices per byte)

Constraint: head_dim * bits must be divisible by 8.

All paths use specialized shift operations — no intermediate bit-stream
expansion. This avoids the memory blowup of the general approach.
"""

import torch


def packed_dim(head_dim: int, bits: int) -> int:
    """Compute the packed byte dimension for a given head_dim and bit-width."""
    assert (head_dim * bits) % 8 == 0, (
        f"head_dim * bits must be divisible by 8, got {head_dim} * {bits} = {head_dim * bits}"
    )
    return (head_dim * bits) // 8


def pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack b-bit quantization indices into uint8 bytes.

    Args:
        indices: (..., head_dim) uint8, values in [0, 2^bits - 1]
        bits: bit-width (1, 2, 3, or 4)

    Returns:
        packed: (..., packed_dim) uint8
    """
    if bits == 8:
        return indices
    if bits == 4:
        return _pack_4bit(indices)
    if bits == 3:
        return _pack_3bit(indices)
    if bits == 2:
        return _pack_2bit(indices)
    if bits == 1:
        return _pack_1bit(indices)
    raise ValueError(f"Unsupported bits={bits}. Supported: 1, 2, 3, 4, 8.")


def unpack_indices(packed: torch.Tensor, bits: int, head_dim: int) -> torch.Tensor:
    """Unpack uint8 bytes back to b-bit quantization indices.

    Args:
        packed: (..., packed_dim) uint8
        bits: bit-width (1, 2, 3, or 4)
        head_dim: original unpacked dimension

    Returns:
        indices: (..., head_dim) uint8, values in [0, 2^bits - 1]
    """
    if bits == 8:
        return packed
    if bits == 4:
        return _unpack_4bit(packed, head_dim)
    if bits == 3:
        return _unpack_3bit(packed, head_dim)
    if bits == 2:
        return _unpack_2bit(packed, head_dim)
    if bits == 1:
        return _unpack_1bit(packed, head_dim)
    raise ValueError(f"Unsupported bits={bits}. Supported: 1, 2, 3, 4, 8.")


# ---------------------------------------------------------------------------
# 4-bit: 2 indices per byte
# byte = low_nibble | (high_nibble << 4)
# ---------------------------------------------------------------------------

def _pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    low = indices[..., 0::2]
    high = indices[..., 1::2]
    return (low | (high << 4)).to(torch.uint8)


def _unpack_4bit(packed: torch.Tensor, head_dim: int) -> torch.Tensor:
    batch_shape = packed.shape[:-1]
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    out = torch.stack([low, high], dim=-1)
    return out.reshape(*batch_shape, head_dim).to(torch.uint8)


# ---------------------------------------------------------------------------
# 3-bit: 8 indices per 3 bytes (24 bits)
#
# Encoding (8 values v0..v7, each 3 bits, into bytes b0, b1, b2):
#   b0 = v0 | (v1 << 3) | (v2 << 6)
#   b1 = (v2 >> 2) | (v3 << 1) | (v4 << 4) | (v5 << 7)
#   b2 = (v5 >> 1) | (v6 << 2) | (v7 << 5)
#
# This uses pure shift/or operations — no intermediate bit tensor.
# ---------------------------------------------------------------------------

def _pack_3bit(indices: torch.Tensor) -> torch.Tensor:
    batch_shape = indices.shape[:-1]
    D = indices.shape[-1]
    assert D % 8 == 0
    # Reshape to (..., D/8, 8) for group processing
    idx = indices.reshape(*batch_shape, D // 8, 8).to(torch.int32)
    v0, v1, v2, v3, v4, v5, v6, v7 = [idx[..., i] for i in range(8)]

    b0 = (v0 | (v1 << 3) | (v2 << 6)) & 0xFF
    b1 = ((v2 >> 2) | (v3 << 1) | (v4 << 4) | (v5 << 7)) & 0xFF
    b2 = ((v5 >> 1) | (v6 << 2) | (v7 << 5)) & 0xFF

    # Stack and reshape: (..., D/8, 3) -> (..., D*3/8)
    packed = torch.stack([b0, b1, b2], dim=-1).to(torch.uint8)
    return packed.reshape(*batch_shape, D * 3 // 8)


def _unpack_3bit(packed: torch.Tensor, head_dim: int) -> torch.Tensor:
    batch_shape = packed.shape[:-1]
    # Reshape to (..., head_dim/8, 3)
    groups = packed.reshape(*batch_shape, head_dim // 8, 3).to(torch.int32)
    b0, b1, b2 = groups[..., 0], groups[..., 1], groups[..., 2]

    v0 = b0 & 0x07
    v1 = (b0 >> 3) & 0x07
    v2 = ((b0 >> 6) | (b1 << 2)) & 0x07
    v3 = (b1 >> 1) & 0x07
    v4 = (b1 >> 4) & 0x07
    v5 = ((b1 >> 7) | (b2 << 1)) & 0x07
    v6 = (b2 >> 2) & 0x07
    v7 = (b2 >> 5) & 0x07

    out = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=-1)
    return out.reshape(*batch_shape, head_dim).to(torch.uint8)


# ---------------------------------------------------------------------------
# 2-bit: 4 indices per byte
# byte = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)
# ---------------------------------------------------------------------------

def _pack_2bit(indices: torch.Tensor) -> torch.Tensor:
    v0 = indices[..., 0::4]
    v1 = indices[..., 1::4]
    v2 = indices[..., 2::4]
    v3 = indices[..., 3::4]
    return (v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)).to(torch.uint8)


def _unpack_2bit(packed: torch.Tensor, head_dim: int) -> torch.Tensor:
    batch_shape = packed.shape[:-1]
    v0 = packed & 0x03
    v1 = (packed >> 2) & 0x03
    v2 = (packed >> 4) & 0x03
    v3 = (packed >> 6) & 0x03
    out = torch.stack([v0, v1, v2, v3], dim=-1)
    return out.reshape(*batch_shape, head_dim).to(torch.uint8)


# ---------------------------------------------------------------------------
# 1-bit: 8 indices per byte
# byte = v0 | (v1 << 1) | ... | (v7 << 7)
# ---------------------------------------------------------------------------

def _pack_1bit(indices: torch.Tensor) -> torch.Tensor:
    batch_shape = indices.shape[:-1]
    D = indices.shape[-1]
    assert D % 8 == 0
    idx = indices.reshape(*batch_shape, D // 8, 8).to(torch.int32)
    packed = idx[..., 0]
    for i in range(1, 8):
        packed = packed | (idx[..., i] << i)
    return packed.to(torch.uint8).reshape(*batch_shape, D // 8)


def _unpack_1bit(packed: torch.Tensor, head_dim: int) -> torch.Tensor:
    batch_shape = packed.shape[:-1]
    groups = packed.reshape(*batch_shape, head_dim // 8).to(torch.int32)
    bits = [(groups >> i) & 1 for i in range(8)]
    out = torch.stack(bits, dim=-1)
    return out.reshape(*batch_shape, head_dim).to(torch.uint8)
