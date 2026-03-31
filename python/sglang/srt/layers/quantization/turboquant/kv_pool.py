"""
TurboQuant KV cache pool for SGLang.

Subclasses MHATokenToKVPool to store KV cache entries in TurboQuant-compressed
format. Supports all three quantizer variants:

  - TurboQuantMSE (integer bits, MSE-optimal)
  - TurboQuantProd (integer bits, unbiased inner product)
  - TurboQuantMixed (non-integer bits like 2.5, 3.5)

PERFORMANCE DESIGN:
  1. Triton fused kernels (triton_ops.py) eliminate intermediate tensors in the
     unpack→lookup and quantize→pack paths.
  2. Pre-allocated decompression buffer avoids per-call memory allocation.
  3. Full-buffer decompression on each get_kv_buffer call (CUDA-graph-safe).

Compression ratios (head_dim=128):
    MSE 3-bit:   48B packed + 2B norm = 50B  vs 256B → 5.12x
    MSE 4-bit:   64B packed + 2B norm = 66B  vs 256B → 3.88x
    Mixed 2.5:   24B(3b) + 16B(2b) + 2B norm = 42B → 6.10x
    Mixed 3.5:   32B(4b) + 24B(3b) + 2B norm = 58B → 4.41x
"""

import logging
import os
from contextlib import nullcontext
from typing import Optional

import torch

from sglang.srt.layers.quantization.turboquant.quantizer import create_quantizer
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    get_tensor_size_bytes,
)

try:
    from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
except ImportError:
    GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"

logger = logging.getLogger(__name__)


class MHATokenToKVPoolTurboQuant(MHATokenToKVPool):
    """KV cache pool with TurboQuant compression and bit-packed storage.

    Memory layout:
        Compressed (persistent, per-layer):
            k_buffer:      (size, head_num, packed_dim)  uint8
            k_norm_buffer: (size, head_num)              float16

        Decompressed cache (shared across layers, reused):
            _k_dequant_buf: (size, head_num, head_dim)   model dtype (bf16)
            _v_dequant_buf: (size, head_num, head_dim)   model dtype (bf16)

        The dequant bufs are ONE pair shared across all layers. Since layers
        are processed sequentially, only one layer's decompressed data is
        needed at a time.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        turboquant_bits: float = 3.0,
        turboquant_variant: str = "mse",
        turboquant_seed: int = 42,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        **kwargs,
    ):
        self.turboquant_bits = turboquant_bits
        self.turboquant_variant = turboquant_variant
        self.turboquant_seed = turboquant_seed
        self._tq_device = device
        self._tq_head_dim = head_dim
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=False,
            enable_kv_cache_copy=False,
        )

    def _create_buffers(self):
        """Allocate compressed + decompression cache buffers."""
        self.tq = create_quantizer(
            head_dim=self._tq_head_dim,
            bits=self.turboquant_bits,
            variant=self.turboquant_variant,
            seed=self.turboquant_seed,
            device=self._tq_device,
        )

        pd = self.tq.packed_dim
        has_rnorm = self.tq.has_residual_norm

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                m = self.size + self.page_size
                n = self.head_num
                d = self._tq_head_dim

                # --- Compressed storage (persistent, per-layer) ---
                self.store_dtype = torch.uint8
                self.k_buffer = [
                    torch.zeros((m, n, pd), dtype=torch.uint8, device=self.device)
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros((m, n, pd), dtype=torch.uint8, device=self.device)
                    for _ in range(self.layer_num)
                ]
                self.k_norm_buffer = [
                    torch.zeros((m, n), dtype=torch.float16, device=self.device)
                    for _ in range(self.layer_num)
                ]
                self.v_norm_buffer = [
                    torch.zeros((m, n), dtype=torch.float16, device=self.device)
                    for _ in range(self.layer_num)
                ]
                if has_rnorm:
                    self.k_rnorm_buffer = [
                        torch.zeros((m, n), dtype=torch.float16, device=self.device)
                        for _ in range(self.layer_num)
                    ]
                    self.v_rnorm_buffer = [
                        torch.zeros((m, n), dtype=torch.float16, device=self.device)
                        for _ in range(self.layer_num)
                    ]
                else:
                    self.k_rnorm_buffer = None
                    self.v_rnorm_buffer = None

                # --- Decompression cache (shared across layers) ---
                self._k_dequant_buf = torch.zeros(
                    (m, n, d), dtype=self.dtype, device=self.device
                )
                self._v_dequant_buf = torch.zeros(
                    (m, n, d), dtype=self.dtype, device=self.device
                )

        # High water mark: only decompress [0:_max_occupied] instead of full buffer.
        # Dramatically reduces decompress cost (rotation matmul) when few tokens are active.
        self._max_occupied = 0
        self._debug_count = 0

        logger.info(
            f"TurboQuant KV cache: {self.turboquant_bits}-bit {self.turboquant_variant}, "
            f"d={self.head_dim}, packed_dim={pd}, heads={self.head_num}, "
            f"layers={self.layer_num}, tokens={self.size}, "
            f"has_residual_norm={has_rnorm}"
        )

    def _clear_buffers(self):
        del self.k_buffer, self.v_buffer
        del self.k_norm_buffer, self.v_norm_buffer
        del self._k_dequant_buf, self._v_dequant_buf
        if self.k_rnorm_buffer is not None:
            del self.k_rnorm_buffer, self.v_rnorm_buffer

    def set_kv_buffer(
        self,
        layer,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale=None,
        v_scale=None,
        layer_id_override: Optional[int] = None,
    ):
        """Compress, pack, and store KV entries."""
        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
        buf_idx = layer_id - self.start_layer

        if buf_idx == 0 and self._debug_count < 10:
            logger.info(
                f"TurboQuant set_kv_buffer: layer={layer_id}, loc_size={loc.numel()}, "
                f"loc_range=[{loc.min().item()},{loc.max().item()}], "
                f"cache_k_shape={list(cache_k.shape)}, dtype={cache_k.dtype}"
            )

        if self.tq.has_residual_norm:
            k_packed, k_norms, k_rnorms = self.tq.compress(cache_k)
            v_packed, v_norms, v_rnorms = self.tq.compress(cache_v)
            self.k_rnorm_buffer[buf_idx][loc] = k_rnorms
            self.v_rnorm_buffer[buf_idx][loc] = v_rnorms
        else:
            k_packed, k_norms = self.tq.compress(cache_k)
            v_packed, v_norms = self.tq.compress(cache_v)

        self.k_buffer[buf_idx][loc] = k_packed
        self.v_buffer[buf_idx][loc] = v_packed
        self.k_norm_buffer[buf_idx][loc] = k_norms
        self.v_norm_buffer[buf_idx][loc] = v_norms

        # Track high water mark for efficient decompress
        if loc.numel() > 0:
            self._max_occupied = max(self._max_occupied, loc.max().item() + 1)

        # Optional roundtrip quality check (TURBOQUANT_DEBUG=1)
        if os.environ.get("TURBOQUANT_DEBUG") and self._debug_count < 5:
            rnorms_k = k_rnorms if self.tq.has_residual_norm else None
            result = self.tq.decompress(k_packed, k_norms, residual_norms=rnorms_k,
                                         out_dtype=cache_k.dtype)
            mse = ((cache_k.float() - result.float()) ** 2).mean().item()
            signal = (cache_k.float() ** 2).mean().item()
            snr = signal / max(mse, 1e-10)
            logger.info(
                f"TurboQuant roundtrip check (layer {layer_id}): "
                f"MSE={mse:.6f}, signal={signal:.6f}, SNR={snr:.1f}x, "
                f"tokens={cache_k.shape[0]}"
            )
            self._debug_count += 1

    # Max tokens to decompress per chunk. Limits intermediate memory for
    # Prod/Mixed variants (QJL unpack creates float32 intermediates).
    # 512K tokens × 4 heads × 128 dim × 4B ≈ 1GB per intermediate tensor.
    _DECOMPRESS_CHUNK = 1 << 19  # 512K tokens

    def _decompress_chunked(self, buf, norm_buf, rnorm_buf, out_buf):
        """Decompress buffer in chunks to avoid OOM on large pools."""
        m = buf.shape[0]
        if m <= self._DECOMPRESS_CHUNK:
            rnorms = rnorm_buf if rnorm_buf is not None else None
            self.tq.decompress(
                buf, norm_buf, residual_norms=rnorms,
                out_dtype=self.dtype, output=out_buf,
            )
        else:
            for start in range(0, m, self._DECOMPRESS_CHUNK):
                end = min(start + self._DECOMPRESS_CHUNK, m)
                rnorms = rnorm_buf[start:end] if rnorm_buf is not None else None
                self.tq.decompress(
                    buf[start:end], norm_buf[start:end],
                    residual_norms=rnorms,
                    out_dtype=self.dtype, output=out_buf[start:end],
                )

    def _get_key_buffer(self, layer_id: int) -> torch.Tensor:
        """Decompress occupied portion of buffer and return."""
        buf_idx = layer_id - self.start_layer
        n = self._max_occupied
        if n > 0:
            if buf_idx == 0 and self._debug_count < 10:
                logger.info(f"TurboQuant _get_key_buffer: layer={layer_id}, max_occupied={n}")
                self._debug_count += 1
            rnorm_buf = self.k_rnorm_buffer[buf_idx][:n] if self.k_rnorm_buffer is not None else None
            self._decompress_chunked(
                self.k_buffer[buf_idx][:n], self.k_norm_buffer[buf_idx][:n],
                rnorm_buf, self._k_dequant_buf[:n],
            )
        return self._k_dequant_buf

    def _get_value_buffer(self, layer_id: int) -> torch.Tensor:
        """Decompress occupied portion of buffer and return."""
        buf_idx = layer_id - self.start_layer
        n = self._max_occupied
        if n > 0:
            rnorm_buf = self.v_rnorm_buffer[buf_idx][:n] if self.v_rnorm_buffer is not None else None
            self._decompress_chunked(
                self.v_buffer[buf_idx][:n], self.v_norm_buffer[buf_idx][:n],
                rnorm_buf, self._v_dequant_buf[:n],
            )
        return self._v_dequant_buf

    def get_kv_size_bytes(self):
        """Report actual memory usage of compressed buffers."""
        total_k = sum(get_tensor_size_bytes(b) for b in self.k_buffer)
        total_v = sum(get_tensor_size_bytes(b) for b in self.v_buffer)
        total_k += sum(get_tensor_size_bytes(b) for b in self.k_norm_buffer)
        total_v += sum(get_tensor_size_bytes(b) for b in self.v_norm_buffer)
        if self.k_rnorm_buffer is not None:
            total_k += sum(get_tensor_size_bytes(b) for b in self.k_rnorm_buffer)
            total_v += sum(get_tensor_size_bytes(b) for b in self.v_rnorm_buffer)
        return total_k, total_v

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """Copy compressed entries between locations."""
        if tgt_loc.numel() == 0:
            return
        # Update high water mark for target positions
        self._max_occupied = max(self._max_occupied, tgt_loc.max().item() + 1)
        for i in range(self.layer_num):
            self.k_buffer[i][tgt_loc] = self.k_buffer[i][src_loc]
            self.v_buffer[i][tgt_loc] = self.v_buffer[i][src_loc]
            self.k_norm_buffer[i][tgt_loc] = self.k_norm_buffer[i][src_loc]
            self.v_norm_buffer[i][tgt_loc] = self.v_norm_buffer[i][src_loc]
            if self.k_rnorm_buffer is not None:
                self.k_rnorm_buffer[i][tgt_loc] = self.k_rnorm_buffer[i][src_loc]
                self.v_rnorm_buffer[i][tgt_loc] = self.v_rnorm_buffer[i][src_loc]
