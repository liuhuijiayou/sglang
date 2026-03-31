"""
TurboQuant: Data-oblivious vector quantization for KV cache compression.

Reference: "TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate"
           https://arxiv.org/abs/2504.19874

Quantizer variants:
  - TurboQuantMSE:   Algorithm 1 — MSE-optimal (default)
  - TurboQuantProd:  Algorithm 2 — Unbiased inner product (MSE + QJL residual)
  - TurboQuantMixed: Section 6.4 — Non-integer bit-width via channel splitting

STATUS: Experimental feature. Enabled via --kv-cache-dtype turboquant_{N}bit.
"""

from sglang.srt.layers.quantization.turboquant.kv_pool import (
    MHATokenToKVPoolTurboQuant,
)
from sglang.srt.layers.quantization.turboquant.quantizer import (
    TurboQuantMixed,
    TurboQuantMSE,
    TurboQuantProd,
    create_quantizer,
)

__all__ = [
    "TurboQuantMSE",
    "TurboQuantProd",
    "TurboQuantMixed",
    "create_quantizer",
    "MHATokenToKVPoolTurboQuant",
]
