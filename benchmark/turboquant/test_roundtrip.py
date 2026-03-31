#!/usr/bin/env python3
"""Minimal test to verify TurboQuant compress→decompress roundtrip on GPU."""
import torch

# Must run on the server with GPU
device = "cuda"

from sglang.srt.layers.quantization.turboquant.quantizer import TurboQuantMSE

for bits in [2, 3, 4]:
    tq = TurboQuantMSE(128, bits, seed=42, device=device)

    # Simulate KV cache values (bf16, similar to model output)
    x = torch.randn(10, 4, 128, device=device, dtype=torch.bfloat16)

    # ===== Test 1: Direct roundtrip =====
    packed, norms = tq.compress(x)
    x_hat = tq.decompress(packed, norms, out_dtype=torch.bfloat16)
    mse1 = ((x.float() - x_hat.float()) ** 2).mean().item()
    signal = (x.float() ** 2).mean().item()

    # ===== Test 2: Roundtrip via output buffer (like kv_pool) =====
    out_buf = torch.zeros_like(x)
    tq.decompress(packed, norms, output=out_buf)
    mse2 = ((x.float() - out_buf.float()) ** 2).mean().item()

    # ===== Test 3: Simulate full kv_pool flow =====
    m = 100  # pool size
    k_buffer = torch.zeros(m, 4, tq.packed_dim, dtype=torch.uint8, device=device)
    k_norm_buf = torch.zeros(m, 4, dtype=torch.float16, device=device)
    k_dequant = torch.zeros(m, 4, 128, dtype=torch.bfloat16, device=device)

    loc = torch.arange(5, 15, device=device)  # store at positions 5-14
    k_buffer[loc] = packed
    k_norm_buf[loc] = norms

    # Decompress [0:15] like kv_pool does
    n = 15
    tq.decompress(k_buffer[:n], k_norm_buf[:n], output=k_dequant[:n])
    mse3 = ((x.float() - k_dequant[5:15].float()) ** 2).mean().item()

    # Check if uninitialized positions (0-4) are zeros
    uninit_max = k_dequant[:5].abs().max().item()

    print(f"\n{bits}-bit MSE:")
    print(f"  Test 1 (direct):    MSE={mse1:.6f}, SNR={signal/max(mse1,1e-10):.1f}x")
    print(f"  Test 2 (output):    MSE={mse2:.6f}, SNR={signal/max(mse2,1e-10):.1f}x")
    print(f"  Test 3 (kv_pool):   MSE={mse3:.6f}, SNR={signal/max(mse3,1e-10):.1f}x")
    print(f"  Uninit positions:   max_abs={uninit_max:.6f} (should be ~0)")
    print(f"  Input[:5]:  {x[0,0,:5].tolist()}")
    print(f"  Output[:5]: k_dequant[5,0,:5] = {k_dequant[5,0,:5].tolist()}")

    # CRITICAL: check if results match
    if mse3 > 10 * mse1:
        print(f"  *** BUG: kv_pool MSE is {mse3/max(mse1,1e-10):.0f}x worse than direct! ***")
    if uninit_max > 0.01:
        print(f"  *** BUG: uninitialized positions are non-zero! ***")
