#!/usr/bin/env python3
"""
TurboQuant end-to-end performance benchmark.

Measures compress (set_kv_buffer path) and decompress (get_key_buffer path)
throughput across all quantizer variants, bit-widths, and realistic batch sizes.

Designed for GPU execution. Falls back to manual timing on CPU.

Usage:
    # Full GPU benchmark (recommended)
    python bench_e2e_perf.py

    # Specific configs
    python bench_e2e_perf.py --bits 3 4 --variants mse --device cuda

    # Save results
    python bench_e2e_perf.py --output results_perf.json

    # Quick CI mode (smaller sweep)
    python bench_e2e_perf.py --ci

Model reference (Llama-3.1-8B-Instruct):
    head_dim=128, num_kv_heads=8, num_layers=32
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Import TurboQuant — handle both installed and development paths
# ---------------------------------------------------------------------------
try:
    from sglang.srt.layers.quantization.turboquant.quantizer import (
        TurboQuantMixed,
        TurboQuantMSE,
        TurboQuantProd,
        create_quantizer,
    )
except ImportError:
    import importlib.util
    import types

    sys.path.insert(0, "python")
    for pkg in [
        "sglang", "sglang.srt", "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.turboquant",
    ]:
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)
            if "." in pkg:
                setattr(sys.modules[pkg.rsplit(".", 1)[0]], pkg.split(".")[-1], sys.modules[pkg])

    _base = "python/sglang/srt/layers/quantization/turboquant"
    for name, fname in [
        ("codebook", "codebook.py"), ("rotation", "rotation.py"),
        ("packing", "packing.py"), ("qjl", "qjl.py"), ("quantizer", "quantizer.py"),
    ]:
        fqn = f"sglang.srt.layers.quantization.turboquant.{name}"
        spec = importlib.util.spec_from_file_location(fqn, f"{_base}/{fname}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[fqn] = mod
        spec.loader.exec_module(mod)

    from sglang.srt.layers.quantization.turboquant.quantizer import (
        TurboQuantMixed,
        TurboQuantMSE,
        TurboQuantProd,
        create_quantizer,
    )


# ---------------------------------------------------------------------------
# Timing utilities — follow SGLang conventions
# ---------------------------------------------------------------------------
_USE_TRITON_BENCH = False
try:
    import triton.testing
    _USE_TRITON_BENCH = True
except ImportError:
    pass


def _sync(device: str):
    if "cuda" in device:
        torch.cuda.synchronize()


def bench_fn(fn: Callable, device: str, warmup: int = 10, rep: int = 50) -> Dict[str, float]:
    """Benchmark a function, returning timing statistics in milliseconds.

    Uses triton.testing.do_bench on CUDA when available, otherwise manual timing.
    Returns dict with keys: median_ms, min_ms, max_ms.
    """
    if _USE_TRITON_BENCH and "cuda" in device:
        ms, min_ms, max_ms = triton.testing.do_bench(
            fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8]
        )
        return {"median_ms": ms, "p20_ms": min_ms, "p80_ms": max_ms}

    # Manual timing fallback
    for _ in range(warmup):
        fn()
    _sync(device)

    times = []
    for _ in range(rep):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return {
        "median_ms": times[len(times) // 2],
        "p20_ms": times[int(len(times) * 0.2)],
        "p80_ms": times[int(len(times) * 0.8)],
    }


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class BenchResult:
    variant: str        # "mse", "prod", "mixed"
    bits: float         # 2, 2.5, 3, 3.5, 4
    operation: str      # "compress", "decompress", "compress_op_*"
    num_tokens: int     # batch dimension
    num_heads: int
    head_dim: int
    median_ms: float
    p20_ms: float
    p80_ms: float
    throughput_mtok_s: float  # million tokens / sec
    bytes_per_token: float    # packed bytes per token (K only, one head)


# ---------------------------------------------------------------------------
# Microbenchmark: individual operations
# ---------------------------------------------------------------------------
def bench_ops_breakdown(
    tq, x: torch.Tensor, device: str, warmup: int, rep: int
) -> List[BenchResult]:
    """Benchmark individual operations in the compress/decompress pipeline."""
    results = []
    ntok = x.shape[0]
    nh = x.shape[1] if x.ndim == 3 else 1
    d = x.shape[-1]
    label_prefix = f"{type(tq).__name__} {getattr(tq, 'bits', getattr(tq, 'target_bits', '?'))}b"

    # -- Compress sub-ops (MSE path) --
    if isinstance(tq, TurboQuantMSE):
        from sglang.srt.layers.quantization.turboquant.rotation import rotate, unrotate
        from sglang.srt.layers.quantization.turboquant.packing import pack_indices, unpack_indices

        norms = x.float().norm(dim=-1)
        x_unit = x / norms.to(dtype=x.dtype).unsqueeze(-1).clamp(min=1e-12)

        ops = [
            ("norm",      lambda: x.float().norm(dim=-1)),
            ("normalize", lambda: x / norms.to(dtype=x.dtype).unsqueeze(-1).clamp(min=1e-12)),
            ("rotate",    lambda: rotate(x_unit, tq.R)),
        ]
        y = rotate(x_unit, tq.R)
        ops.append(("bucketize", lambda: torch.bucketize(
            y.float() if y.dtype != tq.boundaries.dtype else y, tq.boundaries
        ).to(torch.uint8)))
        indices = torch.bucketize(
            y.float() if y.dtype != tq.boundaries.dtype else y, tq.boundaries
        ).to(torch.uint8)
        ops.append(("pack",      lambda: pack_indices(indices, tq.bits)))

        packed = pack_indices(indices, tq.bits)
        ops.append(("unpack",    lambda: unpack_indices(packed, tq.bits, d)))
        unpacked = unpack_indices(packed, tq.bits, d)
        ops.append(("lookup",    lambda: tq.centroids[unpacked.long()]))
        y_hat = tq.centroids[unpacked.long()]
        if y_hat.is_cuda and torch.bfloat16 in (torch.bfloat16,):
            y_hat_cast = y_hat.to(torch.bfloat16)
            ops.append(("unrotate",  lambda: unrotate(y_hat_cast, tq.R)))
        else:
            ops.append(("unrotate",  lambda: unrotate(y_hat, tq.R)))

        for op_name, op_fn in ops:
            t = bench_fn(op_fn, device, warmup=warmup, rep=rep)
            results.append(BenchResult(
                variant="mse", bits=tq.bits, operation=f"op_{op_name}",
                num_tokens=ntok, num_heads=nh, head_dim=d,
                median_ms=t["median_ms"], p20_ms=t["p20_ms"], p80_ms=t["p80_ms"],
                throughput_mtok_s=ntok * nh / t["median_ms"] / 1000 if t["median_ms"] > 0 else 0,
                bytes_per_token=tq.packed_dim,
            ))

    return results


# ---------------------------------------------------------------------------
# End-to-end compress/decompress benchmark
# ---------------------------------------------------------------------------
def bench_e2e(
    variant: str,
    bits: float,
    num_tokens_list: List[int],
    num_heads: int,
    head_dim: int,
    device: str,
    dtype: torch.dtype,
    warmup: int,
    rep: int,
) -> List[BenchResult]:
    """Benchmark end-to-end compress and decompress throughput."""
    results = []

    tq = create_quantizer(head_dim=head_dim, bits=bits, variant=variant, device=device)
    packed_bytes = tq.packed_dim
    has_rnorm = tq.has_residual_norm

    for ntok in num_tokens_list:
        x = torch.randn(ntok, num_heads, head_dim, dtype=dtype, device=device)

        # --- Compress ---
        def do_compress():
            return tq.compress(x)

        t_c = bench_fn(do_compress, device, warmup=warmup, rep=rep)
        tput_c = ntok * num_heads / t_c["median_ms"] / 1000  # Mtok/s

        results.append(BenchResult(
            variant=variant, bits=bits, operation="compress",
            num_tokens=ntok, num_heads=num_heads, head_dim=head_dim,
            median_ms=t_c["median_ms"], p20_ms=t_c["p20_ms"], p80_ms=t_c["p80_ms"],
            throughput_mtok_s=tput_c, bytes_per_token=packed_bytes,
        ))

        # Prepare packed data for decompress benchmark
        if has_rnorm:
            packed, norms, rnorms = tq.compress(x)
        else:
            packed, norms = tq.compress(x)
            rnorms = None

        # --- Decompress ---
        def do_decompress():
            return tq.decompress(packed, norms, residual_norms=rnorms, out_dtype=dtype)

        t_d = bench_fn(do_decompress, device, warmup=warmup, rep=rep)
        tput_d = ntok * num_heads / t_d["median_ms"] / 1000

        results.append(BenchResult(
            variant=variant, bits=bits, operation="decompress",
            num_tokens=ntok, num_heads=num_heads, head_dim=head_dim,
            median_ms=t_d["median_ms"], p20_ms=t_d["p20_ms"], p80_ms=t_d["p80_ms"],
            throughput_mtok_s=tput_d, bytes_per_token=packed_bytes,
        ))

    return results


# ---------------------------------------------------------------------------
# Simulated KV cache pattern: write 1 token, read full buffer
# ---------------------------------------------------------------------------
def bench_kv_cache_pattern(
    variant: str,
    bits: float,
    buffer_sizes: List[int],
    num_heads: int,
    head_dim: int,
    device: str,
    dtype: torch.dtype,
    warmup: int,
    rep: int,
) -> List[BenchResult]:
    """Simulate the actual KV cache access pattern during autoregressive decode:
    compress 1 token (write), then decompress the full buffer (read).
    This is the critical hot path for inference latency.
    """
    results = []
    tq = create_quantizer(head_dim=head_dim, bits=bits, variant=variant, device=device)
    has_rnorm = tq.has_residual_norm

    for buf_size in buffer_sizes:
        # Pre-fill buffer
        x_buf = torch.randn(buf_size, num_heads, head_dim, dtype=dtype, device=device)
        if has_rnorm:
            packed_buf, norms_buf, rnorms_buf = tq.compress(x_buf)
        else:
            packed_buf, norms_buf = tq.compress(x_buf)
            rnorms_buf = None

        # Single token for write
        x_one = torch.randn(1, num_heads, head_dim, dtype=dtype, device=device)

        # --- Write: compress 1 token ---
        def do_write():
            return tq.compress(x_one)
        t_w = bench_fn(do_write, device, warmup=warmup, rep=rep)

        results.append(BenchResult(
            variant=variant, bits=bits, operation="kv_write_1tok",
            num_tokens=buf_size, num_heads=num_heads, head_dim=head_dim,
            median_ms=t_w["median_ms"], p20_ms=t_w["p20_ms"], p80_ms=t_w["p80_ms"],
            throughput_mtok_s=num_heads / t_w["median_ms"] / 1000 if t_w["median_ms"] > 0 else 0,
            bytes_per_token=tq.packed_dim,
        ))

        # --- Read: decompress full buffer ---
        def do_read():
            return tq.decompress(packed_buf, norms_buf, residual_norms=rnorms_buf, out_dtype=dtype)
        t_r = bench_fn(do_read, device, warmup=warmup, rep=rep)
        tput_r = buf_size * num_heads / t_r["median_ms"] / 1000

        results.append(BenchResult(
            variant=variant, bits=bits, operation="kv_read_full",
            num_tokens=buf_size, num_heads=num_heads, head_dim=head_dim,
            median_ms=t_r["median_ms"], p20_ms=t_r["p20_ms"], p80_ms=t_r["p80_ms"],
            throughput_mtok_s=tput_r, bytes_per_token=tq.packed_dim,
        ))

    return results


# ---------------------------------------------------------------------------
# Baseline: FP16 no-quantization (memcpy equivalent)
# ---------------------------------------------------------------------------
def bench_baseline(
    num_tokens_list: List[int],
    num_heads: int,
    head_dim: int,
    device: str,
    dtype: torch.dtype,
    warmup: int,
    rep: int,
) -> List[BenchResult]:
    """Measure raw memory copy as a reference baseline (no quantization)."""
    results = []

    for ntok in num_tokens_list:
        src = torch.randn(ntok, num_heads, head_dim, dtype=dtype, device=device)
        dst = torch.empty_like(src)

        def do_copy():
            dst.copy_(src)

        t = bench_fn(do_copy, device, warmup=warmup, rep=rep)
        bw = src.nelement() * src.element_size() / t["median_ms"] / 1e6  # GB/s

        results.append(BenchResult(
            variant="baseline_fp16", bits=16, operation="memcpy",
            num_tokens=ntok, num_heads=num_heads, head_dim=head_dim,
            median_ms=t["median_ms"], p20_ms=t["p20_ms"], p80_ms=t["p80_ms"],
            throughput_mtok_s=ntok * num_heads / t["median_ms"] / 1000 if t["median_ms"] > 0 else 0,
            bytes_per_token=head_dim * 2,  # fp16
        ))

    return results


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def print_results(results: List[BenchResult], title: str):
    print(f"\n{'=' * 95}")
    print(f"  {title}")
    print(f"{'=' * 95}")
    print(f"  {'variant':<10} {'bits':>4} {'operation':<18} {'tokens':>7} "
          f"{'median':>8} {'p20':>8} {'p80':>8} {'Mtok/s':>8} {'B/tok':>6}")
    print(f"  {'-' * 10} {'-' * 4} {'-' * 18} {'-' * 7} "
          f"{'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 6}")
    for r in results:
        bits_str = f"{r.bits:.1f}" if r.bits != int(r.bits) else f"{int(r.bits)}"
        print(f"  {r.variant:<10} {bits_str:>4} {r.operation:<18} {r.num_tokens:>7} "
              f"{r.median_ms:>7.3f}{'ms':1} {r.p20_ms:>7.3f}{'ms':1} {r.p80_ms:>7.3f}{'ms':1} "
              f"{r.throughput_mtok_s:>7.2f}{'M':1} {r.bytes_per_token:>5.0f}B")


def print_kv_summary(results: List[BenchResult]):
    """Print a decode-focused summary: per-layer cost extrapolation."""
    reads = [r for r in results if r.operation == "kv_read_full"]
    if not reads:
        return

    print(f"\n{'=' * 70}")
    print(f"  Decode latency estimate (32-layer model, read K+V per layer)")
    print(f"{'=' * 70}")
    print(f"  {'config':<20} {'buf_tokens':>10} {'per_read':>10} "
          f"{'×64(K+V)':>10} {'vs_fp16':>10}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")

    # Find baseline reads for comparison
    baselines = {r.num_tokens: r.median_ms for r in results
                 if r.variant == "baseline_fp16" and r.operation == "memcpy"}

    for r in reads:
        per_read_ms = r.median_ms
        total_64 = per_read_ms * 64  # 32 layers × 2 (K + V)
        bits_str = f"{r.bits:.1f}" if r.bits != int(r.bits) else f"{int(r.bits)}"
        label = f"{r.variant}_{bits_str}b"
        baseline_ms = baselines.get(r.num_tokens, 0)
        ratio = f"{per_read_ms / baseline_ms:.1f}x" if baseline_ms > 0 else "N/A"
        print(f"  {label:<20} {r.num_tokens:>10} {per_read_ms:>9.3f}ms "
              f"{total_64:>9.1f}ms {ratio:>10}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="TurboQuant end-to-end performance benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bits", type=float, nargs="+", default=[2, 3, 4])
    parser.add_argument("--mixed-bits", type=float, nargs="+", default=[2.5, 3.5],
                        help="Non-integer bit-widths for mixed-precision benchmark")
    parser.add_argument("--variants", type=str, nargs="+", default=["mse", "prod"],
                        choices=["mse", "prod"])
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4,
                        help="Number of KV heads (GQA). Default: 4 for Qwen2.5-7B")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--ci", action="store_true",
                        help="Reduced sweep for CI (faster)")
    parser.add_argument("--breakdown", action="store_true",
                        help="Include per-operation microbenchmark breakdown")
    args = parser.parse_args()

    device = args.device
    dtype = torch.bfloat16
    d = args.head_dim
    nh = args.num_heads

    if args.ci:
        num_tokens_list = [1, 512, 4096]
        buffer_sizes = [4096]
        args.bits = [3, 4]
        args.mixed_bits = [3.5]
        args.variants = ["mse"]
        args.rep = 20
    else:
        num_tokens_list = [1, 8, 64, 512, 2048, 8192]
        buffer_sizes = [1024, 4096, 8192, 16384]

    print(f"TurboQuant Performance Benchmark")
    print(f"  Device: {device}")
    if "cuda" in device:
        print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  dtype: {dtype}, head_dim: {d}, num_heads: {nh}")
    print(f"  warmup: {args.warmup}, rep: {args.rep}")
    print(f"  triton.testing available: {_USE_TRITON_BENCH}")

    all_results: List[BenchResult] = []

    # --- Baseline ---
    print("\n>>> Running baseline (FP16 memcpy)...")
    baseline_results = bench_baseline(
        num_tokens_list, nh, d, device, dtype, args.warmup, args.rep
    )
    all_results.extend(baseline_results)
    print_results(baseline_results, "Baseline: FP16 memory copy")

    # --- E2E: integer bit-widths ---
    for variant in args.variants:
        for bits in args.bits:
            bits_int = int(bits)
            print(f"\n>>> Running {variant} {bits_int}-bit e2e...")
            r = bench_e2e(variant, bits_int, num_tokens_list, nh, d, device, dtype,
                          args.warmup, args.rep)
            all_results.extend(r)
            print_results(r, f"E2E: {variant} {bits_int}-bit")

    # --- E2E: mixed precision ---
    for bits in args.mixed_bits:
        print(f"\n>>> Running mixed {bits}-bit e2e...")
        r = bench_e2e("mixed", bits, num_tokens_list, nh, d, device, dtype,
                      args.warmup, args.rep)
        all_results.extend(r)
        print_results(r, f"E2E: mixed {bits}-bit")

    # --- KV cache decode pattern ---
    print("\n>>> Running KV cache decode pattern simulation...")
    kv_results = []
    # Always include baseline for comparison
    kv_results.extend(bench_baseline(
        buffer_sizes, nh, d, device, dtype, args.warmup, args.rep
    ))
    for variant in args.variants:
        for bits in args.bits:
            r = bench_kv_cache_pattern(
                variant, int(bits), buffer_sizes, nh, d, device, dtype,
                args.warmup, args.rep
            )
            kv_results.extend(r)
    for bits in args.mixed_bits:
        r = bench_kv_cache_pattern(
            "mixed", bits, buffer_sizes, nh, d, device, dtype,
            args.warmup, args.rep
        )
        kv_results.extend(r)
    all_results.extend(kv_results)
    print_results(
        [r for r in kv_results if r.operation in ("kv_read_full", "memcpy")],
        "KV Cache Decode: full buffer read latency"
    )
    print_kv_summary(kv_results)

    # --- Per-op breakdown (optional) ---
    if args.breakdown:
        for bits in args.bits:
            tq = TurboQuantMSE(head_dim=d, bits=int(bits), device=device)
            x = torch.randn(4096, nh, d, dtype=dtype, device=device)
            print(f"\n>>> Running MSE {int(bits)}-bit op breakdown (4096 tokens)...")
            bd = bench_ops_breakdown(tq, x, device, args.warmup, args.rep)
            all_results.extend(bd)
            print_results(bd, f"Op breakdown: MSE {int(bits)}-bit, 4096 tokens × {nh} heads")

    # --- Save ---
    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        print(f"\nResults saved to {args.output}")

    print(f"\n{'=' * 70}")
    print(f"  Benchmark complete. {len(all_results)} measurements.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
