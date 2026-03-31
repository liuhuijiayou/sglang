#!/usr/bin/env python3
"""
TurboQuant compression ratio calculator.

Computes theoretical and actual memory usage for TurboQuant KV cache
compression vs FP16 baseline, clearly showing the gap due to missing
bit-packing in the current MVP.

Usage:
  python bench_compression.py
  python bench_compression.py --bits 2 3 4 --head-dim 128 --num-kv-heads 8 --num-layers 32 --seq-len 8192
  python bench_compression.py --output compression_results.csv
"""

import argparse
import csv
import json


def compute_compression(
    bits: int,
    head_dim: int,
    num_kv_heads: int,
    num_layers: int,
    seq_len: int,
) -> dict:
    """Compute memory usage for one configuration.

    Returns a dict with all computed values.
    """
    # ---------------------------------------------------------------
    # FP16 baseline: 2 bytes per element
    # Per token per head: head_dim * 2 bytes (K) + head_dim * 2 bytes (V)
    # ---------------------------------------------------------------
    fp16_per_token_per_head_kv = head_dim * 2 * 2  # K + V, 2 bytes each
    fp16_per_token = fp16_per_token_per_head_kv * num_kv_heads
    fp16_total = fp16_per_token * seq_len * num_layers

    # ---------------------------------------------------------------
    # TurboQuant THEORETICAL (with bit-packing):
    # Per token per head:
    #   indices: head_dim * bits / 8 bytes (K) + same for V
    #   norms:   2 bytes (fp16) for K + 2 bytes for V
    # ---------------------------------------------------------------
    # TurboQuant (with bit-packing, current implementation):
    # Per token per head:
    #   indices: head_dim * bits / 8 bytes (bit-packed into uint8)
    #   norms:   2 bytes (fp16)
    # ---------------------------------------------------------------
    packed_indices_per_head = (head_dim * bits + 7) // 8  # ceiling division
    norm_per_head = 2  # fp16
    tq_per_token_per_head_kv = (packed_indices_per_head + norm_per_head) * 2  # K+V
    tq_per_token = tq_per_token_per_head_kv * num_kv_heads
    tq_total = tq_per_token * seq_len * num_layers

    # ---------------------------------------------------------------
    # Rotation matrix overhead (shared, not per-token)
    # One (head_dim x head_dim) float32 matrix
    # ---------------------------------------------------------------
    rotation_overhead = head_dim * head_dim * 4  # float32

    # ---------------------------------------------------------------
    # Codebook overhead (shared, not per-token)
    # 2^bits centroids + (2^bits - 1) boundaries, all float32
    # ---------------------------------------------------------------
    num_levels = 1 << bits
    codebook_overhead = (num_levels + num_levels - 1) * 4  # float32

    shared_overhead = rotation_overhead + codebook_overhead

    return {
        "bits": bits,
        "head_dim": head_dim,
        "num_kv_heads": num_kv_heads,
        "num_layers": num_layers,
        "seq_len": seq_len,
        # Per-token per-head (bytes, K only)
        "fp16_per_head_K_bytes": head_dim * 2,
        "tq_per_head_K_bytes": packed_indices_per_head + norm_per_head,
        # Per-token (bytes, K+V)
        "fp16_per_token_bytes": fp16_per_token,
        "tq_per_token_bytes": tq_per_token,
        # Total (bytes)
        "fp16_total_MB": fp16_total / (1024 * 1024),
        "tq_total_MB": tq_total / (1024 * 1024),
        # Compression ratio
        "compression_ratio": fp16_total / tq_total if tq_total > 0 else float("inf"),
        # Overhead
        "shared_overhead_KB": shared_overhead / 1024,
        # Details
        "packed_indices_bytes_per_head": packed_indices_per_head,
        "norm_bytes_per_head": norm_per_head,
    }


def print_report(configs: list):
    """Pretty-print a comparison table."""
    print("=" * 80)
    print("TurboQuant Compression Ratio Report (with bit-packing)")
    print("=" * 80)

    for c in configs:
        print(f"\n--- {c['bits']}-bit, d={c['head_dim']}, "
              f"heads={c['num_kv_heads']}, layers={c['num_layers']}, "
              f"seq_len={c['seq_len']} ---")
        print(f"  FP16 baseline:")
        print(f"    Per head (K only) : {c['fp16_per_head_K_bytes']} bytes")
        print(f"    Per token (K+V)   : {c['fp16_per_token_bytes']} bytes")
        print(f"    Total KV cache    : {c['fp16_total_MB']:.1f} MB")
        print(f"  TurboQuant {c['bits']}-bit (bit-packed):")
        print(f"    Per head (K only) : {c['tq_per_head_K_bytes']} bytes  "
              f"({c['packed_indices_bytes_per_head']}B packed idx + "
              f"{c['norm_bytes_per_head']}B norm)")
        print(f"    Per token (K+V)   : {c['tq_per_token_bytes']} bytes")
        print(f"    Total KV cache    : {c['tq_total_MB']:.1f} MB")
        print(f"    Compression ratio : {c['compression_ratio']:.2f}x")
        print(f"    Shared overhead   : {c['shared_overhead_KB']:.1f} KB "
              f"(rotation matrix + codebook, amortized)")


def main():
    parser = argparse.ArgumentParser(description="TurboQuant compression ratio calculator")
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-kv-heads", type=int, default=8,
                        help="Number of KV heads (GQA). Llama-3.1-8B uses 8.")
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--seq-len", type=int, nargs="+", default=[4096, 8192, 16384],
                        help="Sequence lengths to evaluate")
    parser.add_argument("--output", type=str, default="results_compression.csv")
    args = parser.parse_args()

    configs = []
    for bits in args.bits:
        for seq_len in args.seq_len:
            c = compute_compression(
                bits=bits,
                head_dim=args.head_dim,
                num_kv_heads=args.num_kv_heads,
                num_layers=args.num_layers,
                seq_len=seq_len,
            )
            configs.append(c)

    print_report(configs)

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(configs[0].keys()))
            writer.writeheader()
            writer.writerows(configs)
        print(f"\nResults saved to {args.output}")

        json_out = args.output.replace(".csv", ".json")
        with open(json_out, "w") as f:
            json.dump(configs, f, indent=2)
        print(f"Results saved to {json_out}")


if __name__ == "__main__":
    main()
