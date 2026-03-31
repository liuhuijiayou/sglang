#!/usr/bin/env python3
"""
Aggregate and compare TurboQuant E2E serving benchmark results.

Reads the JSONL output files from bench_e2e_serving.sh and produces:
  1. Side-by-side throughput/latency comparison table
  2. Relative performance (normalized to FP16 baseline)
  3. JSON summary for downstream analysis

Usage:
    # After running bench_e2e_serving.sh:
    python compare_serving_results.py --results-dir results/serving/
    python compare_serving_results.py --results-dir results/serving/ --output comparison.json
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List


def parse_bench_serving_jsonl(filepath: str) -> Dict:
    """Parse the last line of a bench_serving JSONL output file."""
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        return {}
    return json.loads(lines[-1])


def collect_results(results_dir: str, profile: str = "default") -> Dict[str, Dict]:
    """Collect all benchmark results matching the profile."""
    # Try profile-specific pattern first, then generic
    pattern = os.path.join(results_dir, f"bench_{profile}_*.jsonl")
    results = {}

    files = sorted(glob.glob(pattern))
    if not files:
        # Fallback: try bench_*.jsonl (no profile prefix)
        pattern = os.path.join(results_dir, "bench_*.jsonl")
        files = sorted(glob.glob(pattern))
        prefix = "bench_"
    else:
        prefix = f"bench_{profile}_"

    for filepath in files:
        filename = os.path.basename(filepath)
        label = filename.replace(prefix, "").replace(".jsonl", "")
        try:
            data = parse_bench_serving_jsonl(filepath)
            if data:
                results[label] = data
        except Exception as e:
            print(f"Warning: Failed to parse {filepath}: {e}", file=sys.stderr)

    return results


def print_comparison(results: Dict[str, Dict], baseline_key: str = "fp16"):
    """Print a formatted comparison table."""
    if not results:
        print("No results found.")
        return

    baseline = results.get(baseline_key, {})
    baseline_out_tps = baseline.get("output_throughput", 1)

    # Key metrics to compare
    metrics = [
        ("output_throughput", "Out tok/s", ".1f"),
        ("input_throughput", "In tok/s", ".1f"),
        ("total_throughput", "Total tok/s", ".1f"),
        ("mean_ttft_ms", "TTFT mean(ms)", ".2f"),
        ("median_ttft_ms", "TTFT med(ms)", ".2f"),
        ("p99_ttft_ms", "TTFT p99(ms)", ".2f"),
        ("mean_tpot_ms", "TPOT mean(ms)", ".2f"),
        ("p99_tpot_ms", "TPOT p99(ms)", ".2f"),
        ("mean_itl_ms", "ITL mean(ms)", ".2f"),
        ("p99_itl_ms", "ITL p99(ms)", ".2f"),
        ("mean_e2e_latency_ms", "E2E mean(ms)", ".1f"),
        ("p99_e2e_latency_ms", "E2E p99(ms)", ".1f"),
    ]

    labels = list(results.keys())

    # ---- Absolute values table ----
    print("=" * (25 + 15 * len(labels)))
    print("  Absolute Metrics")
    print("=" * (25 + 15 * len(labels)))

    header = f"{'Metric':<25}"
    for label in labels:
        header += f"{label:>15}"
    print(header)
    print("-" * (25 + 15 * len(labels)))

    for key, display_name, fmt in metrics:
        row = f"{display_name:<25}"
        for label in labels:
            val = results[label].get(key, 0)
            row += f"{val:>15{fmt}}"
        print(row)

    # ---- Relative to baseline table ----
    if baseline_key in results:
        print()
        print("=" * (25 + 15 * len(labels)))
        print(f"  Relative to {baseline_key} (higher = better for throughput, lower = better for latency)")
        print("=" * (25 + 15 * len(labels)))

        header = f"{'Metric':<25}"
        for label in labels:
            header += f"{label:>15}"
        print(header)
        print("-" * (25 + 15 * len(labels)))

        throughput_metrics = {"output_throughput", "input_throughput", "total_throughput"}
        for key, display_name, fmt in metrics:
            baseline_val = baseline.get(key, 0)
            if baseline_val == 0:
                continue
            row = f"{display_name:<25}"
            for label in labels:
                val = results[label].get(key, 0)
                ratio = val / baseline_val
                if key in throughput_metrics:
                    # Higher is better: show as ratio
                    marker = "+" if ratio >= 1.0 else ""
                    pct = (ratio - 1) * 100
                    row += f"{marker}{pct:>13.1f}%"
                else:
                    # Lower is better (latency): show as ratio
                    marker = "+" if ratio > 1.0 else ""
                    pct = (ratio - 1) * 100
                    row += f"{marker}{pct:>13.1f}%"
            print(row)

    # ---- Key takeaway ----
    print()
    print("Key takeaway:")
    for label in labels:
        if label == baseline_key:
            continue
        out_tps = results[label].get("output_throughput", 0)
        ratio = out_tps / baseline_out_tps if baseline_out_tps > 0 else 0
        pct_change = (ratio - 1) * 100
        direction = "faster" if pct_change > 0 else "slower"
        print(f"  {label}: {out_tps:.1f} tok/s ({pct_change:+.1f}% {direction} than {baseline_key})")


def main():
    parser = argparse.ArgumentParser(description="Compare TurboQuant serving benchmark results")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--profile", type=str, default="default",
                        help="Profile name matching bench_e2e_serving.sh output files")
    parser.add_argument("--baseline", type=str, default="fp16")
    parser.add_argument("--output", type=str, default=None,
                        help="Save comparison to JSON")
    args = parser.parse_args()

    results = collect_results(args.results_dir, args.profile)

    if not results:
        print(f"No results found in {args.results_dir} for profile '{args.profile}'")
        print(f"Expected files like: bench_{args.profile}_fp16.jsonl")
        return

    print(f"Found {len(results)} configurations: {', '.join(results.keys())}")
    print()
    print_comparison(results, baseline_key=args.baseline)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nRaw results saved to {args.output}")


if __name__ == "__main__":
    main()
