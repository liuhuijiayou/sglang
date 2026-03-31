#!/usr/bin/env python3
"""
Needle-in-a-Haystack (NIAH) evaluation for SGLang server.

Self-contained script — does NOT depend on any sglang imports.
Sends requests to a running SGLang server via the OpenAI-compatible API.

The needle: "The best thing to do in San Francisco is eat a sandwich
and sit in Dolores Park on a sunny day."
(Same as TurboQuant paper's evaluation)

Usage:
  # Start server first, then:
  python bench_niah.py --port 30001
  python bench_niah.py --port 30001 --context-lengths 1024 4096 8192 --depth-percents 25 50 75
  python bench_niah.py --port 30001 --output results_niah.json --label "turboquant_4bit"
"""

import argparse
import json
import random
import string
import time
from typing import Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Constants matching the TurboQuant paper's NIAH setup
# ---------------------------------------------------------------------------
DEFAULT_NEEDLE = (
    "The best thing to do in San Francisco is eat a sandwich "
    "and sit in Dolores Park on a sunny day."
)
DEFAULT_QUESTION = (
    "What is the best thing to do in San Francisco? "
    "Answer with the exact sentence from the context."
)
KEY_PHRASES = ["sandwich", "Dolores Park", "sunny day", "San Francisco"]

# Filler text for building haystacks
FILLER_PARAGRAPH = (
    "The history of computing is a fascinating tale that spans centuries. "
    "From the earliest mechanical calculators to modern quantum computers, "
    "humans have continually sought to automate complex calculations. "
    "Charles Babbage designed the Analytical Engine in the 1830s, "
    "which many consider the first general-purpose computer. "
    "Ada Lovelace wrote what is often considered the first computer program "
    "for this machine. The development of electronic computers in the 1940s "
    "revolutionized science, business, and daily life. "
    "Today, billions of transistors fit on a single chip, "
    "enabling artificial intelligence and machine learning at scale. "
)


# ---------------------------------------------------------------------------
# Haystack construction
# ---------------------------------------------------------------------------
def build_haystack(target_tokens: int, chars_per_token: float = 4.0) -> str:
    """Build filler text of approximately target_tokens length."""
    target_chars = int(target_tokens * chars_per_token)
    repeats = (target_chars // len(FILLER_PARAGRAPH)) + 1
    text = (FILLER_PARAGRAPH * repeats)[:target_chars]
    return text


def insert_needle(haystack: str, needle: str, depth_percent: float) -> str:
    """Insert needle at the specified depth percentage in the haystack."""
    if depth_percent <= 0:
        return needle + "\n" + haystack
    if depth_percent >= 100:
        return haystack + "\n" + needle

    insert_pos = int(len(haystack) * depth_percent / 100)
    # Find a sentence boundary near the insert position
    for i in range(insert_pos, min(insert_pos + 200, len(haystack))):
        if haystack[i] == ".":
            insert_pos = i + 1
            break
    return haystack[:insert_pos] + "\n" + needle + "\n" + haystack[insert_pos:]


# ---------------------------------------------------------------------------
# Server interaction
# ---------------------------------------------------------------------------
def query_server(
    base_url: str,
    context: str,
    question: str,
    max_tokens: int = 128,
    temperature: float = 0.0,
) -> dict:
    """Send a completion request to the SGLang server.

    Returns dict with 'response' (str) and 'latency_ms' (float).
    """
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": f"<context>\n{context}\n</context>\n\n{question}",
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    t0 = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.time() - t0) * 1000
        content = data["choices"][0]["message"]["content"]
        return {"response": content, "latency_ms": latency_ms}
    except Exception as e:
        return {"response": f"ERROR: {e}", "latency_ms": (time.time() - t0) * 1000}


def score_response(response: str) -> float:
    """Score the response by checking for key phrases from the needle.

    Returns a score in [0, 1]. 1.0 means all key phrases found.
    Paper reference: TurboQuant achieves 0.997 recall on NIAH.
    """
    response_lower = response.lower()
    matches = sum(1 for phrase in KEY_PHRASES if phrase.lower() in response_lower)
    return matches / len(KEY_PHRASES)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def run_niah(
    base_url: str,
    context_lengths: List[int],
    depth_percents: List[float],
    num_repeats: int = 1,
    label: str = "default",
) -> dict:
    """Run the full NIAH evaluation grid.

    Returns a dict with per-cell scores and overall recall.
    """
    results = {
        "label": label,
        "context_lengths": context_lengths,
        "depth_percents": depth_percents,
        "grid": {},  # {ctx_len: {depth: {"score": float, "latency_ms": float}}}
        "all_scores": [],
        "all_latencies": [],
    }

    total_tests = len(context_lengths) * len(depth_percents) * num_repeats
    print(f"\nRunning NIAH: {len(context_lengths)} ctx_lens x "
          f"{len(depth_percents)} depths x {num_repeats} repeats = {total_tests} tests")
    print(f"Label: {label}")

    test_num = 0
    for ctx_len in context_lengths:
        results["grid"][ctx_len] = {}
        for depth in depth_percents:
            scores = []
            latencies = []
            for rep in range(num_repeats):
                test_num += 1
                haystack = build_haystack(ctx_len)
                context = insert_needle(haystack, DEFAULT_NEEDLE, depth)

                result = query_server(base_url, context, DEFAULT_QUESTION)
                score = score_response(result["response"])
                scores.append(score)
                latencies.append(result["latency_ms"])

                status = "OK" if score >= 0.75 else "MISS"
                response_preview = result["response"][:200].replace("\n", " ")
                print(
                    f"  [{test_num}/{total_tests}] ctx={ctx_len:>6} depth={depth:>5.1f}% "
                    f"score={score:.2f} latency={result['latency_ms']:.0f}ms [{status}]"
                )
                print(f"    Response: {response_preview}")

            avg_score = sum(scores) / len(scores)
            avg_latency = sum(latencies) / len(latencies)
            results["grid"][ctx_len][depth] = {
                "score": avg_score,
                "latency_ms": avg_latency,
            }
            results["all_scores"].append(avg_score)
            results["all_latencies"].append(avg_latency)

    # Overall recall
    results["overall_recall"] = (
        sum(results["all_scores"]) / len(results["all_scores"])
        if results["all_scores"]
        else 0.0
    )
    results["avg_latency_ms"] = (
        sum(results["all_latencies"]) / len(results["all_latencies"])
        if results["all_latencies"]
        else 0.0
    )

    # Print summary grid
    print(f"\n{'=' * 70}")
    print(f"NIAH Results Grid — {label}")
    print(f"{'=' * 70}")
    header = f"{'ctx_len':>8}"
    for d in depth_percents:
        header += f"  {d:>6.0f}%"
    print(header)
    print("-" * len(header))
    for ctx_len in context_lengths:
        row = f"{ctx_len:>8}"
        for d in depth_percents:
            s = results["grid"][ctx_len][d]["score"]
            marker = "  " if s >= 0.75 else " !"
            row += f"  {s:>5.2f}{marker}"
        print(row)
    print(f"\nOVERALL RECALL: {results['overall_recall']:.4f}")
    print(f"AVG LATENCY:    {results['avg_latency_ms']:.0f} ms")
    print(f"Paper reference: 0.997 (TurboQuant 4-bit, Llama-3.1-8B)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Needle-in-a-Haystack evaluation")
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--context-lengths", type=int, nargs="+",
                        default=[1024, 4096, 8192, 16384])
    parser.add_argument("--depth-percents", type=float, nargs="+",
                        default=[25, 50, 75])
    parser.add_argument("--num-repeats", type=int, default=1,
                        help="Number of repeats per (ctx_len, depth) pair")
    parser.add_argument("--label", type=str, default="default",
                        help="Label for this run (e.g., 'fp16_baseline', 'turboquant_4bit')")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--warmup", action="store_true",
                        help="Send a warmup request first")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    # Health check
    print(f"Checking server at {base_url}...")
    try:
        resp = requests.get(f"{base_url}/health", timeout=10)
        print(f"  Server status: {resp.status_code}")
    except Exception as e:
        print(f"  ERROR: Cannot reach server at {base_url}: {e}")
        print("  Make sure the SGLang server is running.")
        return

    # Warmup
    if args.warmup:
        print("Sending warmup request...")
        query_server(base_url, "Hello world. " * 100, "Say hello.", max_tokens=10)
        print("  Warmup done.")

    results = run_niah(
        base_url=base_url,
        context_lengths=args.context_lengths,
        depth_percents=args.depth_percents,
        num_repeats=args.num_repeats,
        label=args.label,
    )

    if args.output:
        # Convert grid keys to strings for JSON serialization
        serializable = {**results}
        serializable["grid"] = {
            str(k): {str(d): v for d, v in inner.items()}
            for k, inner in results["grid"].items()
        }
        with open(args.output, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
