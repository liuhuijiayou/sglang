#!/usr/bin/env python3
"""
TurboQuant distortion benchmark — MSE + Prod + Mixed.

Validates all three quantizer variants against paper bounds:
  - TurboQuantMSE  (Algorithm 1): MSE distortion, Theorem 1
  - TurboQuantProd (Algorithm 2): unbiased inner product, Theorem 2
  - TurboQuantMixed: non-integer bit-widths (2.5, 3.5)

Metrics:
  Common:  MSE, relative L2, cosine similarity
  Prod:    inner-product bias (should be ~0), inner-product distortion D_prod
  All:     dot-product error |<y,x> - <y,x_hat>|^2 / ||y||^2

Usage:
  # MSE only (default)
  python bench_mse.py --bits 2 3 4

  # MSE + Prod side-by-side
  python bench_mse.py --bits 2 3 4 --variants mse prod

  # Include non-integer bit-widths
  python bench_mse.py --bits 2 3 4 --mixed-bits 2.5 3.5

  # With real KV vectors
  python bench_mse.py --bits 2 3 4 --variants mse prod --real-kv --model-path /models/Qwen2.5-7B-Instruct
"""

import argparse
import csv
import json
import math
import sys
import time
from typing import Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Import quantizers (handles missing sglang deps gracefully)
# ---------------------------------------------------------------------------
def _load_quantizers():
    """Import TurboQuant quantizer classes."""
    try:
        from sglang.srt.layers.quantization.turboquant.quantizer import (
            TurboQuantMSE, TurboQuantProd, TurboQuantMixed, create_quantizer,
        )
        return TurboQuantMSE, TurboQuantProd, TurboQuantMixed, create_quantizer
    except ImportError:
        import importlib, types, os
        base = os.path.join(os.path.dirname(__file__), "..", "..", "python")
        if os.path.isdir(base):
            sys.path.insert(0, os.path.abspath(base))
        for pkg in [
            "sglang", "sglang.srt", "sglang.srt.layers",
            "sglang.srt.layers.quantization",
            "sglang.srt.layers.quantization.turboquant",
        ]:
            if pkg not in sys.modules:
                sys.modules[pkg] = types.ModuleType(pkg)
                if "." in pkg:
                    parent = pkg.rsplit(".", 1)[0]
                    setattr(sys.modules[parent], pkg.split(".")[-1], sys.modules[pkg])

        tq_base = os.path.join(base, "sglang/srt/layers/quantization/turboquant")
        for name in ["codebook", "rotation", "packing", "qjl", "quantizer"]:
            fqn = f"sglang.srt.layers.quantization.turboquant.{name}"
            path = os.path.join(tq_base, f"{name}.py")
            spec = importlib.util.spec_from_file_location(fqn, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[fqn] = mod
            spec.loader.exec_module(mod)

        q = sys.modules["sglang.srt.layers.quantization.turboquant.quantizer"]
        return q.TurboQuantMSE, q.TurboQuantProd, q.TurboQuantMixed, q.create_quantizer


# ---------------------------------------------------------------------------
# Naive scalar quantizer baseline
# ---------------------------------------------------------------------------
class NaiveScalarQuantizer:
    """Uniform scalar quantizer without rotation. Baseline."""

    def __init__(self, head_dim: int, bits: int, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = bits
        self.num_levels = 1 << bits

    def compress(self, x):
        norms = x.float().norm(dim=-1)
        x_unit = x.float() / norms.unsqueeze(-1).clamp(min=1e-12)
        max_val = 3.0 / math.sqrt(self.head_dim)
        x_clamp = x_unit.clamp(-max_val, max_val)
        x_scaled = (x_clamp + max_val) / (2 * max_val)
        indices = (x_scaled * (self.num_levels - 1)).round().long().clamp(0, self.num_levels - 1)
        return indices.to(torch.uint8), norms.to(torch.float16)

    def decompress(self, indices, norms, out_dtype=torch.float32):
        max_val = 3.0 / math.sqrt(self.head_dim)
        x_hat = indices.float() / (self.num_levels - 1) * (2 * max_val) - max_val
        x_hat = x_hat * norms.float().unsqueeze(-1)
        return x_hat.to(out_dtype)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    label: str,
    n_y_probes: int = 200,
) -> Dict:
    """Compute distortion metrics."""
    x_f = x.float()
    xh_f = x_hat.float()
    d = x_f.shape[-1]

    mse_per_vec = (x_f - xh_f).pow(2).sum(dim=-1)
    mse = mse_per_vec.mean().item()

    x_norms = x_f.norm(dim=-1).clamp(min=1e-12)
    rel_l2 = ((x_f - xh_f).norm(dim=-1) / x_norms).mean().item()

    cos_sim = torch.nn.functional.cosine_similarity(x_f, xh_f, dim=-1).mean().item()

    # Inner-product error: E[|<y,x> - <y,x_hat>|^2] / ||y||^2
    y = torch.randn(n_y_probes, d, device=x_f.device)
    y_norms_sq = y.pow(2).sum(dim=-1)
    dot_x = x_f @ y.t()
    dot_xh = xh_f @ y.t()
    dot_err_sq = (dot_x - dot_xh).pow(2)
    dot_err = (dot_err_sq / y_norms_sq.unsqueeze(0)).mean().item()

    # Per-vector scale ratio: ||x_hat|| / ||x|| - 1 (proxy for systematic bias)
    xh_norms = xh_f.norm(dim=-1).clamp(min=1e-12)
    scale_bias = (xh_norms / x_norms).mean().item() - 1.0

    return {
        "method": label,
        "mse": mse,
        "relative_l2": rel_l2,
        "cosine_similarity": cos_sim,
        "dot_product_error": dot_err,
        "scale_bias": scale_bias,
    }


def print_result(m: Dict, paper_bound=None, lower_bound=None, show_bias=False):
    """Print one result line."""
    ratio_str = f"{m['mse']/lower_bound:.2f}x" if lower_bound else ""
    line = (
        f"  {m['method']:<28s} "
        f"MSE={m['mse']:.5f}  cos={m['cosine_similarity']:.5f}  "
        f"relL2={m['relative_l2']:.5f}  dot_err={m['dot_product_error']:.6f}"
    )
    if paper_bound:
        line += f"  paper={paper_bound}  ratio={ratio_str}"
    if show_bias:
        line += f"  scale_bias={m['scale_bias']:+.4f}"
    print(line)


# ---------------------------------------------------------------------------
# Real KV vector extraction
# ---------------------------------------------------------------------------
def extract_real_kv_vectors(model_path: str, device: str = "cuda") -> Optional[torch.Tensor]:
    """Extract real KV vectors from one forward pass."""
    print(f"Loading model from {model_path}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()

    prompt = (
        "The quick brown fox jumps over the lazy dog. " * 20
        + "In machine learning, attention mechanisms allow models to focus on "
        + "different parts of the input sequence. The key-value cache stores "
        + "intermediate representations for efficient autoregressive generation."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    kv_vectors = []
    def hook_fn(module, input, output):
        if hasattr(output, '__len__') and len(output) >= 3 and output[2] is not None:
            past = output[2]
            if isinstance(past, tuple) and len(past) >= 1:
                k = past[0]
                kv_vectors.append(k[0].detach().cpu().float())

    # Register hook on first attention layer
    attn_layers = []
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1] if "." in name else name
        if leaf == "self_attn":
            attn_layers.append((name, module))
    if not attn_layers:
        for name, module in model.named_modules():
            leaf = name.rsplit(".", 1)[-1] if "." in name else name
            if "attn" in leaf.lower() and "proj" not in leaf:
                attn_layers.append((name, module))

    if attn_layers:
        handle = attn_layers[0][1].register_forward_hook(hook_fn)
        print(f"  Hook on: {attn_layers[0][0]}")
    else:
        print("  WARNING: no attention layer found")
        del model; torch.cuda.empty_cache()
        return None

    with torch.no_grad():
        model(**inputs, use_cache=True)

    handle.remove()
    del model; torch.cuda.empty_cache()

    if kv_vectors:
        kv = kv_vectors[0]  # (num_kv_heads, seq_len, head_dim)
        kv = kv.permute(1, 0, 2)  # (seq_len, num_kv_heads, head_dim)
        print(f"  KV shape: {kv.shape}")
        return kv
    return None


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def bench_on_vectors(
    x: torch.Tensor,
    bits_list: List,
    variants: List[str],
    mixed_bits: List[float],
    device: str,
    vector_type: str,
    create_quantizer,
    results: List[Dict],
):
    """Run benchmark on a set of vectors for all bit-widths and variants."""
    d = x.shape[-1]

    # Paper bounds
    paper_mse = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
    lower_mse = {b: 1.0 / (4 ** b) for b in range(1, 9)}
    # Theorem 2: D_prod <= (sqrt(3)*pi^2/d) * 4^{-b} * ||y||^2
    # For ||y||=1: D_prod <= sqrt(3)*pi^2/d * 4^{-b}
    paper_dprod = {b: math.sqrt(3) * math.pi**2 / d / (4**b) for b in range(1, 9)}

    for bits in bits_list:
        bits_int = int(bits)

        for variant in variants:
            tq = create_quantizer(head_dim=d, bits=bits_int, variant=variant, device=device)
            if tq.has_residual_norm:
                packed, norms, rnorms = tq.compress(x)
                x_hat = tq.decompress(packed, norms, residual_norms=rnorms, out_dtype=torch.float32)
            else:
                packed, norms = tq.compress(x)
                x_hat = tq.decompress(packed, norms, out_dtype=torch.float32)

            label = f"TurboQuant_{variant}-{bits}bit"
            m = compute_metrics(x, x_hat, label)
            m["vector_type"] = vector_type
            m["bits"] = bits
            m["variant"] = variant
            m["dim"] = d

            if variant == "mse":
                m["paper_mse_bound"] = paper_mse.get(bits_int)
                m["lower_mse_bound"] = lower_mse.get(bits_int)
                m["paper_dprod_bound"] = None
            else:
                m["paper_mse_bound"] = None
                m["lower_mse_bound"] = None
                m["paper_dprod_bound"] = paper_dprod.get(bits_int)

            results.append(m)
            show_bias = (variant == "prod")
            pb = paper_mse.get(bits_int) if variant == "mse" else None
            lb = lower_mse.get(bits_int) if variant == "mse" else None
            print_result(m, paper_bound=pb, lower_bound=lb, show_bias=show_bias)

            if variant == "prod":
                # Extra: Prod-specific metrics
                dprod = m["dot_product_error"]
                dprod_bound = paper_dprod.get(bits_int)
                within = dprod <= dprod_bound * 1.3 if dprod_bound else "N/A"
                print(f"    D_prod={dprod:.6f}  bound={dprod_bound:.6f}  within_1.3x={within}")

        # Naive baseline (only for integer bits, once per bit-width)
        if bits_int == bits:  # skip for non-integer
            nq = NaiveScalarQuantizer(head_dim=d, bits=bits_int, device=device)
            idx_n, norms_n = nq.compress(x)
            x_hat_n = nq.decompress(idx_n, norms_n)
            m_n = compute_metrics(x, x_hat_n, f"Naive-{bits}bit")
            m_n["vector_type"] = vector_type
            m_n["bits"] = bits
            m_n["variant"] = "naive"
            m_n["dim"] = d
            m_n["paper_mse_bound"] = None
            m_n["lower_mse_bound"] = lower_mse.get(bits_int)
            m_n["paper_dprod_bound"] = None
            results.append(m_n)
            print_result(m_n)

    # Non-integer (mixed) bit-widths
    for bits in mixed_bits:
        tq = create_quantizer(head_dim=d, bits=bits, variant="mse", device=device)
        packed, norms = tq.compress(x)
        x_hat = tq.decompress(packed, norms, out_dtype=torch.float32)
        label = f"TurboQuant_mixed-{bits}bit"
        m = compute_metrics(x, x_hat, label)
        m["vector_type"] = vector_type
        m["bits"] = bits
        m["variant"] = "mixed"
        m["dim"] = d
        m["paper_mse_bound"] = None
        m["lower_mse_bound"] = 1.0 / (4 ** bits)
        m["paper_dprod_bound"] = None
        results.append(m)
        print_result(m)


def run_benchmark(args):
    _, _, _, create_quantizer = _load_quantizers()
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    results = []

    print("=" * 72)
    print("TurboQuant Distortion Benchmark")
    print(f"  variants: {args.variants}, bits: {args.bits}, mixed: {args.mixed_bits}")
    print(f"  dim={args.dim}, n={args.num_vectors}, device={device}")
    print("=" * 72)

    # --- Random unit vectors (paper setting) ---
    print(f"\n{'='*60}")
    print(f"  Random unit vectors (S^{{d-1}}, d={args.dim}, n={args.num_vectors})")
    print(f"{'='*60}")
    x_unit = torch.randn(args.num_vectors, args.dim, device=device)
    x_unit = x_unit / x_unit.norm(dim=-1, keepdim=True)

    bench_on_vectors(
        x_unit, args.bits, args.variants, args.mixed_bits,
        device, "random_unit", create_quantizer, results,
    )

    # FP16 reference
    x_hat_fp16 = x_unit.half().float()
    m16 = compute_metrics(x_unit, x_hat_fp16, "FP16-reference")
    m16.update(vector_type="random_unit", bits=16, variant="fp16", dim=args.dim,
               paper_mse_bound=None, lower_mse_bound=None, paper_dprod_bound=None)
    results.append(m16)
    print(f"  {'FP16-reference':<28s} MSE={m16['mse']:.8f}  cos={m16['cosine_similarity']:.8f}")

    # --- Random non-unit vectors ---
    print(f"\n{'='*60}")
    print(f"  Random non-unit vectors (d={args.dim}, n={args.num_vectors})")
    print(f"{'='*60}")
    x_nonunit = torch.randn(args.num_vectors, args.dim, device=device) * 2.5

    bench_on_vectors(
        x_nonunit, args.bits, args.variants, args.mixed_bits,
        device, "random_nonunit", create_quantizer, results,
    )

    # --- Real KV vectors ---
    if args.real_kv and args.model_path:
        print(f"\n{'='*60}")
        print(f"  Real KV vectors from {args.model_path}")
        print(f"{'='*60}")
        kv = extract_real_kv_vectors(args.model_path, device=device)
        if kv is not None:
            kv_flat = kv.reshape(-1, kv.shape[-1]).to(device)
            real_dim = kv_flat.shape[-1]
            print(f"  dim={real_dim}, n={kv_flat.shape[0]}")
            bench_on_vectors(
                kv_flat, args.bits, args.variants, args.mixed_bits,
                device, "real_kv", create_quantizer, results,
            )

    # --- MSE vs Prod comparison table ---
    if "mse" in args.variants and "prod" in args.variants:
        print(f"\n{'='*60}")
        print("  MSE vs Prod comparison (random_unit)")
        print(f"{'='*60}")
        print(f"  {'bits':<6s} {'MSE_mse':>10s} {'MSE_prod':>10s} "
              f"{'Dprod':>12s} {'Dprod_bound':>12s} {'within?':>8s}")
        print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*8}")
        for bits in args.bits:
            mse_r = [r for r in results
                     if r["bits"]==bits and r["variant"]=="mse" and r["vector_type"]=="random_unit"]
            prod_r = [r for r in results
                      if r["bits"]==bits and r["variant"]=="prod" and r["vector_type"]=="random_unit"]
            if mse_r and prod_r:
                m, p = mse_r[0], prod_r[0]
                dprod_bound = p.get("paper_dprod_bound", 0) or 0
                within = p["dot_product_error"] <= dprod_bound * 1.3
                print(f"  {bits:<6}  {m['mse']:>9.5f}  {p['mse']:>9.5f}  "
                      f"{p['dot_product_error']:>11.6f}  {dprod_bound:>11.6f}  "
                      f"{'OK' if within else 'FAIL':>7s}")

        # Multi-seed unbiasedness test for Prod
        print(f"\n  Prod unbiasedness test (multi-seed, 20 seeds x 500 vectors):")
        d = args.dim
        n_seeds = 20
        n_test = min(500, args.num_vectors)
        x_test = x_unit[:n_test]
        y_test = torch.randn(50, d, device=device)
        y_test = y_test / y_test.norm(dim=-1, keepdim=True)  # unit y
        true_dots = (x_test.float() @ y_test.t())  # (n_test, 50)

        for bits in args.bits:
            ip_estimates = []
            for seed in range(42, 42 + n_seeds):
                from sglang.srt.layers.quantization.turboquant.quantizer import TurboQuantProd
                tq_p = TurboQuantProd(head_dim=d, bits=int(bits), seed=seed, device=device)
                packed_p, norms_p, rnorms_p = tq_p.compress(x_test)
                xh_p = tq_p.decompress(packed_p, norms_p, residual_norms=rnorms_p, out_dtype=torch.float32)
                est_dots = (xh_p @ y_test.t())
                ip_estimates.append(est_dots)

            avg_est = torch.stack(ip_estimates).mean(dim=0)  # (n_test, 50)
            bias = (avg_est - true_dots).mean().item()
            rel_bias = abs(bias) / true_dots.abs().mean().item()
            print(f"    b={bits}: mean_bias={bias:+.6f}  "
                  f"rel_bias={rel_bias:.4f}  "
                  f"{'UNBIASED' if rel_bias < 0.05 else 'BIASED'}")

    # --- Save ---
    fieldnames = [
        "method", "vector_type", "variant", "bits", "dim",
        "mse", "relative_l2", "cosine_similarity", "dot_product_error", "scale_bias",
        "paper_mse_bound", "lower_mse_bound", "paper_dprod_bound",
    ]

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV: {args.output}")

    json_path = args.output.replace(".csv", ".json") if args.output else None
    if json_path:
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"JSON: {json_path}")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="TurboQuant distortion benchmark")
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--mixed-bits", type=float, nargs="+", default=[],
                        help="Non-integer bit-widths (e.g., 2.5 3.5)")
    parser.add_argument("--variants", type=str, nargs="+", default=["mse"],
                        choices=["mse", "prod"],
                        help="Quantizer variants to benchmark")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num-vectors", type=int, default=10000)
    parser.add_argument("--real-kv", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output", type=str, default="results_mse.csv")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
