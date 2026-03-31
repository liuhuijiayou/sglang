# TurboQuant Validation Guide

## One Command, Everything

```bash
docker exec -it sglang-dev bash
cd /workspace/sglang
export MODEL_PATH=/models/Qwen2.5-7B-Instruct

# Full validation (~60 min): MSE + compression + perf + NIAH + E2E serving
bash benchmark/turboquant/run_validation.sh

# Quick mode (~20 min): smaller sweeps everywhere
bash benchmark/turboquant/run_validation.sh --quick

# No server phases (~5 min): MSE + compression + perf only
bash benchmark/turboquant/run_validation.sh --skip-server
```

Model config (num_kv_heads, num_layers, head_dim) auto-detected from `config.json`.

---

## Phases

| # | Phase | What | Time | Server? |
|---|-------|------|------|---------|
| 1 | Distortion | MSE + Prod + Mixed on random/real KV vectors | ~3min | No |
| 2 | Compression | Theoretical compression ratios | ~1s | No |
| 2.5 | Perf | Decompress throughput + per-op breakdown | ~3min | No |
| 3 | NIAH | FP16, 4bit, 3bit, 4bit_prod, 3.5bit | ~25min | Yes |
| 4 | Serving | E2E throughput: FP16, 4bit, 3bit, 4bit_prod, 3.5bit, 2bit | ~30min | Yes |

## What Gets Validated

| Variant | Algorithm | Key Metric | Configs |
|---------|-----------|-----------|---------|
| MSE | 1 | MSE <= paper bound (Thm 1) | 2/3/4 bit |
| Prod | 2 | D_prod <= paper bound (Thm 2), unbiased | 3/4 bit |
| Mixed | Sec 6.4 | Non-integer bit-widths | 2.5/3.5 bit |
| Naive | baseline | Should be 2-3x worse than TurboQuant | 2/3/4 bit |

## Output Files

```
benchmark/turboquant/results/
  mse_random.csv / .json          # Phase 1: random vectors
  mse_with_real_kv.csv / .json    # Phase 1: real KV from model
  compression.csv                  # Phase 2
  perf.json / perf.txt            # Phase 2.5: decompress throughput
  niah_fp16_baseline.json / .txt   # Phase 3
  niah_turboquant_4bit.json
  niah_turboquant_3bit.json
  niah_turboquant_4bit_prod.json
  niah_turboquant_3.5bit.json
  server_*.log                     # Server logs (for debugging)
  serving/
    bench_fp16.jsonl / .txt        # Phase 4
    bench_turboquant_4bit.jsonl
    bench_turboquant_3bit.jsonl
    bench_turboquant_4bit_prod.jsonl
    bench_turboquant_3.5bit.jsonl
    bench_turboquant_2bit.jsonl
    comparison.json / .txt         # Phase 4 summary table
```

**Copy everything to me for analysis:**
```bash
tar czf turboquant_results.tar.gz benchmark/turboquant/results/
```

## Pass Criteria

**Phase 1 (Distortion):**
- MSE variant: ratio to lower bound <= 2.7x (Theorem 3)
- Prod variant: D_prod within paper bound x 1.3
- Prod unbiasedness: multi-seed rel_bias < 0.05
- Cosine sim: b=3 > 0.98, b=4 > 0.995

**Phase 3 (NIAH):**
- FP16 baseline recall should be high (>0.95) — validates test setup
- 4-bit recall should be close to FP16 (paper: 0.997)
- 3-bit may have slight degradation

**Phase 4 (Serving):**
- 4-bit throughput >= 85% of FP16
- 3-bit throughput >= 75% of FP16

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Unsupported kv_cache_dtype` | Code not synced |
| Server OOM | Lower `--mem-fraction-static` to 0.60 in script |
| NIAH all 0 | Check server log; run Phase 1 first |
| `--real-kv` skips | Check "Hook on:" message in output |
| Prod crash | Ensure quantizer.py has `output` param on Prod decompress |
