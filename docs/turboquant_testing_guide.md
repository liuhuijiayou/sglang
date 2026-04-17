# TurboQuant GPU Testing Guide

TurboQuant KV cache compression for SGLang. Branch: `feat/turboquant-kv-cache`

**Architecture**: WHT rotation + Lloyd-Max scalar quantization + per-vector norm + bit-packing.
Uses Walsh-Hadamard Transform (deterministic, O(d log d)) instead of random rotation matrix.
Incremental dequant with dirty tracking — only re-dequantizes newly written tokens.

## Prerequisites

- NVIDIA GPU with CUDA support
- SGLang installed from source (`pip install -e ".[all]"`)
- A chat model, e.g. `Qwen/Qwen2.5-7B-Instruct` or `meta-llama/Llama-3.2-1B-Instruct`

## 1. Unit Tests

```bash
# CPU tests (codebook, config, sign vectors)
python -m pytest test/srt/test_turboquant.py -v -k "not GPU"

# GPU tests (roundtrip quality, WHT self-inverse, attention score preservation)
python -m pytest test/srt/test_turboquant.py -v -k "GPU"
```

**Expected:**
- `test_wht_self_inverse`: max error < 1e-4
- `test_3bit_roundtrip`: cosine > 0.9
- `test_4bit_roundtrip`: cosine > 0.95
- `test_attention_score_preservation`: cosine > 0.9

## 2. Server Smoke Test

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --port 30001 \
    --kv-cache-dtype turboquant_3bit \
    --disable-cuda-graph \
    --mem-fraction-static 0.85
```

Wait for `"The server is fired up"`, then:

```bash
# Simple math
curl -s http://localhost:30001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"default","messages":[{"role":"user","content":"What is 2+3? Reply with just the number."}],"max_tokens":32,"temperature":0}' \
    | python3 -m json.tool

# Coherence check
curl -s http://localhost:30001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"default","messages":[{"role":"user","content":"Explain what a binary tree is in 2 sentences."}],"max_tokens":128,"temperature":0}' \
    | python3 -m json.tool
```

**Expected:**
- Math: should output "5"
- Coherence: meaningful English text, NOT garbage like "are are are..."
- Response time: < 5 seconds

## 3. NIAH (Needle-in-a-Haystack) — Target: >99%

### Quick test (5 needle types, multiple depths)

```bash
python benchmark/turboquant/bench_niah_turboquant.py --port 30001 \
    --context-lengths 512 1024 2048 \
    --depth-percents 10 25 50 75 90 \
    --verbose
```

### Full test with repeats

```bash
python benchmark/turboquant/bench_niah_turboquant.py --port 30001 \
    --context-lengths 512 1024 2048 4096 \
    --depth-percents 10 25 50 75 90 \
    --repeat 3 --verbose
```

### Long context (if model supports)

```bash
python benchmark/turboquant/bench_niah_turboquant.py --port 30001 \
    --context-lengths 8192 16384 \
    --depth-percents 10 30 50 70 90 \
    --repeat 2 --verbose
```

**Expected output format:**
```
NIAH Results Matrix:

 depth%     512    1024   2048   4096
    10%   3/3    3/3    3/3    3/3
    25%   3/3    3/3    3/3    3/3
    50%   3/3    3/3    3/3    3/3
    75%   3/3    3/3    3/3    3/3
    90%   3/3    3/3    3/3    3/3

Overall: 60/60 (100.0%)
```

### Multi-bit comparison

```bash
for bits in 2 3 4; do
    echo "=== turboquant_${bits}bit ==="
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --port 3000${bits} \
        --kv-cache-dtype turboquant_${bits}bit \
        --disable-cuda-graph --mem-fraction-static 0.85 &
    sleep 60
    python benchmark/turboquant/bench_niah_turboquant.py --port 3000${bits} \
        --context-lengths 1024 2048 4096 --depth-percents 10 25 50 75 90
    kill %1 2>/dev/null; sleep 5
done
```

## 4. Performance — Target: >90% of BF16

### TurboQuant throughput

```bash
python -m sglang.bench_serving --backend sglang --port 30001 \
    --num-prompts 10 --dataset-name random \
    --random-input-len 512 --random-output-len 128
```

### BF16 baseline

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct --port 30002 --mem-fraction-static 0.85

python -m sglang.bench_serving --backend sglang --port 30002 \
    --num-prompts 10 --dataset-name random \
    --random-input-len 512 --random-output-len 128
```

**Expected:** TurboQuant decode tok/s should be > 90% of BF16 (thanks to incremental dequant).

## 5. Memory Savings

```bash
# Compare GPU memory with turboquant_3bit vs bf16
# turboquant_3bit: ~5.3x KV cache compression
# At 8192 context: KV memory should be ~19% of bf16
nvidia-smi  # compare VRAM usage between the two servers
```

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Server crashes on startup | CUDA graph | Add `--disable-cuda-graph` |
| Garbage output | Wrong rotation type | Ensure code uses WHT (hadamard_transform), not random rotation |
| NIAH < 80% | WHT not applied or codebook wrong | Run unit tests, check `test_attention_score_preservation` |
| Slow decode (< 50% BF16) | Full buffer re-dequant | Check dirty tracking: `_k_dequant_valid` should be mostly True during decode |
| OOM | Dequant buffers | Reduce `--mem-fraction-static` |

## Files

| File | What |
|------|------|
| `python/sglang/srt/layers/quantization/kv_turboquant.py` | Core: WHT + Lloyd-Max + pack/unpack |
| `python/sglang/srt/mem_cache/memory_pool.py` | `MHATokenToKVPoolTurboQuant` with dirty tracking |
| `python/sglang/srt/server_args.py` | `turboquant_*bit` choices |
| `python/sglang/srt/model_executor/model_runner.py` | dtype mapping |
| `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` | pool selection |
| `test/srt/test_turboquant.py` | Unit tests |
| `benchmark/turboquant/bench_niah_turboquant.py` | NIAH benchmark (5 needles, matrix output) |
