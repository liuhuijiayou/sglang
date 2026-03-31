#!/bin/bash
# ==========================================================================
# TurboQuant Full Validation — One Script, All Phases
#
# Usage:
#   bash run_validation.sh                          # all phases (~60min)
#   bash run_validation.sh --skip-server            # phases 1+2+2.5 only (~5min)
#   bash run_validation.sh --quick                  # quick mode everywhere
#   bash run_validation.sh --model /models/XXX      # custom model
#
# Model config auto-detected from config.json.
# ==========================================================================
set -euo pipefail

# ---- Configuration ----
MODEL_PATH="${MODEL_PATH:-/models/Qwen2.5-7B-Instruct}"
PORT="${PORT:-30001}"
RESULTS_DIR="${RESULTS_DIR:-benchmark/turboquant/results}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_SERVER=false
QUICK=false
SERVING_RESULTS_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-server|--skip-niah) SKIP_SERVER=true; shift;;
        --quick)     QUICK=true; shift;;
        --model)     MODEL_PATH="$2"; shift 2;;
        --port)      PORT="$2"; shift 2;;
        *)           echo "Unknown arg: $1"; exit 1;;
    esac
done

# Auto-detect model config from config.json
CONFIG_FILE="$MODEL_PATH/config.json"
if [ -f "$CONFIG_FILE" ]; then
    NUM_KV_HEADS=$(python3 -c "
import json; c = json.load(open('$CONFIG_FILE'))
print(c.get('num_key_value_heads', c.get('num_attention_heads', 8)))
")
    NUM_LAYERS=$(python3 -c "
import json; c = json.load(open('$CONFIG_FILE'))
print(c.get('num_hidden_layers', 32))
")
    HEAD_DIM=$(python3 -c "
import json; c = json.load(open('$CONFIG_FILE'))
hs = c.get('hidden_size', 4096); nh = c.get('num_attention_heads', 32)
print(c.get('head_dim', hs // nh))
")
    echo "Auto-detected: num_kv_heads=$NUM_KV_HEADS, num_layers=$NUM_LAYERS, head_dim=$HEAD_DIM"
else
    echo "WARNING: $CONFIG_FILE not found, using defaults (Qwen2.5-7B)"
    NUM_KV_HEADS=4; NUM_LAYERS=28; HEAD_DIM=128
fi

SERVING_RESULTS_DIR="$RESULTS_DIR/serving"
mkdir -p "$RESULTS_DIR" "$SERVING_RESULTS_DIR"

NUM_VECTORS=10000
NIAH_CONTEXTS="1024 4096 8192"
SERVING_PROMPTS=200
if [ "$QUICK" = true ]; then
    NUM_VECTORS=2000
    NIAH_CONTEXTS="1024 4096"
    SERVING_PROMPTS=50
fi

echo "============================================================"
echo "TurboQuant Full Validation"
echo "============================================================"
echo "Model:   $MODEL_PATH"
echo "Config:  ${NUM_LAYERS}L × ${NUM_KV_HEADS}KVH × ${HEAD_DIM}d"
echo "Quick:   $QUICK"
echo "Results: $RESULTS_DIR"
echo "Date:    $(date -Iseconds)"
echo "============================================================"

PHASE_TIMES=()
phase_start() { PHASE_START_TS=$(date +%s); }
phase_end() {
    local elapsed=$(( $(date +%s) - PHASE_START_TS ))
    PHASE_TIMES+=("$1: ${elapsed}s")
    echo "  [${1} done in ${elapsed}s]"
}

# ==================================================================
# PHASE 1: MSE / Distortion (no server needed)
# ==================================================================
echo ""
echo "========================================"
echo "PHASE 1: MSE / Distortion"
echo "========================================"
phase_start

python "$SCRIPT_DIR/bench_mse.py" \
    --bits 2 3 4 \
    --mixed-bits 2.5 3.5 \
    --variants mse prod \
    --dim "$HEAD_DIM" \
    --num-vectors "$NUM_VECTORS" \
    --output "$RESULTS_DIR/mse_random.csv"

python "$SCRIPT_DIR/bench_mse.py" \
    --bits 2 3 4 \
    --mixed-bits 2.5 3.5 \
    --variants mse prod \
    --dim "$HEAD_DIM" \
    --num-vectors "$NUM_VECTORS" \
    --real-kv \
    --model-path "$MODEL_PATH" \
    --output "$RESULTS_DIR/mse_with_real_kv.csv"

phase_end "Phase1_MSE"

# ==================================================================
# PHASE 2: Compression Ratio (no server needed)
# ==================================================================
echo ""
echo "========================================"
echo "PHASE 2: Compression Ratio"
echo "========================================"
phase_start

python "$SCRIPT_DIR/bench_compression.py" \
    --bits 2 3 4 \
    --head-dim "$HEAD_DIM" \
    --num-kv-heads "$NUM_KV_HEADS" \
    --num-layers "$NUM_LAYERS" \
    --seq-len 4096 8192 16384 \
    --output "$RESULTS_DIR/compression.csv"

phase_end "Phase2_Compression"

# ==================================================================
# PHASE 2.5: Decompress Performance (no server, needs GPU)
# ==================================================================
echo ""
echo "========================================"
echo "PHASE 2.5: Decompress Performance"
echo "========================================"
phase_start

PERF_ARGS="--num-heads $NUM_KV_HEADS --head-dim $HEAD_DIM --breakdown"
if [ "$QUICK" = true ]; then
    PERF_ARGS="$PERF_ARGS --ci"
fi

python "$SCRIPT_DIR/bench_e2e_perf.py" \
    $PERF_ARGS \
    --output "$RESULTS_DIR/perf.json" \
    2>&1 | tee "$RESULTS_DIR/perf.txt"

phase_end "Phase2.5_Perf"

# ==================================================================
# Server-dependent phases (NIAH + E2E serving)
# ==================================================================
if [ "$SKIP_SERVER" = true ]; then
    echo ""
    echo "========================================"
    echo "Server phases SKIPPED (--skip-server)"
    echo "========================================"
    echo "Results: $RESULTS_DIR"
    printf "\nTiming: "; printf "%s  " "${PHASE_TIMES[@]}"; echo ""
    exit 0
fi

# ---- Server helper functions ----
start_server() {
    local label="$1"
    local extra_args="$2"

    # Kill any leftover server
    local pids=$(lsof -ti:$PORT 2>/dev/null || true)
    [ -n "$pids" ] && kill $pids 2>/dev/null || true && sleep 2

    echo "  Starting server ($label)..."
    python -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --port "$PORT" \
        --mem-fraction-static 0.80 \
        --disable-cuda-graph \
        $extra_args \
        > "$RESULTS_DIR/server_${label}.log" 2>&1 &
    SERVER_PID=$!

    local max_wait=300 waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "  Server ready (${waited}s)"
            return 0
        fi
        sleep 5; waited=$((waited + 5))
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "  ERROR: Server died. Log: $RESULTS_DIR/server_${label}.log"
            return 1
        fi
    done
    echo "  ERROR: Server timeout"; kill $SERVER_PID 2>/dev/null || true; return 1
}

stop_server() {
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 3
}

get_model_name() {
    curl -s "http://localhost:$PORT/v1/models" | \
        python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null \
        || echo "$MODEL_PATH"
}

# ==================================================================
# PHASE 3: NIAH
# ==================================================================
echo ""
echo "========================================"
echo "PHASE 3: Needle-in-a-Haystack"
echo "========================================"
phase_start

NIAH_CONFIGS=(
    "fp16_baseline:"
    "turboquant_2bit:--kv-cache-dtype turboquant_2bit"
    "turboquant_3bit:--kv-cache-dtype turboquant_3bit"
    "turboquant_4bit:--kv-cache-dtype turboquant_4bit"
    "turboquant_3bit_prod:--kv-cache-dtype turboquant_3bit --turboquant-variant prod"
    "turboquant_4bit_prod:--kv-cache-dtype turboquant_4bit --turboquant-variant prod"
    "turboquant_2.5bit:--kv-cache-dtype turboquant_2.5bit"
    "turboquant_3.5bit:--kv-cache-dtype turboquant_3.5bit"
)

for config in "${NIAH_CONFIGS[@]}"; do
    IFS=':' read -r label extra_args <<< "$config"
    echo ""
    echo "---- NIAH: $label ----"
    if start_server "$label" "$extra_args"; then
        python "$SCRIPT_DIR/bench_niah.py" \
            --port "$PORT" \
            --context-lengths $NIAH_CONTEXTS \
            --depth-percents 25 50 75 \
            --warmup \
            --label "$label" \
            --output "$RESULTS_DIR/niah_${label}.json" \
            2>&1 | tee "$RESULTS_DIR/niah_${label}.txt"
        stop_server
    else
        echo "  SKIPPING $label (server failed)"
        stop_server 2>/dev/null || true
    fi
done

phase_end "Phase3_NIAH"

# ==================================================================
# PHASE 4: E2E Serving Throughput
# ==================================================================
echo ""
echo "========================================"
echo "PHASE 4: E2E Serving Throughput"
echo "========================================"
phase_start

SERVING_CONFIGS=(
    "fp16:"
    "turboquant_2bit:--kv-cache-dtype turboquant_2bit"
    "turboquant_3bit:--kv-cache-dtype turboquant_3bit"
    "turboquant_4bit:--kv-cache-dtype turboquant_4bit"
    "turboquant_3bit_prod:--kv-cache-dtype turboquant_3bit --turboquant-variant prod"
    "turboquant_4bit_prod:--kv-cache-dtype turboquant_4bit --turboquant-variant prod"
    "turboquant_2.5bit:--kv-cache-dtype turboquant_2.5bit"
    "turboquant_3.5bit:--kv-cache-dtype turboquant_3.5bit"
)

for config in "${SERVING_CONFIGS[@]}"; do
    IFS=':' read -r label extra_args <<< "$config"
    echo ""
    echo "---- Serving: $label ----"
    if start_server "$label" "$extra_args"; then
        local_model=$(get_model_name)
        python3 -m sglang.bench_serving \
            --backend sglang \
            --host localhost --port $PORT \
            --model "$local_model" \
            --dataset-name random \
            --random-input-len 512 \
            --random-output-len 128 \
            --num-prompts $SERVING_PROMPTS \
            --request-rate inf \
            --warmup-requests 5 \
            --disable-stream \
            --output-file "$SERVING_RESULTS_DIR/bench_${label}.jsonl" \
            2>&1 | tee "$SERVING_RESULTS_DIR/bench_${label}.txt"
        stop_server
    else
        echo "  SKIPPING $label (server failed)"
        stop_server 2>/dev/null || true
    fi
done

# Comparison table
python "$SCRIPT_DIR/compare_serving_results.py" \
    --results-dir "$SERVING_RESULTS_DIR" \
    --output "$SERVING_RESULTS_DIR/comparison.json" \
    2>&1 | tee "$SERVING_RESULTS_DIR/comparison.txt"

phase_end "Phase4_Serving"

# ==================================================================
# SUMMARY
# ==================================================================
echo ""
echo "============================================================"
echo "ALL PHASES COMPLETE"
echo "============================================================"
echo ""
printf "Timing:\n"
for t in "${PHASE_TIMES[@]}"; do echo "  $t"; done
echo ""
echo "Output files:"
find "$RESULTS_DIR" -type f | sort | while read f; do
    echo "  $f  ($(du -h "$f" | cut -f1))"
done
echo ""
echo "Copy these to your local machine for analysis:"
echo "  scp -r server:$(pwd)/$RESULTS_DIR ."
