#!/bin/bash
# ===========================================================================
# TurboQuant End-to-End Inference Throughput & Latency Benchmark
#
# Compares FP16 baseline vs TurboQuant KV cache compression using SGLang's
# canonical bench_serving benchmark. Measures:
#   - Output token throughput (tok/s)
#   - TTFT (time to first token)
#   - TPOT (time per output token)
#   - ITL (inter-token latency)
#   - End-to-end request latency
#
# Usage:
#   bash bench_e2e_serving.sh                      # defaults: Llama-3.1-8B, 3+4 bit
#   bash bench_e2e_serving.sh --model /models/XXX   # custom model
#   bash bench_e2e_serving.sh --quick               # fewer prompts, faster
#   bash bench_e2e_serving.sh --decode-heavy         # long output, stress decode
#   bash bench_e2e_serving.sh --prefill-heavy        # long input, stress prefill
# ===========================================================================
set -euo pipefail

# ---- Defaults ----
MODEL_PATH="${MODEL_PATH:-/models/Qwen2.5-7B-Instruct}"
PORT="${PORT:-30001}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/sglang/benchmark/turboquant/results/serving}"
NUM_PROMPTS=200
REQUEST_RATE="inf"     # burst mode for max throughput
WARMUP=10
INPUT_LEN=512
OUTPUT_LEN=128
EXTRA_SERVER_ARGS=""
QUICK=false
PROFILE_NAME="default"

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)       MODEL_PATH="$2"; shift 2;;
        --port)        PORT="$2"; shift 2;;
        --results-dir) RESULTS_DIR="$2"; shift 2;;
        --num-prompts) NUM_PROMPTS="$2"; shift 2;;
        --input-len)   INPUT_LEN="$2"; shift 2;;
        --output-len)  OUTPUT_LEN="$2"; shift 2;;
        --quick)       NUM_PROMPTS=50; WARMUP=3; QUICK=true; shift;;
        --decode-heavy)
            # Stress decode: short input, long output
            INPUT_LEN=128; OUTPUT_LEN=512; PROFILE_NAME="decode_heavy"; shift;;
        --prefill-heavy)
            # Stress prefill: long input, short output
            INPUT_LEN=2048; OUTPUT_LEN=32; PROFILE_NAME="prefill_heavy"; shift;;
        --extra-server-args) EXTRA_SERVER_ARGS="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

mkdir -p "$RESULTS_DIR"

echo "=================================================================="
echo "TurboQuant E2E Serving Benchmark"
echo "=================================================================="
echo "Model:       $MODEL_PATH"
echo "Input len:   $INPUT_LEN"
echo "Output len:  $OUTPUT_LEN"
echo "Num prompts: $NUM_PROMPTS"
echo "Profile:     $PROFILE_NAME"
echo "Results:     $RESULTS_DIR"
echo "Date:        $(date -Iseconds)"
echo "=================================================================="

# ---- Configurations to benchmark ----
# Format: "label:kv_cache_dtype_arg"
CONFIGS=(
    "fp16:"
    "turboquant_4bit:--kv-cache-dtype turboquant_4bit"
    "turboquant_3bit:--kv-cache-dtype turboquant_3bit"
)

if [ "$QUICK" = false ]; then
    CONFIGS+=(
        "turboquant_2bit:--kv-cache-dtype turboquant_2bit"
        "turboquant_3bit_prod:--kv-cache-dtype turboquant_3bit --turboquant-variant prod"
        "turboquant_4bit_prod:--kv-cache-dtype turboquant_4bit --turboquant-variant prod"
        "turboquant_2.5bit:--kv-cache-dtype turboquant_2.5bit"
        "turboquant_3.5bit:--kv-cache-dtype turboquant_3.5bit"
    )
fi

# ---- Helper functions ----
wait_for_server() {
    local max_wait=300
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "  Server ready (${waited}s)"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "  ERROR: Server did not start within ${max_wait}s"
    return 1
}

kill_server() {
    # Kill any sglang server on the port
    local pids=$(lsof -ti:$PORT 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "  Killing server (pids: $pids)"
        kill $pids 2>/dev/null || true
        sleep 3
    fi
}

run_one_config() {
    local label="$1"
    local extra_args="$2"

    echo ""
    echo "============================================================"
    echo "  Config: $label"
    echo "============================================================"

    # Kill any existing server
    kill_server

    # Start server
    echo "  Starting server..."
    local server_cmd="python -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --port $PORT \
        --mem-fraction-static 0.85 \
        --disable-cuda-graph \
        $extra_args \
        $EXTRA_SERVER_ARGS"

    echo "  CMD: $server_cmd"
    eval $server_cmd > "$RESULTS_DIR/server_${label}.log" 2>&1 &
    local server_pid=$!

    if ! wait_for_server; then
        echo "  Server failed. Log: $RESULTS_DIR/server_${label}.log"
        kill $server_pid 2>/dev/null || true
        return 1
    fi

    # Get model info from server
    local model_name
    model_name=$(curl -s "http://localhost:$PORT/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "$MODEL_PATH")

    # Run bench_serving
    local output_file="$RESULTS_DIR/bench_${PROFILE_NAME}_${label}.jsonl"

    echo "  Running bench_serving (${NUM_PROMPTS} prompts, in=${INPUT_LEN}, out=${OUTPUT_LEN})..."
    python3 -m sglang.bench_serving \
        --backend sglang \
        --host localhost \
        --port $PORT \
        --model "$model_name" \
        --dataset-name random \
        --random-input-len $INPUT_LEN \
        --random-output-len $OUTPUT_LEN \
        --num-prompts $NUM_PROMPTS \
        --request-rate $REQUEST_RATE \
        --warmup-requests $WARMUP \
        --disable-stream \
        --output-file "$output_file" \
        2>&1 | tee "$RESULTS_DIR/bench_${PROFILE_NAME}_${label}.txt"

    # Stop server
    echo "  Stopping server..."
    kill $server_pid 2>/dev/null || true
    wait $server_pid 2>/dev/null || true
    sleep 3
    echo "  Done with $label"
}

# ---- Run all configs ----
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r label extra_args <<< "$config"
    run_one_config "$label" "$extra_args" || echo "  WARNING: $label failed, continuing..."
done

# ---- Summary ----
echo ""
echo "=================================================================="
echo "SUMMARY"
echo "=================================================================="
echo ""

# Parse results from each output file
printf "%-20s %10s %10s %10s %10s %10s\n" \
    "Config" "Out tok/s" "TTFT(ms)" "TPOT(ms)" "ITL_p99" "E2E(ms)"
printf "%-20s %10s %10s %10s %10s %10s\n" \
    "--------------------" "----------" "----------" "----------" "----------" "----------"

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r label extra_args <<< "$config"
    result_file="$RESULTS_DIR/bench_${PROFILE_NAME}_${label}.jsonl"
    if [ -f "$result_file" ]; then
        # Extract metrics from the JSONL (last line is the summary)
        python3 -c "
import json, sys
try:
    with open('$result_file') as f:
        lines = [l.strip() for l in f if l.strip()]
    data = json.loads(lines[-1])
    out_tps = data.get('output_throughput', 0)
    ttft = data.get('mean_ttft_ms', 0)
    tpot = data.get('mean_tpot_ms', 0)
    itl_p99 = data.get('p99_itl_ms', 0)
    e2e = data.get('mean_e2e_latency_ms', 0)
    print(f'$label'.ljust(20) + f'{out_tps:>10.1f} {ttft:>10.2f} {tpot:>10.2f} {itl_p99:>10.2f} {e2e:>10.1f}')
except Exception as e:
    print(f'$label'.ljust(20) + f'  PARSE ERROR: {e}')
" 2>/dev/null || echo "$(printf '%-20s %s' "$label" "  FILE NOT FOUND")"
    else
        echo "$(printf '%-20s %s' "$label" "  NO RESULTS")"
    fi
done

echo ""
echo "Detailed results in: $RESULTS_DIR/"
echo "Server logs:         $RESULTS_DIR/server_*.log"
echo ""
echo "To compare manually:"
echo "  cat $RESULTS_DIR/bench_${PROFILE_NAME}_fp16.txt"
echo "  cat $RESULTS_DIR/bench_${PROFILE_NAME}_turboquant_4bit.txt"
