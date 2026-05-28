#!/bin/bash
# Run eval_with_tools.py via vLLM against the live paperclip MCP server.
#
# Usage:
#   bash qwen_rl/run_eval.sh                                               # global_step_100, bs=1
#   bash qwen_rl/run_eval.sh --checkpoint global_step_100
#   bash qwen_rl/run_eval.sh --checkpoint global_step_100 --limit 10
#   bash qwen_rl/run_eval.sh --checkpoint global_step_100 --batch-size 4
#   bash qwen_rl/run_eval.sh --checkpoint global_step_100 --batch-size 4 --limit 10

set -euo pipefail
cd /workspaces/gxl

PYTHON=/workspaces/gxl/qwen_rl/verl-tool/.venv/bin/python
VLLM=$(/workspaces/gxl/qwen_rl/verl-tool/.venv/bin/python -c "import vllm, os; print(os.path.join(os.path.dirname(vllm.__file__), '../../../bin/vllm'))" 2>/dev/null || echo "vllm")
EVAL_FILE=qwen_rl/data/_test_50.json
CHECKPOINT="global_step_100"
WORKDIR="$(pwd)"
LIMIT=0
BATCH_SIZE=20
VLLM_PORT=8001

for arg in "$@"; do
    case "$arg" in
        --checkpoint) NEXT_IS_CKPT=1 ;;
        --limit)      NEXT_IS_LIMIT=1 ;;
        --batch-size) NEXT_IS_BS=1 ;;
        --full)       LIMIT=0 ;;
        *)
            if [[ "${NEXT_IS_CKPT:-0}" == "1" ]]; then
                CHECKPOINT="$arg"; NEXT_IS_CKPT=0
            elif [[ "${NEXT_IS_LIMIT:-0}" == "1" ]]; then
                LIMIT="$arg"; NEXT_IS_LIMIT=0
            elif [[ "${NEXT_IS_BS:-0}" == "1" ]]; then
                BATCH_SIZE="$arg"; NEXT_IS_BS=0
            fi
            ;;
    esac
done

# Resolve checkpoint to absolute path — vLLM requires absolute paths
if [[ "$CHECKPOINT" == /* ]]; then
    CKPT_PATH="$CHECKPOINT"
else
    CKPT_PATH="${WORKDIR}/qwen_rl/checkpoints_maintext/$CHECKPOINT"
fi

# Auto-append _merged if directory has no config.json
if [[ ! -f "${CKPT_PATH}/config.json" && -f "${CKPT_PATH}_merged/config.json" ]]; then
    echo "    (using ${CHECKPOINT}_merged — original has no config.json)"
    CKPT_PATH="${CKPT_PATH}_merged"
    CHECKPOINT="${CHECKPOINT}_merged"
fi

OUT="qwen_rl/eval_results/${CHECKPOINT//\//_}_bs${BATCH_SIZE}.json"
echo "=== Eval: $CHECKPOINT (bs=$BATCH_SIZE) ==="

LIMIT_FLAG=""
if [[ "$LIMIT" -gt 0 ]]; then
    LIMIT_FLAG="--limit $LIMIT"
    echo "    Questions: $LIMIT"
else
    echo "    Questions: all"
fi

# ── Kill any leftover vLLM on this port ───────────────────────────────────────
OLD_PID=$(lsof -ti tcp:${VLLM_PORT} 2>/dev/null || true)
if [[ -n "$OLD_PID" ]]; then
    echo "Killing existing process on port $VLLM_PORT (pid $OLD_PID)..."
    kill "$OLD_PID" 2>/dev/null || true
    sleep 3
fi

# ── Start vLLM ─────────────────────────────────────────────────────────────────
echo "Starting vLLM on port $VLLM_PORT ..."
/workspaces/gxl/qwen_rl/verl-tool/.venv/bin/vllm serve "$CKPT_PATH" \
    --port "$VLLM_PORT" \
    --dtype bfloat16 \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    &
VLLM_PID=$!

# Wait until vLLM is ready
echo -n "Waiting for vLLM..."
for i in $(seq 1 60); do
    if curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo " ready"
        break
    fi
    sleep 5
    echo -n "."
done

# ── Run eval ───────────────────────────────────────────────────────────────────
mkdir -p qwen_rl/eval_results

ANTHROPIC_API_KEY=$(grep ^ANTHROPIC_API_KEY .env | cut -d= -f2 | tr -d ' ') \
$PYTHON qwen_rl/eval_with_tools.py \
    --vllm-url    "http://localhost:${VLLM_PORT}" \
    --eval-file   "$EVAL_FILE" \
    --output      "$OUT" \
    --batch-size  "$BATCH_SIZE" \
    $LIMIT_FLAG

# ── Cleanup ────────────────────────────────────────────────────────────────────
kill "$VLLM_PID" 2>/dev/null || true

echo ""
echo "Results: $OUT"
