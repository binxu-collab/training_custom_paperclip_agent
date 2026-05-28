#!/bin/bash
# Filter SFT trajectories for efficiency, then convert to parquet.
#
# Pipeline:
#   filter_sft_trajectories.py  →  convert_to_sft_parquet.py
#
# Usage:
#   bash qwen_rl/run_filter_sft.sh              # full run
#   bash qwen_rl/run_filter_sft.sh --test       # 10-trace smoke test
#   bash qwen_rl/run_filter_sft.sh --no-llm     # rule-only filter (no API calls)
#   bash qwen_rl/run_filter_sft.sh --dry-run    # stats only, no files written
#
# Key knobs (edit below):
#   MIN_SCORE     LLM efficiency score threshold to keep a trace  (1–5, default 3)
#   MAX_CALLS     hard-reject traces longer than this              (default 12)
#   LLM_THRESHOLD only LLM-judge traces with >= this many calls   (default 5)
#   JUDGE_MODEL   cheap model for judging                         (haiku by default)

set -euo pipefail
cd /workspaces/gxl

# ── Tuneable parameters ────────────────────────────────────────────────────────
RAW_INPUT="qwen_rl/data/xulong_biomedrxiv_sft_raw.json"
FILTERED="qwen_rl/data/xulong_biomedrxiv_sft_filtered.json"
PARQUET_OUTPUT="qwen_rl/data/biomedrxiv_sft"
SYSTEM_PROMPT_YAML="agents/papers/papers_reader.yaml"
HOLDOUT="qwen_rl/data/_test_50.json"

ENGINE_URL="http://localhost:8000"
JUDGE_MODEL="claude-opus-4-5"
MIN_SCORE=3
MAX_CALLS=12
LLM_THRESHOLD=5
MAX_CONCURRENT=20
VAL_SPLIT=0.1

# ── Arg parsing ────────────────────────────────────────────────────────────────
EXTRA_FLAGS=""
for arg in "$@"; do
    case "$arg" in
        --test)
            echo "=== TEST MODE (10 traces) ==="
            RAW_INPUT="qwen_rl/data/xulong_biomedrxiv_sft_test.json"
            FILTERED="qwen_rl/data/_test_10_filtered.json"
            PARQUET_OUTPUT="qwen_rl/data/_test_10_sft"
            ;;
        --no-llm)
            echo "=== Rule-only mode (no LLM judge) ==="
            EXTRA_FLAGS="$EXTRA_FLAGS --no-llm-judge"
            ;;
        --dry-run)
            echo "=== Dry run (stats only) ==="
            EXTRA_FLAGS="$EXTRA_FLAGS --dry-run"
            ;;
    esac
done

mkdir -p qwen_rl/data

# ── Step 1: Filter trajectories ────────────────────────────────────────────────
echo ""
echo "=== Step 1: Filter trajectories ==="
echo "  Input:        $RAW_INPUT"
echo "  Output:       $FILTERED"
echo "  Judge model:  $JUDGE_MODEL  (min-score=$MIN_SCORE)"
echo "  Max calls:    $MAX_CALLS  |  LLM threshold: $LLM_THRESHOLD"
echo ""

python3 qwen_rl/filter_sft_trajectories.py \
    --input           "$RAW_INPUT" \
    --output          "$FILTERED" \
    --engine-url      "$ENGINE_URL" \
    --judge-model     "$JUDGE_MODEL" \
    --min-score       "$MIN_SCORE" \
    --max-calls       "$MAX_CALLS" \
    --llm-threshold   "$LLM_THRESHOLD" \
    --max-concurrent  "$MAX_CONCURRENT" \
    $EXTRA_FLAGS

# Skip parquet conversion on dry-run
if echo "$EXTRA_FLAGS" | grep -q "\-\-dry-run"; then
    echo "Dry run complete — skipping parquet conversion."
    exit 0
fi

# ── Step 2: Convert to parquet ─────────────────────────────────────────────────
echo ""
echo "=== Step 2: Convert to parquet ==="
echo "  Input:   $FILTERED"
echo "  Output:  ${PARQUET_OUTPUT}_train.parquet  /  ${PARQUET_OUTPUT}_val.parquet"
echo ""

python3 qwen_rl/convert_to_sft_parquet.py \
    --input         "$FILTERED" \
    --output        "$PARQUET_OUTPUT" \
    --system-prompt "$SYSTEM_PROMPT_YAML" \
    --holdout       "$HOLDOUT" \
    --val-split     "$VAL_SPLIT"

echo ""
echo "=== Done ==="
echo "  Filtered JSON: $FILTERED"
echo "  Train parquet: ${PARQUET_OUTPUT}_train.parquet"
echo "  Val   parquet: ${PARQUET_OUTPUT}_val.parquet"
