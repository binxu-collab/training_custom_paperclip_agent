#!/bin/bash
# Run biomedrxiv SFT data collection
# Sessions history is cleared and data is saved after every BATCH_SIZE questions.
# Usage:
#   ./qwen_rl/run_collect_sft.sh           # full run (main text questions)
#   ./qwen_rl/run_collect_sft.sh --test    # test run (first 10 questions)
#   ./qwen_rl/run_collect_sft.sh --supp    # supplementary questions (biomedrxiv_supp.csv)
#   ./qwen_rl/run_collect_sft.sh --supp --test  # supp test (first 10)

set -e
cd /workspaces/gxl

INPUT="apps/evals/inputs_final/xulong_biomedrxiv_evals.json"
OUTPUT="qwen_rl/data/xulong_biomedrxiv_sft_raw.json"
AGENT="papers"
MAX_CONCURRENT=4
ENGINE_URL="http://localhost:8000"
BATCH_SIZE=10
SUPP_MODE=0

for arg in "$@"; do
    case "$arg" in
        --supp) SUPP_MODE=1 ;;
    esac
done

if [[ "$SUPP_MODE" == "1" ]]; then
    # Convert CSV → evals JSON, then collect
    echo "=== SUPP MODE: converting biomedrxiv_supp.csv ==="
    python3 qwen_rl/convert_supp_csv_to_evals.py \
        --input  biomedrxiv_supp.csv \
        --output qwen_rl/data/supp_evals.json
    INPUT="qwen_rl/data/supp_evals.json"
    OUTPUT="qwen_rl/data/supp_biomedrxiv_sft_raw.json"
fi

if [[ "$1" == "--test" ]] || [[ "$2" == "--test" ]]; then
    echo "=== TEST MODE: first 10 questions ==="
    INPUT_ACTUAL="${INPUT%.json}_test10.json"
    if [[ "$SUPP_MODE" == "1" ]]; then
        OUTPUT="qwen_rl/data/supp_biomedrxiv_sft_test.json"
    else
        OUTPUT="qwen_rl/data/xulong_biomedrxiv_sft_test.json"
    fi
    python3 -c "
import json
with open('$INPUT') as f:
    d = json.load(f)
d['evals'] = d['evals'][:10]
with open('$INPUT_ACTUAL', 'w') as f:
    json.dump(d, f, indent=2)
print(f'Wrote {len(d[\"evals\"])} questions to $INPUT_ACTUAL')
"
    INPUT_ACTUAL="$INPUT_ACTUAL"
else
    echo "=== FULL RUN: all questions ==="
    INPUT_ACTUAL="$INPUT"
fi

mkdir -p qwen_rl/data

python3 qwen_rl/collect_sft_data.py \
    --input "$INPUT_ACTUAL" \
    --agent "$AGENT" \
    --output "$OUTPUT" \
    --max-concurrent "$MAX_CONCURRENT" \
    --engine-url "$ENGINE_URL" \
    --batch-size "$BATCH_SIZE"

echo "Done. Output: $OUTPUT"
