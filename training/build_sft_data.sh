#!/bin/bash
# Build final SFT parquet:
#   1. Filter supp traces that passed LLM judge
#   2. Merge with xulong_biomedrxiv_sft_filtered.json
#   3. Convert to parquet (no val split)
#
# Usage:
#   bash qwen_rl/build_sft_data.sh

set -euo pipefail
cd /workspaces/gxl

ACCURACY="qwen_rl/data/supp_accuracy.json"
SUPP_RAW="qwen_rl/data/supp_biomedrxiv_sft_raw.json"
XULONG="qwen_rl/data/xulong_biomedrxiv_sft_filtered.json"
MERGED="qwen_rl/data/merged_biomedrxiv_sft.json"
OUTPUT="qwen_rl/data/merged_biomedrxiv_sft"
SYSTEM_PROMPT_YAML="agents/papers/papers_reader.yaml"

echo "=== Step 1: Filter passed supp traces + merge with xulong ==="
python3 - <<'PYEOF'
import json, sys

ACCURACY   = "qwen_rl/data/supp_accuracy.json"
SUPP_RAW   = "qwen_rl/data/supp_biomedrxiv_sft_raw.json"
XULONG     = "qwen_rl/data/xulong_biomedrxiv_sft_filtered.json"
MERGED     = "qwen_rl/data/merged_biomedrxiv_sft.json"

with open(ACCURACY) as f:
    acc = json.load(f)
with open(SUPP_RAW) as f:
    raw = json.load(f)
with open(XULONG) as f:
    xulong = json.load(f)

supp_passed = [r for r in acc["results"] if r["pass"]]
print(f"Supp passed:   {len(supp_passed)}/{acc['total']}  ({acc['accuracy']:.1%})")
print(f"Xulong traces: {len(xulong['traces'])}")

merged_traces = xulong["traces"] + supp_passed
print(f"Merged total:  {len(merged_traces)}")

out = {
    "agent":         raw["agent"],
    "system_prompt": raw["system_prompt"],
    "tools":         raw["tools"],
    "traces":        merged_traces,
}

with open(MERGED, "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"Saved → {MERGED}")
PYEOF

echo ""
echo "=== Step 2: Convert to parquet (no val split) ==="
python3 qwen_rl/convert_to_sft_parquet.py \
    --input         "$MERGED" \
    --output        "$OUTPUT" \
    --system-prompt "$SYSTEM_PROMPT_YAML" \
    --val-split     0

echo ""
echo "Done. Output: ${OUTPUT}_train.parquet"
