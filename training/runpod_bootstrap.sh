#!/bin/bash
# Bootstrap a fresh RunPod pod (H100 80GB, pytorch:2.4+ base image) for
# Qwen3-14B SFT. Run from /workspace/gxl after rsyncing the repo:
#
#   ssh root@<pod-ip> -p <ssh-port>
#   cd /workspace/gxl && bash qwen_rl/runpod_bootstrap.sh
#
# Idempotent — safe to re-run. Skips work that's already done.

set -e

REPO_ROOT=/workspace/gxl
VERLTOOL=$REPO_ROOT/qwen_rl/verl-tool
DATA_DIR=$REPO_ROOT/qwen_rl/data
HF_CACHE=/workspace/hf-cache

mkdir -p $HF_CACHE /workspace/checkpoints_32k_1ep
export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "=== 1. Check required files ==="
test -f $DATA_DIR/merged_biomedrxiv_sft_train.parquet \
    || { echo "MISSING: $DATA_DIR/merged_biomedrxiv_sft_train.parquet — rsync repo first"; exit 1; }
test -d $VERLTOOL || { echo "MISSING: $VERLTOOL — rsync repo first"; exit 1; }
echo "OK"

echo ""
echo "=== 2. Install uv ==="
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi
uv --version

echo ""
echo "=== 3. Create verl-tool venv + install deps ==="
cd $VERLTOOL
if [ ! -d .venv ]; then
    uv venv --python 3.11
fi
# Activate
. .venv/bin/activate
# Install verl-tool itself (uses pyproject.toml) — editable so the verl/ subdir is importable
uv pip install -e .
# Core training deps not pinned in verl-tool's pyproject
uv pip install \
    "torch>=2.4" \
    "transformers>=4.46" \
    "accelerate>=1.0" \
    "peft>=0.13" \
    "datasets>=3.0" \
    "pyarrow" \
    "pandas" \
    "wandb" \
    "hf-transfer" \
    "flash-attn==2.7.4.post1" --no-build-isolation || \
    uv pip install "torch>=2.4" "transformers>=4.46" "accelerate>=1.0" "peft>=0.13" \
                   "datasets>=3.0" "pyarrow" "pandas" "wandb" "hf-transfer"

echo ""
echo "=== 4. Pre-download Qwen3-14B to volume cache ==="
$VERLTOOL/.venv/bin/python -c "
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-14B', cache_dir='$HF_CACHE')
"

echo ""
echo "=== 5. Wandb login check ==="
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARN: \$WANDB_API_KEY not set — training will log to console only."
    echo "      Set it via RunPod pod env vars, or run: wandb login <key>"
fi

echo ""
echo "=== Bootstrap done. To start training: ==="
echo "  cd $REPO_ROOT/qwen_rl && bash train_sft_32k_1ep.sh"
