#!/bin/bash
# Qwen3-14B + LoRA SFT, 32k context, 1 epoch — smoke test on RunPod H100 80GB.
#
# Assumed pod layout (Network Volume mounted at /workspace):
#   /workspace/gxl/                    rsynced repo (this script runs from gxl/qwen_rl/)
#   /workspace/gxl/qwen_rl/data/       merged_biomedrxiv_sft_train.parquet
#   /workspace/hf-cache/               HF model cache (persists across pod restarts)
#   /workspace/checkpoints_32k_1ep/    LoRA output
#
# Run after runpod_bootstrap.sh has finished.

set -e

REPO_ROOT=/workspace/gxl
VERLTOOL=$REPO_ROOT/qwen_rl/verl-tool
PYTHON=$VERLTOOL/.venv/bin/python
TORCHRUN=$VERLTOOL/.venv/bin/torchrun
PYTHONPATH=$VERLTOOL/verl

export HF_HOME=/workspace/hf-cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=1

DATA=$REPO_ROOT/qwen_rl/data/merged_biomedrxiv_sft_train.parquet

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=$PYTHONPATH \
$TORCHRUN --standalone --nnodes=1 --nproc_per_node=1 \
    -m verl.trainer.sft_trainer \
    data.train_files=$DATA \
    data.val_files=$DATA \
    data.messages_key=messages \
    data.tools_key=tools \
    data.max_length=32768 \
    data.truncation=right \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=8 \
    data.pad_mode=no_padding \
    data.max_token_len_per_gpu=32768 \
    model.path=Qwen/Qwen3-14B \
    model.lora_rank=16 \
    model.lora_alpha=32 \
    model.enable_gradient_checkpointing=True \
    model.enable_activation_offload=False \
    engine.model_dtype=bfloat16 \
    engine.optimizer_offload=False \
    engine.param_offload=False \
    trainer.project_name=biomedrxiv-sft \
    trainer.experiment_name=qwen3-14b-lora-merged-32k-1ep \
    trainer.total_epochs=1 \
    trainer.default_local_dir=/workspace/checkpoints_32k_1ep \
    trainer.logger=[console,wandb] \
    trainer.save_freq=50 \
    trainer.test_freq=99999 \
    trainer.resume_mode=disable
