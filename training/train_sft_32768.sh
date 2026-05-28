#!/bin/bash
set -e

# Context length: 32768 (within Qwen3-14B native 40960 — no YaRN needed)

PYTHON=/workspaces/gxl/qwen_rl/verl-tool/.venv/bin/python
TORCHRUN=/workspaces/gxl/qwen_rl/verl-tool/.venv/bin/torchrun
PYTHONPATH=/workspaces/gxl/qwen_rl/verl-tool/verl

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=$PYTHONPATH \
$TORCHRUN --standalone --nnodes=1 --nproc_per_node=1 \
    -m verl.trainer.sft_trainer \
    data.train_files=/workspaces/gxl/qwen_rl/data/maintext_sft_train.parquet \
    data.val_files=/workspaces/gxl/qwen_rl/data/maintext_sft_val.parquet \
    data.messages_key=messages \
    data.tools_key=tools \
    data.max_length=16382 \
    data.truncation=right \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=4 \
    data.pad_mode=no_padding \
    model.path=Qwen/Qwen3-14B \
    model.lora_rank=16 \
    model.lora_alpha=32 \
    model.enable_gradient_checkpointing=True \
    engine.model_dtype=bfloat16 \
    engine.optimizer_offload=False \
    engine.param_offload=False \
    model.enable_activation_offload=False \
    data.max_token_len_per_gpu=16382 \
    trainer.project_name=biomedrxiv-sft \
    trainer.experiment_name=qwen3-14b-lora-maintext-32k \
    trainer.total_epochs=3 \
    trainer.default_local_dir=/workspaces/gxl/qwen_rl/checkpoints_maintext_16k \
    trainer.logger=[console,wandb] \
    trainer.save_freq=200 \
    trainer.test_freq=1000 \
    trainer.resume_mode=disable
