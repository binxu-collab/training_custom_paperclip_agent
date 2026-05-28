# Training Custom Paperclip Agent

SFT + RL training pipeline for Qwen3-14B on biomedical paper QA using the paperclip MCP tool.

## Overview

This repo contains the training code to fine-tune Qwen3-14B to use the `paperclip` tool
for answering questions about biomedical papers (bioRxiv, medRxiv, PMC).

## Quick Start (RunPod H100 80GB)

### 1. Sync repo + bootstrap
```bash
rsync -avz --exclude='.venv' --exclude='checkpoints*' --exclude='wandb' \
    ./ root@<pod-ip>:/workspace/gxl/ -e "ssh -p <port>"

ssh root@<pod-ip> -p <port>
cd /workspace/gxl
WANDB_API_KEY=<your-key> bash runpod_bootstrap.sh
```

### 2. Train (SFT, 32K context, 1 epoch)
```bash
WANDB_API_KEY=<your-key> bash train_sft_32k_1ep.sh
```

## Training Scripts

| Script | Description |
|--------|-------------|
| `runpod_bootstrap.sh` | Install deps + download Qwen3-14B on RunPod H100 |
| `train_sft_32k_1ep.sh` | SFT LoRA training, 32K context, 1 epoch |
| `train_sft.sh` | SFT, 8K context |
| `train_sft_32768.sh` | SFT, 32K context, multi-epoch |
| `train_rl.sh` | RL/GRPO training with tool use |

## Data Pipeline

```
collect_sft_data.py      # Run agent on eval questions → raw traces
filter_sft_trajectories.py  # Filter for high-quality traces
convert_to_sft_parquet.py   # Convert to verl-tool parquet format
```

## Model Config

- Base: `Qwen/Qwen3-14B`
- Method: LoRA (rank=16, alpha=32)
- Context: 32K tokens
- Precision: bfloat16

## Dependencies

Training uses [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool) framework.
Run `runpod_bootstrap.sh` to install everything.
