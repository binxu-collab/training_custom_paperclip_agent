# Training a Custom Paperclip Agent

Fine-tuning Qwen3-14B to use the `paperclip` tool for biomedical paper QA — achieving **93% accuracy** on a 100-question benchmark, matching or exceeding frontier models (Claude Opus 4.7: 92%, Claude Sonnet 4.6: 91%) while running locally at a fraction of the cost.

## What is this?

[Paperclip](https://github.com/GXL-ai/gxl) is an MCP tool that provides access to 8M+ full-text biomedical papers from bioRxiv, medRxiv, and PubMed Central. Instead of stuffing entire papers into a prompt (slow, expensive), a paperclip agent retrieves only what it needs — looking up papers by DOI, grepping for specific values, reading targeted sections.

This repo contains the full pipeline to:
1. **Collect** high-quality SFT traces by running a teacher model (Claude) on biomedical QA questions
2. **Filter** traces for efficiency (remove redundant tool calls, reward concise retrieval)
3. **Train** Qwen3-14B with LoRA SFT, then optionally fine-tune further with RL (GRPO)

## Results

Evaluated on 100 biomedical QA questions covering main text, supplementary PDFs, and supplementary DOCX files:

| Model | Method | Accuracy | Time (100Q) |
|---|---|---|---|
| **Q3-14B-SFT** (this repo) | MCP agent | **93%** | 5.6 min |
| Claude Opus 4.7 | Full context | 92% | 7.5 min |
| Claude Sonnet 4.6 | Full context | 91% | 7.9 min |
| Gemini 3.1 Flash Lite | Full context | 91% | 6.7 min |
| Gemini 2.5 Flash | Full context | 89% | 6.6 min |
| Q3-14B (base) | Full context (80K limit) | 53% | 18 min |

The SFT model learns to use the paperclip tool efficiently: average **4 tool calls per question** vs naive approaches that burn the full context window.

## Repository Structure

```
├── runpod_bootstrap.sh        # One-shot setup for RunPod H100 80GB
├── train_sft_32k_1ep.sh       # Main SFT training script (32K context, 1 epoch)
├── train_sft.sh               # SFT, 8K context
├── train_sft_32768.sh         # SFT, 32K context, multi-epoch
├── train_rl.sh                # RL/GRPO training with live tool use
│
├── collect_sft_data.py        # Run teacher agent → raw traces
├── filter_sft_trajectories.py # Rule + LLM filtering for trace quality
├── convert_to_sft_parquet.py  # Convert JSON traces → verl-tool parquet
├── build_rl_data.py           # Build RL training dataset
├── merge_sft_checkpoint.py    # Merge LoRA weights into base model
│
├── eval_context_baseline.py   # Evaluate full-context baseline models
├── run_eval.sh                # Run agent evals
├── run_collect_sft.sh         # Shell wrapper for data collection
├── run_filter_sft.sh          # Shell wrapper for filtering
├── build_sft_data.sh          # End-to-end data build pipeline
│
├── papers_reader.yaml         # Agent config (MCP URL, tools, system prompt)
└── data_samples/              # Sample eval questions for testing
    ├── _test_10.json
    └── _test_100.json
```

## Quick Start

### Prerequisites

- RunPod H100 80GB pod with a **150GB Network Volume** mounted at `/workspace`
- `WANDB_API_KEY` for training logs (optional but recommended)
- Training data: `merged_biomedrxiv_sft_train.parquet` (contact us for access)

### 1. Clone and sync to RunPod

```bash
git clone https://github.com/binxu-collab/training_custom_paperclip_agent.git
cd training_custom_paperclip_agent

# Sync to pod (replace with your pod's IP and port)
rsync -avz --exclude='.venv' --exclude='checkpoints*' \
    ./ root@<pod-ip>:/workspace/gxl/ -e "ssh -p <port>"
```

### 2. Bootstrap the pod

```bash
ssh root@<pod-ip> -p <port>
cd /workspace/gxl
WANDB_API_KEY=<your-key> bash runpod_bootstrap.sh
```

This will:
- Install `uv` and create a Python 3.11 venv
- Install [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool), PyTorch, transformers, peft, flash-attn
- Download `Qwen/Qwen3-14B` (~28GB) to `/workspace/hf-cache`

### 3. Run SFT training

```bash
WANDB_API_KEY=<your-key> bash train_sft_32k_1ep.sh
```

Training config: LoRA rank=16, 32K context, bfloat16, 1 epoch.
Expected runtime: **~2 hours** on H100 80GB.
Output: `/workspace/checkpoints_32k_1ep/`

### 4. (Optional) RL fine-tuning

After merging the LoRA checkpoint:
```bash
python merge_sft_checkpoint.py --checkpoint /workspace/checkpoints_32k_1ep/global_step_xxx
WANDB_API_KEY=<your-key> bash train_rl.sh
```

RL uses GRPO with live paperclip tool calls as the reward signal.

## Data Pipeline

To collect your own SFT data from a set of QA questions:

```bash
# 1. Collect traces (runs Claude as teacher agent)
bash run_collect_sft.sh

# 2. Filter for quality
bash run_filter_sft.sh

# 3. Build final parquet
bash build_sft_data.sh
```

The teacher agent runs ~4 tool calls per question on average. Filtering removes traces with redundant calls, empty results, or low LLM efficiency scores. The final dataset has ~1,300 traces.

## Model Config

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen3-14B` |
| Method | LoRA |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Max context | 32,768 tokens |
| Precision | bfloat16 |
| Optimizer | AdamW via FSDP |
| Epochs | 1 (SFT), 5 (RL) |

## Dependencies

Training is built on [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool), which extends the [verl](https://github.com/volcengine/verl) RL framework with multi-turn tool-use support.

Key packages: `torch>=2.4`, `transformers>=4.46`, `peft>=0.13`, `flash-attn==2.7.4.post1`, `vllm<=0.11.0`
