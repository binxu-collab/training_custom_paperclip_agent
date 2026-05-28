# Training a Custom Paperclip Agent

Fine-tuning Qwen3-14B to use the `paperclip` tool for biomedical paper QA — achieving **93% accuracy** on a 100-question benchmark, matching or exceeding frontier models (Claude Opus 4.7: 92%, Claude Sonnet 4.6: 91%) while running as a local agent at a fraction of the cost.

## What is this?

[Paperclip](https://github.com/GXL-ai/gxl) is an MCP tool that provides access to 8M+ full-text biomedical papers from bioRxiv, medRxiv, and PubMed Central. Instead of stuffing entire papers into a prompt (slow, expensive), a paperclip agent retrieves only what it needs — looking up papers by DOI, grepping for specific values, reading targeted sections.

This repo contains the full pipeline to:
1. **Collect** high-quality SFT traces by running a teacher model (Claude) on biomedical QA questions
2. **Filter** traces for efficiency (remove redundant tool calls, reward concise retrieval)
3. **Train** Qwen3-14B with LoRA SFT, then optionally continue with RL (GRPO)

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

The SFT model learns to use the paperclip tool efficiently: average **4 tool calls per question** vs naive full-context approaches.

---

## Environment Setup

### Requirements

- A **RunPod pod** with H100 80GB GPU (recommended: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`)
- A **RunPod Network Volume** (150GB) mounted at `/workspace` — this persists the model cache and checkpoints across pod restarts
- Python 3.11
- CUDA 12.4+

### Directory layout expected on the pod

```
/workspace/
├── gxl/                          ← this repo (cloned/rsynced here)
│   ├── runpod_bootstrap.sh
│   ├── train_sft_32k_1ep.sh
│   ├── verl-tool/                ← training framework (submodule)
│   └── data/
│       └── merged_biomedrxiv_sft_train.parquet
├── hf-cache/                     ← Hugging Face model cache (auto-created)
└── checkpoints_32k_1ep/          ← LoRA output (auto-created)
```

### Step 1 — Get the training framework (verl-tool)

The training scripts depend on [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool). Clone it into the repo:

```bash
git clone https://github.com/binxu-collab/training_custom_paperclip_agent.git
cd training_custom_paperclip_agent
git clone https://github.com/TIGER-AI-Lab/verl-tool.git
```

### Step 2 — Sync to RunPod pod

```bash
# From your local machine
rsync -avz \
    --exclude='.venv' \
    --exclude='checkpoints*' \
    --exclude='wandb' \
    --exclude='__pycache__' \
    ./ root@<pod-ip>:/workspace/gxl/ \
    -e "ssh -p <ssh-port>"
```

### Step 3 — Run the bootstrap script

SSH into the pod and run:

```bash
ssh root@<pod-ip> -p <ssh-port>
cd /workspace/gxl
WANDB_API_KEY=<your-wandb-key> bash runpod_bootstrap.sh
```

The bootstrap script is **idempotent** (safe to re-run). It will:

1. **Install `uv`** — fast Python package manager
2. **Create a Python 3.11 virtualenv** at `verl-tool/.venv`
3. **Install all dependencies** into the venv:
   - `verl-tool` (editable install from `verl-tool/pyproject.toml`)
   - `torch>=2.4`, `transformers>=4.46`, `accelerate>=1.0`
   - `peft>=0.13` (LoRA)
   - `datasets>=3.0`, `pyarrow`, `pandas`
   - `flash-attn==2.7.4.post1` (compiled with `--no-build-isolation`)
   - `wandb`, `hf-transfer`
4. **Download Qwen3-14B** (~28GB) to `/workspace/hf-cache` using `hf-transfer` for fast downloads

Expected bootstrap time: **~20–30 minutes** (mostly model download).

> **Note:** The venv and model are stored on the Network Volume, so they survive pod restarts. Re-running bootstrap after a restart completes in seconds.

### Step 4 — Verify the setup

```bash
# Check venv exists
ls /workspace/gxl/verl-tool/.venv/bin/torchrun

# Check model downloaded
ls /workspace/hf-cache/models--Qwen--Qwen3-14B/

# Check training data present
ls /workspace/gxl/data/merged_biomedrxiv_sft_train.parquet
```

---

## Running SFT Training

```bash
cd /workspace/gxl
WANDB_API_KEY=<your-key> bash train_sft_32k_1ep.sh
```

This runs `torchrun` with the `verl.trainer.sft_trainer` module. Key hyperparameters:

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen3-14B` |
| LoRA rank / alpha | 16 / 32 |
| Max sequence length | 32,768 tokens |
| Micro batch size | 1 per GPU |
| Global train batch | 8 |
| Epochs | 1 |
| Precision | bfloat16 |
| Gradient checkpointing | enabled |

Checkpoints are saved every 50 steps to `/workspace/checkpoints_32k_1ep/`.

Expected training time: **~2 hours** on a single H100 80GB.

---

## Running RL Fine-tuning (optional)

After SFT, merge the LoRA weights and run GRPO:

```bash
# Merge LoRA into full model
python merge_sft_checkpoint.py \
    --checkpoint /workspace/checkpoints_32k_1ep/global_step_<N>

# Run RL training (GRPO with live tool calls as reward signal)
WANDB_API_KEY=<your-key> bash train_rl.sh
```

RL config: n=8 rollouts per prompt, lr=1e-6, 5 epochs, max 8 tool turns per episode.

---

## Data Pipeline

To collect your own SFT data from a set of QA questions:

```bash
# 1. Run Claude as teacher agent on your eval questions → raw JSON traces
bash run_collect_sft.sh

# 2. Filter: remove inefficient traces (redundant calls, empty results)
#    LLM-based scoring for traces with 5+ tool calls
bash run_filter_sft.sh

# 3. Merge datasets and convert to parquet
bash build_sft_data.sh
```

The resulting `merged_biomedrxiv_sft_train.parquet` has ~1,300 traces, each containing:
- The user question
- Full multi-turn conversation (system prompt → tool calls → tool results → final answer)
- Formatted for verl-tool's `MultiTurnSFT` trainer

---

## Repository Structure

```
├── runpod_bootstrap.sh        # Pod setup: install deps + download model
├── train_sft_32k_1ep.sh       # SFT training, 32K context, 1 epoch
├── train_sft.sh               # SFT training, 8K context
├── train_sft_32768.sh         # SFT training, 32K context, multi-epoch
├── train_rl.sh                # RL/GRPO training with live tool use
│
├── collect_sft_data.py        # Run teacher agent → raw traces
├── filter_sft_trajectories.py # Rule + LLM filtering for trace quality
├── convert_to_sft_parquet.py  # JSON traces → verl-tool parquet format
├── build_rl_data.py           # Build RL training dataset from evals
├── merge_sft_checkpoint.py    # Merge LoRA weights into base model
│
├── eval_context_baseline.py   # Evaluate full-context baseline models
├── run_eval.sh                # Run agent evals via inference engine
├── run_collect_sft.sh         # Shell wrapper for data collection
├── run_filter_sft.sh          # Shell wrapper for filtering pipeline
├── build_sft_data.sh          # End-to-end data build
│
├── papers_reader.yaml         # Agent config (MCP URL, tools, system prompt)
└── data_samples/
    ├── _test_10.json          # 10-question smoke test
    └── _test_100.json         # 100-question benchmark
```

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool) | latest | Multi-turn SFT + GRPO training framework |
| torch | ≥2.4 | PyTorch |
| transformers | ≥4.46 | Model loading + tokenization |
| peft | ≥0.13 | LoRA |
| accelerate | ≥1.0 | FSDP distributed training |
| flash-attn | 2.7.4.post1 | Fast attention kernels |
| vllm | ≤0.11.0 | Fast inference for RL rollouts |
| datasets / pyarrow | ≥3.0 | Parquet data loading |
| wandb | latest | Training metrics |
| hf-transfer | latest | Fast HuggingFace model downloads |
