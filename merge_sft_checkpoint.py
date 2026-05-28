"""
Merge a verl SFT LoRA checkpoint into a full HuggingFace model directory.

Usage:
  python qwen_rl/merge_sft_checkpoint.py \
      --checkpoint qwen_rl/checkpoints/global_step_100 \
      --output     qwen_rl/checkpoints/global_step_100_merged
"""

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL  = "Qwen/Qwen3-14B"
LORA_RANK   = 16
LORA_ALPHA  = 32
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]


def merge(checkpoint_dir: str, output_dir: str):
    ckpt = Path(checkpoint_dir)
    out  = Path(output_dir)
    pt_file = ckpt / "model_world_size_1_rank_0.pt"

    if not pt_file.exists():
        raise FileNotFoundError(f"No model weights at {pt_file}")

    print(f"Loading base model: {BASE_MODEL}")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    print(f"Loading LoRA checkpoint: {pt_file}")
    lora_cfg = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    state_dict = torch.load(pt_file, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:3]}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]}")

    print("Merging LoRA into base model...")
    model = model.merge_and_unload()

    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {out}")
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output",     required=True)
    args = parser.parse_args()
    merge(args.checkpoint, args.output)
