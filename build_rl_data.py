"""
Build RL training parquet from biomedrxiv SFT data + eval JSON.

Output schema (matches verl-tool RL format):
  data_source, prompt, ability, reward_model, extra_info
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def load_system_prompt(yaml_path: str) -> str:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["system_prompt"].strip()


def from_eval_json(eval_file: str, system_prompt: str) -> list[dict]:
    with open(eval_file) as f:
        data = json.load(f)
    evals = data.get("evals", data) if isinstance(data, dict) else data
    rows = []
    for item in evals:
        question = item["input"]
        criteria = item.get("criteria", [])
        rows.append({
            "data_source": "biomedrxiv",
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": question},
            ],
            "ability": "paper_qa",
            "reward_model": {"ground_truth": criteria},
            "extra_info": {
                "question": question,
                "category": item.get("category", "unknown"),
            },
        })
    return rows


def from_sft_parquet(parquet_file: str) -> list[dict]:
    """Convert SFT trajectories: keep only system+user turns as prompt, use assistant reply as criteria."""
    df = pd.read_parquet(parquet_file)
    rows = []
    for _, row in df.iterrows():
        messages = row["messages"]
        # extract system + first user message
        prompt = [m for m in messages if m["role"] in ("system", "user")][:2]
        if len(prompt) < 2:
            continue
        # extract last assistant message as pseudo ground truth
        assistant_turns = [m["content"] for m in messages if m["role"] == "assistant"]
        ground_truth = assistant_turns[-1] if assistant_turns else ""
        rows.append({
            "data_source": "biomedrxiv_sft",
            "prompt": prompt,
            "ability": "paper_qa",
            "reward_model": {"ground_truth": [ground_truth]},
            "extra_info": {"category": "sft"},
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json",    default="qwen_rl/data/_test_50.json")
    parser.add_argument("--sft-parquet",  default=None,
                        help="Optional SFT parquet to add as additional prompts")
    parser.add_argument("--agent-yaml",   default="agents/papers/papers_reader.yaml")
    parser.add_argument("--output-train", default="qwen_rl/data/rl_train.parquet")
    parser.add_argument("--output-val",   default="qwen_rl/data/rl_val.parquet")
    parser.add_argument("--val-split",    type=float, default=0.1)
    args = parser.parse_args()

    system_prompt = load_system_prompt(args.agent_yaml)
    rows = from_eval_json(args.eval_json, system_prompt)

    if args.sft_parquet:
        rows += from_sft_parquet(args.sft_parquet)

    split = max(1, int(len(rows) * args.val_split))
    val_rows, train_rows = rows[:split], rows[split:]

    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(args.output_train, index=False)
    pd.DataFrame(val_rows).to_parquet(args.output_val, index=False)
    print(f"train: {len(train_rows)} rows → {args.output_train}")
    print(f"val:   {len(val_rows)} rows → {args.output_val}")


if __name__ == "__main__":
    main()
