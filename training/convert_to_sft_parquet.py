"""
Convert collected Claude traces (JSON) to verl-tool MultiTurnSFT parquet format.

The output parquet has columns:
  - messages: list of dicts [{role, content, tool_calls?, tool_call_id?}]
  - tools:    list of OpenAI tool specs (for chat template)

Usage:
    python convert_to_sft_parquet.py \
        --input   qwen_rl/data/biomedrxiv_sft_filtered.json \
        --output  qwen_rl/data/biomedrxiv_sft \
        --system-prompt agents/papers/papers_reader.yaml \
        --val-split 0.1
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# Tool spec exposed to the Qwen model (OpenAI function-calling format).
# Matches the paperclip interface used in papers_reader.yaml.
PAPERCLIP_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "paperclip",
        "description": (
            "Query a biomedical paper filesystem (8M+ papers from bioRxiv, medRxiv, PMC). "
            "Pass shell-like commands: lookup, scan, grep, cat, head, search, map, sql, links, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to run, e.g. 'lookup doi 10.1101/...' or 'grep -n \"pattern\" /papers/UUID/content.lines'",
                },
                "description": {
                    "type": "string",
                    "description": "One-sentence description of what this call is trying to find.",
                },
            },
            "required": ["command"],
        },
    },
}

# Tool name used in the raw collected data (MCP server prefix + tool name)
_RAW_TOOL_NAME = "papers__paperclip"
# Tool name we expose to the Qwen model
_QWEN_TOOL_NAME = "paperclip"


def load_system_prompt(yaml_path: str | None) -> str | None:
    """Load system_prompt from a YAML agent config file, or return None."""
    if not yaml_path:
        return None
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"System prompt YAML not found: {yaml_path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    prompt = cfg.get("system_prompt", "")
    if not prompt:
        raise ValueError(f"No 'system_prompt' key found in {yaml_path}")
    return prompt.strip()


def build_messages(
    system_prompt: str,
    question: str,
    tool_calls: list,
    final_response: str,
) -> list[dict]:
    """
    Reconstruct full multi-turn conversation from a Claude trace.

    Structure:
        system
        user (question)
        [for each tool call:]
            assistant  (tool_calls array — tool name normalised to paperclip)
            tool       (result)
        assistant  (final response — internal planning markers stripped)
    """
    messages: list[dict] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": question})

    for tc in tool_calls:
        fn = tc.get("function", {})
        tc_id = tc.get("id") or f"call_{len(messages)}"

        # Normalise tool name: papers__paperclip → paperclip
        raw_name = fn.get("name", "")
        qwen_name = _QWEN_TOOL_NAME if raw_name == _RAW_TOOL_NAME else raw_name

        # Strip 'description' field from arguments — it's Claude-internal metadata
        args_str = fn.get("arguments", "{}")
        try:
            args = json.loads(args_str)
            args.pop("description", None)
            args_str = json.dumps(args, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            pass

        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": qwen_name,
                    "arguments": args_str,
                },
            }],
        })

        # Tool result
        result = tc.get("result", "")
        messages.append({
            "role": "tool",
            "content": result,
            "tool_call_id": tc_id,
        })

    clean_response = _clean_final_response(final_response)
    if clean_response:
        messages.append({"role": "assistant", "content": clean_response})

    return messages


def _clean_final_response(text: str) -> str:
    """
    Strip Claude-internal planning text from the final response:
      - Everything before ---FINAL_RESPONSE--- (including that marker)
      - CLASS: / BUDGET: lines that leaked without the marker
    """
    if "---FINAL_RESPONSE---" in text:
        return text.split("---FINAL_RESPONSE---", 1)[1].strip()

    lines = text.split("\n")
    cleaned = [
        line for line in lines
        if not re.match(r"^(CLASS|BUDGET):\s*", line)
    ]
    return "\n".join(cleaned).strip()


def load_holdout_questions(holdout_path: str) -> set[str]:
    """Load questions from a holdout eval file to exclude from training."""
    with open(holdout_path) as f:
        d = json.load(f)
    # Support both eval format (evals[].input) and trace format (traces[].question)
    if "evals" in d:
        return {e["input"] for e in d["evals"]}
    if "traces" in d:
        return {t["question"] for t in d["traces"]}
    return set()


def convert(
    input_path: str,
    output_path: str,
    system_prompt_yaml: str | None = None,
    val_split: float = 0.1,
    holdout_path: str | None = None,
) -> None:
    with open(input_path) as f:
        data = json.load(f)

    traces = data.get("traces", [])

    # Exclude holdout questions from training data
    if holdout_path:
        holdout_qs = load_holdout_questions(holdout_path)
        before = len(traces)
        traces = [t for t in traces if t.get("question", "") not in holdout_qs]
        removed = before - len(traces)
        print(f"Holdout:       {holdout_path}  ({len(holdout_qs)} questions, removed {removed} overlapping traces)")

    # System prompt: prefer YAML override, fall back to embedded prompt in raw data
    if system_prompt_yaml:
        system_prompt = load_system_prompt(system_prompt_yaml)
        print(f"System prompt: {system_prompt_yaml} ({len(system_prompt)} chars)")
    else:
        system_prompt = data.get("system_prompt", "")
        print(f"System prompt: from input JSON ({len(system_prompt)} chars)")

    print(f"Tool spec:     {_QWEN_TOOL_NAME}  (renamed from {_RAW_TOOL_NAME})")
    print(f"Converting {len(traces)} traces...")

    rows = []
    skipped = 0
    for trace in traces:
        question      = trace.get("question", "")
        tool_calls    = trace.get("tool_calls", [])
        final_response = trace.get("final_response", "")

        if not question or not final_response:
            skipped += 1
            continue

        messages = build_messages(system_prompt, question, tool_calls, final_response)

        if not any(m["role"] == "assistant" for m in messages):
            skipped += 1
            continue

        rows.append({
            "messages": messages,
            "tools": [PAPERCLIP_TOOL_SPEC],
        })

    print(f"Valid rows: {len(rows)}, skipped: {skipped}")
    if not rows:
        print("No valid rows, exiting.")
        return

    df = pd.DataFrame(rows)

    # Shuffle so train/val share the same tool-call-length distribution
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n_val   = max(1, int(len(df) * val_split)) if val_split > 0 else 0
    n_train = len(df) - n_val

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    train_path = out.parent / (out.stem + "_train.parquet")
    val_path   = out.parent / (out.stem + "_val.parquet")

    df.iloc[:n_train].to_parquet(train_path, index=False)
    print(f"Train: {n_train} rows → {train_path}")

    if n_val > 0:
        df.iloc[n_train:].to_parquet(val_path, index=False)
        print(f"Val:   {n_val} rows → {val_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",         required=True,
                        help="Filtered traces JSON (from filter_sft_trajectories.py)")
    parser.add_argument("--output",        required=True,
                        help="Output parquet path stem (_train/_val suffixes added)")
    parser.add_argument("--system-prompt", default=None,
                        help="YAML agent config to read system_prompt from "
                             "(e.g. agents/papers/papers_reader.yaml). "
                             "Overrides the system_prompt embedded in --input.")
    parser.add_argument("--val-split",     type=float, default=0.1,
                        help="Fraction for validation set (default 0.1)")
    parser.add_argument("--holdout",       default=None,
                        help="Eval JSON file whose questions must be excluded from training "
                             "(e.g. qwen_rl/data/_test_50.json)")
    args = parser.parse_args()

    convert(args.input, args.output, args.system_prompt, args.val_split, args.holdout)
