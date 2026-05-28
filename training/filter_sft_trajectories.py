"""
Filter SFT traces by removing inefficient/redundant tool call trajectories.

Two-stage filtering:
  Stage 1: Rule-based detection of obvious waste patterns (free, instant)
  Stage 2: LLM efficiency scoring for longer traces (cheap model, async)

Usage:
    python filter_sft_trajectories.py \\
        --input  data/xulong_biomedrxiv_sft_raw.json \\
        --output data/xulong_biomedrxiv_sft_filtered.json \\
        [--engine-url  http://localhost:8000] \\
        [--judge-model claude-opus-4-5] \\
        [--min-score   3]   # keep traces with LLM score >= this (1–5) \\
        [--max-calls   12]  # hard-reject traces longer than this \\
        [--llm-threshold 5] # only LLM-judge traces with >= this many calls \\
        [--max-concurrent 20] \\
        [--no-llm-judge]    # rule-only (no API calls) \\
        [--dry-run]         # print stats, write nothing
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_LLM_THRESHOLD = 5      # call LLM for traces with >= this many tool calls
DEFAULT_MAX_CALLS     = 12     # hard-reject above this length
DEFAULT_MIN_SCORE     = 3      # keep if LLM score >= this
DEFAULT_MAX_CONCURRENT = 20
DEFAULT_MODEL = "claude-opus-4-5"

JUDGE_SYSTEM_PROMPT = """\
You are an expert at evaluating the efficiency of AI agent tool-use trajectories.

The agent has ONE tool: `papers__paperclip`, a biomedical paper database interface.
Common commands: `lookup doi <doi>`, `scan <path> "kw"`, `grep -n "pat" <path>`, \
`cat <path>`, `search "query"`, `curl <url>`.

Rate EFFICIENCY of the command sequence on a scale of 1–5:
  5 = optimal path, minimum necessary calls, no redundancy
  4 = good, one or two minor inefficiencies
  3 = acceptable, some wasted steps but not excessive
  2 = poor, significant redundancy or unnecessary exploration
  1 = very wasteful, large number of redundant/duplicate steps

Penalise heavily for:
- Running the same command (or near-identical command) twice
- `scan` immediately followed by `cat` on the same file (scan was pointless)
- Grepping the same file 3+ times with overlapping patterns
- Multiple curl/fetch requests to the same paper URL

Return ONLY valid JSON, no prose:
{"score": <1-5>, "wasted_steps": [<0-indexed step numbers>], "reason": "<one sentence>"}
"""


# ─── helpers ──────────────────────────────────────────────────────────────────

def extract_command(tc: dict) -> str:
    """Extract the command string from a tool call dict."""
    fn = tc.get("function", {})
    args_str = fn.get("arguments", "{}")
    try:
        args = json.loads(args_str)
        return args.get("command", args_str)
    except (json.JSONDecodeError, TypeError):
        return str(args_str)


def extract_file_path(cmd: str) -> str | None:
    """Return first /papers/… or /session_files/… path found in a command."""
    m = re.search(r'(/(?:papers|session_files)/\S+)', cmd)
    return m.group(1) if m else None


# ─── Stage 1: rule-based checks ───────────────────────────────────────────────

def _identical_consecutive(commands: list[str]) -> list[str]:
    issues = []
    for i in range(1, len(commands)):
        if commands[i].strip() == commands[i - 1].strip():
            issues.append(
                f"step {i}: identical to step {i-1} ({commands[i][:60]!r})"
            )
    return issues


def _scan_then_cat(commands: list[str]) -> list[str]:
    issues = []
    for i in range(1, len(commands)):
        prev, curr = commands[i - 1], commands[i]
        if "scan " in prev and ("cat " in curr or curr.startswith("cat ")):
            p1, p2 = extract_file_path(prev), extract_file_path(curr)
            if p1 and p1 == p2:
                issues.append(f"steps {i-1}-{i}: scan then cat same file ({p1})")
    return issues


def _repeated_grep(commands: list[str]) -> list[str]:
    """Flag if the same file is grep-ed ≥3 times in any 5-call window."""
    seen: set[str] = set()
    issues = []
    n = len(commands)
    for i in range(n):
        window = commands[max(0, i - 4): i + 1]
        counts: dict[str, int] = {}
        for cmd in window:
            if "grep " in cmd:
                path = extract_file_path(cmd)
                if path:
                    counts[path] = counts.get(path, 0) + 1
        for path, cnt in counts.items():
            if cnt >= 3:
                key = f"grep:{path}:{i}"
                if key not in seen:
                    seen.add(key)
                    issues.append(f"near step {i}: {cnt}× grep on same file ({path})")
    return issues


def _repeated_curl(commands: list[str]) -> list[str]:
    curl_bases: dict[str, list[int]] = {}
    for i, cmd in enumerate(commands):
        if "curl " in cmd:
            m = re.search(r'https?://\S+', cmd)
            if m:
                url = m.group(0).rstrip("'\";")
                parsed = urlparse(url)
                base = f"{parsed.netloc}{parsed.path.split('.')[0]}"
                curl_bases.setdefault(base, []).append(i)
    issues = []
    for base, idxs in curl_bases.items():
        if len(idxs) >= 2:
            issues.append(
                f"steps {idxs}: {len(idxs)}× curl to same base URL ({base})"
            )
    return issues


def rule_based_filter(trace: dict) -> tuple[str, list[str]]:
    """
    Returns:
        ('clean',    [])      — no issues detected
        ('flagged',  [issues])— minor issues; send to LLM judge
        ('rejected', [issues])— clear severe waste; discard directly
    """
    commands = [extract_command(tc) for tc in trace.get("tool_calls", [])]

    issues: list[str] = []
    issues += _identical_consecutive(commands)
    issues += _scan_then_cat(commands)
    issues += _repeated_grep(commands)
    issues += _repeated_curl(commands)

    if not issues:
        return "clean", []

    # Identical calls or curl-spam are clear-cut → reject outright
    severe = any("identical" in iss or "curl" in iss for iss in issues)
    return ("rejected" if severe else "flagged"), issues


# ─── Stage 2: LLM efficiency judge (Anthropic API direct) ────────────────────

async def _judge_one(
    client: anthropic.AsyncAnthropic,
    trace: dict,
    model: str,
    sem: asyncio.Semaphore,
    idx: int,
) -> dict:
    commands = [extract_command(tc) for tc in trace.get("tool_calls", [])]
    numbered = "\n".join(f"[{i}] {cmd}" for i, cmd in enumerate(commands))
    user_msg = f"Question: {trace['question']}\n\nTool call sequence:\n{numbered}"

    async with sem:
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=256,
                temperature=0.0,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = resp.content[0].text if resp.content else ""
        except Exception as exc:
            logger.warning(f"[{idx}] LLM judge request failed: {exc}")
            return {
                **trace,
                "efficiency_score": None,
                "efficiency_reason": f"judge error: {exc}",
                "wasted_steps": [],
            }

    try:
        content = raw
        if "```json" in content:
            content = content.split("```json", 1)[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```", 1)[1].split("```")[0].strip()
        parsed = json.loads(content)
        score  = int(parsed.get("score", 3))
        wasted = parsed.get("wasted_steps", [])
        reason = parsed.get("reason", "")
    except Exception as exc:
        logger.warning(f"[{idx}] Failed to parse judge JSON: {exc} | raw: {raw[:200]}")
        score, wasted, reason = 3, [], f"parse error: {exc}"

    return {**trace, "efficiency_score": score, "efficiency_reason": reason, "wasted_steps": wasted}


async def llm_judge_batch(
    traces: list[dict],
    model: str,
    max_concurrent: int,
) -> list[dict]:
    sem = asyncio.Semaphore(max_concurrent)
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    tasks = [
        _judge_one(client, t, model, sem, i)
        for i, t in enumerate(traces)
    ]
    return list(await asyncio.gather(*tasks))


# ─── main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="Filter SFT trajectories for efficiency")
    parser.add_argument("--input",  required=True, help="Raw traces JSON")
    parser.add_argument("--output", required=True, help="Filtered output JSON")
    parser.add_argument("--engine-url",    default="http://localhost:8000")
    parser.add_argument("--judge-model",   default=DEFAULT_MODEL)
    parser.add_argument("--min-score",     type=int, default=DEFAULT_MIN_SCORE,
                        help="Min LLM score to keep (1–5, default 3)")
    parser.add_argument("--max-calls",     type=int, default=DEFAULT_MAX_CALLS,
                        help="Hard-reject traces with more tool calls (default 12)")
    parser.add_argument("--llm-threshold", type=int, default=DEFAULT_LLM_THRESHOLD,
                        help="Only LLM-judge traces with >= this many calls (default 5)")
    parser.add_argument("--max-concurrent",type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--no-llm-judge",  action="store_true",
                        help="Skip LLM judge; treat all flagged as rejected")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Print stats only, write no files")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    traces: list[dict] = data.get("traces", [])
    logger.info(f"Loaded {len(traces)} traces from {args.input}")

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    auto_keep:    list[dict] = []  # short + clean → keep without LLM
    to_judge:     list[dict] = []  # need LLM scoring
    rule_rejected:list[dict] = []  # rule-rejected or hard-reject

    for trace in traces:
        n = len(trace.get("tool_calls", []))

        if n > args.max_calls:
            trace["filter_stage"]  = "hard_reject"
            trace["filter_issues"] = [f"too many calls: {n} > {args.max_calls}"]
            rule_rejected.append(trace)
            continue

        status, issues = rule_based_filter(trace)
        trace["filter_stage"]  = f"rule_{status}"
        trace["filter_issues"] = issues

        if status == "rejected":
            rule_rejected.append(trace)
        elif status == "flagged" or n >= args.llm_threshold:
            # flagged traces always go to LLM; so do long-but-clean traces
            to_judge.append(trace)
        else:
            auto_keep.append(trace)

    logger.info(
        f"Stage 1 → auto-keep: {len(auto_keep)}, to-judge: {len(to_judge)}, "
        f"rule-rejected: {len(rule_rejected)}"
    )

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    llm_keep:   list[dict] = []
    llm_reject: list[dict] = []

    if to_judge:
        if args.no_llm_judge:
            # Without LLM: keep clean long traces, reject flagged ones
            for t in to_judge:
                if "flagged" in t.get("filter_stage", ""):
                    t["filter_stage"] = "rule_flagged_reject"
                    llm_reject.append(t)
                else:
                    t["filter_stage"] = "rule_long_keep"
                    llm_keep.append(t)
            logger.info(
                f"Stage 2 (no-llm) → keep: {len(llm_keep)}, reject: {len(llm_reject)}"
            )
        else:
            logger.info(
                f"Stage 2: LLM-judging {len(to_judge)} traces "
                f"with {args.judge_model} (min-score={args.min_score}) ..."
            )
            judged = await llm_judge_batch(
                to_judge, args.judge_model, args.max_concurrent
            )
            for t in judged:
                score = t.get("efficiency_score")
                if score is None or score >= args.min_score:
                    t["filter_stage"] = "llm_keep"
                    llm_keep.append(t)
                else:
                    t["filter_stage"] = "llm_reject"
                    llm_reject.append(t)
            logger.info(
                f"Stage 2 → keep: {len(llm_keep)}, reject: {len(llm_reject)}"
            )

    kept     = auto_keep + llm_keep
    rejected = rule_rejected + llm_reject

    # ── Summary ───────────────────────────────────────────────────────────────
    n_in  = len(traces)
    n_out = len(kept)

    avg_in  = sum(len(t.get("tool_calls", [])) for t in traces) / max(n_in, 1)
    avg_out = sum(len(t.get("tool_calls", [])) for t in kept)   / max(n_out, 1)

    score_dist: dict[int, int] = {}
    for t in kept + rejected:
        s = t.get("efficiency_score")
        if s is not None:
            score_dist[s] = score_dist.get(s, 0) + 1

    print(f"\n{'='*55}")
    print(f"  Input  traces:      {n_in}")
    print(f"  Kept:               {n_out}  ({100*n_out/max(n_in,1):.1f}%)")
    print(f"  Rejected:           {len(rejected)}  ({100*len(rejected)/max(n_in,1):.1f}%)")
    print(f"  Avg calls  (in):    {avg_in:.2f}")
    print(f"  Avg calls  (out):   {avg_out:.2f}")
    if score_dist:
        print(f"  Score distribution: {dict(sorted(score_dist.items()))}")
    print(f"{'='*55}\n")

    if args.dry_run:
        logger.info("Dry run — no output written.")
        return

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "agent":         data.get("agent", ""),
        "system_prompt": data.get("system_prompt", ""),
        "tools":         data.get("tools", []),
        "traces":        kept,
        "filter_stats": {
            "input":              n_in,
            "kept":               n_out,
            "rejected":           len(rejected),
            "avg_calls_input":    round(avg_in, 2),
            "avg_calls_output":   round(avg_out, 2),
            "score_distribution": score_dist,
        },
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f"Written {n_out} filtered traces → {out}")


if __name__ == "__main__":
    asyncio.run(main())
