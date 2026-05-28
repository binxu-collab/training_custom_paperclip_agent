"""
Context-stuffed baseline: read pre-fetched paper content from local files,
then ask a model to answer each question from full context (no tool calls).

Supports Anthropic models directly and any OpenAI-compatible provider (e.g. OpenRouter).

Usage:
  python qwen_rl/eval_context_baseline.py
  python qwen_rl/eval_context_baseline.py --model claude-opus-4-5 --concurrency 10
  python qwen_rl/eval_context_baseline.py --limit 10
  # OpenRouter (Qwen, etc.)
  python qwen_rl/eval_context_baseline.py \
      --model qwen/qwen3-8b --provider openrouter --concurrency 4
"""

import argparse
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONTEXT_DIR = Path(__file__).parent / "paper_contexts"
EVAL_FILE = Path(__file__).parent / "data/_test_50.json"
JUDGE_MODEL = "claude-opus-4-5"

_DOI_RE = re.compile(r"DOI:\s*([\w./\-]+)")

ANSWER_SYSTEM = """\
You are a precise scientific reader. You are given the complete content of a biomedical \
paper (main text and any supplementary files) followed by a specific question about it.

Answer the question based solely on the provided paper content. Be concise and exact — \
give the specific number, value, or phrase asked for. If the answer is a statistic or \
numerical value, look carefully through tables, figure captions, and supplement content.

Do not say you cannot find the answer unless you have genuinely checked all sections.
"""

JUDGE_SYSTEM = """\
You are an evaluator checking whether an AI answer satisfies specific criteria.
Reply ONLY with valid JSON: {"pass": true/false, "reason": "one sentence"}.
"""


async def answer_question_anthropic(
    client: anthropic.AsyncAnthropic,
    question: str,
    context: str,
    model: str,
    sem: asyncio.Semaphore,
) -> str:
    user_msg = f"<paper_content>\n{context}\n</paper_content>\n\nQuestion: {question}"
    async with sem:
        for attempt in range(6):
            try:
                create_kwargs = dict(
                    model=model,
                    max_tokens=512,
                    system=ANSWER_SYSTEM,
                    messages=[{"role": "user", "content": user_msg}],
                )
                # claude-opus-4-7 and some newer models don't support temperature=0.0
                if "opus-4-7" not in model and "claude-4-7" not in model:
                    create_kwargs["temperature"] = 0.0
                resp = await client.messages.create(**create_kwargs)
                return resp.content[0].text if resp.content else ""
            except anthropic.RateLimitError:
                if attempt == 5:
                    raise
                wait = 60 * (attempt + 1)
                logger.warning(
                    f"Rate limited (attempt {attempt+1}), waiting {wait}s..."
                )
                await asyncio.sleep(wait)
        return ""  # unreachable


async def answer_question_openai(
    client: AsyncOpenAI,
    question: str,
    context: str,
    model: str,
    sem: asyncio.Semaphore,
    thinking: bool = False,
) -> tuple[str, int]:
    # Qwen3: append /think or /no_think to control extended reasoning
    think_tag = "/think" if thinking else "/no_think"
    user_msg = f"<paper_content>\n{context}\n</paper_content>\n\nQuestion: {question} {think_tag}"
    async with sem:
        for attempt in range(6):
            try:
                chunks = []
                ttft_ms = -1
                t0 = time.perf_counter()
                stream = await client.chat.completions.create(
                    model=model,
                    max_tokens=512,
                    temperature=0.0,
                    stream=True,
                    messages=[
                        {"role": "system", "content": ANSWER_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        if ttft_ms < 0:
                            ttft_ms = int((time.perf_counter() - t0) * 1000)
                        chunks.append(delta)
                text = "".join(chunks)
                # strip <think> blocks (Qwen3 reasoning)
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                return text, max(ttft_ms, 0)
            except Exception as e:
                if attempt == 5:
                    raise
                wait = 30 * (attempt + 1)
                logger.warning(f"Error (attempt {attempt+1}): {e}, waiting {wait}s...")
                await asyncio.sleep(wait)
        return "", 0  # unreachable


async def answer_question(
    client, question, context, model, sem, provider="anthropic", thinking=False
) -> tuple[str, int]:
    if provider == "anthropic":
        text = await answer_question_anthropic(client, question, context, model, sem)
        return text, -1
    return await answer_question_openai(client, question, context, model, sem, thinking=thinking)


async def judge_answer(
    client: anthropic.AsyncAnthropic,  # judge always uses Anthropic
    question: str,
    answer: str,
    criteria: list[str],
    sem: asyncio.Semaphore,
) -> dict:
    criteria_str = "\n".join(f"- {c}" for c in criteria)
    user_msg = f"Question: {question}\n\nAnswer:\n{answer}\n\nCriteria:\n{criteria_str}"
    async with sem:
        try:
            resp = await client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=128,
                temperature=0.0,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = resp.content[0].text if resp.content else "{}"
            for tag in ["```json", "```"]:
                if tag in raw:
                    raw = raw.split(tag, 1)[1].split("```")[0].strip()
                    break
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"Judge error: {e}")
            return {"pass": False, "reason": f"judge error: {e}"}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", default=str(EVAL_FILE))
    parser.add_argument("--context-dir", default=str(CONTEXT_DIR))
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: auto-derived from model name)",
    )
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "openrouter", "vllm"],
        help="Which provider to use for the answer model (judge always uses Anthropic)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL (e.g. http://localhost:8001/v1 for local vLLM)",
    )
    parser.add_argument(
        "--thinking", action="store_true", default=False,
        help="Enable Qwen3 extended reasoning (/think). Default: off (/no_think).",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    # Qwen3-8B on Alibaba via OpenRouter: max 98304 tokens
    # observed ~1.2 chars/token for paper content; 100K chars ≈ 83K tokens — safe ceiling
    parser.add_argument("--max-context-chars", type=int, default=100_000)
    args = parser.parse_args()

    with open(args.eval_file) as f:
        data = json.load(f)
    evals = data.get("evals", data) if isinstance(data, dict) else data
    if args.limit:
        evals = evals[: args.limit]

    if args.output is None:
        slug = args.model.replace("/", "_").replace("-", "_").replace(".", "_")
        think_tag = "_think" if args.thinking else "_nothink"
        args.output = f"qwen_rl/eval_results/context_baseline_{slug}{think_tag}.json"

    logger.info(
        f"Loaded {len(evals)} questions | model={args.model} | provider={args.provider}"
    )

    if args.provider == "openrouter":
        ans_client = AsyncOpenAI(
            base_url=args.base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    elif args.provider == "vllm":
        ans_client = AsyncOpenAI(
            base_url=args.base_url or "http://localhost:8001/v1",
            api_key="dummy",
        )
    else:
        ans_client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    judge_client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    ans_sem = asyncio.Semaphore(args.concurrency)
    jdg_sem = asyncio.Semaphore(args.concurrency)

    context_dir = Path(args.context_dir)
    results = []

    async def process(item):
        question = item["input"]
        category = item.get("category", "unknown")
        criteria = item.get("criteria", [])

        doi_m = _DOI_RE.search(question)
        doi = doi_m.group(1).strip() if doi_m else None

        # load pre-fetched context
        context = ""
        if doi:
            ctx_path = context_dir / (doi.replace("/", "_") + ".txt")
            if ctx_path.exists():
                context = ctx_path.read_text(encoding="utf-8")
                if len(context) > args.max_context_chars:
                    logger.warning(
                        f"Truncating context for {doi}: {len(context)} → {args.max_context_chars} chars"
                    )
                    context = context[: args.max_context_chars]
            else:
                logger.warning(f"No context file for {doi}")

        ctx_kb = len(context) // 1024
        logger.info(f"Q: {question[:80]}...  [{category}] ctx={ctx_kb}KB")

        t0 = time.perf_counter()
        answer, ttft_ms = await answer_question(
            ans_client, question, context, args.model, ans_sem, args.provider,
            thinking=args.thinking,
        )
        answer_ms = int((time.perf_counter() - t0) * 1000)

        judgment = await judge_answer(judge_client, question, answer, criteria, jdg_sem)
        passed = judgment.get("pass", False)

        ttft_str = f"  ttft={ttft_ms}ms" if ttft_ms >= 0 else ""
        logger.info(
            f"  {'PASS' if passed else 'FAIL'}  total={answer_ms}ms{ttft_str}  {judgment.get('reason','')[:80]}"
        )
        return {
            "question": question,
            "category": category,
            "doi": doi,
            "ctx_chars": len(context),
            "answer": answer,
            "criteria": criteria,
            "judgment": judgment,
            "pass": passed,
            "answer_ms": answer_ms,
            "ttft_ms": ttft_ms,
        }

    wall_t0 = time.perf_counter()
    tasks = [process(item) for item in evals]
    results = await asyncio.gather(*tasks)
    wall_ms = int((time.perf_counter() - wall_t0) * 1000)

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["pass"])
    accuracy = passed / total if total else 0.0
    avg_ms = int(sum(r["answer_ms"] for r in results) / total) if total else 0
    ttft_results = [r for r in results if r.get("ttft_ms", -1) >= 0]
    avg_ttft_ms = (
        int(sum(r["ttft_ms"] for r in ttft_results) / len(ttft_results))
        if ttft_results
        else -1
    )

    by_cat: dict[str, list[bool]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r["pass"])

    print(f"\n{'='*55}")
    print(f"  Model:      {args.model}  ({'think' if args.thinking else 'no_think'})")
    print(f"  Total:      {total}")
    print(f"  Passed:     {passed}  ({accuracy:.1%})")
    print(f"  Wall time:  {wall_ms/1000:.1f}s")
    print(f"  Avg answer: {avg_ms}ms / question")
    if avg_ttft_ms >= 0:
        print(f"  Avg TTFT:   {avg_ttft_ms}ms / question  (prefill time)")
    for cat, vals in sorted(by_cat.items()):
        p = sum(vals)
        cat_rs = [r for r in results if r["category"] == cat]
        cat_ms = int(sum(r["answer_ms"] for r in cat_rs) / len(vals))
        cat_ttfts = [r["ttft_ms"] for r in cat_rs if r.get("ttft_ms", -1) >= 0]
        ttft_str = f"  ttft={int(sum(cat_ttfts)/len(cat_ttfts))}ms" if cat_ttfts else ""
        print(
            f"  {cat:25s}: {p}/{len(vals)} = {p/len(vals):.1%}  (avg {cat_ms}ms{ttft_str})"
        )
    print(f"{'='*55}\n")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {
                "model": args.model,
                "eval_file": args.eval_file,
                "accuracy": accuracy,
                "passed": passed,
                "total": total,
                "wall_ms": wall_ms,
                "avg_answer_ms": avg_ms,
                "avg_ttft_ms": avg_ttft_ms,
                "by_category": {
                    cat: {
                        "passed": sum(v),
                        "total": len(v),
                        "accuracy": f"{sum(v)/len(v):.1%}",
                        "avg_ms": int(
                            sum(r["answer_ms"] for r in results if r["category"] == cat)
                            / len(v)
                        ),
                    }
                    for cat, v in by_cat.items()
                },
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"Saved → {out}")


if __name__ == "__main__":
    asyncio.run(main())
