"""
Collect SFT training data by running Claude on eval questions and saving full
multi-turn conversation traces (system prompt + tool calls + results + final response).

Usage:
    python collect_sft_data.py \
        --input apps/evals/inputs_final/biomedrxiv_evals.json \
        --agent agents/biomedrxiv/biomedrxiv \
        --output qwen_rl/data/biomedrxiv_sft_raw.json \
        --max-concurrent 5 \
        --only-passing          # only keep traces where Claude answered correctly
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path

import httpx
import yaml

# Reuse AgentRunner from evals
sys.path.insert(0, str(Path(__file__).parent.parent / "apps/evals/src"))
from evals.components.agent_runner import AgentRunner, AGENT_CONFIG_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


PAPERS_MCP_URL = "http://localhost:8083"


async def fetch_tool_calls_from_messages(engine_url: str, session_id: str, agent_id: str) -> list[dict] | None:
    """Rebuild tool_calls with results from the session's DB messages.
    Returns a flat list of tool calls (with 'result' field), or None on failure.
    """
    if not session_id or not agent_id:
        return None
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{engine_url}/api/sessions/{session_id}/messages",
                params={"agent_id": agent_id, "group_limit": 100},
                timeout=15.0,
            )
            r.raise_for_status()
            data = r.json()
        messages = data.get("messages", [])
        # Build tool_result lookup: tool_call_id -> content
        tool_results: dict[str, str] = {}
        for msg in messages:
            if msg.get("role") == "tool" and msg.get("tool_call_id"):
                content = msg.get("message_content", "")
                if isinstance(content, dict):
                    import json as _json
                    content = _json.dumps(content)
                tool_results[msg["tool_call_id"]] = str(content)
        # Extract tool_calls from assistant messages, attach results
        tool_calls = []
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in (msg.get("tool_calls") or []):
                    tc_id = tc.get("id", "")
                    entry = dict(tc)
                    if tc_id in tool_results:
                        entry["result"] = tool_results[tc_id]
                    tool_calls.append(entry)
        return tool_calls if tool_calls else None
    except Exception as e:
        logger.warning(f"Failed to fetch tool calls from messages for session {session_id}: {e}")
        return None


async def fetch_tool_results(engine_url: str, session_id: str, agent_id: str) -> dict[str, str]:
    """Fetch tool results from the session's message history.
    Returns a dict mapping tool_call_id -> result content.
    """
    if not session_id or not agent_id:
        return {}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{engine_url}/api/sessions/{session_id}/messages",
                params={"agent_id": agent_id, "group_limit": 100},
                timeout=15.0,
            )
            r.raise_for_status()
            data = r.json()
        results = {}
        for msg in data.get("messages", []):
            if msg.get("role") == "tool" and msg.get("tool_call_id"):
                results[msg["tool_call_id"]] = msg.get("message_content", "")
        return results
    except Exception as e:
        logger.warning(f"Failed to fetch tool results for session {session_id}: {e}")
        return {}


async def delete_session(engine_url: str, session_id: str) -> None:
    """Delete a session from the inference engine and papers MCP server."""
    if not session_id:
        return
    async with httpx.AsyncClient() as client:
        try:
            await client.delete(f"{engine_url}/api/sessions/{session_id}", timeout=10.0)
        except Exception as e:
            logger.warning(f"Failed to delete inference session {session_id}: {e}")
        try:
            await client.delete(f"{PAPERS_MCP_URL}/sessions/{session_id}", timeout=10.0)
        except Exception as e:
            logger.warning(f"Failed to delete papers MCP session {session_id}: {e}")


async def collect_one(
    runner: AgentRunner,
    agent_name: str,
    question: str,
    semaphore: asyncio.Semaphore,
    engine_url: str,
    no_tool_results: bool = False,
) -> dict | None:
    """Run one question and return the structured trace."""
    async with semaphore:
        try:
            result = await runner.run(
                agent_name=agent_name,
                prompt=question,
                collect_subagent_traces=False,
            )
        except Exception as e:
            logger.warning(f"Run failed: {e}")
            return None

        if result.get("error"):
            return None

        raw = result.get("raw", {})
        tool_calls = raw.get("tool_calls", [])
        final_response = raw.get("content", "")
        session_id = raw.get("session_id", "")
        agent_id = raw.get("agent_id", "")

        trace = {
            "question": question,
            "session_id": session_id,
            "tool_calls": tool_calls,   # list of OpenAI-format tool_calls with .result attached
            "final_response": final_response,
        }

        if no_tool_results:
            # Strip results — keep only call metadata (id, function name/arguments)
            trace["tool_calls"] = [
                {k: v for k, v in tc.items() if k != "result"}
                for tc in tool_calls
            ]
        else:
            # Rebuild tool_calls from DB messages (avoids SSE delta overwrite bug)
            db_tool_calls = await fetch_tool_calls_from_messages(engine_url, session_id, agent_id)
            if db_tool_calls:
                trace["tool_calls"] = db_tool_calls

        # Clean up session immediately after collecting the trace
        await delete_session(engine_url, session_id)

        return trace


def clear_sessions_history(sessions_dir: str = "/workspaces/gxl/sessions") -> None:
    """Truncate sessions table in DB and clear the sessions filesystem directory."""
    import subprocess
    try:
        import psycopg2
        ip = subprocess.check_output(
            ["docker", "inspect", "gxl-inference-db", "--format",
             "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}"],
            text=True,
        ).strip()
        conn = psycopg2.connect(f"postgresql://gxl:gxl@{ip}:5432/gxl_inference_dev")
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sessions")
        n = cur.fetchone()[0]
        cur.execute("TRUNCATE sessions CASCADE")
        conn.commit()
        conn.close()
        logger.info(f"Cleared {n} sessions from DB")
    except Exception as e:
        logger.warning(f"DB session clear failed: {e}")

    sessions_path = Path(sessions_dir)
    if sessions_path.exists():
        cleared = 0
        for item in sessions_path.iterdir():
            try:
                if item.is_dir():
                    import shutil
                    shutil.rmtree(item)
                else:
                    item.unlink()
                cleared += 1
            except Exception as e:
                logger.warning(f"Failed to remove {item}: {e}")
        logger.info(f"Cleared {cleared} items from {sessions_dir}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Eval JSON file")
    parser.add_argument("--agent", required=True, help="Agent name or path")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument("--only-passing", action="store_true",
                        help="Cross-reference with a judge to keep only correct traces")
    parser.add_argument("--engine-url", default="http://localhost:8000")
    parser.add_argument("--judge-results", default=None,
                        help="Optional: path to existing eval results JSON to filter by passed=True")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Clear session history and save after every N questions (0 = disabled)")
    parser.add_argument("--no-tool-results", action="store_true",
                        help="Omit tool call results from traces (keep only call metadata)")
    args = parser.parse_args()

    # Load questions
    with open(args.input) as f:
        eval_data = json.load(f)
    questions = [e["input"] for e in eval_data["evals"]]
    logger.info(f"Loaded {len(questions)} questions from {args.input}")

    # Optionally pre-filter to only questions Claude already answered correctly
    passing_questions = set(questions)
    if args.judge_results:
        with open(args.judge_results) as f:
            judge_data = json.load(f)
        passing_questions = {r["input"] for r in judge_data if r.get("passed")}
        logger.info(f"Filtering to {len(passing_questions)} passing questions from judge results")
        questions = [q for q in questions if q in passing_questions]

    # Init runner
    runner = AgentRunner(engine_url=args.engine_url, timeout=300.0)
    await runner.prefetch_tools(args.agent)

    # Resolve agent config (system prompt + tools) once upfront
    agent_stem = Path(args.agent).name
    agent_config_path = AGENT_CONFIG_ROOT / "agents" / args.agent / f"{agent_stem}.yaml"
    if not agent_config_path.exists():
        agent_config_path = AGENT_CONFIG_ROOT / "agents" / f"{args.agent}.yaml"
    if not agent_config_path.exists():
        agent_config_path = AGENT_CONFIG_ROOT / args.agent / f"{agent_stem}.yaml"
    if not agent_config_path.exists():
        agent_config_path = AGENT_CONFIG_ROOT / f"{args.agent}.yaml"
    system_prompt = ""
    if agent_config_path.exists():
        with open(agent_config_path) as f:
            cfg = yaml.safe_load(f)
        system_prompt = cfg.get("system_prompt", "")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def save_output(traces: list) -> None:
        tools_spec = runner._tools_cache.get(args.agent, [])
        payload = {
            "agent": args.agent,
            "system_prompt": system_prompt,
            "tools": tools_spec,
            "traces": traces,
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(traces)} traces to {output_path}")

    semaphore = asyncio.Semaphore(args.max_concurrent)
    traces: list[dict] = []

    if args.batch_size > 0:
        # Process in batches; save + clear sessions after each batch
        for batch_start in range(0, len(questions), args.batch_size):
            batch = questions[batch_start: batch_start + args.batch_size]
            batch_num = batch_start // args.batch_size + 1
            total_batches = (len(questions) + args.batch_size - 1) // args.batch_size
            logger.info(f"Batch {batch_num}/{total_batches}: {len(batch)} questions")

            tasks = [collect_one(runner, args.agent, q, semaphore, args.engine_url, args.no_tool_results) for q in batch]
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result:
                    traces.append(result)
                    logger.info(f"Collected {len(traces)}/{len(questions)} traces total")

            logger.info(f"Batch {batch_num} done — saving and clearing sessions...")
            save_output(traces)
            clear_sessions_history()
    else:
        # Original behaviour: collect all at once, save at the end
        tasks = [collect_one(runner, args.agent, q, semaphore, args.engine_url, args.no_tool_results) for q in questions]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                traces.append(result)
                logger.info(f"Collected {len(traces)}/{len(questions)} traces")

        save_output(traces)

    await runner.close()


if __name__ == "__main__":
    asyncio.run(main())
