"""
Parallel Execution Framework for Virtual Filesystem Modules.

Provides:
- Batch pre-fetching for efficient data loading
- Parallel task execution with subagents
- Multiple reduce strategies for aggregating results
"""

import asyncio
import json
import logging
import random
import re
import time
import uuid
from datetime import datetime
from typing import Any

from gxl_inference_client.agent import Agent
from shared.core.environment import get_inference_url

logger = logging.getLogger(__name__)


def _generate_id(prefix: str = "r") -> str:
    """Generate a short unique ID."""
    import hashlib

    unique = f"{time.time()}{random.random()}"
    return f"{prefix}_{hashlib.md5(unique.encode()).hexdigest()[:8]}"


async def _gather_with_straggler_timeout(
    coros: list,
    total: int,
    logger,
    quorum_pct: float = 0.75,
    straggler_multiplier: float = 3.0,
    min_straggler_timeout: float = 10.0,
    absolute_max: float = 120.0,
) -> list:
    """Run coroutines concurrently, applying adaptive timeouts to stragglers.

    Once `quorum_pct` fraction of tasks complete, a deadline is set for the
    rest based on observed durations:

        deadline = now + max(median_duration * straggler_multiplier, min_straggler_timeout)

    Tasks still running after the deadline are cancelled and recorded as
    timeouts. This prevents a single slow task from blocking the entire batch.

    Returns a list in the same order as `coros`, with timeout entries as:
        {"status": "timeout", "error": "Timed out ..."}
    """
    tasks_map: dict[asyncio.Task, int] = {}
    for i, coro in enumerate(coros):
        task = asyncio.ensure_future(coro)
        tasks_map[task] = i

    results: dict[int, object] = {}
    durations: list[float] = []
    start = time.perf_counter()
    pending = set(tasks_map.keys())
    quorum_needed = max(1, int(total * quorum_pct))
    deadline: float | None = None

    while pending:
        # If deadline is set, compute remaining wait
        if deadline is not None:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            timeout = remaining
        else:
            timeout = absolute_max

        done, pending = await asyncio.wait(
            pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            idx = tasks_map[task]
            elapsed = time.perf_counter() - start
            durations.append(elapsed)
            try:
                results[idx] = task.result()
            except Exception as e:
                results[idx] = e

        # Check if quorum reached — set deadline for stragglers
        if deadline is None and len(durations) >= quorum_needed:
            durations_sorted = sorted(durations)
            median = durations_sorted[len(durations_sorted) // 2]
            straggler_budget = max(
                median * straggler_multiplier,
                min_straggler_timeout,
            )
            # Deadline from start, capped by absolute_max
            deadline = start + min(straggler_budget, absolute_max)
            remaining_count = total - len(durations)
            logger.info(
                f"[PARALLEL] Quorum reached ({len(durations)}/{total}). "
                f"Median: {median:.1f}s. "
                f"Straggler deadline: +{max(0, deadline - time.perf_counter()):.1f}s "
                f"for {remaining_count} remaining tasks."
            )

        if not pending:
            break

    # Cancel any remaining tasks
    timed_out_count = 0
    for task in pending:
        task.cancel()
        idx = tasks_map[task]
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        results[idx] = {
            "status": "timeout",
            "error": f"Timed out after {elapsed_ms}ms (straggler cutoff)",
            "time_ms": elapsed_ms,
            "path": "",
            "query": "",
            "output": None,
        }
        timed_out_count += 1

    if timed_out_count:
        logger.warning(f"[PARALLEL] Cancelled {timed_out_count} straggler tasks")

    # Suppress CancelledError from cancelled tasks
    for task in pending:
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    # Return in original order
    return [results.get(i) for i in range(total)]


class ParallelExecutor:
    """Execute tasks in parallel using subagents with batch data pre-fetching.

    Usage:
        executor = ParallelExecutor(
            document_store=my_store,
            agent_config="my_agent_config",
            session_manager=session_mgr,
        )
        results = await executor.execute(
            tasks=[{"path": "/docs/abc/", "query": "What is the main finding?"}],
        max_concurrent=10,
        )
    """

    def __init__(
        self,
        document_store,
        agent_config: str,
        session_manager=None,
        max_content_chars: int = 1000000,
    ):
        self.document_store = document_store
        self.agent_config = agent_config
        self.session_manager = session_manager
        self.max_content_chars = max_content_chars

    async def _fetch_agent_rollout(
        self, base_url: str, session_id: str, agent_id: str
    ) -> list[dict]:
        """Fetch conversation history for a specific subagent.

        Returns a simplified rollout showing tool calls and key events.
        Fetches all message pages so debugging views see the full subagent trace.
        """
        import httpx

        try:
            url = f"{base_url}/api/sessions/{session_id}/agents/{agent_id}/messages"
            logger.info(f"[ROLLOUT] Fetching rollout from: {url}")

            async with httpx.AsyncClient(timeout=10.0) as client:
                page_size = 100
                before: str | None = None
                message_pages: list[list[dict]] = []

                while True:
                    params: dict[str, str | int] = {"group_limit": page_size}
                    if before:
                        params["before"] = before

                    response = await client.get(url, params=params)
                    if response.status_code != 200:
                        logger.warning(
                            f"[ROLLOUT] Failed to fetch rollout for {agent_id}: {response.status_code} - {response.text[:200]}"
                        )
                        if not message_pages:
                            return []
                        break

                    data = response.json()
                    page_messages = (
                        data.get("messages", []) if isinstance(data, dict) else data
                    )
                    message_pages.append(page_messages)

                    if not isinstance(data, dict) or not data.get("has_more"):
                        break

                    before = data.get("next_before")
                    if not before:
                        break

                # The API pages backwards from the newest groups, so reverse the
                # page list to rebuild the full rollout in chronological order.
                messages = [
                    message
                    for page_messages in reversed(message_pages)
                    for message in page_messages
                ]
                logger.info(
                    f"[ROLLOUT] Got {len(messages)} messages for {agent_id} "
                    f"across {len(message_pages)} page(s)"
                )

                # Simplify rollout for display: extract tool calls and key content
                rollout = []
                for msg in messages:
                    role = msg.get("role", "")
                    # API returns "message_content", not "content"
                    content = msg.get("message_content", "") or msg.get("content", "")
                    tool_calls = msg.get("tool_calls") or []
                    tool_call_id = msg.get("tool_call_id")

                    if role == "user":
                        # Skip the initial long prompt with full document
                        if len(content) > 1000:
                            rollout.append(
                                {
                                    "type": "query",
                                    "content": "[Document context provided]",
                                }
                            )
                        else:
                            rollout.append(
                                {
                                    "type": "query",
                                    "content": content[:500],
                                }
                            )
                    elif role == "assistant":
                        if tool_calls:
                            for tc in tool_calls:
                                func = tc.get("function", {})
                                rollout.append(
                                    {
                                        "type": "tool_call",
                                        "tool": func.get("name", "unknown"),
                                        "args": func.get("arguments", "{}"),
                                    }
                                )
                        elif content:
                            # Final response
                            rollout.append(
                                {
                                    "type": "response",
                                    "content": content[:500]
                                    + ("..." if len(content) > 500 else ""),
                                }
                            )
                    elif role == "tool":
                        # Tool result
                        rollout.append(
                            {
                                "type": "tool_result",
                                "tool_call_id": tool_call_id,
                                "result": content[:300]
                                + ("..." if len(content) > 300 else ""),
                            }
                        )

                logger.info(
                    f"[ROLLOUT] Processed rollout for {agent_id}: {len(rollout)} steps, {sum(1 for s in rollout if s.get('type') == 'tool_call')} tool calls"
                )
                return rollout

        except Exception as e:
            logger.warning(f"[ROLLOUT] Error fetching rollout for {agent_id}: {e}")
            return []

    def _write_progress(
        self,
        session_id: str,
        total: int,
        completed: int,
        failed: int,
        started_at: float | None = None,
        started_count: int | None = None,
    ):
        """Write map progress to a JSON file for frontend polling."""
        try:
            import os

            base = os.getenv("LOCAL_SESSION_STORAGE_ROOT", "/workspaces/gxl/sessions")
            progress_dir = os.path.join(base, session_id)
            os.makedirs(progress_dir, exist_ok=True)
            progress_file = os.path.join(progress_dir, "map_progress.json")
            in_progress = (
                (started_count if started_count is not None else total)
                - completed
                - failed
            )
            progress = {
                "total": total,
                "completed": completed,
                "failed": failed,
                "in_progress": max(0, in_progress),
                "started_at": started_at,
            }
            with open(progress_file, "w") as f:
                json.dump(progress, f)
        except Exception as e:
            logger.warning(f"[PARALLEL] Failed to write progress: {e}")

    async def _bulk_write_traces(
        self,
        base_url: str,
        session_id: str,
        traces: list[list[dict]],
        http_client=None,
    ):
        """Bulk-write all deferred subagent traces to the DB via the inference API.

        Flattens all per-subagent traces into a single batch INSERT,
        avoiding the N*M per-round DB writes that would otherwise occur.
        """
        import httpx

        flat_messages = []
        for trace in traces:
            for msg in trace:
                flat_messages.append(
                    {
                        "message_id": msg.get("message_id"),
                        "session_id": session_id,
                        "role": msg.get("role"),
                        "content": msg.get("content", ""),
                        "tool_calls": msg.get("tool_calls"),
                        "tool_call_id": msg.get("tool_call_id"),
                        "agent_id": msg.get("agent_id"),
                        "model": msg.get("model"),
                    }
                )

        if not flat_messages:
            return

        logger.info(
            f"[PARALLEL] Bulk-writing {len(flat_messages)} deferred trace messages "
            f"from {len(traces)} subagents"
        )

        owns_client = http_client is None
        client = http_client or httpx.AsyncClient(timeout=30.0)
        try:
            resp = await client.post(
                f"{base_url}/api/sessions/{session_id}/messages/bulk",
                json={"messages": flat_messages},
                timeout=30.0,
            )
            if resp.status_code in (200, 201):
                logger.info(
                    f"[PARALLEL] Bulk write succeeded: {len(flat_messages)} messages"
                )
            else:
                logger.warning(
                    f"[PARALLEL] Bulk write returned {resp.status_code}: {resp.text[:200]}"
                )
        except Exception as e:
            logger.error(f"[PARALLEL] Bulk trace write failed: {e}")
        finally:
            if owns_client:
                await client.aclose()

    async def _bulk_update_agent_prompts(
        self,
        base_url: str,
        session_id: str,
        updates: list[tuple[str, str]],
        http_client=None,
    ):
        """Update agent rows with the full system prompt (including injected doc content).

        The initial pre-registration only saves the base prompt from the YAML config.
        This writes back the runtime prompt that was actually sent to the LLM.
        """
        import httpx

        BATCH_SIZE = 20
        updated = 0
        owns_client = http_client is None
        client = http_client or httpx.AsyncClient(timeout=30.0)
        try:
            for i in range(0, len(updates), BATCH_SIZE):
                batch = updates[i : i + BATCH_SIZE]
                coros = [
                    client.post(
                        f"{base_url}/api/sessions/{session_id}/agents",
                        json={
                            "agent_id": agent_id,
                            "system_prompt": system_prompt,
                        },
                        timeout=10.0,
                    )
                    for agent_id, system_prompt in batch
                ]
                results = await asyncio.gather(*coros, return_exceptions=True)
                updated += sum(
                    1
                    for r in results
                    if not isinstance(r, Exception)
                    and getattr(r, "status_code", 500) in (200, 201, 409)
                )
            logger.info(
                f"[PARALLEL] Updated system prompts for {updated}/{len(updates)} agents"
            )
        except Exception as e:
            logger.error(f"[PARALLEL] Bulk agent prompt update failed: {e}")
        finally:
            if owns_client:
                await client.aclose()

    def _clear_progress(self, session_id: str):
        """Remove the progress file after map completes."""
        try:
            import os

            base = os.getenv("LOCAL_SESSION_STORAGE_ROOT", "/workspaces/gxl/sessions")
            progress_file = os.path.join(base, session_id, "map_progress.json")
            if os.path.exists(progress_file):
                os.remove(progress_file)
        except Exception as e:
            logger.warning(f"[PARALLEL] Failed to clear progress: {e}")

    async def _load_agent_config(self) -> dict:
        """Load and cache the agent config YAML (read once, reuse for all subagents)."""
        if not hasattr(self, "_cached_agent_config"):
            import os
            from pathlib import Path

            import yaml

            GXL_ROOT = Path(os.environ.get("GXL_ROOT", "/workspaces/gxl"))
            config_path = GXL_ROOT / "agents" / f"{self.agent_config}.yaml"
            if not config_path.exists():
                raise FileNotFoundError(f"Agent config not found at {config_path}")
            with open(config_path) as f:
                self._cached_agent_config = yaml.safe_load(f) or {}
        return self._cached_agent_config

    async def _load_agent_tools(self, base_url: str, config: dict) -> list[dict]:
        """Load and cache tools from MCP servers (fetch once, reuse for all subagents).

        Converts MCP tool format to OpenAI function-calling format so the LLM
        can actually recognise and invoke them.
        """
        if not hasattr(self, "_cached_tools"):
            import httpx

            tools = []
            mcp_servers = config.get("mcp_servers", [])
            async with httpx.AsyncClient(timeout=30.0) as client:
                for server in mcp_servers:
                    server_name = server.get("name", "mcp")
                    server_url = server.get("url", "")
                    included = server.get("included_tools", [])
                    excluded = server.get("excluded_tools", [])
                    if not server_url:
                        continue
                    try:
                        url = (
                            server_url
                            if server_url.endswith("/tools/list")
                            else f"{server_url}/tools/list"
                        )
                        resp = await client.get(url)
                        if resp.status_code == 200:
                            data = resp.json()
                            # JSON-RPC wraps in {"result": {"tools": [...]}}
                            if "result" in data and isinstance(data["result"], dict):
                                server_tools = data["result"].get("tools", [])
                            else:
                                server_tools = data.get("tools", [])
                            for tool in server_tools:
                                name = tool.get("name", "")
                                if included and name not in included:
                                    continue
                                if excluded and name in excluded:
                                    continue

                                if "function" in tool:
                                    tool_copy = tool.copy()
                                    tool_copy["function"] = tool["function"].copy()
                                    tool_copy["function"][
                                        "name"
                                    ] = f"{server_name}__{tool['function']['name']}"
                                    tools.append(tool_copy)
                                else:
                                    tools.append(
                                        {
                                            "type": "function",
                                            "function": {
                                                "name": f"{server_name}__{name}",
                                                "description": tool.get(
                                                    "description", ""
                                                ),
                                                "parameters": tool.get(
                                                    "inputSchema", {}
                                                ),
                                            },
                                        }
                                    )
                    except Exception as e:
                        logger.warning(
                            f"[PARALLEL] Failed to fetch tools from {server_url}: {e}"
                        )
            self._cached_tools = tools
            logger.info(
                f"[PARALLEL] Cached {len(tools)} tools from {len(mcp_servers)} MCP servers"
            )
        return self._cached_tools

    async def execute(
        self,
        tasks: list[dict],
        max_concurrent: int = 25,
        batch_size: int = 20,
        output_schema: dict = None,
        include_rollouts: bool = False,
        session_id: str = "default",
        parent_agent_id: str = None,
        extract_document_id: callable = None,
    ) -> list[dict]:
        """Execute tasks in parallel using semaphore-based concurrency.

        All tasks are launched immediately but a semaphore limits how many
        run at the same time.  As each task finishes, the next one starts
        without waiting for an entire sub-batch to drain.

        Args:
            tasks: List of {path, query} dicts
            max_concurrent: Max parallel subagents (semaphore width)
            batch_size: Ignored (kept for backward compat)
            output_schema: Optional JSON schema for structured output
            include_rollouts: Whether to fetch full per-paper subagent traces
            session_id: Session ID for subagents
            parent_agent_id: Parent agent ID for tracing
            extract_document_id: Function to extract document_id from path

        Returns:
            List of result dicts
        """
        if not tasks:
            return []

        total = len(tasks)
        map_started_at = time.time()

        # Write initial progress (0 started — pre-fetch phase)
        self._write_progress(
            session_id, total, 0, 0, started_at=map_started_at, started_count=0
        )

        try:
            # --- One-time setup (fixes #3 & #4: global pre-fetch + cached config) ---

            # Ensure session exists in inference engine
            actual_session_id = session_id
            if self.session_manager:
                try:
                    actual_session_id = (
                        await self.session_manager.auto_initialize_sandbox_for_session(
                            session_id
                        )
                    )
                    logger.info(f"Initialized sandbox for session: {actual_session_id}")
                except Exception as e:
                    logger.warning(f"Could not initialize session sandbox: {e}")

            base_url = get_inference_url()

            # Create session in inference engine if it doesn't exist
            try:
                import httpx

                async with httpx.AsyncClient() as client:
                    check_resp = await client.get(
                        f"{base_url}/api/sessions/{actual_session_id}",
                        timeout=5.0,
                    )
                    if check_resp.status_code == 404:
                        logger.info(
                            f"Creating session in inference engine: {actual_session_id}"
                        )
                        create_resp = await client.post(
                            f"{base_url}/api/sessions",
                            json={
                                "session_id": actual_session_id,
                                "user_id": parent_agent_id or "mcp_parallel_user",
                                "title": f"Parallel Session - {actual_session_id}",
                                "model": "anthropic/claude-sonnet-4",
                                "system_prompt": "You are a research assistant analyzing scientific papers.",
                            },
                            timeout=10.0,
                        )
                        if create_resp.status_code not in (200, 201):
                            logger.warning(
                                f"Failed to create session: {create_resp.status_code}"
                            )
                        else:
                            logger.info(
                                f"Created session in inference engine: {actual_session_id}"
                            )
                    else:
                        logger.info(
                            f"Session exists in inference engine: {actual_session_id}"
                        )
            except Exception as e:
                logger.warning(f"Could not ensure session exists: {e}")

            # Determine if config uses plain content (no line numbers/block IDs)
            plain_content_mode = "full_content" in self.agent_config

            # Pre-fetch: full content when injecting into prompt, metadata-only
            # when subagents read content via tools.
            prefetch_start = time.perf_counter()
            doc_ids = []
            if extract_document_id:
                for task in tasks:
                    doc_id = extract_document_id(task.get("path", ""))
                    if doc_id:
                        doc_ids.append(doc_id)

            documents_data = {}
            if doc_ids:
                if plain_content_mode:
                    documents_data = await self.document_store.batch_get_documents(
                        doc_ids
                    )
                else:
                    documents_data = await self.document_store.batch_get_metadata(
                        doc_ids
                    )

            prefetch_time = (time.perf_counter() - prefetch_start) * 1000
            logger.info(
                f"[PARALLEL] Pre-fetched {'content' if plain_content_mode else 'metadata'} "
                f"for {len(documents_data)} documents in {prefetch_time:.0f}ms"
            )

            # Fix #4: Cache agent config + tools (read once, reuse for all subagents)
            agent_config_data = await self._load_agent_config()
            cached_tools = await self._load_agent_tools(base_url, agent_config_data)
            logger.info(
                f"[PARALLEL] Cached agent config ({self.agent_config}) "
                f"and {len(cached_tools)} tools"
            )

            # Build paper text helper
            def build_document_text(doc_data: dict) -> tuple[str, bool]:
                """Build document text from pre-fetched blocks."""
                parts = []
                current_section = None
                total_chars = 0
                truncated = False
                is_youtube = (
                    doc_data.get("metadata", {}).get("source_type") == "youtube"
                )

                for block in doc_data.get("blocks", []):
                    section = block.get("section")
                    if section and section != current_section:
                        current_section = section
                        header = f"\n\n## {section}\n\n"
                        if total_chars + len(header) < self.max_content_chars:
                            parts.append(header)
                            total_chars += len(header)

                    content = block.get("content", "")

                    if plain_content_mode:
                        line_num = block.get("line_number", len(parts) + 1)
                        line_text = f"L{line_num}: {content}\n"
                    else:
                        line_num = block.get("line_number", len(parts))
                        raw_block_id = block.get("block_id")

                        encoded_bid = None
                        if raw_block_id is not None:
                            try:
                                from modules.virtual_filesystem.block_id_codec import (
                                    encode_block_id,
                                )

                                encoded_bid = encode_block_id(int(raw_block_id))
                            except (ValueError, TypeError):
                                encoded_bid = str(raw_block_id)

                        start_sec = block.get("start_sec")
                        if is_youtube and start_sec is not None:
                            hours = int(start_sec // 3600)
                            minutes = int((start_sec % 3600) // 60)
                            seconds = int(start_sec % 60)
                            ts_str = (
                                f"{hours}:{minutes:02d}:{seconds:02d}"
                                if hours > 0
                                else f"{minutes}:{seconds:02d}"
                            )
                            line_text = (
                                f"L{line_num} [{encoded_bid}] [{ts_str}] (timestamp_sec={start_sec}) {content}\n"
                                if encoded_bid
                                else f"[LINE {line_num}] [{ts_str}] (timestamp_sec={start_sec}) {content}\n"
                            )
                        else:
                            line_text = (
                                f"L{line_num} [{encoded_bid}]: {content}\n"
                                if encoded_bid
                                else f"[LINE {line_num}] {content}\n"
                            )

                    if total_chars + len(line_text) < self.max_content_chars:
                        parts.append(line_text)
                        total_chars += len(line_text)
                    else:
                        truncated = True
                        break

                return "".join(parts), truncated

            # --- Task execution with global scheduler ---

            from shared.core.global_scheduler import get_global_scheduler

            scheduler = await get_global_scheduler()

            exec_start_time = time.perf_counter()
            subagent_ids = [f"doc_explorer_{uuid.uuid4().hex[:8]}" for _ in tasks]
            all_deferred_traces: list[list[dict]] = []
            agent_prompt_updates: list[tuple[str, str]] = []
            traces_lock = asyncio.Lock()

            # Shared HTTP client for all tasks (connection reuse, no per-task overhead)
            import httpx

            shared_limits = httpx.Limits(
                max_connections=max(max_concurrent * 2, 200),
                max_keepalive_connections=max(max_concurrent, 100),
            )
            shared_client = httpx.AsyncClient(timeout=180.0, limits=shared_limits)

            # Derive user_id for scheduler fairness from parent_agent_id or session
            scheduler_user_id = parent_agent_id or session_id

            # Pre-register all agents in a single batch via the shared client
            reg_payloads = []
            for subagent_id in subagent_ids:
                p = {
                    "agent_id": subagent_id,
                    "system_prompt": agent_config_data.get("system_prompt", ""),
                    "model": agent_config_data.get(
                        "model", "google/gemini-3-flash-preview"
                    ),
                }
                if parent_agent_id:
                    p["parent_agent_id"] = parent_agent_id
                reg_payloads.append(p)

            reg_start = time.perf_counter()
            REG_BATCH_SIZE = 20
            reg_results = []
            for i in range(0, len(reg_payloads), REG_BATCH_SIZE):
                batch = reg_payloads[i : i + REG_BATCH_SIZE]
                batch_coros = [
                    shared_client.post(
                        f"{base_url}/api/sessions/{actual_session_id}/agents",
                        json=p,
                        timeout=10.0,
                    )
                    for p in batch
                ]
                reg_results.extend(
                    await asyncio.gather(*batch_coros, return_exceptions=True)
                )
            reg_ok = sum(
                1
                for r in reg_results
                if not isinstance(r, Exception)
                and getattr(r, "status_code", 500) in (200, 201, 409)
            )
            logger.info(
                f"[PARALLEL] Batch-registered {reg_ok}/{len(reg_payloads)} agents "
                f"in {(time.perf_counter() - reg_start) * 1000:.0f}ms"
            )

            async def execute_single_task(
                task: dict,
                subagent_id: str,
                task_idx: int,
            ) -> dict:
                """Execute a single task with global scheduler + deferred persistence."""
                path = task.get("path", "")
                query = task.get("query", "")
                doc_id = extract_document_id(path) if extract_document_id else None

                # Estimate tokens for scheduler (rough: 4 chars ≈ 1 token + output buffer)
                doc_data = documents_data.get(doc_id, {}) if doc_id else {}
                doc_text, truncated = ("", False)
                if doc_data:
                    doc_text, truncated = build_document_text(doc_data)
                estimated_tokens = len(doc_text) // 4 + 2000

                async with scheduler.acquire(scheduler_user_id, estimated_tokens):
                    task_start = time.perf_counter()
                    submit_offset = (task_start - exec_start_time) * 1000

                    logger.info(
                        f"[PARALLEL] Task {task_idx} [{subagent_id}] STARTED at +{submit_offset:.0f}ms"
                    )

                    try:
                        if not doc_data:
                            return {
                                "path": path,
                                "query": query,
                                "output": None,
                                "status": "error",
                                "error": f"Document not found: {doc_id}",
                                "time_ms": round(
                                    (time.perf_counter() - task_start) * 1000
                                ),
                            }

                        metadata = doc_data.get("metadata", {})
                        title = metadata.get("title", "")
                        # Use shortened document_id from metadata (bio_/med_/PMC)
                        if metadata.get("document_id"):
                            doc_id = metadata["document_id"]

                        schema_instructions = ""
                        if output_schema:
                            schema_keys = list(output_schema.keys())
                            if plain_content_mode:
                                line_format_desc = "L<number>: content"
                                line_example = "L87: Our cohort consisted of 5,000 patients..."
                                line_extract_hint = "Extract JUST the number after L, e.g. L87 → 87"
                            else:
                                line_format_desc = "[LINE X] content"
                                line_example = "[LINE 87] Our cohort consisted of 5,000 patients..."
                                line_extract_hint = "Extract JUST the number from [LINE X], don't include the brackets"
                            schema_instructions = f"""

## REQUIRED OUTPUT FORMAT

You MUST return a JSON object with EXACTLY these keys: {schema_keys}
PLUS a "_citations" array.

Schema:
{json.dumps(output_schema, indent=2)}

## CRITICAL: EVERY VALUE NEEDS A CITATION

Each value you provide MUST be backed up by a citation from the document.
- Numbers like "5 samples" → cite the exact line where this appears
- Specific claims → cite the source text
- If you cannot find supporting text, use "Not found" as the value

Rules:
- Use EXACTLY the key names shown above (case-sensitive)
- If information is not found, use "Not found" as the value
- Do NOT guess or infer values without textual evidence
- Return ONLY the JSON object, no other text

## _citations Array Format

For EACH field, add a citation entry with:
- "field": the key name this citation supports
- "line": the line number (shown as {line_format_desc} at the start of each line in the document)
- "content": the exact text that supports your answer (copy verbatim, 10-50 words)

The document shows lines like: {line_example}

Example output:
{{
  "sample_size": "5,000 patients",
  "method": "transformer-based classifier",
  "_citations": [
    {{"field": "sample_size", "line": 87, "content": "Our cohort consisted of 5,000 patients from three hospitals"}},
    {{"field": "method", "line": 142, "content": "We employed a transformer-based classifier with 12 attention heads"}}
  ]
}}

IMPORTANT: 
- {line_extract_hint}
- If a field has "Not found", do NOT include a citation for it
- Do NOT embed citations in the values - use ONLY the _citations array
"""

                        source_type = metadata.get("source_type", "")
                        doc_header = f"# Document: {title}\n"
                        if source_type == "youtube":
                            video_id = metadata.get("video_id") or metadata.get(
                                "identifier", ""
                            )
                            doc_header += f"# Source Type: YouTube Transcript\n"
                            doc_header += f"# Video ID: {video_id}\n"
                            doc_header += f'# IMPORTANT: Use video_id "{video_id}" (not the document UUID) in all citations.\n'
                            doc_header += f"# Timestamps are shown as (timestamp_sec=X) on each line — include these in citations.\n"
                        elif source_type:
                            doc_header += f"# Source Type: {source_type}\n"
                            doc_header += f"# Document ID: {doc_id}\n"

                        base_system_prompt = agent_config_data.get("system_prompt", "")
                        model_name = agent_config_data.get("model", "")

                        if model_name.startswith("local/"):
                            # Self-hosted wrappers (e.g. qwen-rl at :8200) run their own
                            # paperclip tool loop. Send only the DOI + query — the wrapper
                            # fetches whatever content it needs.
                            # Format MUST match the SFT/RL training distribution exactly:
                            # "According to paper DOI: <doi>, <query>" on a single line.
                            system_prompt_with_doc = base_system_prompt
                            doi = metadata.get("doi") or doc_id
                            agentic_query = (
                                f"According to paper DOI: {doi}, {query}"
                                f"{schema_instructions}"
                            )
                        elif plain_content_mode:
                            system_prompt_with_doc = (
                                f"{base_system_prompt}\n\n"
                                f"---\n"
                                f"# {title}\n\n"
                                f"{doc_text[:self.max_content_chars]}\n"
                                f"---"
                            )
                            agentic_query = f"{query}\n{schema_instructions}"
                        else:
                            system_prompt_with_doc = (
                                f"{base_system_prompt}\n\n"
                                f"---\n"
                                f"{doc_header}"
                                f"The following is the output of `cat content.lines` "
                                f"for the paper you are analyzing:\n\n"
                                f"{doc_text[:self.max_content_chars]}\n"
                                f"---"
                            )
                            agentic_query = f"{query}\n{schema_instructions}"

                        agent = Agent(
                            model=agent_config_data.get(
                                "model", "google/gemini-3-flash-preview"
                            ),
                            system_prompt=system_prompt_with_doc,
                            agent_id=subagent_id,
                            session_id=actual_session_id,
                            base_url=base_url,
                            timeout=120.0,
                            max_iterations=1,
                            parent_agent_id=parent_agent_id,
                        )
                        # No tools — content is in the system prompt.
                        agent.tool_list = []
                        if doc_id:
                            agent.paper_document_id = doc_id

                        response = await agent.call_async(
                            agentic_query,
                            defer_persistence=True,
                            http_client=shared_client,
                        )

                        # Collect deferred trace for bulk write later
                        if (
                            hasattr(agent, "_last_deferred_trace")
                            and agent._last_deferred_trace
                        ):
                            async with traces_lock:
                                all_deferred_traces.append(agent._last_deferred_trace)
                                agent_prompt_updates.append(
                                    (subagent_id, system_prompt_with_doc)
                                )

                        # Build rollout from deferred trace instead of extra HTTP call
                        rollout = []
                        if hasattr(agent, "_last_deferred_trace"):
                            for msg in agent._last_deferred_trace:
                                role = msg.get("role", "")
                                content = msg.get("content", "")
                                tool_calls_data = msg.get("tool_calls", [])
                                if role == "assistant" and tool_calls_data:
                                    for tc in tool_calls_data:
                                        func = tc.get("function", {})
                                        rollout.append(
                                            {
                                                "type": "tool_call",
                                                "tool": func.get("name", "unknown"),
                                                "args": func.get("arguments", "{}"),
                                            }
                                        )
                                elif role == "assistant" and content:
                                    rollout.append(
                                        {
                                            "type": "response",
                                            "content": content[:500]
                                            + ("..." if len(content) > 500 else ""),
                                        }
                                    )

                        structured_output = None
                        if output_schema:
                            import re

                            cleaned = response.strip()

                            # Strip markdown code fences
                            if cleaned.startswith("```"):
                                first_nl = cleaned.find("\n")
                                if first_nl != -1:
                                    cleaned = cleaned[first_nl + 1 :]
                                if cleaned.endswith("```"):
                                    cleaned = cleaned[:-3]
                                cleaned = cleaned.strip()

                            # Remove all {{...}} double-brace citation markers before
                            # attempting JSON parsing — they are never valid JSON and
                            # any variant (block_id, document_id, image, etc.) breaks
                            # json.loads.
                            cleaned = re.sub(r"\{\{[^}]*\}\}", "", cleaned)

                            # Try parsing directly
                            try:
                                parsed = json.loads(cleaned)
                                if isinstance(parsed, dict):
                                    structured_output = parsed
                            except (json.JSONDecodeError, ValueError):
                                pass

                            # Fallback: extract all top-level JSON objects,
                            # keep the last one (reader subagents often produce
                            # a preliminary JSON from the abstract, then a
                            # complete one after reading the full paper).
                            if structured_output is None:
                                candidates = []
                                scan = 0
                                while scan < len(cleaned):
                                    start = cleaned.find("{", scan)
                                    if start == -1:
                                        break
                                    depth = 0
                                    in_str = False
                                    esc = False
                                    found_end = False
                                    for idx in range(start, len(cleaned)):
                                        c = cleaned[idx]
                                        if esc:
                                            esc = False
                                            continue
                                        if c == "\\":
                                            esc = True
                                            continue
                                        if c == '"' and not esc:
                                            in_str = not in_str
                                            continue
                                        if in_str:
                                            continue
                                        if c == "{":
                                            depth += 1
                                        elif c == "}":
                                            depth -= 1
                                            if depth == 0:
                                                try:
                                                    obj = json.loads(
                                                        cleaned[start : idx + 1]
                                                    )
                                                    if isinstance(obj, dict):
                                                        candidates.append(obj)
                                                except (
                                                    json.JSONDecodeError,
                                                    ValueError,
                                                ):
                                                    pass
                                                scan = idx + 1
                                                found_end = True
                                                break
                                    if not found_end:
                                        break
                                if candidates:
                                    structured_output = candidates[-1]

                        task_end = time.perf_counter()
                        end_offset = (task_end - exec_start_time) * 1000
                        duration = (task_end - task_start) * 1000

                        logger.info(
                            f"[PARALLEL] Task {task_idx} [{subagent_id}] COMPLETED at +{end_offset:.0f}ms (took {duration:.0f}ms) | Overhead: start_delay={submit_offset:.0f}ms"
                        )

                        result = {
                            "path": path,
                            "query": query,
                            "response": response,
                            "title": title,
                            "document_id": doc_id,
                            "status": "success",
                            "time_ms": round(duration),
                        }
                        if include_rollouts:
                            result["rollout"] = rollout

                        if structured_output is not None:
                            result["output"] = json.dumps(structured_output)
                            result["output_structured"] = True
                        else:
                            result["output"] = response
                            result["output_structured"] = False

                        return result

                    except Exception as e:
                        task_end = time.perf_counter()
                        logger.error(
                            f"[PARALLEL] Task {task_idx} [{subagent_id}] ERROR: {e}"
                        )

                        return {
                            "path": path,
                            "query": query,
                            "output": None,
                            "status": "error",
                            "error": str(e),
                            "time_ms": round((task_end - task_start) * 1000),
                        }

            # Progress tracking
            progress_lock = asyncio.Lock()
            completed_count = 0
            failed_count = 0

            async def tracked_task(coro, task_idx):
                nonlocal completed_count, failed_count
                try:
                    result = await coro
                    is_error = (
                        isinstance(result, dict) and result.get("status") == "error"
                    )
                    async with progress_lock:
                        if is_error:
                            failed_count += 1
                        else:
                            completed_count += 1
                        self._write_progress(
                            session_id,
                            total,
                            completed_count + failed_count,
                            failed_count,
                            started_at=map_started_at,
                            started_count=total,
                        )
                    return result
                except Exception:
                    async with progress_lock:
                        failed_count += 1
                        self._write_progress(
                            session_id,
                            total,
                            completed_count + failed_count,
                            failed_count,
                            started_at=map_started_at,
                            started_count=total,
                        )
                    raise

            logger.info(
                f"[PARALLEL] ===== Launching {total} tasks via global scheduler ====="
            )

            # Update progress to show all tasks are now running
            self._write_progress(
                session_id, total, 0, 0, started_at=map_started_at, started_count=total
            )

            coros = [
                tracked_task(
                    execute_single_task(task, sid, i),
                    i,
                )
                for i, (task, sid) in enumerate(zip(tasks, subagent_ids))
            ]

            gather_start = time.perf_counter()
            try:
                raw_results = await _gather_with_straggler_timeout(coros, total, logger)

                # Bulk-write all deferred traces to DB in one transaction
                if all_deferred_traces:
                    await self._bulk_write_traces(
                        base_url, actual_session_id, all_deferred_traces,
                        http_client=shared_client,
                    )

                # Persist the full system prompts (with injected doc content)
                if agent_prompt_updates:
                    await self._bulk_update_agent_prompts(
                        base_url, actual_session_id, agent_prompt_updates,
                        http_client=shared_client,
                    )
            finally:
                await shared_client.aclose()

            gather_duration = (time.perf_counter() - gather_start) * 1000
            timed_out = sum(
                1
                for r in raw_results
                if isinstance(r, dict) and r.get("status") == "timeout"
            )
            logger.info(
                f"[PARALLEL] All {total} tasks finished in {gather_duration:.0f}ms"
                + (f" ({timed_out} timed out)" if timed_out else "")
            )

            sched_stats = scheduler.stats()
            logger.info(
                f"[PARALLEL] Scheduler stats: {sched_stats['global_in_flight']} in-flight, "
                f"RPM used: {sched_stats['rpm_used']}, TPM used: {sched_stats['tpm_used']}"
            )

            results = []
            for i, result in enumerate(raw_results):
                if isinstance(result, Exception):
                    results.append(
                        {
                            "path": tasks[i].get("path", ""),
                            "query": tasks[i].get("query", ""),
                            "output": None,
                            "status": "error",
                            "error": str(result),
                        }
                    )
                elif isinstance(result, dict) and result.get("status") == "timeout":
                    result["path"] = result.get("path") or tasks[i].get("path", "")
                    result["query"] = result.get("query") or tasks[i].get("query", "")
                    results.append(result)
                else:
                    results.append(result)

            return results

        finally:
            self._clear_progress(session_id)


class ReduceStrategies:
    """Common reduce strategies for aggregating parallel results."""

    @staticmethod
    def concat(results: list[dict]) -> str:
        """Concatenate all outputs with separators."""
        parts = []
        for r in results:
            path = r.get("path", "unknown")
            output = r.get("output", "")
            parts.append(f"[{path}]\n{output}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def extract_fields(results: list[dict], fields: list[str]) -> list[dict]:
        """Extract specific fields from structured outputs."""
        extracted = []
        for r in results:
            try:
                output = r.get("output", "{}")
                if isinstance(output, str):
                    output = json.loads(output)

                item = {"path": r.get("path")}
                for field in fields:
                    item[field] = output.get(field)
                extracted.append(item)
            except (json.JSONDecodeError, AttributeError):
                pass
        return extracted

    @staticmethod
    def to_table(
        results: list[dict],
        columns: list[str],
    ) -> dict:
        """Convert results to table format."""
        rows = []
        for r in results:
            try:
                output = r.get("output", "{}")
                if isinstance(output, str):
                    output = json.loads(output)

                row = {}
                for col in columns:
                    val = output.get(col, "")
                    if isinstance(val, list):
                        val = "; ".join(str(v) for v in val[:3])
                    row[col] = str(val)[:200] if val else ""

                row["_title"] = r.get("title", "")[:50]
                row["_document_id"] = r.get("document_id", "")
                rows.append(row)
            except (json.JSONDecodeError, AttributeError):
                pass

        return {
            "columns": ["_title"] + columns,
            "rows": rows,
        }

    @staticmethod
    def aggregate_numeric(
        results: list[dict],
        field: str,
    ) -> dict:
        """Aggregate a numeric field across results."""
        values = []
        for r in results:
            try:
                output = r.get("output", "{}")
                if isinstance(output, str):
                    output = json.loads(output)
                val = output.get(field)
                if val is not None and isinstance(val, (int, float)):
                    values.append(val)
            except (json.JSONDecodeError, AttributeError):
                pass

        if not values:
            return {"error": f"No numeric values found for field: {field}"}

        return {
            "field": field,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "values": values,
        }

    @staticmethod
    def vote(results: list[dict], field: str = None) -> dict:
        """Simple majority voting on outputs or a specific field."""
        votes = {}
        for r in results:
            if r.get("status") != "success":
                continue

            try:
                output = r.get("output", "")
                if field:
                    if isinstance(output, str):
                        output = json.loads(output)
                    val = str(output.get(field, ""))
                else:
                    val = str(output)

                val = val.strip()[:500]  # Truncate for voting
                votes[val] = votes.get(val, 0) + 1
            except (json.JSONDecodeError, AttributeError):
                pass

        if not votes:
            return {"error": "No valid votes"}

        winner = max(votes.items(), key=lambda x: x[1])
        return {
            "winner": winner[0],
            "votes": winner[1],
            "total": sum(votes.values()),
            "distribution": votes,
        }
