"""
Virtual Filesystem Base Classes

Abstract interfaces for building virtual filesystem modules over document collections.
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.types import Tool

from ..base import ToolModule

logger = logging.getLogger(__name__)


@dataclass
class ParsedPath:
    """Result of parsing a virtual filesystem path."""

    type: str  # e.g., "root", "documents_list", "document", "section", "file", "youtube_*"
    document_id: str | None = None
    section: str | None = None
    filename: str | None = None
    filter: str | None = None
    error: str | None = None
    video_id: str | None = None  # For YouTube paths: /youtube/{video_id}/...
    extra: dict = field(default_factory=dict)


class PathParser(ABC):
    """Abstract path parser for virtual filesystem paths.

    Subclasses implement domain-specific path structures:
    - BioMedRxiv: /papers/{uuid}/sections/{name}.lines
    - FDA: /documents/{doc_id}/pages/{page}.lines
    - ClinicalTrials: /trials/{nct_id}/arms/{arm}.lines
    """

    @property
    @abstractmethod
    def root_name(self) -> str:
        """Root path name (e.g., 'papers', 'documents', 'trials')."""
        pass

    @abstractmethod
    def parse(self, path: str) -> ParsedPath:
        """Parse a path string into components."""
        pass

    def normalize(self, path: str) -> str:
        """Normalize a path string."""
        path = path.strip().rstrip("/")
        if not path.startswith("/"):
            path = "/" + path
        return path


class DocumentStore(ABC):
    """Abstract interface for document storage backends.

    Subclasses implement domain-specific storage:
    - PostgreSQL with content_blocks table
    - Elasticsearch indexes
    - File-based storage
    """

    @abstractmethod
    async def get_document(self, document_id: str) -> dict | None:
        """Fetch a single document's metadata."""
        pass

    @abstractmethod
    async def get_document_content(self, document_id: str) -> list[dict]:
        """Fetch all content blocks for a document.

        Returns list of dicts with:
        - line_number: int
        - content: str
        - section: str | None
        - block_type: str | None
        """
        pass

    @abstractmethod
    async def search_documents(
        self,
        query: str = None,
        filters: dict = None,
        limit: int = 20,
    ) -> list[dict]:
        """Search for documents matching criteria."""
        pass

    @abstractmethod
    async def grep_content(
        self,
        regex: str,
        document_ids: list[str] = None,
        section_filter: str = None,
        limit: int = 50,
    ) -> list[dict]:
        """Regex search on document content."""
        pass

    async def batch_get_documents(
        self,
        document_ids: list[str],
    ) -> dict[str, dict]:
        """Batch fetch multiple documents and their content.

        Returns dict mapping document_id to:
        {
            "metadata": {...},
            "blocks": [...]
        }

        Default implementation calls get_document + get_document_content for each.
        Override for optimized batch queries.
        """
        result = {}
        for doc_id in document_ids:
            metadata = await self.get_document(doc_id)
            if metadata:
                blocks = await self.get_document_content(doc_id)
                result[doc_id] = {
                    "metadata": metadata,
                    "blocks": blocks,
                }
        return result

    async def batch_get_metadata(
        self,
        document_ids: list[str],
    ) -> dict[str, dict]:
        """Batch fetch metadata only (no content blocks).

        Returns dict mapping document_id to {"metadata": {...}}.
        Used by agentic map operations where subagents read content via tools.

        Default implementation calls get_document for each.
        Override for optimized batch queries.
        """
        result = {}
        for doc_id in document_ids:
            metadata = await self.get_document(doc_id)
            if metadata:
                result[doc_id] = {"metadata": metadata}
        return result


class VirtualFilesystemModule(ToolModule):
    """Base class for virtual filesystem modules with parallel execution.

    Provides:
    - Standard filesystem-like tools (ls, cat, head, tail, grep, find, stat)
    - Parallel execution framework with subagents
    - Results caching and registry
    - Reduce strategies for aggregating results

    Subclasses must implement:
    - get_name(): Module name
    - get_path_parser(): Return a PathParser instance
    - get_document_store(): Return a DocumentStore instance
    - get_tools(): Define domain-specific tools (can call super and extend)

    Subclasses may override:
    - _batch_cite(): Batch citation lookup (default falls back to individual _get_citation calls)
    """

    def __init__(self):
        super().__init__()
        self._tools = []
        self._handlers = {}
        self._results_cache = {}  # In-memory cache for results
        self._path_parser = None
        self._document_store = None

    @abstractmethod
    def get_path_parser(self) -> PathParser:
        """Return the path parser for this filesystem."""
        pass

    @abstractmethod
    def get_document_store(self) -> DocumentStore:
        """Return the document store backend."""
        pass

    @property
    def path_parser(self) -> PathParser:
        """Lazy-loaded path parser."""
        if self._path_parser is None:
            self._path_parser = self.get_path_parser()
        return self._path_parser

    @property
    def document_store(self) -> DocumentStore:
        """Lazy-loaded document store."""
        if self._document_store is None:
            self._document_store = self.get_document_store()
        return self._document_store

    def get_handlers(self) -> dict[str, Callable]:
        """Return tool handlers."""
        return self._handlers

    # =========================================================================
    # Citation Support
    # =========================================================================

    async def _batch_cite(
        self,
        doc_id: str,
        line_numbers: list[int],
        supplement_filename: str | None = None,
    ) -> dict | None:
        """Batch citation lookup for the terminal's cite command.

        Returns None if the document is not found.
        Otherwise returns::

            {
                "doc_meta": {
                    "doc_title": str,
                    "doi": str,
                    "authors": str,
                    "source": str,
                    "month_year": str,
                },
                "lines": {
                    <line_number (1-indexed)>: {
                        "content": str,
                        "section": str,
                        "block_type": str,
                        "citation_info": dict,   # module-specific extras
                    } | None,   # None means line not found
                    ...
                },
            }

        Override in subclasses for batch-optimized DB queries.
        Default implementation calls ``_get_citation()`` per line.
        """
        if not hasattr(self, "_get_citation"):
            return None

        doc_found = False
        lines_result: dict[int, dict | None] = {}
        doc_meta = {
            "doc_title": "",
            "doi": "",
            "authors": "",
            "source": "",
            "month_year": "",
        }

        for ln in line_numbers:
            result = await self._get_citation(
                document_id=doc_id,
                line_number=ln,
                supplement_filename=supplement_filename,
                session_id="default",
            )
            if "error" in result:
                lines_result[ln] = None
            else:
                doc_found = True
                if not doc_meta["doc_title"]:
                    doc_meta["doc_title"] = result.get("doc_title", "")
                    doc_meta["doi"] = result.get("doi", "")
                    doc_meta["authors"] = result.get("authors", "")
                    doc_meta["source"] = result.get("source", "")
                    doc_meta["month_year"] = result.get("month_year", "")
                ci = {}
                for key in (
                    "source_type",
                    "source_path",
                    "xml_id",
                    "xpath",
                    "page",
                    "bbox",
                    "block_id",
                ):
                    if result.get(key) is not None:
                        ci[key] = result[key]
                lines_result[ln] = {
                    "content": result.get("content_preview", result.get("content", "")),
                    "section": result.get("section", result.get("section_name", "")),
                    "block_type": result.get("block_type", ""),
                    "citation_info": ci,
                }

        if not doc_found and not any(v is not None for v in lines_result.values()):
            return None

        return {"doc_meta": doc_meta, "lines": lines_result}

    # =========================================================================
    # Results Registry (session-based caching)
    # =========================================================================

    def _save_results(
        self,
        results_id: str,
        data: dict,
        session_id: str = "default",
    ) -> str:
        """Save results to session sandbox and in-memory cache."""
        cache_key = f"{session_id}:{results_id}"
        self._results_cache[cache_key] = data

        # Also persist to disk if session manager available
        if self.session_manager:
            try:
                import json

                sandbox_path = self.session_manager.get_sandbox_path(session_id)
                results_dir = sandbox_path / "results"
                results_dir.mkdir(parents=True, exist_ok=True)

                results_file = results_dir / f"{results_id}.json"
                with open(results_file, "w") as f:
                    json.dump(data, f, indent=2, default=str)
            except Exception as e:
                logger.warning(f"Could not persist results to disk: {e}")

        return results_id

    def _load_results(
        self,
        results_id: str,
        session_id: str = "default",
    ) -> dict | None:
        """Load results from cache or session sandbox."""
        cache_key = f"{session_id}:{results_id}"

        # Check in-memory cache first
        if cache_key in self._results_cache:
            return self._results_cache[cache_key]

        # Try loading from disk
        if self.session_manager:
            try:
                import json

                sandbox_path = self.session_manager.get_sandbox_path(session_id)
                results_file = sandbox_path / "results" / f"{results_id}.json"

                if results_file.exists():
                    with open(results_file) as f:
                        data = json.load(f)
                    self._results_cache[cache_key] = data
                    return data
            except Exception as e:
                logger.warning(f"Could not load results from disk: {e}")

        return None

    # =========================================================================
    # Common Tool Implementations
    # =========================================================================

    async def _ls(
        self,
        path: str,
        query: str = None,
        limit: int = 20,
        session_id: str = "default",
    ) -> dict:
        """List contents of a virtual path."""
        parsed = self.path_parser.parse(path)

        if parsed.error:
            return {"error": parsed.error, "path": path}

        if parsed.type == "root":
            return {
                "path": "/",
                "type": "root",
                "contents": [f"/{self.path_parser.root_name}/"],
            }

        if parsed.type == "documents_list":
            # Delegate to document store search
            results = await self.document_store.search_documents(
                query=query,
                limit=limit,
            )
            return {
                "path": path,
                "type": "documents_list",
                "count": len(results),
                "items": results,
            }

        if parsed.type == "document":
            # Show document structure
            doc = await self.document_store.get_document(parsed.document_id)
            if not doc:
                return {"error": f"Document not found: {parsed.document_id}"}

            return {
                "path": path,
                "type": "document",
                "document_id": parsed.document_id,
                "title": doc.get("title"),
                "contents": self._get_document_contents(doc),
            }

        return {"error": f"Cannot list path: {path}", "parsed": parsed.__dict__}

    def _get_document_contents(self, doc: dict) -> list[str]:
        """Get listing of document contents. Override for domain-specific structure."""
        return [
            "meta.json",
            "content.lines",
            "sections/",
        ]

    async def _cat(
        self,
        path: str,
        start: int = None,
        end: int = None,
        session_id: str = "default",
    ) -> dict:
        """Read file contents."""
        parsed = self.path_parser.parse(path)

        if parsed.error:
            return {"error": parsed.error, "path": path}

        if parsed.type == "file" and parsed.filename == "meta.json":
            doc = await self.document_store.get_document(parsed.document_id)
            if not doc:
                return {"error": f"Document not found: {parsed.document_id}"}
            return {"path": path, "type": "json", "content": doc}

        if parsed.type == "file" and parsed.filename == "content.lines":
            blocks = await self.document_store.get_document_content(parsed.document_id)
            if not blocks:
                return {"error": f"Content not found: {parsed.document_id}"}

            lines = self._format_content_lines(blocks, start, end)
            return {
                "path": path,
                "type": "lines",
                "total_lines": len(blocks),
                "lines": lines,
            }

        return {"error": f"Cannot read path: {path}"}

    def _format_content_lines(
        self,
        blocks: list[dict],
        start: int = None,
        end: int = None,
    ) -> list[dict]:
        """Format content blocks as numbered lines."""
        lines = []
        for block in blocks:
            line_num = block.get("line_number", len(lines) + 1)

            # Apply range filter
            if start and line_num < start:
                continue
            if end and line_num > end:
                break

            lines.append(
                {
                    "line": line_num,
                    "content": block.get("content", ""),
                }
            )

        return lines

    async def _head(
        self,
        path: str,
        n: int = 20,
        session_id: str = "default",
    ) -> dict:
        """Read first N lines of a file."""
        return await self._cat(path, start=1, end=n, session_id=session_id)

    async def _tail(
        self,
        path: str,
        n: int = 20,
        session_id: str = "default",
    ) -> dict:
        """Read last N lines of a file."""
        parsed = self.path_parser.parse(path)
        if parsed.error:
            return {"error": parsed.error}

        blocks = await self.document_store.get_document_content(parsed.document_id)
        total = len(blocks)
        start = max(1, total - n + 1)
        return await self._cat(path, start=start, session_id=session_id)

    async def _stat(
        self,
        path: str,
        session_id: str = "default",
    ) -> dict:
        """Get metadata/stats about a file or folder."""
        parsed = self.path_parser.parse(path)

        if parsed.error:
            return {"error": parsed.error}

        if parsed.type == "document" and parsed.document_id:
            doc = await self.document_store.get_document(parsed.document_id)
            blocks = await self.document_store.get_document_content(parsed.document_id)

            return {
                "path": path,
                "type": "document",
                "document_id": parsed.document_id,
                "title": doc.get("title") if doc else None,
                "line_count": len(blocks),
                "metadata": doc,
            }

        return {"path": path, "parsed": parsed.__dict__}

    # =========================================================================
    # Parallel Execution (map) — shared implementation
    # =========================================================================

    def get_reader_agent_config(self) -> str:
        """Return the agent config path for map reader subagents.

        Override in subclasses. E.g. "fda/fda_reviewer/fda_reviewer_reader"
        or "papers/papers_reader".
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_reader_agent_config()"
        )

    def get_reduce_config(self) -> tuple[str, str | None]:
        """Return (model, system_prompt) for reduce operations.

        Override in subclasses to load from a YAML config file.
        """
        return "google/gemini-2.5-flash", None

    def get_map_table_columns(self) -> list[str]:
        """Return column names for the map result table artifact.

        Override in subclasses for domain-specific columns.
        Default: ["response"]
        """
        return ["response"]

    def build_map_table_row(
        self, index: int, result: dict, meta: dict, response_text: str
    ) -> dict:
        """Build a single row for the map result table.

        Override in subclasses to add domain-specific columns (e.g.
        application_number, authors, month_year).
        """
        title = result.get("title") or result.get("path") or f"Document #{index+1}"
        return {
            "document": title[:80],
            "document_id": result.get("document_id", ""),
            "response": response_text,
        }

    _MAP_TIMEOUT_SECONDS = 600.0

    async def _parallel(
        self,
        from_results: str = None,
        tasks: list[dict] = None,
        limit: int = None,
        offset: int = None,
        query: str = None,
        output_schema: dict = None,
        max_concurrent: int = 25,
        include_rollouts: bool = False,
        batch_size: int = 20,
        session_id: str = "default",
        agent_id: str = None,
        **kwargs,
    ) -> dict:
        """Execute parallel document exploration (map phase).

        Called by VirtualTerminal._cmd_map. Returns dict with map_id,
        artifact_id, tasks_executed, tasks_successful, time_ms, results.
        """
        from .parallel import ParallelExecutor

        start_time = time.perf_counter()

        resolved_tasks = []
        source_items_meta: list[dict] = []

        if tasks:
            resolved_tasks = tasks

        elif from_results:
            saved = self.results_registry.load(from_results, session_id)

            if not saved and hasattr(self, "_load_artifact_as_papers") and from_results.startswith("a_"):
                saved = await self._load_artifact_as_papers(from_results, session_id)

            if not saved:
                available = self.results_registry.list_results(session_id, prefix="s")
                hint = (
                    f" Available: {', '.join(available[:5])}"
                    if available
                    else " No results saved in this session yet. Run a search first."
                )
                return {"error": f"Results not found: {from_results}.{hint}"}

            items = saved.get("items") or saved.get("papers") or []
            if offset:
                items = items[offset:]
            if limit:
                items = items[:limit]
            else:
                items = items[:10]

            if not query:
                return {"error": "Must provide 'query' parameter"}

            source_items_meta = items

            document_ids = [r.get("document_id") for r in items]
            if hasattr(self, "_normalize_document_ids"):
                document_ids = await self._normalize_document_ids(document_ids)

            root = self.path_parser.root_name
            resolved_tasks = [
                {"path": f"/{root}/{doc_id}/", "query": query}
                for doc_id in document_ids
            ]
        else:
            return {"error": "Must provide 'tasks' or 'from_results'"}

        if not resolved_tasks:
            return {"error": "No tasks to execute"}

        executor = ParallelExecutor(
            document_store=self.get_document_store(),
            agent_config=self.get_reader_agent_config(),
            session_manager=self.session_manager,
        )

        def extract_doc_id(path: str) -> str | None:
            parsed = self.path_parser.parse(path)
            return parsed.document_id if hasattr(parsed, "document_id") else None

        try:
            outputs = await asyncio.wait_for(
                executor.execute(
                    tasks=resolved_tasks,
                    max_concurrent=max_concurrent,
                    batch_size=batch_size,
                    output_schema=output_schema,
                    include_rollouts=include_rollouts,
                    session_id=session_id,
                    parent_agent_id=agent_id,
                    extract_document_id=extract_doc_id,
                ),
                timeout=self._MAP_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"map: timeout after {self._MAP_TIMEOUT_SECONDS:.0f}s — retry with a smaller --limit"
            )

        save_data = {"results": outputs, "query": query, "output_schema": output_schema}
        if from_results and saved and saved.get("source"):
            save_data["source"] = saved["source"]
        results_id = self.results_registry.save(
            data=save_data,
            session_id=session_id,
            prefix="m",
        )

        successful = [
            r for r in outputs if r.get("status") == "success" or "title" in r
        ]
        table_output = self._build_map_table(successful, source_items=source_items_meta)
        source_docs = [
            {"document_id": r.get("document_id", ""), "title": r.get("title", "")[:80]}
            for r in successful
            if r.get("document_id")
        ]
        table_artifact = {
            "artifact_id": results_id,
            "artifact_type": "reduce_table",
            "created_at": datetime.now().isoformat(),
            "source": {
                "source_id": results_id,
                "paper_count": len(source_docs),
                "papers": source_docs,
            },
            "output": table_output,
            "citations": table_output.get("citations", []),
        }
        self._save_artifact(results_id, table_artifact, session_id)

        return {
            "map_id": results_id,
            "artifact_id": results_id,
            "tasks_executed": len(outputs),
            "tasks_successful": sum(1 for o in outputs if o.get("status") == "success"),
            "time_ms": round((time.perf_counter() - start_time) * 1000),
            "results": outputs[:10],
        }

    # =========================================================================
    # Reduce — shared implementation
    # =========================================================================

    _REDUCE_CHAR_BUDGET = 4_000_000

    async def _reduce(
        self,
        from_map: str = None,
        from_parallel: str = None,
        from_results: str = None,
        question: str = None,
        session_id: str = "default",
        strategy: str = None,
        columns: list[str] = None,
        fields: list[str] = None,
        max_items: int = None,
        **kwargs,
    ) -> dict:
        """Synthesize map results into a cohesive narrative using an LLM."""
        from gxl_inference_client.agent import Agent
        from shared.core.environment import get_inference_url

        start_time = time.perf_counter()

        source_id = from_map or from_parallel or from_results
        if not source_id:
            return {"error": "Must provide from_map or from_results"}

        saved = self.results_registry.load(source_id, session_id)
        if not saved:
            return {"error": f"Results not found: {source_id}"}

        if "results" in saved:
            results = saved["results"]
        elif "outputs" in saved:
            results = saved["outputs"]
        elif "papers" in saved:
            results = saved["papers"]
        else:
            return {"error": f"Invalid results format in {source_id}"}

        successful = [
            r for r in results if r.get("status") == "success" or "title" in r
        ]

        if not successful:
            return {"error": "No successful results to reduce", "source": source_id}

        context_parts = []
        chars_used = 0
        included = 0
        for i, r in enumerate(successful):
            title = r.get("title") or r.get("path", f"Item {i+1}")
            doc_id = r.get("document_id", "")
            output = r.get("output") or r.get("abstract", "")
            if isinstance(output, dict):
                output = json.dumps(output, indent=2)
            elif isinstance(output, str) and doc_id:
                output = self._uplift_block_citations(output, doc_id)
            header = f"## Document {i+1}: {title}"
            if doc_id:
                header += f"\ndocument_id: {doc_id}"
            entry = f"{header}\n{output}"
            if chars_used + len(entry) > self._REDUCE_CHAR_BUDGET:
                break
            context_parts.append(entry)
            chars_used += len(entry)
            included += 1

        truncated = len(successful) - included
        context = "\n\n".join(context_parts)
        task = question or "Summarize the key findings"

        truncation_note = (
            f"\n\n[Note: {truncated} additional documents were excluded to stay within context limits.]"
            if truncated > 0
            else ""
        )

        prompt = f"""Task: {task}

{context}{truncation_note}"""

        reduce_model, reduce_system_prompt = self.get_reduce_config()
        try:
            reducer_id = f"reducer_{uuid.uuid4().hex[:6]}"
            base_url = get_inference_url()

            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                reducer_payload = {"agent_id": reducer_id, "model": reduce_model}
                parent_id = kwargs.get("agent_id")
                if parent_id:
                    reducer_payload["parent_agent_id"] = parent_id
                resp = await client.post(
                    f"{base_url}/api/sessions/{session_id}/agents",
                    json=reducer_payload,
                    timeout=10.0,
                )
                if resp.status_code not in (200, 201, 409):
                    logger.warning(
                        f"Failed to register reducer agent {reducer_id}: {resp.status_code}"
                    )

            agent = Agent(
                agent_id=reducer_id,
                session_id=session_id,
                model=reduce_model,
                system_prompt=reduce_system_prompt,
                base_url=base_url,
                timeout=90.0,
            )
            output_text = await agent.call_async(prompt)
        except Exception as e:
            logger.warning(f"LLM synthesis failed (model={reduce_model}): {e}")
            fallback = []
            for i, r in enumerate(successful):
                title = r.get("title") or f"Item {i+1}"
                text = r.get("output") or r.get("abstract", "")
                if isinstance(text, dict):
                    text = json.dumps(text)
                if text:
                    fallback.append(f"**{title}**: {str(text)[:500]}")
            output_text = (
                (
                    f"*LLM synthesis unavailable — showing {len(successful)} results:*\n\n"
                    + "\n\n".join(fallback)
                )
                if fallback
                else f"[Synthesis unavailable - {len(successful)} results]"
            )

        citation_index = self._extract_citation_index(successful)
        if citation_index:
            ref_lines = [
                "\n---\nCITATION REFERENCE (use these block_id citations in your response):"
            ]
            for c in citation_index:
                ref_lines.append(
                    f'- {{"block_id": {c["block_id"]}}} — "{c["context"]}" '
                    f'({c["paper_title"]})'
                )
            output_text += "\n".join(ref_lines)

        artifact_id = f"r_{uuid.uuid4().hex[:8]}"
        source_docs = [
            {"document_id": r.get("document_id", ""), "title": r.get("title", "")[:80]}
            for r in successful
            if r.get("document_id")
        ]
        artifact = {
            "artifact_id": artifact_id,
            "artifact_type": "reduce_summarize",
            "created_at": datetime.now().isoformat(),
            "source": {
                "source_id": source_id,
                "paper_count": len(source_docs),
                "papers": source_docs,
            },
            "output": output_text,
            "citations": [],
        }
        self._save_artifact(artifact_id, artifact, session_id)

        return {
            "artifact_id": artifact_id,
            "source": source_id,
            "items_processed": len(successful),
            "time_ms": round((time.perf_counter() - start_time) * 1000),
            "output": output_text,
        }

    # =========================================================================
    # Shared helpers
    # =========================================================================

    _BLOCK_ID_PATTERN = re.compile(r'\{\{"block_id":\s*"?([^"}\s]+)"?\}\}')

    @classmethod
    def _extract_citation_index(cls, results: list[dict]) -> list[dict]:
        """Extract block_id → surrounding text mappings from reader outputs."""
        citations = []
        seen: set[str] = set()

        for r in results:
            raw = r.get("output", "")
            if not raw or not isinstance(raw, str):
                continue
            title = (r.get("title") or r.get("path", ""))[:60]

            for m in cls._BLOCK_ID_PATTERN.finditer(raw):
                block_id = m.group(1)
                if block_id in seen:
                    continue
                seen.add(block_id)

                start = max(0, m.start() - 150)
                context = raw[start : m.start()].strip()
                for sep in [". ", ".\n", "\n\n"]:
                    idx = context.rfind(sep)
                    if idx != -1:
                        context = context[idx + len(sep) :]
                        break
                context = context[:120]

                citations.append(
                    {
                        "block_id": block_id,
                        "context": context,
                        "paper_title": title,
                    }
                )

        return citations

    @classmethod
    def _uplift_block_citations(cls, text: str, doc_id: str) -> str:
        """Transform {{"block_id": "X"}} into full {{"citation": {...}}} format.

        Extracts preceding sentence as the content field so the reducer LLM
        doesn't need to perform the format transformation itself.
        """
        if not doc_id or "block_id" not in text:
            return text

        matches = list(cls._BLOCK_ID_PATTERN.finditer(text))
        if not matches:
            return text

        parts = []
        last_end = 0
        for m in matches:
            parts.append(text[last_end : m.start()])

            block_id = m.group(1)
            preceding = text[max(0, m.start() - 200) : m.start()].strip()
            for sep in [". ", ".\n", "\n\n"]:
                idx = preceding.rfind(sep)
                if idx != -1:
                    preceding = preceding[idx + len(sep) :]
                    break
            content = preceding[:150].replace('"', '\\"').replace("\n", " ")

            parts.append(
                '{{"citation": {"doc_id": "' + doc_id
                + '", "block_id": "' + str(block_id)
                + '", "content": "' + content
                + '"}}}'
            )
            last_end = m.end()

        parts.append(text[last_end:])
        return "".join(parts)

    def _build_map_table(
        self,
        results: list[dict],
        source_items: list[dict] | None = None,
    ) -> dict:
        """Build a deterministic table from map results.

        Uses get_map_table_columns() and build_map_table_row() for
        domain-specific customization.
        """
        meta_by_id: dict[str, dict] = {}
        if source_items:
            for item in source_items:
                did = item.get("document_id", "")
                if did:
                    meta_by_id[did] = item

        fixed_columns = self.get_map_table_columns()
        rows = []
        all_citations: list[dict] = []

        for i, r in enumerate(results):
            title = r.get("title") or r.get("path") or f"Document #{i+1}"
            doc_id = r.get("document_id", "")
            meta = meta_by_id.get(doc_id, {})

            response_text = self._format_map_output(r, title, i, all_citations)

            row = self.build_map_table_row(i, r, meta, response_text)
            rows.append(row)

        return {
            "columns": fixed_columns,
            "rows": rows,
            "citations": all_citations,
        }

    @staticmethod
    def _format_map_output(
        result: dict, title: str, index: int, all_citations: list[dict]
    ) -> str:
        """Format a single map result's output for the table.

        Extracts citations from JSON output and returns display text.
        """
        doc_id = result.get("document_id", "")
        raw_output = result.get("output")
        if raw_output is None or raw_output == "":
            if result.get("status") == "error":
                return f"[Error: {result.get('error', 'Unknown error')}]"
            return "[No output]"
        elif isinstance(raw_output, dict):
            display = {k: v for k, v in raw_output.items() if not k.startswith("_")}
            return json.dumps(display, indent=1) if display else str(raw_output)
        elif isinstance(raw_output, str):
            try:
                parsed = json.loads(raw_output)
                if isinstance(parsed, dict):
                    paper_citations = parsed.pop("_citations", [])
                    if isinstance(paper_citations, list):
                        for cit in paper_citations:
                            if isinstance(cit, dict):
                                all_citations.append({
                                    "document_id": doc_id,
                                    "line_number": cit.get("line"),
                                    "content": cit.get("content", "")[:200],
                                    "field": cit.get("field", ""),
                                    "_paper_title": title[:60],
                                    "_row_idx": index,
                                })
                    display = {k: v for k, v in parsed.items() if not k.startswith("_")}
                    return json.dumps(display, indent=1) if display else raw_output[:500]
                else:
                    return str(parsed)[:500]
            except (json.JSONDecodeError, ValueError):
                return raw_output[:500]
        else:
            return str(raw_output)[:500]

    def _save_artifact(self, artifact_id: str, artifact: dict, session_id: str):
        """Save artifact to local disk. Override for other backends (e.g. GXLFileSystem)."""
        try:
            base = os.getenv("LOCAL_SESSION_STORAGE_ROOT", "/workspaces/gxl/sessions")
            artifacts_dir = os.path.join(base, session_id, "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            filepath = os.path.join(artifacts_dir, f"{artifact_id}.json")
            with open(filepath, "w") as f:
                json.dump(artifact, f, indent=2, default=str)
            logger.info(f"Saved artifact {artifact_id}")
        except Exception as e:
            logger.warning(f"Failed to save artifact: {e}")
