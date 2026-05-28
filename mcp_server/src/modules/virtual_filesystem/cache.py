"""
Results Registry - Session-based caching for search results and artifacts.

Provides:
- In-memory caching for fast access
- Disk persistence via session sandbox
- Cursor-based pagination for large result sets
"""

import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

from cachetools import LRUCache

logger = logging.getLogger(__name__)


def _generate_id(prefix: str = "r") -> str:
    """Generate a short unique ID."""
    unique = f"{time.time()}{random.random()}"
    return f"{prefix}_{hashlib.md5(unique.encode()).hexdigest()[:8]}"


class ResultsRegistry:
    """Manage search results and artifacts with session-based persistence.

    Results are stored with IDs like:
    - s_abc123: Search results from papers_find/papers_grep/funded-by
    - m_xyz789: Map (parallel execution) results
    - r_def456: Reduce operation results/artifacts

    Usage:
        registry = ResultsRegistry()

        # Save results
        results_id = registry.save(
            data={"papers": [...]},
            session_id="abc",
            prefix="s",
        )

        # Load results
        data = registry.load(results_id, session_id="abc")
    """

    def __init__(self, session_manager=None, storage_root: str = None):
        self.session_manager = session_manager
        self._storage_root = storage_root or os.getenv(
            "LOCAL_SESSION_STORAGE_ROOT", "/workspaces/gxl/sessions"
        )
        self._cache: LRUCache = LRUCache(maxsize=500)

    def _get_session_dir(self, session_id: str) -> Path:
        """Get the session results directory, creating if needed."""
        session_dir = Path(self._storage_root) / session_id / "results"
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def save(
        self,
        data: dict,
        session_id: str = "default",
        prefix: str = "r",
        results_id: str = None,
    ) -> str:
        """Save results to in-memory cache and schedule async disk persist.

        The in-memory write is synchronous (instant); the disk write runs
        in a background thread so it never blocks the caller.

        Returns:
            Results ID
        """
        if results_id is None:
            results_id = _generate_id(prefix)

        cache_key = f"{session_id}:{results_id}"
        self._cache[cache_key] = data

        # Fire-and-forget disk persist via thread to avoid blocking
        import threading

        sid, rid = session_id, results_id

        def _persist():
            try:
                results_dir = self._get_session_dir(sid)
                results_file = results_dir / f"{rid}.json"
                with open(results_file, "w") as f:
                    json.dump(data, f, indent=2, default=str)
                logger.debug(f"Persisted results to {results_file}")
            except Exception as e:
                logger.warning(f"Could not persist results: {e}")

        threading.Thread(target=_persist, daemon=True).start()

        return results_id

    def load(
        self,
        results_id: str,
        session_id: str = "default",
    ) -> dict | None:
        """Load results from cache or disk.

        Args:
            results_id: Results identifier
            session_id: Session identifier

        Returns:
            Results data or None if not found
        """
        cache_key = f"{session_id}:{results_id}"

        # Check in-memory cache
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try loading from disk
        try:
            results_dir = self._get_session_dir(session_id)
            results_file = results_dir / f"{results_id}.json"

            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                self._cache[cache_key] = data
                return data
        except Exception as e:
            logger.warning(f"Could not load results: {e}")

        return None

    def delete(
        self,
        results_id: str,
        session_id: str = "default",
    ) -> bool:
        """Delete results from cache and disk.

        Returns:
            True if deleted, False if not found
        """
        cache_key = f"{session_id}:{results_id}"

        deleted = False

        # Remove from cache
        if cache_key in self._cache:
            del self._cache[cache_key]
            deleted = True

        # Remove from disk
        if self.session_manager:
            try:
                sandbox_path = self.session_manager.get_sandbox_path(session_id)
                results_file = sandbox_path / "results" / f"{results_id}.json"

                if results_file.exists():
                    results_file.unlink()
                    deleted = True
            except Exception as e:
                logger.warning(f"Could not delete results file: {e}")

        return deleted

    def list_results(
        self,
        session_id: str = "default",
        prefix: str = None,
    ) -> list[str]:
        """List all result IDs for a session.

        Args:
            session_id: Session identifier
            prefix: Optional filter by prefix (r, p, a)

        Returns:
            List of result IDs
        """
        results = set()

        # From cache
        for key in self._cache:
            if key.startswith(f"{session_id}:"):
                rid = key.split(":", 1)[1]
                if prefix is None or rid.startswith(f"{prefix}_"):
                    results.add(rid)

        # From disk
        if self.session_manager:
            try:
                sandbox_path = self.session_manager.get_sandbox_path(session_id)
                results_dir = sandbox_path / "results"

                if results_dir.exists():
                    for f in results_dir.glob("*.json"):
                        rid = f.stem
                        if prefix is None or rid.startswith(f"{prefix}_"):
                            results.add(rid)
            except Exception as e:
                logger.warning(f"Could not list results: {e}")

        return sorted(results)

    def get_stats(self, session_id: str = "default") -> dict:
        """Get statistics about cached results."""
        results = self.list_results(session_id)

        by_prefix = {}
        for rid in results:
            prefix = rid.split("_")[0] if "_" in rid else "unknown"
            by_prefix[prefix] = by_prefix.get(prefix, 0) + 1

        return {
            "session_id": session_id,
            "total_results": len(results),
            "by_type": by_prefix,
            "cache_size": sum(1 for k in self._cache if k.startswith(f"{session_id}:")),
        }

    def clear_cache(self, session_id: str = None):
        """Clear in-memory cache.

        Args:
            session_id: Clear only this session, or all if None
        """
        if session_id:
            keys_to_delete = [k for k in self._cache if k.startswith(f"{session_id}:")]
            for k in keys_to_delete:
                del self._cache[k]
        else:
            self._cache.clear()


class CursorPaginator:
    """Handle cursor-based pagination for large result sets.

    When results exceed a threshold, stores full results and returns
    a cursor ID that can be used to fetch subsequent pages.
    """

    def __init__(
        self,
        registry: ResultsRegistry,
        page_size: int = 20,
        cursor_threshold: int = 100,
    ):
        self.registry = registry
        self.page_size = page_size
        self.cursor_threshold = cursor_threshold

    def paginate(
        self,
        results: list,
        session_id: str = "default",
        metadata: dict = None,
    ) -> dict:
        """Paginate results, returning cursor if needed.

        Returns:
            {
                "items": [...],  # First page
                "total": N,
                "cursor_id": "c_xxx" or None,
                "has_more": bool,
            }
        """
        total = len(results)

        if total <= self.cursor_threshold:
            # Return all results
            return {
                "items": results,
                "total": total,
                "cursor_id": None,
                "has_more": False,
            }

        # Store full results with cursor
        cursor_id = _generate_id("c")
        self.registry.save(
            data={
                "items": results,
                "metadata": metadata or {},
                "created_at": time.time(),
            },
            session_id=session_id,
            results_id=cursor_id,
        )

        return {
            "items": results[: self.page_size],
            "total": total,
            "cursor_id": cursor_id,
            "has_more": True,
            "page": 1,
            "page_size": self.page_size,
        }

    def get_page(
        self,
        cursor_id: str,
        page: int = 1,
        session_id: str = "default",
    ) -> dict:
        """Get a specific page from cursor results.

        Args:
            cursor_id: Cursor identifier
            page: Page number (1-indexed)
            session_id: Session identifier

        Returns:
            Same format as paginate()
        """
        data = self.registry.load(cursor_id, session_id)

        if not data:
            return {"error": f"Cursor not found: {cursor_id}"}

        items = data.get("items", [])
        total = len(items)

        start = (page - 1) * self.page_size
        end = start + self.page_size

        return {
            "items": items[start:end],
            "total": total,
            "cursor_id": cursor_id,
            "has_more": end < total,
            "page": page,
            "page_size": self.page_size,
            "total_pages": (total + self.page_size - 1) // self.page_size,
        }
