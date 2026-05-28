"""Unified search-backend clients for the Papers MCP.

Everything the Papers MCP needs to talk to the new search infrastructure
lives here:

- :func:`get_opensearch_client` — lazy HTTP client for the OpenSearch cluster
  on the ``gxl-search`` GCE VM (title+abstract BM25 across preprints, pmc,
  abstract_only).
- :func:`get_qdrant_client` — lazy HTTP client for the Qdrant cluster on the
  same VM (3072-dim Gemini embeddings per corpus collection).

Both clients use plain ``httpx`` (no heavy SDKs) so they work identically on
Cloud Run (via the VPC connector) and locally (via the SSH tunnels used by
``scripts/search_service/gxl_search.py``).

Index / collection map (source of truth, mirrored from the migration
scripts under ``scripts/search_service/``)::

    corpus          OS index        Qdrant collection   source filter
    biomedrxiv      preprints       biomedrxiv          source IN ('biorxiv','medrxiv')
    arxiv           preprints       arxiv               source = 'arxiv'
    pmc             pmc             pmc                 (none)
    abstract_only   abstract_only   abstract_only       (none)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Corpus → (index, collection, source filter) map
# ---------------------------------------------------------------------------

# Canonical corpus keys used internally.
CORPUS_BIOMEDRXIV = "biomedrxiv"
CORPUS_ARXIV = "arxiv"
CORPUS_PMC = "pmc"
CORPUS_ABSTRACT_ONLY = "abstract_only"

PREPRINTS_OS_INDEX = "preprints"

ALL_CORPORA: tuple[str, ...] = (
    CORPUS_BIOMEDRXIV,
    CORPUS_ARXIV,
    CORPUS_PMC,
    CORPUS_ABSTRACT_ONLY,
)

# Default corpora for search (excludes abstract_only unless explicitly requested).
DEFAULT_CORPORA: tuple[str, ...] = (
    CORPUS_BIOMEDRXIV,
    CORPUS_ARXIV,
    CORPUS_PMC,
)

# OpenSearch index per corpus. arxiv shares the preprints index (filtered by
# source). Everything else is 1:1.
OS_INDEX_BY_CORPUS: dict[str, str] = {
    CORPUS_BIOMEDRXIV: PREPRINTS_OS_INDEX,
    CORPUS_ARXIV: PREPRINTS_OS_INDEX,
    CORPUS_PMC: "pmc",
    CORPUS_ABSTRACT_ONLY: "abstract_only",
}

# OS `source` field values that belong to this corpus. `None` means the
# entire index is this corpus (no filter needed).
OS_SOURCE_FILTER_BY_CORPUS: dict[str, list[str] | None] = {
    CORPUS_BIOMEDRXIV: ["biorxiv", "medrxiv"],
    CORPUS_ARXIV: ["arxiv"],
    CORPUS_PMC: None,
    CORPUS_ABSTRACT_ONLY: None,
}

# Qdrant collection per corpus (each corpus has its own collection).
QDRANT_COLLECTION_BY_CORPUS: dict[str, str] = {
    CORPUS_BIOMEDRXIV: "biomedrxiv",
    CORPUS_ARXIV: "arxiv",
    CORPUS_PMC: "pmc",
    CORPUS_ABSTRACT_ONLY: "abstract_only",
}

# `source` field values (as stored in both OS `_source` and Qdrant payload)
# that map back to a corpus. Used for routing hydration / content lookups.
_SOURCE_VALUE_TO_CORPUS: dict[str, str] = {
    "biorxiv": CORPUS_BIOMEDRXIV,
    "medrxiv": CORPUS_BIOMEDRXIV,
    "arxiv": CORPUS_ARXIV,
    "pmc": CORPUS_PMC,
    "openalex": CORPUS_ABSTRACT_ONLY,
    "abstracts": CORPUS_ABSTRACT_ONLY,
}


def corpus_for_source(source: str | None) -> str:
    """Map a ``source`` field value back to its corpus key."""
    if not source:
        return CORPUS_BIOMEDRXIV
    return _SOURCE_VALUE_TO_CORPUS.get(source.lower(), CORPUS_BIOMEDRXIV)


def resolve_corpora(requested: list[str] | str | None) -> list[str]:
    """Normalise user-facing ``source`` filters to the list of corpora to hit.

    Accepts legacy values (``biorxiv``, ``medrxiv``, ``pmc``, ``openalex``,
    ``abstracts``) plus the new corpus keys (``biomedrxiv``, ``arxiv``,
    ``abstract_only``, ``all``). Returns a de-duplicated list in canonical order.

    By default, ``abstract_only`` is excluded, including for ``"all"``.
    Explicitly request ``"abstracts"`` / ``"openalex"`` /
    ``"abstract_only"`` to include it.
    """
    if requested is None or requested == []:
        return list(DEFAULT_CORPORA)
    if requested == "all" or requested == ["all"]:
        return list(DEFAULT_CORPORA)
    if isinstance(requested, str):
        requested = [requested]

    out: list[str] = []
    for raw in requested:
        if not raw:
            continue
        key = raw.lower()
        if key == "all":
            for corpus in DEFAULT_CORPORA:
                if corpus not in out:
                    out.append(corpus)
            continue
        if key in ALL_CORPORA:
            if key not in out:
                out.append(key)
            continue
        # User-facing aliases.
        if key in ("abstracts", "openalex"):
            if CORPUS_ABSTRACT_ONLY not in out:
                out.append(CORPUS_ABSTRACT_ONLY)
            continue
        mapped_corpus = _SOURCE_VALUE_TO_CORPUS.get(key)
        if mapped_corpus and mapped_corpus not in out:
            out.append(mapped_corpus)
    return out or list(DEFAULT_CORPORA)


# ---------------------------------------------------------------------------
# OpenSearch client
# ---------------------------------------------------------------------------

_OS_TIMEOUT_SECONDS = float(os.environ.get("OPENSEARCH_TIMEOUT_SECONDS", "15"))
_OS_IDLE_RESET_SECONDS = 120


class OpenSearchClient:
    """Tiny ``httpx``-backed OpenSearch client (search + msearch + mget)."""

    def __init__(self, base_url: str, auth: tuple[str, str] | None = None):
        self.base_url = base_url.rstrip("/")
        self._auth = auth
        self._client: httpx.Client | None = None
        self._lock = threading.Lock()
        self._last_used: float = 0

    def _get_httpx(self) -> httpx.Client:
        now = time.time()
        with self._lock:
            if self._client is not None and self._last_used:
                idle = now - self._last_used
                if idle > _OS_IDLE_RESET_SECONDS:
                    try:
                        self._client.close()
                    except Exception:
                        pass
                    self._client = None
            if self._client is None:
                self._client = httpx.Client(
                    base_url=self.base_url,
                    timeout=_OS_TIMEOUT_SECONDS,
                    auth=self._auth,
                    follow_redirects=False,
                )
            return self._client

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

    def _post(self, path: str, body: Any) -> dict:
        client = self._get_httpx()
        resp = client.post(path, json=body)
        self._last_used = time.time()
        resp.raise_for_status()
        return resp.json()

    def search(self, index: str, body: dict, **_kw: Any) -> dict:
        """Run a search query against one or more indices (comma-separated).

        Extra keyword arguments (``request_timeout``, etc.) are accepted for
        elasticsearch-py compatibility but ignored; the timeout is fixed at
        construction time.
        """
        return self._post(f"/{index}/_search", body)

    def count(self, index: str, body: dict | None = None, **_kw: Any) -> dict:
        """Cheap ``_count`` query, for keep-alive pings."""
        return self._post(f"/{index}/_count", body or {"query": {"match_all": {}}})

    def msearch(
        self,
        index_body_pairs: list[tuple[str, dict]] | None = None,
        *,
        body: list | None = None,
        **_kw: Any,
    ) -> dict:
        """Batch search with elasticsearch-py-compatible ``body=`` shape.

        Accepts either:
        - the native ``index_body_pairs=[(index, body), ...]`` form, or
        - the elasticsearch-py ``body=[{"index": idx}, body, ...]`` form.

        Returns a dict ``{"responses": [...]}`` matching the ES API.
        """
        import json as _json

        if index_body_pairs is None and body is not None:
            # Unpack elasticsearch-py-style body: alternating header + query.
            index_body_pairs = []
            i = 0
            while i < len(body) - 1:
                header = body[i]
                query_body = body[i + 1]
                idx = (
                    header.get("index") if isinstance(header, dict) else ""
                ) or ""
                if isinstance(idx, list):
                    idx = ",".join(idx)
                index_body_pairs.append((idx, query_body))
                i += 2
        if not index_body_pairs:
            return {"responses": []}

        client = self._get_httpx()
        lines: list[str] = []
        for idx, qbody in index_body_pairs:
            lines.append(_json.dumps({"index": idx}))
            lines.append(_json.dumps(qbody))
        payload = "\n".join(lines) + "\n"
        resp = client.post(
            "/_msearch",
            content=payload,
            headers={"Content-Type": "application/x-ndjson"},
        )
        self._last_used = time.time()
        resp.raise_for_status()
        return resp.json()

    def msearch_pairs(self, index_body_pairs: list[tuple[str, dict]]) -> list[dict]:
        """``msearch`` returning the unwrapped ``responses`` list (convenience)."""
        return self.msearch(index_body_pairs).get("responses", [])

    def mget(self, index: str, ids: list[str], source_fields: list[str] | None = None) -> list[dict]:
        """Multi-get documents by ``_id``. Returns the raw ``docs`` list.

        OpenSearch's ``_mget`` does not allow ``_source`` in the request body
        when using the ``ids`` form (it errors with ``"unknown key [_source]
        for a START_ARRAY"``). To still pull only the fields we need we send
        ``_source`` as a query-string parameter instead.
        """
        client = self._get_httpx()
        params: dict[str, str] = {}
        if source_fields:
            params["_source"] = ",".join(source_fields)
        resp = client.post(f"/{index}/_mget", json={"ids": ids}, params=params)
        self._last_used = time.time()
        resp.raise_for_status()
        return resp.json().get("docs", [])


_os_client: OpenSearchClient | None = None
_os_lock = threading.Lock()


def get_opensearch_client() -> OpenSearchClient | None:
    """Lazily build a singleton OpenSearch client.

    Reads ``OPENSEARCH_URL`` (e.g. ``http://10.142.0.5:9200`` behind the VPC
    connector) plus optional ``OPENSEARCH_USER`` / ``OPENSEARCH_PASSWORD``.
    """
    global _os_client
    if _os_client is not None:
        return _os_client

    if os.environ.get("PAPERS_DISABLE_OPENSEARCH", "").lower() in ("1", "true", "yes"):
        logger.info("OpenSearch disabled via PAPERS_DISABLE_OPENSEARCH")
        return None

    url = os.environ.get("OPENSEARCH_URL") or os.environ.get("OS_URL")
    if not url:
        logger.warning(
            "OpenSearch not configured (set OPENSEARCH_URL to the gxl-search "
            "internal endpoint or the local SSH-tunnel URL)"
        )
        return None

    user = os.environ.get("OPENSEARCH_USER")
    pw = os.environ.get("OPENSEARCH_PASSWORD")
    auth = (user, pw) if user and pw else None

    with _os_lock:
        if _os_client is None:
            _os_client = OpenSearchClient(url, auth=auth)
            logger.info("Connected to OpenSearch at %s", url)
    return _os_client


def reset_opensearch_client() -> OpenSearchClient | None:
    """Force OpenSearch client recreation after transport-level failures."""
    global _os_client
    with _os_lock:
        if _os_client is not None:
            try:
                _os_client.close()
            except Exception:
                pass
            _os_client = None
    return get_opensearch_client()


# ---------------------------------------------------------------------------
# Qdrant client
# ---------------------------------------------------------------------------

_QDRANT_TIMEOUT_SECONDS = float(os.environ.get("QDRANT_TIMEOUT_SECONDS", "15"))


class QdrantClient:
    """Minimal Qdrant REST client (search + scroll)."""

    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client: httpx.Client | None = None
        self._lock = threading.Lock()

    def _get_httpx(self) -> httpx.Client:
        with self._lock:
            if self._client is None:
                headers: dict[str, str] = {}
                if self._api_key:
                    headers["api-key"] = self._api_key
                self._client = httpx.Client(
                    base_url=self.base_url,
                    timeout=_QDRANT_TIMEOUT_SECONDS,
                    headers=headers,
                    follow_redirects=False,
                )
            return self._client

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

    def search(
        self,
        collection: str,
        vector: list[float],
        limit: int,
        *,
        with_payload: bool = True,
        query_filter: dict | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]:
        # All four collections are configured with binary quantization
        # (`{binary: {always_ram: true}}`) plus fp32 originals on disk. We
        # explicitly request candidate-selection on the quantized index and
        # rescore the oversampled top-K against the full-precision vectors
        # from disk, so scores returned to callers match fp32 semantics.
        # Oversampling of 2.0 is the recommended starting point for binary
        # quantization to recover recall lost to 1-bit quantization.
        body: dict[str, Any] = {
            "vector": vector,
            "limit": limit,
            "with_payload": with_payload,
            "params": {
                "quantization": {
                    "ignore": False,
                    "rescore": True,
                    "oversampling": 2.0,
                },
            },
        }
        if query_filter:
            body["filter"] = query_filter
        if score_threshold is not None:
            body["score_threshold"] = score_threshold
        client = self._get_httpx()
        resp = client.post(f"/collections/{collection}/points/search", json=body)
        resp.raise_for_status()
        return resp.json().get("result", [])


_qdrant_client: QdrantClient | None = None
_qdrant_lock = threading.Lock()


def get_qdrant_client() -> QdrantClient | None:
    """Lazily build a singleton Qdrant client.

    Reads ``QDRANT_URL`` and optional ``QDRANT_API_KEY``.
    """
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client

    if os.environ.get("PAPERS_DISABLE_QDRANT", "").lower() in ("1", "true", "yes"):
        logger.info("Qdrant disabled via PAPERS_DISABLE_QDRANT")
        return None

    url = os.environ.get("QDRANT_URL")
    if not url:
        logger.warning(
            "Qdrant not configured (set QDRANT_URL to the gxl-search "
            "internal endpoint or the local SSH-tunnel URL)"
        )
        return None

    api_key = os.environ.get("QDRANT_API_KEY")

    with _qdrant_lock:
        if _qdrant_client is None:
            _qdrant_client = QdrantClient(url, api_key=api_key)
            logger.info("Connected to Qdrant at %s", url)
    return _qdrant_client


def opensearch_url() -> str | None:
    """Currently-configured OpenSearch URL (or ``None`` if disabled/unset)."""
    if os.environ.get("PAPERS_DISABLE_OPENSEARCH", "").lower() in ("1", "true", "yes"):
        return None
    return os.environ.get("OPENSEARCH_URL") or os.environ.get("OS_URL")


def qdrant_url() -> str | None:
    """Currently-configured Qdrant URL (or ``None`` if disabled/unset)."""
    if os.environ.get("PAPERS_DISABLE_QDRANT", "").lower() in ("1", "true", "yes"):
        return None
    return os.environ.get("QDRANT_URL")


__all__ = [
    "ALL_CORPORA",
    "CORPUS_ABSTRACT_ONLY",
    "CORPUS_ARXIV",
    "CORPUS_BIOMEDRXIV",
    "CORPUS_PMC",
    "DEFAULT_CORPORA",
    "OS_INDEX_BY_CORPUS",
    "OS_SOURCE_FILTER_BY_CORPUS",
    "OpenSearchClient",
    "QDRANT_COLLECTION_BY_CORPUS",
    "QdrantClient",
    "corpus_for_source",
    "get_opensearch_client",
    "get_qdrant_client",
    "opensearch_url",
    "qdrant_url",
    "reset_opensearch_client",
    "resolve_corpora",
]
