"""
Papers Filesystem Module - Using VirtualFilesystemModule base.

Extends the generalized virtual filesystem framework for literature access.
"""

import asyncio
import json
import logging
import os
import random
import re
import time
import uuid
from datetime import datetime, timedelta
from typing import Any

from mcp.types import TextContent, Tool

from ..virtual_filesystem.base import (
    DocumentStore,
    ParsedPath,
    PathParser,
    VirtualFilesystemModule,
)
from ..virtual_filesystem.cache import ResultsRegistry, _generate_id

from ..virtual_filesystem.terminal import VirtualTerminal
from .short_ids import resolve, shorten, shorten_result, shorten_results

logger = logging.getLogger(__name__)

# Lazy-loaded clients
_es_client = None
_db_conn = None

# Grep timeout (seconds)
GREP_TIMEOUT_SECONDS = 15


_db_last_used = 0  # Timestamp of last successful query
_DB_HEALTH_CHECK_INTERVAL = 30  # Only health check if idle > 30 seconds

# OpenAlex connections (papers+mapping on biomedrxiv-prod, edges on dedicated instance)
_oa_papers_conn = None
_oa_papers_last_used = 0
_oa_edges_conn = None
_oa_edges_last_used = 0

# OpenAlex in Paperclip search — no longer a separate index; abstract_only covers it.
OPENALEX_SEARCH_ENABLED = os.getenv("OPENALEX_SEARCH", "1").lower() in (
    "1",
    "true",
    "yes",
)

# Abstract-only OpenSearch index (title + abstract, no full text on disk)
ABSTRACT_ONLY_OS_INDEX = "abstract_only"
ABSTRACT_ONLY_SEARCH_ENABLED = os.getenv("ABSTRACT_ONLY_SEARCH", "1").lower() in (
    "1",
    "true",
    "yes",
)


def _get_db_connection():
    """Get PostgreSQL connection with smart health check.

    Only checks connection health if it's been idle for >30 seconds.
    This avoids the ~5-20ms overhead of SELECT 1 on every call.

    Supports two configuration methods:
    1. BIOMEDRXIV_DB_URL (full connection string, preferred)
    2. Individual vars: BIOMEDRXIV_DB_HOST, BIOMEDRXIV_DB_PASSWORD, etc.
    """
    global _db_conn, _db_last_used
    import psycopg2

    def _create_connection():
        """Create a new database connection with keepalives.

        TCP keepalives prevent CloudSQL from closing idle connections.
        - keepalives=1: Enable TCP keepalives
        - keepalives_idle=300: Start sending keepalives after 5 min idle
        - keepalives_interval=30: Send keepalive every 30 sec
        - keepalives_count=5: Close after 5 failed keepalives

        Connection stays alive indefinitely as long as MCP server runs.
        """
        db_url = os.getenv("BIOMEDRXIV_DB_URL")
        if db_url:
            # Pass URL directly - psycopg2 handles socket URLs with ?host= query param
            conn = psycopg2.connect(
                db_url,
                keepalives=1,
                keepalives_idle=300,
                keepalives_interval=30,
                keepalives_count=5,
            )
            conn.autocommit = True
            return conn

        host = os.getenv("BIOMEDRXIV_DB_HOST")
        password = os.getenv("BIOMEDRXIV_DB_PASSWORD")

        if not host or not password:
            raise ValueError(
                "Database not configured. Set BIOMEDRXIV_DB_URL or BIOMEDRXIV_DB_HOST + BIOMEDRXIV_DB_PASSWORD."
            )

        conn = psycopg2.connect(
            host=host,
            port=int(os.getenv("BIOMEDRXIV_DB_PORT", "5432")),
            database=os.getenv("BIOMEDRXIV_DB_NAME", "biomedrxiv"),
            user=os.getenv("BIOMEDRXIV_DB_USER", "postgres"),
            password=password,
            keepalives=1,
            keepalives_idle=300,
            keepalives_interval=30,
            keepalives_count=5,
        )
        conn.autocommit = True
        return conn

    def _is_connection_healthy(conn):
        """Check if connection is alive with a quick query."""
        if conn is None or conn.closed:
            return False
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:
            return False

    now = time.time()

    # Fast path: if connection exists and was used recently, skip health check
    if _db_conn is not None and not _db_conn.closed:
        if (now - _db_last_used) < _DB_HEALTH_CHECK_INTERVAL:
            _db_last_used = now
            return _db_conn

    # Slow path: check health if connection is old or missing
    if not _is_connection_healthy(_db_conn):
        if _db_conn is not None:
            try:
                _db_conn.close()
            except Exception:
                pass
            logger.info("[DB] Reconnecting - connection was stale")
        _db_conn = _create_connection()
        logger.info("[DB] New connection established")

    _db_last_used = now
    return _db_conn


def _find_paper_versions(oa_id: int, title: str) -> list[int]:
    """Find all OpenAlex IDs for different versions of the same paper.

    Uses the OpenAlex API to search by title and returns OA IDs for works
    whose titles are near-identical (handles minor preprint→published changes).
    """
    import httpx
    try:
        resp = httpx.get(
            "https://api.openalex.org/works",
            params={"search": title, "per_page": 5},
            timeout=3.0,
        )
        if resp.status_code != 200:
            return [oa_id]
        results = resp.json().get("results", [])
        ids = []
        title_words = set(title.lower().split())
        for w in results:
            w_title = (w.get("title") or "").lower().strip()
            w_words = set(w_title.split())
            # Jaccard similarity: papers often change 1-2 words between versions
            intersection = len(title_words & w_words)
            union = len(title_words | w_words)
            if union > 0 and intersection / union >= 0.85:
                w_id_str = (w.get("id") or "").rsplit("/", 1)[-1]
                if w_id_str.startswith("W"):
                    w_id_str = w_id_str[1:]
                try:
                    ids.append(int(w_id_str))
                except ValueError:
                    pass
        return ids if ids else [oa_id]
    except Exception:
        return [oa_id]


def _resolve_biomedrxiv_creds() -> tuple[str, str, str]:
    """Resolve host/password/user for the biomedrxiv-prod Cloud SQL instance.

    Checks explicit env vars first, falls back to parsing BIOMEDRXIV_DB_URL.
    """
    host = os.getenv("BIOMEDRXIV_DB_HOST")
    pw = os.getenv("BIOMEDRXIV_DB_PASSWORD")
    user = os.getenv("BIOMEDRXIV_DB_USER", "postgres")
    if host and pw:
        return host, pw, user
    db_url = os.getenv("BIOMEDRXIV_DB_URL", "")
    if db_url:
        from urllib.parse import urlparse, unquote, parse_qs
        parsed = urlparse(db_url)
        pw = unquote(parsed.password or "") if parsed.password else ""
        user = parsed.username or "postgres"
        qs = parse_qs(parsed.query)
        host = qs.get("host", [None])[0] or parsed.hostname or ""
        return host, pw, user
    return (
        os.getenv("BIOMEDRXIV_DB_HOST", "/cloudsql/gxl-prod:us-central1:biomedrxiv-prod"),
        "",
        "postgres",
    )


def _get_oa_papers_conn():
    """Connection to the openalex database on biomedrxiv-prod (papers + mapping)."""
    global _oa_papers_conn, _oa_papers_last_used
    import psycopg2

    now = time.time()
    if _oa_papers_conn is not None and not _oa_papers_conn.closed:
        if (now - _oa_papers_last_used) < _DB_HEALTH_CHECK_INTERVAL:
            _oa_papers_last_used = now
            return _oa_papers_conn

    def _healthy(conn):
        if conn is None or conn.closed:
            return False
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:
            return False

    if not _healthy(_oa_papers_conn):
        if _oa_papers_conn:
            try:
                _oa_papers_conn.close()
            except Exception:
                pass
        host, pw, user = _resolve_biomedrxiv_creds()
        _oa_papers_conn = psycopg2.connect(
            host=host, database="openalex", user=user, password=pw,
            keepalives=1, keepalives_idle=300, keepalives_interval=30, keepalives_count=5,
        )
        _oa_papers_conn.autocommit = True

    _oa_papers_last_used = now
    return _oa_papers_conn


def _get_oa_edges_conn():
    """Connection to the dedicated openalex instance (citation_edges)."""
    global _oa_edges_conn, _oa_edges_last_used
    import psycopg2

    now = time.time()
    if _oa_edges_conn is not None and not _oa_edges_conn.closed:
        if (now - _oa_edges_last_used) < _DB_HEALTH_CHECK_INTERVAL:
            _oa_edges_last_used = now
            return _oa_edges_conn

    def _healthy(conn):
        if conn is None or conn.closed:
            return False
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:
            return False

    if not _healthy(_oa_edges_conn):
        if _oa_edges_conn:
            try:
                _oa_edges_conn.close()
            except Exception:
                pass
        host = os.getenv("OA_EDGES_DB_HOST", "/cloudsql/gxl-prod:us-central1:openalex")
        _, fallback_pw, _ = _resolve_biomedrxiv_creds()
        pw = os.getenv("OA_EDGES_DB_PASSWORD", fallback_pw)
        _oa_edges_conn = psycopg2.connect(
            host=host, database="openalex", user="postgres", password=pw,
            keepalives=1, keepalives_idle=300, keepalives_interval=30, keepalives_count=5,
        )
        _oa_edges_conn.autocommit = True

    _oa_edges_last_used = now
    return _oa_edges_conn


def _get_es_client():
    """Get OpenSearch client for Papers (title+abstract BM25).

    Uses the shared ``OPENSEARCH_URL`` env var.  Returns the singleton
    ``OpenSearchClient`` from ``search_backends`` so every Papers call-site
    talks to the same OpenSearch cluster that the Papers MCP uses.
    """
    global _es_client
    if _es_client is None:
        from mcps.papers.servers.search_backends import get_opensearch_client

        _es_client = get_opensearch_client()
        if _es_client is not None:
            logger.info("Papers filesystem using OpenSearch client from search_backends")
        else:
            logger.warning(
                "OpenSearch not configured (set OPENSEARCH_URL)"
            )
    return _es_client


# =============================================================================
# Papers-specific PathParser
# =============================================================================


class PapersPathParser(PathParser):
    """Parse paths like /papers/{uuid}/sections/{name}.lines"""

    @property
    def root_name(self) -> str:
        return "papers"

    def parse(self, path: str) -> ParsedPath:
        """Parse a virtual path into components."""
        path = self.normalize(path)
        parts = [p for p in path.split("/") if p]

        if len(parts) == 0:
            return ParsedPath(type="root")

        if parts[0] != "papers":
            return ParsedPath(
                type="invalid",
                error=f"Unknown root: /{parts[0]}. Use /papers/",
            )

        if len(parts) == 1:
            return ParsedPath(type="documents_list")

        doc_id = parts[1]

        if len(parts) == 2:
            return ParsedPath(type="document", document_id=doc_id)

        subpath = parts[2]

        if subpath == "meta.json":
            return ParsedPath(type="file", document_id=doc_id, filename="meta.json")

        if subpath == "content.lines":
            return ParsedPath(type="file", document_id=doc_id, filename="content.lines")

        if subpath == "sections":
            if len(parts) == 3:
                return ParsedPath(type="sections_list", document_id=doc_id)
            section_file = parts[3]
            if section_file.endswith(".lines"):
                section_name = section_file[:-6]
                return ParsedPath(
                    type="section", document_id=doc_id, section=section_name
                )
            return ParsedPath(
                type="invalid", error=f"Invalid section file: {section_file}"
            )

        if subpath == "supplements":
            if len(parts) == 3:
                return ParsedPath(type="supplements_list", document_id=doc_id)
            filename = parts[3]
            if filename.endswith(".lines"):
                return ParsedPath(
                    type="supplement_text", document_id=doc_id, filename=filename
                )
            else:
                return ParsedPath(
                    type="supplement_file", document_id=doc_id, filename=filename
                )

        if subpath == "figures":
            if len(parts) == 3:
                return ParsedPath(type="figures_list", document_id=doc_id)
            filename = parts[3]
            return ParsedPath(type="figure_file", document_id=doc_id, filename=filename)

        # Treat any other subpath as a section/block_type filter
        return ParsedPath(type="document_section", document_id=doc_id, filter=subpath)


# =============================================================================
# Papers-specific DocumentStore
# =============================================================================


class PapersStore(DocumentStore):
    """PostgreSQL + Elasticsearch backend for Papers."""

    @staticmethod
    def _want_openalex_in_search(raw_sources: list | None) -> bool:
        if not OPENALEX_SEARCH_ENABLED:
            return False
        if not raw_sources:
            return True
        ls = [s.lower() for s in raw_sources]
        if "all" in ls:
            return True
        return "openalex" in ls

    @staticmethod
    def _month_year_from_pub_year(pub_year) -> str:
        if pub_year is None:
            return ""
        try:
            return f"June_{int(pub_year)}"
        except (TypeError, ValueError):
            return ""

    @staticmethod
    def _deduplicate_by_doi(results: list[dict]) -> list[dict]:
        """Deduplicate results by DOI, keeping only the most recent version.

        Papers with the same DOI (different versions) are deduplicated by
        preferring the one with the most recent month_year (e.g., "July_2024" > "April_2024").
        """
        if not results:
            return results

        # Month ordering for comparison
        month_order = {
            "January": 1,
            "February": 2,
            "March": 3,
            "April": 4,
            "May": 5,
            "June": 6,
            "July": 7,
            "August": 8,
            "September": 9,
            "October": 10,
            "November": 11,
            "December": 12,
        }

        def parse_month_year(my: str) -> tuple:
            """Parse 'July_2024' into (2024, 7) for comparison."""
            if not my or "_" not in my:
                return (0, 0)
            parts = my.split("_")
            if len(parts) != 2:
                return (0, 0)
            month_str, year_str = parts
            try:
                year = int(year_str)
                month = month_order.get(month_str, 0)
                return (year, month)
            except ValueError:
                return (0, 0)

        # Group by DOI
        doi_groups: dict[str, list[dict]] = {}
        no_doi_results = []

        for r in results:
            doi = r.get("doi")
            if doi:
                if doi not in doi_groups:
                    doi_groups[doi] = []
                doi_groups[doi].append(r)
            else:
                no_doi_results.append(r)

        # For each DOI, keep only the most recent
        deduped = []
        for doi, group in doi_groups.items():
            if len(group) == 1:
                deduped.append(group[0])
            else:
                # Prefer full_text over abstract_only, then most recent month_year.
                def _doi_sort_key(x: dict) -> tuple:
                    prefer_ft = 1 if x.get("text_access") != "abstract_only" else 0
                    my = parse_month_year(x.get("month_year", ""))
                    if my == (0, 0) and x.get("pub_year") is not None:
                        try:
                            my = (int(x["pub_year"]), 6)
                        except (TypeError, ValueError):
                            pass
                    return (prefer_ft, my)

                group.sort(key=_doi_sort_key, reverse=True)
                deduped.append(group[0])

        # Preserve original order based on first appearance of each DOI
        seen_dois = set()
        ordered_deduped = []
        for r in results:
            doi = r.get("doi")
            if doi:
                if doi not in seen_dois:
                    seen_dois.add(doi)
                    # Find the deduped version for this DOI
                    for d in deduped:
                        if d.get("doi") == doi:
                            ordered_deduped.append(d)
                            break
            else:
                ordered_deduped.append(r)

        return ordered_deduped

    @staticmethod
    def _parse_one_es_hit(hit: dict) -> dict | None:
        """Parse one ES hit into a paper dict (bio/medrxiv, PMC, or OpenAlex)."""
        src = hit.get("_source") or {}
        if src.get("oa_id") is not None:
            oa_id = src["oa_id"]
            paper_id = src.get("paper_id")
            snippet = (src.get("abstract") or "")[:600]
            base = {
                "score": hit.get("_score"),
                "pub_year": src.get("pub_year"),
                "doi": (src.get("doi") or "").strip(),
                "title": src.get("title") or "",
                "authors": src.get("first_author") or "",
                "month_year": PapersStore._month_year_from_pub_year(src.get("pub_year")),
                "abstract_snippet": snippet,
                "openalex_id": oa_id,
            }
            if paper_id:
                return {
                    **base,
                    "document_id": paper_id,
                    "source": src.get("source_name") or "openalex",
                    "text_access": "full_text",
                }
            return {
                **base,
                "document_id": f"oa_{oa_id}",
                "source": "openalex",
                "text_access": "abstract_only",
            }
        doc_id = src.get("document_id") or src.get("pmc_id")
        if not doc_id:
            return None
        is_abstract_only = str(src.get("source") or "").lower() == "abstract_only"
        snippet = (src.get("abstract") or src.get("abstract_text") or "")[:600]
        return {
            "document_id": doc_id,
            "source": src.get("source", "pmc" if src.get("pmc_id") else "biorxiv"),
            "score": hit.get("_score"),
            "pub_year": src.get("pub_year"),
            "doi": (src.get("doi") or "").strip() if src.get("doi") else "",
            "title": src.get("title") or "",
            "authors": src.get("authors") or "",
            "abstract_snippet": snippet,
            "text_access": "abstract_only" if is_abstract_only else "full_text",
        }

    @staticmethod
    def _parse_es_hits(response: dict) -> list[dict]:
        """Parse ES response into a list of paper dicts with document_id and source."""
        results = []
        for hit in response.get("hits", {}).get("hits", []):
            row = PapersStore._parse_one_es_hit(hit)
            if row:
                results.append(row)
        return results

    async def get_document(self, document_id: str) -> dict | None:
        """Fetch paper metadata."""
        uuid = resolve(document_id)
        conn = _get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT document_id::text, title, doi, source, authors, month_year, 
                       abstract_text, created_at
                FROM documents WHERE document_id::text = %s
            """,
                (uuid,),
            )
            row = cur.fetchone()

        if not row:
            return None

        return shorten_result({
            "document_id": row[0],
            "title": row[1],
            "doi": row[2],
            "source": row[3],
            "authors": row[4],
            "month_year": row[5],
            "abstract": row[6],
            "created_at": str(row[7]) if row[7] else None,
        })

    async def get_document_content(self, document_id: str) -> list[dict]:
        """Fetch all content blocks for a paper."""
        uuid = resolve(document_id)
        conn = _get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT line_number, content, section, block_type
                FROM content_blocks
                WHERE document_id::text = %s
                ORDER BY line_number
            """,
                (uuid,),
            )
            rows = cur.fetchall()

        return [
            {
                "line_number": row[0],
                "content": row[1],
                "section": row[2],
                "block_type": row[3],
            }
            for row in rows
        ]

    # Article types considered "research" for PMC filtering
    _PMC_RESEARCH_TYPES = [
        "research-article", "review-article", "case-report", "brief-report",
        "systematic-review", "data-paper", "methods-article",
        "rapid-communication", "discussion",
    ]

    def _get_search_indices(self, sources: list | None, es) -> list[str]:
        """Return OpenSearch indices (preprints, pmc, abstract_only)."""
        ls = [s.lower() for s in sources] if sources else []

        ao_available = ABSTRACT_ONLY_SEARCH_ENABLED and es is not None

        if not sources or "all" in ls:
            out = ["preprints", "pmc"]
            if ao_available:
                out.append(ABSTRACT_ONLY_OS_INDEX)
            return out
        indices = []
        if any(s in ls for s in ("biorxiv", "medrxiv")):
            indices.append("preprints")
        if "pmc" in ls:
            indices.append("pmc")
        if "openalex" in ls and ao_available:
            indices.append(ABSTRACT_ONLY_OS_INDEX)
        if "abstract_only" in ls and ao_available:
            indices.append(ABSTRACT_ONLY_OS_INDEX)
        return indices or ["preprints"]

    def _filters_for_search_index(
        self,
        idx: str,
        raw_sources: list | None,
        doc_indices: list[str],
        time_filters: list,
        pmc_article_filter: dict,
    ) -> list:
        """Filters for one OpenSearch index."""
        fl = list(time_filters)
        if idx == ABSTRACT_ONLY_OS_INDEX:
            return fl
        fl.append(pmc_article_filter)
        if not raw_sources:
            return fl
        ls = [s.lower() for s in raw_sources]
        if "pmc" in doc_indices:
            return fl
        bio_vals = [s for s in raw_sources if str(s).lower() in ("biorxiv", "medrxiv")]
        if bio_vals and idx == "preprints":
            fl.append({"terms": {"source": [b.lower() for b in bio_vals]}})
        return fl

    @staticmethod
    def _since_to_pub_year(since_str: str) -> int | None:
        """Convert since string like '2y', '6m', '30d' to a minimum pub_year."""
        import re as _re
        from datetime import datetime, timedelta
        m = _re.match(r"(\d+)([dmy])", since_str.strip().lower())
        if not m:
            return None
        val, unit = int(m.group(1)), m.group(2)
        now = datetime.now()
        if unit == "d":
            dt = now - timedelta(days=val)
        elif unit == "m":
            dt = now - timedelta(days=val * 30)
        else:
            dt = now - timedelta(days=val * 365)
        return dt.year

    async def search_documents(
        self,
        query: str = None,
        filters: dict = None,
        limit: int = 100,
    ) -> list[dict]:
        """Search Elasticsearch: bioRxiv/medRxiv, PMC, and OpenAlex (BM25 on title+abstract).

        OpenAlex hits use ``text_access``: ``full_text`` (mapped to PaperCLIP) or
        ``abstract_only`` (``oa_<id>``). Requires index ``OPENALEX_ES_INDEX``.
        Abstract-only corpus (``ABSTRACT_ONLY_ES_INDEX``) uses ``source=abstract_only``
        and ``text_access=abstract_only`` (no full text on disk).
        """
        es = _get_es_client()
        if not es:
            raise RuntimeError(
                "OpenSearch not available. Search disabled. "
                "Set OPENSEARCH_URL and restart."
            )

        if not filters:
            filters = {}

        search_mode = filters.get("search_mode", "any")

        must = []
        if query:
            if search_mode == "phrase":
                must.append({"bool": {"should": [
                    {"match_phrase": {"title": {"query": query, "boost": 3}}},
                    {"match_phrase": {"abstract_text": {"query": query, "boost": 2}}},
                    {"match_phrase": {"abstract": {"query": query, "boost": 2}}},
                    {"match_phrase": {"first_author": {"query": query, "boost": 1}}},
                ], "minimum_should_match": 1}})
            elif search_mode == "all":
                must.append({"multi_match": {
                    "query": query,
                    "fields": [
                        "title^3", "abstract_text^2", "abstract^2", "authors",
                        "first_author^1",
                    ],
                    "type": "cross_fields", "operator": "and",
                }})
            elif search_mode in ("50%", "75%"):
                must.append({"multi_match": {
                    "query": query,
                    "fields": [
                        "title^3", "abstract_text^2", "abstract^2", "authors",
                        "first_author^1",
                    ],
                    "type": "best_fields", "minimum_should_match": search_mode,
                }})
            else:
                must.append({"bool": {"should": [
                    {"match": {"title": {"query": query, "boost": 3}}},
                    {"match": {"abstract": {"query": query, "boost": 2}}},
                    {"match": {"abstract_text": {"query": query, "boost": 2}}},
                    {"match": {"authors": {"query": query, "boost": 1}}},
                    {"match": {"first_author": {"query": query, "boost": 1}}},
                ], "minimum_should_match": 1}})

        # Determine which indices to search
        raw_sources = filters.get("source")
        if isinstance(raw_sources, str):
            raw_sources = [raw_sources]
        doc_indices = self._get_search_indices(raw_sources, es)

        pmc_article_filter = {
            "bool": {
                "should": [
                    {"terms": {"article_type": self._PMC_RESEARCH_TYPES}},
                    {"term": {"source": "biorxiv"}},
                    {"term": {"source": "medrxiv"}},
                ],
                "minimum_should_match": 1,
            }
        }

        if not filters.get("since") and not filters.get("year") and not filters.get("all_time"):
            filters.setdefault("since", "2y")

        time_filters = []
        since = filters.get("since")
        if since:
            min_year = self._since_to_pub_year(since)
            if min_year:
                time_filters.append({"range": {"pub_year": {"gte": min_year}}})
        year = filters.get("year")
        if year:
            time_filters.append({"term": {"pub_year": int(year)}})

        sort_mode = filters.get("sort")
        sort_clause = None
        if sort_mode == "date":
            sort_clause = [{"pub_year": {"order": "desc", "missing": "_last"}}, "_score"]

        def _body_for_index(idx: str) -> dict:
            idx_filters = self._filters_for_search_index(
                idx, raw_sources, doc_indices, time_filters, pmc_article_filter
            )
            return {
                "query": {
                    "bool": {
                        "must": must if must else [{"match_all": {}}],
                        "filter": idx_filters,
                    }
                },
                "size": limit,
                "_source": True,
            }

        try:
            import asyncio

            if len(doc_indices) <= 1:
                idx0 = doc_indices[0]
                body = _body_for_index(idx0)
                if sort_clause:
                    body["sort"] = sort_clause
                response = await asyncio.to_thread(es.search, index=idx0, body=body)
                results = self._parse_es_hits(response)
                logger.info("[search] single-index %s: %d results", idx0, len(results))
            else:
                per_source = max(limit, 20)

                def _interleaved():
                    all_hits = {}
                    for idx in doc_indices:
                        b = _body_for_index(idx)
                        b["size"] = per_source
                        if sort_clause:
                            b["sort"] = sort_clause
                        try:
                            resp = es.search(index=idx, body=b)
                            all_hits[idx] = self._parse_es_hits(resp)
                        except Exception as exc:
                            logger.warning("ES search on %s failed: %s", idx, exc)
                            all_hits[idx] = []
                    merged = []
                    iters = {k: iter(v) for k, v in all_hits.items() if v}
                    while len(merged) < limit and iters:
                        exhausted = []
                        for k, it in iters.items():
                            r = next(it, None)
                            if r:
                                merged.append(r)
                            else:
                                exhausted.append(k)
                        for k in exhausted:
                            del iters[k]
                    return merged[:limit]

                results = await asyncio.to_thread(_interleaved)

            hydrated = await self._hydrate_results(results)
            return shorten_results(self._deduplicate_by_doi(hydrated))
        except Exception as e:
            logger.error(f"OpenSearch search failed: {e}")
            raise RuntimeError(f"OpenSearch search failed: {e}") from e

    async def _hydrate_results(self, results: list[dict]) -> list[dict]:
        """Enrich minimal search results (doc_id + score) with full metadata from PG."""
        import asyncio, re as _re

        if not results:
            return results

        bio_ids = []
        for r in results:
            did = r.get("document_id", "")
            if _re.match(r"^PMC\d+$", did, _re.IGNORECASE):
                continue
            if isinstance(did, str) and did.lower().startswith("oa_"):
                continue
            if r.get("source") == "abstract_only":
                continue
            bio_ids.append(did)

        pmc_ids = [
            r["document_id"]
            for r in results
            if _re.match(r"^PMC\d+$", r.get("document_id", ""), _re.IGNORECASE)
        ]

        bio_meta: dict = {}
        pmc_meta: dict = {}
        oa_meta: dict[int, dict] = {}

        def _fetch_bio(ids: list):
            if not ids:
                return
            conn = _get_db_connection()
            resolved = [resolve(d) for d in ids]
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT document_id::text, title, doi, source, authors, month_year,
                              abstract_text, created_at
                       FROM documents WHERE document_id::text = ANY(%s)""",
                    (resolved,),
                )
                for row in cur.fetchall():
                    bio_meta[row[0]] = {
                        "document_id": row[0],
                        "title": row[1],
                        "doi": row[2],
                        "source": row[3],
                        "authors": row[4],
                        "month_year": row[5],
                        "abstract_text": row[6],
                    }

        def _fetch_pmc():
            if not pmc_ids:
                return
            try:
                from apps.tools.src.mcps.papers.servers.papers_server import _get_papers_module

                module = _get_papers_module()
                if not module:
                    return
                pmc_conn = module._get_pmc_db_connection()
                with pmc_conn.cursor() as cur:
                    cur.execute(
                        """SELECT pmc_id, title, doi, authors, journal_title,
                                  pub_year, pub_date, article_type, source, abstract_text, tldr
                           FROM documents WHERE pmc_id = ANY(%s)""",
                        (pmc_ids,),
                    )
                    for row in cur.fetchall():
                        pmc_meta[row[0]] = {
                            "document_id": row[0],
                            "title": row[1],
                            "doi": row[2],
                            "authors": row[3],
                            "journal": row[4],
                            "pub_year": row[5],
                            "pub_date": str(row[6]) if row[6] else None,
                            "article_type": row[7],
                            "source": row[8] or "pmc",
                            "abstract_text": row[9],
                            "tldr": row[10],
                        }
            except Exception as e:
                logger.warning("PMC hydration failed: %s", e)

        def _fetch_oa():
            oa_numeric: list[int] = []
            for r in results:
                did = r.get("document_id", "")
                if isinstance(did, str) and did.lower().startswith("oa_"):
                    try:
                        oa_numeric.append(int(did[3:], 10))
                    except ValueError:
                        pass
            if not oa_numeric:
                return
            try:
                conn = _get_oa_papers_conn()
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT p.oa_id, p.title, p.doi, p.first_author, p.pub_year, p.abstract,
                               m.paper_id
                        FROM papers p
                        LEFT JOIN oa_paper_mapping m ON m.oa_id = p.oa_id
                        WHERE p.oa_id = ANY(%s)
                        """,
                        (oa_numeric,),
                    )
                    for row in cur.fetchall():
                        oid, title, doi, fa, py, abstract, paper_id = row
                        snip = (abstract or "")[:600]
                        my = PapersStore._month_year_from_pub_year(py)
                        entry = {
                            "title": title or "",
                            "doi": (doi or "").strip(),
                            "authors": fa or "",
                            "month_year": my,
                            "abstract_snippet": snip,
                        }
                        if paper_id:
                            entry["paper_id"] = paper_id
                        oa_meta[int(oid)] = entry
            except Exception as e:
                logger.warning("[search] OpenAlex PG hydrate failed: %s", e)

        await asyncio.gather(
            asyncio.to_thread(_fetch_pmc),
            asyncio.to_thread(_fetch_oa),
        )

        mapped_bio = [m["paper_id"] for m in oa_meta.values() if m.get("paper_id")]
        bio_ids_all = list(dict.fromkeys(bio_ids + mapped_bio))
        await asyncio.to_thread(_fetch_bio, bio_ids_all)

        hydrated = []
        for r in results:
            doc_id = r.get("document_id", "")
            if isinstance(doc_id, str) and doc_id.lower().startswith("oa_"):
                try:
                    oid = int(doc_id[3:], 10)
                except ValueError:
                    r.setdefault("text_access", "abstract_only")
                    hydrated.append(r)
                    continue
                meta = oa_meta.get(oid)
                if meta:
                    merged = {**r, **{k: v for k, v in meta.items() if k != "paper_id"}}
                    if meta.get("paper_id"):
                        merged["document_id"] = meta["paper_id"]
                        merged["text_access"] = "full_text"
                        full_id = resolve(meta["paper_id"])
                        extra = bio_meta.get(full_id)
                        if extra:
                            merged = {**merged, **extra}
                    else:
                        merged["text_access"] = "abstract_only"
                    hydrated.append(merged)
                else:
                    r.setdefault("text_access", "abstract_only")
                    hydrated.append(r)
                continue

            full_id = (
                resolve(doc_id)
                if doc_id and not doc_id.upper().startswith("PMC")
                else doc_id
            )
            meta = bio_meta.get(full_id) or pmc_meta.get(doc_id)
            if meta:
                merged = {**r, **meta}
                merged.setdefault("text_access", "full_text")
                hydrated.append(merged)
            else:
                if r.get("source") == "abstract_only":
                    r.setdefault("text_access", "abstract_only")
                else:
                    r.setdefault("text_access", "full_text")
                hydrated.append(r)
        return hydrated

    async def _search_postgres(
        self,
        query: str = None,
        filters: dict = None,
        limit: int = 100,
    ) -> list[dict]:
        """Fallback PostgreSQL search."""
        conn = _get_db_connection()
        conditions = []
        params = []

        if query:
            conditions.append(
                "(title ILIKE %s OR abstract_text ILIKE %s OR authors ILIKE %s)"
            )
            pattern = f"%{query}%"
            params.extend([pattern, pattern, pattern])

        if filters:
            if "source" in filters:
                conditions.append("source = %s")
                params.append(filters["source"])

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with conn.cursor() as cur:
            # Fetch extra results to account for deduplication
            cur.execute(
                f"""
                SELECT document_id::text, title, doi, authors, month_year, source
                FROM documents
                {where}
                ORDER BY created_at DESC
                LIMIT %s
            """,
                params + [limit * 2],  # Fetch extra for deduplication
            )
            rows = cur.fetchall()

        results = [
            {
                "document_id": row[0],
                "title": row[1],
                "doi": row[2],
                "authors": row[3],
                "month_year": row[4],
                "source": row[5],
                "path": f"/papers/{row[0]}/",
            }
            for row in rows
        ]

        # Deduplicate by DOI, keeping most recent version, then limit
        return shorten_results(self._deduplicate_by_doi(results)[:limit])

    async def grep_content(
        self,
        regex: str,
        document_ids: list[str] = None,
        section_filter: str = None,
        limit: int = 50,
    ) -> list[dict]:
        """Regex search on paper content."""
        conn = _get_db_connection()

        conditions = ["content ~* %s"]
        params = [regex]

        if document_ids:
            resolved_ids = [resolve(d) for d in document_ids]
            conditions.append("document_id::text = ANY(%s)")
            params.append(resolved_ids)

        if section_filter:
            conditions.append("(section ILIKE %s OR block_type ILIKE %s)")
            params.extend([f"%{section_filter}%", f"%{section_filter}%"])

        with conn.cursor() as cur:
            # Set timeout
            cur.execute(f"SET statement_timeout = '{GREP_TIMEOUT_SECONDS * 1000}'")

            cur.execute(
                f"""
                SELECT document_id::text, line_number, content, section, block_type
                FROM content_blocks
                WHERE {' AND '.join(conditions)}
                ORDER BY document_id, line_number
                LIMIT %s
            """,
                params + [limit * 2],
            )
            rows = cur.fetchall()

            cur.execute("RESET statement_timeout")

        results = []
        for row in rows:
            content = row[2] or ""
            try:
                match_obj = re.search(regex, content, re.IGNORECASE)
                match_text = match_obj.group(0) if match_obj else content[:100]
            except re.error:
                match_text = content[:100]

            results.append(
                {
                    "document_id": shorten(row[0]),
                    "line_number": row[1],
                    "content": content[:300],
                    "match": match_text,
                    "section": row[3],
                    "block_type": row[4],
                }
            )

        return results

    async def batch_get_documents(self, document_ids: list[str]) -> dict[str, dict]:
        """Batch fetch multiple papers - optimized with single queries."""
        if not document_ids:
            return {}

        resolved_ids = [resolve(d) for d in document_ids]
        conn = _get_db_connection()
        result = {}

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT document_id::text, title, doi, source, authors, month_year
                FROM documents 
                WHERE document_id::text = ANY(%s)
            """,
                (resolved_ids,),
            )
            for row in cur.fetchall():
                doc_id = row[0]
                short_id = shorten(doc_id, row[3])
                result[doc_id] = {
                    "metadata": {
                        "document_id": short_id,
                        "title": row[1],
                        "doi": row[2],
                        "source": row[3],
                        "authors": row[4],
                        "month_year": row[5],
                    },
                    "blocks": [],
                }

            # Query 2: Fetch ALL content blocks at once
            cur.execute(
                """
                SELECT document_id::text, line_number, content, section, block_type
                FROM content_blocks
                WHERE document_id::text = ANY(%s)
                ORDER BY document_id, line_number
            """,
                (resolved_ids,),
            )
            for row in cur.fetchall():
                doc_id = row[0]
                if doc_id in result:
                    result[doc_id]["blocks"].append(
                        {
                            "line_number": row[1],
                            "content": row[2],
                            "section": row[3],
                            "block_type": row[4],
                        }
                    )

        return result

    async def batch_get_metadata(self, document_ids: list[str]) -> dict[str, dict]:
        """Batch fetch metadata only (no content blocks) - single query."""
        if not document_ids:
            return {}

        conn = _get_db_connection()
        result = {}

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT document_id::text, title, doi, source, authors, month_year
                FROM documents
                WHERE document_id::text = ANY(%s)
            """,
                (document_ids,),
            )
            for row in cur.fetchall():
                doc_id = row[0]
                result[doc_id] = {
                    "metadata": {
                        "document_id": doc_id,
                        "title": row[1],
                        "doi": row[2],
                        "source": row[3],
                        "authors": row[4],
                        "month_year": row[5],
                    },
                }

        return result


# =============================================================================
# Papers Module (extends VirtualFilesystemModule)
# =============================================================================


class PapersModule(VirtualFilesystemModule):
    """Papers research tools using the virtual filesystem framework."""

    def __init__(self):
        super().__init__()
        self._results_registry = None
        self._parallel_executor = None
        self._terminals: dict[str, VirtualTerminal] = {}  # Per-session terminals
        self._setup_tools()

    def get_name(self) -> str:
        return "papers"

    def get_description(self) -> str:
        return (
            "Research tools for 450K+ bioRxiv/medRxiv preprints with parallel analysis"
        )

    def get_path_parser(self) -> PathParser:
        return PapersPathParser()

    def get_document_store(self) -> DocumentStore:
        return PapersStore()

    @property
    def results_registry(self) -> ResultsRegistry:
        if self._results_registry is None:
            self._results_registry = ResultsRegistry(self.session_manager)
        return self._results_registry

    def _get_document_contents(self, doc: dict) -> list[str]:
        """Override to show Papers-specific structure."""
        return [
            "meta.json",
            "content.lines",
            "sections/",
            "supplements/",
            "figures/",
        ]

    def _setup_tools(self):
        """Define filesystem + parallel tools.

        Note: Basic filesystem operations (ls, cat, head, tail, find, grep) are
        consolidated into bash for a cleaner interface.
        """
        from collections.abc import Callable

        tools_def = [
            # papers_stat
            {
                "name": "papers_stat",
                "description": "Get metadata/stats about a file or folder.",
                "handler": self._stat,
                "schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to stat"},
                    },
                    "required": ["path"],
                },
            },
            # papers_map (parallel execution across papers)
            {
                "name": "papers_map",
                "description": """Execute parallel paper exploration tasks.

Each paper is assigned a dedicated reader agent with full tool access (grep, cat, etc.)
that explores the paper and extracts the requested information.

INPUT:
1. FROM RESULTS: papers_map(from_results="s_abc", query="...")
2. EXPLICIT: papers_map(tasks=[{"path": "/papers/uuid/", "query": "..."}])

OUTPUT:
- output_schema: Forces JSON output matching schema
  Example: output_schema={"method": "string", "accuracy": "number"}""",
                "handler": self._parallel,
                "schema": {
                    "type": "object",
                    "properties": {
                        "from_results": {
                            "type": "string",
                            "description": "search_id (s_ prefix)",
                        },
                        "tasks": {
                            "type": "array",
                            "description": "Explicit tasks",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "query": {"type": "string"},
                                },
                            },
                        },
                        "limit": {"type": "number", "description": "Max papers"},
                        "query": {"type": "string", "description": "Query for all"},
                        "output_schema": {
                            "type": "object",
                            "description": "JSON schema",
                        },
                        "max_concurrent": {
                            "type": "number",
                            "description": "Max parallel (default: 25)",
                        },
                        "include_rollouts": {
                            "type": "boolean",
                            "description": "Include per-paper subagent rollouts for debugging/UI",
                        },
                    },
                },
            },
            # papers_reduce
            {
                "name": "papers_reduce",
                "description": """Synthesize and summarize results from papers_map (reduce phase).

STRATEGIES:
1. summarize - LLM synthesizes outputs into a cohesive narrative
2. table - Structure outputs into a comparison table  
3. themes - Extract common themes/patterns with examples
4. consensus - Find agreement/disagreement across papers
5. bullet_points - Distill key findings into bullet points
6. extract - Extract specific fields from structured outputs

EXAMPLES:
- papers_reduce(from_map="m_abc", strategy="summarize")
- papers_reduce(from_map="m_abc", strategy="table", columns=["method", "accuracy"])
- papers_reduce(from_map="m_abc", strategy="themes")

ARTIFACT OUTPUT:
Every reduction is saved as a citable artifact with artifact_id.""",
                "handler": self._reduce,
                "schema": {
                    "type": "object",
                    "properties": {
                        "from_map": {
                            "type": "string",
                            "description": "map_id from papers_map",
                        },
                        "from_parallel": {
                            "type": "string",
                            "description": "Legacy alias for from_map",
                        },
                        "from_results": {
                            "type": "string",
                            "description": "search_id (s_ prefix) from search",
                        },
                        "strategy": {
                            "type": "string",
                            "enum": [
                                "summarize",
                                "table",
                                "themes",
                                "consensus",
                                "bullet_points",
                                "extract",
                            ],
                        },
                        "question": {"type": "string", "description": "Focus question"},
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "fields": {"type": "array", "items": {"type": "string"}},
                        "max_items": {"type": "number"},
                    },
                    "required": ["strategy"],
                },
            },
            # papers_ask_image
            {
                "name": "papers_ask_image",
                "description": """Analyze a figure/image from a paper using a vision model.

WORKFLOW:
1. Use papers_ls /papers/{uuid}/figures/ to list available figures
2. Use papers_ask_image to analyze a specific figure

Example:
  papers_ask_image(path="/papers/PMC12345/figures/fig1.jpg", question="What does this figure show?")""",
                "handler": self._ask_image,
                "schema": {
                    "type": "object",
                    "properties": {
                        "document_id": {"type": "string", "description": "Paper UUID (e.g., 'PMC12345' or 'bio_abc123')"},
                        "figure_id": {
                            "type": "string",
                            "description": "Figure filename (e.g., 'fig1.jpg')",
                        },
                        "question": {
                            "type": "string",
                            "description": "Question about the figure",
                        },
                    },
                    "required": ["document_id", "figure_id", "question"],
                },
            },
            # papers_get_citation
            {
                "name": "papers_get_citation",
                "description": """Get citation info for a specific line number in a paper or supplement.

Use this when you need to cite content. Returns citation_info needed for proper citation JSON.

WORKFLOW:
1. Read content with papers_cat or bash (shows 'LINE: content' format)
2. Call papers_get_citation(document_id, line_number, supplement_filename?)
3. Use returned info to build your citation

EXAMPLES:
  papers_get_citation(document_id="abc123", line_number=42)  # main content
  papers_get_citation(document_id="abc123", line_number=5, supplement_filename="613278_file03.content.md.lines")

Returns: {source_type, source_path, xml_id, xpath, section, block_type, ...}""",
                "handler": self._get_citation,
                "schema": {
                    "type": "object",
                    "properties": {
                        "document_id": {"type": "string", "description": "Paper UUID"},
                        "line_number": {
                            "type": "integer",
                            "description": "Line number from content",
                        },
                        "supplement_filename": {
                            "type": "string",
                            "description": "Optional: supplement filename (e.g. '613278_file03.content.md.lines') for citing supplement content",
                        },
                    },
                    "required": ["document_id", "line_number"],
                },
            },
            # === TERMINAL ===
            {
                "name": "bash",
                "description": """Unix shell on the Papers virtual filesystem. Use standard shell commands.

FILESYSTEM:
  /papers/{uuid}/              # Paper directory
  /papers/{uuid}/meta.json     # Metadata (title, authors, doi)
  /papers/{uuid}/content.lines # Full text
  /papers/{uuid}/sections/     # Sections (Methods.lines, etc.)
  /papers/{uuid}/figures/      # Figures (use ask_image /papers/{uuid}/figures/<file> to analyze)
  /papers/{uuid}/supplements/  # Supplement PDFs as text

SPECIAL COMMANDS:
  search QUERY       Find papers (semantic search, 25 results)
  search -r PATTERN  Regex search across all papers
  search -e "phrase" Exact phrase match
  cite LINE          Get citation info for a line number

RESTRICTED (not allowed):
  rm, mv, cp, chmod, sudo, curl, wget, ssh, eval, exec

OUTPUT NOTE:
  All .lines files output "LINE: content" format for citation tracking.

Returns: {stdout, stderr, exit_code, cwd}""",
                "handler": self._shell,
                "schema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute",
                        },
                    },
                    "required": ["command"],
                },
            },
        ]

        # Build tools and handlers (wrapped with TextContent)
        for tool_def in tools_def:
            tool = Tool(
                name=tool_def["name"],
                description=tool_def["description"],
                inputSchema=tool_def["schema"],
            )
            self._tools.append(tool)
            self._handlers[tool_def["name"]] = self._create_handler(tool_def["handler"])

        logger.info(f"Registered {len(self._tools)} filesystem-v2 tools")

    def _create_handler(self, func):
        """Wrap handler with error handling and TextContent formatting."""
        import inspect
        sig = inspect.signature(func)
        _accepts_agent_id = (
            'agent_id' in sig.parameters
            or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        )

        async def handler(arguments: dict, session_id: str = "default", **kwargs):
            try:
                extra = {}
                if _accepts_agent_id and "agent_id" in kwargs:
                    extra["agent_id"] = kwargs["agent_id"]
                result = await func(session_id=session_id, **extra, **arguments)
                return [
                    TextContent(
                        type="text", text=json.dumps(result, indent=2, default=str)
                    )
                ]
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": str(e), "arguments": arguments},
                            indent=2,
                        ),
                    )
                ]

        return handler

    def get_tools(self) -> list[Tool]:
        return self._tools

    # =========================================================================
    # Tool Implementations
    # =========================================================================

    async def _ls(
        self,
        path: str,
        query: str = None,
        limit: int = 20,
        session_id: str = "default",
    ) -> dict:
        """List contents of a virtual path - handles all path types."""
        start_time = time.perf_counter()
        parsed = self.path_parser.parse(path)

        if parsed.error:
            return {"error": parsed.error, "path": path}

        conn = _get_db_connection()

        if parsed.type == "root":
            return {
                "path": "/",
                "contents": ["papers/"],
                "hint": "Use /papers/ to browse papers",
            }

        if parsed.type == "documents_list":
            # Get total count
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM documents")
                total_count = cur.fetchone()[0]

            if query:
                result = await self._find(
                    query=query, limit=limit, session_id=session_id
                )
                result["path"] = "/papers/"
                result["total_papers"] = total_count
                return result

            with conn.cursor() as cur:
                cur.execute(
                    "SELECT document_id, title, doi, source FROM documents "
                    "ORDER BY created_at DESC LIMIT %s",
                    (min(limit, 20),),
                )
                rows = cur.fetchall()

            from .short_ids import shorten

            return {
                "path": "/papers/",
                "total_papers": total_count,
                "showing": len(rows),
                "contents": [
                    {
                        "path": f"/papers/{shorten(str(r[0]), r[3])}/",
                        "title": r[1][:80] if r[1] else None,
                        "doi": r[2],
                        "source": r[3],
                    }
                    for r in rows
                ],
                "hint": f"Showing {len(rows)} of {total_count:,} papers. Use papers_find to search, or cd /papers/UUID/ to explore a specific paper.",
                "time_ms": round((time.perf_counter() - start_time) * 1000),
            }

        if parsed.type == "document":
            uuid = parsed.document_id
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT title, doi, source FROM documents WHERE document_id::text = %s",
                    (uuid,),
                )
                doc = cur.fetchone()

            if not doc:
                return {"error": f"Paper not found: {uuid}"}

            with conn.cursor() as cur:
                cur.execute(
                    """SELECT COUNT(*) as total_lines,
                              COUNT(DISTINCT section) as section_count,
                              COUNT(CASE WHEN citation_info->>'source_type' LIKE '%%supplement%%' THEN 1 END) as has_supplements,
                              COUNT(CASE WHEN block_type = 'figure' THEN 1 END) as figure_count
                       FROM content_blocks WHERE document_id = %s""",
                    (uuid,),
                )
                stats = cur.fetchone()

            contents = ["meta.json", f"content.lines  ({stats[0]} lines)", "sections/"]
            if stats[2]:
                contents.append("supplements/")
            if stats[3]:
                contents.append("figures/")

            return {
                "path": f"/papers/{uuid}/",
                "title": doc[0],
                "doi": doc[1],
                "source": doc[2],
                "contents": contents,
                "time_ms": round((time.perf_counter() - start_time) * 1000),
            }

        if parsed.type == "sections_list":
            uuid = parsed.document_id
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT section, COUNT(*) as lines
                       FROM content_blocks WHERE document_id = %s AND section IS NOT NULL
                       GROUP BY section ORDER BY MIN(line_number)""",
                    (uuid,),
                )
                rows = cur.fetchall()

            return {
                "path": f"/papers/{uuid}/sections/",
                "contents": [
                    {"name": f"{r[0]}.lines", "lines": r[1]} for r in rows if r[0]
                ],
                "time_ms": round((time.perf_counter() - start_time) * 1000),
            }

        if parsed.type == "supplements_list":
            uuid = parsed.document_id
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT DISTINCT 
                           citation_info->>'source_path' as source_path,
                           citation_info->>'source_type' as source_type,
                           COUNT(*) as lines
                       FROM content_blocks 
                       WHERE document_id = %s 
                       AND citation_info->>'source_type' LIKE '%%supplement%%'
                       GROUP BY citation_info->>'source_path', citation_info->>'source_type'""",
                    (uuid,),
                )
                rows = cur.fetchall()

            contents = []
            seen_images = set()

            for row in rows:
                source_path = row[0] or ""
                source_type = row[1] or ""
                lines = row[2]
                filename = source_path.split("/")[-1] if source_path else "unknown"
                if "pdf" in source_type.lower():
                    contents.append(
                        {
                            "name": filename,
                            "type": "pdf",
                            "lines": lines,
                            "text_file": f"{filename}.lines",
                        }
                    )
                else:
                    contents.append({"name": filename, "lines": lines})

            # Also extract image references from supplement content
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT content, citation_info->>'source_path' as source_path
                       FROM content_blocks 
                       WHERE document_id = %s 
                       AND citation_info->>'source_type' LIKE '%%supplement%%'
                       AND content LIKE '![]%%'""",
                    (uuid,),
                )
                img_rows = cur.fetchall()

            for row in img_rows:
                content = row[0] or ""
                source_path = row[1] or ""
                # Extract image filename from ![](filename)
                match = re.search(r"!\[\]\(([^)]+)\)", content)
                if match:
                    img_filename = match.group(1)
                    if img_filename not in seen_images:
                        seen_images.add(img_filename)
                        # Get parent supplement name
                        parent = (
                            source_path.split("/")[-1] if source_path else "supplement"
                        )
                        contents.append(
                            {
                                "name": img_filename,
                                "type": "image",
                                "parent": parent,
                            }
                        )

            return {
                "path": f"/papers/{uuid}/supplements/",
                "count": len(contents),
                "contents": contents,
                "time_ms": round((time.perf_counter() - start_time) * 1000),
            }

        if parsed.type == "figures_list":
            uuid = parsed.document_id
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT citation_info->>'xml_id' as figure_id,
                              content,
                              citation_info->>'graphic' as graphic
                       FROM content_blocks 
                       WHERE document_id = %s AND block_type = 'figure'
                       ORDER BY line_number""",
                    (uuid,),
                )
                rows = cur.fetchall()

            contents = []
            for row in rows:
                figure_id = row[0]
                content = row[1] or ""
                graphic = row[2]
                if graphic:
                    # Use actual graphic filename as the name (like a real filesystem)
                    contents.append(
                        {
                            "name": graphic,
                            "type": "image",
                            "figure_id": figure_id,
                            "caption": (
                                content[:80] + "..." if len(content) > 80 else content
                            ),
                        }
                    )
                elif figure_id:
                    contents.append(
                        {
                            "name": figure_id,
                            "type": "xml_figure",
                            "caption": content[:80],
                        }
                    )
                elif content.startswith("![]("):
                    match = re.search(r"!\[\]\(([^)]+)\)", content)
                    if match:
                        contents.append(
                            {"name": match.group(1), "type": "supplement_image"}
                        )

            return {
                "path": f"/papers/{uuid}/figures/",
                "count": len(contents),
                "contents": contents,
                "time_ms": round((time.perf_counter() - start_time) * 1000),
            }

        if parsed.type == "figure_file":
            # Show info about a specific figure - match by graphic filename OR figure_id
            uuid = parsed.document_id
            filename = parsed.filename
            filename_base = (
                filename.replace(".tif", "").replace(".tiff", "")
                .replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
                .replace(".gif", "")
            )
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT content, 
                              citation_info->>'graphic' as graphic,
                              citation_info->>'xml_id' as xml_id,
                              line_number
                       FROM content_blocks 
                       WHERE document_id = %s 
                       AND block_type = 'figure'
                       AND (citation_info->>'graphic' = %s 
                            OR citation_info->>'xml_id' = %s 
                            OR citation_info->>'xml_id' ILIKE %s
                            OR citation_info->>'graphic' ILIKE %s)
                       LIMIT 1""",
                    (
                        uuid,
                        filename,
                        filename,
                        f"%{filename_base}%",
                        f"%{filename_base}%",
                    ),
                )
                row = cur.fetchone()

            if not row:
                return {
                    "error": f"Figure not found: {filename}",
                    "hint": f"Use 'ls /papers/{uuid}/figures/' to see available figures",
                }

            content, graphic, xml_id, line_number = row
            return {
                "path": f"/papers/{uuid}/figures/{graphic or filename}",
                "type": "figure",
                "figure_id": xml_id,
                "graphic": graphic,
                "caption": content[:200] + "..." if len(content) > 200 else content,
                "line_number": line_number,
                "hint": f"Use ask_image /papers/{uuid}/figures/{graphic or xml_id} 'What does this figure show?' to analyze this figure",
                "time_ms": round((time.perf_counter() - start_time) * 1000),
            }

        # Supplement file (image) - show info
        if parsed.type == "supplement_file":
            filename = parsed.filename
            if filename.endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif")):
                return {
                    "path": f"/papers/{parsed.document_id}/supplements/{filename}",
                    "type": "image",
                    "filename": filename,
                    "hint": f"Use ask_image /papers/{parsed.document_id}/supplements/{filename} 'What does this figure show?' to analyze this image",
                }

        return {"error": f"Cannot list path: {path}", "path_type": parsed.type}

    async def _cat(
        self,
        path: str,
        start: int = None,
        end: int = None,
        session_id: str = "default",
    ) -> dict:
        """Read file contents - handles all file types."""
        start_time = time.perf_counter()
        parsed = self.path_parser.parse(path)

        if parsed.error:
            return {"error": parsed.error, "path": path}

        conn = _get_db_connection()

        # meta.json
        if parsed.type == "file" and parsed.filename == "meta.json":
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT document_id, title, doi, source, authors, month_year, abstract_text FROM documents WHERE document_id::text = %s",
                    (parsed.document_id,),
                )
                row = cur.fetchone()
            if not row:
                return {"error": f"Document not found: {parsed.document_id}"}
            return {
                "path": path,
                "type": "json",
                "content": {
                    "document_id": row[0],
                    "title": row[1],
                    "doi": row[2],
                    "source": row[3],
                    "authors": row[4],
                    "month_year": row[5],
                    "abstract": row[6],
                },
                "time_ms": round((time.perf_counter() - start_time) * 1000),
            }

        # content.lines - full document
        if parsed.type == "file" and parsed.filename == "content.lines":
            return await self._read_lines(parsed.document_id, start, end, start_time)

        # Section file
        if parsed.type == "section":
            return await self._read_lines(
                parsed.document_id, start, end, start_time, section=parsed.section
            )

        # Supplement text
        if parsed.type == "supplement_text":
            return await self._read_supplement_lines(
                parsed.document_id, parsed.filename, start, end, start_time
            )

        # Figure file — generate a signed download URL when available
        if parsed.type == "figure_file":
            filename = parsed.filename
            resolved = self._resolve_figure_gcs_path(parsed.document_id, filename)
            if "gcs_path" in resolved:
                from .tools import generate_signed_download_url
                url = generate_signed_download_url(resolved["gcs_path"])
                if url:
                    return {
                        "type": "binary",
                        "download_url": url,
                        "filename": resolved.get("filename", filename),
                        "caption": resolved.get("caption", ""),
                        "hint": "Redirect to a file to save: paperclip cat /papers/<id>/figures/<filename> > <filename>",
                        "time_ms": round((time.perf_counter() - start_time) * 1000),
                    }
            return {
                "type": "binary",
                "error": f"Cannot read image file: {filename}",
                "hint": (
                    f"To save the image, redirect to a file:\n"
                    f"  paperclip cat /papers/{parsed.document_id}/figures/{filename} > {filename}\n"
                    f"To analyze with a vision model instead:\n"
                    f'  paperclip ask-image /papers/{parsed.document_id}/figures/{filename} "What does this figure show?"'
                ),
            }

        # Supplement image file — same signed-URL path
        if parsed.type == "supplement_file":
            filename = parsed.filename
            if filename.endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif")):
                resolved = self._resolve_figure_gcs_path(parsed.document_id, filename)
                if "gcs_path" in resolved:
                    from .tools import generate_signed_download_url
                    url = generate_signed_download_url(resolved["gcs_path"])
                    if url:
                        return {
                            "type": "binary",
                            "download_url": url,
                            "filename": resolved.get("filename", filename),
                            "time_ms": round((time.perf_counter() - start_time) * 1000),
                        }
                return {
                    "type": "binary",
                    "error": f"Cannot read image file: {filename}",
                    "hint": (
                        f"To save, redirect to a file:\n"
                        f"  paperclip cat /papers/{parsed.document_id}/supplements/{filename} > {filename}\n"
                        f"To analyze with a vision model:\n"
                        f'  paperclip ask-image /papers/{parsed.document_id}/supplements/{filename} "describe"'
                    ),
                }

        return {"error": f"Cannot read path: {path}"}

    async def _read_lines(
        self,
        document_id: str,
        start: int = None,
        end: int = None,
        start_time: float = None,
        section: str = None,
    ) -> dict:
        """Read content blocks as lines."""
        start_time = start_time or time.perf_counter()

        try:
            conn = _get_db_connection()
        except Exception as e:
            return {
                "error": f"Database connection error: {e}",
                "error_type": "connection",
            }

        # Get total line count first (for [N lines above] display)
        count_query = "SELECT COUNT(*) FROM content_blocks WHERE document_id = %s"
        count_params = [document_id]
        if section:
            count_query += " AND section = %s"
            count_params.append(section)

        # Include id for citation support (block_id)
        query = (
            "SELECT id, line_number, content FROM content_blocks WHERE document_id = %s"
        )
        params = [document_id]

        if section:
            query += " AND section = %s"
            params.append(section)

        if start:
            query += " AND line_number >= %s"
            params.append(start)
        if end:
            query += " AND line_number <= %s"
            params.append(end)

        query += " ORDER BY line_number"

        try:
            with conn.cursor() as cur:
                cur.execute(count_query, count_params)
                total_lines = cur.fetchone()[0]

                cur.execute(query, params)
                rows = cur.fetchall()
        except Exception as e:
            error_str = str(e)
            if any(
                x in error_str.lower() for x in ["ssl", "connection", "timeout", "eof"]
            ):
                return {
                    "error": f"Database connection lost: {e}",
                    "error_type": "connection",
                }
            return {"error": f"Database query error: {e}", "error_type": "query"}

        if not rows:
            return {
                "error": f"No content found for {document_id}"
                + (f" section {section}" if section else ""),
                "error_type": "not_found",
            }

        return {
            "document_id": document_id,
            "section": section,
            # Include block_id (database id) for citations
            # Add +1 to line numbers since DB is 0-indexed but we display 1-indexed
            "lines": [
                {"block_id": r[0], "line": r[1] + 1, "content": r[2]} for r in rows
            ],
            "count": len(rows),
            "total_lines": total_lines,
            "time_ms": round((time.perf_counter() - start_time) * 1000),
        }

    async def _read_supplement_lines(
        self,
        document_id: str,
        filename: str,
        start: int = None,
        end: int = None,
        start_time: float = None,
    ) -> dict:
        """Read supplement content by filename."""
        conn = _get_db_connection()
        start_time = start_time or time.perf_counter()

        # Match by source_path ending with filename - include id for citations
        query = """
            SELECT id, line_number, content 
            FROM content_blocks 
            WHERE document_id = %s 
            AND citation_info->>'source_path' LIKE %s
        """
        params = [document_id, f"%{filename.replace('%', '%%')}"]

        if start:
            query += " AND line_number >= %s"
            params.append(start)
        if end:
            query += " AND line_number <= %s"
            params.append(end)

        query += " ORDER BY line_number"

        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        if not rows:
            return {"error": f"Supplement not found: {filename}"}

        # Get total lines for this supplement
        total_lines = len(rows)  # Supplements are small, this is fine

        return {
            "document_id": document_id,
            "filename": filename,
            # Include block_id (database id) for citations
            # Add +1 to line numbers since DB is 0-indexed but we display 1-indexed
            "lines": [
                {"block_id": r[0], "line": r[1] + 1, "content": r[2]} for r in rows
            ],
            "count": len(rows),
            "total_lines": total_lines,
            "time_ms": round((time.perf_counter() - start_time) * 1000),
        }

    async def _head(
        self,
        path: str,
        n: int = 10,
        session_id: str = "default",
    ) -> dict:
        """Read first n lines of a file."""
        # Get all lines then slice (since line numbers are global, not per-section)
        result = await self._cat(path=path, session_id=session_id)
        if "lines" in result:
            result["lines"] = result["lines"][:n]
            result["count"] = len(result["lines"])
        return result

    async def _tail(
        self,
        path: str,
        n: int = 10,
        session_id: str = "default",
    ) -> dict:
        """Read last n lines of a file."""
        # Get all lines then slice from end (since line numbers are global, not per-section)
        result = await self._cat(path=path, session_id=session_id)
        if "lines" in result:
            result["lines"] = result["lines"][-n:]
            result["count"] = len(result["lines"])
        return result

    async def _stat(
        self,
        path: str,
        session_id: str = "default",
    ) -> dict:
        """Get stats about a path."""
        start_time = time.perf_counter()
        parsed = self.path_parser.parse(path)

        if parsed.error:
            return {"error": parsed.error}

        conn = _get_db_connection()

        if parsed.type in ("document", "file", "section"):
            doc_id = parsed.document_id

            with conn.cursor() as cur:
                cur.execute(
                    """SELECT d.title, d.doi, d.source, d.authors, d.month_year,
                              COUNT(cb.line_number) as lines,
                              COUNT(DISTINCT cb.section) as sections
                       FROM documents d
                       LEFT JOIN content_blocks cb ON d.document_id = cb.document_id
                       WHERE d.document_id::text = %s
                       GROUP BY d.document_id""",
                    (doc_id,),
                )
                row = cur.fetchone()

            if not row:
                return {"error": f"Document not found: {doc_id}"}

            return {
                "path": path,
                "document_id": doc_id,
                "title": row[0],
                "doi": row[1],
                "source": row[2],
                "authors": row[3],
                "month_year": row[4],
                "total_lines": row[5],
                "section_count": row[6],
                "time_ms": round((time.perf_counter() - start_time) * 1000),
            }

        return {"error": f"Cannot stat path: {path}"}

    async def _find(
        self,
        query: str = None,
        title: str = None,
        author: str = None,
        doi: str = None,
        date_range: str = None,
        source: str = None,
        search_mode: str = "any",
        since: str = None,
        category: str = None,
        journal: str = None,
        article_type: str = None,
        year: str = None,
        sort: str = None,
        all_time: bool = False,
        ranking: str = "hybrid",
        limit: int = 100,
        session_id: str = "default",
        document_ids: list[str] = None,
        **_extra,
    ) -> dict:
        """Find papers matching criteria across all sources (bioRxiv + medRxiv + PMC).

        Uses hybrid search (ES BM25 + Vertex AI vector) by default.

        Args:
            document_ids: Hard-scope results to these paper IDs only (repo-scoped search).
        """
        start_time = time.perf_counter()

        search_query = query or title or author
        filters = {}
        if source:
            filters["source"] = source if isinstance(source, list) else [source]
        if date_range:
            filters["date_range"] = date_range
        if search_mode:
            filters["search_mode"] = search_mode
        if since:
            filters["since"] = since
        if category:
            filters["category"] = category
        if journal:
            filters["journal"] = journal
        if article_type:
            filters["article_type"] = article_type
        if year:
            filters["year"] = str(year)
        if sort:
            filters["sort"] = sort
        if all_time:
            filters["all_time"] = True
        if ranking and ranking != "hybrid":
            filters["ranking"] = ranking

        results = await self.document_store.search_documents(
            query=search_query,
            filters=filters,
            limit=min(limit, 1000),
            document_ids=document_ids,
        )

        # Save results
        results_id = self.results_registry.save(
            data={"papers": results, "query": search_query},
            session_id=session_id,
            prefix="s",
        )

        # Also save as a table artifact so it can be cited/viewed as a table.
        # Fire-and-forget: a first-call GCS TLS+OAuth handshake used to add
        # ~6s to the hot path of one-shot CLI invocations. The artifact still
        # lands in GCS a beat after the response returns, so `map --from`,
        # the table viewer, and any artifact-citation path keep working.
        asyncio.create_task(
            self._save_search_artifact(
                results_id, results, search_query, session_id
            )
        )

        return {
            "results_id": results_id,
            "query": search_query,
            "count": len(results),
            "papers": results[:20],  # Preview
            "time_ms": round((time.perf_counter() - start_time) * 1000),
        }

    async def _lookup(
        self,
        field: str,
        value: str,
        limit: int = 100,
        session_id: str = "default",
    ) -> dict:
        """Look up papers by a specific metadata field.

        Args:
            field: Database column name (doi, authors, title, month_year, source, abstract_text)
            value: Value to search for (partial match using ILIKE)
            limit: Maximum results to return
        """
        start_time = time.perf_counter()
        conn = _get_db_connection()

        # Validate field to prevent SQL injection
        allowed_fields = {
            "doi",
            "document_id",
            "authors",
            "title",
            "month_year",
            "source",
            "abstract_text",
        }
        if field not in allowed_fields:
            return {
                "error": f"Invalid field: {field}. Allowed: {', '.join(allowed_fields)}"
            }

        # For DOI / document_id, use exact match only
        if field in ("doi", "document_id"):
            with conn.cursor() as cur:
                cur.execute(
                    f"""SELECT document_id::text, title, doi, authors, month_year, source
                        FROM documents
                        WHERE {field} = %s
                        ORDER BY created_at DESC
                        LIMIT 1""",
                    (value,),
                )
                row = cur.fetchone()
                if row:
                    return {
                        "total": 1,
                        "results": [
                            {
                                "document_id": row[0],
                                "title": row[1],
                                "doi": row[2],
                                "authors": row[3],
                                "month_year": row[4],
                                "source": row[5],
                                "path": f"/papers/{row[0]}/",
                            }
                        ],
                        "time_ms": round((time.perf_counter() - start_time) * 1000),
                    }
            label = "arXiv ID" if field == "document_id" else "DOI"
            return {
                "total": 0,
                "results": [],
                "field": field,
                "value": value,
                "note": f"Exact {label} match not found. Check the ID is correct.",
                "time_ms": round((time.perf_counter() - start_time) * 1000),
            }

        with conn.cursor() as cur:
            # First get count (unique DOIs)
            cur.execute(
                f"SELECT COUNT(DISTINCT doi) FROM documents WHERE {field} ILIKE %s",
                (f"%{value}%",),
            )
            total = cur.fetchone()[0]

            # Then get results - fetch extra for deduplication
            cur.execute(
                f"""SELECT document_id::text, title, doi, authors, month_year, source
                    FROM documents
                    WHERE {field} ILIKE %s
                    ORDER BY created_at DESC
                    LIMIT %s""",
                (f"%{value}%", limit * 2),
            )
            rows = cur.fetchall()

        results = []
        for row in rows:
            results.append(
                {
                    "document_id": row[0],
                    "title": row[1],
                    "doi": row[2],
                    "authors": row[3],
                    "month_year": row[4],
                    "source": row[5],
                    "path": f"/papers/{row[0]}/",
                }
            )

        # Deduplicate by DOI, keeping most recent version
        results = PapersStore._deduplicate_by_doi(results)[:limit]

        return {
            "total": total,
            "results": results,
            "field": field,
            "value": value,
            "time_ms": round((time.perf_counter() - start_time) * 1000),
        }

    async def _grep(
        self,
        regex: str,
        path: str = "/papers/",
        from_results: str = None,
        top_k: int = None,
        limit: int = 50,
        session_id: str = "default",
        section_filter: str = None,
    ) -> dict:
        """Regex search on paper content."""
        start_time = time.perf_counter()

        parsed = self.path_parser.parse(path)
        document_ids = None
        if not section_filter:
            section_filter = parsed.filter if parsed.type == "document_section" else None

        # Get document IDs from previous results if specified
        if from_results:
            saved = self.results_registry.load(from_results, session_id)
            if saved and "papers" in saved:
                papers = saved["papers"]
                if top_k:
                    papers = sorted(
                        papers, key=lambda p: p.get("score", 0), reverse=True
                    )[:top_k]
                document_ids = [
                    p.get("document_id") for p in papers if p.get("document_id")
                ]

        elif parsed.document_id:
            document_ids = [parsed.document_id]

        # Execute grep
        try:
            matches = await self.document_store.grep_content(
                regex=regex,
                document_ids=document_ids,
                section_filter=section_filter,
                limit=limit,
            )
        except Exception as e:
            if "timeout" in str(e).lower():
                return {
                    "error": "Query timed out - pattern too slow",
                    "timeout_seconds": GREP_TIMEOUT_SECONDS,
                    "help": "Use papers_find first to narrow scope",
                }
            raise

        # Group by document
        doc_matches = {}
        for m in matches:
            doc_id = m["document_id"]
            if doc_id not in doc_matches:
                doc_matches[doc_id] = {
                    "document_id": doc_id,
                    "path": f"/papers/{doc_id}/",
                    "matches": [],
                }
            doc_matches[doc_id]["matches"].append(m)

        papers = list(doc_matches.values())[:limit]

        # Save results
        results_id = self.results_registry.save(
            data={"papers": papers, "regex": regex},
            session_id=session_id,
            prefix="s",
        )

        # Also save as a table artifact so it can be cited/viewed as a table
        await self._save_search_artifact(results_id, papers, f"regex: {regex}", session_id)

        return {
            "results_id": results_id,
            "regex": regex,
            "matched_docs": len(papers),
            "total_matches": len(matches),
            "papers": papers,
            "time_ms": round((time.perf_counter() - start_time) * 1000),
        }

    async def _load_artifact_as_papers(
        self, artifact_id: str, session_id: str
    ) -> dict | None:
        """Load an export artifact (a_xxx) and convert rows to papers format.

        This allows `map --from a_xxx` to work with export artifacts that
        contain a document_id column, bridging the gap between SQL exports
        and the search-results format that _parallel expects.
        """
        try:
            from gxl_filesystem import GXLFileSystem

            fs = GXLFileSystem(session_id=session_id)
            content = await fs.read_file(f"artifacts/{artifact_id}.json")
            artifact = json.loads(content)
        except (FileNotFoundError, json.JSONDecodeError, Exception):
            return None

        rows = artifact.get("output", {}).get("rows", [])
        if not rows:
            return None

        # Rows must contain document_id to be usable for map
        if "document_id" not in rows[0]:
            return None

        papers = []
        for row in rows:
            doc_id = row.get("document_id", "")
            if not doc_id:
                continue
            papers.append(
                {
                    "document_id": doc_id,
                    "title": row.get("title", ""),
                    "path": f"/papers/{doc_id}/",
                    **{
                        k: v
                        for k, v in row.items()
                        if k not in ("document_id", "title")
                    },
                }
            )

        return {"papers": papers} if papers else None

    # -- Config hooks for base class _parallel / _reduce --

    def get_reader_agent_config(self) -> str:
        return "papers/papers_reader_full_content"

    def get_reduce_config(self) -> tuple[str, str | None]:
        from pathlib import Path

        import yaml

        root = Path(os.environ.get("GXL_ROOT", "/workspaces/gxl"))
        config_path = root / "agents" / "papers" / "papers_reducer.yaml"
        if not config_path.exists():
            config_path = (
                root / "agents" / "configs" / "papers" / "papers_reducer.yaml"
            )
        if not config_path.exists():
            raise FileNotFoundError(
                f"Reducer config not found at {config_path}. "
                f"Cannot fall back to a default model — fix the config path."
            )
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        model = config.get("model")
        if not model:
            raise ValueError(
                f"No 'model' field in {config_path}. "
                f"Cannot fall back to a default model — add a model to the config."
            )
        return model, config.get("system_prompt")

    def get_map_table_columns(self) -> list[str]:
        return ["authors", "month_year", "source", "text_access", "response"]

    def build_map_table_row(
        self, index: int, result: dict, meta: dict, response_text: str
    ) -> dict:
        title = result.get("title") or result.get("path") or f"Paper #{index+1}"
        return {
            "paper": title[:80],
            "document_id": result.get("document_id", ""),
            "authors": meta.get("authors", result.get("authors", "-")),
            "month_year": meta.get("month_year", result.get("month_year", "-")),
            "source": meta.get("source", result.get("source", "-")),
            "text_access": meta.get("text_access", result.get("text_access", "full_text")),
            "response": response_text,
        }

    async def _save_artifact(self, artifact_id: str, artifact: dict, session_id: str):
        """Save artifact via GXLFileSystem (respects SANDBOX_PROVIDER env var)."""
        try:
            from gxl_filesystem import GXLFileSystem

            fs = GXLFileSystem(session_id=session_id)
            file_path = f"artifacts/{artifact_id}.json"
            content = json.dumps(artifact, indent=2, default=str)
            await fs.write_file(file_path, content)
            logger.info(f"Saved artifact {artifact_id} to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save artifact {artifact_id}: {e}")
            raise

    # --- REMOVED: _parallel, _reduce, _extract_citation_index, _get_reduce_config,
    # --- _build_table now live in VirtualFilesystemModule base class ---

    async def _save_search_artifact(
        self, results_id: str, papers: list[dict], query: str, session_id: str
    ):
        """Save search results as a table artifact so they can be cited/viewed as tables."""
        try:
            columns = [
                "title",
                "authors",
                "doi",
                "month_year",
                "source",
                "document_id",
                "text_access",
            ]
            rows = []
            for p in papers:
                rows.append(
                    {
                        "title": p.get("title", ""),
                        "authors": p.get("authors", ""),
                        "doi": p.get("doi", ""),
                        "month_year": p.get("month_year", ""),
                        "source": p.get("source", ""),
                        "document_id": p.get("document_id", ""),
                        "text_access": p.get("text_access", "full_text"),
                    }
                )
            artifact = {
                "artifact_id": results_id,
                "artifact_type": "reduce_table",
                "created_at": datetime.now().isoformat(),
                "source": {
                    "description": f"Search: {query}",
                    "paper_count": len(papers),
                },
                "output": {
                    "columns": columns,
                    "rows": rows,
                },
            }
            await self._save_artifact(results_id, artifact, session_id)
        except Exception as e:
            logger.warning(f"Failed to save search artifact: {e}")

    # =========================================================================
    # Figure GCS path resolution (shared by ask_image + download)
    # =========================================================================

    def _resolve_figure_gcs_path(self, document_id: str, figure_id: str) -> dict:
        """Resolve a figure to its GCS path without downloading bytes.

        Returns dict with 'gcs_path', 'filename', 'caption', or 'error'.
        """
        figure_id_base = (
            figure_id.replace(".tif", "").replace(".tiff", "")
            .replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
        )
        conn = _get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """SELECT content, citation_info->>'source_path', citation_info->>'xml_id',
                          citation_info->>'xpath', citation_info->>'graphic'
                   FROM content_blocks
                   WHERE document_id = %s AND block_type = 'figure'
                   AND (
                       citation_info->>'graphic' = %s
                       OR citation_info->>'xml_id' = %s
                       OR citation_info->>'xml_id' = %s
                       OR citation_info->>'graphic' ILIKE %s
                   )
                   LIMIT 1""",
                (document_id, figure_id, figure_id, figure_id_base, f"%{figure_id_base}%"),
            )
            row = cur.fetchone()

        if not row:
            return {"error": f"Figure not found: {figure_id} in paper {document_id}"}

        caption, source_path, xml_id, _xpath, graphic = row
        if not source_path:
            return {"error": "No source path for this figure"}

        if not source_path.startswith("gs://"):
            bucket = os.getenv("BIOMEDRXIV_GCS_BUCKET", "rxiv_dev")
            source_path = f"gs://{bucket}/{source_path}"

        base_path = source_path.rsplit("/", 1)[0]

        candidates = []
        if graphic:
            candidates.append(f"{base_path}/{graphic}")
        if xml_id:
            for ext in [".tif", ".tiff", ".jpg", ".jpeg", ".png"]:
                candidates.append(f"{base_path}/{xml_id}{ext}")

        from .tools import _get_gcs_client

        for gcs_path in candidates:
            client = _get_gcs_client()
            if not client:
                break
            path_no_prefix = gcs_path[5:]
            parts = path_no_prefix.split("/", 1)
            if len(parts) != 2:
                continue
            bucket_name, blob_path = parts
            try:
                blob = client.bucket(bucket_name).blob(blob_path)
                if blob.exists():
                    return {
                        "gcs_path": gcs_path,
                        "filename": graphic or f"{xml_id}{os.path.splitext(gcs_path)[1]}",
                        "caption": caption or "",
                    }
            except Exception:
                continue

        return {"error": f"Image file not found in GCS for figure {figure_id}"}

    # =========================================================================
    # papers_ask_image - Vision model analysis
    # =========================================================================

    async def _ask_image(
        self,
        document_id: str,
        figure_id: str,
        question: str,
        session_id: str = "default",
    ) -> dict:
        """Analyze a figure/image using a vision model."""
        import base64

        from .tools import download_image_from_gcs

        start_time = time.perf_counter()
        conn = _get_db_connection()

        # Find the figure in the database
        # Match by: graphic filename (e.g., 657517v1_figa15.tif), xml_id (e.g., figa15), or partial match
        # Strip extension for partial matching
        figure_id_base = (
            figure_id.replace(".tif", "")
            .replace(".tiff", "")
            .replace(".jpg", "")
            .replace(".jpeg", "")
            .replace(".png", "")
        )

        with conn.cursor() as cur:
            cur.execute(
                """SELECT content, citation_info->>'source_path', citation_info->>'xml_id',
                          citation_info->>'xpath', citation_info->>'graphic'
                   FROM content_blocks
                   WHERE document_id = %s
                   AND block_type = 'figure'
                   AND (
                       citation_info->>'graphic' = %s
                       OR citation_info->>'xml_id' = %s
                       OR citation_info->>'xml_id' = %s
                       OR citation_info->>'graphic' ILIKE %s
                   )
                   LIMIT 1""",
                (
                    document_id,
                    figure_id,
                    figure_id,
                    figure_id_base,
                    f"%{figure_id_base}%",
                ),
            )
            row = cur.fetchone()

        if not row:
            return {
                "error": f"Figure not found: {figure_id} in paper {document_id}",
                "hint": "Use bash 'ls figures/' to list available figures. Use the graphic filename (e.g., 657517v1_fig1.tif) or xml_id (e.g., fig1).",
            }

        caption, source_path, xml_id, xpath, graphic = row

        if not source_path:
            return {"error": "No source path for this figure"}

        # Get image from GCS
        # source_path is like "gs://bucket/path/to/file.xml" or just "path/to/file.xml"
        # We need to get the directory and then construct the image path

        # Ensure we have the gs:// prefix
        if not source_path.startswith("gs://"):
            # Try to add the bucket prefix - check environment variable
            bucket = os.getenv("BIOMEDRXIV_GCS_BUCKET", "rxiv_dev")
            source_path = f"gs://{bucket}/{source_path}"

        base_path = source_path.rsplit("/", 1)[0]
        image_bytes = None
        image_filename = None
        tried_paths = []  # For debugging

        if graphic:
            img_path = f"{base_path}/{graphic}"
            tried_paths.append(img_path)
            img_bytes = download_image_from_gcs(img_path)
            if img_bytes:
                image_bytes = img_bytes
                image_filename = graphic

        if not image_bytes:
            for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                for pattern in [f"{xml_id}{ext}", f"{xml_id.upper()}{ext}"]:
                    img_path = f"{base_path}/{pattern}"
                    tried_paths.append(img_path)
                    img_bytes = download_image_from_gcs(img_path)
                    if img_bytes:
                        image_bytes = img_bytes
                        image_filename = pattern
                        break
                if image_bytes:
                    break

        if not image_bytes:
            # Log the paths we tried for debugging
            logger.warning(
                f"Image not found. Tried paths: {tried_paths[:5]}"
            )  # Only log first 5
            return {
                "figure_id": xml_id,
                "caption": caption,
                "warning": "Image file not found in GCS",
                "tried_paths": tried_paths[:3],  # Show first 3 paths for debugging
                "source_path": source_path,
                "graphic": graphic,
                "question": question,
                "time_ms": round((time.perf_counter() - start_time) * 1000),
            }

        # Convert TIF to PNG if needed
        if image_filename and image_filename.lower().endswith((".tif", ".tiff")):
            try:
                import io

                from PIL import Image

                img = Image.open(io.BytesIO(image_bytes))
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")
                # Resize if too large
                max_dim = 1024
                if max(img.size) > max_dim:
                    ratio = max_dim / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
                mime_type = "image/png"
            except Exception as e:
                logger.warning(f"TIF conversion failed: {e}")
                return {"error": f"Image conversion failed: {e}"}
        else:
            suffix = (image_filename or "").lower().split(".")[-1]
            mime_type = "image/jpeg" if suffix in ["jpg", "jpeg"] else "image/png"

        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        # Call vision model via InferenceClient
        try:
            from gxl_inference_client.client import InferenceClient

            logger.info(
                f"[ASK_IMAGE] Calling vision model with {len(image_bytes)} bytes image"
            )

            # Build multimodal message
            message_history = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Figure caption: {caption}\n\nQuestion: {question}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_b64}"
                            },
                        },
                    ],
                }
            ]

            async with InferenceClient(timeout=120.0) as client:
                result = await client.chat(
                    message_history=message_history,
                    model="google/gemini-3-flash-preview",
                    agent_id=f"vision_{uuid.uuid4().hex[:6]}",
                )

            logger.info(f"[ASK_IMAGE] Response keys: {list(result.keys())}")

            # Extract content from response
            # InferenceClient.chat returns {"response": {...}, "usage": {...}}
            analysis = ""
            if "response" in result:
                inner = result["response"]
                if isinstance(inner, dict) and "choices" in inner:
                    analysis = (
                        inner.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                elif isinstance(inner, str):
                    analysis = inner
            elif "content" in result:
                analysis = result["content"]

            logger.info(f"[ASK_IMAGE] Analysis length: {len(analysis)}")

        except Exception as e:
            logger.error(f"[ASK_IMAGE] Exception: {e}")
            return {"error": f"Vision model error: {e}"}

        # Create a nice label for the figure
        label = xml_id or figure_id_base
        if label.startswith("fig"):
            if label.startswith("figa"):
                label = "Appendix Figure " + label[4:]
            else:
                label = "Figure " + label[3:]
        elif label.startswith("alg"):
            label = "Algorithm " + label[3:]

        return {
            "figure_id": xml_id,
            "graphic": graphic or image_filename,
            "label": label,
            "caption": caption,
            "question": question,
            "analysis": analysis,
            "time_ms": round((time.perf_counter() - start_time) * 1000),
            # Citation info for the model to use when citing this image
            "citation_info": {
                "type": "image",
                "doc_id": document_id,
                "figure_id": xml_id or figure_id_base,
                "graphic": graphic or image_filename,
                "label": label,
                "caption": caption[:200] if caption else None,
            },
        }

    # =========================================================================
    # Batch citation (used by terminal cite command)
    # =========================================================================

    async def _batch_cite(
        self,
        doc_id: str,
        line_numbers: list[int],
        supplement_filename: str | None = None,
    ) -> dict | None:
        """Batch citation lookup optimized for Papers (single DB round-trip)."""
        conn = _get_db_connection()

        with conn.cursor() as cur:
            cur.execute(
                "SELECT title, doi, source, authors, month_year FROM documents WHERE document_id = %s",
                (doc_id,),
            )
            doc_row = cur.fetchone()

        if not doc_row:
            return None

        doc_meta = {
            "doc_title": doc_row[0] or "",
            "doi": doc_row[1] or "",
            "source": doc_row[2] or "",
            "authors": doc_row[3] or "",
            "month_year": doc_row[4] or "",
        }

        db_lines = [ln - 1 for ln in line_numbers]
        placeholders = ",".join(["%s"] * len(db_lines))

        with conn.cursor() as cur:
            if supplement_filename:
                cur.execute(
                    f"""SELECT line_number, content, block_type, section, citation_info
                        FROM content_blocks
                        WHERE document_id = %s AND line_number IN ({placeholders})
                          AND citation_info->>'source_path' LIKE %s
                        ORDER BY line_number""",
                    [doc_id] + db_lines + [f"%{supplement_filename}%"],
                )
            else:
                cur.execute(
                    f"""SELECT line_number, content, block_type, section, citation_info
                        FROM content_blocks
                        WHERE document_id = %s AND line_number IN ({placeholders})
                        ORDER BY line_number""",
                    [doc_id] + db_lines,
                )
            rows = {r[0]: r for r in cur.fetchall()}

        lines_result: dict[int, dict | None] = {}
        for ln in line_numbers:
            db_ln = ln - 1
            row = rows.get(db_ln)
            if not row:
                lines_result[ln] = None
                continue
            _, content, block_type, section, citation_info = row
            if not content or not content.strip():
                lines_result[ln] = None
                continue
            ci = citation_info or {}
            lines_result[ln] = {
                "content": content,
                "section": section or "",
                "block_type": block_type or "",
                "citation_info": {
                    "source_type": ci.get("source_type", ""),
                    "source_path": ci.get("source_path", ""),
                    "xml_id": ci.get("xml_id", ""),
                    "xpath": ci.get("xpath", ""),
                },
            }

        return {"doc_meta": doc_meta, "lines": lines_result}

    # =========================================================================
    # papers_get_citation - Get citation info for a line
    # =========================================================================

    async def _get_citation(
        self,
        document_id: str,
        line_number: int,
        supplement_filename: str = None,
        session_id: str = "default",
    ) -> dict:
        """Get citation info for a specific line number.

        Args:
            document_id: Paper UUID
            line_number: Line number within the content (1-indexed as displayed)
            supplement_filename: If provided, look up line in this specific supplement file
        """
        start_time = time.perf_counter()
        conn = _get_db_connection()

        # Convert from 1-indexed (display) to 0-indexed (database)
        db_line_number = line_number - 1

        with conn.cursor() as cur:
            if supplement_filename:
                # Query for supplement-specific line (filter by source_path)
                cur.execute(
                    """SELECT line_number, content, block_type, section, citation_info
                       FROM content_blocks
                       WHERE document_id = %s 
                         AND line_number = %s
                         AND citation_info->>'source_path' LIKE %s
                       LIMIT 1""",
                    (
                        document_id,
                        db_line_number,
                        f"%{supplement_filename.replace('%', '%%')}",
                    ),
                )
            else:
                # Query for any content at this line (including supplements from content.lines)
                # content.lines includes both main XML and supplement PDFs
                cur.execute(
                    """SELECT line_number, content, block_type, section, citation_info
                       FROM content_blocks
                       WHERE document_id = %s 
                         AND line_number = %s
                       LIMIT 1""",
                    (document_id, db_line_number),
                )
            row = cur.fetchone()

        if not row:
            return {"error": f"Line {line_number} not found in {document_id}"}

        line_num, content, block_type, section, citation_info = row

        # Get document metadata
        with conn.cursor() as cur:
            cur.execute(
                "SELECT title, doi, source, authors, month_year FROM documents WHERE document_id = %s",
                (document_id,),
            )
            doc_row = cur.fetchone()

        result = {
            "document_id": document_id,
            "line_number": line_num,
            "content_preview": content[:100] + "..." if len(content) > 100 else content,
            "block_type": block_type,
            "section": section,
            "doc_title": doc_row[0] if doc_row else None,
            "doi": doc_row[1] if doc_row else None,
            "source": doc_row[2] if doc_row else None,
            "authors": doc_row[3] if doc_row else None,
            "month_year": doc_row[4] if doc_row else None,
        }

        if citation_info:
            result["source_type"] = citation_info.get("source_type")
            result["source_path"] = citation_info.get("source_path")
            result["xml_id"] = citation_info.get("xml_id")
            result["xpath"] = citation_info.get("xpath")
            if citation_info.get("page"):
                result["page"] = citation_info.get("page")
            # Include bbox if available and not a placeholder [0,0,1,1]
            bbox = citation_info.get("bbox")
            if bbox and bbox != [0, 0, 1, 1]:
                result["bbox"] = bbox

        result["time_ms"] = round((time.perf_counter() - start_time) * 1000)
        return result

    # =========================================================================
    # Terminal Tool
    # =========================================================================

    def _get_terminal(self, session_id: str) -> VirtualTerminal:
        """Get or create a terminal for a session."""
        if session_id not in self._terminals:
            terminal = PapersTerminal(filesystem_module=self)
            self._terminals[session_id] = terminal
            logger.info(f"Created new terminal for session {session_id}")
        return self._terminals[session_id]

    async def _shell(
        self,
        command: str,
        session_id: str = "default",
        paper_uuid: str | None = None,
    ) -> dict:
        """Execute a shell command in the virtual filesystem."""
        start_time = time.perf_counter()

        # Get terminal for this session
        terminal = self._get_terminal(session_id)

        # If paper_uuid is provided, auto-cd into that paper's directory
        if paper_uuid and terminal.cwd == terminal.root_path:
            paper_dir = f"/papers/{paper_uuid}/"
            await terminal.execute(f"cd {paper_dir}", session_id=session_id)
            logger.info(f"[shell] Auto-cd to {paper_dir} for paper-scoped session")

        # Execute command
        result = await terminal.execute(command, session_id=session_id)

        # Format response
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "cwd": result.cwd,
            "prompt": terminal.get_prompt(),
            "time_ms": round((time.perf_counter() - start_time) * 1000),
        }


# =============================================================================
# Papers Terminal
# =============================================================================


class PapersTerminal(VirtualTerminal):
    """Papers-specific terminal.

    Extends VirtualTerminal with:
    - hostname / home_dir set to biomedrxiv / /papers/
    - funded-by command (requires biomedrxiv Elasticsearch indices)
    - lookup-citation command (requires biomedrxiv DB)
    """

    def __init__(self, filesystem_module=None):
        super().__init__(filesystem_module)
        self.hostname = "papers"
        self.home_dir = "/papers/"
        self.cwd = "/papers/"
        self.env["HOME"] = "/papers/"
        self.env["PWD"] = "/papers/"
        # Register biomedrxiv-only commands
        self._handlers["funded-by"] = self._cmd_funded_by
        self._handlers["lookup-citation"] = self._cmd_lookup_citation
        self._handlers["oa"] = self._cmd_oa
        self._handlers["papers_that_cite"] = self._cmd_papers_that_cite
        self._handlers["websearch"] = self._cmd_websearch
        self._handlers["curl"] = self._cmd_curl
        self._handlers["links"] = self._cmd_links
        self._handlers["links-search"] = self._cmd_links_search
        self._handlers["links-browse"] = self._cmd_links_browse
        self._handlers["links-stats"] = self._cmd_links_stats

    async def _cmd_oa(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ):
        """Search 150M+ OpenAlex papers via full-text search.

        Usage:
            oa search "gene therapy"
            oa search "CRISPR delivery" -n 20
            oa search "single cell RNA" --abstract
        """
        from ..virtual_filesystem.terminal import TerminalResult

        if not args or args[0] != "search":
            return TerminalResult(
                stderr="oa: subcommand required. Usage: oa search <query> [-n N] [--abstract]",
                exit_code=1, cwd=self.cwd,
            )

        sub_args = args[1:]
        limit = 10
        show_abstract = False
        query_parts = []
        i = 0
        while i < len(sub_args):
            a = sub_args[i]
            if a == "-n" and i + 1 < len(sub_args):
                try:
                    limit = int(sub_args[i + 1])
                except ValueError:
                    pass
                i += 2
                continue
            if a in ("--abstract", "-a"):
                show_abstract = True
                i += 1
                continue
            query_parts.append(a)
            i += 1

        query = " ".join(query_parts).strip()
        if not query:
            return TerminalResult(
                stderr='oa search: query required. Usage: oa search "search terms"',
                exit_code=1, cwd=self.cwd,
            )

        import asyncio
        import functools

        def _run_query():
            import psycopg2
            host, pw, user = _resolve_biomedrxiv_creds()
            conn = psycopg2.connect(
                host=host, database="openalex", user=user, password=pw,
                options="-c statement_timeout=15000",
                connect_timeout=10,
            )
            try:
                conn.autocommit = True
                cur = conn.cursor()
                cur.execute(
                    """SELECT p.oa_id, p.title, p.first_author, p.pub_year,
                              p.cited_by_count, p.doi, p.work_type, p.source_name,
                              m.paper_id,
                              CASE WHEN p.abstract IS NOT NULL THEN LEFT(p.abstract, 300) END AS abstract_snip
                       FROM papers p
                       LEFT JOIN oa_paper_mapping m ON m.oa_id = p.oa_id
                       WHERE fts @@ plainto_tsquery('simple', %s)
                       LIMIT %s""",
                    (query, limit),
                )
                return cur.fetchall()
            finally:
                conn.close()

        try:
            loop = asyncio.get_event_loop()
            rows = await loop.run_in_executor(None, _run_query)

            if not rows:
                return TerminalResult(stdout=f"No results for '{query}'\n", cwd=self.cwd)

            lines = [f"OpenAlex search: '{query}' ({len(rows)} results)\n"]
            for oa_id, title, author, year, cited, doi, wtype, source, paper_id, abstract in rows:
                pc_tag = f"  📎 {paper_id}" if paper_id else ""
                lines.append(
                    f"  [{year or '?'}] {author or 'Unknown'} — {(title or 'Untitled')[:90]}"
                    f"\n        cited: {cited or 0}  doi: {doi or 'n/a'}  type: {wtype or '?'}"
                    f"  oa_id: {oa_id}{pc_tag}"
                )
                if show_abstract and abstract:
                    lines.append(f"        {abstract}...")
                lines.append("")

            return TerminalResult(stdout="\n".join(lines) + "\n", cwd=self.cwd)

        except Exception as e:
            err_msg = str(e)
            if "statement timeout" in err_msg.lower() or "cancel" in err_msg.lower():
                return TerminalResult(
                    stderr=f"oa search: query too broad, timed out after 15s. Try more specific terms.",
                    exit_code=1, cwd=self.cwd,
                )
            return TerminalResult(stderr=f"oa search: {e}", exit_code=1, cwd=self.cwd)

    async def _cmd_papers_that_cite(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ):
        """Find all papers that cite a given paper.

        Accepts PaperCLIP IDs (bio_xxx, PMCxxx), DOIs, or OpenAlex oa_ids.

        Usage:
            papers_that_cite bio_4f78753a6feb
            papers_that_cite PMC7194329
            papers_that_cite 10.1101/2024.01.15.575613
            papers_that_cite --oa 2741809807
            papers_that_cite bio_4f78753a6feb -n 20
        """
        from ..virtual_filesystem.terminal import TerminalResult

        limit = 10
        use_oa_id = False
        id_parts = []
        i = 0
        while i < len(args):
            a = args[i]
            if a == "-n" and i + 1 < len(args):
                try:
                    limit = int(args[i + 1])
                except ValueError:
                    pass
                i += 2
                continue
            if a == "--oa":
                use_oa_id = True
                i += 1
                continue
            id_parts.append(a)
            i += 1

        paper_id = " ".join(id_parts).strip()
        if not paper_id:
            return TerminalResult(
                stderr="papers_that_cite: paper ID required.\n"
                       "  Usage: papers_that_cite <paper_id> [-n N] [--oa]\n"
                       "  Accepts: bio_xxx, PMCxxx, DOI, or --oa <oa_id>",
                exit_code=1, cwd=self.cwd,
            )

        import asyncio

        def _run_citation_query():
            import psycopg2
            host, pw, user = _resolve_biomedrxiv_creds()
            papers_conn = psycopg2.connect(
                host=host, database="openalex", user=user, password=pw,
                options="-c statement_timeout=15000",
                connect_timeout=10,
            )
            papers_conn.autocommit = True

            edges_host = os.getenv("OA_EDGES_DB_HOST", "/cloudsql/gxl-prod:us-central1:openalex")
            edges_pw = os.getenv("OA_EDGES_DB_PASSWORD", pw)
            edges_conn = psycopg2.connect(
                host=edges_host, database="openalex", user="postgres", password=edges_pw,
                options="-c statement_timeout=15000",
                connect_timeout=10,
            )
            edges_conn.autocommit = True

            try:
                pcur = papers_conn.cursor()

                oa_id = None
                if use_oa_id:
                    oa_id = int(paper_id)
                elif paper_id.startswith("10."):
                    pcur.execute("SELECT oa_id FROM papers WHERE doi = %s LIMIT 1", (paper_id,))
                    row = pcur.fetchone()
                    if row:
                        oa_id = row[0]
                else:
                    resolved = resolve(paper_id)
                    pcur.execute(
                        "SELECT oa_id FROM oa_paper_mapping WHERE paper_id = %s LIMIT 1",
                        (resolved,),
                    )
                    row = pcur.fetchone()
                    if row:
                        oa_id = row[0]

                if oa_id is None:
                    return None, None, None, paper_id

                pcur.execute(
                    "SELECT title, first_author, pub_year, cited_by_count, doi FROM papers WHERE oa_id = %s",
                    (oa_id,),
                )
                target_row = pcur.fetchone()
                target = target_row[:4] if target_row else None

                # Find all versions (preprint + published) via OpenAlex API
                # so citations to both the preprint and published version are aggregated
                all_oa_ids = [oa_id]
                if target_row and target_row[0]:
                    alt_oa_ids = _find_paper_versions(oa_id, target_row[0])
                    for alt_id in alt_oa_ids:
                        if alt_id != oa_id:
                            all_oa_ids.append(alt_id)
                    if len(all_oa_ids) > 1:
                        best_id = oa_id
                        best_cited = target[3] or 0
                        ph = ",".join(["%s"] * len(all_oa_ids))
                        pcur.execute(
                            f"SELECT oa_id, title, first_author, pub_year, cited_by_count FROM papers WHERE oa_id IN ({ph}) ORDER BY cited_by_count DESC NULLS LAST LIMIT 1",
                            all_oa_ids,
                        )
                        best_row = pcur.fetchone()
                        if best_row and (best_row[4] or 0) > best_cited:
                            target = best_row[1:5]

                ecur = edges_conn.cursor()
                id_placeholders = ",".join(["%s"] * len(all_oa_ids))
                ecur.execute(
                    f"SELECT DISTINCT citing_oa_id FROM citation_edges WHERE cited_oa_id IN ({id_placeholders}) LIMIT %s",
                    all_oa_ids + [limit + 50],
                )
                citing_ids = [r[0] for r in ecur.fetchall()]

                if not citing_ids:
                    return target, [], None, paper_id

                placeholders = ",".join(["%s"] * len(citing_ids))
                pcur.execute(
                    f"""SELECT p.oa_id, p.title, p.first_author, p.pub_year,
                               p.cited_by_count, p.doi, p.work_type,
                               m.paper_id
                        FROM papers p
                        LEFT JOIN oa_paper_mapping m ON m.oa_id = p.oa_id
                        WHERE p.oa_id IN ({placeholders})
                        ORDER BY p.cited_by_count DESC NULLS LAST
                        LIMIT %s""",
                    citing_ids + [limit],
                )
                return target, pcur.fetchall(), oa_id, paper_id
            finally:
                papers_conn.close()
                edges_conn.close()

        try:
            if use_oa_id:
                try:
                    int(paper_id)
                except ValueError:
                    return TerminalResult(
                        stderr=f"papers_that_cite: invalid oa_id '{paper_id}'",
                        exit_code=1, cwd=self.cwd,
                    )

            loop = asyncio.get_event_loop()
            target, rows, oa_id, pid = await loop.run_in_executor(None, _run_citation_query)

            if target is None and rows is None:
                return TerminalResult(
                    stderr=f"papers_that_cite: could not resolve '{pid}' to an OpenAlex ID.\n"
                           "  Try: papers_that_cite --oa <numeric_oa_id>",
                    exit_code=1, cwd=self.cwd,
                )

            target_desc = (
                f"{target[1] or 'Unknown'} ({target[2] or '?'}) — {(target[0] or 'Untitled')[:80]}"
                if target else f"oa_id={oa_id}"
            )

            if not rows:
                return TerminalResult(
                    stdout=f"Target: {target_desc}\n\nNo citing papers found in OpenAlex.\n",
                    cwd=self.cwd,
                )

            cited_count = target[3] if target else "?"
            lines = [
                f"Target: {target_desc}",
                f"Cited by: {cited_count} papers (OpenAlex)"
                f" — showing {len(rows)} from graph\n",
            ]
            for oa, title, author, year, cited, doi, wtype, pc_id in rows:
                pc_tag = f"  📎 {pc_id}" if pc_id else ""
                lines.append(
                    f"  [{year or '?'}] {author or 'Unknown'} — {(title or 'Untitled')[:90]}"
                    f"\n        cited: {cited or 0}  doi: {doi or 'n/a'}  type: {wtype or '?'}"
                    f"  oa_id: {oa}{pc_tag}"
                )
                lines.append("")

            return TerminalResult(stdout="\n".join(lines) + "\n", cwd=self.cwd)

        except Exception as e:
            err_msg = str(e)
            if "statement timeout" in err_msg.lower() or "cancel" in err_msg.lower():
                return TerminalResult(
                    stderr="papers_that_cite: query timed out. Try a different paper ID.",
                    exit_code=1, cwd=self.cwd,
                )
            return TerminalResult(
                stderr=f"papers_that_cite: {e}", exit_code=1, cwd=self.cwd,
            )

    async def _cmd_websearch(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ):
        """Search the web via Tavily as a last-resort fallback when the filesystem lacks a document.

        Only use this after confirming the paper is not in /papers/ via search/lookup.

        Usage:
            websearch "CRISPR gene editing nature 2023"
            websearch -n 10 "base editing clinical trial"
            websearch --all-domains "specific author paper title"

        Options:
            -n N           Max results (default: 10)
            -d DEPTH       Search depth: basic or advanced (default: advanced)
            --all-domains  Search the open web instead of default domains

        Examples:
            websearch "Doudna CRISPR 2012 Science"
            websearch -n 5 --all-domains "biorxiv preprint specific author"
        """
        from ..virtual_filesystem.terminal import TerminalResult

        # Parse flags
        query_parts = []
        n_results = 10
        search_depth = "advanced"
        include_domains: list[str] = []
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "-n" and i + 1 < len(args):
                try:
                    n_results = int(args[i + 1])
                except ValueError:
                    pass
                i += 2
                continue
            if arg == "-d" and i + 1 < len(args):
                search_depth = args[i + 1]
                i += 2
                continue
            if arg == "--all-domains":
                include_domains = []
                i += 1
                continue
            query_parts.append(arg)
            i += 1

        query = " ".join(query_parts).strip()
        if not query:
            return TerminalResult(
                stderr='websearch: query required. Usage: websearch "search terms"',
                exit_code=1,
                cwd=self.cwd,
            )

        try:
            from tavily import TavilyClient

            api_key = os.getenv("TAVILY_API_KEY", "")
            if not api_key:
                return TerminalResult(
                    stderr="websearch: TAVILY_API_KEY not set",
                    exit_code=1,
                    cwd=self.cwd,
                )

            client = TavilyClient(api_key)
            call_kwargs: dict = {
                "query": query,
                "search_depth": search_depth,
                "max_results": n_results,
            }
            if include_domains:
                call_kwargs["include_domains"] = include_domains

            response = client.search(**call_kwargs)
        except ImportError:
            return TerminalResult(
                stderr="websearch: tavily-python not installed. Run: pip install tavily-python",
                exit_code=1,
                cwd=self.cwd,
            )
        except Exception as e:
            return TerminalResult(
                stderr=f"websearch: Tavily error: {e}",
                exit_code=1,
                cwd=self.cwd,
            )

        results = response.get("results", [])
        if not results:
            return TerminalResult(
                stdout=f"websearch: no results for '{query}'\n",
                cwd=self.cwd,
            )

        output_lines = [
            f"websearch: '{query}'  ({len(results)} results, depth={search_depth})",
            "",
        ]
        for r in results:
            score_str = f"  [score={r.get('score', 0):.2f}]" if "score" in r else ""
            output_lines.append(f"  {r['url']}{score_str}")
            snippet = (r.get("content") or "")[:150].replace("\n", " ")
            if snippet:
                output_lines.append(f"    {snippet}...")
            output_lines.append("")

        output_lines.append(
            "Use curl to download PDF links: curl \"https://...\""
        )

        return TerminalResult(
            stdout="\n".join(output_lines) + "\n",
            cwd=self.cwd,
        )

    async def _cmd_curl(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ):
        """Download a URL and save to /session_files/, processing PDFs via Datalab.

        Use this to download PDFs or pages surfaced by websearch.

        Usage:
            curl URL                      # Download and process automatically
            curl -o OUTFILE URL           # Save to specific /session_files/ path
            curl --no-process URL         # Download raw only, skip Datalab

        Examples:
            curl "https://www.biorxiv.org/content/10.1101/2024.01.01.000000v1.full.pdf"
            curl -o /session_files/paper.pdf "https://..."
        """
        from ..virtual_filesystem.terminal import TerminalResult

        if not args:
            return TerminalResult(
                stderr='curl: URL required. Usage: curl "https://..."',
                exit_code=1,
                cwd=self.cwd,
            )

        output_path: str | None = None
        no_process = False
        url: str | None = None
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "-o" and i + 1 < len(args):
                output_path = args[i + 1]
                i += 2
                continue
            if arg == "--no-process":
                no_process = True
                i += 1
                continue
            if arg.startswith("http://") or arg.startswith("https://"):
                url = arg
            i += 1

        if not url:
            return TerminalResult(
                stderr="curl: no URL found in arguments",
                exit_code=1,
                cwd=self.cwd,
            )

        try:
            import aiohttp

            async with aiohttp.ClientSession() as http_session:
                async with http_session.get(
                    url, allow_redirects=True, timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status != 200:
                        return TerminalResult(
                            stderr=f"curl: HTTP {resp.status} for {url}",
                            exit_code=1,
                            cwd=self.cwd,
                        )
                    content_type = resp.headers.get("Content-Type", "")
                    data = await resp.read()
        except Exception as e:
            return TerminalResult(
                stderr=f"curl: download failed: {e}",
                exit_code=1,
                cwd=self.cwd,
            )

        is_pdf = data.startswith(b"%PDF") or "pdf" in content_type.lower()
        filename = url.rsplit("/", 1)[-1] or "download"
        if filename.lower() in ("download", "download.pdf"):
            parts = url.rstrip("/").split("/")
            if len(parts) >= 2 and parts[-2]:
                filename = parts[-2]
        if is_pdf and not filename.lower().endswith(".pdf"):
            filename += ".pdf"

        if not output_path:
            output_path = f"/session_files/{filename}"

        if is_pdf and not no_process:
            try:
                self._ensure_local_session_dir(session_id)
                session_root = os.path.dirname(self._ensure_local_session_dir(session_id))
                from inference_service.datalab import process_pdf_via_datalab

                await process_pdf_via_datalab(data, filename, session_root)
                stem = filename[:-4] if filename.lower().endswith(".pdf") else filename

                # Sync Datalab output to sandbox provider (GCS when SANDBOX_PROVIDER=e2b)
                local_paper_dir = os.path.join(session_root, "files", stem)
                if os.path.isdir(local_paper_dir):
                    for root_dir, _dirs, fnames in os.walk(local_paper_dir):
                        for fn in fnames:
                            fpath = os.path.join(root_dir, fn)
                            rel = os.path.relpath(fpath, os.path.join(session_root, "files"))
                            sandbox_path = f"/session_files/{rel}"
                            try:
                                file_content = open(fpath, "r", encoding="utf-8", errors="replace").read()
                                await self._session_files_write(sandbox_path, file_content, session_id)
                            except Exception:
                                pass

                return TerminalResult(
                    stdout=(
                        f"curl: downloaded {filename} ({len(data):,} bytes)\n"
                        f"Processed via Datalab → /session_files/{stem}/\n"
                    ),
                    cwd=self.cwd,
                )
            except Exception as e:
                logger.warning(f"curl: Datalab processing failed for {url}: {e}")

        # Save as text or base64-encoded PDF
        if is_pdf:
            import base64

            encoded = base64.b64encode(data).decode("ascii")
            await self._session_files_write(output_path, encoded, session_id)
            return TerminalResult(
                stdout=(
                    f"curl: downloaded {filename} ({len(data):,} bytes)\n"
                    f"Saved (base64) to {output_path}\n"
                ),
                cwd=self.cwd,
            )
        else:
            try:
                text = data.decode("utf-8", errors="replace")
            except Exception:
                text = repr(data[:500])
            await self._session_files_write(output_path, text, session_id)
            return TerminalResult(
                stdout=f"curl: downloaded {filename} ({len(data):,} bytes) → {output_path}\n",
                cwd=self.cwd,
            )

    # ── Paper Links commands ─────────────────────────────────────────

    def _get_paper_links_conn(self):
        from mcps.papers.servers.papers_server import _get_paper_links_connection

        return _get_paper_links_connection()

    def _resolve_document_id(self, raw_id: str) -> str:
        """Resolve a short ID (bio_xxx, med_xxx) or PMC ID to full UUID."""
        from .short_ids import resolve

        return resolve(raw_id)

    async def _cmd_links(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> "TerminalResult":
        """Get database accession links (GEO, PDB, GitHub, SRA, etc.) for a paper.

        Usage:
            links <paper_id>               All database references
            links <paper_id> --db geo      Only GEO accessions
            links <paper_id> --db github   Only GitHub links
        """
        from ..virtual_filesystem.terminal import TerminalResult

        db_name = None
        doc_id = None
        skip_next = False
        for i, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            if arg == "--db" and i + 1 < len(args):
                db_name = args[i + 1].lower()
                skip_next = True
            elif not arg.startswith("-") and doc_id is None:
                doc_id = arg

        if not doc_id:
            # If we're inside a paper directory, use that
            import re as _re

            m = _re.match(r"^/papers/([^/]+)/?", self.cwd)
            if m:
                doc_id = m.group(1)
            else:
                return TerminalResult(
                    stderr="links: usage: links <paper_id> [--db <database>]",
                    exit_code=1,
                    cwd=self.cwd,
                )

        document_id = self._resolve_document_id(doc_id)

        try:
            conn = self._get_paper_links_conn()
            with conn.cursor() as cur:
                if db_name:
                    cur.execute(
                        """SELECT db_name, db_category, accession, section, confidence
                        FROM paper_db_links
                        WHERE document_id = %s AND db_name = %s
                        ORDER BY section""",
                        (document_id, db_name),
                    )
                else:
                    cur.execute(
                        """SELECT db_name, db_category, accession, section, confidence
                        FROM paper_db_links
                        WHERE document_id = %s
                        ORDER BY db_name, section""",
                        (document_id,),
                    )
                rows = cur.fetchall()

            if not rows:
                filter_msg = f" for db={db_name}" if db_name else ""
                return TerminalResult(
                    stdout=f"No database links found{filter_msg} in paper {doc_id}\n",
                    cwd=self.cwd,
                )

            databases: dict[str, list[dict]] = {}
            seen: set[tuple] = set()
            for db, cat, acc, section, conf in rows:
                key = (db, acc)
                if key in seen:
                    continue
                seen.add(key)
                databases.setdefault(db, []).append({
                    "accession": acc,
                    "section": section or "",
                    "category": cat or "",
                })

            lines = [f"Database links for {doc_id}: {len(seen)} accessions across {len(databases)} databases\n"]
            for db, entries in sorted(databases.items()):
                lines.append(f"\n  {db} ({len(entries)}):")
                for e in entries[:50]:
                    sec = f"  [{e['section']}]" if e["section"] else ""
                    lines.append(f"    {e['accession']}{sec}")
                if len(entries) > 50:
                    lines.append(f"    ... and {len(entries) - 50} more")

            return TerminalResult(stdout="\n".join(lines) + "\n", cwd=self.cwd)

        except Exception as e:
            return TerminalResult(
                stderr=f"links: {e}", exit_code=1, cwd=self.cwd
            )

    async def _cmd_links_search(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> "TerminalResult":
        """Find papers that reference a specific database accession.

        Usage:
            links-search GSE224252                     Search across all databases
            links-search --db geo GSE224252            Filter to specific database
            links-search --db pdb --source biorxiv 5UWA
            links-search -n 50 --db github <url>       Limit results
        """
        from ..virtual_filesystem.terminal import TerminalResult

        db_name = None
        source = None
        limit = 25
        accession = None
        skip_next = False

        for i, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            if arg == "--db" and i + 1 < len(args):
                db_name = args[i + 1].lower()
                skip_next = True
            elif arg == "--source" and i + 1 < len(args):
                source = args[i + 1].lower()
                skip_next = True
            elif arg == "-n" and i + 1 < len(args):
                try:
                    limit = min(int(args[i + 1]), 200)
                except ValueError:
                    pass
                skip_next = True
            elif not arg.startswith("-"):
                accession = arg

        if not db_name and not accession:
            return TerminalResult(
                stderr="links-search: provide at least an accession or --db filter\n"
                "  links-search GSE224252\n"
                "  links-search --db geo GSE224252\n"
                "  links-search --db pdb --source biorxiv 5UWA",
                exit_code=1,
                cwd=self.cwd,
            )

        try:
            conn = self._get_paper_links_conn()
            conditions = []
            params: list = []
            if db_name:
                conditions.append("db_name = %s")
                params.append(db_name)
            if accession:
                conditions.append("accession = %s")
                params.append(accession)
            if source:
                conditions.append("source = %s")
                params.append(source)

            where = " AND ".join(conditions)
            params.append(limit)

            with conn.cursor() as cur:
                cur.execute(
                    f"""SELECT DISTINCT document_id, source, db_name, accession, section
                    FROM paper_db_links
                    WHERE {where}
                    LIMIT %s""",
                    params,
                )
                rows = cur.fetchall()

            if not rows:
                query_parts = []
                if accession:
                    query_parts.append(accession)
                if db_name:
                    query_parts.append(f"db={db_name}")
                if source:
                    query_parts.append(f"source={source}")
                return TerminalResult(
                    stdout=f"No papers found referencing {' '.join(query_parts)}\n",
                    cwd=self.cwd,
                )

            from .short_ids import shorten

            papers: dict[str, list] = {}
            for doc_id, src, db, acc, section in rows:
                short = shorten(doc_id, src) or doc_id[:12]
                if short not in papers:
                    papers[short] = []
                papers[short].append(f"{db}:{acc}" + (f" [{section}]" if section else ""))

            query_desc = accession or db_name or ""
            lines = [f"Papers referencing {query_desc}: {len(papers)} found\n"]
            for paper_id, refs in list(papers.items())[:limit]:
                lines.append(f"  {paper_id}")
                for ref in refs[:5]:
                    lines.append(f"    {ref}")
                if len(refs) > 5:
                    lines.append(f"    ... +{len(refs) - 5} more")

            return TerminalResult(stdout="\n".join(lines) + "\n", cwd=self.cwd)

        except Exception as e:
            return TerminalResult(
                stderr=f"links-search: {e}", exit_code=1, cwd=self.cwd
            )

    async def _cmd_links_browse(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> "TerminalResult":
        """Browse papers by database type, category, or source.

        Usage:
            links-browse --db github                    Papers with GitHub repos
            links-browse --category structural          Papers with PDB/EMDB/etc.
            links-browse --db geo --db github --all     Papers with BOTH
            links-browse --db github --source biorxiv   Filter by source
            links-browse -n 50 --db pdb                 Limit results
        """
        from ..virtual_filesystem.terminal import TerminalResult

        db_names: list[str] = []
        category = None
        source = None
        require_all = False
        limit = 25
        offset = 0
        skip_next = False

        for i, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            if arg == "--db" and i + 1 < len(args):
                db_names.append(args[i + 1].lower())
                skip_next = True
            elif arg == "--category" and i + 1 < len(args):
                category = args[i + 1].lower()
                skip_next = True
            elif arg == "--source" and i + 1 < len(args):
                source = args[i + 1].lower()
                skip_next = True
            elif arg in ("--all", "--require-all"):
                require_all = True
            elif arg == "-n" and i + 1 < len(args):
                try:
                    limit = min(int(args[i + 1]), 200)
                except ValueError:
                    pass
                skip_next = True
            elif arg == "--offset" and i + 1 < len(args):
                try:
                    offset = int(args[i + 1])
                except ValueError:
                    pass
                skip_next = True

        if not db_names and not category and not source:
            return TerminalResult(
                stderr="links-browse: provide at least one filter\n"
                "  links-browse --db github\n"
                "  links-browse --category structural\n"
                "  links-browse --db geo --db github --all",
                exit_code=1,
                cwd=self.cwd,
            )

        try:
            conn = self._get_paper_links_conn()

            if db_names and require_all and len(db_names) > 1:
                params: list = [db_names, len(db_names)]
                source_filter = ""
                if source:
                    source_filter = "AND source = %s"
                    params.append(source)
                params.extend([limit, offset])

                with conn.cursor() as cur:
                    cur.execute(
                        f"""WITH matching_docs AS (
                            SELECT document_id
                            FROM paper_db_links
                            WHERE db_name = ANY(%s) {source_filter}
                            GROUP BY document_id
                            HAVING COUNT(DISTINCT db_name) = %s
                            LIMIT %s OFFSET %s
                        )
                        SELECT p.document_id, p.source, p.db_name, p.accession, p.section
                        FROM matching_docs m
                        JOIN paper_db_links p ON p.document_id = m.document_id
                        AND p.db_name = ANY(%s)
                        ORDER BY p.document_id""",
                        params + [db_names],
                    )
                    rows = cur.fetchall()
            else:
                conditions = []
                params = []
                if len(db_names) == 1:
                    conditions.append("db_name = %s")
                    params.append(db_names[0])
                elif db_names:
                    conditions.append("db_name = ANY(%s)")
                    params.append(db_names)
                if category:
                    conditions.append("db_category = %s")
                    params.append(category)
                if source:
                    conditions.append("source = %s")
                    params.append(source)

                where = " AND ".join(conditions)
                params.extend([limit, offset])

                with conn.cursor() as cur:
                    cur.execute(
                        f"""SELECT DISTINCT document_id, source, db_name, accession, section
                        FROM paper_db_links
                        WHERE {where}
                        ORDER BY document_id
                        LIMIT %s OFFSET %s""",
                        params,
                    )
                    rows = cur.fetchall()

            if not rows:
                return TerminalResult(
                    stdout="No papers found matching filters.\n",
                    cwd=self.cwd,
                )

            from .short_ids import shorten

            papers: dict[str, dict] = {}
            for doc_id, src, db, acc, section in rows:
                short = shorten(doc_id, src) or doc_id[:12]
                if short not in papers:
                    papers[short] = {"source": src, "links": []}
                papers[short]["links"].append(f"{db}:{acc}")

            filter_desc = []
            if db_names:
                filter_desc.append(f"db={'&'.join(db_names)}")
            if category:
                filter_desc.append(f"category={category}")
            if source:
                filter_desc.append(f"source={source}")

            lines = [f"Papers matching {' '.join(filter_desc)}: {len(papers)} found\n"]
            for paper_id, info in list(papers.items()):
                unique_dbs = {l.split(":")[0] for l in info["links"]}
                lines.append(f"  {paper_id}  [{', '.join(sorted(unique_dbs))}]")

            if len(papers) == limit:
                lines.append(f"\n  (showing {limit} results, use --offset {offset + limit} for next page)")

            return TerminalResult(stdout="\n".join(lines) + "\n", cwd=self.cwd)

        except Exception as e:
            return TerminalResult(
                stderr=f"links-browse: {e}", exit_code=1, cwd=self.cwd
            )

    async def _cmd_links_stats(
        self, args: list[str], stdin: str = "", session_id: str = "default", **kwargs
    ) -> "TerminalResult":
        """Summary statistics for database references.

        Usage:
            links-stats                 Global corpus stats (top databases)
            links-stats <paper_id>      Per-paper breakdown
            links-stats -n 50           Show top 50 databases
        """
        from ..virtual_filesystem.terminal import TerminalResult
        from mcps.papers.servers.papers_server import (
            _paper_links_stats_cache,
            _paper_links_stats_cache_time,
        )
        import mcps.papers.servers.papers_server as _ps

        doc_id = None
        top_n = 20
        skip_next = False
        for i, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            if arg == "-n" and i + 1 < len(args):
                try:
                    top_n = int(args[i + 1])
                except ValueError:
                    pass
                skip_next = True
            elif not arg.startswith("-"):
                doc_id = arg

        try:
            conn = self._get_paper_links_conn()

            if doc_id:
                document_id = self._resolve_document_id(doc_id)
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT db_name, db_category, COUNT(*), COUNT(DISTINCT accession)
                        FROM paper_db_links
                        WHERE document_id = %s
                        GROUP BY db_name, db_category
                        ORDER BY COUNT(*) DESC""",
                        (document_id,),
                    )
                    rows = cur.fetchall()

                if not rows:
                    return TerminalResult(
                        stdout=f"No database links found for {doc_id}\n",
                        cwd=self.cwd,
                    )

                total = sum(cnt for _, _, cnt, _ in rows)
                lines = [f"Link stats for {doc_id}: {total} total across {len(rows)} databases\n"]
                for db, cat, cnt, uniq in rows:
                    lines.append(f"  {db:20s}  {uniq:4d} unique  ({cnt:4d} mentions)  [{cat or ''}]")

                return TerminalResult(stdout="\n".join(lines) + "\n", cwd=self.cwd)

            # Global stats with caching
            now = time.time()
            CACHE_TTL = 300

            if _ps._paper_links_stats_cache and (now - _ps._paper_links_stats_cache_time) < CACHE_TTL:
                result = _ps._paper_links_stats_cache
                cached = True
            else:
                with conn.cursor() as cur:
                    cur.execute("SELECT reltuples::bigint FROM pg_class WHERE relname = 'paper_db_links'")
                    est_total = cur.fetchone()[0]

                    cur.execute(
                        """SELECT db_name, db_category, COUNT(*)
                        FROM paper_db_links
                        GROUP BY db_name, db_category
                        ORDER BY COUNT(*) DESC"""
                    )
                    db_rows = cur.fetchall()

                categories: dict[str, int] = {}
                databases = []
                for db, cat, cnt in db_rows:
                    databases.append({"name": db, "category": cat, "count": cnt})
                    categories[cat] = categories.get(cat, 0) + cnt

                result = {
                    "total_links_estimate": est_total,
                    "databases": databases,
                    "categories": categories,
                }
                _ps._paper_links_stats_cache = result
                _ps._paper_links_stats_cache_time = now
                cached = False

            lines = [
                f"Paper Links corpus: ~{result['total_links_estimate']:,} total links"
                + (" (cached)" if cached else "")
                + "\n"
            ]

            lines.append(f"\nTop {top_n} databases:")
            for db_info in result["databases"][:top_n]:
                lines.append(
                    f"  {db_info['name']:20s}  {db_info['count']:>10,}  [{db_info.get('category', '')}]"
                )

            lines.append(f"\nCategories:")
            cats = result["categories"]
            if isinstance(cats, dict):
                sorted_cats = sorted(cats.items(), key=lambda x: -x[1])
            else:
                sorted_cats = [(c["name"], c["count"]) for c in cats]
            for name, count in sorted_cats:
                lines.append(f"  {name:20s}  {count:>10,}")

            return TerminalResult(stdout="\n".join(lines) + "\n", cwd=self.cwd)

        except Exception as e:
            return TerminalResult(
                stderr=f"links-stats: {e}", exit_code=1, cwd=self.cwd
            )


