"""
Papers MCP Server — unified literature access (bioRxiv, medRxiv, arXiv, PMC).

Provides the PapersServer and PapersModule classes that power the
virtual filesystem, search, grep, and citation tools for biomedical papers.
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
import traceback
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

try:
    import re2 as _re2_mod

    _HAS_RE2 = True
except ImportError:
    _HAS_RE2 = False


def _compile_regex(pattern: str, ignore_case: bool = True):
    """Compile regex with RE2 (linear-time, 4-5x faster) when possible.

    Falls back to Python re for patterns RE2 doesn't support
    (backreferences, lookahead/lookbehind, etc.).
    """
    if _HAS_RE2:
        try:
            prefix = "(?i)" if ignore_case else ""
            return _re2_mod.compile(prefix + pattern)
        except Exception:
            pass
    flags = re.IGNORECASE if ignore_case else 0
    return re.compile(pattern, flags)


ABSTRACT_SNIPPET_LEN = int(os.environ.get("ABSTRACT_SNIPPET_LEN", "200"))

# Add src directory to Python path
src_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# slab-grep client lives alongside this server
_slab_grep_dir = str(Path(__file__).resolve().parent / "slab-grep")
if _slab_grep_dir not in sys.path:
    sys.path.insert(0, _slab_grep_dir)

from gxl_inference_client.agent import Agent
from mcp.types import TextContent, Tool
from modules.papers.filesystem import PapersTerminal
from modules.papers.short_ids import resolve, shorten, shorten_result, shorten_results
from modules.virtual_filesystem.base import (
    DocumentStore,
    ParsedPath,
    PathParser,
    VirtualFilesystemModule,
)
from modules.virtual_filesystem.block_id_codec import (
    decode_block_id,
    decode_pmc_block_id,
    encode_pmc_block_id,
    is_encoded_block_id,
    is_pmc_block_id,
)
from modules.virtual_filesystem.cache import ResultsRegistry, _generate_id
from modules.virtual_filesystem.parallel import ParallelExecutor, ReduceStrategies
from modules.virtual_filesystem.terminal import VirtualTerminal
from shared.core.base_server import MCPServer
from shared.core.config import ServerConfig
from shared.core.environment import get_inference_url, load_environment
from shared.core.response_manager import ResponseManager
from shared.core.session_manager import SessionManager
from shared.core.transports import HTTPTransport, StdioTransport

from mcps.papers.servers.compact_fmt import FORMATTERS as COMPACT_FORMATTERS
from mcps.papers.servers.search_backends import (
    ALL_CORPORA as _ALL_CORPORA,
    CORPUS_ABSTRACT_ONLY,
    CORPUS_ARXIV,
    CORPUS_BIOMEDRXIV,
    CORPUS_PMC,
    OS_INDEX_BY_CORPUS,
    OS_SOURCE_FILTER_BY_CORPUS,
    PREPRINTS_OS_INDEX,
    QDRANT_COLLECTION_BY_CORPUS,
    corpus_for_source,
    get_opensearch_client,
    get_qdrant_client,
    reset_opensearch_client,
    resolve_corpora,
)

logger = logging.getLogger(__name__)

# Lazy-loaded clients
_es_client = None
_db_pool = None  # psycopg2 ThreadedConnectionPool
_paper_links_conn = None
_paper_links_last_used: float = 0
_paper_links_stats_cache: dict | None = None
_paper_links_stats_cache_time: float = 0

# Image cache: gcs_path -> bytes (populated by prefetch for paper sessions)
_image_cache: dict[str, bytes] = {}
_prefetch_tasks: dict[str, bool] = {}  # paper_uuid -> True if prefetch started

# Hydration metadata cache: document_id -> metadata dict
# Invalidated every _HYDRATION_CACHE_TTL_S seconds.
_hydration_cache: dict[str, dict] = {}
_hydration_cache_ts: float = 0
_HYDRATION_CACHE_TTL_S = 3600 * 4  # 4 hours
_HYDRATION_CACHE_MAX = 50_000

# Grep timeout (seconds)
GREP_TIMEOUT_SECONDS = 15

# Thread-safe connection pools for grep P2 batch fetches.
# Each pool is a queue of pre-established connections to avoid per-batch connect overhead.
import queue as _queue
import threading as _threading

_GREP_POOL_SIZE = 5
_GREP_STMT_TIMEOUT_MS = 30_000
_grep_bio_pool: _queue.Queue | None = None
_grep_pmc_pool: _queue.Queue | None = None
_grep_pool_lock = _threading.Lock()


def _create_grep_conn(db_url: str):
    """Create a single pooled connection with keepalives + statement timeout."""
    import psycopg2

    conn = psycopg2.connect(
        db_url,
        connect_timeout=15,
        keepalives=1,
        keepalives_idle=60,
        keepalives_interval=10,
        keepalives_count=3,
        options=f"-c statement_timeout={_GREP_STMT_TIMEOUT_MS}",
    )
    conn.autocommit = True
    return conn


def _init_grep_pools():
    """Lazily create bio + pmc connection pools for grep P2."""
    global _grep_bio_pool, _grep_pmc_pool
    with _grep_pool_lock:
        if _grep_bio_pool is not None:
            return
        import re as _re

        bio_url = os.environ.get("BIOMEDRXIV_DB_URL", "")
        pmc_url = _re.sub(r"/biomedrxiv(\?|$)", r"/pmc\1", bio_url) if bio_url else ""

        _grep_bio_pool = _queue.Queue()
        _grep_pmc_pool = _queue.Queue()
        for _ in range(_GREP_POOL_SIZE):
            if bio_url:
                try:
                    _grep_bio_pool.put(_create_grep_conn(bio_url))
                except Exception:
                    pass
            if pmc_url:
                try:
                    _grep_pmc_pool.put(_create_grep_conn(pmc_url))
                except Exception:
                    pass
        _grep_bio_pool._url = bio_url
        _grep_pmc_pool._url = pmc_url


def _grep_pool_get(pool: _queue.Queue):
    """Borrow a connection from the pool, reconnecting if stale."""
    try:
        conn = pool.get(timeout=10)
    except _queue.Empty:
        conn = _create_grep_conn(pool._url)
    if conn.closed:
        conn = _create_grep_conn(pool._url)
    return conn


def _grep_pool_put(pool: _queue.Queue, conn):
    """Return a connection to the pool (or discard if broken)."""
    if conn is None or conn.closed:
        try:
            conn = _create_grep_conn(pool._url)
        except Exception:
            return
    try:
        pool.put_nowait(conn)
    except _queue.Full:
        try:
            conn.close()
        except Exception:
            pass


def _load_enabled_sources() -> set[str]:
    """Read enabled_sources from ENABLED_SOURCES env var or agents/papers/papers.yaml."""
    env_val = os.environ.get("ENABLED_SOURCES")
    if env_val:
        return {s.strip() for s in env_val.split(",") if s.strip()}
    try:
        import yaml

        config_path = (
            Path(os.environ.get("GXL_ROOT", "/workspaces/gxl"))
            / "agents"
            / "papers"
            / "papers.yaml"
        )
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            sources = config.get("enabled_sources")
            if sources is not None:
                return set(sources)
    except Exception:
        pass
    return {"biorxiv", "medrxiv", "arxiv", "pmc", "openalex", "abstracts"}


ENABLED_SOURCES = _load_enabled_sources()

# Default statement timeout applied at connection level (2 minutes).
# Individual tools can override this lower (e.g. _raw_sql uses 15s)
# but no query can exceed this unless explicitly raised.
_DB_STATEMENT_TIMEOUT_MS = 120_000

# Database connection retry settings
_DB_MAX_RETRIES = 3
_DB_RETRY_DELAY = 1.0  # seconds, doubles each retry

# Elasticsearch connection retry settings
_ES_MAX_RETRIES = 4
_ES_RETRY_DELAY = 1.0  # seconds, doubles each retry

_db_last_used = 0  # Timestamp of last successful query
_DB_HEALTH_CHECK_INTERVAL = 30  # Only health check if idle > 30 seconds
_DB_POOL_MIN = 2
_DB_POOL_MAX = 10

_es_last_used: float = 0  # Timestamp of last successful ES query
_ES_IDLE_RESET_SECONDS = 120  # Reset ES client if idle > 2 minutes (TLS likely stale)


def _extract_cloudsql_socket_path(db_url: str) -> str | None:
    """Extract the Cloud SQL socket file path from a DB URL.

    Example URL: postgresql://user:pass@/dbname?host=/cloudsql/project:region:instance
    Returns:     /cloudsql/project:region:instance/.s.PGSQL.5432
    """
    match = re.search(r"host=(/cloudsql/[^&]+)", db_url)
    if match:
        return f"{match.group(1)}/.s.PGSQL.5432"
    return None


def _extract_cloudsql_instance(db_url: str) -> str | None:
    """Extract Cloud SQL instance name from a DB URL.

    Returns e.g. 'project:region:instance' or None.
    """
    match = re.search(r"host=/cloudsql/([^&]+)", db_url)
    return match.group(1) if match else None


def _ensure_cloudsql_proxy(db_url: str) -> None:
    """Restart the Cloud SQL proxy if its socket file is missing.

    The proxy can lose its socket if another service's startup or the
    devcontainer lifecycle deleted it.  This function detects a missing
    socket, kills any zombie proxy for the same instance, and starts a
    fresh one.
    """
    import shutil
    import subprocess

    socket_path = _extract_cloudsql_socket_path(db_url)
    if not socket_path:
        return

    if os.path.exists(socket_path):
        return

    instance = _extract_cloudsql_instance(db_url)
    if not instance:
        return

    socket_dir = os.path.dirname(os.path.dirname(socket_path))  # /cloudsql
    logger.warning(
        f"[DB] Cloud SQL socket missing: {socket_path}. Restarting proxy for {instance}..."
    )

    if not shutil.which("cloud-sql-proxy"):
        logger.error("[DB] cloud-sql-proxy not installed, cannot auto-restart")
        return

    # Kill any zombie proxy for this instance
    try:
        subprocess.run(
            ["pkill", "-f", f"cloud-sql-proxy.*{instance}"],
            timeout=5,
            capture_output=True,
        )
        time.sleep(1)
    except Exception:
        pass

    # Ensure the socket directory exists
    instance_dir = os.path.join(socket_dir, instance)
    os.makedirs(instance_dir, exist_ok=True)

    # Start a new proxy
    try:
        proc = subprocess.Popen(
            ["cloud-sql-proxy", instance, "--unix-socket", socket_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait for the socket to appear (up to 10 seconds)
        for _ in range(20):
            time.sleep(0.5)
            if os.path.exists(socket_path):
                logger.info(
                    f"[DB] Cloud SQL proxy restarted successfully (PID: {proc.pid})"
                )
                return
        logger.error(
            f"[DB] Cloud SQL proxy started (PID: {proc.pid}) but socket not created after 10s"
        )
    except Exception as e:
        logger.error(f"[DB] Failed to restart Cloud SQL proxy: {e}")


def _get_db_connect_kwargs() -> dict:
    """Build psycopg2 connection keyword args from environment."""
    common = dict(
        keepalives=1,
        keepalives_idle=300,
        keepalives_interval=30,
        keepalives_count=5,
        connect_timeout=10,
        options=f"-c statement_timeout={_DB_STATEMENT_TIMEOUT_MS}",
    )
    db_url = os.getenv("BIOMEDRXIV_DB_URL")
    if db_url:
        _ensure_cloudsql_proxy(db_url)
        return {"dsn": db_url, **common}

    host = os.getenv("BIOMEDRXIV_DB_HOST")
    password = os.getenv("BIOMEDRXIV_DB_PASSWORD")
    if not host or not password:
        raise ValueError(
            "Database not configured. Set BIOMEDRXIV_DB_URL or BIOMEDRXIV_DB_HOST + BIOMEDRXIV_DB_PASSWORD."
        )
    return dict(
        host=host,
        port=int(os.getenv("BIOMEDRXIV_DB_PORT", "5432")),
        database=os.getenv("BIOMEDRXIV_DB_NAME", "biomedrxiv"),
        user=os.getenv("BIOMEDRXIV_DB_USER", "postgres"),
        password=password,
        **common,
    )


def _init_db_pool():
    """Create the ThreadedConnectionPool (lazy, once)."""
    global _db_pool
    if _db_pool is not None:
        return _db_pool
    from psycopg2.pool import ThreadedConnectionPool

    kwargs = _get_db_connect_kwargs()
    _db_pool = ThreadedConnectionPool(
        minconn=_DB_POOL_MIN, maxconn=_DB_POOL_MAX, **kwargs
    )
    logger.info(
        f"[DB] Connection pool created (min={_DB_POOL_MIN}, max={_DB_POOL_MAX})"
    )
    return _db_pool


def _create_db_connection():
    """Create a standalone connection (used by legacy callers and force-reconnect)."""
    import psycopg2

    kwargs = _get_db_connect_kwargs()
    conn = psycopg2.connect(**kwargs)
    conn.autocommit = True
    return conn


def _force_reconnect():
    """Force reconnection of both primary connection and pool."""
    global _db_conn, _db_pool, _db_last_used
    if _db_conn is not None:
        try:
            _db_conn.close()
        except Exception:
            pass
        _db_conn = None
    if _db_pool is not None:
        try:
            _db_pool.closeall()
        except Exception:
            pass
        _db_pool = None
    _db_conn = _create_db_connection()
    _db_last_used = time.time()
    logger.info("[DB] Forced reconnection (primary + pool reset)")
    return _db_conn


def _is_connection_error(exc):
    """Check if an exception is a connection-related error that warrants retry.

    Statement timeouts (QueryCanceled) are NOT connection errors — retrying
    would just burn CPU on the same doomed query.
    """
    import psycopg2

    error_str = str(exc).lower()

    # Statement timeouts raise QueryCanceled (subclass of OperationalError).
    # These should NOT be retried — the query is inherently too slow.
    if "statement timeout" in error_str or "canceling statement" in error_str:
        return False

    # Connection errors that should trigger retry
    if isinstance(exc, (psycopg2.OperationalError, psycopg2.InterfaceError)):
        return True
    connection_errors = [
        "connection",
        "server closed",
        "ssl",
        "eof",
        "broken pipe",
        "connection reset",
        "network",
    ]
    return any(err in error_str for err in connection_errors)


# Module-level reference so PapersStore can access the cached PMC connection
_papers_module_instance: "PapersModule | None" = None


def _get_papers_module() -> "PapersModule | None":
    return _papers_module_instance


_db_conn = None  # Primary shared connection (main thread, backward compat)


def _get_db_connection():
    """Get the primary shared PostgreSQL connection.

    Most callers use this — it returns a single long-lived connection
    with smart health checking.  For parallel work (hydration threads),
    use _get_pooled_connection() / _return_pooled_connection() instead.
    """
    global _db_conn, _db_last_used
    import psycopg2

    def _is_connection_healthy(conn):
        if conn is None or conn.closed:
            return False
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:
            return False

    now = time.time()
    if _db_conn is not None and not _db_conn.closed:
        if (now - _db_last_used) < _DB_HEALTH_CHECK_INTERVAL:
            _db_last_used = now
            return _db_conn

    if not _is_connection_healthy(_db_conn):
        if _db_conn is not None:
            try:
                _db_conn.close()
            except Exception:
                pass
            logger.info("[DB] Reconnecting primary connection — was stale")
        _db_conn = _create_db_connection()
        logger.info("[DB] Primary connection established")

    _db_last_used = now
    return _db_conn


def _get_pooled_connection():
    """Get a connection from the ThreadedConnectionPool.

    Use this for parallel work (asyncio.to_thread callbacks) where
    multiple threads need their own connection simultaneously.
    Always pair with _return_pooled_connection().
    """
    try:
        pool = _init_db_pool()
        conn = pool.getconn()
        if conn.closed:
            pool.putconn(conn, close=True)
            conn = pool.getconn()
        conn.autocommit = True
        return conn
    except Exception as e:
        logger.warning(
            f"[DB] Pool getconn failed ({e}), creating standalone connection"
        )
        return _create_db_connection()


def _return_pooled_connection(conn):
    """Return a connection to the pool."""
    if _db_pool is not None:
        try:
            _db_pool.putconn(conn)
        except Exception:
            pass


def _rrf_merge(
    results_a: list[dict], results_b: list[dict], limit: int = 25, k: int = 60
) -> list[dict]:
    """Reciprocal Rank Fusion: merge two ranked result lists."""
    scores: dict[str, float] = {}
    doc_data: dict[str, dict] = {}
    for rank, r in enumerate(results_a):
        did = r.get("document_id", "")
        scores[did] = scores.get(did, 0) + 1.0 / (k + rank + 1)
        doc_data.setdefault(did, r)
    for rank, r in enumerate(results_b):
        did = r.get("document_id", "")
        scores[did] = scores.get(did, 0) + 1.0 / (k + rank + 1)
        doc_data.setdefault(did, r)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_data[did] for did, _ in ranked[:limit]]


def _execute_with_retry(func, *args, **kwargs):
    """Execute a database function with automatic retry on connection errors.

    If a connection error occurs, forces a reconnection and retries.
    Uses exponential backoff between retries.

    Args:
        func: Function to execute (should use _get_db_connection internally)
        *args, **kwargs: Arguments to pass to the function

    Returns:
        The result of func(*args, **kwargs)

    Raises:
        The last exception if all retries fail
    """
    last_exception = None
    delay = _DB_RETRY_DELAY

    for attempt in range(_DB_MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if not _is_connection_error(e):
                # Not a connection error, don't retry
                raise

            last_exception = e
            if attempt < _DB_MAX_RETRIES - 1:
                logger.warning(
                    f"[DB] Connection error on attempt {attempt + 1}/{_DB_MAX_RETRIES}: {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                _force_reconnect()
            else:
                logger.error(
                    f"[DB] All {_DB_MAX_RETRIES} attempts failed. Last error: {e}"
                )

    raise last_exception


def _get_paper_links_connection():
    """Get connection to the paper_links database (52M+ accession links).

    Same Cloud SQL instance as biomedrxiv, different database name.
    Uses the same health check / reconnect pattern as _get_db_connection.
    """
    global _paper_links_conn, _paper_links_last_used
    import psycopg2

    now = time.time()

    if _paper_links_conn is not None and not _paper_links_conn.closed:
        if (now - _paper_links_last_used) < _DB_HEALTH_CHECK_INTERVAL:
            _paper_links_last_used = now
            return _paper_links_conn

    def _healthy(conn):
        if conn is None or conn.closed:
            return False
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:
            return False

    if not _healthy(_paper_links_conn):
        if _paper_links_conn is not None:
            try:
                _paper_links_conn.close()
            except Exception:
                pass
            logger.info("[paper_links DB] Reconnecting")

        db_url = os.getenv("BIOMEDRXIV_DB_URL")
        if db_url:
            paper_links_url = re.sub(r"/[^/?]+(\?)", r"/paper_links\1", db_url)
            _ensure_cloudsql_proxy(db_url)
            _paper_links_conn = psycopg2.connect(
                paper_links_url,
                keepalives=1,
                keepalives_idle=300,
                keepalives_interval=30,
                keepalives_count=5,
                connect_timeout=10,
                options=f"-c statement_timeout={_DB_STATEMENT_TIMEOUT_MS}",
            )
        else:
            host = os.getenv("BIOMEDRXIV_DB_HOST")
            password = os.getenv("BIOMEDRXIV_DB_PASSWORD")
            if not host or not password:
                raise ValueError(
                    "Database not configured. Set BIOMEDRXIV_DB_URL or BIOMEDRXIV_DB_HOST + BIOMEDRXIV_DB_PASSWORD."
                )
            _paper_links_conn = psycopg2.connect(
                host=host,
                port=int(os.getenv("BIOMEDRXIV_DB_PORT", "5432")),
                database="paper_links",
                user=os.getenv("BIOMEDRXIV_DB_USER", "postgres"),
                password=password,
                keepalives=1,
                keepalives_idle=300,
                keepalives_interval=30,
                keepalives_count=5,
                connect_timeout=10,
                options=f"-c statement_timeout={_DB_STATEMENT_TIMEOUT_MS}",
            )
        _paper_links_conn.autocommit = True
        logger.info("[paper_links DB] Connected")

    _paper_links_last_used = now
    return _paper_links_conn


_gcs_bucket_cache = None


def _get_gcs_bucket():
    """Get GCS bucket for supplement file listing."""
    global _gcs_bucket_cache
    if _gcs_bucket_cache is None:
        from google.cloud import storage

        _gcs_bucket_cache = storage.Client().bucket("gxl-collections")
    return _gcs_bucket_cache


async def _prefetch_paper_images(paper_uuid: str):
    """Prefetch all figure images for a paper into the in-memory cache."""
    if paper_uuid in _prefetch_tasks:
        return
    _prefetch_tasks[paper_uuid] = True
    try:
        conn = _get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT graphic, source_path FROM figures WHERE document_id = %s",
                (paper_uuid,),
            )
            rows = cur.fetchall()

        from google.cloud import storage

        client = storage.Client()
        count = 0
        for graphic, source_path in rows:
            if not source_path:
                continue
            img_path = source_path
            if img_path in _image_cache:
                count += 1
                continue
            try:
                bucket_name = img_path.split("/")[2]
                blob_path = "/".join(img_path.split("/")[3:])
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                img_bytes = blob.download_as_bytes()
                _image_cache[img_path] = img_bytes
                count += 1
            except Exception:
                pass
        logger.info(
            f"[prefetch] Cached {count}/{len(rows)} images for paper {paper_uuid[:8]}"
        )
    except Exception as e:
        logger.error(f"[prefetch] Failed for {paper_uuid[:8]}: {e}")


def _get_es_client():
    """Return the OpenSearch client for Papers (formerly ES).

    Kept under the ``_get_es_client`` name so existing call-sites that use
    elasticsearch-py-compatible APIs (``es.search``, ``es.msearch``,
    ``es.count``) continue to work unchanged against OpenSearch.
    """
    return get_opensearch_client()


def _reset_es_client():
    """Force OpenSearch client recreation after transport-level failures."""
    return reset_opensearch_client()


def _is_es_connection_error(exc: Exception) -> bool:
    """Return True for transient TLS/transport failures worth retrying."""
    error_str = str(exc).lower()
    transient_markers = [
        "tls",
        "ssl",
        "eof",
        "broken pipe",
        "connection timeout",
        "connection aborted",
        "connection reset",
        "temporarily unavailable",
        "temporary failure",
        "server disconnected",
        "n/a duration",
        "node not available",
    ]
    return any(marker in error_str for marker in transient_markers)


def _execute_es_with_retry(operation, *, operation_name: str):
    """Run an ES operation with reconnection/backoff for transient failures."""
    global _es_last_used
    last_exception = None
    delay = _ES_RETRY_DELAY

    for attempt in range(_ES_MAX_RETRIES):
        es = _get_es_client()
        if not es:
            raise RuntimeError("Elasticsearch not configured")

        try:
            result = operation(es)
            _es_last_used = time.time()
            return result
        except Exception as e:
            if not _is_es_connection_error(e):
                raise

            last_exception = e
            if attempt < _ES_MAX_RETRIES - 1:
                logger.warning(
                    f"[ES] {operation_name} failed on attempt "
                    f"{attempt + 1}/{_ES_MAX_RETRIES}: {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
                delay *= 2
                _reset_es_client()
            else:
                logger.error(
                    f"[ES] {operation_name} failed after {_ES_MAX_RETRIES} attempts: {e}"
                )

    raise last_exception


def _prewarm_es():
    """Prime ES transport/TLS before the first real user query hits a revision."""
    global _es_last_used
    if not _get_es_client():
        return

    for operation_name, index_name in (
        ("preprints warmup", PREPRINTS_OS_INDEX),
        ("pmc warmup", "pmc"),
    ):
        _execute_es_with_retry(
            lambda es, idx=index_name: es.search(
                index=idx,
                body={"size": 0, "query": {"match_all": {}}},
            ),
            operation_name=operation_name,
        )
    _es_last_used = time.time()


_ES_KEEPALIVE_INTERVAL = 45  # seconds between pings
_es_keepalive_task = None


async def _es_keepalive_loop():
    """Background task: ping ES every N seconds to keep connections warm.

    A cheap _count on the smallest index avoids the 2-3s cold-start penalty
    that occurs when ES connections go idle for >60s and TLS must renegotiate.
    """
    global _es_last_used
    while True:
        await asyncio.sleep(_ES_KEEPALIVE_INTERVAL)
        try:
            es = _get_es_client()
            if es:
                await asyncio.to_thread(
                    es.count,
                    index=PREPRINTS_OS_INDEX,
                    body={"query": {"match_all": {}}},
                )
                _es_last_used = time.time()
        except Exception as e:
            logger.debug(f"[ES keepalive] ping failed: {e}")


def _start_es_keepalive():
    """Schedule the ES keep-alive loop (call once from event loop)."""
    global _es_keepalive_task
    if _es_keepalive_task is None or _es_keepalive_task.done():
        _es_keepalive_task = asyncio.create_task(_es_keepalive_loop())
        logger.info(f"[ES] Keep-alive ping started (every {_ES_KEEPALIVE_INTERVAL}s)")


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

        doc_id = resolve(parts[1])

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
                if section_name.endswith(".md"):
                    section_name = section_name[:-3]
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
            if filename.endswith(".md.lines") or filename.endswith(".cheatsheet.md"):
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

        if subpath == "reviews":
            if len(parts) == 3:
                return ParsedPath(type="reviews_list", document_id=doc_id)
            filename = parts[3]
            return ParsedPath(type="review_file", document_id=doc_id, filename=filename)

        # Treat any other subpath as a section/block_type filter
        return ParsedPath(type="document_section", document_id=doc_id, filter=subpath)


# =============================================================================
# Papers-specific DocumentStore
# =============================================================================


class PapersStore(DocumentStore):
    """PostgreSQL + Elasticsearch backend for Papers."""

    @staticmethod
    def _rrf_merge_multi(
        result_lists: list[list[dict]], limit: int = 25, k: int = 60
    ) -> list[dict]:
        """Reciprocal Rank Fusion across N result lists."""
        scores = {}
        doc_data = {}
        for result_list in result_lists:
            for rank, r in enumerate(result_list):
                did = r.get("document_id")
                if not did:
                    continue
                scores[did] = scores.get(did, 0) + 1.0 / (k + rank + 1)
                if did not in doc_data:
                    doc_data[did] = r
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        results = []
        for did, score in ranked[:limit]:
            entry = doc_data[did].copy()
            entry["score"] = score
            results.append(entry)
        return results

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
                # Sort by month_year descending (most recent first)
                group.sort(
                    key=lambda x: parse_month_year(x.get("month_year", "")),
                    reverse=True,
                )
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
    def _dedup_cross_source(results: list[dict]) -> list[dict]:
        """Drop lower-priority duplicates when the same paper appears in
        multiple corpora.

        Priority (highest first): PMC > biorxiv/medrxiv/arxiv > openalex.
        PMC and preprint sources have full text; OpenAlex is abstract-only.

        Linking uses both DOI (exact) and title (lowercase, stripped, >30
        chars) so we catch matches even when one corpus lacks a DOI.

        Also replaces ``source="pmc"`` with the journal name when available.
        """
        if not results:
            return results

        def _is_openalex(r: dict) -> bool:
            src = (r.get("source") or "").lower()
            doc_id = r.get("document_id") or ""
            return src == "openalex" or doc_id.startswith("oa_")

        def _is_pmc(r: dict) -> bool:
            src = (r.get("source") or "").lower()
            doc_id = (r.get("document_id") or "").upper()
            return src == "pmc" or doc_id.startswith("PMC")

        def _is_preprint(r: dict) -> bool:
            return (r.get("source") or "").lower() in (
                "biorxiv", "medrxiv", "arxiv",
            )

        # Build lookup sets for full-text sources (PMC + preprints).
        fulltext_dois: set[str] = set()
        fulltext_titles: set[str] = set()
        pmc_titles: set[str] = set()
        pmc_dois: set[str] = set()

        for r in results:
            doi = (r.get("doi") or "").strip().lower()
            title = (r.get("title") or "").strip().lower()
            if _is_pmc(r):
                if doi:
                    pmc_dois.add(doi)
                    fulltext_dois.add(doi)
                if len(title) > 30:
                    pmc_titles.add(title)
                    fulltext_titles.add(title)
            elif _is_preprint(r):
                if doi:
                    fulltext_dois.add(doi)
                if len(title) > 30:
                    fulltext_titles.add(title)

        drop_indices: set[int] = set()
        for idx, r in enumerate(results):
            doi = (r.get("doi") or "").strip().lower()
            title = (r.get("title") or "").strip().lower()

            if _is_openalex(r):
                # Drop OpenAlex when any full-text source has the same paper.
                if (doi and doi in fulltext_dois) or (
                    len(title) > 30 and title in fulltext_titles
                ):
                    drop_indices.add(idx)
            elif _is_preprint(r):
                # Drop preprint when PMC (peer-reviewed) has it.
                if (doi and doi in pmc_dois) or (
                    len(title) > 30 and title in pmc_titles
                ):
                    drop_indices.add(idx)

        for r in results:
            if _is_pmc(r):
                journal = r.get("journal", "")
                if journal:
                    r["source"] = journal

        return [r for idx, r in enumerate(results) if idx not in drop_indices]

    async def get_document(self, document_id: str) -> dict | None:
        """Fetch paper metadata with automatic retry on connection errors."""
        document_id = resolve(document_id)

        def _fetch():
            conn = _get_db_connection()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT document_id::text, title, doi, source, authors, month_year, 
                           abstract_text, created_at
                    FROM documents WHERE document_id::text = %s
                """,
                    (document_id,),
                )
                row = cur.fetchone()

            if not row:
                return None

            return {
                "document_id": shorten(row[0], row[3]),
                "title": row[1],
                "doi": row[2],
                "source": row[3],
                "authors": row[4],
                "month_year": row[5],
                "abstract": row[6],
                "created_at": str(row[7]) if row[7] else None,
            }

        return _execute_with_retry(_fetch)

    async def get_document_content(self, document_id: str) -> list[dict]:
        """Fetch all content blocks for a paper with automatic retry on connection errors."""
        document_id = resolve(document_id)

        def _fetch():
            conn = _get_db_connection()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT line_number, content, section, block_type
                    FROM content_blocks
                    WHERE document_id::text = %s
                    ORDER BY line_number
                """,
                    (document_id,),
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

        return _execute_with_retry(_fetch)

    # ------------------------------------------------------------------
    # Date / category helpers for search filters
    # ------------------------------------------------------------------

    @staticmethod
    def _since_to_pub_year(since_str: str) -> int | None:
        """Convert a duration string like '30d', '6m', '2y' to a minimum pub_year.

        Used as a range filter on the ES `pub_year` integer field
        (works for both bioRxiv and PMC).
        """
        from datetime import date

        today = date.today()
        s = since_str.strip().lower()

        try:
            if s.endswith("d"):
                cutoff = today - timedelta(days=int(s[:-1]))
            elif s.endswith("w"):
                cutoff = today - timedelta(weeks=int(s[:-1]))
            elif s.endswith("m"):
                months = int(s[:-1])
                year = today.year
                month = today.month - months
                while month <= 0:
                    month += 12
                    year -= 1
                cutoff = today.replace(year=year, month=month, day=1)
            elif s.endswith("y"):
                cutoff = today.replace(year=today.year - int(s[:-1]))
            else:
                return None
        except (ValueError, OverflowError):
            return None

        return cutoff.year

    @staticmethod
    def _since_to_date(since_str: str):
        """Convert duration string like '30d', '6m', '2y' to a cutoff date object."""
        from datetime import date

        today = date.today()
        s = since_str.strip().lower()
        try:
            if s.endswith("d"):
                return today - timedelta(days=int(s[:-1]))
            elif s.endswith("w"):
                return today - timedelta(weeks=int(s[:-1]))
            elif s.endswith("m"):
                months = int(s[:-1])
                year, month = today.year, today.month - months
                while month <= 0:
                    month += 12
                    year -= 1
                return today.replace(year=year, month=month, day=1)
            elif s.endswith("y"):
                return today.replace(year=today.year - int(s[:-1]))
        except (ValueError, OverflowError):
            pass
        return None

    @staticmethod
    def _month_year_values_since(since_str: str) -> list[str]:
        """Convert a duration string like '30d', '7d', '6m', '1y' into
        the list of ``Month_YYYY`` keywords that fall within that window.

        Returns an empty list if the string can't be parsed.
        """
        import calendar
        from datetime import date

        today = date.today()
        s = since_str.strip().lower()

        try:
            if s.endswith("d"):
                cutoff = today - timedelta(days=int(s[:-1]))
            elif s.endswith("w"):
                cutoff = today - timedelta(weeks=int(s[:-1]))
            elif s.endswith("m"):
                months = int(s[:-1])
                year = today.year
                month = today.month - months
                while month <= 0:
                    month += 12
                    year -= 1
                cutoff = today.replace(year=year, month=month, day=1)
            elif s.endswith("y"):
                cutoff = today.replace(year=today.year - int(s[:-1]))
            else:
                return []
        except (ValueError, OverflowError):
            return []

        # Enumerate every month from cutoff to today (inclusive)
        values: list[str] = []
        cur = cutoff.replace(day=1)
        while cur <= today:
            values.append(f"{calendar.month_name[cur.month]}_{cur.year}")
            if cur.month == 12:
                cur = cur.replace(year=cur.year + 1, month=1)
            else:
                cur = cur.replace(month=cur.month + 1)
        return values

    def _resolve_category_doc_ids(
        self, es, category: str, limit: int = 10_000
    ) -> list[str] | None:
        """Resolve bioRxiv category → list of document_id strings (or None).

        We no longer maintain a full-text content index in OpenSearch; instead
        we look the category up directly against the biomedrxiv ``documents``
        table, which stores bioRxiv subject areas on each row.
        """
        try:
            conn = _get_db_connection()
            if not conn:
                return None
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT document_id::text FROM documents
                       WHERE categories ILIKE %s
                       LIMIT %s""",
                    (f"%{category}%", limit),
                )
                return [row[0] for row in cur.fetchall()]
        except Exception as e:
            logger.warning(f"Category filter failed: {e}")
            return None

    def _resolve_pmc_id_filters(
        self,
        journal: str | None = None,
        limit: int = 10_000,
    ) -> list[str] | None:
        """Resolve PMC IDs from the PMC database matching a journal name (ILIKE)."""
        try:
            module = _get_papers_module()
            if not module:
                return None
            conn = module._get_pmc_db_connection()
            if not journal:
                return None
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pmc_id FROM documents WHERE journal_title ILIKE %s LIMIT %s",
                    (f"%{journal}%", limit),
                )
                return [row[0] for row in cur.fetchall()]
        except Exception as e:
            logger.warning(f"PMC journal filter failed: {e}")
            return None

    def _resolve_funder_doc_ids(
        self, es, funder_query: str, limit: int = 10_000
    ) -> tuple[list[str], int]:
        """Query the legacy content index for papers whose Funding section matches.

        The ``biomedrxiv_content`` index has been deprecated in favour of
        per-corpus OpenSearch (title+abstract only) + Postgres trigram/BM25
        for full-text. This helper now returns an empty result so that any
        caller passing ``--funder`` degrades gracefully instead of erroring.
        """
        return ([], 0)

    # Painless script to convert month_year keyword to a sortable integer.
    # "February_2026" → 202602
    _MONTH_YEAR_SORT_SCRIPT = {
        "_script": {
            "type": "number",
            "script": {
                "source": (
                    "def m=['january':1,'february':2,'march':3,'april':4,"
                    "'may':5,'june':6,'july':7,'august':8,'september':9,"
                    "'october':10,'november':11,'december':12];"
                    "if(!doc.containsKey('month_year')||doc['month_year'].size()==0) return 0;"
                    "def v=doc['month_year'].value.toLowerCase();"
                    "int idx=v.indexOf('_');"
                    "if(idx<0) return 0;"
                    "def mn=v.substring(0,idx);"
                    "def yr=Integer.parseInt(v.substring(idx+1));"
                    "def mi=m.getOrDefault(mn,0);"
                    "return yr*100+mi;"
                ),
                "lang": "painless",
            },
            "order": "desc",
        }
    }

    # ── Hybrid search: vector embedding support ──────────────────────────
    _embed_client = None

    @classmethod
    def _get_embed_client(cls):
        if cls._embed_client is None:
            try:
                from google import genai

                api_key = os.environ.get("GEMINI_API_KEY")
                if api_key:
                    cls._embed_client = genai.Client(api_key=api_key)
                else:
                    cls._embed_client = genai.Client(
                        vertexai=True,
                        project=os.environ.get("GCP_PROJECT", "gxl-prod"),
                        location=os.environ.get("GCP_REGION", "us-central1"),
                    )
            except Exception as e:
                logger.warning(f"Could not init embedding client: {e}")
        return cls._embed_client

    @classmethod
    def _embed_query(
        cls, query: str, model: str = "gemini-embedding-001", dims: int = 768
    ) -> list[float] | None:
        """Embed a search query using Gemini for vector search."""
        client = cls._get_embed_client()
        if not client:
            return None
        try:
            result = client.models.embed_content(
                model=model,
                contents=[query],
                config={"task_type": "RETRIEVAL_QUERY", "output_dimensionality": dims},
            )
            return result.embeddings[0].values
        except Exception as e:
            logger.warning(f"Query embedding failed ({model}): {e}")
            return None

    @classmethod
    async def _qdrant_vector_search(
        cls,
        query_vec: list[float],
        limit: int = 25,
        sources: list[str] | None = None,
    ) -> list[dict]:
        """Dense-vector search via Qdrant (3072-dim gemini-embedding-2).

        Routes to one or more corpus-specific collections
        (``biomedrxiv``, ``arxiv``, ``pmc``, ``abstract_only``) based on the
        requested sources. Results are merged by score and capped at
        ``limit``.
        """
        import asyncio

        client = get_qdrant_client()
        if not client:
            return []

        corpora = cls._static_resolve_corpora(sources)
        if not corpora:
            return []

        # Pull a bit more per-collection so post-merge we still have ``limit`` hits.
        per_limit = limit if len(corpora) == 1 else max(limit, 10)

        def _one(corpus: str) -> list[dict]:
            coll = QDRANT_COLLECTION_BY_CORPUS[corpus]
            try:
                hits = client.search(coll, query_vec, per_limit, with_payload=True)
            except Exception as e:
                logger.warning(f"Qdrant search on {coll} failed: {e}")
                return []
            out: list[dict] = []
            for p in hits:
                payload = p.get("payload") or {}
                doc_id = (
                    payload.get("document_id")
                    or payload.get("pmc_id")
                    or payload.get("oa_id")
                )
                if not doc_id:
                    continue
                source_val = payload.get("source")
                if not source_val:
                    if isinstance(doc_id, str) and doc_id.startswith("PMC"):
                        source_val = "pmc"
                    elif isinstance(doc_id, str) and doc_id.startswith("oa_"):
                        source_val = "openalex"
                    else:
                        source_val = corpus
                entry: dict[str, Any] = {
                    "document_id": doc_id,
                    "source": source_val,
                    "score": p.get("score", 0.0),
                    "corpus": corpus,
                    "backend": "qdrant",
                }
                # Carry hydration fields that live in Qdrant payload (added
                # by the minimal-payload backfill). Anything missing is
                # filled in later by ``_hydrate_results``.
                for k in (
                    "title",
                    "tldr",
                    "doi",
                    "authors",
                    "pub_date",
                    "pub_year",
                    "journal",
                    "journal_title",
                    "abstract_snippet",
                ):
                    v = payload.get(k)
                    if v not in (None, ""):
                        entry[k] = v
                out.append(entry)
            return out

        per_corpus = await asyncio.gather(
            *(asyncio.to_thread(_one, c) for c in corpora)
        )

        merged: list[dict] = []
        for group in per_corpus:
            merged.extend(group)
        merged.sort(key=lambda x: x.get("score") or 0.0, reverse=True)

        # Dedupe by document_id, keep best score.
        seen: set[str] = set()
        deduped: list[dict] = []
        for r in merged:
            did = r["document_id"]
            if did in seen:
                continue
            seen.add(did)
            deduped.append(r)
            if len(deduped) >= limit:
                break
        return deduped

    @staticmethod
    def _static_resolve_corpora(sources: list[str] | None) -> list[str]:
        """Classmethod-friendly corpus resolver that honours ENABLED_SOURCES."""
        requested = resolve_corpora(sources)
        enabled = {corpus_for_source(v) for v in ENABLED_SOURCES} or set(_ALL_CORPORA)
        return [c for c in requested if c in enabled]

    @staticmethod
    def _rrf_merge(
        results_a: list[dict], results_b: list[dict], limit: int = 25, k: int = 60
    ) -> list[dict]:
        """Reciprocal Rank Fusion: merge two result lists."""
        return PapersModule._rrf_merge_multi([results_a, results_b], limit=limit, k=k)

    @staticmethod
    def _rrf_merge_multi(
        result_lists: list[list[dict]], limit: int = 25, k: int = 60
    ) -> list[dict]:
        """Reciprocal Rank Fusion across N result lists."""
        scores = {}
        doc_data = {}

        for result_list in result_lists:
            for rank, r in enumerate(result_list):
                did = r.get("document_id")
                if not did:
                    continue
                scores[did] = scores.get(did, 0) + 1.0 / (k + rank + 1)
                if did not in doc_data:
                    doc_data[did] = r

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        results = []
        for did, score in ranked[:limit]:
            entry = doc_data[did].copy()
            entry["score"] = score
            results.append(entry)
        return results

    PMC_RESEARCH_TYPES = [
        "research-article",
        "review-article",
        "case-report",
        "brief-report",
        "systematic-review",
        "data-paper",
        "methods-article",
        "rapid-communication",
        "discussion",
    ]

    PMC_ARTICLE_TYPE_FILTER = {"terms": {"article_type": PMC_RESEARCH_TYPES}}

    PMC_ARTICLE_TYPE_SQL = (
        "article_type IN ("
        "'research-article','review-article','case-report','brief-report',"
        "'systematic-review','data-paper','methods-article',"
        "'rapid-communication','discussion')"
    )

    def _get_search_indices(
        self, sources: list[str] | None
    ) -> tuple[list[str], list[str]]:
        """Return (doc_indices, content_indices) based on requested sources.

        ``doc_indices`` are the OpenSearch indices to hit for title+abstract
        BM25. ``content_indices`` is currently always ``[]`` — full-text
        block-level search now lives in PostgreSQL (``_search_pg_bm25`` /
        ``grep``) rather than a separate OpenSearch ``_content`` index.

        Respects the ENABLED_SOURCES env var — disabled sources are never
        included.
        """
        corpora = self._resolve_corpora_from_sources(sources)
        doc_idx: list[str] = []
        for corpus in corpora:
            idx = OS_INDEX_BY_CORPUS[corpus]
            if idx not in doc_idx:
                doc_idx.append(idx)
        if not doc_idx:
            doc_idx = [PREPRINTS_OS_INDEX]
        return doc_idx, []

    def _resolve_corpora_from_sources(
        self, sources: list[str] | None
    ) -> list[str]:
        """Normalise user-facing ``source`` filter to corpora, honouring ENABLED_SOURCES."""
        requested = resolve_corpora(sources)
        # Map the ENABLED_SOURCES set (legacy source values) to corpus keys
        # so the user's config still gates new corpora.
        enabled_corpora: set[str] = set()
        for val in ENABLED_SOURCES:
            enabled_corpora.add(corpus_for_source(val))
        if not enabled_corpora:
            enabled_corpora = set(_ALL_CORPORA)
        return [c for c in requested if c in enabled_corpora]

    # Fields pulled from OpenSearch per corpus. Corpora with TL;DRs in
    # Postgres (biomedrxiv, arxiv, pmc) skip ``abstract`` to shrink the
    # response payload dramatically (~6x smaller). ``abstract_only``
    # (OpenAlex) still needs ``abstract`` because it has no TL;DR yet.
    _OS_SOURCE_FIELDS_BASE = [
        "document_id",
        "pmc_id",
        "oa_id",
        "source",
        "pub_year",
        "pub_date",
        "title",
        "doi",
        "authors",
        "journal_title",
        "article_type",
        "month_year",
        "categories",
    ]
    _OS_SOURCE_FIELDS_WITH_ABSTRACT = _OS_SOURCE_FIELDS_BASE + ["abstract"]
    # Back-compat alias (older call sites).
    _OS_SOURCE_FIELDS = _OS_SOURCE_FIELDS_WITH_ABSTRACT
    _ES_SOURCE_FIELDS = _OS_SOURCE_FIELDS

    # Corpora whose TL;DRs live in Postgres (no need to fetch abstract).
    _CORPORA_WITH_TLDR = frozenset({
        CORPUS_BIOMEDRXIV, CORPUS_ARXIV, CORPUS_PMC,
    })

    def _os_source_for_corpus(self, corpus: str) -> list[str]:
        if corpus in self._CORPORA_WITH_TLDR:
            return self._OS_SOURCE_FIELDS_BASE
        return self._OS_SOURCE_FIELDS_WITH_ABSTRACT

    def _normalise_os_hit(self, hit: dict) -> dict:
        """Extract doc id + source + score (and stash ``_source``) from an OS hit.

        The full ``_source`` dict is attached as ``__os_source`` so the
        hydration step can read OpenSearch fields directly without a second
        roundtrip.
        """
        src = hit.get("_source", {}) or {}
        doc_id = (
            src.get("document_id")
            or src.get("pmc_id")
            or src.get("oa_id")
            or hit.get("_id")
        )
        source_val = src.get("source")
        if not source_val:
            if isinstance(doc_id, str) and doc_id.startswith("PMC"):
                source_val = "pmc"
            elif isinstance(doc_id, str) and doc_id.startswith("oa_"):
                source_val = "openalex"
            else:
                source_val = "biorxiv"
        return {
            "document_id": doc_id,
            "source": source_val,
            "score": hit.get("_score"),
            "backend": "opensearch",
            "__os_source": src,
        }

    # Back-compat alias.
    _normalise_es_hit = _normalise_os_hit

    async def _hydrate_results(self, results: list[dict]) -> list[dict]:
        """Enrich ranked search hits with metadata.

        Core fields (title, abstract, doi, authors, pub_year, journal, etc.)
        come from the OpenSearch ``_source`` stashed on each hit as
        ``__os_source``. TL;DRs live only in Postgres, so we issue a
        parallel batch lookup (biomedrxiv / pmc DBs) to pick them up.

        Hits lacking ``__os_source`` (e.g. from Qdrant-only flows or legacy
        callers) fall back to an OpenSearch ``_mget`` and then to a Postgres
        read for anything still missing.
        """
        if not results:
            return results

        import asyncio

        global _hydration_cache, _hydration_cache_ts

        now = time.time()
        if now - _hydration_cache_ts > _HYDRATION_CACHE_TTL_S:
            _hydration_cache = {}
            _hydration_cache_ts = now

        # 1. Start by copying OS _source fields into each hit.
        for r in results:
            os_src = r.pop("__os_source", None) or {}
            if os_src:
                # Truncate abstract early so we never hold multi-KB strings
                # in memory longer than needed (the full text is only used
                # for BM25 ranking server-side; we only need a snippet).
                ab = os_src.get("abstract")
                if ab and len(ab) > ABSTRACT_SNIPPET_LEN:
                    os_src["abstract"] = ab[:ABSTRACT_SNIPPET_LEN]
                for k, v in os_src.items():
                    r.setdefault(k, v)

        # 2. Classify hits for TL;DR / OS mget lookups.
        needs_mget_biomedrxiv: list[str] = []
        needs_mget_pmc: list[str] = []
        needs_tldr_biomedrxiv: list[str] = []
        needs_tldr_pmc: list[str] = []
        for r in results:
            doc_id = r.get("document_id")
            if not doc_id:
                continue
            source = (r.get("source") or "").lower()
            # OpenAlex doesn't have TL;DRs and OS `_source` is always set on
            # hits from our pipeline, so skip PG entirely.
            if source == "openalex" or (
                isinstance(doc_id, str) and doc_id.startswith("oa_")
            ):
                continue
            cached_meta = _hydration_cache.get(doc_id)
            if cached_meta:
                for k, v in cached_meta.items():
                    r.setdefault(k, v)
            is_pmc = source == "pmc" or (
                isinstance(doc_id, str) and doc_id.startswith("PMC")
            )
            is_arxiv = source == "arxiv" or (
                isinstance(doc_id, str) and doc_id.startswith("arxiv_")
            )
            if not r.get("title"):
                if is_pmc:
                    needs_mget_pmc.append(doc_id)
                elif not is_arxiv:
                    needs_mget_biomedrxiv.append(doc_id)
            if not r.get("tldr") and not cached_meta:
                if is_pmc:
                    needs_tldr_pmc.append(doc_id)
                else:
                    needs_tldr_biomedrxiv.append(doc_id)

        # 3. Fetch missing OS docs via _mget (rare path).
        def _os_mget() -> dict[str, dict]:
            out: dict[str, dict] = {}
            os_client = get_opensearch_client()
            if not os_client:
                return out
            if needs_mget_biomedrxiv:
                try:
                    docs = os_client.mget(
                        PREPRINTS_OS_INDEX,
                        needs_mget_biomedrxiv,
                        source_fields=self._OS_SOURCE_FIELDS,
                    )
                    for d in docs:
                        if d.get("found"):
                            src = d.get("_source") or {}
                            did = (
                                src.get("document_id")
                                or src.get("pmc_id")
                                or d.get("_id")
                            )
                            if did:
                                out[did] = src
                except Exception as e:
                    logger.warning(f"OpenSearch _mget (preprints) failed: {e}")
            if needs_mget_pmc:
                try:
                    docs = os_client.mget(
                        "pmc",
                        needs_mget_pmc,
                        source_fields=self._OS_SOURCE_FIELDS,
                    )
                    for d in docs:
                        if d.get("found"):
                            src = d.get("_source") or {}
                            did = src.get("pmc_id") or src.get("document_id") or d.get("_id")
                            if did:
                                out[did] = src
                except Exception as e:
                    logger.warning(f"OpenSearch _mget (pmc) failed: {e}")
            return out

        # 4. Fetch TL;DRs from Postgres.
        def _fetch_tldr_bio() -> dict[str, dict]:
            if not needs_tldr_biomedrxiv:
                return {}
            try:
                conn = _get_db_connection()
                if not conn:
                    return {}
                with conn.cursor() as cur:
                    # document_id is text (UUIDs for biorxiv/medrxiv,
                    # arxiv IDs like "2507.07988" for arxiv). A plain
                    # text-array ANY hits the btree index on document_id.
                    cur.execute(
                        """SELECT document_id, tldr, month_year, pub_date
                           FROM documents WHERE document_id = ANY(%s)""",
                        (needs_tldr_biomedrxiv,),
                    )
                    out: dict[str, dict] = {}
                    for row in cur.fetchall():
                        pub_date = row[3]
                        out[row[0]] = {
                            "tldr": row[1] or "",
                            "month_year": row[2] or "",
                            "pub_date": str(pub_date) if pub_date else "",
                        }
                    return out
            except Exception as e:
                logger.warning(f"Bio TL;DR hydration failed: {e}")
                return {}

        def _fetch_tldr_pmc() -> dict[str, dict]:
            if not needs_tldr_pmc:
                return {}
            module = _get_papers_module()
            if not module:
                return {}
            try:
                conn = module._get_pmc_db_connection()
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT pmc_id, tldr, pub_date
                           FROM documents WHERE pmc_id = ANY(%s)""",
                        (needs_tldr_pmc,),
                    )
                    out: dict[str, dict] = {}
                    for row in cur.fetchall():
                        pub_date = row[2]
                        out[row[0]] = {
                            "tldr": row[1] or "",
                            "pub_date": str(pub_date) if pub_date else "",
                        }
                    return out
            except Exception as e:
                logger.warning(f"PMC TL;DR hydration failed: {e}")
                return {}

        os_mget_task = asyncio.to_thread(_os_mget)
        tldr_bio_task = asyncio.to_thread(_fetch_tldr_bio)
        tldr_pmc_task = asyncio.to_thread(_fetch_tldr_pmc)
        os_meta, tldr_bio, tldr_pmc = await asyncio.gather(
            os_mget_task, tldr_bio_task, tldr_pmc_task
        )

        # 5. Merge everything back into the result rows.
        all_tldr = {**tldr_bio, **tldr_pmc}
        for r in results:
            doc_id = r.get("document_id")
            if not doc_id:
                continue
            os_fields = os_meta.get(doc_id)
            if os_fields:
                for k, v in os_fields.items():
                    r.setdefault(k, v)
            pg_fields = all_tldr.get(doc_id)
            if pg_fields:
                for k, v in pg_fields.items():
                    r.setdefault(k, v)
            # abstract_snippet for back-compat callers (UI + legacy code).
            if not r.get("abstract_snippet"):
                tldr = r.get("tldr") or ""
                abstract = r.get("abstract") or ""
                if ABSTRACT_SNIPPET_LEN > 0 and abstract:
                    abstract = abstract[:ABSTRACT_SNIPPET_LEN]
                r["abstract_snippet"] = tldr if tldr else abstract
            # Populate pub_year from pub_date when OS returned only one.
            if not r.get("pub_year") and r.get("pub_date"):
                try:
                    r["pub_year"] = int(str(r["pub_date"])[:4])
                except ValueError:
                    pass
            # Cache final metadata blob for future requests.
            if len(_hydration_cache) < _HYDRATION_CACHE_MAX:
                _hydration_cache[doc_id] = {
                    "title": r.get("title") or "",
                    "doi": r.get("doi") or "",
                    "authors": r.get("authors") or "",
                    "source": r.get("source") or "",
                    "pub_year": r.get("pub_year"),
                    "pub_date": r.get("pub_date") or "",
                    "abstract_snippet": r.get("abstract_snippet") or "",
                    "tldr": r.get("tldr") or "",
                    "month_year": r.get("month_year") or "",
                    "journal": r.get("journal") or r.get("journal_title") or "",
                    "article_type": r.get("article_type") or "",
                }

        return results

    async def _deep_search_documents(
        self,
        query: str,
        filters: dict | None,
        limit: int,
    ) -> list[dict]:
        """Deep search: full-text content blocks (deprecated).

        The old ``biomedrxiv_content`` / ``pmc_content`` OpenSearch indices
        have been retired in favour of Postgres-native full-text search
        (``_search_pg_bm25`` and the ``grep`` tool). This wrapper now always
        returns an empty list so that ``depth=deep`` callers degrade
        gracefully; use ``paperclip grep`` or ``_search_pg_bm25`` for actual
        full-text queries.
        """
        logger.debug(
            "_deep_search_documents is deprecated (content indices removed); "
            "returning []."
        )
        return []

        # Unreachable legacy body kept for reference.
        es = _get_es_client()  # type: ignore[unreachable]
        if not es:
            return []

        raw_sources = filters.get("source") if filters else None
        if isinstance(raw_sources, str):
            raw_sources = [raw_sources]
        _, content_indices = self._get_search_indices(raw_sources)
        index_str = ",".join(content_indices)

        search_mode = filters.get("search_mode", "any") if filters else "any"

        if search_mode == "phrase":
            text_query = {"match_phrase": {"content": {"query": query}}}
        elif search_mode == "all":
            text_query = {"match": {"content": {"query": query, "operator": "and"}}}
        else:
            text_query = {"match": {"content": {"query": query}}}

        # Also boost matches in title
        es_query = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {"title": {"query": query, "boost": 3}}},
                        text_query,
                    ],
                    "minimum_should_match": 1,
                }
            },
            # Collapse on document identifier to get one result per paper
            "collapse": {
                "field": "document_id",
                "inner_hits": {
                    "name": "best_block",
                    "size": 1,
                    "_source": ["content", "section"],
                },
            },
            "size": limit,
            "_source": [
                "document_id",
                "pmc_id",
                "title",
                "doi",
                "authors",
                "source",
                "pub_year",
            ],
        }

        try:
            import asyncio

            response = await asyncio.to_thread(
                es.search, index=index_str, body=es_query
            )
            results = []
            for hit in response["hits"]["hits"]:
                src = hit["_source"]
                doc_id = src.get("document_id") or src.get("pmc_id")
                best = (
                    hit.get("inner_hits", {})
                    .get("best_block", {})
                    .get("hits", {})
                    .get("hits", [])
                )
                snippet = best[0]["_source"].get("content", "")[:150] if best else ""
                section = best[0]["_source"].get("section", "") if best else ""
                results.append(
                    {
                        "document_id": doc_id,
                        "title": src.get("title", ""),
                        "doi": src.get("doi", ""),
                        "authors": src.get("authors", ""),
                        "month_year": str(src.get("pub_year", "")),
                        "source": src.get("source", "pmc" if src.get("pmc_id") else ""),
                        "abstract_snippet": (
                            f"[{section}] {snippet}" if section else snippet
                        ),
                        "score": hit.get("_score"),
                    }
                )
            return results
        except Exception as e:
            logger.warning(f"Deep search failed ({index_str}): {e}")
            return []

    async def search_documents(
        self,
        query: str = None,
        filters: dict = None,
        limit: int = 25,
        document_ids: list[str] = None,
    ) -> list[dict]:
        """Search for papers using Elasticsearch.

        Supported filters (via ``filters`` dict):
          - search_mode: "any", "all", "phrase", "50%", "75%"
          - source: "biorxiv" | "medrxiv" | "pmc" | "abstracts" | list of those | "all"
          - since: duration string like "30d", "7d", "6m", "1y"
          - category: bioRxiv subject area (e.g. "Neuroscience")
          - journal: journal name (ILIKE match, PMC only)
          - article_type: e.g. "research-article", "review-article" (PMC only)
          - year: publication year (e.g. "2024")
          - sort: "date" to sort by recency instead of relevance
          - depth: "shallow" (default, title+abstract) | "deep" (full text paragraphs)

        Args:
            document_ids: Optional list of document IDs to restrict search to.
                         Supports both bioRxiv UUIDs and PMC IDs (PMCnnnnn).
                         Used for grep | search chaining.
        """
        es = _get_es_client()
        if not es:
            from mcps.papers.servers.search_backends import get_qdrant_client as _gqc

            if not _gqc():
                raise RuntimeError(
                    "No search backend available. Set OPENSEARCH_URL and/or "
                    "QDRANT_URL (and unset PAPERS_DISABLE_OPENSEARCH / "
                    "PAPERS_DISABLE_QDRANT) to the gxl-search endpoint."
                )
            logger.warning(
                "OpenSearch unavailable — running vector-only (Qdrant). "
                "Unset PAPERS_DISABLE_OPENSEARCH or fix OPENSEARCH_URL to restore BM25."
            )

        if not filters:
            filters = {}

        # When scoped to specific document IDs (grep | search), bypass date filter
        if document_ids:
            filters["all_time"] = True

        # Default to 2024+ unless caller explicitly opts out or sets their own date filter
        if (
            not filters.get("since")
            and not filters.get("year")
            and not filters.get("all_time")
        ):
            filters.setdefault("since", "2y")

        search_mode = filters.get("search_mode", "any")

        must = []
        if query:
            if search_mode == "phrase":
                must.append(
                    {
                        "bool": {
                            "should": [
                                {
                                    "match_phrase": {
                                        "title": {"query": query, "boost": 3}
                                    }
                                },
                                {
                                    "match_phrase": {
                                        "abstract": {"query": query, "boost": 2}
                                    }
                                },
                                {
                                    "match_phrase": {
                                        "abstract_text": {"query": query, "boost": 2}
                                    }
                                },
                                {
                                    "match_phrase": {
                                        "abstract": {"query": query, "boost": 2}
                                    }
                                },
                            ],
                            "minimum_should_match": 1,
                        }
                    }
                )
            elif search_mode == "all":
                must.append(
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "title^3",
                                "abstract_text^2",
                                "abstract^2",
                                "authors",
                            ],
                            "type": "cross_fields",
                            "operator": "and",
                        }
                    }
                )
            elif search_mode in ("50%", "75%"):
                must.append(
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "title^3",
                                "abstract_text^2",
                                "abstract^2",
                                "authors",
                            ],
                            "type": "best_fields",
                            "minimum_should_match": search_mode,
                        }
                    }
                )
            else:
                must.append(
                    {
                        "bool": {
                            "should": [
                                {"match": {"title": {"query": query, "boost": 3}}},
                                {"match": {"abstract": {"query": query, "boost": 2}}},
                                {
                                    "match": {
                                        "abstract_text": {"query": query, "boost": 2}
                                    }
                                },
                                {"match": {"authors": {"query": query, "boost": 1}}},
                            ],
                            "minimum_should_match": 1,
                        }
                    }
                )

        # Determine which indices to search
        raw_sources = filters.get("source") if filters else None
        if isinstance(raw_sources, str):
            raw_sources = [raw_sources]
        doc_indices, content_indices = self._get_search_indices(raw_sources)

        filter_clauses = []

        # NOTE: Per-corpus source filtering (biorxiv/medrxiv/arxiv) is applied
        # inside the BM25 query builder below (based on the corpus, not on
        # filters["source"]). These cross-cutting filters apply to all hits.

        # Default: exclude non-research PMC articles (editorials, corrections,
        # etc.). This must not drop non-PMC hits, so we OR it with a non-pmc
        # source clause.
        if CORPUS_PMC in self._resolve_corpora_from_sources(
            filters.get("source") if filters else None
        ):
            filter_clauses.append(
                {
                    "bool": {
                        "should": [
                            self.PMC_ARTICLE_TYPE_FILTER,
                            {"terms": {"source": ["biorxiv", "medrxiv", "arxiv", "openalex"]}},
                        ],
                        "minimum_should_match": 1,
                    }
                }
            )

        if filters:
            # --since: unified pub_year range filter (works across all corpora)
            since = filters.get("since")
            if since:
                min_year = self._since_to_pub_year(since)
                if min_year:
                    filter_clauses.append({"range": {"pub_year": {"gte": min_year}}})

            # --category: try a direct term filter on the OS `categories` field
            # (bioRxiv has it; other corpora will simply return 0 hits if the
            # field is absent). Deeper category resolution via the old ES
            # content index has been removed.
            category = filters.get("category")
            if category:
                filter_clauses.append(
                    {
                        "bool": {
                            "should": [
                                {"match": {"categories": {"query": category, "operator": "and"}}},
                                {"match": {"category": {"query": category, "operator": "and"}}},
                            ],
                            "minimum_should_match": 1,
                        }
                    }
                )

            # --article-type: direct OS filter on the pmc index's article_type
            article_type = filters.get("article_type")
            if article_type and "pmc" in doc_indices:
                filter_clauses.append({"term": {"article_type": article_type}})

            # --year: direct OS range on pub_year (works for all corpora)
            year = filters.get("year")
            if year:
                filter_clauses.append({"term": {"pub_year": int(year)}})

            # --journal: two-phase — resolve pmc_ids from PMC PostgreSQL
            journal = filters.get("journal")
            if journal and "pmc" in doc_indices:
                pmc_ids = self._resolve_pmc_id_filters(journal=journal)
                if pmc_ids:
                    filter_clauses.append({"terms": {"pmc_id": pmc_ids}})
                elif pmc_ids is not None:
                    filter_clauses.append({"term": {"pmc_id": "__no_match__"}})

        # Scope to specific document IDs (for grep | search chaining)
        if document_ids:
            import re as _re

            resolved_ids = [resolve(d) for d in document_ids]
            bio_ids = [
                d for d in resolved_ids if not _re.match(r"^PMC\d+$", d, _re.IGNORECASE)
            ]
            pmc_ids_filter = [
                d for d in resolved_ids if _re.match(r"^PMC\d+$", d, _re.IGNORECASE)
            ]
            id_should = []
            if bio_ids:
                id_should.append({"terms": {"document_id": bio_ids}})
            if pmc_ids_filter:
                id_should.append({"terms": {"pmc_id": pmc_ids_filter}})
            if id_should:
                filter_clauses.append(
                    {"bool": {"should": id_should, "minimum_should_match": 1}}
                )
            logger.info(
                f"grep|search scoped to {len(bio_ids)} bio + {len(pmc_ids_filter)} pmc IDs"
            )

        # Sort: by date (recency) or default (relevance)
        sort_mode = filters.get("sort") if filters else None
        sort_clause = None
        if sort_mode == "date":
            sort_clause = [
                {"pub_year": {"order": "desc", "missing": "_last"}},
                "_score",
            ]

        es_query = {
            "query": {
                "bool": {
                    "must": must if must else [{"match_all": {}}],
                    "filter": filter_clauses,
                }
            },
            "size": limit,
            "_source": self._ES_SOURCE_FIELDS,
        }
        if sort_clause:
            es_query["sort"] = sort_clause

        index_str = ",".join(doc_indices)

        filters.pop("search_type", None)
        ranking = filters.pop("ranking", "hybrid")

        search_backend = os.getenv("SEARCH_BACKEND", "es")
        has_category = filters.get("category") if filters else False
        if (
            ranking == "bm25"
            and search_backend == "postgres"
            and query
            and not has_category
        ):
            raw_src = filters.get("source")
            bio_only = raw_src and "pmc" not in (
                raw_src if isinstance(raw_src, list) else [raw_src]
            )
            if bio_only:
                try:
                    pg_results = await self._search_pg_bm25(
                        query,
                        filters=filters,
                        limit=limit,
                        document_ids=document_ids,
                    )
                    if pg_results:
                        hydrated = await self._hydrate_results(pg_results)
                        deduped = self._deduplicate_by_doi(hydrated)
                        return self._dedup_cross_source(deduped)
                except Exception as e:
                    logger.warning(f"[bm25-pg] failed, falling back to ES: {e}")

        try:
            can_vector = query and not sort_clause

            raw_sources = filters.get("source") if filters else None
            if isinstance(raw_sources, str):
                raw_sources = [raw_sources]

            corpora = self._resolve_corpora_from_sources(raw_sources)

            # Per-call backend diagnostics. Surfaced via ``last_search_meta``
            # so the MCP tool layer (``_find``) can return it to the user and
            # prove which backend served the request.
            from mcps.papers.servers.search_backends import (
                opensearch_url as _os_url,
                qdrant_url as _q_url,
            )

            _meta: dict[str, Any] = {
                "ranking": ranking,
                "corpora": list(corpora),
                "opensearch": {
                    "url": _os_url(),
                    "enabled": bool(es) and _os_url() is not None,
                    "ms": None,  # wall-clock for the OS call itself
                    "hits": 0,
                    "error": None,
                },
                "qdrant": {
                    "url": _q_url(),
                    "enabled": _q_url() is not None,
                    "ms": None,  # total (embed + search)
                    "embed_ms": None,  # Gemini embedding API
                    "search_ms": None,  # just the Qdrant HTTP call
                    "hits": 0,
                    "error": None,
                },
                "hydrate_ms": None,
                "total_ms": None,
            }
            _meta_t0 = time.perf_counter()

            async def _bm25_es():
                """OpenSearch BM25 across all requested corpora (per-corpus source filter)."""
                if not es:
                    _meta["opensearch"]["error"] = "client-disabled"
                    return []
                if not corpora:
                    return []
                per_source = max(limit, 15) if len(corpora) > 1 else limit

                def _build_body_for_corpus(corpus: str) -> dict:
                    src_filter = OS_SOURCE_FILTER_BY_CORPUS.get(corpus)
                    extra_filters: list[dict] = []
                    if src_filter:
                        extra_filters.append({"terms": {"source": src_filter}})
                    body = {
                        "query": {
                            "bool": {
                                "must": es_query["query"]["bool"]["must"],
                                "filter": filter_clauses + extra_filters,
                            }
                        },
                        "size": per_source,
                        "_source": self._os_source_for_corpus(corpus),
                    }
                    if sort_clause:
                        body["sort"] = sort_clause
                    return body

                pairs = [
                    (OS_INDEX_BY_CORPUS[c], _build_body_for_corpus(c)) for c in corpora
                ]

                def _run():
                    try:
                        responses = es.msearch_pairs(pairs)
                    except Exception as exc:
                        logger.warning(
                            f"OpenSearch msearch failed, falling back to sequential: {exc}"
                        )
                        responses = []
                        for idx, body in pairs:
                            try:
                                responses.append(es.search(index=idx, body=body))
                            except Exception as e:
                                logger.warning(f"OpenSearch search on {idx} failed: {e}")
                                responses.append({"hits": {"hits": []}})
                    per_corpus_hits: list[list[dict]] = []
                    for (idx, _), resp_item in zip(pairs, responses):
                        if "error" in resp_item:
                            logger.warning(
                                f"OpenSearch msearch error on {idx}: {resp_item['error']}"
                            )
                            per_corpus_hits.append([])
                            continue
                        per_corpus_hits.append(
                            [
                                self._normalise_os_hit(h)
                                for h in resp_item.get("hits", {}).get("hits", [])
                            ]
                        )
                    # Interleave for balanced source representation.
                    merged: list[dict] = []
                    iters = [iter(hs) for hs in per_corpus_hits if hs]
                    while len(merged) < limit and iters:
                        drained: list[int] = []
                        for i, it in enumerate(iters):
                            nxt = next(it, None)
                            if nxt is None:
                                drained.append(i)
                            else:
                                merged.append(nxt)
                                if len(merged) >= limit:
                                    break
                        for i in reversed(drained):
                            iters.pop(i)
                    return merged[:limit]

                _t0 = time.perf_counter()
                try:
                    hits = await asyncio.to_thread(_run)
                except Exception as exc:
                    _meta["opensearch"]["error"] = str(exc)
                    _meta["opensearch"]["ms"] = round(
                        (time.perf_counter() - _t0) * 1000
                    )
                    raise
                _meta["opensearch"]["ms"] = round(
                    (time.perf_counter() - _t0) * 1000
                )
                _meta["opensearch"]["hits"] = len(hits)
                logger.info(
                    "[opensearch] %s corpora=%s hits=%d in %sms",
                    _meta["opensearch"]["url"],
                    ",".join(corpora),
                    len(hits),
                    _meta["opensearch"]["ms"],
                )
                return hits

            async def _vector_v2():
                """Qdrant dense-vector search (3072-dim gemini-embedding-2-preview)."""
                if not can_vector:
                    return []
                if not _meta["qdrant"]["enabled"]:
                    _meta["qdrant"]["error"] = "client-disabled"
                    return []
                _t0 = time.perf_counter()
                _t_embed = time.perf_counter()
                vec = await asyncio.to_thread(
                    self._embed_query,
                    query,
                    model="gemini-embedding-2-preview",
                    dims=3072,
                )
                _meta["qdrant"]["embed_ms"] = round(
                    (time.perf_counter() - _t_embed) * 1000
                )
                if not vec:
                    _meta["qdrant"]["error"] = "embedding-unavailable"
                    _meta["qdrant"]["ms"] = round(
                        (time.perf_counter() - _t0) * 1000
                    )
                    return []
                _t_search = time.perf_counter()
                try:
                    hits = await self._qdrant_vector_search(
                        vec, limit=limit, sources=raw_sources
                    )
                except Exception as exc:
                    _meta["qdrant"]["error"] = str(exc)
                    _meta["qdrant"]["search_ms"] = round(
                        (time.perf_counter() - _t_search) * 1000
                    )
                    _meta["qdrant"]["ms"] = round(
                        (time.perf_counter() - _t0) * 1000
                    )
                    raise
                _meta["qdrant"]["search_ms"] = round(
                    (time.perf_counter() - _t_search) * 1000
                )
                _meta["qdrant"]["ms"] = round(
                    (time.perf_counter() - _t0) * 1000
                )
                _meta["qdrant"]["hits"] = len(hits)
                logger.info(
                    "[qdrant] %s hits=%d embed=%sms search=%sms total=%sms",
                    _meta["qdrant"]["url"],
                    len(hits),
                    _meta["qdrant"]["embed_ms"],
                    _meta["qdrant"]["search_ms"],
                    _meta["qdrant"]["ms"],
                )
                return hits

            if ranking == "vector":
                results = await _vector_v2() or await _bm25_es()
            elif ranking == "hybrid":

                async def _safe_vector():
                    try:
                        return await _vector_v2()
                    except Exception as e:
                        logger.warning(
                            f"[hybrid] vector search failed (BM25 only): {e}"
                        )
                        return []

                bm25_results, vec_results = await asyncio.gather(
                    _bm25_es(), _safe_vector()
                )

                if vec_results and bm25_results:
                    results = _rrf_merge(bm25_results, vec_results, limit=limit)
                else:
                    results = bm25_results or vec_results or []
            else:
                results = await _bm25_es()

            _t_hy = time.perf_counter()
            hydrated = await self._hydrate_results(results)
            _meta["hydrate_ms"] = round((time.perf_counter() - _t_hy) * 1000)
            deduped = self._deduplicate_by_doi(hydrated)
            _meta["returned"] = len(deduped)
            _meta["total_ms"] = round((time.perf_counter() - _meta_t0) * 1000)
            self.last_search_meta = _meta
            return self._dedup_cross_source(deduped)
        except Exception as e:
            self.last_search_meta = _meta
            logger.error(
                f"OpenSearch search failed ({index_str}): {e}", exc_info=True
            )
            raise RuntimeError(f"Search failed: {e}") from e

    async def search_deep(
        self,
        query: str,
        limit: int = 25,
        min_match: str = "50%",
    ) -> list[dict]:
        """Alias for search_documents (which now includes deep content-block search)."""
        return await self.search_documents(query=query, limit=limit)

    def _parse_es_hits(self, response: dict) -> list[dict]:
        """Parse ES response hits — supports both biomedrxiv_documents and pmc_documents."""
        return [self._normalise_es_hit(h) for h in response["hits"]["hits"]]

    def _build_es_query(
        self, query: str, search_mode: str = "any", limit: int = 25
    ) -> dict:
        """Build ES query body for a search term."""
        must = []
        if query:
            if search_mode == "phrase":
                must.append(
                    {
                        "bool": {
                            "should": [
                                {
                                    "match_phrase": {
                                        "title": {"query": query, "boost": 3}
                                    }
                                },
                                {
                                    "match_phrase": {
                                        "abstract": {"query": query, "boost": 2}
                                    }
                                },
                                {
                                    "match_phrase": {
                                        "abstract_text": {"query": query, "boost": 2}
                                    }
                                },
                                {
                                    "match_phrase": {
                                        "abstract": {"query": query, "boost": 2}
                                    }
                                },
                            ],
                            "minimum_should_match": 1,
                        }
                    }
                )
            elif search_mode == "all":
                must.append(
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "title^3",
                                "abstract_text^2",
                                "abstract^2",
                                "authors",
                            ],
                            "type": "cross_fields",
                            "operator": "and",
                        }
                    }
                )
            elif search_mode in ("50%", "75%"):
                must.append(
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "title^3",
                                "abstract_text^2",
                                "abstract^2",
                                "authors",
                            ],
                            "type": "best_fields",
                            "minimum_should_match": search_mode,
                        }
                    }
                )
            else:
                must.append(
                    {
                        "bool": {
                            "should": [
                                {"match": {"title": {"query": query, "boost": 3}}},
                                {"match": {"abstract": {"query": query, "boost": 2}}},
                                {
                                    "match": {
                                        "abstract_text": {"query": query, "boost": 2}
                                    }
                                },
                                {"match": {"authors": {"query": query, "boost": 1}}},
                            ],
                            "minimum_should_match": 1,
                        }
                    }
                )

        return {
            "query": {
                "bool": {
                    "must": must if must else [{"match_all": {}}],
                }
            },
            "size": limit,
            "_source": [
                "document_id",
                "title",
                "doi",
                "authors",
                "month_year",
                "source",
            ],
        }

    async def combined_search_documents(
        self,
        queries: list[str],
        search_mode: str = "any",
        limit: int = 200,
        depth: str = "shallow",
        sources: list[str] | None = None,
    ) -> dict:
        """Run multiple search terms as a single ES bool.should query.

        Each term becomes a should clause so documents matching more terms
        score higher.  Returns a single ranked result set.

        depth: "shallow" (title+abstract, default) | "deep" (full text paragraphs)
        """
        es = _get_es_client()
        if not es:
            papers = await self._search_postgres(
                " OR ".join(queries), {"search_mode": search_mode}, limit
            )
            return {"total": len(papers), "papers": papers, "error": None}

        should_clauses = []
        for q in queries:
            if search_mode == "phrase":
                should_clauses.append(
                    {
                        "bool": {
                            "should": [
                                {"match_phrase": {"title": {"query": q, "boost": 3}}},
                                {
                                    "match_phrase": {
                                        "abstract": {"query": q, "boost": 2}
                                    }
                                },
                                {
                                    "match_phrase": {
                                        "abstract_text": {"query": q, "boost": 2}
                                    }
                                },
                                {
                                    "match_phrase": {
                                        "abstract": {"query": q, "boost": 2}
                                    }
                                },
                            ],
                            "minimum_should_match": 1,
                        }
                    }
                )
            else:
                should_clauses.append(
                    {
                        "bool": {
                            "should": [
                                {"match": {"title": {"query": q, "boost": 3}}},
                                {
                                    "bool": {
                                        "should": [
                                            {
                                                "match": {
                                                    "abstract_text": {
                                                        "query": q,
                                                        "boost": 2,
                                                    }
                                                }
                                            },
                                            {
                                                "match": {
                                                    "abstract": {"query": q, "boost": 2}
                                                }
                                            },
                                        ],
                                        "minimum_should_match": 1,
                                    }
                                },
                                {"match": {"authors": {"query": q, "boost": 1}}},
                            ],
                            "minimum_should_match": 1,
                        }
                    }
                )

        doc_indices, _ = self._get_search_indices(sources)

        filter_clauses = []
        if "pmc" in doc_indices:
            filter_clauses.append(
                {
                    "bool": {
                        "should": [
                            self.PMC_ARTICLE_TYPE_FILTER,
                            {"term": {"source": "biorxiv"}},
                            {"term": {"source": "medrxiv"}},
                        ],
                        "minimum_should_match": 1,
                    }
                }
            )

        body = {
            "query": {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1,
                    "filter": filter_clauses,
                }
            },
            "size": limit,
            "_source": [
                "document_id",
                "pmc_id",
                "title",
                "doi",
                "authors",
                "month_year",
                "pub_year",
                "source",
            ],
        }

        try:
            import asyncio

            response = await asyncio.to_thread(
                _execute_es_with_retry,
                lambda client: client.search(
                    index=",".join(doc_indices),
                    body=body,
                ),
                operation_name="combined document search",
            )
            papers = [self._normalise_es_hit(h) for h in response["hits"]["hits"]]
            papers = await self._hydrate_results(papers)
            papers = self._deduplicate_by_doi(papers)
            papers = self._dedup_cross_source(papers)
            total = response["hits"]["total"]
            if isinstance(total, dict):
                total = total.get("value", len(papers))
            return {"total": total, "papers": papers, "error": None}
        except Exception as e:
            logger.warning(f"ES combined search failed: {e}")
            return {"total": 0, "papers": [], "error": str(e)}

    async def msearch_documents(
        self,
        queries: list[str],
        search_mode: str = "any",
        limit: int = 25,
    ) -> list[dict]:
        """Execute multiple searches in a single ES request using msearch API.

        Returns list of results, one per query:
        [
            {"query": "CRISPR", "total": 47, "papers": [...], "error": None},
            {"query": "cancer", "total": 0, "papers": [], "error": None},
            ...
        ]
        """
        es = _get_es_client()
        if not es:
            # Fallback to sequential searches
            results = []
            for q in queries:
                try:
                    papers = await self._search_postgres(
                        q, {"search_mode": search_mode}, limit
                    )
                    results.append(
                        {
                            "query": q,
                            "total": len(papers),
                            "papers": papers,
                            "error": None,
                        }
                    )
                except Exception as e:
                    results.append(
                        {"query": q, "total": 0, "papers": [], "error": str(e)}
                    )
            return results

        # Build msearch body (alternating header + query)
        # Use both indices so PMC and preprint results are interleaved by relevance.
        msearch_body = []
        for query in queries:
            msearch_body.append({"index": f"{PREPRINTS_OS_INDEX},pmc"})
            msearch_body.append(self._build_es_query(query, search_mode, limit))

        try:
            import asyncio

            response = await asyncio.to_thread(
                _execute_es_with_retry,
                lambda client: client.msearch(body=msearch_body),
                operation_name="document msearch",
            )

            # Collect all papers across queries for batch hydration
            all_papers = []
            query_slices = []  # (query, total, start_idx, end_idx) or (query, error)
            for query, res in zip(queries, response["responses"]):
                if "error" in res:
                    query_slices.append((query, 0, -1, -1, str(res["error"])))
                else:
                    papers = [self._normalise_es_hit(h) for h in res["hits"]["hits"]]
                    start_idx = len(all_papers)
                    all_papers.extend(papers)
                    end_idx = len(all_papers)
                    total = res["hits"]["total"]
                    if isinstance(total, dict):
                        total = total.get("value", len(papers))
                    query_slices.append((query, total, start_idx, end_idx, None))

            # Batch hydrate all papers at once
            if all_papers:
                all_papers = await self._hydrate_results(all_papers)

            results = []
            for query, total, start, end, error in query_slices:
                if error is not None:
                    results.append(
                        {"query": query, "total": 0, "papers": [], "error": error}
                    )
                else:
                    papers = self._deduplicate_by_doi(all_papers[start:end])
                    results.append(
                        {
                            "query": query,
                            "total": total,
                            "papers": papers,
                            "error": None,
                        }
                    )
            return results
        except Exception as e:
            logger.warning(f"ES msearch failed: {e}")
            return [
                {"query": q, "total": 0, "papers": [], "error": str(e)} for q in queries
            ]

    async def _search_postgres(
        self,
        query: str = None,
        filters: dict = None,
        limit: int = 25,
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
                src = filters["source"]
                if isinstance(src, list):
                    conditions.append("source = ANY(%s)")
                    params.append(src)
                else:
                    conditions.append("source = %s")
                    params.append(src)

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
        return self._deduplicate_by_doi(results)[:limit]

    # ------------------------------------------------------------------
    # BM25 search via PostgreSQL (replaces Elasticsearch for ranking)
    # ------------------------------------------------------------------

    _corpus_stats_cache: tuple[int, float, float] | None = None  # (N, avgdl, ts)

    def _get_corpus_stats(self) -> tuple[int, float]:
        """Return (total_docs, avg_doc_length), cached for 1 hour."""
        now = time.time()
        if self._corpus_stats_cache and (now - self._corpus_stats_cache[2]) < 3600:
            return self._corpus_stats_cache[0], self._corpus_stats_cache[1]

        conn = _get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT total_docs, avg_doc_length FROM search_corpus_stats WHERE id=1"
                )
                row = cur.fetchone()
                if row:
                    N, avgdl = int(row[0]), float(row[1])
                else:
                    # Fast estimate from pg_class (no table scan)
                    cur.execute(
                        """
                        SELECT reltuples::int FROM pg_class WHERE relname = 'documents'
                    """
                    )
                    r = cur.fetchone()
                    N = int(r[0]) if r and r[0] > 0 else 470000
                    # Average tsvector length for title+abstract is ~40-60 terms
                    avgdl = 50.0
        except Exception as e:
            logger.warning(f"[bm25] corpus stats failed: {e}")
            N, avgdl = 470000, 50.0

        self._corpus_stats_cache = (N, avgdl, now)
        return N, avgdl

    async def _search_pg_bm25(
        self,
        query: str,
        filters: dict | None = None,
        limit: int = 25,
        document_ids: list[str] | None = None,
    ) -> list[dict]:
        """BM25-ranked search using PostgreSQL GIN index + Python re-ranking.

        Flow:
          1. Build tsquery from query + search_mode
          2. GIN retrieval with SQL filters (source, since, year, etc.)
          3. Get document frequency for each query term
          4. Compute BM25 scores in Python
          5. Return top-k results (document_id, source, score)
        """
        import math

        if not filters:
            filters = {}

        conn = _get_db_connection()
        search_mode = filters.get("search_mode", "any")
        sort_mode = filters.get("sort")

        # --- Build tsquery ---
        # Use AND retrieval for all modes to get high-quality candidates,
        # then ts_rank_cd handles scoring (weights: title=A, abstract=B).
        # For "phrase" mode, use exact phrase matching.
        min_match_frac = None
        if search_mode == "phrase":
            tsquery_expr = "phraseto_tsquery('english', %s)"
        elif search_mode in ("50%", "75%"):
            tsquery_expr = "websearch_to_tsquery('english', %s)"
            min_match_frac = 0.5 if search_mode == "50%" else 0.75
        else:
            # "any" and "all" both use AND for GIN retrieval
            # ts_rank_cd scoring handles relevance differentiation
            tsquery_expr = "plainto_tsquery('english', %s)"

        # --- Build WHERE clauses ---
        # Separate tsquery match clause from filter clauses
        # so the retrieve method can use a CTE for the tsquery
        filter_parts: list[str] = []
        filter_params: list = []

        if (
            not filters.get("since")
            and not filters.get("year")
            and not filters.get("all_time")
        ):
            filters.setdefault("since", "2y")

        if filters.get("source") and filters["source"] not in ("all",):
            src = filters["source"]
            if isinstance(src, list):
                filter_parts.append("d.source = ANY(%s)")
                filter_params.append(src)
            else:
                filter_parts.append("d.source = %s")
                filter_params.append(src)

        since = filters.get("since")
        if since:
            min_date = self._since_to_date(since)
            if min_date:
                filter_parts.append("d.pub_date >= %s")
                filter_params.append(min_date)

        year = filters.get("year")
        if year:
            filter_parts.append("EXTRACT(YEAR FROM d.pub_date) = %s")
            filter_params.append(int(year))

        if document_ids:
            resolved = [resolve(d) for d in document_ids]
            filter_parts.append("d.document_id::text = ANY(%s)")
            filter_params.append(resolved)

        filter_sql = (" AND " + " AND ".join(filter_parts)) if filter_parts else ""
        candidate_limit = min(max(limit * 20, 200), 500)

        try:
            t0 = time.perf_counter()

            candidates = await asyncio.to_thread(
                self._pg_bm25_retrieve,
                conn,
                tsquery_expr,
                query,
                filter_sql,
                filter_params,
                candidate_limit if sort_mode != "date" else limit,
                sort_mode,
            )
            retrieve_ms = (time.perf_counter() - t0) * 1000

            if not candidates:
                return []

            if sort_mode == "date":
                logger.info(
                    f"[bm25-pg] date sort: {len(candidates)} results in {retrieve_ms:.0f}ms"
                )
                return [
                    {"document_id": c["document_id"], "source": c["source"], "score": 0}
                    for c in candidates[:limit]
                ]

            # Candidates already ordered by ts_rank_cd from the DB.
            # Use tsrank directly as score — it's cover density ranking
            # which already provides good BM25-like relevance.
            # BM25 re-ranking is done async in background to warm the DF cache
            # for subsequent queries.
            t1 = time.perf_counter()

            if min_match_frac:
                terms = self._extract_tsquery_terms(conn, query, search_mode)
                n_terms = len(terms) if terms else 1
            else:
                terms = None

            scored = []
            for c in candidates:
                if min_match_frac and terms:
                    # For 50%/75% modes, check match fraction via tsrank > 0
                    # ts_rank_cd already ensures all AND/OR terms match via GIN,
                    # but in OR mode (websearch_to_tsquery), we may have partial matches
                    pass  # GIN with websearch_to_tsquery already handles this
                scored.append(
                    {
                        "document_id": c["document_id"],
                        "source": c["source"],
                        "score": c["tsrank"],
                    }
                )

            # Already sorted by tsrank DESC from DB query
            score_ms = (time.perf_counter() - t1) * 1000
            total_ms = (time.perf_counter() - t0) * 1000

            logger.info(
                f"[bm25-pg] query='{query[:40]}' candidates={len(candidates)} "
                f"retrieve={retrieve_ms:.0f}ms score={score_ms:.0f}ms total={total_ms:.0f}ms"
            )

            # Trigger async DF cache warming for future BM25 scoring
            if not terms:
                terms = self._extract_tsquery_terms(conn, query, search_mode)
            asyncio.get_event_loop().call_soon(
                lambda: asyncio.ensure_future(
                    asyncio.to_thread(self._get_term_dfs, conn, terms)
                )
            )

            return scored[:limit]

        except Exception as e:
            logger.error(f"[bm25-pg] search failed: {e}", exc_info=True)
            raise

    def _pg_bm25_retrieve(
        self,
        conn,
        tsquery_expr: str,
        query: str,
        filter_sql: str,
        filter_params: list,
        limit: int,
        sort_mode: str | None,
    ) -> list[dict]:
        """Synchronous GIN candidate retrieval (runs in thread pool).

        Uses a CTE to compute the tsquery once, then filters + ranks.
        """
        if sort_mode == "date":
            sql = f"""
                WITH q AS (SELECT {tsquery_expr} AS tsq)
                SELECT d.document_id::text, d.source,
                       length(d.search_vector) AS doc_term_count,
                       0::float AS tsrank
                FROM documents d, q
                WHERE d.search_vector @@ q.tsq {filter_sql}
                ORDER BY d.pub_date DESC NULLS LAST
                LIMIT %s
            """
        else:
            sql = f"""
                WITH q AS (SELECT {tsquery_expr} AS tsq)
                SELECT d.document_id::text, d.source,
                       length(d.search_vector) AS doc_term_count,
                       ts_rank_cd(d.search_vector, q.tsq) AS tsrank
                FROM documents d, q
                WHERE d.search_vector @@ q.tsq {filter_sql}
                ORDER BY tsrank DESC
                LIMIT %s
            """

        # query param goes first (for the CTE), then filter params, then limit
        all_params = [query] + filter_params + [limit]
        with conn.cursor() as cur:
            cur.execute(sql, all_params)
            rows = cur.fetchall()

        return [
            {
                "document_id": doc_id,
                "source": source or "biorxiv",
                "doc_term_count": dtc,
                "tsrank": float(tsrank),
            }
            for doc_id, source, dtc, tsrank in rows
        ]

    _term_df_cache: dict[str, tuple[int, float]] = {}  # term -> (df, timestamp)

    def _get_term_dfs(self, conn, terms: list[str]) -> dict[str, int]:
        """Get document frequency for each term, with 24-hour cache.

        Uses a fast estimation approach: count matching rows with LIMIT to
        cap scan time, then use the GIN index for efficient lookups.
        Falls back to a conservative estimate if the query times out.
        """
        now = time.time()
        result = {}
        uncached = []

        for term in terms:
            cached = self._term_df_cache.get(term)
            if cached and (now - cached[1]) < 86400:
                result[term] = cached[0]
            else:
                uncached.append(term)

        if uncached:
            # Batch estimate: use a single query with ts_stat on a sample
            # or fall back to per-term EXPLAIN-based estimates
            for term in uncached:
                try:
                    with conn.cursor() as cur:
                        # Use EXPLAIN to get row estimate from GIN index
                        # This is O(1) — reads from the planner stats, not actual data
                        cur.execute(
                            "EXPLAIN (FORMAT JSON) SELECT 1 FROM documents "
                            "WHERE search_vector @@ to_tsquery('english', %s)",
                            (term,),
                        )
                        plan = cur.fetchone()[0]
                        # Extract estimated rows from the plan
                        df = int(plan[0]["Plan"].get("Plan Rows", 1))
                        result[term] = max(df, 1)
                        self._term_df_cache[term] = (max(df, 1), now)
                except Exception:
                    result[term] = 1

        return result

    @staticmethod
    def _query_to_or_tsquery(query: str) -> str:
        """Convert 'CRISPR gene therapy' → 'CRISPR | gene | therapy' for OR tsquery."""
        words = query.split()
        if not words:
            return query
        return " | ".join(words)

    def _extract_tsquery_terms(self, conn, query: str, search_mode: str) -> list[str]:
        """Parse a user query into individual stemmed terms via Postgres."""
        if search_mode == "phrase":
            func = "phraseto_tsquery"
        elif search_mode == "all":
            func = "plainto_tsquery"
        else:
            func = "websearch_to_tsquery"

        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT {func}('english', %s)::text", (query,))
                tsq_text = cur.fetchone()[0]
            if not tsq_text:
                return []
            terms = []
            for token in re.split(r"[&|!<>\s()]+", tsq_text):
                token = token.strip("' ")
                if token and not token.startswith(":"):
                    terms.append(token)
            return terms
        except Exception as e:
            logger.warning(f"[bm25] term extraction failed: {e}")
            return query.lower().split()[:8]

    async def grep_content(
        self,
        regex: str,
        document_ids: list[str] = None,
        section_filter: str = None,
        limit: int = 50,
        exhaustive: bool = False,
        source_filter: str = None,
    ) -> list[dict]:
        """Regex search on paper content.

        Corpus-wide search tries the slab-grep service first (sub-second),
        falls back to PG tsvector pipeline if slab-grep is unavailable.

        When document_ids is provided, uses direct PG regex on content_blocks.
        Set exhaustive=True to process all P1 candidates without early exit.

        source_filter: "biomedrxiv"/"biorxiv" for bio-only, "pmc" for pmc-only,
                       None or "all" for both.
        """
        import re as _re

        # Corpus-wide: try bitmap-grep slab service first (sub-second, in-memory)
        if document_ids is None:
            slab_result = await self._try_slab_grep(
                regex, section_filter, limit, source_filter=source_filter
            )
            if slab_result is not None:
                return slab_result
            # Fallback: PG tsvector pipeline
            return await self._grep_tsv_pg(
                regex,
                section_filter,
                limit,
                exhaustive=exhaustive,
                source_filter=source_filter,
            )

        from modules.papers.short_ids import is_arxiv_id, bare_arxiv_id

        pmc_ids = [d for d in document_ids if _re.match(r"^PMC\d+$", d, _re.IGNORECASE)]
        arxiv_ids = [d for d in document_ids if d not in pmc_ids and is_arxiv_id(d)]
        bio_ids = [d for d in document_ids if d not in pmc_ids and d not in arxiv_ids]

        def _search_arxiv():
            if not arxiv_ids:
                return []
            module = _get_papers_module()
            if not module:
                return []
            arxiv_conn = module._get_arxiv_db_connection()
            bare_ids = [bare_arxiv_id(d) for d in arxiv_ids]
            conditions = ["content ~* %s"]
            params = [regex]
            conditions.append("document_id = ANY(%s)")
            params.append(bare_ids)
            if section_filter:
                conditions.append("(section ILIKE %s OR block_type ILIKE %s)")
                params.extend([f"%{section_filter}%", f"%{section_filter}%"])
            with arxiv_conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = '{GREP_TIMEOUT_SECONDS * 1000}'")
                cur.execute(
                    f"""
                    SELECT document_id, line_number, content, section, block_type
                    FROM content_blocks
                    WHERE {' AND '.join(conditions)}
                    ORDER BY document_id, line_number LIMIT %s
                """,
                    params + [limit * 2],
                )
                rows = cur.fetchall()
                cur.execute("RESET statement_timeout")
            return rows

        def _search_bio():
            if not bio_ids:
                return []
            conn = _get_db_connection()
            conditions = ["content ~* %s"]
            params = [regex]
            conditions.append("document_id::text = ANY(%s)")
            params.append(bio_ids)
            if section_filter:
                conditions.append("(section ILIKE %s OR block_type ILIKE %s)")
                params.extend([f"%{section_filter}%", f"%{section_filter}%"])
            with conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = '{GREP_TIMEOUT_SECONDS * 1000}'")
                cur.execute(
                    f"""
                    SELECT document_id::text, line_number, content, section, block_type
                    FROM content_blocks
                    WHERE {' AND '.join(conditions)}
                    ORDER BY document_id, line_number LIMIT %s
                """,
                    params + [limit * 2],
                )
                rows = cur.fetchall()
                cur.execute("RESET statement_timeout")
            return rows

        def _search_pmc():
            if not pmc_ids or "pmc" not in ENABLED_SOURCES:
                return []
            module = _get_papers_module()
            if not module:
                return []
            pmc_conn = module._get_pmc_db_connection()
            conditions = ["content ~* %s"]
            params = [regex]
            conditions.append("pmc_id = ANY(%s)")
            params.append(pmc_ids)
            if section_filter:
                conditions.append("(section ILIKE %s OR block_type ILIKE %s)")
                params.extend([f"%{section_filter}%", f"%{section_filter}%"])
            with pmc_conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = '{GREP_TIMEOUT_SECONDS * 1000}'")
                cur.execute(
                    f"""
                    SELECT pmc_id, line_number, content, section, block_type
                    FROM content_blocks
                    WHERE {' AND '.join(conditions)}
                    ORDER BY pmc_id, line_number LIMIT %s
                """,
                    params + [limit * 2],
                )
                rows = cur.fetchall()
                cur.execute("RESET statement_timeout")
            return rows

        def _run_both():
            bio_rows = _search_bio()
            pmc_rows = _search_pmc()
            arxiv_rows = _search_arxiv()
            all_rows = list(bio_rows) + list(pmc_rows) + list(arxiv_rows)
            out = []
            for row in all_rows:
                doc_id = row[0]
                line_num = row[1]
                content = row[2] or ""
                try:
                    match_obj = _re.search(regex, content, _re.IGNORECASE)
                    match_text = match_obj.group(0) if match_obj else content[:100]
                except _re.error:
                    match_text = content[:100]
                is_pmc = _re.match(r"^PMC\d+$", doc_id, _re.IGNORECASE)
                out.append(
                    {
                        "document_id": doc_id if is_pmc else shorten(doc_id),
                        "line_number": line_num,
                        "content": content[:300],
                        "match": match_text,
                        "section": row[3],
                        "block_type": row[4],
                    }
                )
            return out

        return _execute_with_retry(_run_both)

    async def _grep_trigram(
        self,
        regex: str,
        section_filter: str = None,
        limit: int = 50,
    ) -> list[dict]:
        """Three-phase ES trigram grep over the entire corpus.

        Phase 1: Query paper_trigram index with n-gram term queries
                 to find candidate paper IDs.
        Phase 2: Fetch content blocks for candidates from
                 biomedrxiv_content / pmc_content.
        Phase 3: Python regex verification on fetched blocks.
        """
        import re as _re
        import time as _time

        t0 = _time.perf_counter()
        es = _get_es_client()
        if not es:
            return []

        # --- Extract literal runs and generate n-grams ---
        literal_runs = _re.findall(r"[a-zA-Z0-9]{3,}", regex)
        if not literal_runs:
            return []

        trigrams_by_word: list[list[str]] = []
        fourgrams_by_word: list[list[str]] = []
        for run in literal_runs:
            low = run.lower()
            tris = [low[i : i + 3] for i in range(len(low) - 2)]
            fours = [low[i : i + 4] for i in range(len(low) - 3)]
            if tris:
                trigrams_by_word.append(tris)
            if fours:
                fourgrams_by_word.append(fours)

        # Round-robin select n-grams across words for balanced representation
        def _round_robin(word_lists: list[list[str]], max_clauses: int) -> list[str]:
            selected = []
            indices = [0] * len(word_lists)
            while len(selected) < max_clauses:
                added = False
                for wi, grams in enumerate(word_lists):
                    if indices[wi] < len(grams) and len(selected) < max_clauses:
                        g = grams[indices[wi]]
                        if g not in selected:
                            selected.append(g)
                        indices[wi] += 1
                        added = True
                if not added:
                    break
            return selected

        MAX_CLAUSES = 6
        four_selected = (
            _round_robin(fourgrams_by_word, MAX_CLAUSES) if fourgrams_by_word else []
        )
        tri_selected = (
            _round_robin(trigrams_by_word, MAX_CLAUSES) if trigrams_by_word else []
        )

        # Prefer 4-grams (more selective), fill remaining with trigrams
        must_clauses = []
        for fg in four_selected:
            if len(must_clauses) < MAX_CLAUSES:
                must_clauses.append({"term": {"content.fourgram": fg}})
        for tg in tri_selected:
            if len(must_clauses) >= MAX_CLAUSES:
                break
            must_clauses.append({"term": {"content": tg}})

        if not must_clauses:
            return []

        # Phase 1: find candidate paper IDs per source (parallel)
        PER_SOURCE_CAP = 1000
        from concurrent.futures import ThreadPoolExecutor

        def _p1_query(source_filter: str) -> list[str]:
            body = {
                "bool": {
                    "must": list(must_clauses),
                    "filter": [{"term": {"source": source_filter}}],
                }
            }
            try:
                resp = es.search(
                    index="paper_trigram",
                    body={"query": body, "size": PER_SOURCE_CAP, "_source": False},
                    request_timeout=30,
                )
                return [h["_id"] for h in resp["hits"]["hits"]]
            except Exception as e:
                logger.warning(f"grep_trigram P1 error ({source_filter}): {e}")
                return []

        with ThreadPoolExecutor(max_workers=2) as pool:
            bio_fut = pool.submit(_p1_query, "biorxiv")
            pmc_fut = (
                pool.submit(_p1_query, "pmc") if "pmc" in ENABLED_SOURCES else None
            )
            bio_candidates = bio_fut.result()
            pmc_candidates = pmc_fut.result() if pmc_fut else []

        candidates = [(pid, "biorxiv") for pid in bio_candidates] + [
            (pid, "pmc") for pid in pmc_candidates
        ]

        t1 = _time.perf_counter()
        if not candidates:
            return []

        # Phase 2: fetch content blocks (parallel, capped per source)
        P2_PAPER_CAP = 200  # fetch blocks for top N papers per source

        def _fetch_blocks(index: str, id_field: str, paper_ids: list[str]):
            if not paper_ids:
                return []
            capped = paper_ids[:P2_PAPER_CAP]
            blocks = []
            try:
                resp = es.search(
                    index=index,
                    body={
                        "query": {"terms": {id_field: capped}},
                        "_source": [
                            id_field,
                            "content",
                            "section",
                            "block_type",
                            "line_number",
                        ],
                        "size": 10000,
                    },
                    request_timeout=15,
                )
                for hit in resp["hits"]["hits"]:
                    s = hit["_source"]
                    blocks.append(
                        {
                            "paper_id": s.get(id_field, ""),
                            "content": s.get("content", ""),
                            "section": s.get("section", ""),
                            "block_type": s.get("block_type", ""),
                            "line_number": s.get("line_number", 0),
                        }
                    )
            except Exception as e:
                logger.warning(f"grep_trigram P2 error ({index}): {e}")
            return blocks

        all_blocks = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            bio_fut2 = pool.submit(
                _fetch_blocks, "biomedrxiv_content", "document_id", bio_candidates
            )
            pmc_fut2 = pool.submit(
                _fetch_blocks, "pmc_content", "pmc_id", pmc_candidates
            )
            all_blocks.extend(bio_fut2.result())
            all_blocks.extend(pmc_fut2.result())

        t2 = _time.perf_counter()

        # Phase 3: regex verification (RE2 when available — 4-5x faster, linear time)
        try:
            compiled = _compile_regex(regex)
        except Exception:
            return []

        results = []
        for block in all_blocks:
            m = compiled.search(block["content"])
            if m:
                doc_id = block["paper_id"]
                line_num = block.get("line_number", 0)
                is_pmc = _re.match(r"^PMC\d+$", doc_id, _re.IGNORECASE)

                content = block["content"]
                start = max(0, m.start() - 100)
                end = min(len(content), m.end() + 100)
                snippet = content[start:end]

                results.append(
                    {
                        "document_id": doc_id if is_pmc else shorten(doc_id),
                        "line_number": line_num,
                        "content": snippet[:400],
                        "match": m.group(0)[:200],
                        "section": block.get("section", ""),
                        "block_type": block.get("block_type", ""),
                    }
                )
                if len(results) >= limit * 2:
                    break

        t3 = _time.perf_counter()
        p1_ms = (t1 - t0) * 1000
        p2_ms = (t2 - t1) * 1000
        p3_ms = (t3 - t2) * 1000
        logger.info(
            f"grep_trigram: P1={p1_ms:.0f}ms ({len(candidates)} candidates) "
            f"P2={p2_ms:.0f}ms ({len(all_blocks)} blocks) "
            f"P3={p3_ms:.0f}ms ({len(results)} matches)"
        )

        return results

    async def _try_slab_grep(
        self,
        regex: str,
        section_filter: str = None,
        limit: int = 50,
        source_filter: str = None,
    ) -> list[dict] | None:
        """Try the bitmap-grep slab service for corpus-wide search.

        Returns None if the service is unavailable (caller should fallback).
        Section filtering is applied post-hoc: run bitmap grep, resolve
        sections, then filter to matching sections.
        source_filter is handled natively by the slab service (no SQL roundtrip).
        """

        try:
            from slab_grep_client import slab_grep_search
        except ImportError:
            return None

        # Normalize source_filter for the slab service
        slab_source = None
        if source_filter and source_filter.lower() not in ("all", ""):
            slab_source = source_filter.lower()

        # When section filtering, request more results since many will be filtered out
        fetch_limit = limit * 10 if section_filter else limit
        result = await asyncio.to_thread(
            slab_grep_search,
            regex,
            limit=fetch_limit,
            case_insensitive=True,
            source_filter=slab_source,
        )
        if result is None:
            return None

        slab_matches = result.get("matches", [])
        if not slab_matches:
            return []

        elapsed_ms = result.get("elapsed_ms", 0)
        strategy = result.get("strategy", "bitmap")
        logger.info(
            f"[slab-grep] {regex!r}: {len(slab_matches)} hits in {elapsed_ms:.1f}ms ({strategy})"
        )

        import re as _re

        try:
            pat = _re.compile(regex, _re.IGNORECASE)
        except _re.error:
            pat = None

        out = []
        for m in slab_matches:
            doc_id = m["doc_id"]
            context = m.get("context", "")
            matching_line = context
            match_only = ""
            match_start = 0
            if context and pat:
                for line in context.split("\n"):
                    hit = pat.search(line)
                    if hit:
                        matching_line = line.strip()
                        match_only = hit.group(0)
                        match_start = hit.start()
                        break
                else:
                    # Bitmap index returned a candidate but the regex doesn't
                    # actually match any line — skip this false positive.
                    continue
            elif context:
                lines = context.split("\n")
                matching_line = max(lines, key=len).strip() if lines else context

            # Center the content snippet on the match so the matched text
            # is visible even in long single-line paragraphs (e.g. References).
            content_snippet = matching_line[:1000]
            if match_start > 500 and len(matching_line) > 1000:
                snippet_start = max(0, match_start - 400)
                content_snippet = "…" + matching_line[snippet_start : snippet_start + 999]

            out.append(
                {
                    "document_id": doc_id,
                    "paper_id": doc_id,
                    "source": m.get("source", "biorxiv"),
                    "content": content_snippet,
                    "match": match_only or matching_line[:200],
                    "match_context": context[:500] if context else "",
                    "line_number": m.get("line_number", 0),
                    "_match_snippet": matching_line[:80],
                }
            )

        # Resolve section names from content_blocks (needs raw UUIDs)
        await self._resolve_slab_grep_sections(out)

        # Convert raw UUIDs to short display IDs (bio_xxx / med_xxx)
        import re as _re2

        _pmc_pat = _re2.compile(r"^PMC\d+$", _re2.IGNORECASE)
        for m in out:
            raw_id = m["document_id"]
            if not _pmc_pat.match(raw_id):
                display_id = shorten(raw_id, m.get("source"))
                m["document_id"] = display_id
                m["paper_id"] = display_id

        # Apply section filter if requested (case-insensitive substring match)
        if section_filter:
            sf_lower = section_filter.lower()
            out = [m for m in out if sf_lower in (m.get("section") or "").lower()]

        # Deduplicate by content (different versions of the same paper)
        seen_content = set()
        deduped = []
        for m in out:
            key = m["content"][:200]
            if key not in seen_content:
                seen_content.add(key)
                deduped.append(m)

        return deduped[:limit]

    async def _resolve_slab_grep_sections(self, matches: list[dict]):
        """Look up section names for bitmap-grep matches via content_blocks."""
        if not matches:
            return

        def _resolve():
            bio_matches = [m for m in matches if m.get("source") != "pmc"]
            pmc_matches = [m for m in matches if m.get("source") == "pmc"]

            if bio_matches:
                try:
                    conn = _get_db_connection()
                    self._resolve_sections_batch(conn, bio_matches, "document_id::text")
                except Exception as e:
                    logger.debug(f"[slab-grep] section resolve (bio) failed: {e}")

            if pmc_matches:
                try:
                    module = _get_papers_module()
                    if module:
                        pmc_conn = module._get_pmc_db_connection()
                        self._resolve_sections_batch(pmc_conn, pmc_matches, "pmc_id")
                except Exception as e:
                    logger.debug(f"[slab-grep] section resolve (pmc) failed: {e}")

        await asyncio.to_thread(_resolve)

    def _resolve_sections_batch(self, conn, matches: list[dict], id_col: str):
        """For a batch of matches, find which section each belongs to.

        Uses a single query with UNION ALL to resolve all matches at once.
        """
        resolvable = [
            (i, m)
            for i, m in enumerate(matches)
            if m.get("_match_snippet", "") and len(m.get("_match_snippet", "")) >= 10
        ]
        if not resolvable:
            return

        # Build a single query: one subquery per match, UNION ALL'd together
        parts = []
        params = []
        for idx, (i, m) in enumerate(resolvable):
            doc_id = m["document_id"]
            snippet = m["_match_snippet"][:60]
            snippet = (
                snippet.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            )
            parts.append(
                f"""
                (SELECT {idx} AS idx, section FROM content_blocks
                 WHERE {id_col} = %s AND content LIKE %s LIMIT 1)
            """
            )
            params.extend([doc_id, f"%{snippet}%"])

        if not parts:
            return

        try:
            with conn.cursor() as cur:
                cur.execute(" UNION ALL ".join(parts), params)
                for row in cur.fetchall():
                    idx, section = row[0], row[1]
                    if section and idx < len(resolvable):
                        resolvable[idx][1]["section"] = section
        except Exception as e:
            logger.debug(f"[slab-grep] batch section resolve failed: {e}")

    async def _grep_tsv_pg(
        self,
        regex: str,
        section_filter: str = None,
        limit: int = 50,
        exhaustive: bool = False,
        source_filter: str = None,
    ) -> list[dict]:
        """Three-phase grep using PG tsvector + paper_fulltext.

        Phase 1: paper_search tsvector GIN → candidate paper IDs (Unix socket)
        Phase 2+3: Parallel batched fulltext fetch + regex verification.
                   Candidates are split into batches processed in parallel.
                   Early-exits when enough matches are found (unless exhaustive=True).

        Set exhaustive=True (--all flag) to process all P1 candidates without
        early exit, returning up to `limit` matched papers.

        source_filter: "biomedrxiv"/"biorxiv" = bio only, "pmc" = pmc only,
                       None/"all" = both sources.

        Falls back to _grep_trigram (ES) if paper_search table is empty or
        tsvector query fails.
        """
        import re as _re
        import time as _time
        from concurrent.futures import ThreadPoolExecutor

        t0 = _time.perf_counter()

        # Extract literal words from regex for tsquery
        literal_runs = _re.findall(r"[a-zA-Z0-9]{2,}", regex)
        if not literal_runs:
            return await self._grep_trigram(regex, section_filter, limit)

        tsquery_str = " & ".join(w.lower() for w in literal_runs)

        P1_CAP = 10000
        P2_BATCH = 250
        P2_WORKERS = 4
        MATCH_TARGET = P1_CAP if exhaustive else limit * 2

        # Split P1 into biorxiv (fast partial GIN) and PMC (slower seq scan)
        # so biorxiv candidates can start P2 immediately.
        def _p1_query_bio():
            conn = _get_db_connection()
            with conn.cursor() as cur:
                cur.execute("SET statement_timeout = '5000'")
                cur.execute(
                    """
                    SELECT paper_id
                    FROM paper_search
                    WHERE tsv @@ to_tsquery('simple', %s)
                      AND source = 'biorxiv'
                    LIMIT %s
                """,
                    (tsquery_str, P1_CAP),
                )
                rows = [r[0] for r in cur.fetchall()]
                cur.execute("RESET statement_timeout")
            return rows

        def _p1_query_pmc():
            _init_grep_pools()
            conn = _grep_pool_get(_grep_bio_pool)
            try:
                with conn.cursor() as cur:
                    cur.execute("SET statement_timeout = '20000'")
                    cur.execute(
                        """
                        SELECT paper_id
                        FROM paper_search
                        WHERE tsv @@ to_tsquery('simple', %s)
                          AND source = 'pmc'
                        LIMIT %s
                    """,
                        (tsquery_str, P1_CAP),
                    )
                    rows = [r[0] for r in cur.fetchall()]
                    cur.execute("RESET statement_timeout")
                return rows
            except Exception as e:
                logger.warning(f"grep_tsv_pg P1 pmc failed: {e}")
                return []
            finally:
                _grep_pool_put(_grep_bio_pool, conn)

        try:
            compiled = _compile_regex(regex)
        except Exception:
            return []

        # Normalize source_filter
        _sf = (source_filter or "").lower()
        _want_bio = _sf in ("", "all", "biomedrxiv", "biorxiv", "bio")
        _want_pmc = _sf in ("", "all", "pmc")

        # Launch P1 queries; skip sources the caller doesn't want
        from concurrent.futures import ThreadPoolExecutor as _P1Pool

        p1_pool = _P1Pool(max_workers=2)
        bio_fut = p1_pool.submit(_p1_query_bio) if _want_bio else None
        pmc_fut = p1_pool.submit(_p1_query_pmc) if _want_pmc else None

        # Wait for biorxiv P1 first (should be ~50-100ms with partial GIN)
        bio_candidates = []
        if bio_fut is not None:
            try:
                bio_candidates = bio_fut.result(timeout=10)
            except Exception as e:
                logger.warning(f"grep_tsv_pg bio P1 failed: {e}")
                bio_candidates = []

        t1 = _time.perf_counter()

        # ----- FAST PATH: PG-side regex filtering with LIMIT -----
        # Single query: feeds P1 candidates to PG, PG applies regex + LIMIT,
        # returns only matching docs' content. Avoids transferring non-matching
        # documents entirely and stops decompressing after enough matches found.
        # Falls through to batch path on error or unsupported patterns.
        _has_complex_regex = bool(_re.search(r"\\[1-9]|\(\?[<!=]", regex))
        _can_fast_path = (
            not section_filter
            and not _has_complex_regex
            and not exhaustive
            and bio_candidates
        )
        if _can_fast_path:
            try:
                _doc_limit = min(limit * 2, 200)
                conn = _get_db_connection()
                with conn.cursor() as cur:
                    cur.execute("SET statement_timeout = '30000'")
                    cur.execute(
                        """
                        SELECT document_id, content
                        FROM paper_fulltext
                        WHERE document_id = ANY(%s)
                          AND content ~* %s
                        LIMIT %s
                    """,
                        (bio_candidates, regex, _doc_limit),
                    )
                    rows = cur.fetchall()
                    cur.execute("RESET statement_timeout")

                fast_results = []
                fast_paper_ids = set()
                _MAX_LINES = 50
                for doc_id, content in rows:
                    if not content:
                        continue
                    is_pmc = _re.match(r"^PMC\d+$", doc_id, _re.IGNORECASE)
                    display_id = doc_id if is_pmc else shorten(doc_id)
                    fast_paper_ids.add(display_id)
                    paper_hits = 0
                    for lineno, line in enumerate(content.split("\n"), 1):
                        if compiled.search(line):
                            fast_results.append(
                                {
                                    "document_id": display_id,
                                    "line_number": lineno,
                                    "content": line,
                                    "match": compiled.search(line).group(0)[:200],
                                }
                            )
                            paper_hits += 1
                            if paper_hits >= _MAX_LINES:
                                break

                t2 = _time.perf_counter()
                p1_ms = (t1 - t0) * 1000
                p2p3_ms = (t2 - t1) * 1000
                src_label = f", source={source_filter}" if source_filter else ""
                logger.info(
                    f"grep_tsv_pg: FAST PATH P1={p1_ms:.0f}ms (bio={len(bio_candidates)}{src_label}) "
                    f"P2+P3={p2p3_ms:.0f}ms ({len(rows)} matching docs, "
                    f"{len(fast_results)} lines in {len(fast_paper_ids)} papers, "
                    f"limit={_doc_limit}, path=pg_regex_fast)"
                )
                if fast_results:
                    p1_pool.shutdown(wait=False)
                    return fast_results
            except Exception as e:
                logger.warning(f"grep fast path failed, falling back to batch: {e}")

        # ----- BATCH PATH (fallback): original batched P2+P3 -----
        _init_grep_pools()

        MAX_LINES_PER_PAPER = 50

        def _fetch_fulltext_bio(ids):
            conn = _grep_pool_get(_grep_bio_pool)
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT document_id, content FROM paper_fulltext WHERE document_id = ANY(%s)",
                        (ids,),
                    )
                    return cur.fetchall()
            except Exception as e:
                logger.warning(f"grep batch bio error: {e}")
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
                return []
            finally:
                _grep_pool_put(_grep_bio_pool, conn)

        def _fetch_fulltext_pmc(ids):
            conn = _grep_pool_get(_grep_pmc_pool)
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT pmc_id, content FROM paper_fulltext WHERE pmc_id = ANY(%s)",
                        (ids,),
                    )
                    return cur.fetchall()
            except Exception as e:
                logger.warning(f"grep batch pmc error: {e}")
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
                return []
            finally:
                _grep_pool_put(_grep_pmc_pool, conn)

        def _pg_regex_filter_bio(ids: list[str]):
            """Filter in PG: return only doc IDs + content for docs that match regex."""
            conn = _grep_pool_get(_grep_bio_pool)
            try:
                with conn.cursor() as cur:
                    cur.execute("SET statement_timeout = '30000'")
                    cur.execute(
                        """
                        SELECT document_id, content
                        FROM paper_fulltext
                        WHERE document_id = ANY(%s)
                          AND content ~* %s
                    """,
                        (ids, regex),
                    )
                    return cur.fetchall()
            except Exception as e:
                logger.warning(f"pg_regex_filter bio error: {e}")
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
                return []
            finally:
                _grep_pool_put(_grep_bio_pool, conn)

        def _pg_regex_filter_pmc(ids: list[str]):
            """Filter in PG: return only PMC docs that match regex."""
            conn = _grep_pool_get(_grep_pmc_pool)
            try:
                with conn.cursor() as cur:
                    cur.execute("SET statement_timeout = '30000'")
                    cur.execute(
                        """
                        SELECT pmc_id, content
                        FROM paper_fulltext
                        WHERE pmc_id = ANY(%s)
                          AND content ~* %s
                    """,
                        (ids, regex),
                    )
                    return cur.fetchall()
            except Exception as e:
                logger.warning(f"pg_regex_filter pmc error: {e}")
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
                return []
            finally:
                _grep_pool_put(_grep_pmc_pool, conn)

        def _process_batch_pg_regex(bio_ids: list[str], pmc_ids: list[str]):
            """Server-side filter: PG discards non-matching docs, only matches transferred."""
            from concurrent.futures import ThreadPoolExecutor as _TPE

            docs = []
            with _TPE(max_workers=2) as sub:
                futs = []
                if bio_ids:
                    futs.append(sub.submit(_pg_regex_filter_bio, bio_ids))
                if pmc_ids:
                    futs.append(sub.submit(_pg_regex_filter_pmc, pmc_ids))
                for f in futs:
                    docs.extend(f.result())

            batch_matches = []
            for doc_id, content in docs:
                if not content:
                    continue
                is_pmc = _re.match(r"^PMC\d+$", doc_id, _re.IGNORECASE)
                display_id = doc_id if is_pmc else shorten(doc_id)
                lines = content.split("\n")
                paper_hits = 0
                for lineno, line in enumerate(lines, 1):
                    if compiled.search(line):
                        batch_matches.append(
                            {
                                "document_id": display_id,
                                "line_number": lineno,
                                "content": line,
                                "match": compiled.search(line).group(0)[:200],
                            }
                        )
                        paper_hits += 1
                        if paper_hits >= MAX_LINES_PER_PAPER:
                            break
            return batch_matches, len(docs)

        def _process_batch_fulltext(bio_ids: list[str], pmc_ids: list[str]):
            """Fallback: fetch full paper text to Python, then regex verify."""
            from concurrent.futures import ThreadPoolExecutor as _TPE

            docs = []
            with _TPE(max_workers=2) as sub:
                futs = []
                if bio_ids:
                    futs.append(sub.submit(_fetch_fulltext_bio, bio_ids))
                if pmc_ids:
                    futs.append(sub.submit(_fetch_fulltext_pmc, pmc_ids))
                for f in futs:
                    docs.extend(f.result())

            batch_matches = []
            for doc_id, content in docs:
                if not content:
                    continue
                if not compiled.search(content):
                    continue
                is_pmc = _re.match(r"^PMC\d+$", doc_id, _re.IGNORECASE)
                display_id = doc_id if is_pmc else shorten(doc_id)
                lines = content.split("\n")
                paper_hits = 0
                for lineno, line in enumerate(lines, 1):
                    if compiled.search(line):
                        batch_matches.append(
                            {
                                "document_id": display_id,
                                "line_number": lineno,
                                "content": line,
                                "match": compiled.search(line).group(0)[:200],
                            }
                        )
                        paper_hits += 1
                        if paper_hits >= MAX_LINES_PER_PAPER:
                            break
            return batch_matches, len(docs)

        def _fetch_sections_bio(ids, sec_pattern):
            conn = _grep_pool_get(_grep_bio_pool)
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT document_id::text, line_number, content
                           FROM content_blocks
                           WHERE document_id::text = ANY(%s)
                             AND (section ILIKE %s OR block_type ILIKE %s)
                           ORDER BY document_id, line_number""",
                        (ids, sec_pattern, sec_pattern),
                    )
                    return cur.fetchall()
            except Exception as e:
                logger.warning(f"grep section batch bio error: {e}")
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
                return []
            finally:
                _grep_pool_put(_grep_bio_pool, conn)

        def _fetch_sections_pmc(ids, sec_pattern):
            conn = _grep_pool_get(_grep_pmc_pool)
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT pmc_id, line_number, content
                           FROM content_blocks
                           WHERE pmc_id = ANY(%s)
                             AND (section ILIKE %s OR block_type ILIKE %s)
                           ORDER BY pmc_id, line_number""",
                        (ids, sec_pattern, sec_pattern),
                    )
                    return cur.fetchall()
            except Exception as e:
                logger.warning(f"grep section batch pmc error: {e}")
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
                return []
            finally:
                _grep_pool_put(_grep_pmc_pool, conn)

        def _process_batch_sections(bio_ids: list[str], pmc_ids: list[str]):
            """Fetch section-filtered blocks from both DBs in parallel, then regex verify."""
            from concurrent.futures import ThreadPoolExecutor as _TPE

            sec_pattern = f"%{section_filter}%"
            rows = []
            with _TPE(max_workers=2) as sub:
                futs = []
                if bio_ids:
                    futs.append(
                        ("bio", sub.submit(_fetch_sections_bio, bio_ids, sec_pattern))
                    )
                if pmc_ids:
                    futs.append(
                        ("pmc", sub.submit(_fetch_sections_pmc, pmc_ids, sec_pattern))
                    )
                for src, f in futs:
                    rows.extend((src, *r) for r in f.result())

            batch_matches = []
            n_papers = 0
            paper_hits = {}
            seen_papers = set()
            for src, doc_id, lineno, content in rows:
                if doc_id not in seen_papers:
                    seen_papers.add(doc_id)
                    n_papers += 1
                if not content or not compiled.search(content):
                    continue
                display_id = doc_id if src == "pmc" else shorten(doc_id)
                hits = paper_hits.get(display_id, 0)
                if hits >= MAX_LINES_PER_PAPER:
                    continue
                paper_hits[display_id] = hits + 1
                batch_matches.append(
                    {
                        "document_id": display_id,
                        "line_number": lineno,
                        "content": content,
                        "match": compiled.search(content).group(0)[:200],
                    }
                )

            return batch_matches, n_papers

        # Choose batch P2+P3 strategy (only reached if fast path didn't return)
        _use_pg_regex = not section_filter and not _has_complex_regex

        if section_filter:
            _process_batch = _process_batch_sections
        elif _use_pg_regex:
            _process_batch = _process_batch_pg_regex
        else:
            _process_batch = _process_batch_fulltext

        # Stream P2+P3 with lazy batch submission: only P2_WORKERS batches
        # in flight at once so early-exit doesn't wait for wasted work.
        from concurrent.futures import FIRST_COMPLETED
        from concurrent.futures import wait as _wait

        results = []
        matched_paper_ids = set()
        docs_checked = 0
        pmc_candidates = []

        bio_batches = [
            bio_candidates[i : i + P2_BATCH]
            for i in range(0, len(bio_candidates), P2_BATCH)
        ]
        bio_batch_idx = 0
        pmc_batches = []
        pmc_batch_idx = 0

        pool = ThreadPoolExecutor(max_workers=P2_WORKERS)
        try:
            pending = set()

            # Seed with initial bio batches (up to P2_WORKERS)
            while bio_batch_idx < len(bio_batches) and len(pending) < P2_WORKERS:
                pending.add(pool.submit(_process_batch, bio_batches[bio_batch_idx], []))
                bio_batch_idx += 1

            # Submit PMC waiter if needed
            pmc_waiter = None
            if pmc_fut is not None:

                def _await_pmc():
                    try:
                        return pmc_fut.result(timeout=25)
                    except Exception as e:
                        logger.warning(f"grep_tsv_pg pmc P1 timed out or failed: {e}")
                        return []

                pmc_waiter = pool.submit(_await_pmc)
                pending.add(pmc_waiter)

            def _submit_next():
                """Submit one more batch if available and under worker limit."""
                nonlocal bio_batch_idx, pmc_batch_idx
                active = sum(1 for _ in pending if _ is not pmc_waiter)
                if active >= P2_WORKERS:
                    return
                if bio_batch_idx < len(bio_batches):
                    pending.add(
                        pool.submit(_process_batch, bio_batches[bio_batch_idx], [])
                    )
                    bio_batch_idx += 1
                elif pmc_batch_idx < len(pmc_batches):
                    pending.add(
                        pool.submit(_process_batch, [], pmc_batches[pmc_batch_idx])
                    )
                    pmc_batch_idx += 1

            while pending:
                done, pending = _wait(pending, return_when=FIRST_COMPLETED)

                for f in done:
                    if f is pmc_waiter:
                        pmc_candidates = f.result()
                        pmc_batches = [
                            pmc_candidates[i : i + P2_BATCH]
                            for i in range(0, len(pmc_candidates), P2_BATCH)
                        ]
                        pmc_batch_idx = 0
                        continue

                    try:
                        batch_matches, n_docs = f.result()
                    except Exception as e:
                        logger.warning(f"grep_tsv_pg batch error: {e}")
                        _submit_next()
                        continue

                    docs_checked += n_docs
                    for match in batch_matches:
                        did = match["document_id"]
                        if did not in matched_paper_ids:
                            matched_paper_ids.add(did)
                            results.append(match)

                if len(matched_paper_ids) >= MATCH_TARGET:
                    break

                _submit_next()

        finally:
            pool.shutdown(wait=False, cancel_futures=True)
            p1_pool.shutdown(wait=False)

        if not bio_candidates and not pmc_candidates and not results:
            return await self._grep_trigram(regex, section_filter, limit)

        t2 = _time.perf_counter()
        p1_ms = (t1 - t0) * 1000
        p2p3_ms = (t2 - t1) * 1000
        total_candidates = len(bio_candidates) + len(pmc_candidates)
        n_batches = (
            (len(bio_candidates) + P2_BATCH - 1) // P2_BATCH
            + (len(pmc_candidates) + P2_BATCH - 1) // P2_BATCH
            if total_candidates
            else 0
        )
        mode = "exhaustive" if exhaustive else f"early-exit@{MATCH_TARGET}"
        path_label = (
            f"section={section_filter}"
            if section_filter
            else ("pg_regex_batch" if _use_pg_regex else "fulltext_batch")
        )
        src_label = f", source={source_filter}" if source_filter else ""
        logger.info(
            f"grep_tsv_pg: P1={p1_ms:.0f}ms (bio={len(bio_candidates)}, pmc={len(pmc_candidates)}{src_label}) "
            f"P2+P3={p2p3_ms:.0f}ms ({docs_checked} docs checked, "
            f"{len(results)} matches in {len(matched_paper_ids)} papers, "
            f"{n_batches} batches, mode={mode}, path={path_label})"
        )

        return results

    async def batch_get_documents(self, document_ids: list[str]) -> dict[str, dict]:
        """Batch fetch multiple papers (biomedrxiv + PMC) - optimized single queries per DB."""
        if not document_ids:
            return {}

        import re as _re

        pmc_ids = [d for d in document_ids if _re.match(r"^PMC\d+$", d, _re.IGNORECASE)]
        bio_ids = [d for d in document_ids if d not in pmc_ids]

        # Resolve short IDs to UUIDs and build reverse map so results
        # are keyed by the original ID the caller used.
        bio_resolved = {resolve(d): d for d in bio_ids}
        bio_uuids = list(bio_resolved.keys())

        result: dict = {}

        # ── Biomedrxiv / medRxiv ─────────────────────────────────────────────
        if bio_uuids:

            def _fetch_bio():
                conn = _get_db_connection()
                r = {}
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT document_id::text, title, doi, source, authors, month_year
                           FROM documents WHERE document_id::text = ANY(%s)""",
                        (bio_uuids,),
                    )
                    for row in cur.fetchall():
                        uuid_id = row[0]
                        orig_id = bio_resolved.get(uuid_id, uuid_id)
                        r[orig_id] = {
                            "metadata": {
                                "document_id": shorten(uuid_id, row[3]),
                                "title": row[1],
                                "doi": row[2],
                                "source": row[3],
                                "authors": row[4],
                                "month_year": row[5],
                            },
                            "blocks": [],
                        }
                    cur.execute(
                        """SELECT document_id::text, line_number, content, section, block_type
                           FROM content_blocks WHERE document_id::text = ANY(%s)
                           ORDER BY document_id, line_number""",
                        (bio_uuids,),
                    )
                    for row in cur.fetchall():
                        orig_id = bio_resolved.get(row[0], row[0])
                        if orig_id in r:
                            r[orig_id]["blocks"].append(
                                {
                                    "line_number": row[1],
                                    "content": row[2],
                                    "section": row[3],
                                    "block_type": row[4],
                                }
                            )
                return r

            result.update(_execute_with_retry(_fetch_bio))

        # ── PMC ──────────────────────────────────────────────────────────────
        if pmc_ids:

            def _fetch_pmc():
                try:
                    module = _get_papers_module()
                    if module:
                        pmc_conn = module._get_pmc_db_connection()
                    else:
                        import re as re_

                        import psycopg2

                        pmc_url = re_.sub(
                            r"/biomedrxiv(\?|$)",
                            "/pmc\\1",
                            os.getenv("BIOMEDRXIV_DB_URL", ""),
                        )
                        pmc_conn = psycopg2.connect(pmc_url)
                        pmc_conn.autocommit = True
                except Exception as e:
                    logger.warning(f"[batch_get] PMC DB connection failed: {e}")
                    return {}

                r = {}
                upper_ids = [p.upper() for p in pmc_ids]
                with pmc_conn.cursor() as cur:
                    # Metadata — fast (documents table, small, has pmc_id index)
                    cur.execute(
                        """SELECT pmc_id, title, doi, source, authors, pub_year
                           FROM documents WHERE pmc_id = ANY(%s)""",
                        (upper_ids,),
                    )
                    for row in cur.fetchall():
                        doc_id = row[0]
                        r[doc_id] = {
                            "metadata": {
                                "document_id": doc_id,
                                "pmc_id": doc_id,
                                "title": row[1],
                                "doi": row[2],
                                "source": row[3] or "pmc",
                                "authors": row[4],
                                "month_year": str(row[5]) if row[5] else "",
                            },
                            "blocks": [],
                        }

                    # Content — include pub_year in WHERE so PostgreSQL prunes to one partition.
                    # Without pub_year: 16 index scans (~58ms/paper). With: 1 scan (~0.2ms).
                    module = _get_papers_module()
                    for uid in upper_ids:
                        if uid not in r:
                            continue
                        # Use cached pub_year lookup (populated from the metadata query above)
                        raw_year = r[uid]["metadata"].get("month_year")
                        try:
                            pub_year_int = int(str(raw_year)[:4]) if raw_year else None
                        except (ValueError, TypeError):
                            pub_year_int = None
                        # Also populate the module's cache so _read_pmc_lines is instant
                        if module and pub_year_int:
                            module._pmc_pub_year_cache[uid] = pub_year_int

                        if pub_year_int:
                            cur.execute(
                                """SELECT pmc_id, line_number, content, section, block_type
                                   FROM content_blocks
                                   WHERE pmc_id = %s AND pub_year = %s
                                   ORDER BY line_number""",
                                (uid, pub_year_int),
                            )
                        else:
                            cur.execute(
                                """SELECT pmc_id, line_number, content, section, block_type
                                   FROM content_blocks WHERE pmc_id = %s
                                   ORDER BY line_number""",
                                (uid,),
                            )
                        for row in cur.fetchall():
                            r[row[0]]["blocks"].append(
                                {
                                    "line_number": row[1],
                                    "content": row[2],
                                    "section": row[3],
                                    "block_type": row[4],
                                }
                            )
                return r

            try:
                import asyncio

                # Run with a timeout — PMC blocks fetch can be slow on cold partitions
                pmc_result = await asyncio.wait_for(
                    asyncio.to_thread(_fetch_pmc), timeout=15.0
                )
                result.update(pmc_result)
            except asyncio.TimeoutError:
                logger.warning(f"[batch_get] PMC fetch timed out for {pmc_ids}")
                # Return metadata-only stubs so map can still show the papers
                for pid in pmc_ids:
                    if pid not in result:
                        result[pid] = {
                            "metadata": {
                                "document_id": pid,
                                "pmc_id": pid,
                                "title": pid,
                                "source": "pmc",
                            },
                            "blocks": [],
                        }
            except Exception as e:
                logger.warning(f"[batch_get] PMC fetch failed: {e}")

        return result

    async def batch_get_metadata(self, document_ids: list[str]) -> dict[str, dict]:
        """Batch fetch metadata only (no content blocks) - single query with retry.

        Queries both the biomedrxiv DB (for bio_/med_ IDs) and the PMC DB
        (for PMC* IDs), mirroring the split logic in batch_get_documents.
        """
        if not document_ids:
            return {}

        import re as _re

        pmc_ids = [d for d in document_ids if _re.match(r"^PMC\d+$", d, _re.IGNORECASE)]
        bio_ids = [d for d in document_ids if d not in pmc_ids]

        result: dict[str, dict] = {}

        # ── Biomedrxiv / medRxiv ─────────────────────────────────────────────
        if bio_ids:

            def _fetch_bio_metadata():
                conn = _get_db_connection()
                r = {}
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT document_id::text, title, doi, source, authors, month_year
                        FROM documents
                        WHERE document_id::text = ANY(%s)
                    """,
                        (bio_ids,),
                    )
                    for row in cur.fetchall():
                        doc_id = row[0]
                        r[doc_id] = {
                            "metadata": {
                                "document_id": doc_id,
                                "title": row[1],
                                "doi": row[2],
                                "source": row[3],
                                "authors": row[4],
                                "month_year": row[5],
                            },
                        }
                return r

            result.update(_execute_with_retry(_fetch_bio_metadata))

        # ── PMC ──────────────────────────────────────────────────────────────
        if pmc_ids:

            def _fetch_pmc_metadata():
                try:
                    module = _get_papers_module()
                    if module:
                        pmc_conn = module._get_pmc_db_connection()
                    else:
                        import re as re_

                        import psycopg2

                        pmc_url = re_.sub(
                            r"/biomedrxiv(\?|$)",
                            "/pmc\\1",
                            os.getenv("BIOMEDRXIV_DB_URL", ""),
                        )
                        pmc_conn = psycopg2.connect(pmc_url)
                        pmc_conn.autocommit = True
                except Exception as e:
                    logger.warning(
                        f"[batch_get_metadata] PMC DB connection failed: {e}"
                    )
                    return {}

                r = {}
                upper_ids = [p.upper() for p in pmc_ids]
                with pmc_conn.cursor() as cur:
                    cur.execute(
                        """SELECT pmc_id, title, doi, source, authors, pub_year
                           FROM documents WHERE pmc_id = ANY(%s)""",
                        (upper_ids,),
                    )
                    for row in cur.fetchall():
                        doc_id = row[0]
                        r[doc_id] = {
                            "metadata": {
                                "document_id": doc_id,
                                "pmc_id": doc_id,
                                "title": row[1],
                                "doi": row[2],
                                "source": row[3] or "pmc",
                                "authors": row[4],
                                "month_year": str(row[5]) if row[5] else "",
                            },
                        }
                return r

            try:
                import asyncio

                pmc_result = await asyncio.wait_for(
                    asyncio.to_thread(_fetch_pmc_metadata), timeout=15.0
                )
                result.update(pmc_result)
            except asyncio.TimeoutError:
                logger.warning(
                    f"[batch_get_metadata] PMC fetch timed out for {pmc_ids}"
                )
            except Exception as e:
                logger.warning(f"[batch_get_metadata] PMC fetch failed: {e}")

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
        self._pmc_pub_year_cache: dict[str, int] = {}  # pmc_id → pub_year cache
        self._setup_tools()
        # Register as global instance so the store can access PMC connection
        global _papers_module_instance
        _papers_module_instance = self
        # Pre-warm DB connection in background so first user query isn't cold
        import threading

        threading.Thread(target=self._warmup_connections, daemon=True).start()

    def _warmup_connections(self):
        """Fire a trivial query to warm up the CloudSQL proxy connection.
        Runs in background on startup so the first user command isn't slow (~3s cold start).
        """
        try:
            conn = _get_db_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            logger.info("[warmup] biomedrxiv DB connection warmed up")
        except Exception as e:
            logger.warning(f"[warmup] DB warmup failed: {e}")

    def _get_pmc_pub_year(self, pmc_id: str) -> int | None:
        """Look up pub_year for a PMC paper, using an in-memory cache.

        First call: hits documents table (O(1) with pmc_id unique index).
        All subsequent calls: O(1) from the in-process dict.
        The cache is never invalidated (pub_year doesn't change for a paper).
        """
        uid = pmc_id.upper()
        if uid in self._pmc_pub_year_cache:
            return self._pmc_pub_year_cache[uid]
        try:
            conn = self._get_pmc_db_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT pub_year FROM documents WHERE pmc_id = %s", (uid,))
                row = cur.fetchone()
            if row:
                self._pmc_pub_year_cache[uid] = row[0]
                return row[0]
        except Exception as e:
            logger.warning(f"[pub_year_cache] Failed for {pmc_id}: {e}")
        return None

    @staticmethod
    def _get_month_for_doc(uuid, conn):
        """Get month_year for a document from the DB."""
        with conn.cursor() as cur:
            cur.execute(
                "SELECT month_year FROM documents WHERE document_id = %s", (uuid,)
            )
            row = cur.fetchone()
            return row[0] if row else None

    @staticmethod
    def _is_biorxiv(uuid, conn):
        """Check if document is from biorxiv (vs medrxiv)."""
        with conn.cursor() as cur:
            cur.execute("SELECT source FROM documents WHERE document_id = %s", (uuid,))
            row = cur.fetchone()
            return row[0] != "medrxiv" if row else True

    # Whitelist for any filename that gets concatenated into a GCS object
    # path. Covers every PMC/biorxiv figure + supplement filename we've seen
    # in the wild (alphanumerics, dot, underscore, hyphen).
    _SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,200}$")

    @classmethod
    def _is_safe_filename(cls, name: str) -> bool:
        """Return True iff *name* is safe to interpolate into a GCS path.

        Rejects empty strings, path separators, `..`, and anything with
        characters outside the whitelist. GCS doesn't resolve `..` segments,
        so this isn't an active vulnerability today — but any filename we
        concatenate into a bucket path or log line should be validated
        before refactors tighten the blast radius elsewhere.
        """
        return bool(name) and bool(cls._SAFE_FILENAME_RE.fullmatch(name))

    @classmethod
    def _resolve_figure_download(cls, document_id: str, figure_id: str, conn) -> dict:
        """Resolve a figure to a short-lived signed GCS download URL.

        PMC figures live directly under gs://gxl-collections/pmc/articles/{id}/.
        bioRxiv/medRxiv figures are recorded in content_blocks and mirror the
        resolution logic in modules.papers.filesystem.PapersModule
        ._resolve_figure_gcs_path so the MCP server's _cat handler can return
        a `download_url` for `paperclip cat /papers/<id>/figures/<file>`.

        Returns {"download_url", "filename", "caption"} on success or
        {"error": ...} when the figure can't be resolved.
        """
        import os

        from modules.papers.tools import _get_gcs_client, generate_signed_download_url

        if not cls._is_safe_filename(figure_id):
            return {"error": f"Invalid figure filename: {figure_id!r}"}

        is_pmc = document_id.upper().startswith("PMC") and document_id[3:].isdigit()
        if is_pmc:
            gcs_path = f"gs://gxl-collections/pmc/articles/{document_id}/{figure_id}"
            client = _get_gcs_client()
            if not client:
                return {"error": "GCS client unavailable"}
            try:
                blob = client.bucket("gxl-collections").blob(
                    f"pmc/articles/{document_id}/{figure_id}"
                )
                blob_found = blob.exists()
            except Exception:
                # M2: don't echo GCS internals (request IDs, SA names,
                # bucket URLs) to the client. Log server-side, return a
                # generic error.
                logger.exception(
                    "GCS lookup failed for %s/%s", document_id, figure_id
                )
                return {"error": "GCS lookup failed"}
            if not blob_found:
                return {"error": f"Image not found in GCS: {figure_id}"}
            url = generate_signed_download_url(gcs_path)
            if not url:
                return {"error": "Failed to generate signed URL"}
            return {
                "download_url": url,
                "filename": figure_id,
                "caption": "",
            }

        figure_id_base = (
            figure_id.replace(".tif", "").replace(".tiff", "")
            .replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
        )
        with conn.cursor() as cur:
            cur.execute(
                """SELECT content, citation_info->>'source_path', citation_info->>'xml_id',
                          citation_info->>'graphic', citation_info->>'image_uri'
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

        caption, source_path, xml_id, graphic = row[0], row[1], row[2], row[3]
        image_uri = row[4] if len(row) > 4 else None

        client = _get_gcs_client()
        if not client:
            return {"error": "GCS client unavailable"}

        # Arxiv: use image_uri directly (points to the actual image in GCS)
        if image_uri and image_uri.startswith("gs://"):
            path_no_prefix = image_uri[5:]
            parts = path_no_prefix.split("/", 1)
            if len(parts) == 2:
                bucket_name, blob_path = parts
                try:
                    blob = client.bucket(bucket_name).blob(blob_path)
                    if blob.exists():
                        url = generate_signed_download_url(image_uri)
                        if url:
                            return {
                                "download_url": url,
                                "filename": graphic or figure_id,
                                "caption": caption or "",
                            }
                except Exception:
                    pass

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
            for ext in (".tif", ".tiff", ".jpg", ".jpeg", ".png"):
                candidates.append(f"{base_path}/{xml_id}{ext}")

        for gcs_path in candidates:
            path_no_prefix = gcs_path[5:]
            parts = path_no_prefix.split("/", 1)
            if len(parts) != 2:
                continue
            bucket_name, blob_path = parts
            try:
                blob = client.bucket(bucket_name).blob(blob_path)
                if not blob.exists():
                    continue
            except Exception:
                continue
            url = generate_signed_download_url(gcs_path)
            if url:
                return {
                    "download_url": url,
                    "filename": graphic or f"{xml_id}{os.path.splitext(gcs_path)[1]}",
                    "caption": caption or "",
                }

        return {"error": f"Image file not found in GCS for figure {figure_id}"}

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
        """Override to show papers-specific structure."""
        return [
            "meta.json",
            "content.lines",
            "sections/",
            "supplements/",
            "figures/",
        ]

    def _setup_tools(self):
        """Define the single paperclip tool.

        All commands mirror the paperclip CLI: search, lookup, bash, map,
        reduce, ask-image, searches, funded-by, scan, grep, cat, etc.
        Everything routes through the virtual terminal.
        """
        from collections.abc import Callable

        tools_def = [
            {
                "name": "paperclip",
                "description": """Paperclip — virtual filesystem of full-text biomedical papers (PubMed Central, bioRxiv, medRxiv, arXiv), FDA regulatory documents, clinical trials, and international regulatory filings.

Pass any command as a string. **Run `skill` first to load full documentation.** Run `<command> --help` for help on any command.""",
                "handler": self._paperclip,
                "schema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Full command string (e.g. \"search 'CRISPR delivery' -n 20\")",
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

    _MAX_RESPONSE_CHARS = 500_000

    async def _save_large_response(
        self,
        text: str,
        func_name: str,
        session_id: str,
        agent_id: str | None = None,
        paper_uuid: str | None = None,
    ) -> str:
        """Save large response to /.gxl/ and return truncated text
        with instructions the model can actually follow in the virtual shell."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{func_name}_output_{timestamp}.txt"
        filepath = f"/.gxl/{filename}"

        try:
            terminal = self._get_terminal(session_id, agent_id, paper_uuid)
            write_result = await terminal._session_files_write(
                filepath, text, session_id
            )
            if write_result.exit_code != 0:
                raise RuntimeError(write_result.stderr)
        except Exception as e:
            logger.warning(f"Failed to save large response to {filepath}: {e}")
            preview = text[: self._MAX_RESPONSE_CHARS]
            return preview + "\n\n[... output truncated, too large to display ...]"

        preview_limit = 5000
        preview = text[:preview_limit]
        if len(text) > preview_limit:
            preview += "\n[... truncated ...]"

        return (
            f"{preview}\n\n"
            f"[Output too large ({len(text)} chars). Full output saved to /.gxl/{filename}]\n"
            f"[To view: cat /.gxl/{filename}]"
        )

    def _create_handler(self, func):
        """Wrap handler with error handling and TextContent formatting.

        Uses compact text formatters when available (see compact_fmt.py)
        to minimize token usage in tool outputs.  Falls back to JSON.
        Toggle: touch /tmp/biomedrxiv_json_mode to force JSON output.
        """
        compact_formatter = COMPACT_FORMATTERS.get(func.__name__)

        import inspect as _inspect

        _func_params = set(_inspect.signature(func).parameters)

        async def handler(
            arguments: dict,
            session_id: str = "default",
            agent_id: str | None = None,
            paper_uuid: str | None = None,
            **kwargs,
        ):
            try:
                arguments.pop("description", None)
                extra: dict = {}
                if agent_id is not None and "agent_id" in _func_params:
                    extra["agent_id"] = agent_id
                if paper_uuid is not None and "paper_uuid" in _func_params:
                    extra["paper_uuid"] = paper_uuid
                result = await func(session_id=session_id, **arguments, **extra)

                # Use compact formatter if available, fall back to JSON
                use_compact = not os.path.exists("/tmp/biomedrxiv_json_mode")
                if use_compact and compact_formatter:
                    text = compact_formatter(result)
                else:
                    text = json.dumps(result, indent=2, default=str)

                # Save to /.gxl/ if too large, so the model can
                # retrieve it with `cat /.gxl/...` instead of
                # getting a misleading "execute_in_sandbox" message from
                # the generic ResponseManager.
                if len(text) > self._MAX_RESPONSE_CHARS:
                    text = await self._save_large_response(
                        text, func.__name__, session_id, agent_id, paper_uuid
                    )

                tc = TextContent(type="text", text=text)
                # Stash raw dict so disk persistence (base_server) can
                # save structured JSON even when compact format is active.
                object.__setattr__(tc, "_raw_result", result)
                return [tc]
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                return [
                    TextContent(
                        type="text",
                        text=f"ERROR: {func.__name__}: {e}",
                    )
                ]

        return handler

    def get_tools(self) -> list[Tool]:
        return self._tools

    # =========================================================================
    # Tool Implementations
    # =========================================================================

    _pmc_conn_cache: "psycopg2.extensions.connection | None" = None
    _arxiv_conn_cache: "psycopg2.extensions.connection | None" = None

    def _get_arxiv_db_connection(self):
        """Get a cached connection to the arxiv database."""
        import psycopg2

        if self._arxiv_conn_cache is not None:
            try:
                self._arxiv_conn_cache.cursor().execute("SELECT 1")
                return self._arxiv_conn_cache
            except Exception:
                self._arxiv_conn_cache = None

        url = os.environ.get("BIOMEDRXIV_DB_URL", "")
        if url:
            import re

            arxiv_url = re.sub(r"/biomedrxiv(\?|$)", "/arxiv\\1", url)
            conn = psycopg2.connect(
                arxiv_url,
                keepalives=1,
                keepalives_idle=300,
                keepalives_interval=30,
                keepalives_count=5,
            )
        else:
            host = os.environ.get("BIOMEDRXIV_DB_HOST", "")
            password = os.environ.get("BIOMEDRXIV_DB_PASSWORD", "")
            conn = psycopg2.connect(
                host=host,
                port=5432,
                database="arxiv",
                user=os.environ.get("BIOMEDRXIV_DB_USER", "postgres"),
                password=password,
                keepalives=1,
                keepalives_idle=300,
            )
        conn.autocommit = True
        self._arxiv_conn_cache = conn
        return conn

    def _get_pmc_db_connection(self):
        """Get a cached connection to the PMC database."""
        import psycopg2

        # Return cached connection if alive
        if self._pmc_conn_cache is not None:
            try:
                self._pmc_conn_cache.cursor().execute("SELECT 1")
                return self._pmc_conn_cache
            except Exception:
                self._pmc_conn_cache = None

        url = os.environ.get("BIOMEDRXIV_DB_URL", "")
        if url:
            import re

            pmc_url = re.sub(r"/biomedrxiv(\?|$)", "/pmc\\1", url)
            conn = psycopg2.connect(
                pmc_url,
                keepalives=1,
                keepalives_idle=300,
                keepalives_interval=30,
                keepalives_count=5,
            )
        else:
            host = os.environ.get(
                "PMC_DB_HOST", os.environ.get("BIOMEDRXIV_DB_HOST", "")
            )
            password = os.environ.get(
                "PMC_DB_PASSWORD", os.environ.get("BIOMEDRXIV_DB_PASSWORD", "")
            )
            conn = psycopg2.connect(
                host=host,
                port=5432,
                database="pmc",
                user=os.environ.get("BIOMEDRXIV_DB_USER", "postgres"),
                password=password,
                keepalives=1,
                keepalives_idle=300,
            )
        conn.autocommit = True
        self._pmc_conn_cache = conn
        return conn

    async def _ls_pmc_document(self, pmc_id: str, start_time: float) -> dict:
        """List a PMC paper directory."""
        import asyncio

        def _query():
            conn = self._get_pmc_db_connection()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT title, doi, pmid, journal_title, pub_year FROM documents WHERE pmc_id = %s",
                    (pmc_id,),
                )
                doc = cur.fetchone()
                if not doc:
                    return None, None
                cur.execute(
                    "SELECT COUNT(*), COUNT(DISTINCT section) FROM content_blocks WHERE pmc_id = %s",
                    (pmc_id,),
                )
                stats = cur.fetchone()
            return doc, stats

        doc, stats = await asyncio.to_thread(_query)
        if not doc:
            return {"error": f"PMC paper not found: {pmc_id}"}
        total_lines, section_count = stats
        contents = ["meta.json", f"content.lines  ({total_lines} lines)", "sections/"]

        try:
            has_supplements, has_figures, has_reviews = await asyncio.to_thread(
                self._pmc_gcs_checks, pmc_id
            )
            if has_supplements:
                contents.append("supplements/")
            if has_figures:
                contents.append("figures/")
            if has_reviews:
                contents.append("reviews/")
        except Exception:
            pass

        return {
            "path": f"/papers/{pmc_id}/",
            "title": doc[0],
            "doi": doc[1],
            "pmid": doc[2],
            "journal": doc[3],
            "pub_year": doc[4],
            "source": "pmc",
            "contents": contents,
            "time_ms": round((time.perf_counter() - start_time) * 1000),
        }

    # ── OpenAlex (abstract-only) VFS support ────────────────────────────
    # OpenAlex papers are served entirely from OpenSearch (`abstract_only`
    # index) — there is no full text, only a title + abstract + author /
    # journal metadata. The VFS exposes just two entries for each paper:
    #   meta.json             (the OS `_source`, minus the raw abstract)
    #   sections/ABSTRACT.md  (the abstract, one paragraph, markdown-ish)
    # This keeps the paperclip paper-filesystem model uniform across corpora.

    @staticmethod
    def _oa_os_fields() -> list[str]:
        return [
            "document_id",
            "oa_id",
            "source",
            "title",
            "abstract",
            "doi",
            "authors",
            "pub_year",
            "pub_date",
            "journal_title",
            "categories",
        ]

    def _fetch_oa_source(self, oa_id: str) -> dict | None:
        """Pull an OpenAlex paper's ``_source`` from OpenSearch.

        Tries a direct ``_mget`` first (when ``document_id == _id``), then
        falls back to a ``terms`` query on ``document_id`` which always
        works regardless of how the point was indexed.
        """
        os_client = get_opensearch_client()
        if not os_client:
            return None
        try:
            docs = os_client.mget(
                "abstract_only", [oa_id], source_fields=self._oa_os_fields()
            )
            for d in docs:
                if d.get("found"):
                    return d.get("_source") or {}
        except Exception as e:
            logger.debug(f"OpenAlex _mget failed for {oa_id}: {e}")
        try:
            resp = os_client.search(
                "abstract_only",
                {
                    "query": {"term": {"document_id": oa_id}},
                    "size": 1,
                    "_source": self._oa_os_fields(),
                },
            )
            hits = resp.get("hits", {}).get("hits", [])
            if hits:
                return hits[0].get("_source") or {}
        except Exception as e:
            logger.warning(f"OpenAlex lookup failed for {oa_id}: {e}")
        return None

    async def _ls_oa_document(self, oa_id: str, start_time: float) -> dict:
        """List an OpenAlex paper directory (meta.json + sections/)."""
        import asyncio

        src = await asyncio.to_thread(self._fetch_oa_source, oa_id)
        if not src:
            return {"error": f"OpenAlex paper not found: {oa_id}"}
        return {
            "path": f"/papers/{oa_id}/",
            "title": src.get("title") or "",
            "doi": src.get("doi") or "",
            "authors": src.get("authors") or "",
            "pub_year": src.get("pub_year"),
            "source": src.get("source") or "openalex",
            "journal": src.get("journal_title") or "",
            "contents": ["meta.json", "sections/"],
            "hint": (
                "OpenAlex papers are abstract-only; read "
                f"/papers/{oa_id}/sections/ABSTRACT.md for the abstract."
            ),
            "time_ms": round((time.perf_counter() - start_time) * 1000),
        }

    async def _ls_oa_sections(self, oa_id: str, start_time: float) -> dict:
        """List section files for an OpenAlex paper (ABSTRACT.md only)."""
        import asyncio

        src = await asyncio.to_thread(self._fetch_oa_source, oa_id)
        if not src:
            return {"error": f"OpenAlex paper not found: {oa_id}"}
        return {
            "path": f"/papers/{oa_id}/sections/",
            "contents": ["ABSTRACT.md"],
            "time_ms": round((time.perf_counter() - start_time) * 1000),
        }

    async def _cat_oa(
        self, oa_id: str, parsed: ParsedPath, path: str, start_time: float
    ) -> dict:
        """Serve meta.json or sections/ABSTRACT.md for an OpenAlex paper."""
        import asyncio

        src = await asyncio.to_thread(self._fetch_oa_source, oa_id)
        if not src:
            return {"error": f"OpenAlex paper not found: {oa_id}"}

        if parsed.type == "file" and parsed.filename == "meta.json":
            return {
                "path": path,
                "type": "json",
                "content": {
                    "document_id": oa_id,
                    "title": src.get("title") or "",
                    "doi": src.get("doi") or "",
                    "source": src.get("source") or "openalex",
                    "authors": src.get("authors") or "",
                    "pub_year": src.get("pub_year"),
                    "pub_date": src.get("pub_date") or "",
                    "journal": src.get("journal_title") or "",
                    "abstract": src.get("abstract") or "",
                    "categories": src.get("categories") or "",
                },
                "time_ms": round((time.perf_counter() - start_time) * 1000),
            }

        if parsed.type == "section" and (parsed.section or "").lower() == "abstract":
            abstract = (src.get("abstract") or "").strip()
            return {
                "path": path,
                "type": "text",
                "content": abstract or "(no abstract)",
                "time_ms": round((time.perf_counter() - start_time) * 1000),
            }

        return {
            "error": (
                f"OpenAlex papers expose only meta.json and "
                f"sections/ABSTRACT.md (requested: {path})"
            )
        }

    def _pmc_gcs_checks(self, pmc_id: str) -> tuple:
        """Check GCS for supplement files, figure files, and peer review sub-articles.
        Returns (has_supplements, has_figures, has_reviews)."""
        try:
            gcs_bucket = _get_gcs_bucket()
            prefix = f"pmc/articles/{pmc_id}/"
            has_supplements = False
            has_figures = False
            has_reviews = False
            nxml_blob_name = None
            main_exts = {".nxml", ".xml"}
            figure_exts = {".gif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".svg"}
            for blob in gcs_bucket.list_blobs(prefix=prefix, max_results=200):
                rel = blob.name[len(prefix) :]
                if not rel or rel.startswith("_processed/"):
                    continue
                ext = rel[rel.rfind(".") :].lower() if "." in rel else ""
                if ext in main_exts:
                    nxml_blob_name = blob.name
                    continue
                if ext == ".pdf":
                    lower_rel = rel.lower()
                    if (
                        "suppl" in lower_rel
                        or ".s0" in lower_rel
                        or ".s1" in lower_rel
                        or lower_rel.startswith("mmc")
                    ):
                        has_supplements = True
                    continue
                if ext in {
                    ".docx",
                    ".doc",
                    ".xlsx",
                    ".xls",
                    ".pptx",
                    ".csv",
                    ".zip",
                    ".tar",
                    ".gz",
                }:
                    has_supplements = True
                elif ext in figure_exts:
                    has_figures = True
                if has_supplements and has_figures:
                    break
            if not has_supplements:
                proc_prefix = f"pmc/articles/{pmc_id}/_processed/v1/"
                for blob in gcs_bucket.list_blobs(prefix=proc_prefix, max_results=20):
                    rel = blob.name[len(proc_prefix) :]
                    if rel and rel.endswith(".md") and rel != "_manifest.json":
                        has_supplements = True
                        break
            # Check for peer review sub-articles via quick byte scan of NXML
            if nxml_blob_name:
                try:
                    blob = gcs_bucket.blob(nxml_blob_name)
                    # Only download first 5KB to check for sub-article tag
                    header = blob.download_as_bytes(
                        start=0, end=min(blob.size or 500000, 500000)
                    )
                    if b"<sub-article" in header:
                        has_reviews = True
                except Exception:
                    pass
            return has_supplements, has_figures, has_reviews
        except Exception as e:
            logger.warning(f"[PMC GCS check] Failed for {pmc_id}: {e}")
            return False, False, False

    async def _ls_pmc_supplements(self, pmc_id: str, start_time: float) -> dict:
        """List supplement files for a PMC paper from GCS."""
        import asyncio
        import json as _json

        def _list():
            gcs_bucket = _get_gcs_bucket()
            prefix = f"pmc/articles/{pmc_id}/"
            proc_prefix = f"pmc/articles/{pmc_id}/_processed/v1/"

            # Load manifest to find processed supplements
            processed_map = {}  # source_filename -> output_md
            try:
                manifest_blob = gcs_bucket.blob(f"{proc_prefix}_manifest.json")
                if manifest_blob.exists():
                    manifest = _json.loads(manifest_blob.download_as_text())
                    for f in manifest.get("files", []):
                        if f.get("status") == "ok" and f.get("output"):
                            processed_map[f["source"]] = f["output"]
            except Exception:
                pass

            contents = []
            seen = set()
            main_exts = {".nxml", ".xml"}
            figure_exts = {".gif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".svg"}
            skip_figure_exts = {".gif"}  # GIFs are low-res thumbnails

            for blob in gcs_bucket.list_blobs(prefix=prefix, max_results=200):
                rel = blob.name[len(prefix) :]
                if not rel or rel.startswith("_processed/"):
                    continue
                ext = rel[rel.rfind(".") :].lower() if "." in rel else ""
                if ext in main_exts:
                    continue
                lower_rel = rel.lower()
                _is_suppl = (
                    "suppl" in lower_rel
                    or ".s0" in lower_rel
                    or ".s1" in lower_rel
                    or lower_rel.startswith("mmc")
                )
                # Skip figure images (gXXX pattern, fXXX pattern)
                is_figure = ext in figure_exts and not _is_suppl
                if is_figure:
                    continue
                # Main paper PDF (skip unless it's a supplement)
                if ext == ".pdf" and not _is_suppl:
                    continue

                if rel in seen:
                    continue
                seen.add(rel)
                contents.append({"name": rel, "size": blob.size})

                # If this file has a processed .md version, list it too
                if rel in processed_map:
                    md_name = f"{rel}.md.lines"
                    # Count lines in the processed md
                    try:
                        md_blob = gcs_bucket.blob(f"{proc_prefix}{processed_map[rel]}")
                        md_text = md_blob.download_as_text()
                        line_count = md_text.count("\n") + 1
                        contents.append({"name": md_name, "lines": line_count})
                    except Exception:
                        contents.append({"name": md_name})

            return contents

        try:
            contents = await asyncio.to_thread(_list)
        except Exception as e:
            return {
                "path": f"/papers/{pmc_id}/supplements/",
                "count": 0,
                "contents": [],
                "error": str(e),
                "time_ms": round((time.perf_counter() - start_time) * 1000),
            }

        return {
            "path": f"/papers/{pmc_id}/supplements/",
            "count": len(contents),
            "contents": contents,
            "time_ms": round((time.perf_counter() - start_time) * 1000),
        }

    def _parse_pmc_reviews(self, pmc_id: str) -> list:
        """Parse peer review sub-articles from a PMC paper's NXML in GCS.
        Returns list of dicts with review metadata and content."""
        from lxml import etree as _etree

        gcs_bucket = _get_gcs_bucket()
        prefix = f"pmc/articles/{pmc_id}/"

        nxml_blob = None
        for blob in gcs_bucket.list_blobs(prefix=prefix, max_results=200):
            if blob.name.endswith(".nxml"):
                nxml_blob = blob
                break
        if nxml_blob is None:
            return []

        xml_bytes = nxml_blob.download_as_bytes()
        parser = _etree.XMLParser(recover=True)
        root = _etree.fromstring(xml_bytes, parser)
        if root is None:
            return []

        def _extract_text(elem):
            if elem is None:
                return ""
            return " ".join(elem.itertext()).strip()

        reviews = []
        for sub in root.findall("sub-article"):
            art_type = sub.get("article-type", "unknown")
            sub_id = sub.get("id", "")

            # DOI
            doi = None
            for aid in sub.findall(".//article-id"):
                if aid.get("pub-id-type") == "doi" and aid.text:
                    doi = aid.text.strip()

            title_el = sub.find(".//article-title")
            title = _extract_text(title_el) if title_el is not None else art_type

            # Contributors
            contribs = []
            for c in sub.findall(".//contrib"):
                name_el = c.find(".//name")
                if name_el is not None:
                    given = name_el.findtext("given-names", "") or ""
                    surname = name_el.findtext("surname", "") or ""
                    roles = [r.text for r in c.findall("role") if r.text]
                    name_str = f"{given} {surname}".strip()
                    if roles:
                        name_str += f" ({', '.join(roles)})"
                    contribs.append(name_str)

            # Body text
            body = sub.find(".//body")
            body_lines = []
            if body is not None:
                for elem in body.iter():
                    if elem.tag == "p":
                        t = _extract_text(elem)
                        if t:
                            body_lines.append(t)
                    elif elem.tag in ("title", "label"):
                        t = _extract_text(elem)
                        if t and not any(
                            t in bl for bl in body_lines[-2:] if body_lines
                        ):
                            body_lines.append(f"## {t}")

            # Build a clean filename from the title
            safe_title = (title or art_type).lower()
            safe_title = re.sub(r"[^a-z0-9]+", "_", safe_title).strip("_")
            if not safe_title:
                safe_title = sub_id or f"review_{len(reviews)}"

            reviews.append(
                {
                    "filename": safe_title,
                    "title": title,
                    "article_type": art_type,
                    "sub_id": sub_id,
                    "doi": doi,
                    "contributors": contribs,
                    "lines": body_lines,
                }
            )

        return reviews

    async def _ls_pmc_reviews(self, pmc_id: str, start_time: float) -> dict:
        """List peer review rounds for a PMC paper."""
        import asyncio

        try:
            reviews = await asyncio.to_thread(self._parse_pmc_reviews, pmc_id)
        except Exception as e:
            return {
                "path": f"/papers/{pmc_id}/reviews/",
                "count": 0,
                "contents": [],
                "error": str(e),
                "time_ms": round((time.perf_counter() - start_time) * 1000),
            }

        if not reviews:
            return {
                "path": f"/papers/{pmc_id}/reviews/",
                "count": 0,
                "contents": [],
                "note": "No peer review data found for this paper",
                "time_ms": round((time.perf_counter() - start_time) * 1000),
            }

        contents = []
        for r in reviews:
            entry = {
                "name": f"{r['filename']}.lines",
                "title": r["title"],
                "type": r["article_type"],
                "lines": len(r["lines"]),
            }
            if r["doi"]:
                entry["doi"] = r["doi"]
            if r["contributors"]:
                entry["contributors"] = r["contributors"]
            contents.append(entry)

        return {
            "path": f"/papers/{pmc_id}/reviews/",
            "count": len(contents),
            "contents": contents,
            "time_ms": round((time.perf_counter() - start_time) * 1000),
        }

    async def _cat_pmc_review(
        self,
        pmc_id: str,
        filename: str,
        start: int = None,
        end: int = None,
        start_time: float = None,
    ) -> dict:
        """Read a specific peer review file for a PMC paper."""
        import asyncio

        start_time = start_time or time.perf_counter()

        # Strip .lines suffix
        target = filename
        if target.endswith(".lines"):
            target = target[:-6]

        try:
            reviews = await asyncio.to_thread(self._parse_pmc_reviews, pmc_id)
        except Exception as e:
            return {"error": f"Cannot read review: {str(e)[:100]}"}

        matched = None
        for r in reviews:
            if r["filename"] == target:
                matched = r
                break

        if matched is None:
            available = [r["filename"] for r in reviews]
            return {"error": f"Review not found: {target}", "available": available}

        lines = []
        # Header
        lines.append(f"# {matched['title']}")
        if matched.get("doi"):
            lines.append(f"DOI: {matched['doi']}")
        if matched.get("article_type"):
            lines.append(f"Type: {matched['article_type']}")
        if matched.get("contributors"):
            lines.append(f"From: {', '.join(matched['contributors'])}")
        lines.append("")
        lines.extend(matched["lines"])

        all_lines = lines
        if start:
            all_lines = [l for i, l in enumerate(all_lines) if (i + 1) >= start]
        if end:
            all_lines = [l for i, l in enumerate(all_lines) if (i + 1) <= end]

        return {
            "document_id": pmc_id,
            "filename": filename,
            "lines": [
                {"line": (start or 1) + i, "content": l}
                for i, l in enumerate(all_lines[:3000])
            ],
            "count": min(len(all_lines), 3000),
            "total_lines": len(lines),
            "time_ms": round((time.perf_counter() - start_time) * 1000),
        }

    async def _read_pmc_supplement_text(
        self,
        pmc_id: str,
        filename: str,
        start: int = None,
        end: int = None,
        start_time: float = None,
    ) -> dict:
        """Read processed supplement .md from GCS for a PMC paper."""
        import asyncio
        import json as _json

        start_time = start_time or time.perf_counter()

        # "file.pdf.md.lines" → look up processed md for "file.pdf"
        base_filename = filename
        if base_filename.endswith(".md.lines"):
            base_filename = base_filename[:-9]
        elif base_filename.endswith(".cheatsheet.md"):
            base_filename = base_filename[:-14]

        def _read():
            gcs_bucket = _get_gcs_bucket()
            proc_prefix = f"pmc/articles/{pmc_id}/_processed/v1/"

            # Find the right .md via the manifest
            try:
                manifest_blob = gcs_bucket.blob(f"{proc_prefix}_manifest.json")
                manifest = _json.loads(manifest_blob.download_as_text())
                for f in manifest.get("files", []):
                    if f.get("source") == base_filename and f.get("output"):
                        md_blob = gcs_bucket.blob(f"{proc_prefix}{f['output']}")
                        return md_blob.download_as_text()
            except Exception:
                pass

            # Fallback: try {base_filename}.md directly
            md_blob = gcs_bucket.blob(f"{proc_prefix}{base_filename}.md")
            if md_blob.exists():
                return md_blob.download_as_text()

            # Try the base filename directly in the processed folder
            for blob in gcs_bucket.list_blobs(prefix=proc_prefix, max_results=50):
                rel = blob.name[len(proc_prefix) :]
                if rel == "_manifest.json":
                    continue
                if base_filename.split(".")[0] in rel:
                    return blob.download_as_text()
            return None

        try:
            text = await asyncio.to_thread(_read)
        except Exception as e:
            return {"error": f"Cannot read supplement: {str(e)[:100]}"}

        if text is None:
            return {"error": f"Processed supplement not found: {filename}"}

        lines = text.split("\n")
        if start:
            lines = [l for i, l in enumerate(lines) if (i + 1) >= start]
        if end:
            lines = [l for i, l in enumerate(lines) if (i + 1) <= end]

        return {
            "document_id": pmc_id,
            "filename": filename,
            "lines": [
                {"line": (start or 1) + i, "content": l}
                for i, l in enumerate(lines[:2000])
            ],
            "count": min(len(lines), 2000),
            "total_lines": text.count("\n") + 1,
            "time_ms": round((time.perf_counter() - start_time) * 1000),
        }

    async def _cat_pmc_meta(self, pmc_id: str, path: str, start_time: float) -> dict:
        """Return meta.json content for a PMC paper."""
        import asyncio

        def _query():
            conn = self._get_pmc_db_connection()
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT pmc_id, pmid, doi, title, authors, abstract_text,
                              article_type, source, journal_title, pub_year, pub_date,
                              categories, keywords
                       FROM documents WHERE pmc_id = %s""",
                    (pmc_id,),
                )
                return cur.fetchone()

        row = await asyncio.to_thread(_query)
        if not row:
            return {"error": f"PMC paper not found: {pmc_id}"}
        return {
            "path": path,
            "type": "json",
            "content": {
                "document_id": row[0],
                "pmc_id": row[0],
                "pmid": row[1],
                "doi": row[2],
                "title": row[3],
                "authors": row[4],
                "abstract": row[5],
                "article_type": row[6],
                "source": row[7] or "pmc",
                "journal": row[8],
                "pub_year": row[9],
                "pub_date": str(row[10]) if row[10] else None,
                "categories": row[11],
                "keywords": row[12],
            },
            "time_ms": round((time.perf_counter() - start_time) * 1000),
        }

    async def _read_pmc_lines(
        self, pmc_id: str, start: int, end: int, start_time: float, section: str = None
    ) -> dict:
        """Read content lines for a PMC paper."""
        import asyncio

        TRUNCATE_LINES = 100

        # Get pub_year from cache (O(1) after first lookup) for partition pruning
        pub_year = self._get_pmc_pub_year(pmc_id)

        def _query():
            conn = self._get_pmc_db_connection()
            with conn.cursor() as cur:
                # pub_year enables single-partition index scan (~0.2ms vs ~58ms without)
                if section and pub_year:
                    cur.execute(
                        """SELECT line_number, content FROM content_blocks
                           WHERE pmc_id = %s AND section ILIKE %s AND pub_year = %s
                           ORDER BY line_number""",
                        (pmc_id, f"%{section}%", pub_year),
                    )
                elif section:
                    cur.execute(
                        "SELECT line_number, content FROM content_blocks WHERE pmc_id = %s AND section ILIKE %s ORDER BY line_number",
                        (pmc_id, f"%{section}%"),
                    )
                elif pub_year:
                    cur.execute(
                        "SELECT line_number, content FROM content_blocks WHERE pmc_id = %s AND pub_year = %s ORDER BY line_number",
                        (pmc_id, pub_year),
                    )
                else:
                    cur.execute(
                        "SELECT line_number, content FROM content_blocks WHERE pmc_id = %s ORDER BY line_number",
                        (pmc_id,),
                    )
                return cur.fetchall()

        rows = await asyncio.to_thread(_query)
        if start is not None or end is not None:
            s = (start or 1) - 1
            e = end or len(rows)
            rows = rows[s:e]
        truncated = len(rows) > TRUNCATE_LINES
        display = rows[:TRUNCATE_LINES] if truncated else rows
        lines = [
            {
                "line": r[0],
                "content": r[1],
            }
            for r in display
        ]
        return {
            "path": f"/papers/{pmc_id}/content.lines",
            "type": "lines",
            "lines": lines,
            "total_lines": len(rows),
            "truncated": truncated,
            "time_ms": round((time.perf_counter() - start_time) * 1000),
        }

    async def _read_arxiv_lines(
        self, arxiv_id: str, start: int, end: int, start_time: float, section: str = None
    ) -> dict:
        """Read content lines for an arxiv paper from the arxiv database."""
        import asyncio
        from modules.papers.short_ids import bare_arxiv_id

        bare_id = bare_arxiv_id(arxiv_id)

        def _query():
            conn = self._get_arxiv_db_connection()
            with conn.cursor() as cur:
                count_query = "SELECT COUNT(*) FROM content_blocks WHERE document_id = %s"
                count_params = [bare_id]
                if section:
                    count_query += " AND section ILIKE %s"
                    count_params.append(f"%{section}%")
                cur.execute(count_query, count_params)
                total = cur.fetchone()[0]

                query = "SELECT id, line_number, content FROM content_blocks WHERE document_id = %s"
                params = [bare_id]
                if section:
                    query += " AND section ILIKE %s"
                    params.append(f"%{section}%")
                if start:
                    query += " AND line_number >= %s"
                    params.append(start)
                if end:
                    query += " AND line_number <= %s"
                    params.append(end)
                query += " ORDER BY line_number"
                cur.execute(query, params)
                return total, cur.fetchall()

        total_lines, rows = await asyncio.to_thread(_query)
        if not rows:
            return {
                "error": f"No content found for {arxiv_id}"
                + (f" section {section}" if section else ""),
                "error_type": "not_found",
            }
        return {
            "document_id": arxiv_id,
            "section": section,
            "lines": [
                {"line": r[1] + 1, "content": r[2]} for r in rows
            ],
            "count": len(rows),
            "total_lines": total_lines,
            "time_ms": round((time.perf_counter() - start_time) * 1000),
        }

    async def _ls_arxiv_document(self, arxiv_id: str, start_time: float) -> dict:
        """List an arxiv paper directory."""
        import asyncio
        from modules.papers.short_ids import bare_arxiv_id

        bare_id = bare_arxiv_id(arxiv_id)

        def _query():
            conn = self._get_arxiv_db_connection()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT title, doi, source FROM documents WHERE document_id = %s",
                    (bare_id,),
                )
                doc = cur.fetchone()
                if not doc:
                    return None, None
                cur.execute(
                    """SELECT COUNT(*) as total_lines,
                              COUNT(DISTINCT section) as section_count,
                              COUNT(CASE WHEN citation_info->>'source_type' LIKE '%%supplement%%' THEN 1 END) as has_supplements,
                              COUNT(CASE WHEN block_type = 'figure' THEN 1 END) as figure_count
                       FROM content_blocks WHERE document_id = %s""",
                    (bare_id,),
                )
                stats = cur.fetchone()
                return doc, stats

        doc, stats = await asyncio.to_thread(_query)
        if not doc:
            return {"error": f"Paper not found: {arxiv_id}"}

        contents = ["meta.json", f"content.lines  ({stats[0]} lines)", "sections/"]
        if stats[2]:
            contents.append("supplements/")
        if stats[3]:
            contents.append("figures/")
        return {
            "path": f"/papers/{arxiv_id}/",
            "title": doc[0],
            "doi": doc[1],
            "source": doc[2],
            "contents": contents,
            "time_ms": round((time.perf_counter() - start_time) * 1000),
        }

    async def _ls_arxiv_sections(self, arxiv_id: str, start_time: float) -> dict:
        """List sections for an arxiv paper."""
        import asyncio
        from modules.papers.short_ids import bare_arxiv_id

        bare_id = bare_arxiv_id(arxiv_id)

        def _query():
            conn = self._get_arxiv_db_connection()
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT section, COUNT(*) as lines
                       FROM content_blocks
                       WHERE document_id = %s AND section IS NOT NULL
                       GROUP BY section ORDER BY MIN(line_number)""",
                    (bare_id,),
                )
                return cur.fetchall()

        rows = await asyncio.to_thread(_query)
        contents = [
            {"name": f"{r[0]}.lines", "lines": r[1]}
            for r in rows if r[0]
        ]
        return {
            "path": f"/papers/{arxiv_id}/sections/",
            "contents": contents,
            "time_ms": round((time.perf_counter() - start_time) * 1000),
        }

    async def _cat_arxiv_meta(self, arxiv_id: str, path: str, start_time: float) -> dict:
        """Read meta.json for an arxiv paper."""
        import asyncio
        from modules.papers.short_ids import bare_arxiv_id

        bare_id = bare_arxiv_id(arxiv_id)

        def _query():
            conn = self._get_arxiv_db_connection()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT document_id, title, doi, source, authors, pub_date, abstract_text "
                    "FROM documents WHERE document_id = %s",
                    (bare_id,),
                )
                return cur.fetchone()

        row = await asyncio.to_thread(_query)
        if not row:
            return {"error": f"Document not found: {arxiv_id}"}

        return {
            "path": path,
            "type": "json",
            "content": {
                "document_id": arxiv_id,
                "title": row[1],
                "doi": row[2],
                "source": row[3] or "arxiv",
                "authors": row[4],
                "pub_date": str(row[5]) if row[5] else "",
                "abstract": row[6],
            },
            "time_ms": round((time.perf_counter() - start_time) * 1000),
        }

    async def _ls(
        self,
        path: str,
        query: str = None,
        limit: int = 20,
        session_id: str = "default",
    ) -> dict:
        """List contents of a virtual path - handles all paper path types."""
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
                "hint": f"Showing {len(rows)} of {total_count:,} papers. Use papers_find to search, or cd /papers/ID/ to explore a specific paper.",
                "time_ms": round((time.perf_counter() - start_time) * 1000),
            }

        if parsed.type == "document":
            uuid = parsed.document_id
            # Route PMC IDs to the pmc database
            if uuid.upper().startswith("PMC") and uuid[3:].isdigit():
                return await self._ls_pmc_document(uuid, start_time)
            # Route OpenAlex IDs (abstract-only) to OpenSearch
            if uuid.lower().startswith("oa_"):
                return await self._ls_oa_document(uuid, start_time)
            # Route arxiv IDs to the arxiv database
            from modules.papers.short_ids import is_arxiv_id
            if is_arxiv_id(uuid):
                return await self._ls_arxiv_document(uuid, start_time)

            with conn.cursor() as cur:
                cur.execute(
                    "SELECT title, doi, source FROM documents WHERE document_id::text = %s",
                    (uuid,),
                )
                doc = cur.fetchone()

            if not doc:
                return {"error": f"Paper not found: {uuid}"}

            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT COUNT(*) as total_lines,
                                  COUNT(DISTINCT section) as section_count,
                                  COUNT(CASE WHEN citation_info->>'source_type' LIKE '%%supplement%%' THEN 1 END) as has_supplements,
                                  COUNT(CASE WHEN block_type = 'figure' THEN 1 END) as figure_count
                           FROM content_blocks WHERE document_id::text = %s""",
                        (uuid,),
                    )
                    stats = cur.fetchone()
            except Exception:
                stats = (0, 0, 0, 0)

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
            # OpenAlex routing (abstract-only)
            if uuid.lower().startswith("oa_"):
                return await self._ls_oa_sections(uuid, start_time)
            # arxiv routing
            from modules.papers.short_ids import is_arxiv_id
            if is_arxiv_id(uuid):
                return await self._ls_arxiv_sections(uuid, start_time)
            # PMC routing
            if uuid.upper().startswith("PMC") and uuid[3:].isdigit():
                import asyncio

                pub_year = self._get_pmc_pub_year(uuid)

                def _pmc_sections():
                    pmc_conn = self._get_pmc_db_connection()
                    with pmc_conn.cursor() as cur:
                        if pub_year:
                            cur.execute(
                                """SELECT section, COUNT(*) as lines
                                   FROM content_blocks
                                   WHERE pmc_id = %s AND section IS NOT NULL AND pub_year = %s
                                   GROUP BY section ORDER BY MIN(line_number)""",
                                (uuid, pub_year),
                            )
                        else:
                            cur.execute(
                                """SELECT section, COUNT(*) as lines
                                   FROM content_blocks
                                   WHERE pmc_id = %s AND section IS NOT NULL
                                   GROUP BY section ORDER BY MIN(line_number)""",
                                (uuid,),
                            )
                        return cur.fetchall()

                rows = asyncio.get_event_loop().run_until_complete(
                    asyncio.to_thread(_pmc_sections)
                )
            else:
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

            if uuid.upper().startswith("PMC") and uuid[3:].isdigit():
                return await self._ls_pmc_supplements(uuid, start_time)

            from modules.papers.short_ids import is_arxiv_id, bare_arxiv_id
            if is_arxiv_id(uuid):
                import asyncio
                _bare = bare_arxiv_id(uuid)

                def _arxiv_supps():
                    arxiv_conn = self._get_arxiv_db_connection()
                    with arxiv_conn.cursor() as cur:
                        cur.execute(
                            """SELECT DISTINCT
                                   citation_info->>'supplement_filename' as supp_filename,
                                   citation_info->>'source_type' as source_type,
                                   COUNT(*) as lines
                               FROM content_blocks
                               WHERE document_id = %s
                               AND citation_info->>'source_type' LIKE '%%supplement%%'
                               GROUP BY citation_info->>'supplement_filename', citation_info->>'source_type'""",
                            (_bare,),
                        )
                        return cur.fetchall()

                db_rows = await asyncio.to_thread(_arxiv_supps)
            else:
                # 1) Get processed supplements from DB
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT DISTINCT 
                               citation_info->>'supplement_filename' as supp_filename,
                               citation_info->>'source_type' as source_type,
                               COUNT(*) as lines
                           FROM content_blocks 
                           WHERE document_id = %s 
                           AND citation_info->>'source_type' LIKE '%%supplement%%'
                           GROUP BY citation_info->>'supplement_filename', citation_info->>'source_type'""",
                        (uuid,),
                    )
                    db_rows = cur.fetchall()

            contents = []
            seen_files = set()

            for row in db_rows:
                filename = row[0] or "unknown"
                source_type = row[1] or ""
                lines = row[2]

                if filename in seen_files:
                    continue
                seen_files.add(filename)

                contents.append({"name": filename})

                if "pdf" in source_type or "docx" in source_type:
                    contents.append({"name": f"{filename}.md.lines", "lines": lines})
                elif "excel" in source_type:
                    contents.append(
                        {"name": f"{filename}.cheatsheet.md", "lines": lines}
                    )

            # 2) List ALL raw files from GCS content/supplements/
            #    so the agent can see .fa, .csv, .txt, .tif, etc.
            try:
                gcs_bucket = _get_gcs_bucket()
                month = self._get_month_for_doc(uuid, conn)
                if month:
                    source_prefix = (
                        "biorxiv" if self._is_biorxiv(uuid, conn) else "medrxiv"
                    )
                    gcs_prefix = (
                        f"{source_prefix}_extracted/{month}/{uuid}/content/supplements/"
                    )
                    skip_exts = {".jpg", ".json"}
                    for blob in gcs_bucket.list_blobs(
                        prefix=gcs_prefix, max_results=100
                    ):
                        fname = blob.name.split("/")[-1]
                        if not fname or fname in seen_files:
                            continue
                        ext = fname[fname.rfind(".") :].lower() if "." in fname else ""
                        if ext in skip_exts:
                            continue
                        if len(fname) == 36 and "_img." in fname:
                            continue
                        seen_files.add(fname)
                        contents.append({"name": fname})
            except Exception:
                pass

            return {
                "path": f"/papers/{uuid}/supplements/",
                "count": len(contents),
                "contents": contents,
                "time_ms": round((time.perf_counter() - start_time) * 1000),
            }

        if parsed.type == "figures_list":
            uuid = parsed.document_id
            is_pmc = uuid.upper().startswith("PMC") and uuid[3:].isdigit()

            from modules.papers.short_ids import is_arxiv_id, bare_arxiv_id
            is_arxiv = is_arxiv_id(uuid)

            contents = []
            if is_pmc:
                # PMC: list figure images from GCS
                import asyncio

                def _list_pmc_figures():
                    gcs_bucket = _get_gcs_bucket()
                    prefix = f"pmc/articles/{uuid}/"
                    figure_exts = {
                        ".gif",
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".tif",
                        ".tiff",
                        ".svg",
                    }
                    figs = []
                    for blob in gcs_bucket.list_blobs(prefix=prefix, max_results=200):
                        rel = blob.name[len(prefix) :]
                        if not rel or rel.startswith("_processed/"):
                            continue
                        ext = rel[rel.rfind(".") :].lower() if "." in rel else ""
                        if ext not in figure_exts:
                            continue
                        lower_rel = rel.lower()
                        if "suppl" in lower_rel or lower_rel.startswith("mmc"):
                            continue
                        figs.append({"name": rel, "type": "image", "size": blob.size})
                    return figs

                try:
                    contents = await asyncio.to_thread(_list_pmc_figures)
                except Exception as e:
                    logger.warning(f"[PMC figures] GCS listing failed for {uuid}: {e}")
            elif is_arxiv:
                import asyncio

                bare_id = bare_arxiv_id(uuid)

                def _list_arxiv_figures():
                    arxiv_conn = self._get_arxiv_db_connection()
                    with arxiv_conn.cursor() as cur:
                        cur.execute(
                            """SELECT citation_info->>'xml_id' as figure_id,
                                      content,
                                      citation_info->>'graphic' as graphic
                               FROM content_blocks
                               WHERE document_id = %s AND block_type = 'figure'
                               ORDER BY line_number""",
                            (bare_id,),
                        )
                        return cur.fetchall()

                rows = await asyncio.to_thread(_list_arxiv_figures)
                for row in rows:
                    figure_id = row[0]
                    content = row[1] or ""
                    graphic = row[2]
                    if graphic:
                        contents.append(
                            {
                                "name": graphic,
                                "type": "image",
                                "figure_id": figure_id,
                                "caption": (
                                    content[:80] + "..."
                                    if len(content) > 80
                                    else content
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
            else:
                # bioRxiv/medRxiv: list from content_blocks
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

                for row in rows:
                    figure_id = row[0]
                    content = row[1] or ""
                    graphic = row[2]
                    if graphic:
                        contents.append(
                            {
                                "name": graphic,
                                "type": "image",
                                "figure_id": figure_id,
                                "caption": (
                                    content[:80] + "..."
                                    if len(content) > 80
                                    else content
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
                "hint": f"To save a figure: paperclip cat /papers/{uuid}/figures/<filename> > <filename>",
                "time_ms": round((time.perf_counter() - start_time) * 1000),
            }

        if parsed.type == "figure_file":
            # Show info about a specific figure - match by graphic filename OR figure_id
            uuid = parsed.document_id
            filename = parsed.filename

            from modules.papers.short_ids import is_arxiv_id, bare_arxiv_id
            is_pmc = uuid.upper().startswith("PMC") and uuid[3:].isdigit()

            if is_arxiv_id(uuid):
                import asyncio
                _bare = bare_arxiv_id(uuid)

                def _arxiv_fig():
                    arxiv_conn = self._get_arxiv_db_connection()
                    with arxiv_conn.cursor() as cur:
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
                                    OR citation_info->>'xml_id' ILIKE %s)
                               LIMIT 1""",
                            (
                                _bare,
                                filename,
                                filename,
                                f"%{filename.replace('.tif', '').replace('.jpg', '').replace('.png', '')}%",
                            ),
                        )
                        return cur.fetchone()

                row = await asyncio.to_thread(_arxiv_fig)

            elif is_pmc:
                import asyncio

                def _check_pmc_figure():
                    gcs_bucket = _get_gcs_bucket()
                    blob = gcs_bucket.blob(f"pmc/articles/{uuid}/{filename}")
                    if blob.exists():
                        return {"size": blob.size}
                    return None

                try:
                    info = await asyncio.to_thread(_check_pmc_figure)
                except Exception:
                    info = None

                if info:
                    return {
                        "path": f"/papers/{uuid}/figures/{filename}",
                        "type": "figure",
                        "figure_id": filename,
                        "graphic": filename,
                        "caption": "",
                        "line_number": None,
                        "hint": f"Use ask_image /papers/{uuid}/figures/{filename} 'What does this figure show?' to analyze this figure",
                        "time_ms": round((time.perf_counter() - start_time) * 1000),
                    }
                return {
                    "error": f"Figure not found: {filename}",
                    "hint": f"Use 'ls /papers/{uuid}/figures/' to see available figures",
                }

            else:
                # bioRxiv/medRxiv: match by graphic filename OR figure_id
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
                "hint": f"To save: paperclip cat /papers/{uuid}/figures/{graphic or filename} > {graphic or filename}",
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
                    "hint": f"To save: paperclip cat /papers/{parsed.document_id}/supplements/{filename} > {filename}",
                }

        if parsed.type == "reviews_list":
            doc_id = parsed.document_id
            if doc_id.upper().startswith("PMC") and doc_id[3:].isdigit():
                return await self._ls_pmc_reviews(doc_id, start_time)
            return {
                "error": "Peer review data is only available for PMC papers (primarily PLOS journals)"
            }

        if parsed.type == "review_file":
            doc_id = parsed.document_id
            if not (doc_id.upper().startswith("PMC") and doc_id[3:].isdigit()):
                return {"error": "Peer review data is only available for PMC papers"}
            filename = parsed.filename
            if filename.endswith(".lines"):
                filename = filename[:-6]
            return {
                "path": f"/papers/{doc_id}/reviews/{parsed.filename}",
                "hint": f"Use cat /papers/{doc_id}/reviews/{parsed.filename} to read this review file",
            }

        return {"error": f"Cannot list path: {path}", "path_type": parsed.type}

    async def _cat(
        self,
        path: str,
        start: int = None,
        end: int = None,
        session_id: str = "default",
        truncate: bool = True,
    ) -> dict:
        """Read file contents - handles all paper file types."""
        start_time = time.perf_counter()
        parsed = self.path_parser.parse(path)

        if parsed.error:
            return {"error": parsed.error, "path": path}

        conn = _get_db_connection()

        # OpenAlex routing — delegate to OpenSearch-backed (abstract-only) handler
        doc_id = parsed.document_id or ""
        if doc_id.lower().startswith("oa_"):
            return await self._cat_oa(doc_id, parsed, path, start_time)

        # PMC routing — delegate to PMC-specific handlers
        if doc_id.upper().startswith("PMC") and doc_id[3:].isdigit():
            if parsed.type == "file" and parsed.filename == "meta.json":
                return await self._cat_pmc_meta(doc_id, path, start_time)
            if (
                parsed.type in ("file", "content")
                and parsed.filename == "content.lines"
            ):
                slab_result = await self._try_slab_cat(
                    doc_id, start_time, truncate=truncate
                )
                if slab_result is not None:
                    return slab_result
                if start is not None or end is not None:
                    return await self._read_pmc_lines(doc_id, start, end, start_time)
                return {"error": f"Slab service unavailable for {doc_id}. Cat requires the slab-cat service on the grep VM."}
            if parsed.type == "section":
                return await self._read_pmc_lines(
                    doc_id, start, end, start_time, section=parsed.section
                )

        # arxiv routing — slab is the primary path (arxiv is fully indexed)
        from modules.papers.short_ids import is_arxiv_id, bare_arxiv_id
        if is_arxiv_id(doc_id):
            if parsed.type == "file" and parsed.filename == "meta.json":
                return await self._cat_arxiv_meta(doc_id, path, start_time)
            if (
                parsed.type in ("file", "content")
                and parsed.filename == "content.lines"
            ):
                slab_result = await self._try_slab_cat(
                    bare_arxiv_id(doc_id),
                    start_time,
                    timeout=5,
                    truncate=truncate,
                )
                if slab_result is not None:
                    return slab_result
                if start is not None or end is not None:
                    return await self._read_arxiv_lines(doc_id, start, end, start_time)
                return {"error": f"Slab service unavailable for {doc_id}. Cat requires the slab-cat service on the grep VM."}
            if parsed.type == "section":
                return await self._read_arxiv_lines(
                    doc_id, start, end, start_time, section=parsed.section
                )

        # meta.json
        if parsed.type == "file" and parsed.filename == "meta.json":
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT document_id, title, doi, source, authors, pub_date, abstract_text FROM documents WHERE document_id::text = %s",
                    (parsed.document_id,),
                )
                row = cur.fetchone()
            if not row:
                return {"error": f"Document not found: {parsed.document_id}"}
            return {
                "path": path,
                "type": "json",
                "content": {
                    "document_id": shorten(str(row[0]), row[3]),
                    "title": row[1],
                    "doi": row[2],
                    "source": row[3],
                    "authors": row[4],
                    "pub_date": str(row[5]) if row[5] else "",
                    "abstract": row[6],
                },
                "time_ms": round((time.perf_counter() - start_time) * 1000),
            }

        # content.lines - full document (slab-backed, no SQL fallback)
        if parsed.type == "file" and parsed.filename == "content.lines":
            slab_result = await self._try_slab_cat(
                parsed.document_id, start_time, truncate=truncate
            )
            if slab_result is not None:
                return slab_result
            if start is not None or end is not None:
                return await self._read_lines(parsed.document_id, start, end, start_time)
            return {"error": f"Slab service unavailable for {parsed.document_id}. Cat requires the slab-cat service on the grep VM."}

        # Section file (always SQL — slabs store concatenated text)
        if parsed.type == "section":
            return await self._read_lines(
                parsed.document_id, start, end, start_time, section=parsed.section
            )

        # Supplement text
        if parsed.type == "supplement_text":
            doc_id = parsed.document_id
            if doc_id.upper().startswith("PMC") and doc_id[3:].isdigit():
                return await self._read_pmc_supplement_text(
                    doc_id, parsed.filename, start, end, start_time
                )
            from modules.papers.short_ids import is_arxiv_id, bare_arxiv_id
            if is_arxiv_id(doc_id):
                return await self._read_arxiv_supplement_lines(
                    doc_id, parsed.filename, start, end, start_time
                )
            return await self._read_supplement_lines(
                doc_id, parsed.filename, start, end, start_time
            )

        # Peer review file
        if parsed.type == "review_file":
            doc_id = parsed.document_id
            if doc_id.upper().startswith("PMC") and doc_id[3:].isdigit():
                return await self._cat_pmc_review(
                    doc_id, parsed.filename, start, end, start_time
                )
            return {"error": "Peer review data is only available for PMC papers"}

        # Figure file — resolve to a signed GCS download URL so the client
        # can stream bytes when stdout is redirected (paperclip cat ... > file).
        if parsed.type == "figure_file":
            filename = parsed.filename
            doc_id = parsed.document_id
            from modules.papers.short_ids import is_arxiv_id, bare_arxiv_id
            if is_arxiv_id(doc_id):
                arxiv_conn = self._get_arxiv_db_connection()
                resolved = self._resolve_figure_download(
                    bare_arxiv_id(doc_id), filename, arxiv_conn
                )
            else:
                resolved = self._resolve_figure_download(doc_id, filename, conn)
            if resolved.get("download_url"):
                return {
                    "type": "binary",
                    "download_url": resolved["download_url"],
                    "filename": resolved.get("filename", filename),
                    "caption": resolved.get("caption", ""),
                    "hint": (
                        f"Redirect to a file to save: "
                        f"paperclip cat /papers/{doc_id}/figures/{filename} > {filename}"
                    ),
                    "time_ms": round((time.perf_counter() - start_time) * 1000),
                }
            return {
                "type": "binary",
                "error": f"Cannot read image file: {filename}",
                "hint": (
                    f"To save the image, redirect to a file:\n"
                    f"  paperclip cat /papers/{doc_id}/figures/{filename} > {filename}\n"
                    f"To analyze with a vision model instead:\n"
                    f'  paperclip ask-image /papers/{doc_id}/figures/{filename} "Describe this figure"'
                ),
            }

        # Supplement file — handle by extension
        if parsed.type == "supplement_file":
            filename = parsed.filename
            image_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif")
            text_exts = (
                ".fa",
                ".fasta",
                ".csv",
                ".tsv",
                ".txt",
                ".xml",
                ".gff",
                ".bed",
                ".sam",
                ".vcf",
            )

            if filename.endswith(image_exts):
                doc_id = parsed.document_id
                if is_arxiv_id(doc_id):
                    _fig_conn = self._get_arxiv_db_connection()
                    resolved = self._resolve_figure_download(
                        bare_arxiv_id(doc_id), filename, _fig_conn
                    )
                else:
                    resolved = self._resolve_figure_download(doc_id, filename, conn)
                if resolved.get("download_url"):
                    return {
                        "type": "binary",
                        "download_url": resolved["download_url"],
                        "filename": resolved.get("filename", filename),
                        "time_ms": round((time.perf_counter() - start_time) * 1000),
                    }
                return {
                    "type": "binary",
                    "error": f"Cannot read image file: {filename}",
                    "hint": (
                        f"To save, redirect to a file:\n"
                        f"  paperclip cat /papers/{doc_id}/supplements/{filename} > {filename}\n"
                        f"To analyze with a vision model:\n"
                        f'  paperclip ask-image /papers/{doc_id}/supplements/{filename} "describe"'
                    ),
                }

            if filename.endswith(text_exts) or filename.endswith((".fq", ".fastq")):
                if not self._is_safe_filename(filename):
                    return {"error": f"Invalid supplement filename: {filename!r}"}
                try:
                    uuid = parsed.document_id
                    gcs_bucket = _get_gcs_bucket()
                    if uuid.upper().startswith("PMC") and uuid[3:].isdigit():
                        gcs_path = f"pmc/articles/{uuid}/{filename}"
                    elif is_arxiv_id(uuid):
                        _bare = bare_arxiv_id(uuid)
                        gcs_path = f"arxiv_extracted/parsed/{_bare}.pdf/{filename}"
                    else:
                        month = self._get_month_for_doc(uuid, conn)
                        source_prefix = (
                            "biorxiv" if self._is_biorxiv(uuid, conn) else "medrxiv"
                        )
                        gcs_path = f"{source_prefix}_extracted/{month}/{uuid}/content/supplements/{filename}"
                    blob = gcs_bucket.blob(gcs_path)
                    max_bytes = 100_000
                    text = blob.download_as_text()
                    if len(text) > max_bytes:
                        text = (
                            text[:max_bytes]
                            + f"\n\n... [truncated at {max_bytes//1000}KB — use Python to process the full file]"
                        )
                    lines = text.split("\n")
                    return {
                        "document_id": uuid,
                        "filename": filename,
                        "lines": [
                            {"line": i + 1, "content": l}
                            for i, l in enumerate(lines[:2000])
                        ],
                        "count": min(len(lines), 2000),
                        "total_lines": len(lines),
                        "time_ms": round((time.perf_counter() - start_time) * 1000),
                    }
                except Exception:
                    # M2: don't surface GCS internals (request IDs, bucket
                    # names, SA details) to the client.
                    logger.exception(
                        "Supplement text read failed for %s/%s", uuid, filename
                    )
                    return {"error": f"Cannot read {filename}"}

            return {
                "error": f"Cannot read binary file: {filename}",
                "hint": f"Binary file. Download via GCS or use Python to process.",
            }

        return {"error": f"Cannot read path: {path}"}

    async def _try_slab_cat(
        self,
        document_id: str,
        start_time: float,
        timeout: int | None = None,
        truncate: bool = True,
    ) -> dict | None:
        """Read full document text from the slab-cat service.

        Returns formatted line output compatible with _read_lines, or None
        if the slab service is unavailable.
        """
        try:
            from slab_grep_client import slab_grep_cat
        except ImportError:
            return None

        slab_doc_id = resolve(document_id)
        result = await asyncio.to_thread(slab_grep_cat, slab_doc_id, timeout=timeout)
        if result is None or "error" in result:
            return None

        content = result.get("content", "")
        if not content:
            return None

        raw_lines = content.split("\n")
        total_lines = len(raw_lines)

        line_prefix_re = re.compile(r"^/papers/[^:]+:L(\d+):\s?(.*)$")
        normalized_lines: list[dict] = []
        for idx, line in enumerate(raw_lines, start=1):
            m = line_prefix_re.match(line)
            if m:
                try:
                    line_num = int(m.group(1))
                except ValueError:
                    line_num = idx
                normalized_lines.append({"line": line_num, "content": m.group(2)})
            else:
                normalized_lines.append({"line": idx, "content": line})

        # Return the full slab text to callers. The terminal `cat` formatter
        # handles user-facing truncation; internal consumers like `grep` need
        # all lines so matches near the end of a paper (e.g. references) are
        # not silently missed.
        TRUNCATE_LINES = 100
        truncated = False
        lines = normalized_lines

        time_ms = round((time.perf_counter() - start_time) * 1000)
        resp = {
            "document_id": document_id,
            "lines": lines,
            "count": len(lines),
            "total_lines": total_lines,
            "time_ms": time_ms,
            "backend": "slab",
        }
        if truncated:
            resp["truncated"] = True
            resp["hint"] = (
                f"Showing first {TRUNCATE_LINES} of {total_lines} lines. Use line ranges: cat -n 100-200 <path>"
            )

        logger.info(
            f"[slab-cat] {document_id}: {total_lines} lines, "
            f"{result.get('length_bytes', 0)} bytes in {time_ms}ms"
        )
        return resp

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
        count_query = "SELECT COUNT(*) FROM content_blocks WHERE document_id::text = %s"
        count_params = [document_id]
        if section:
            count_query += " AND section = %s"
            count_params.append(section)

        query = "SELECT id, line_number, content FROM content_blocks WHERE document_id::text = %s"
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
            "lines": [
                {"line": r[1] + 1, "content": r[2]} for r in rows
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

        # Derive the actual supplement_filename from the virtual filename
        # e.g. "file02.docx.md.lines" -> "file02.docx"
        #      "file03.xlsx.cheatsheet.md" -> "file03.xlsx"
        #      "file02.pdf.md.lines" -> "file02.pdf"
        #      legacy: "file02.content.md.lines" -> strip to "file02"
        base_filename = filename
        if base_filename.endswith(".md.lines"):
            base_filename = base_filename[:-9]  # strip ".md.lines"
        elif base_filename.endswith(".cheatsheet.md"):
            base_filename = base_filename[:-14]  # strip ".cheatsheet.md"
        elif base_filename.endswith(".lines"):
            base_filename = base_filename[:-6]  # legacy ".lines"

        query = """
            SELECT id, line_number, content 
            FROM content_blocks 
            WHERE document_id = %s 
            AND (citation_info->>'supplement_filename' = %s
                 OR citation_info->>'source_path' LIKE %s)
        """
        params = [document_id, base_filename, f"%{base_filename.replace('%', '%%')}%"]

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
            "lines": [
                {"line": r[1] + 1, "content": r[2]} for r in rows
            ],
            "count": len(rows),
            "total_lines": total_lines,
            "time_ms": round((time.perf_counter() - start_time) * 1000),
        }

    async def _read_arxiv_supplement_lines(
        self,
        document_id: str,
        filename: str,
        start: int = None,
        end: int = None,
        start_time: float = None,
    ) -> dict:
        """Read supplement content for an arxiv paper."""
        import asyncio
        from modules.papers.short_ids import bare_arxiv_id

        bare_id = bare_arxiv_id(document_id)
        start_time = start_time or time.perf_counter()

        base_filename = filename
        if base_filename.endswith(".md.lines"):
            base_filename = base_filename[:-9]
        elif base_filename.endswith(".cheatsheet.md"):
            base_filename = base_filename[:-14]
        elif base_filename.endswith(".lines"):
            base_filename = base_filename[:-6]

        def _query():
            conn = self._get_arxiv_db_connection()
            q = """
                SELECT id, line_number, content
                FROM content_blocks
                WHERE document_id = %s
                AND (citation_info->>'supplement_filename' = %s
                     OR citation_info->>'source_path' LIKE %s)
            """
            params = [bare_id, base_filename, f"%{base_filename.replace('%', '%%')}%"]
            if start:
                q += " AND line_number >= %s"
                params.append(start)
            if end:
                q += " AND line_number <= %s"
                params.append(end)
            q += " ORDER BY line_number"
            with conn.cursor() as cur:
                cur.execute(q, params)
                return cur.fetchall()

        rows = await asyncio.to_thread(_query)
        if not rows:
            return {"error": f"Supplement not found: {filename}"}

        return {
            "document_id": document_id,
            "filename": filename,
            "lines": [{"line": r[1] + 1, "content": r[2]} for r in rows],
            "count": len(rows),
            "total_lines": len(rows),
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
        import asyncio
        start_time = time.perf_counter()
        parsed = self.path_parser.parse(path)

        if parsed.error:
            return {"error": parsed.error}

        if parsed.type in ("document", "file", "section"):
            doc_id = parsed.document_id

            from modules.papers.short_ids import is_arxiv_id, bare_arxiv_id
            if is_arxiv_id(doc_id):
                bare_id = bare_arxiv_id(doc_id)

                def _arxiv_stat():
                    conn = self._get_arxiv_db_connection()
                    with conn.cursor() as cur:
                        cur.execute(
                            """SELECT d.title, d.doi, d.source, d.authors, d.pub_date,
                                      COUNT(cb.line_number) as lines,
                                      COUNT(DISTINCT cb.section) as sections
                               FROM documents d
                               LEFT JOIN content_blocks cb ON d.document_id = cb.document_id
                               WHERE d.document_id = %s
                               GROUP BY d.document_id, d.title, d.doi, d.source, d.authors, d.pub_date""",
                            (bare_id,),
                        )
                        return cur.fetchone()

                row = await asyncio.to_thread(_arxiv_stat)
            elif doc_id.upper().startswith("PMC") and doc_id[3:].isdigit():
                def _pmc_stat():
                    conn = self._get_pmc_db_connection()
                    with conn.cursor() as cur:
                        cur.execute(
                            """SELECT d.title, d.doi, d.source, d.authors, d.pub_date,
                                      COUNT(cb.line_number) as lines,
                                      COUNT(DISTINCT cb.section) as sections
                               FROM documents d
                               LEFT JOIN content_blocks cb ON d.pmc_id = cb.pmc_id
                               WHERE d.pmc_id = %s
                               GROUP BY d.pmc_id, d.title, d.doi, d.source, d.authors, d.pub_date""",
                            (doc_id,),
                        )
                        return cur.fetchone()

                row = await asyncio.to_thread(_pmc_stat)
            else:
                conn = _get_db_connection()
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT d.title, d.doi, d.source, d.authors, d.month_year,
                                  COUNT(cb.line_number) as lines,
                                  COUNT(DISTINCT cb.section) as sections
                           FROM documents d
                           LEFT JOIN content_blocks cb ON d.document_id = cb.document_id
                           WHERE d.document_id::text = %s
                           GROUP BY d.document_id, d.title, d.doi, d.source, d.authors, d.month_year""",
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
                "month_year": str(row[4]) if row[4] else "",
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
        search_mode: str = "any",  # "any", "all", "phrase", "50%", "75%"
        since: str = None,  # e.g. "30d", "7d", "6m", "1y"
        category: str = None,  # e.g. "Neuroscience"
        journal: str = None,  # journal name (ILIKE, PMC)
        article_type: str = None,  # e.g. "research-article", "review-article"
        year: str = None,  # e.g. "2024"
        sort: str = None,  # "date" for recency
        depth: str = "shallow",  # "shallow" (title+abstract) | "deep" (full text)
        all_time: bool = False,  # True = bypass default 2024+ date filter
        ranking: str = "hybrid",  # "hybrid" (default) | "bm25" | "vector"
        document_ids: list[str] = None,  # scope to these IDs (for grep | search)
        limit: int = 25,
        session_id: str = "default",
        **_extra,  # absorb unknown kwargs gracefully
    ) -> dict:
        """Find papers matching criteria.

        Search modes:
        - "any" (default): Match any term (most results, broadest)
        - "50%": At least half the terms must match
        - "75%": At least 75% of terms must match
        - "all": All terms must match (strictest keyword search)
        - "phrase": Exact phrase match (words together in order)

        Ranking modes:
        - "bm25" (default): BM25 keyword ranking — fast, best for specific terms
        - "vector": Semantic vector search — best for conceptual/paraphrased queries
        - "hybrid": BM25 + vector fused via RRF — best for deep recall
        """
        start_time = time.perf_counter()

        # Build search query
        search_query = query or title or author
        filters = {}
        if source:
            # source can be a string "pmc" or a list ["pmc","biorxiv"]
            filters["source"] = source if isinstance(source, list) else [source]
        if date_range:
            filters["date_range"] = date_range
        if search_mode:
            filters["search_mode"] = search_mode
        if since:
            filters["since"] = since
        if category:
            filters["category"] = category
        if sort:
            filters["sort"] = sort
        if journal:
            filters["journal"] = journal
        if article_type:
            filters["article_type"] = article_type
        if year:
            filters["year"] = str(year)
        if all_time:
            filters["all_time"] = True
        if depth and depth != "shallow":
            filters["depth"] = depth
        if ranking:
            filters["ranking"] = ranking
        # Search
        results = await self.document_store.search_documents(
            query=search_query,
            filters=filters,
            limit=min(limit, 500),
            document_ids=document_ids,
        )

        # Shorten UUIDs for display (bio_xxxx / med_xxxx)
        shorten_results(results)

        # Save results
        results_id = self.results_registry.save(
            data={"papers": results, "query": search_query},
            session_id=session_id,
            prefix="s",
        )

        # Save table artifact to GCS in background (don't block the agent)
        asyncio.create_task(
            self._save_search_artifact(results_id, results, search_query, session_id)
        )

        return {
            "results_id": results_id,
            "query": search_query,
            "count": len(results),
            "papers": results,
            "time_ms": round((time.perf_counter() - start_time) * 1000),
            "search_meta": getattr(self.document_store, "last_search_meta", None),
        }

    _FILTER_BATCH_SIZE = 50

    async def _filter(
        self,
        from_results: str,
        query: str,
        session_id: str = "default",
    ) -> dict:
        """Filter search results for relevance using an LLM.

        Cumulative: tracks which papers have already been evaluated so that
        repeated filter calls (after additional searches) only send NEW papers
        to the LLM.  Previously-approved papers are preserved and merged with
        newly-approved ones.  Filter state is persisted in the results registry
        under ``{results_id}__fstate``.

        If the filter *query* changes between calls the state is reset and all
        papers are re-evaluated.
        """
        start_time = time.perf_counter()

        saved = self.results_registry.load(from_results, session_id)
        if not saved:
            return {"error": f"Results not found: {from_results}"}

        papers = saved.get("papers", [])
        if not papers:
            return {"error": "No papers in search results to filter"}

        # --- Load cumulative filter state ---
        fstate_id = f"{from_results}__fstate"
        fstate = self.results_registry.load(fstate_id, session_id) or {}

        prev_query = fstate.get("filter_query", "")
        evaluated_ids: set[str] = set(fstate.get("evaluated_ids", []))
        prev_approved: list[dict] = fstate.get("approved_papers", [])

        if query != prev_query:
            evaluated_ids = set()
            prev_approved = []

        # Separate papers into already-evaluated vs new
        new_papers: list[dict] = []
        for p in papers:
            doc_id = p.get("document_id", "")
            if doc_id and doc_id not in evaluated_ids:
                new_papers.append(p)

        pool_size = len(papers)

        if not new_papers:
            approved_count = len(prev_approved)
            self.results_registry.save(
                data={
                    "papers": prev_approved,
                    "query": saved.get("query", ""),
                    "queries": saved.get("queries", []),
                    "filtered_from": pool_size,
                },
                session_id=session_id,
                results_id=from_results,
            )
            asyncio.create_task(
                self._save_search_artifact(
                    from_results, prev_approved, saved.get("query", ""), session_id
                )
            )
            return {
                "results_id": from_results,
                "original_count": pool_size,
                "filtered_count": approved_count,
                "new_evaluated": 0,
                "newly_approved": 0,
                "previously_approved": approved_count,
                "time_ms": round((time.perf_counter() - start_time) * 1000),
            }

        # --- Filter only the new papers ---
        filter_model = self._get_reduce_model()
        base_url = get_inference_url()
        batch_size = self._FILTER_BATCH_SIZE

        batches = [
            new_papers[i : i + batch_size]
            for i in range(0, len(new_papers), batch_size)
        ]
        num_batches = len(batches)

        async def _run_filter_batch(
            batch_idx: int, batch: list[dict], client: "InferenceClient"
        ) -> set[int]:
            """Filter a single batch, returning global indices into new_papers."""
            global_offset = batch_idx * batch_size
            paper_descriptions = []
            for local_i, p in enumerate(batch):
                title = p.get("title", "Untitled")
                abstract = p.get(
                    "abstract",
                    p.get("abstract_text", p.get("abstract_snippet", "")),
                )
                paper_descriptions.append(f"[{local_i}] {title}\n    {abstract[:400]}")

            batch_text = "\n\n".join(paper_descriptions)
            prompt = (
                f"You are a relevance filter for scientific paper search results.\n\n"
                f"User's query: {query}\n\n"
                f"Below are papers returned by a keyword search. For each paper, decide whether it\n"
                f"is RELEVANT to the user's query. A paper is relevant if it genuinely addresses the\n"
                f"topic the user asked about — not just because it shares a keyword.\n\n"
                f"Watch out for keyword collisions: a paper may contain the search term but refer to\n"
                f"something entirely different (e.g., an acronym with multiple meanings, a method name\n"
                f"that matches a platform name, etc.).\n\n"
                f"Papers:\n{batch_text}\n\n"
                f"Respond with ONLY a JSON array of the integer indices of RELEVANT papers.\n"
                f"Example: [0, 2, 5, 7]\n\n"
                f"If none are relevant, respond with: []"
            )

            result = await client.chat(
                message_history=[{"role": "user", "content": prompt}],
                model=filter_model,
            )
            raw_output = (
                result.get("response", {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            indices: set[int] = set()
            match = re.search(r"\[[\d\s,]*\]", raw_output)
            if match:
                for idx in json.loads(match.group()):
                    if isinstance(idx, int) and 0 <= idx < len(batch):
                        indices.add(global_offset + idx)
            else:
                logger.warning(
                    f"Filter batch {batch_idx + 1}/{num_batches} returned "
                    f"unparseable output: {raw_output[:200]}"
                )
            return indices

        try:
            from gxl_inference_client.client import InferenceClient

            client = InferenceClient(base_url, timeout=60.0)

            try:
                batch_results = await asyncio.gather(
                    *[
                        _run_filter_batch(i, batch, client)
                        for i, batch in enumerate(batches)
                    ],
                    return_exceptions=True,
                )
            finally:
                await client.close()

            new_valid_indices: set[int] = set()
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.warning(
                        f"Filter batch {i + 1}/{num_batches} failed: {result}"
                    )
                else:
                    new_valid_indices |= result

            newly_approved = [new_papers[idx] for idx in sorted(new_valid_indices)]

        except Exception as e:
            logger.warning(f"Filter LLM call failed: {e}")
            return {
                "error": f"Filter failed: {e}",
                "results_id": from_results,
                "original_count": pool_size,
            }

        # --- Merge with previously approved papers ---
        prev_approved_ids = {p.get("document_id") for p in prev_approved}
        deduped_new = [
            p for p in newly_approved if p.get("document_id") not in prev_approved_ids
        ]
        all_approved = prev_approved + deduped_new

        # Update evaluated IDs
        evaluated_ids |= {p.get("document_id", "") for p in new_papers}

        # Persist filter state
        self.results_registry.save(
            data={
                "filter_query": query,
                "evaluated_ids": list(evaluated_ids),
                "approved_papers": all_approved,
            },
            session_id=session_id,
            results_id=fstate_id,
        )

        # Save approved papers as the main results
        self.results_registry.save(
            data={
                "papers": all_approved,
                "query": saved.get("query", ""),
                "queries": saved.get("queries", []),
                "filtered_from": pool_size,
            },
            session_id=session_id,
            results_id=from_results,
        )
        asyncio.create_task(
            self._save_search_artifact(
                from_results, all_approved, saved.get("query", ""), session_id
            )
        )

        return {
            "results_id": from_results,
            "original_count": pool_size,
            "filtered_count": len(all_approved),
            "new_evaluated": len(new_papers),
            "newly_approved": len(deduped_new),
            "previously_approved": len(prev_approved),
            "time_ms": round((time.perf_counter() - start_time) * 1000),
        }

    async def _lookup(
        self,
        field: str,
        value: str,
        limit: int = 25,
        session_id: str = "default",
        force_pmc: bool = False,
    ) -> dict:
        """Look up papers by a specific metadata field.

        Args:
            field: DB column name (supports all documents columns from both biomedrxiv and PMC)
            value: Value to search for (ILIKE partial match, or exact for pmc_id/pmid/pub_year)
            limit: Maximum results to return
            force_pmc: Route to PMC database (for PMC-only columns)
        """
        start_time = time.perf_counter()
        conn = _get_db_connection()

        # Validate field (allowlist prevents SQL injection)
        allowed_bio_fields = {
            "doi",
            "document_id",
            "authors",
            "title",
            "month_year",
            "source",
            "abstract_text",
        }
        pmc_id_fields = {"pmc", "pmc_id", "pmid"}
        pmc_text_fields = {
            "journal_title",
            "publisher_name",
            "article_type",
            "license_type",
            "volume",
            "issue",
            "issn",
        }
        pmc_jsonb_fields = {"keywords", "categories"}
        pmc_numeric_fields = {"pub_year"}

        all_pmc_fields = (
            pmc_id_fields | pmc_text_fields | pmc_jsonb_fields | pmc_numeric_fields
        )

        # PMC ID / PMID exact lookup
        if field in pmc_id_fields and "pmc" not in ENABLED_SOURCES:
            return {
                "total": 0,
                "results": [],
                "time_ms": round((time.perf_counter() - start_time) * 1000),
            }
        if field in pmc_id_fields:
            import asyncio

            def _pmc_id_lookup():
                pmc_conn = self._get_pmc_db_connection()
                with pmc_conn.cursor() as cur:
                    if field in ("pmc", "pmc_id"):
                        val = (
                            value if value.upper().startswith("PMC") else f"PMC{value}"
                        )
                        cur.execute(
                            """SELECT pmc_id, title, doi, authors, source,
                                      journal_title, article_type, pub_year
                               FROM documents WHERE pmc_id = %s LIMIT 1""",
                            (val,),
                        )
                    else:
                        cur.execute(
                            """SELECT pmc_id, title, doi, authors, source,
                                      journal_title, article_type, pub_year
                               FROM documents WHERE pmid = %s LIMIT 1""",
                            (value,),
                        )
                    return cur.fetchall()

            rows = await asyncio.to_thread(_pmc_id_lookup)
            results = [
                {
                    "document_id": r[0],
                    "pmc_id": r[0],
                    "title": r[1],
                    "doi": r[2],
                    "authors": r[3],
                    "source": r[4] or "pmc",
                    "journal": r[5] or "",
                    "article_type": r[6] or "",
                    "pub_year": str(r[7]) if r[7] else "",
                    "path": f"/papers/{r[0]}/",
                }
                for r in rows
            ]
            return {
                "total": len(results),
                "results": results,
                "time_ms": round((time.perf_counter() - start_time) * 1000),
            }

        # PMC text/jsonb/numeric fields → query PMC database
        if (field in all_pmc_fields or force_pmc) and "pmc" not in ENABLED_SOURCES:
            return {
                "total": 0,
                "results": [],
                "time_ms": round((time.perf_counter() - start_time) * 1000),
            }
        if field in all_pmc_fields or force_pmc:
            import asyncio

            def _pmc_field_lookup():
                pmc_conn = self._get_pmc_db_connection()
                with pmc_conn.cursor() as cur:
                    at_filter = PapersStore.PMC_ARTICLE_TYPE_SQL
                    if field in pmc_numeric_fields:
                        try:
                            cur.execute(
                                f"""SELECT pmc_id, title, doi, authors, source,
                                           journal_title, article_type, pub_year
                                    FROM documents WHERE {field} = %s AND {at_filter}
                                    ORDER BY created_at DESC LIMIT %s""",
                                (int(value), limit),
                            )
                        except ValueError:
                            return []
                    elif field in pmc_jsonb_fields:
                        cur.execute(
                            f"""SELECT pmc_id, title, doi, authors, source,
                                       journal_title, article_type, pub_year
                                FROM documents
                                WHERE {field}::text ILIKE %s AND {at_filter}
                                ORDER BY created_at DESC LIMIT %s""",
                            (f"%{value}%", limit),
                        )
                    else:
                        cur.execute(
                            f"""SELECT pmc_id, title, doi, authors, source,
                                       journal_title, article_type, pub_year
                                FROM documents
                                WHERE {field} ILIKE %s AND {at_filter}
                                ORDER BY pub_year DESC NULLS LAST, created_at DESC LIMIT %s""",
                            (f"%{value}%", limit),
                        )
                    return cur.fetchall()

            rows = await asyncio.to_thread(_pmc_field_lookup)
            results = [
                {
                    "document_id": r[0],
                    "pmc_id": r[0],
                    "title": r[1],
                    "doi": r[2] or "",
                    "authors": r[3] or "",
                    "source": r[4] or "pmc",
                    "journal": r[5] or "",
                    "article_type": r[6] or "",
                    "pub_year": str(r[7]) if r[7] else "",
                    "path": f"/papers/{r[0]}/",
                }
                for r in rows
            ]
            results_id = self.results_registry.save(
                data={"papers": results, "query": f"lookup {field} {value}"},
                session_id=session_id,
                prefix="s",
            )
            return {
                "total": len(results),
                "results": results,
                "results_id": results_id,
                "time_ms": round((time.perf_counter() - start_time) * 1000),
            }

        if field not in allowed_bio_fields:
            return {
                "error": (
                    f"Invalid field: '{field}'. "
                    f"bioRxiv fields: doi, arxiv, authors, title, month_year, source, abstract_text. "
                    f"PMC fields: pmc, pmid, journal, publisher, type, keywords, "
                    f"category, license, year, volume, issue, issn."
                )
            }

        # For DOI / document_id, use exact match only
        if field in ("doi", "document_id"):
            with conn.cursor() as cur:
                cur.execute(
                    f"""SELECT document_id::text, title, doi, authors, pub_date, source
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
                                "pub_date": str(row[4]) if row[4] else "",
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

        # Trigram indexes: authors/doi are on lower(col), title is on col directly.
        # Match query shape to index expression to ensure GIN trgm index is used.
        lower_trgm_fields = {"authors", "doi"}
        if field in lower_trgm_fields:
            where = f"lower({field}) LIKE lower(%s)"
        elif field == "title":
            where = f"{field} ILIKE %s"
        else:
            where = f"{field} ILIKE %s"
        val = f"%{value}%"

        with conn.cursor() as cur:
            # Skip COUNT(*) — it requires scanning all matching rows and is slow for common terms.
            # Fetch limit*2 results and report how many unique DOIs we found.
            cur.execute(
                f"""SELECT document_id::text, title, doi, authors, pub_date, source
                    FROM documents
                    WHERE {where}
                    ORDER BY created_at DESC
                    LIMIT %s""",
                (val, limit * 2),
            )
            rows = cur.fetchall()
            total = len(
                set(r[2] for r in rows if r[2])
            )  # dedupe count from fetched rows

        results = []
        for row in rows:
            results.append(
                {
                    "document_id": row[0],
                    "title": row[1],
                    "doi": row[2],
                    "authors": row[3],
                    "pub_date": str(row[4]) if row[4] else "",
                    "source": row[5],
                    "path": f"/papers/{row[0]}/",
                }
            )

        # Deduplicate by DOI, keeping most recent version
        results = PapersStore._deduplicate_by_doi(results)[:limit]
        shorten_results(results)

        results_id = self.results_registry.save(
            data={"papers": results, "query": f"lookup {field} {value}"},
            session_id=session_id,
            prefix="s",
        )
        return {
            "total": total,
            "results": results,
            "results_id": results_id,
            "field": field,
            "value": value,
            "time_ms": round((time.perf_counter() - start_time) * 1000),
        }

    # SQL query timeout (seconds)
    _SQL_TIMEOUT_SECONDS = 15
    _SQL_MAX_ROWS = 200

    # Forbidden SQL keywords (must never appear in a SELECT query)
    _SQL_FORBIDDEN_KEYWORDS = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "TRUNCATE",
        "CREATE",
        "GRANT",
        "REVOKE",
        "COPY",
        "EXECUTE",
        "DO",
        "CALL",
        "SET",
        "VACUUM",
        "CLUSTER",
        "REINDEX",
        "LOCK",
        "DISCARD",
        "RESET",
        "REASSIGN",
        "SECURITY",
        "COMMENT",
        "IMPORT",
        "LOAD",
        "REFRESH",
        "LISTEN",
        "NOTIFY",
        "PREPARE",
        "DEALLOCATE",
    ]

    _SQL_FORBIDDEN_PATTERN = re.compile(
        r"\b(" + "|".join(_SQL_FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE
    )

    # Unified schema reference
    DB_SCHEMA_HINT = """Unified documents table — queries run across all sources (bioRxiv, medRxiv, arXiv, PMC):

  documents
    id               TEXT PRIMARY KEY    -- paper identifier (bioRxiv UUID, PMC ID like 'PMC12345678', or arXiv arx_ prefix)
    title            TEXT                -- paper title
    doi              TEXT                -- Digital Object Identifier
    authors          TEXT                -- comma-separated author list
    source           TEXT                -- 'biorxiv', 'medrxiv', 'pmc', or 'arxiv'
    abstract_text    TEXT                -- paper abstract
    pub_date         TEXT                -- publication date (e.g. '2024-09' for bioRxiv, year for PMC)
    journal_title    TEXT                -- journal name (PMC only, NULL for preprints)
    article_type     TEXT                -- e.g. 'research-article', 'review-article' (PMC only)
    pmid             TEXT                -- PubMed ID (PMC only)
    keywords         JSONB               -- keyword array (PMC only)
    categories       JSONB               -- subject category array (PMC only)
    pub_year         INT                 -- publication year (PMC only)
    publisher_name   TEXT                -- publisher (PMC only)
    license_type     TEXT                -- license (PMC only)
    volume           TEXT                -- journal volume (PMC only)
    issue            TEXT                -- journal issue (PMC only)
    issn             TEXT                -- journal ISSN (PMC only)
    created_at       TIMESTAMP           -- when the record was indexed

Notes:
  - Use ILIKE for case-insensitive text matching
  - Only the documents table is available
  - keywords and categories are JSONB arrays (PMC papers)
  - Some columns are NULL for preprint sources (journal_title, pmid, etc.)"""

    PMC_DB_SCHEMA_HINT = DB_SCHEMA_HINT

    _SQL_BLOCKED_TABLES = re.compile(
        r"\b(content_blocks|figures|sections|api_keys|cli_events|users|"
        r"sessions|agents|messages|migrations|oauth_codes|device_codes|"
        r"pg_stat_activity|pg_roles|pg_auth_members|pg_database|"
        r"pg_class|pg_namespace|pg_proc|pg_shadow|pg_authid|"
        r"information_schema)\b",
        re.IGNORECASE,
    )

    # Column mapping: bioRxiv → unified
    _BIO_COLUMN_MAP = {
        "document_id": "id",
        "month_year": "pub_date",
    }
    # Column mapping: PMC → unified
    _PMC_COLUMN_MAP = {
        "pmc_id": "id",
        "pub_year": "pub_date",
    }

    def _validate_sql(self, query: str) -> tuple[str | None, str]:
        """Validate and normalize a SQL query. Returns (error, normalized_query)."""
        normalized = query.strip()
        while normalized.startswith("--"):
            if "\n" not in normalized:
                break
            normalized = normalized.split("\n", 1)[1].strip()
        while normalized.startswith("/*"):
            end = normalized.find("*/")
            if end == -1:
                return "Unterminated comment in query", ""
            normalized = normalized[end + 2 :].strip()

        if not normalized.upper().startswith("SELECT"):
            return "Only SELECT queries are allowed.", ""

        match = self._SQL_FORBIDDEN_PATTERN.search(normalized)
        if match:
            return f"Forbidden SQL keyword: {match.group(0).upper()}", ""

        # Block SELECT ... INTO (creates a table — bypasses write validators)
        stripped = re.sub(r"'[^']*'", "", normalized)
        stripped = re.sub(r'"[^"]*"', "", stripped)

        if re.search(r"\bINTO\s+(?!TEMP\b)\w+", stripped, re.IGNORECASE):
            return "SELECT INTO is not allowed (creates tables). Use plain SELECT.", ""

        blocked = self._SQL_BLOCKED_TABLES.search(stripped)
        if blocked:
            return (
                f"Only the `documents` table is available. `{blocked.group(0)}` is not accessible via SQL.",
                "",
            )

        if ";" in stripped.rstrip(";"):
            return (
                "Multiple SQL statements are not allowed. Send one SELECT at a time.",
                "",
            )

        normalized = normalized.rstrip().rstrip(";")

        return None, normalized

    def _rewrite_sql_for_bio(self, query: str) -> str:
        """Rewrite unified column names to bioRxiv-specific names."""
        q = query
        q = re.sub(r"\bid\b(?!\s*=\s*\d)", "document_id", q)
        q = re.sub(r"\bpub_date\b", "month_year", q)
        # These columns don't exist in bioRxiv — replace with NULL placeholders
        # handled by the executor; just leave them as-is and let errors be caught
        return q

    def _rewrite_sql_for_pmc(self, query: str) -> str:
        """Rewrite unified column names to PMC-specific names."""
        q = query
        q = re.sub(r"\bid\b(?!\s*=\s*\d)", "pmc_id", q)
        q = re.sub(r"\bpub_date\b", "pub_year::text", q)
        q = re.sub(r"\bmonth_year\b", "pub_year::text", q)
        q = re.sub(r"\bdocument_id\b", "pmc_id", q)
        return q

    def _run_sql_on_db(self, normalized: str, use_pmc: bool) -> tuple[list[str], list]:
        """Execute a validated SQL query on a specific database. Returns (columns, rows).

        Rewrites unified column names (id, pub_date) to DB-specific names before execution,
        then normalizes result column names back to the unified schema.

        Runs inside a READ ONLY transaction with a server-side LIMIT to
        prevent writes (e.g. SELECT INTO) and cap resource usage even if
        the client-side validator is bypassed.
        """
        if use_pmc:
            rewritten = self._rewrite_sql_for_pmc(normalized)
            conn = self._get_pmc_db_connection()
        else:
            rewritten = self._rewrite_sql_for_bio(normalized)
            conn = _get_db_connection()
        timeout_ms = self._SQL_TIMEOUT_SECONDS * 1000
        conn.autocommit = False
        try:
            with conn.cursor() as cur:
                cur.execute("BEGIN READ ONLY")
                cur.execute("SET LOCAL ROLE paperclip_query_ro")
                cur.execute(f"SET LOCAL statement_timeout = {timeout_ms}")
                cur.execute(f"SET LOCAL lock_timeout = '5s'")
                wrapped = f"SELECT * FROM ({rewritten}) _q LIMIT {self._SQL_MAX_ROWS}"
                cur.execute(wrapped)
                columns = (
                    [desc[0] for desc in cur.description] if cur.description else []
                )
                rows = cur.fetchall()
                col_map = self._PMC_COLUMN_MAP if use_pmc else self._BIO_COLUMN_MAP
                columns = [col_map.get(c, c) for c in columns]
                return columns, rows
        finally:
            try:
                conn.rollback()
            except Exception:
                pass
            conn.autocommit = True

    def _clean_sql_rows(self, columns: list[str], rows: list, label: str = "") -> list:
        """Convert non-serializable types and shorten UUIDs."""
        clean_rows = []
        for row in rows:
            clean_row = []
            for val in row:
                if isinstance(val, datetime):
                    clean_row.append(val.isoformat())
                elif isinstance(val, uuid.UUID):
                    clean_row.append(str(val))
                elif isinstance(val, (dict, list)):
                    clean_row.append(val)
                else:
                    clean_row.append(val)
            clean_rows.append(clean_row)

        doc_col = None
        src_col = None
        for i, col in enumerate(columns):
            if col == "document_id":
                doc_col = i
            elif col == "source":
                src_col = i
        if doc_col is not None:
            _uuid_re = re.compile(
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                re.IGNORECASE,
            )
            for row in clean_rows:
                val = row[doc_col]
                if isinstance(val, str) and _uuid_re.match(val):
                    src = row[src_col] if src_col is not None else None
                    row[doc_col] = shorten(val, src)

        return clean_rows

    async def _raw_sql(
        self,
        query: str,
        session_id: str = "default",
        source: str = "all",
    ) -> dict:
        """Execute a read-only SQL query against the papers database(s).

        Only SELECT statements are allowed. A statement timeout and row limit are enforced.

        Args:
            query: A SQL SELECT statement.
            session_id: Session ID.
            source: 'all' (both, default), 'biorxiv'/'medrxiv', or 'pmc'.

        Returns:
            dict with 'columns', 'rows', 'count', 'time_ms', 'source', or 'error'.
        """
        start_time = time.perf_counter()
        source_lower = source.lower()

        error, normalized = self._validate_sql(query)
        if error:
            return {"error": error}

        # Determine which DBs to query
        query_bio = source_lower in ("all", "biorxiv", "medrxiv")
        query_pmc = source_lower in ("all", "pmc") and "pmc" in ENABLED_SOURCES

        if source_lower == "all":
            return await self._raw_sql_unified(
                normalized, query_bio, query_pmc, start_time, session_id
            )

        # Single-source mode
        use_pmc = source_lower == "pmc"
        db_label = "PMC" if use_pmc else "bioRxiv"
        logger.info(f"[SQL {db_label}] Executing: {normalized[:200]}...")

        try:
            columns, rows = _execute_with_retry(
                lambda: self._run_sql_on_db(normalized, use_pmc)
            )

            truncated = len(rows) > self._SQL_MAX_ROWS
            rows = rows[: self._SQL_MAX_ROWS]
            clean_rows = self._clean_sql_rows(columns, rows, db_label)

            elapsed = round((time.perf_counter() - start_time) * 1000)
            logger.info(f"[SQL {db_label}] OK: {len(clean_rows)} rows, {elapsed}ms")

            result = {
                "columns": columns,
                "rows": clean_rows,
                "count": len(clean_rows),
                "source": "pmc" if use_pmc else "biorxiv",
                "time_ms": elapsed,
            }
            if truncated:
                result["truncated"] = True
                result["note"] = f"Results capped at {self._SQL_MAX_ROWS} rows."
            return result

        except Exception as e:
            elapsed = round((time.perf_counter() - start_time) * 1000)
            error_msg = str(e).strip()
            try:
                if use_pmc:
                    conn = self._get_pmc_db_connection()
                else:
                    conn = _get_db_connection()
                conn.rollback()
                with conn.cursor() as cur:
                    cur.execute(f"SET statement_timeout = {_DB_STATEMENT_TIMEOUT_MS}")
            except Exception:
                pass
            logger.warning(f"[SQL {db_label}] Error after {elapsed}ms: {error_msg}")
            return {"error": f"SQL error: {error_msg}", "time_ms": elapsed}

    async def _raw_sql_unified(
        self,
        normalized: str,
        query_bio: bool,
        query_pmc: bool,
        start_time: float,
        session_id: str,
    ) -> dict:
        """Execute SQL against both databases and merge results."""
        import asyncio

        bio_result = None
        pmc_result = None
        bio_error = None
        pmc_error = None

        def _run_bio():
            try:
                return self._run_sql_on_db(normalized, use_pmc=False)
            except Exception as e:
                return str(e)

        def _run_pmc():
            try:
                return self._run_sql_on_db(normalized, use_pmc=True)
            except Exception as e:
                return str(e)

        # Run both in parallel
        loop = asyncio.get_event_loop()
        futures = []
        if query_bio:
            futures.append(("bio", loop.run_in_executor(None, _run_bio)))
        if query_pmc:
            futures.append(("pmc", loop.run_in_executor(None, _run_pmc)))

        for label, fut in futures:
            result = await fut
            if isinstance(result, str):
                if label == "bio":
                    bio_error = result
                else:
                    pmc_error = result
            else:
                if label == "bio":
                    bio_result = result
                else:
                    pmc_result = result

        # If both failed, return error
        if bio_result is None and pmc_result is None:
            elapsed = round((time.perf_counter() - start_time) * 1000)
            errors = []
            if bio_error:
                errors.append(f"bioRxiv: {bio_error}")
            if pmc_error:
                errors.append(f"PMC: {pmc_error}")
            return {"error": " | ".join(errors), "time_ms": elapsed}

        # Merge results — columns must match for UNION-style merge
        bio_cols, bio_rows = bio_result if bio_result else ([], [])
        pmc_cols, pmc_rows = pmc_result if pmc_result else ([], [])

        if bio_cols and pmc_cols and bio_cols == pmc_cols:
            # Same columns — merge rows directly
            all_rows = list(bio_rows) + list(pmc_rows)
            all_rows = all_rows[: self._SQL_MAX_ROWS]
            clean_rows = self._clean_sql_rows(bio_cols, all_rows)
            elapsed = round((time.perf_counter() - start_time) * 1000)
            sources_used = []
            if bio_result:
                sources_used.append(f"bioRxiv ({len(bio_rows)} rows)")
            if pmc_result:
                sources_used.append(f"PMC ({len(pmc_rows)} rows)")
            return {
                "columns": bio_cols,
                "rows": clean_rows,
                "count": len(clean_rows),
                "source": "all",
                "sources_detail": " + ".join(sources_used),
                "time_ms": elapsed,
                "truncated": len(list(bio_rows) + list(pmc_rows)) > self._SQL_MAX_ROWS,
            }
        else:
            # Different columns — show results separately with headers
            sections = []
            all_rows_combined = []
            combined_cols = None

            if bio_cols and bio_rows:
                bio_cleaned = self._clean_sql_rows(
                    bio_cols, list(bio_rows)[: self._SQL_MAX_ROWS]
                )
                sections.append(
                    {
                        "label": "bioRxiv/medRxiv",
                        "columns": bio_cols,
                        "rows": bio_cleaned,
                        "count": len(bio_cleaned),
                    }
                )

            if pmc_cols and pmc_rows:
                pmc_cleaned = self._clean_sql_rows(
                    pmc_cols, list(pmc_rows)[: self._SQL_MAX_ROWS]
                )
                sections.append(
                    {
                        "label": "PMC",
                        "columns": pmc_cols,
                        "rows": pmc_cleaned,
                        "count": len(pmc_cleaned),
                    }
                )

            # If only one DB succeeded, return its results directly
            if len(sections) == 1:
                s = sections[0]
                elapsed = round((time.perf_counter() - start_time) * 1000)
                note = None
                if bio_error:
                    note = f"(PMC skipped: {bio_error[:100]})"
                elif pmc_error:
                    note = f"(bioRxiv skipped: {pmc_error[:100]})"
                result = {
                    "columns": s["columns"],
                    "rows": s["rows"],
                    "count": s["count"],
                    "source": "pmc" if not bio_result else "biorxiv",
                    "time_ms": elapsed,
                }
                if note:
                    result["note"] = note
                return result

            # Both returned different columns — return as sections
            elapsed = round((time.perf_counter() - start_time) * 1000)
            return {
                "sections": sections,
                "source": "all",
                "time_ms": elapsed,
            }

    # Max rows for export queries
    _EXPORT_MAX_ROWS = 1_000
    _EXPORT_TIMEOUT_SECONDS = 60

    async def _export_sql(
        self,
        query: str,
        description: str = "",
        session_id: str = "default",
    ) -> dict:
        """Execute a SQL query and export results as a CSV file + table artifact.

        Like _raw_sql but designed for large result sets (up to 1K rows).
        Saves results to:
          1. CSV file in session storage (for download)
          2. JSON table artifact (for the table visualization tab)

        Args:
            query: A SQL SELECT statement.
            description: Human-readable description of the export (e.g. "NIH-funded papers 2024").
            session_id: Session ID for file storage.

        Returns:
            dict with 'artifact_id', 'csv_path', 'count', 'columns', 'time_ms', or 'error'.
        """
        import csv
        import io

        start_time = time.perf_counter()

        # --- Reuse _raw_sql validation (guardrails 1-3) ---
        normalized = query.strip()
        while normalized.startswith("--"):
            # Strip a leading SQL line comment — only advance if there's a newline,
            # otherwise the input is not a SQL comment and we break to avoid an infinite loop.
            if "\n" not in normalized:
                break
            normalized = normalized.split("\n", 1)[1].strip()
        while normalized.startswith("/*"):
            end = normalized.find("*/")
            if end == -1:
                return {"error": "Unterminated comment in query"}
            normalized = normalized[end + 2 :].strip()

        if not normalized.upper().startswith("SELECT"):
            return {"error": "Only SELECT queries are allowed."}

        match = self._SQL_FORBIDDEN_PATTERN.search(normalized)
        if match:
            return {"error": f"Forbidden SQL keyword: {match.group(0).upper()}"}

        stripped = re.sub(r"'[^']*'", "", normalized)
        stripped = re.sub(r'"[^"]*"', "", stripped)
        if ";" in stripped.rstrip(";"):
            return {
                "error": "Multiple SQL statements are not allowed. Send one SELECT at a time."
            }

        # --- Enforce LIMIT cap for export (max 1K, but respect smaller user LIMITs) ---
        upper = normalized.upper()
        if "LIMIT" in upper:
            limit_match = re.search(r"LIMIT\s+(\d+)", normalized, re.IGNORECASE)
            if limit_match:
                user_limit = int(limit_match.group(1))
                capped = min(user_limit, self._EXPORT_MAX_ROWS)
                normalized = re.sub(
                    r"LIMIT\s+\d+",
                    f"LIMIT {capped}",
                    normalized,
                    flags=re.IGNORECASE,
                )
        else:
            normalized = normalized.rstrip().rstrip(";")
            normalized += f" LIMIT {self._EXPORT_MAX_ROWS}"

        logger.info(f"[SQL export] Executing: {normalized[:200]}...")

        # --- Execute inside READ ONLY transaction ---
        try:

            def _run_query():
                conn = _get_db_connection()
                timeout_ms = self._EXPORT_TIMEOUT_SECONDS * 1000
                conn.autocommit = False
                try:
                    with conn.cursor() as cur:
                        cur.execute("BEGIN READ ONLY")
                        cur.execute("SET LOCAL ROLE paperclip_query_ro")
                        cur.execute(f"SET LOCAL statement_timeout = {timeout_ms}")
                        cur.execute(f"SET LOCAL lock_timeout = '5s'")
                        cur.execute(normalized)
                        columns = (
                            [desc[0] for desc in cur.description]
                            if cur.description
                            else []
                        )
                        rows = cur.fetchall()
                        return columns, rows
                finally:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    conn.autocommit = True

            columns, rows = _execute_with_retry(_run_query)

            truncated = len(rows) >= self._EXPORT_MAX_ROWS
            # Hard cap: enforce row limit regardless of SQL LIMIT
            rows = rows[: self._EXPORT_MAX_ROWS]

            elapsed = round((time.perf_counter() - start_time) * 1000)
            logger.info(f"[SQL export] Got {len(rows)} rows in {elapsed}ms")

            if not rows:
                return {
                    "count": 0,
                    "columns": columns,
                    "time_ms": elapsed,
                    "message": "Query returned 0 rows. Nothing to export.",
                }

            # --- Convert rows for JSON serialization ---
            clean_rows = []
            for row in rows:
                clean_row = []
                for val in row:
                    if isinstance(val, datetime):
                        clean_row.append(val.isoformat())
                    elif isinstance(val, uuid.UUID):
                        clean_row.append(str(val))
                    elif isinstance(val, (dict, list)):
                        clean_row.append(json.dumps(val, default=str))
                    elif val is None:
                        clean_row.append("")
                    else:
                        clean_row.append(val)
                clean_rows.append(clean_row)

            # Shorten document_id UUIDs
            doc_col = None
            src_col = None
            for i, col in enumerate(columns):
                if col == "document_id":
                    doc_col = i
                elif col == "source":
                    src_col = i
            if doc_col is not None:
                _uuid_re = re.compile(
                    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                    re.IGNORECASE,
                )
                for row in clean_rows:
                    val = row[doc_col]
                    if isinstance(val, str) and _uuid_re.match(val):
                        src = row[src_col] if src_col is not None else None
                        row[doc_col] = shorten(val, src)

            # --- 1. Save CSV to session artifact store (GXLFileSystem) ---
            # Historically this called a ``_upload_artifact_to_engine`` method
            # that was never implemented, which caused every ``export`` to
            # raise ``AttributeError``. We now route through the same
            # ``GXLFileSystem`` machinery the JSON table artifact uses below,
            # so CSV + JSON land side-by-side under ``artifacts/`` and
            # respect ``SANDBOX_PROVIDER`` (local / GCS / GCR / E2B).
            artifact_id = f"a_{uuid.uuid4().hex[:8]}"
            csv_filename = f"{artifact_id}.csv"

            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(columns)
            writer.writerows(clean_rows)
            csv_content = csv_buffer.getvalue()

            self._save_raw_file_bg(
                f"artifacts/{csv_filename}", csv_content, session_id
            )

            logger.info(
                f"[SQL export] Queued CSV {csv_filename} ({len(clean_rows)} rows) "
                f"for session={session_id}"
            )

            # --- 2. Save JSON table artifact (for UI visualization) ---
            # Hard cap: ensure we never exceed _EXPORT_MAX_ROWS in the artifact
            if len(clean_rows) > self._EXPORT_MAX_ROWS:
                logger.warning(
                    f"[SQL export] Row count {len(clean_rows)} exceeds cap "
                    f"{self._EXPORT_MAX_ROWS}, truncating"
                )
                clean_rows = clean_rows[: self._EXPORT_MAX_ROWS]
                truncated = True

            row_dicts = []
            for row in clean_rows:
                row_dict = {}
                for i, col in enumerate(columns):
                    row_dict[col] = row[i] if i < len(row) else ""
                row_dicts.append(row_dict)

            artifact = {
                "artifact_id": artifact_id,
                "artifact_type": "reduce_table",  # Reuse existing table renderer
                "created_at": datetime.now().isoformat(),
                "source": {
                    "description": description or "SQL export",
                    "query": query.strip(),
                    "row_count": len(clean_rows),
                },
                "output": {
                    "columns": columns,
                    "rows": row_dicts,
                },
                "csv_file": csv_filename,
            }
            if truncated:
                artifact["source"]["truncated"] = True
                artifact["source"][
                    "note"
                ] = f"Results capped at {self._EXPORT_MAX_ROWS} rows."

            self._save_artifact_bg(artifact_id, artifact, session_id)

            result = {
                "artifact_id": artifact_id,
                "csv_path": f"artifacts/{csv_filename}",
                "count": len(clean_rows),
                "columns": columns,
                "truncated": truncated,
                "time_ms": elapsed,
            }
            if description:
                result["description"] = description
            return result

        except Exception as e:
            elapsed = round((time.perf_counter() - start_time) * 1000)
            error_msg = str(e).strip()
            try:
                conn = _get_db_connection()
                conn.rollback()
                with conn.cursor() as cur:
                    cur.execute(f"SET statement_timeout = {_DB_STATEMENT_TIMEOUT_MS}")
            except Exception:
                pass
            logger.warning(f"[SQL export] Error after {elapsed}ms: {error_msg}")
            return {
                "error": f"SQL error: {error_msg}",
                "time_ms": elapsed,
            }

    async def _grep(
        self,
        regex: str,
        path: str = "/papers/",
        from_results: str = None,
        top_k: int = None,
        limit: int = 50,
        session_id: str = "default",
        exhaustive: bool = False,
        source_filter: str = None,
        section_filter: str = None,
    ) -> dict:
        """Regex search on paper content."""
        start_time = time.perf_counter()

        parsed = self.path_parser.parse(path)
        document_ids = None
        if not section_filter:
            if parsed.type == "document_section":
                section_filter = parsed.filter
            elif parsed.type == "section" and parsed.section:
                section_filter = parsed.section

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
                    resolve(p.get("document_id"))
                    for p in papers
                    if p.get("document_id")
                ]

        elif parsed.document_id:
            document_ids = [resolve(parsed.document_id)]

        # Execute grep
        try:
            matches = await self.document_store.grep_content(
                regex=regex,
                document_ids=document_ids,
                section_filter=section_filter,
                limit=limit,
                exhaustive=exhaustive,
                source_filter=source_filter,
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

        # Save table artifact to GCS in background (don't block the agent)
        asyncio.create_task(
            self._save_search_artifact(
                results_id, papers, f"regex: {regex}", session_id
            )
        )

        return {
            "results_id": results_id,
            "regex": regex,
            "matched_docs": len(papers),
            "total_matches": len(matches),
            "papers": papers,
            "time_ms": round((time.perf_counter() - start_time) * 1000),
        }

    def _load_artifact_as_papers(
        self, artifact_id: str, session_id: str
    ) -> dict | None:
        """Load an export artifact (a_xxx) via inference engine and convert rows to papers format.

        This allows `map --from a_xxx` to work with export artifacts that
        contain a document_id column, bridging the gap between SQL exports
        and the search-results format that _parallel expects.
        """
        import httpx

        base_url = get_inference_url()
        file_path = f"artifacts/{artifact_id}.json"
        try:
            resp = httpx.get(
                f"{base_url}/api/sessions/{session_id}/files/{file_path}",
                timeout=15.0,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            artifact = json.loads(data.get("content", "{}"))
        except Exception:
            return None

        rows = artifact.get("output", {}).get("rows", [])
        if not rows:
            return None

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

    async def _parallel(
        self,
        from_results: str = None,
        tasks: list[dict] = None,
        limit: int = None,
        query: str = None,
        output_schema: dict = None,
        max_concurrent: int = 10000,
        batch_size: int = 20,
        session_id: str = "default",
        agent_id: str = None,
        **kwargs,
    ) -> dict:
        """Execute parallel paper exploration.

        Each paper gets a dedicated reader subagent with full tool access
        (grep, cat, head, etc.) that explores the paper independently.
        """
        start_time = time.perf_counter()

        # Resolve tasks
        resolved_tasks = []
        source_info = {}

        # source_papers_meta: full metadata from original search results,
        # used to populate deterministic table columns (authors, date, etc.)
        source_papers_meta = []

        if tasks:
            resolved_tasks = tasks
            source_info = {"mode": "explicit_tasks", "count": len(tasks)}

        elif from_results:
            saved = self.results_registry.load(from_results, session_id)

            # Fallback: if not in results_registry and looks like an export
            # artifact (a_xxx), load from the artifact JSON on disk.
            if not saved and from_results.startswith("a_"):
                saved = self._load_artifact_as_papers(from_results, session_id)

            if not saved:
                return {"error": f"Results not found: {from_results}"}

            papers = saved.get("papers", [])
            if limit:
                papers = papers[:limit]

            if not query:
                return {"error": "Must provide 'query' parameter"}

            source_papers_meta = papers

            resolved_tasks = [
                {
                    "path": p.get("path") or f"/papers/{p.get('document_id')}/",
                    "query": query,
                }
                for p in papers
            ]
            source_info = {
                "mode": "from_results",
                "ref_id": from_results,
                "total_papers": len(saved.get("papers", [])),
                "tasks_count": len(resolved_tasks),
            }

        else:
            return {"error": "Must provide 'tasks' or 'from_results'"}

        if not resolved_tasks:
            return {"error": "No tasks to execute"}

        MAX_MAP_TASKS = 100
        if len(resolved_tasks) > MAX_MAP_TASKS:
            logger.warning(
                f"[PARALLEL] Truncating map from {len(resolved_tasks)} to {MAX_MAP_TASKS} tasks"
            )
            resolved_tasks = resolved_tasks[:MAX_MAP_TASKS]
            if source_papers_meta:
                source_papers_meta = source_papers_meta[:MAX_MAP_TASKS]

        executor = ParallelExecutor(
            document_store=self.document_store,
            agent_config="papers/papers_reader_full_content",
            session_manager=self.session_manager,
        )

        def extract_doc_id(path: str) -> str | None:
            parsed = self.path_parser.parse(path)
            return parsed.document_id

        _MAP_TIMEOUT_SECONDS = 600.0
        try:
            results = await asyncio.wait_for(
                executor.execute(
                    tasks=resolved_tasks,
                    max_concurrent=max_concurrent,
                    batch_size=batch_size,
                    output_schema=output_schema,
                    session_id=session_id,
                    parent_agent_id=agent_id,
                    extract_document_id=extract_doc_id,
                ),
                timeout=_MAP_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"map: timeout after {_MAP_TIMEOUT_SECONDS:.0f}s — the map took too long to complete. Retry the query with a smaller --limit (e.g. --limit 30) or a narrower search set."
            )

        # Save results
        map_id = self.results_registry.save(
            data={"results": results},
            session_id=session_id,
            prefix="m",  # m for map
        )

        # Always save a table artifact so the UI can display per-paper results
        successful = [
            r for r in results if r.get("status") == "success" or "title" in r
        ]
        table_output = self._build_table(successful, source_papers=source_papers_meta)
        source_papers = [
            {"document_id": r.get("document_id", ""), "title": r.get("title", "")[:80]}
            for r in successful
            if r.get("document_id")
        ]
        table_artifact = {
            "artifact_id": map_id,
            "artifact_type": "reduce_table",
            "created_at": datetime.now().isoformat(),
            "source": {
                "source_id": map_id,
                "paper_count": len(source_papers),
                "papers": source_papers,
            },
            "output": table_output,
            "citations": table_output.get("citations", []),
        }
        self._save_artifact_bg(map_id, table_artifact, session_id)

        return {
            "map_id": map_id,
            "artifact_id": map_id,
            "source": source_info,
            "tasks_executed": len(results),
            "tasks_successful": sum(1 for r in results if r.get("status") == "success"),
            "results": results,
            "time_ms": round((time.perf_counter() - start_time) * 1000),
        }

    # =========================================================================
    # papers_reduce - LLM synthesis of map results
    # =========================================================================

    async def _reduce(
        self,
        from_map: str = None,
        from_parallel: str = None,  # Legacy alias for from_map
        from_results: str = None,
        question: str = None,
        max_items: int = 1000,
        session_id: str = "default",
        # Legacy params — ignored but accepted so old calls don't break
        strategy: str = None,
        columns: list[str] = None,
        fields: list[str] = None,
        **kwargs,
    ) -> dict:
        """Synthesize map results into a cohesive narrative using an LLM."""
        start_time = time.perf_counter()

        # Load source data (from_map preferred, from_parallel for legacy support)
        source_id = from_map or from_parallel or from_results
        if not source_id:
            return {"error": "Must provide from_map or from_results"}

        saved = self.results_registry.load(source_id, session_id)
        if not saved:
            return {"error": f"Results not found: {source_id}"}

        # Get results from parallel output or search results
        if "results" in saved:
            results = saved["results"]
        elif "papers" in saved:
            results = saved["papers"]
        else:
            return {"error": f"Invalid results format in {source_id}"}

        results = results[:max_items] if max_items else results
        successful = [
            r for r in results if r.get("status") == "success" or "title" in r
        ]

        if not successful:
            return {"error": "No successful results to reduce", "source": source_id}

        # Build context from per-paper outputs.
        # Use the FINAL response from each subagent's rollout — the last
        # "response" step is the definitive answer, stripped of reasoning.
        # Falls back to the stored "output" field if no rollout is available.
        # Keep each entry short: reduce synthesizes answers, doesn't re-read papers.
        MAX_CHARS_PER_PAPER = 500
        context_parts = []
        for i, r in enumerate(successful):
            title = r.get("title") or r.get("path", f"Item {i+1}")
            doc_id = r.get("document_id", "")

            # Prefer last "response" step in rollout (most recent subagent output)
            final_answer = None
            rollout = r.get("rollout") or []
            for step in reversed(rollout):
                if step.get("type") == "response" and step.get("content"):
                    final_answer = step["content"]
                    break

            if not final_answer:
                # Fall back to stored output field
                final_answer = (
                    r.get("output") or r.get("response") or r.get("abstract", "")
                )
            if isinstance(final_answer, dict):
                final_answer = json.dumps(final_answer)

            snippet = str(final_answer)[:MAX_CHARS_PER_PAPER]
            header = f"[{i+1}] {title[:80]}"
            if doc_id:
                header += f" (doc:{doc_id[:8]})"
            context_parts.append(f"{header}\n{snippet}")

        context = "\n\n".join(context_parts)
        task = question or "Summarize the key findings"

        citation_rules = """
CITATION RULES:
- Cite papers by document_id: {{"document_id": "ID"}}.
- When referencing a specific claim, include the line number: {{"document_id": "ID", "line": N}}.
- IMPORTANT: Always use DOUBLE braces: {{"document_id": "ID"}} and {{"document_id": "ID", "line": N}}.
  Single braces will NOT render. Double braces are required.
- Always include the paper title when citing a paper. Format:
  *"Title of Paper"* {{"document_id": "ID"}}. The title appears in the [N] ... line
  for each paper above."""

        prompt = f"""You are synthesizing findings from {len(successful)} papers.

Task: {task}

{context}

Write a concise synthesis (2-4 paragraphs) that directly addresses the task.
{citation_rules}"""

        # Cap total prompt at ~1M tokens (~4 chars/token) to stay within model limits
        max_prompt_chars = 4_000_000
        if len(prompt) > max_prompt_chars:
            prompt = prompt[:max_prompt_chars]

        # Call LLM for synthesis
        reduce_model = self._get_reduce_model()
        try:
            reducer_id = f"reducer_{uuid.uuid4().hex[:6]}"
            base_url = get_inference_url()

            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                reducer_payload = {
                    "agent_id": reducer_id,
                    "model": reduce_model,
                }
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
                base_url=base_url,
                timeout=90.0,
            )
            output = await agent.call_async(prompt, defer_persistence=True)
        except Exception as e:
            logger.warning(f"LLM synthesis failed (model={reduce_model}): {e}")
            # Fall back to concatenated per-paper outputs
            fallback_parts = []
            for i, r in enumerate(successful):
                title = r.get("title") or r.get("path", f"Item {i+1}")
                text = r.get("output") or r.get("abstract", "")
                if isinstance(text, dict):
                    text = json.dumps(text)
                if text:
                    fallback_parts.append(f"**{title}**: {str(text)[:500]}")
            if fallback_parts:
                output = (
                    f"*LLM synthesis unavailable — showing individual results from {len(successful)} papers:*\n\n"
                    + "\n\n".join(fallback_parts)
                )
            else:
                output = f"[Synthesis unavailable - {len(successful)} results]"

        

        # Save as artifact
        artifact_id = f"r_{uuid.uuid4().hex[:8]}"
        source_papers = [
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
                "paper_count": len(source_papers),
                "papers": source_papers,
            },
            "output": output,
            "citations": [],
        }
        self._save_artifact_bg(artifact_id, artifact, session_id)

        return {
            "artifact_id": artifact_id,
            "source": source_id,
            "items_processed": len(successful),
            "output": output,
            "time_ms": round((time.perf_counter() - start_time) * 1000),
        }

    def _get_reduce_model(self) -> str:
        """Load the model from the biomedrxiv_reader config for reduce operations.

        Raises if the config file or model field is missing.
        """
        import yaml

        config_path = (
            Path(os.environ.get("GXL_ROOT", "/workspaces/gxl"))
            / "agents"
            / "papers"
            / "papers_reader_full_content.yaml"
        )
        if not config_path.exists():
            raise FileNotFoundError(
                f"Reduce agent config not found at {config_path}. "
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
        return model

    def _build_table(
        self,
        results: list[dict],
        source_papers: list[dict] | None = None,
        columns: list[str] = None,
    ) -> dict:
        """Build a deterministic table from map results using paper metadata.

        Columns are fixed and code-driven — never inferred from LLM output:
          paper | authors | month_year | source | response

        Args:
            results: Per-paper result dicts from the parallel executor.
            source_papers: Original paper metadata from search results (keyed
                by document_id). Used to populate authors/month_year/source.
            columns: Ignored (kept for backward compat). Columns are always
                the fixed set above.
        """
        # Build a lookup from document_id → paper metadata for O(1) access
        meta_by_id: dict[str, dict] = {}
        if source_papers:
            for p in source_papers:
                did = p.get("document_id", "")
                if did:
                    meta_by_id[did] = p

        fixed_columns = ["authors", "month_year", "source", "response"]

        rows = []
        all_citations = []
        all_rollouts = []

        for i, r in enumerate(results):
            title = r.get("title") or r.get("path") or f"Paper #{i+1}"
            doc_id = r.get("document_id", "")
            meta = meta_by_id.get(doc_id, {})

            # Extract the LLM response text (plain text, not parsed as columns)
            raw_output = r.get("output")
            if raw_output is None or raw_output == "":
                if r.get("status") == "error":
                    response_text = f"[Error: {r.get('error', 'Unknown error')}]"
                else:
                    response_text = "[No output]"
            elif isinstance(raw_output, dict):
                # Structured output — flatten to readable text
                display = {k: v for k, v in raw_output.items() if not k.startswith("_")}
                response_text = (
                    json.dumps(display, indent=1) if display else str(raw_output)
                )
            elif isinstance(raw_output, str):
                # Try to extract _citations from JSON output before using as text
                try:
                    parsed = json.loads(raw_output)
                    if isinstance(parsed, dict):
                        paper_citations = parsed.pop("_citations", [])
                        if isinstance(paper_citations, list):
                            for cit in paper_citations:
                                if isinstance(cit, dict):
                                    all_citations.append(
                                        {
                                            "document_id": doc_id,
                                            "line_number": cit.get("line"),
                                            "content": cit.get("content", "")[:200],
                                            "field": cit.get("field", ""),
                                            "_paper_title": title[:60],
                                            "_row_idx": i,
                                        }
                                    )
                        display = {
                            k: v for k, v in parsed.items() if not k.startswith("_")
                        }
                        response_text = (
                            json.dumps(display, indent=1)
                            if display
                            else raw_output[:500]
                        )
                    else:
                        response_text = str(parsed)[:500]
                except (json.JSONDecodeError, ValueError):
                    response_text = raw_output[:500]
            else:
                response_text = str(raw_output)[:500]

            row = {
                "paper": title[:80],
                "document_id": doc_id,
                "authors": meta.get("authors", r.get("authors", "-")),
                "month_year": meta.get("month_year", r.get("month_year", "-")),
                "source": meta.get("source", r.get("source", "-")),
                "response": response_text,
            }
            rows.append(row)

            # Collect rollouts for UI display
            rollout = r.get("rollout", [])
            if rollout:
                all_rollouts.append(
                    {
                        "paper_idx": i,
                        "paper_title": title[:60],
                        "document_id": doc_id,
                        "time_ms": r.get("time_ms", 0),
                        "steps": rollout,
                    }
                )

        return {
            "columns": fixed_columns,
            "rows": rows,
            "citations": all_citations,
            "rollouts": all_rollouts,
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

    def _save_artifact_bg(self, artifact_id: str, artifact: dict, session_id: str):
        """Fire-and-forget wrapper — schedules _save_artifact without blocking."""

        async def _do():
            try:
                await self._save_artifact(artifact_id, artifact, session_id)
            except Exception as e:
                logger.warning(f"[bg] artifact save failed for {artifact_id}: {e}")

        asyncio.create_task(_do())

    async def _save_raw_file(self, file_path: str, content: str, session_id: str):
        """Save raw text content (e.g. CSV) via GXLFileSystem.

        Companion to :meth:`_save_artifact` for non-JSON payloads such as the
        CSV side of an ``export`` artifact. Uses the same provider machinery
        (``SANDBOX_PROVIDER`` env var) so local/GCS/GCR/E2B all work.
        """
        try:
            from gxl_filesystem import GXLFileSystem

            fs = GXLFileSystem(session_id=session_id)
            await fs.write_file(file_path, content)
            logger.info(f"Saved raw file {file_path} ({len(content)} bytes)")
        except Exception as e:
            logger.error(f"Failed to save raw file {file_path}: {e}")
            raise

    def _save_raw_file_bg(self, file_path: str, content: str, session_id: str):
        """Fire-and-forget wrapper — schedules _save_raw_file without blocking."""

        async def _do():
            try:
                await self._save_raw_file(file_path, content, session_id)
            except Exception as e:
                logger.warning(f"[bg] raw file save failed for {file_path}: {e}")

        asyncio.create_task(_do())

    async def _save_search_artifact(
        self, results_id: str, papers: list[dict], query: str, session_id: str
    ):
        """Save search results as a table artifact so they can be cited/viewed as tables."""
        try:
            columns = ["title", "authors", "doi", "month_year", "source", "document_id"]
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
    # search_and_filter - Combined parallel search + per-paper LLM filtering
    # =========================================================================

    def _get_filter_model(self) -> str:
        """Load model from papers_filter.yaml config."""
        import yaml

        config_path = (
            Path(os.environ.get("GXL_ROOT", "/workspaces/gxl"))
            / "agents"
            / "configs"
            / "papers"
            / "papers_filter.yaml"
        )
        if not config_path.exists():
            return "google/gemini-3-flash-preview"
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        return config.get("model", "google/gemini-3-flash-preview")

    async def _search_and_filter(
        self,
        queries: list[str],
        task: str,
        n: int = 25,
        session_id: str = "default",
        **kwargs,
    ) -> dict:
        """Combined parallel search + per-paper LLM relevance filtering.

        1. msearch all queries in parallel (each returns up to N papers)
        2. Deduplicate aggregated results by document_id
        3. Score each paper for relevance in parallel using a fast LLM
        4. Rank by score, return top N
        """
        import asyncio

        start_time = time.perf_counter()
        n = min(n, 200)

        # --- Step 1: Parallel msearch ---
        search_start = time.perf_counter()
        if hasattr(self, "document_store") and hasattr(
            self.document_store, "msearch_documents"
        ):
            msearch_results = await self.document_store.msearch_documents(
                queries=queries, search_mode="any", limit=n
            )
        else:
            msearch_results = []
            for q in queries:
                try:
                    result = await self._find(
                        query=q, search_mode="any", limit=n, session_id=session_id
                    )
                    papers = result.get("results", result.get("papers", []))
                    msearch_results.append(
                        {
                            "query": q,
                            "total": len(papers),
                            "papers": papers,
                            "error": None,
                        }
                    )
                except Exception as e:
                    msearch_results.append(
                        {"query": q, "total": 0, "papers": [], "error": str(e)}
                    )
        search_ms = round((time.perf_counter() - search_start) * 1000)

        # Aggregate & deduplicate
        seen_ids = set()
        all_papers = []
        per_query_counts = []
        for res in msearch_results:
            count = 0
            for p in res.get("papers", []):
                doc_id = p.get("document_id")
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_papers.append(p)
                    count += 1
            per_query_counts.append({"query": res["query"], "count": count})

        total_before_filter = len(all_papers)
        logger.info(
            f"[search_and_filter] {len(queries)} queries -> {total_before_filter} "
            f"unique papers in {search_ms}ms"
        )

        if not all_papers:
            return {
                "papers": [],
                "total_searched": 0,
                "total_after_filter": 0,
                "n_requested": n,
                "per_query_counts": per_query_counts,
                "search_ms": search_ms,
                "filter_ms": 0,
                "total_ms": round((time.perf_counter() - start_time) * 1000),
            }

        # --- Step 2: Parallel per-paper relevance scoring ---
        filter_start = time.perf_counter()
        filter_model = self._get_filter_model()

        from gxl_inference_client.client import InferenceClient

        sem = asyncio.Semaphore(10)
        system_msg = (
            "You are a paper relevance judge. Given a user's research task "
            "and a paper's title and abstract, determine if the paper is relevant.\n\n"
            'Respond with ONLY a JSON object: {"relevant": true/false, "score": N}\n\n'
            "where score is 1-10 (10=directly addresses the exact topic, "
            "7-9=highly relevant, 4-6=partially relevant, 1-3=marginal). "
            "If not relevant, score should be 0.\n\n"
            "Do NOT explain your reasoning. Output ONLY the JSON object."
        )

        async def score_paper(client: InferenceClient, paper: dict) -> dict:
            title = paper.get("title", "Untitled")
            abstract = paper.get(
                "abstract",
                paper.get("abstract_text", paper.get("abstract_snippet", "")),
            )
            prompt = (
                f"Task: {task}\n\n"
                f"Paper title: {title}\n"
                f"Abstract: {abstract[:600]}\n\n"
                f"Is this paper relevant to the task?"
            )
            async with sem:
                try:
                    result = await client.chat(
                        message_history=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": prompt},
                        ],
                        model=filter_model,
                        agent_id=f"sf_{uuid.uuid4().hex[:6]}",
                    )

                    text = ""
                    if "response" in result:
                        inner = result["response"]
                        if isinstance(inner, dict) and "choices" in inner:
                            text = (
                                inner.get("choices", [{}])[0]
                                .get("message", {})
                                .get("content", "")
                            )
                        elif isinstance(inner, str):
                            text = inner
                    elif "content" in result:
                        text = result["content"]

                    text = text.strip()
                    if text.startswith("```"):
                        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                    parsed = json.loads(text)
                    relevant = parsed.get("relevant", False)
                    score = int(parsed.get("score", 0))
                    return {**paper, "_relevant": relevant, "_score": score}

                except Exception as e:
                    logger.debug(f"Filter scoring failed for '{title[:50]}': {e}")
                    return {**paper, "_relevant": False, "_score": 0}

        async with InferenceClient(timeout=15.0) as client:
            scored_papers = await asyncio.gather(
                *[score_paper(client, p) for p in all_papers]
            )
        filter_ms = round((time.perf_counter() - filter_start) * 1000)

        # --- Step 3: Rank and pick top N ---
        relevant_papers = [p for p in scored_papers if p.get("_relevant")]
        relevant_papers.sort(key=lambda p: p.get("_score", 0), reverse=True)
        top_n = relevant_papers[:n]

        # Strip internal fields before returning
        clean_papers = []
        for p in top_n:
            score = p.pop("_score", 0)
            p.pop("_relevant", None)
            p["relevance_score"] = score
            clean_papers.append(p)

        total_ms = round((time.perf_counter() - start_time) * 1000)

        logger.info(
            f"[search_and_filter] {total_before_filter} -> {len(relevant_papers)} relevant "
            f"-> top {len(clean_papers)} returned in {total_ms}ms "
            f"(search={search_ms}ms, filter={filter_ms}ms)"
        )

        # Save results to registry + artifact
        results_id = _generate_id("s")
        self.results_registry.save(
            data={
                "papers": clean_papers,
                "query": task,
                "queries": queries,
            },
            session_id=session_id,
            results_id=results_id,
        )
        asyncio.create_task(
            self._save_search_artifact(
                results_id, clean_papers, "; ".join(queries), session_id
            )
        )

        return {
            "results_id": results_id,
            "papers": clean_papers,
            "total_searched": total_before_filter,
            "total_relevant": len(relevant_papers),
            "total_returned": len(clean_papers),
            "n_requested": n,
            "per_query_counts": per_query_counts,
            "search_ms": search_ms,
            "filter_ms": filter_ms,
            "total_ms": total_ms,
        }

    # =========================================================================
    # papers_ask_image - Vision model analysis (Gemini direct)
    # =========================================================================

    _ASK_IMAGE_MODEL = "gemini-3.1-flash-lite-preview"

    _ASK_IMAGE_FUNCTIONS = {
        "describe": (
            "Describe this scientific figure in detail. Include: what type of "
            "visualization it is, what the axes/labels represent, key data trends "
            "or patterns, and any notable findings."
        ),
        "extract-data": (
            "Extract all quantitative data from this figure. List every numerical "
            "value, measurement, statistic, p-value, sample size, or data point you "
            "can identify. Present the data in a structured format (tables where possible)."
        ),
        "ocr": (
            "Extract ALL text visible in this image. Include axis labels, legends, "
            "titles, annotations, watermarks, and any embedded text. Preserve layout "
            "where possible."
        ),
        "compare": (
            "This figure may contain multiple panels or sub-figures. Identify each "
            "panel, describe what it shows, and compare the findings across panels. "
            "Highlight similarities and differences."
        ),
        "methods": (
            "Based on this figure, identify the experimental methods, assays, or "
            "computational techniques that were likely used to generate this data. "
            "Be specific."
        ),
        "summarize": (
            "Provide a one-paragraph summary of the key finding shown in this "
            "figure, suitable for inclusion in a literature review."
        ),
    }

    def _get_gemini_vision_client(self):
        """Lazy-init a google-genai Client for vision calls."""
        if (
            not hasattr(self, "_gemini_vision_client")
            or self._gemini_vision_client is None
        ):
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                self._gemini_vision_client = genai.Client(api_key=api_key)
            else:
                self._gemini_vision_client = genai.Client(
                    vertexai=True,
                    project=os.environ.get("GCP_PROJECT", "gxl-prod"),
                    location=os.environ.get("GCP_REGION", "us-central1"),
                )
        return self._gemini_vision_client

    async def _download_figure_bytes(self, document_id: str, figure_id: str):
        """Resolve a figure to raw bytes + metadata.

        Handles both bioRxiv/medRxiv (DB lookup → GCS) and PMC (direct GCS).
        Returns (image_bytes, mime_type, caption, label, error).
        """
        import asyncio

        resolved_id = document_id
        is_pmc = bool(re.match(r"^PMC\d+$", resolved_id, re.IGNORECASE))

        caption = ""
        label = figure_id
        image_bytes = None
        mime_type = None

        if is_pmc:
            # PMC: image files live directly in GCS under pmc/articles/{id}/
            gcs_bucket = _get_gcs_bucket()
            gcs_path = f"pmc/articles/{resolved_id}/{figure_id}"
            blob = gcs_bucket.blob(gcs_path)
            exists = await asyncio.to_thread(blob.exists)
            if not exists:
                return None, None, None, None, f"Image not found: {figure_id}"
            image_bytes = await asyncio.to_thread(blob.download_as_bytes)
        else:
            # bioRxiv/medRxiv/arxiv: look up in content_blocks, download via GCS
            from modules.papers.tools import download_image_from_gcs
            from modules.papers.short_ids import is_arxiv_id, bare_arxiv_id

            if is_arxiv_id(document_id):
                conn = self._get_arxiv_db_connection()
                lookup_id = bare_arxiv_id(document_id)
            else:
                conn = _get_db_connection()
                lookup_id = document_id

            figure_id_base = figure_id
            for ext in (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".gif"):
                figure_id_base = figure_id_base.replace(ext, "")

            with conn.cursor() as cur:
                cur.execute(
                    """SELECT content, citation_info->>'source_path',
                              citation_info->>'xml_id', citation_info->>'graphic',
                              citation_info->>'image_uri'
                       FROM content_blocks
                       WHERE document_id = %s AND block_type = 'figure'
                       AND (citation_info->>'graphic' = %s
                            OR citation_info->>'xml_id' = %s
                            OR citation_info->>'xml_id' = %s
                            OR citation_info->>'graphic' ILIKE %s)
                       LIMIT 1""",
                    (
                        lookup_id,
                        figure_id,
                        figure_id,
                        figure_id_base,
                        f"%{figure_id_base}%",
                    ),
                )
                row = cur.fetchone()

            if not row:
                return (
                    None,
                    None,
                    None,
                    None,
                    (
                        f"Figure not found: {figure_id} in paper {document_id}. "
                        "Use 'ls figures/' to list available figures."
                    ),
                )

            caption = row[0] or ""
            source_path = row[1]
            xml_id = row[2] or figure_id_base
            graphic = row[3]
            image_uri = row[4] if len(row) > 4 else None

            # Arxiv: use image_uri directly (points to the actual image in GCS)
            if image_uri and image_uri.startswith("gs://"):
                img_bytes = await asyncio.to_thread(
                    download_image_from_gcs, image_uri
                )
                if img_bytes:
                    image_bytes = img_bytes
                    figure_id = graphic or figure_id

            if not image_bytes:
                if not source_path:
                    return None, None, None, None, "No source path for this figure"

                if not source_path.startswith("gs://"):
                    bucket = os.getenv("BIOMEDRXIV_GCS_BUCKET", "rxiv_dev")
                    source_path = f"gs://{bucket}/{source_path}"

                base_path = source_path.rsplit("/", 1)[0]

                for candidate in ([graphic] if graphic else []) + [
                    f"{xml_id}.jpg",
                    f"{xml_id}.jpeg",
                    f"{xml_id}.png",
                    f"{xml_id}.tif",
                    f"{xml_id}.tiff",
                ]:
                    if not candidate:
                        continue
                    img_bytes = await asyncio.to_thread(
                        download_image_from_gcs, f"{base_path}/{candidate}"
                    )
                    if img_bytes:
                        image_bytes = img_bytes
                        figure_id = candidate
                        break

            if not image_bytes:
                return None, None, caption, None, "Image file not found in GCS"

            label = xml_id or figure_id_base
            if label.startswith("figa"):
                label = "Appendix Figure " + label[4:]
            elif label.startswith("fig"):
                label = "Figure " + label[3:]
            elif label.startswith("alg"):
                label = "Algorithm " + label[3:]

        # Determine MIME type and convert TIF→PNG
        ext = ("." + figure_id.rsplit(".", 1)[-1]).lower() if "." in figure_id else ""
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
            ".svg": "image/svg+xml",
        }.get(ext, "image/png")

        if ext in (".tif", ".tiff"):
            try:
                import io

                from PIL import Image

                img = Image.open(io.BytesIO(image_bytes))
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")
                max_dim = 2048
                if max(img.size) > max_dim:
                    ratio = max_dim / max(img.size)
                    img = img.resize(
                        (int(img.size[0] * ratio), int(img.size[1] * ratio)),
                        Image.Resampling.LANCZOS,
                    )
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                image_bytes = buf.getvalue()
                mime_type = "image/png"
            except Exception as e:
                return None, None, caption, label, f"TIF conversion failed: {e}"

        return image_bytes, mime_type, caption, label, None

    async def _ask_image(
        self,
        document_id: str,
        figure_id: str,
        question: str = None,
        fn: str = None,
        session_id: str = "default",
        **kwargs,
    ) -> dict:
        """Analyze a figure/image using Gemini vision (direct).

        Supports free-form questions OR built-in functions (fn=).
        Works for both bioRxiv/medRxiv and PMC papers.
        Used by the MCP tool *and* by the REST endpoint.
        """
        import asyncio

        start_time = time.perf_counter()

        # Resolve prompt from fn or question
        if fn:
            fn = fn.strip().lower()
            if fn not in self._ASK_IMAGE_FUNCTIONS:
                return {
                    "error": f"Unknown function '{fn}'. "
                    f"Available: {', '.join(sorted(self._ASK_IMAGE_FUNCTIONS))}",
                }
            prompt = self._ASK_IMAGE_FUNCTIONS[fn]
            if question:
                prompt = f"{prompt}\n\nAdditional context: {question}"
        elif question:
            prompt = question
        else:
            return {"error": "'question' or 'fn' is required"}

        # Download and prepare image
        image_bytes, mime_type, caption, label, err = await self._download_figure_bytes(
            document_id, figure_id
        )
        if err:
            return {
                "error": err,
                "figure_id": figure_id,
                "time_ms": round((time.perf_counter() - start_time) * 1000),
            }

        # Prepend caption context if available
        if caption:
            full_prompt = f"Figure caption: {caption}\n\n{prompt}"
        else:
            full_prompt = prompt

        # Call Gemini vision
        try:
            from google.genai import types

            gemini = self._get_gemini_vision_client()
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

            logger.info(
                f"[ASK_IMAGE] {document_id}/{figure_id} → {self._ASK_IMAGE_MODEL} "
                f"({len(image_bytes)} bytes, fn={fn})"
            )

            response = await asyncio.to_thread(
                gemini.models.generate_content,
                model=self._ASK_IMAGE_MODEL,
                contents=[image_part, full_prompt],
            )
            analysis = response.text or ""
            logger.info(f"[ASK_IMAGE] Response: {len(analysis)} chars")
        except Exception as e:
            logger.error(f"[ASK_IMAGE] Gemini error: {e}")
            return {"error": f"Vision model error: {e}"}

        result = {
            "figure_id": figure_id,
            "label": label or figure_id,
            "caption": caption,
            "analysis": analysis,
            "time_ms": round((time.perf_counter() - start_time) * 1000),
            "citation_info": {
                "type": "image",
                "doc_id": document_id,
                "figure_id": figure_id,
                "label": label or figure_id,
                "caption": (caption[:200] if caption else None),
            },
        }
        if fn:
            result["fn"] = fn
        else:
            result["question"] = question
        return result

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

        source = doc_row[2] if doc_row else None
        result = {
            "document_id": shorten(document_id, source),
            "line_number": line_num,
            "content_preview": content[:100] + "..." if len(content) > 100 else content,
            "block_type": block_type,
            "section": section,
            "doc_title": doc_row[0] if doc_row else None,
            "doi": doc_row[1] if doc_row else None,
            "source": source,
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

    def _get_terminal(
        self,
        session_id: str,
        agent_id: str | None = None,
        paper_uuid: str | None = None,
    ) -> VirtualTerminal:
        """Get or create a terminal for a session (or per-agent within a session)."""
        key = f"{session_id}:{agent_id}" if agent_id else session_id
        if key not in self._terminals:
            terminal = PapersTerminal(filesystem_module=self)
            if paper_uuid:
                paper_path = f"/papers/{paper_uuid}/"
                terminal.cwd = paper_path
                terminal.env["PWD"] = paper_path
            self._terminals[key] = terminal
            logger.info(f"Created new terminal for {key} (cwd={terminal.cwd})")
        return self._terminals[key]

    async def _paperclip(
        self,
        command: str,
        session_id: str = "default",
        agent_id: str | None = None,
    ) -> dict:
        """Route a paperclip command string through the virtual terminal.

        Strips optional 'bash' prefix and forwards the command to _shell,
        which delegates to the PapersTerminal. The terminal handles all
        built-in commands (search, lookup, map, reduce, ask_image, grep,
        cat, ls, scan, funded-by, etc.) directly.
        """
        cmd = command.strip()
        if cmd.startswith("bash "):
            cmd = cmd[5:].strip()
            if (cmd.startswith("'") and cmd.endswith("'")) or (
                cmd.startswith('"') and cmd.endswith('"')
            ):
                cmd = cmd[1:-1]
        return await self._shell(
            command=cmd,
            session_id=session_id,
            agent_id=agent_id,
        )

    async def _shell(
        self,
        command: str,
        session_id: str = "default",
        agent_id: str | None = None,
        paper_uuid: str | None = None,
    ) -> dict:
        """Execute a shell command in the virtual filesystem."""
        start_time = time.perf_counter()

        # Get terminal for this session+agent (each subagent gets its own terminal)
        terminal = self._get_terminal(session_id, agent_id, paper_uuid)

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


# ============================================================================
# Core Server
# ============================================================================


def build_citation_routes(app, server: MCPServer):
    """Build papers citation HTTP endpoints."""
    import base64 as _base64

    from fastapi import HTTPException, Query, Request
    from fastapi.responses import HTMLResponse, Response
    from modules.papers.tools import (
        download_image_from_gcs,
        download_xml_from_gcs,
        render_xml_to_html,
    )

    # Get biomedrxiv module
    papers_module = None
    for mod in server.modules.values():
        if hasattr(mod, "get_name") and mod.get_name() == "papers":
            papers_module = mod
            break

    if not papers_module:
        logger.warning(
            "Papers module not found - citation endpoints will not be available"
        )
        return

    @app.get("/papers/search")
    async def search_papers(q: str, limit: int = 5, source: str = "biomedrxiv"):
        """Search papers by DOI, PMC ID, title, or author.

        source: "biomedrxiv" (default) | "pmc" | "all"
        For PMC source, searches the pmc database.
        """
        import re as _re

        import psycopg2

        results = []

        # PMC ID direct lookup
        pmc_id_match = _re.match(r"^PMC(\d+)$", q.strip(), _re.IGNORECASE)

        try:
            if (source in ("pmc", "all") or pmc_id_match) and "pmc" in ENABLED_SOURCES:
                # Use Elasticsearch for fast PMC search (pmc_documents index, 6.5M docs)
                try:
                    es = _get_es_client()
                    if es:
                        if pmc_id_match:
                            # Exact PMC ID via term query
                            es_resp = es.search(
                                index="pmc",
                                body={
                                    "query": {"term": {"pmc_id": q.strip().upper()}},
                                    "size": limit,
                                },
                            )
                        else:
                            es_resp = es.search(
                                index="pmc",
                                body={
                                    "query": {
                                        "bool": {
                                            "must": [
                                                {
                                                    "multi_match": {
                                                        "query": q,
                                                        "fields": [
                                                            "title^3",
                                                            "abstract^2",
                                                            "authors",
                                                        ],
                                                        "type": "best_fields",
                                                    }
                                                }
                                            ],
                                            "filter": [
                                                PapersModule.PMC_ARTICLE_TYPE_FILTER
                                            ],
                                        }
                                    },
                                    "size": limit,
                                },
                            )
                        for hit in es_resp["hits"]["hits"]:
                            src = hit["_source"]
                            pmc_id = src.get("pmc_id", hit["_id"])
                            results.append(
                                {
                                    "document_id": pmc_id,
                                    "pmc_id": pmc_id,
                                    "title": src.get("title", ""),
                                    "doi": src.get("doi", ""),
                                    "authors": src.get("authors", ""),
                                    "month_year": str(src.get("pub_year", "")),
                                    "source": "pmc",
                                }
                            )
                    elif papers_module:
                        # Fallback: DB lookup (slower, only for exact PMC ID)
                        if pmc_id_match:
                            pmc_conn = papers_module._get_pmc_db_connection()
                            with pmc_conn.cursor() as cur:
                                cur.execute(
                                    "SELECT pmc_id, title, doi, authors, pub_year FROM documents WHERE pmc_id = %s LIMIT %s",
                                    (q.strip(), limit),
                                )
                                for r in cur.fetchall():
                                    results.append(
                                        {
                                            "document_id": r[0],
                                            "pmc_id": r[0],
                                            "title": r[1] or "",
                                            "doi": r[2] or "",
                                            "authors": r[3] or "",
                                            "month_year": str(r[4]) if r[4] else "",
                                            "source": "pmc",
                                        }
                                    )
                except Exception as pmc_e:
                    logger.warning(f"PMC search failed: {pmc_e}")

            if source in ("biomedrxiv", "all") and not pmc_id_match:
                # Try Elasticsearch first (relevance-ranked), fall back to SQL
                es_done = False
                try:
                    es = _get_es_client()
                    if es:
                        # Check for exact DOI
                        doi_match = _re.match(r"^10\.\d{4,}/", q.strip())
                        if doi_match:
                            es_resp = es.search(
                                index=PREPRINTS_OS_INDEX,
                                body={
                                    "query": {"term": {"doi": q.strip()}},
                                    "size": limit,
                                },
                            )
                        else:
                            es_resp = es.search(
                                index=PREPRINTS_OS_INDEX,
                                body={
                                    "query": {
                                        "multi_match": {
                                            "query": q,
                                            "fields": [
                                                "title^3",
                                                "abstract_text^2",
                                                "authors",
                                            ],
                                            "type": "best_fields",
                                        }
                                    },
                                    "size": limit,
                                },
                            )
                        for hit in es_resp["hits"]["hits"]:
                            src = hit["_source"]
                            results.append(
                                {
                                    "document_id": str(
                                        src.get("document_id", hit["_id"])
                                    ),
                                    "title": src.get("title", ""),
                                    "doi": src.get("doi", ""),
                                    "authors": src.get("authors", ""),
                                    "month_year": src.get("month_year", ""),
                                    "source": src.get("source", ""),
                                }
                            )
                        es_done = True
                except Exception as es_e:
                    logger.warning(
                        f"ES search failed for widget, falling back to SQL: {es_e}"
                    )

                if not es_done:
                    conn = _get_db_connection()
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT document_id, title, doi, authors, month_year, source FROM documents WHERE LOWER(doi) = LOWER(%s) LIMIT %s",
                            (q, limit),
                        )
                        rows = cur.fetchall()
                        if not rows:
                            cur.execute(
                                """SELECT document_id, title, doi, authors, month_year, source
                                   FROM documents
                                   WHERE title ILIKE %s OR authors ILIKE %s
                                   ORDER BY title ILIKE %s DESC, month_year DESC NULLS LAST
                                   LIMIT %s""",
                                (f"%{q}%", f"%{q}%", f"%{q}%", limit),
                            )
                            rows = cur.fetchall()
                    for r in rows:
                        results.append(
                            {
                                "document_id": str(r[0]),
                                "title": r[1] or "",
                                "doi": r[2] or "",
                                "authors": r[3] or "",
                                "month_year": r[4] or "",
                                "source": r[5] or "",
                            }
                        )

            return {"results": shorten_results(results[:limit])}
        except Exception as e:
            logger.error(f"Paper search failed: {e}")
            return {"results": []}

    @app.get("/papers/lookup-doi")
    async def lookup_paper_by_doi(doi: str):
        """Fast exact DOI lookup across bioRxiv/medRxiv and PMC concurrently."""
        import asyncio

        results = []

        def _search_biomedrxiv():
            try:
                conn = _get_db_connection()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT document_id::text, title, doi, source, authors, month_year "
                        "FROM documents WHERE doi = %s LIMIT 1",
                        (doi,),
                    )
                    row = cur.fetchone()
                if row:
                    results.append(
                        shorten_result(
                            {
                                "document_id": row[0],
                                "title": row[1] or "",
                                "doi": row[2] or "",
                                "source": row[3] or "",
                                "authors": row[4] or "",
                                "month_year": row[5] or "",
                            }
                        )
                    )
            except Exception as e:
                logger.warning(f"bioRxiv DOI lookup failed: {e}")

        def _search_pmc():
            try:
                if not papers_module:
                    return
                pmc_conn = papers_module._get_pmc_db_connection()
                with pmc_conn.cursor() as cur:
                    cur.execute(
                        "SELECT pmc_id, title, doi, authors, pub_year "
                        "FROM documents WHERE doi = %s LIMIT 1",
                        (doi,),
                    )
                    row = cur.fetchone()
                if row:
                    results.append(
                        {
                            "document_id": row[0],
                            "pmc_id": row[0],
                            "title": row[1] or "",
                            "doi": row[2] or "",
                            "source": "pmc",
                            "authors": row[3] or "",
                            "month_year": str(row[4]) if row[4] else "",
                        }
                    )
            except Exception as e:
                logger.warning(f"PMC DOI lookup failed: {e}")

        await asyncio.gather(
            asyncio.to_thread(_search_biomedrxiv),
            asyncio.to_thread(_search_pmc),
        )

        return {"result": results[0] if results else None}

    @app.get("/papers/{document_id}/content")
    async def get_paper_content(
        document_id: str, max_chars: int = 100000
    ):  # ~25K tokens
        """Return full paper content (meta + content.lines) as JSON.

        Supports biomedrxiv short IDs, full UUIDs, and PMC IDs.
        """
        import re as _re

        import psycopg2

        document_id = resolve(document_id)
        is_pmc = bool(_re.match(r"^PMC\d+$", document_id, _re.IGNORECASE))
        try:
            if is_pmc and "pmc" not in ENABLED_SOURCES:
                raise HTTPException(status_code=404, detail="PMC source is not enabled")
            if is_pmc:
                pmc_conn = (
                    papers_module._get_pmc_db_connection() if papers_module else None
                )
                if not pmc_conn:
                    raise HTTPException(
                        status_code=503, detail="PMC database not configured"
                    )
                with pmc_conn.cursor() as cur:
                    cur.execute(
                        "SELECT pmc_id, title, doi, authors, pub_year, source, abstract_text FROM documents WHERE pmc_id = %s",
                        (document_id,),
                    )
                    row = cur.fetchone()
                    if not row:
                        pmc_conn.close()
                        raise HTTPException(
                            status_code=404, detail="PMC paper not found"
                        )
                    doc_id, title, doi, authors, month_year, source, abstract = row
                    # pub_year from the row above — use it directly for partition pruning
                    # Also warm the module's cache so subsequent calls are instant
                    if papers_module and month_year:
                        papers_module._pmc_pub_year_cache[doc_id.upper()] = int(
                            month_year
                        )
                    cur.execute(
                        """SELECT line_number, line_number, content FROM content_blocks
                           WHERE pmc_id = %s AND pub_year = %s
                           ORDER BY line_number""",
                        (document_id, month_year),
                    )
                    blocks = cur.fetchall()
            else:
                conn = _get_db_connection()
                with conn.cursor() as cur:
                    # Meta
                    cur.execute(
                        """SELECT document_id, title, doi, authors, month_year, source, abstract_text
                           FROM documents WHERE document_id = %s LIMIT 1""",
                        (document_id,),
                    )
                    row = cur.fetchone()
                    if not row:
                        raise HTTPException(status_code=404, detail="Paper not found")
                    doc_id, title, doi, authors, month_year, source, abstract = row

                    cur.execute(
                        """SELECT line_number, content
                           FROM content_blocks
                           WHERE document_id = %s
                           ORDER BY line_number""",
                        (document_id,),
                    )
                    blocks = cur.fetchall()

            import json as _json

            meta = {
                "document_id": str(doc_id) if is_pmc else shorten(str(doc_id), source),
                "pmc_id": str(doc_id) if is_pmc else None,
                "title": title or "",
                "doi": doi or "",
                "authors": authors or "",
                "month_year": str(month_year) if month_year else "",
                "source": source or ("pmc" if is_pmc else ""),
                "abstract": abstract or "",
            }

            lines = []
            total_chars = 0
            truncated = False
            for line_num, content in blocks:
                line = f"L{line_num}: {content}"
                if total_chars + len(line) > max_chars:
                    truncated = True
                    break
                lines.append(line)
                total_chars += len(line) + 1

            return {
                "meta": meta,
                "content_lines": "\n".join(lines),
                "total_lines": len(lines),
                "total_blocks": len(blocks),
                "truncated": truncated,
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"get_paper_content failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/citations/resolve")
    async def resolve_block_ids(request: dict):
        """Resolve block_ids (content_blocks.id) to full citation data.

        Accepts: { "block_ids": ["123", "456", ...] }
        Returns: { "results": { "123": { doc_id, content, line_number, ... }, ... } }
        """
        from fastapi import Request

        block_ids = request.get("block_ids", [])
        if not block_ids:
            return {"results": {}}

        try:
            results = {}

            # Route each block_id to the right DB:
            #   PMC block_ids:         letter strings decoding to ≥ 10^12 (e.g. "bilszdaeto")
            #   biomedrxiv block_ids:  7-letter strings (e.g. "aexlhvf") or plain integers
            pmc_requests: list[tuple[str, str, int]] = (
                []
            )  # (orig_bid, pmc_id, line_number)
            bio_id_map: dict[int, str] = {}  # numeric_id → orig_bid

            hex_block_ids: list[str] = (
                []
            )  # block_id varchar column values (e.g. "bf3d71d")

            for bid in block_ids:
                bid_str = str(bid).strip()
                if is_encoded_block_id(bid_str) and is_pmc_block_id(bid_str):
                    try:
                        pmc_id, line_num = decode_pmc_block_id(bid_str)
                        pmc_requests.append((bid_str, pmc_id, line_num))
                    except ValueError:
                        pass
                elif bid_str.isdigit():
                    bio_id_map[int(bid_str)] = bid_str
                elif is_encoded_block_id(bid_str):
                    try:
                        bio_id_map[decode_block_id(bid_str)] = bid_str
                    except ValueError:
                        pass
                else:
                    hex_block_ids.append(bid_str)

            # ── Biomedrxiv resolution ─────────────────────────────────────────
            if bio_id_map:
                conn = _get_db_connection()
                placeholders = ",".join(["%s"] * len(bio_id_map))
                with conn.cursor() as cur:
                    cur.execute(
                        f"""SELECT cb.id, cb.document_id::text, cb.line_number, cb.content,
                                   cb.block_type, cb.section, cb.citation_info,
                                   d.title, d.doi, d.source, d.authors, d.month_year
                            FROM content_blocks cb
                            JOIN documents d ON cb.document_id = d.document_id
                            WHERE cb.id IN ({placeholders})""",
                        list(bio_id_map.keys()),
                    )
                    for row in cur.fetchall():
                        (
                            cb_id,
                            doc_id,
                            line_number,
                            content,
                            block_type,
                            section,
                            ci,
                            title,
                            doi,
                            source,
                            authors,
                            month_year,
                        ) = row
                        ci = ci or {}
                        result_key = bio_id_map.get(cb_id, str(cb_id))
                        default_xml = (
                            f"{source}_xml"
                            if source in ("biorxiv", "medrxiv", "pmc")
                            else "biorxiv_xml"
                        )
                        results[result_key] = {
                            "doc_id": shorten(doc_id, source),
                            "content": content or "",
                            "line_number": line_number,
                            "block_type": block_type or "",
                            "section": section or "",
                            "doc_title": title or "",
                            "doi": doi or "",
                            "source": source or "",
                            "authors": authors or "",
                            "month_year": month_year or "",
                            "source_type": ci.get("source_type", default_xml),
                            "source_path": ci.get("source_path", ""),
                            "xml_id": ci.get("xml_id", ""),
                            "xpath": ci.get("xpath", ""),
                        }

            # ── Hex block_id resolution (varchar column) ────────────────────
            if hex_block_ids:
                conn = _get_db_connection()
                placeholders = ",".join(["%s"] * len(hex_block_ids))
                with conn.cursor() as cur:
                    cur.execute(
                        f"""SELECT cb.block_id, cb.document_id::text, cb.line_number, cb.content,
                                   cb.block_type, cb.section, cb.citation_info,
                                   d.title, d.doi, d.source, d.authors, d.month_year
                            FROM content_blocks cb
                            JOIN documents d ON cb.document_id = d.document_id
                            WHERE cb.block_id IN ({placeholders})""",
                        hex_block_ids,
                    )
                    for row in cur.fetchall():
                        (
                            block_id_val,
                            doc_id,
                            line_number,
                            content,
                            block_type,
                            section,
                            ci,
                            title,
                            doi,
                            source,
                            authors,
                            month_year,
                        ) = row
                        ci = ci or {}
                        default_xml = (
                            f"{source}_xml"
                            if source in ("biorxiv", "medrxiv", "pmc")
                            else "biorxiv_xml"
                        )
                        results[block_id_val] = {
                            "doc_id": shorten(doc_id, source),
                            "content": content or "",
                            "line_number": line_number,
                            "block_type": block_type or "",
                            "section": section or "",
                            "doc_title": title or "",
                            "doi": doi or "",
                            "source": source or "",
                            "authors": authors or "",
                            "month_year": month_year or "",
                            "source_type": ci.get("source_type", default_xml),
                            "source_path": ci.get("source_path", ""),
                            "xml_id": ci.get("xml_id", ""),
                            "xpath": ci.get("xpath", ""),
                        }

            # ── PMC resolution ────────────────────────────────────────────────
            if pmc_requests and papers_module:
                try:
                    pmc_conn = papers_module._get_pmc_db_connection()
                    with pmc_conn.cursor() as cur:
                        for orig_bid, pmc_id, line_num in pmc_requests:
                            pub_year = papers_module._get_pmc_pub_year(pmc_id)
                            if pub_year:
                                cur.execute(
                                    """SELECT cb.pmc_id, cb.line_number, cb.content,
                                              cb.block_type, cb.section, cb.citation_info,
                                              d.title, d.doi, d.source, d.authors, d.pub_year,
                                              d.journal_title
                                       FROM content_blocks cb
                                       JOIN documents d USING (pmc_id)
                                       WHERE cb.pmc_id = %s AND cb.line_number = %s AND cb.pub_year = %s""",
                                    (pmc_id, line_num, pub_year),
                                )
                            else:
                                cur.execute(
                                    """SELECT cb.pmc_id, cb.line_number, cb.content,
                                              cb.block_type, cb.section, cb.citation_info,
                                              d.title, d.doi, d.source, d.authors, d.pub_year,
                                              d.journal_title
                                       FROM content_blocks cb
                                       JOIN documents d USING (pmc_id)
                                       WHERE cb.pmc_id = %s AND cb.line_number = %s""",
                                    (pmc_id, line_num),
                                )
                            row = cur.fetchone()
                            if row:
                                ci = row[5] or {}
                                results[orig_bid] = {
                                    "doc_id": row[0],
                                    "content": row[2] or "",
                                    "line_number": row[1],
                                    "block_type": row[3] or "",
                                    "section": row[4] or "",
                                    "doc_title": row[6] or "",
                                    "doi": row[7] or "",
                                    "source": row[8] or "pmc",
                                    "authors": row[9] or "",
                                    "month_year": str(row[10]) if row[10] else "",
                                    "journal": row[11] or "",
                                    "source_type": ci.get("source_type", "pmc_xml"),
                                    "source_path": ci.get("source_path", ""),
                                    "xml_id": "",
                                    "xpath": "",
                                }
                except Exception as pmc_e:
                    logger.warning(f"PMC citation resolve failed: {pmc_e}")

            return {"results": results}
        except Exception as e:
            logger.error(f"Failed to resolve block_ids: {e}")
            return {"results": {}}

    @app.get("/citations/metadata/{doc_id}/{line_number}")
    async def get_citation_metadata(doc_id: str, line_number: int):
        """Get citation metadata for a specific line. Supports biomedrxiv and PMC."""
        import re as _re

        is_pmc = bool(_re.match(r"^PMC\d+$", doc_id, _re.IGNORECASE))
        try:
            if is_pmc and papers_module:
                pub_year = papers_module._get_pmc_pub_year(doc_id)
                pmc_conn = papers_module._get_pmc_db_connection()
                with pmc_conn.cursor() as cur:
                    if pub_year:
                        cur.execute(
                            """SELECT cb.pmc_id, cb.line_number, cb.content, cb.block_type,
                                      cb.section, d.title, d.doi, d.source, d.authors, d.pub_year,
                                      d.journal_title
                               FROM content_blocks cb JOIN documents d USING (pmc_id)
                               WHERE cb.pmc_id = %s AND cb.line_number = %s AND cb.pub_year = %s""",
                            (doc_id, line_number, pub_year),
                        )
                    else:
                        cur.execute(
                            """SELECT cb.pmc_id, cb.line_number, cb.content, cb.block_type,
                                      cb.section, d.title, d.doi, d.source, d.authors, d.pub_year,
                                      d.journal_title
                               FROM content_blocks cb JOIN documents d USING (pmc_id)
                               WHERE cb.pmc_id = %s AND cb.line_number = %s""",
                            (doc_id, line_number),
                        )
                    row = cur.fetchone()
                if not row:
                    raise HTTPException(
                        status_code=404,
                        detail=f"PMC line {doc_id}:{line_number} not found",
                    )
                return {
                    "document_id": row[0],
                    "line_number": row[1],
                    "content": row[2] or "",
                    "block_type": row[3] or "",
                    "section": row[4] or "",
                    "doc_title": row[5] or "",
                    "doi": row[6] or "",
                    "source": row[7] or "pmc",
                    "authors": row[8] or "",
                    "month_year": str(row[9]) if row[9] else "",
                    "journal": row[10] or "",
                }
            else:
                resolved_id = resolve(doc_id)
                result = await papers_module._get_citation(
                    document_id=resolved_id,
                    line_number=line_number + 1,
                )
                if "error" in result:
                    raise HTTPException(status_code=404, detail=result["error"])
                return result
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error fetching citation metadata: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/citations/document/{doc_id}")
    async def get_document_metadata(doc_id: str):
        """Get document-level metadata. Supports short IDs, UUIDs, and PMC IDs."""
        import re as _re

        resolved = resolve(doc_id)
        is_pmc = bool(_re.match(r"^PMC\d+$", resolved, _re.IGNORECASE))
        try:
            if is_pmc and papers_module:
                pmc_conn = papers_module._get_pmc_db_connection()
                with pmc_conn.cursor() as cur:
                    cur.execute(
                        "SELECT title, doi, source, authors, pub_year, journal_title FROM documents WHERE pmc_id = %s",
                        (resolved,),
                    )
                    row = cur.fetchone()
                if not row:
                    raise HTTPException(
                        status_code=404, detail=f"PMC document {doc_id} not found"
                    )
                return {
                    "document_id": resolved,
                    "pmc_id": resolved,
                    "doc_title": row[0],
                    "doi": row[1] or "",
                    "source": row[2] or "pmc",
                    "authors": row[3] or "",
                    "month_year": str(row[4]) if row[4] else "",
                    "journal": row[5] or "",
                }
            else:
                conn = _get_db_connection()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT document_id::text, title, doi, source, authors, month_year FROM documents WHERE document_id::text = %s",
                        (resolved,),
                    )
                    row = cur.fetchone()
                if not row:
                    raise HTTPException(
                        status_code=404, detail=f"Document {doc_id} not found"
                    )
                return {
                    "document_id": shorten(row[0], row[3]),
                    "doc_title": row[1],
                    "doi": row[2],
                    "source": row[3],
                    "authors": row[4],
                    "month_year": row[5],
                }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error fetching document metadata: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/images/{doc_id}/{filename:path}")
    async def get_figure_image(doc_id: str, filename: str, b: str = Query(None)):
        """Serve a Papers figure image.

        The ``b`` query parameter carries the GCS directory (base64url-encoded),
        allowing the path to be resolved without a database lookup.  When ``b``
        is absent (legacy callers such as SupplementTabContent), the GCS path
        is derived from the content_blocks table instead.
        """
        from modules.papers.short_ids import resolve as resolve_short_id

        doc_id = resolve_short_id(doc_id)

        gcs_path: str | None = None

        if b:
            try:
                gcs_dir = _base64.urlsafe_b64decode(b.encode()).decode()
            except Exception:
                raise HTTPException(
                    status_code=400, detail="Invalid GCS base parameter"
                )
            gcs_path = f"{gcs_dir.rstrip('/')}/{filename}"
        else:
            # Fallback: look up the source path via the DB (one query per unique image).
            try:
                conn = _get_db_connection()
                with conn.cursor() as cur:
                    # Try matching by graphic field first
                    cur.execute(
                        """SELECT citation_info->>'source_path'
                           FROM content_blocks
                           WHERE document_id = %s
                             AND block_type IN ('figure', 'table')
                             AND (
                                 citation_info->>'graphic' = %s
                                 OR citation_info->>'graphic' ILIKE %s
                             )
                           LIMIT 1""",
                        (doc_id, filename, f"%{filename.rsplit('.', 1)[0]}%"),
                    )
                    row = cur.fetchone()
                    if not row:
                        # Fallback: derive GCS dir from any content block's source_path
                        cur.execute(
                            """SELECT citation_info->>'source_path'
                               FROM content_blocks
                               WHERE document_id = %s
                                 AND citation_info->>'source_path' IS NOT NULL
                               LIMIT 1""",
                            (doc_id,),
                        )
                        row = cur.fetchone()
                if row and row[0]:
                    source_path = row[0]
                    if not source_path.startswith("gs://"):
                        bucket = os.getenv("BIOMEDRXIV_GCS_BUCKET", "rxiv_dev")
                        source_path = f"gs://{bucket}/{source_path}"
                    gcs_path = f"{source_path.rsplit('/', 1)[0]}/{filename}"
            except Exception as e:
                logger.warning(f"DB fallback for image {doc_id}/{filename} failed: {e}")

        if not gcs_path:
            raise HTTPException(
                status_code=404,
                detail=f"Image {filename} not found for document {doc_id}",
            )

        image_bytes = _image_cache.get(gcs_path)
        if image_bytes is None:
            image_bytes = download_image_from_gcs(gcs_path)
            if image_bytes:
                _image_cache[gcs_path] = image_bytes

        if not image_bytes:
            raise HTTPException(
                status_code=404,
                detail=f"Image {filename} not found",
            )

        suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        mime_type = "image/png"

        if suffix in {"tif", "tiff"}:
            try:
                import io

                from PIL import Image

                img = Image.open(io.BytesIO(image_bytes))
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
            except Exception as e:
                logger.error(f"TIFF conversion failed for {filename}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed converting figure image {filename}",
                )
        elif suffix in {"jpg", "jpeg"}:
            mime_type = "image/jpeg"
        elif suffix == "gif":
            mime_type = "image/gif"
        elif suffix == "webp":
            mime_type = "image/webp"

        return Response(
            content=image_bytes,
            media_type=mime_type,
            headers={"Cache-Control": "public, max-age=3600"},
        )

    # --- ArXiv PDF page rendering (for citations.gxl.ai) -----------------------

    _arxiv_pdf_cache: dict[str, bytes] = {}
    _ARXIV_PDF_CACHE_MAX = 20

    @app.get("/page/{doc_id}/{page_num}")
    async def get_arxiv_pdf_page(
        doc_id: str,
        page_num: int,
        scale: float = Query(1.5, ge=0.5, le=4.0),
    ):
        """Render a single PDF page as base64 PNG for an arXiv paper.

        Returns JSON with image_base64, width, height (original PDF points),
        and the rendered pixel dimensions.
        """
        import asyncio

        import fitz

        from modules.papers.short_ids import bare_arxiv_id, is_arxiv_id
        from modules.papers.short_ids import resolve as resolve_short_id

        resolved = resolve_short_id(doc_id)
        if not is_arxiv_id(doc_id) and not is_arxiv_id(resolved):
            raise HTTPException(
                status_code=400, detail="PDF page view is only supported for arXiv papers"
            )

        bare_id = bare_arxiv_id(doc_id)
        storage_key = bare_id.replace("/", "") + ".pdf"
        gcs_path = f"gs://gxl-collections/arxiv_extracted/{storage_key}"

        pdf_bytes = _arxiv_pdf_cache.get(gcs_path)
        if pdf_bytes is None:
            pdf_bytes = await asyncio.to_thread(download_image_from_gcs, gcs_path)
            if not pdf_bytes:
                raise HTTPException(
                    status_code=404, detail=f"PDF not found for {doc_id}"
                )
            if len(_arxiv_pdf_cache) >= _ARXIV_PDF_CACHE_MAX:
                oldest = next(iter(_arxiv_pdf_cache))
                del _arxiv_pdf_cache[oldest]
            _arxiv_pdf_cache[gcs_path] = pdf_bytes

        def _render_page(raw_bytes: bytes, page_number: int, render_scale: float):
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
            total = len(doc)
            if page_number < 1 or page_number > total:
                doc.close()
                return None, f"Page {page_number} out of range (1-{total})"
            page = doc[page_number - 1]
            rect = page.rect
            mat = fitz.Matrix(render_scale, render_scale)
            pix = page.get_pixmap(matrix=mat)
            png_data = pix.tobytes("png")
            w, h = rect.width, rect.height
            pw, ph = pix.width, pix.height
            doc.close()
            return {
                "png": png_data,
                "width": w,
                "height": h,
                "pixel_width": pw,
                "pixel_height": ph,
                "total_pages": total,
            }, None

        try:
            result, err = await asyncio.to_thread(
                _render_page, pdf_bytes, page_num, scale
            )
            if err:
                raise HTTPException(status_code=400, detail=err)

            import base64 as _b64

            return {
                "image_base64": _b64.standard_b64encode(result["png"]).decode("utf-8"),
                "page": page_num,
                "total_pages": result["total_pages"],
                "width": result["width"],
                "height": result["height"],
                "pixel_width": result["pixel_width"],
                "pixel_height": result["pixel_height"],
                "scale": scale,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error rendering page: {str(e)}"
            )

    @app.get("/blocks/{doc_id}")
    async def get_block_info(
        doc_id: str,
        lines: str = Query(..., description="Comma-separated line numbers (1-indexed)"),
    ):
        """Return page and bbox info for specific content_block lines of an arXiv paper.

        Line numbers are 1-indexed (as displayed in the virtual filesystem).
        Returns blocks with page, bbox (pixel coordinates), and content.
        Also returns page_dimensions (width/height in the bbox coordinate system).
        """
        import asyncio

        from modules.papers.short_ids import bare_arxiv_id, is_arxiv_id
        from modules.papers.short_ids import resolve as resolve_short_id

        resolved = resolve_short_id(doc_id)
        if not is_arxiv_id(doc_id) and not is_arxiv_id(resolved):
            raise HTTPException(
                status_code=400, detail="Block info is only supported for arXiv papers"
            )

        try:
            line_nums = [int(x.strip()) for x in lines.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid line numbers")

        if not line_nums:
            return {"blocks": []}

        bare_id = bare_arxiv_id(doc_id)
        db_lines = [n - 1 for n in line_nums]

        def _query():
            conn = papers_module._get_arxiv_db_connection()
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT line_number, content, block_type, section, citation_info
                       FROM content_blocks
                       WHERE document_id = %s AND line_number = ANY(%s)
                       ORDER BY line_number""",
                    (bare_id, db_lines),
                )
                return cur.fetchall()

        rows = await asyncio.to_thread(_query)

        # Compute page dimensions in the bbox coordinate system.
        # Datalab processes PDFs at ~191 DPI internally (72 DPI * scale_factor).
        # We derive this from the cached PDF page rect rather than fetching
        # the Datalab JSON from GCS.
        page_dims = None
        try:
            import fitz as _fitz

            stem = bare_id.replace("/", "")
            gcs_path = f"gs://gxl-collections/arxiv_extracted/{stem}.pdf"
            pdf_bytes = _arxiv_pdf_cache.get(gcs_path)
            if pdf_bytes is None:
                pdf_bytes = await asyncio.to_thread(download_image_from_gcs, gcs_path)
                if pdf_bytes:
                    if len(_arxiv_pdf_cache) >= _ARXIV_PDF_CACHE_MAX:
                        oldest = next(iter(_arxiv_pdf_cache))
                        del _arxiv_pdf_cache[oldest]
                    _arxiv_pdf_cache[gcs_path] = pdf_bytes

            if pdf_bytes:
                def _get_dims(raw: bytes):
                    doc = _fitz.open(stream=raw, filetype="pdf")
                    rect = doc[0].rect
                    doc.close()
                    # Datalab's scale factor: 1624px / 612pt ≈ 2.6536
                    # (measured empirically from Datalab output)
                    scale = 1624.0 / 612.0
                    return {
                        "width": round(rect.width * scale, 1),
                        "height": round(rect.height * scale, 1),
                    }

                page_dims = await asyncio.to_thread(_get_dims, pdf_bytes)
        except Exception:
            pass

        blocks = []
        for row in rows:
            line_num, content, block_type, section, citation_info = row
            block = {
                "line_number": line_num + 1,
                "content": content,
                "block_type": block_type,
                "section": section,
            }
            if citation_info:
                page = citation_info.get("page")
                bbox = citation_info.get("bbox")
                if page:
                    block["page"] = page
                if bbox and bbox != [0, 0, 1, 1]:
                    block["bbox"] = bbox
            blocks.append(block)

        return {"doc_id": doc_id, "blocks": blocks, "page_dimensions": page_dims}

    # --- End ArXiv PDF page rendering -------------------------------------------

    @app.get("/citations/render/{doc_id}/{line_number}")
    async def render_citation(
        doc_id: str,
        line_number: int,
        context: int = Query(10, description="Lines of context before and after"),
        content: str = Query(None),
    ):
        """Render source context from content_blocks as HTML with the cited line highlighted.

        Returns surrounding content blocks as formatted HTML. Works for all sources
        (biorxiv, medrxiv, PMC) — pulls from the database, not XML files.

        line_number is 1-indexed (as shown in the virtual filesystem) while the
        database content_blocks.line_number is 0-indexed.
        """
        import html as _html
        import re as _re

        try:
            resolved_id = resolve(doc_id)
            is_pmc = bool(_re.match(r"^PMC\d+$", resolved_id, _re.IGNORECASE))

            db_line = line_number - 1
            start_line = max(0, db_line - context)
            end_line = db_line + context

            if is_pmc and papers_module:
                pub_year = papers_module._get_pmc_pub_year(resolved_id)
                pmc_conn = papers_module._get_pmc_db_connection()
                with pmc_conn.cursor() as cur:
                    q = """SELECT line_number, content, section, block_type
                           FROM content_blocks
                           WHERE pmc_id = %s AND line_number BETWEEN %s AND %s"""
                    params = [resolved_id, start_line, end_line]
                    if pub_year:
                        q += " AND pub_year = %s"
                        params.append(pub_year)
                    q += " ORDER BY line_number"
                    cur.execute(q, params)
                    rows = cur.fetchall()
            else:
                conn = _get_db_connection()
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT line_number, content, section, block_type
                           FROM content_blocks
                           WHERE document_id::text = %s AND line_number BETWEEN %s AND %s
                           ORDER BY line_number""",
                        (resolved_id, start_line, end_line),
                    )
                    rows = cur.fetchall()

            if not rows:
                raise HTTPException(
                    status_code=404,
                    detail=f"No content found for {doc_id} around line {line_number}",
                )

            # Build HTML with the cited line highlighted
            parts = [
                "<!DOCTYPE html><html><head><style>",
                'body { margin: 0; padding: 16px; font-family: "Source Serif 4", Georgia, serif; font-size: 13px; line-height: 1.8; color: #4b5563; }',
                ".section-header { font-size: 11px; font-weight: 600; color: #374151; text-transform: uppercase; letter-spacing: 0.05em; margin: 16px 0 6px; padding-bottom: 3px; border-bottom: 1px solid #e5e7eb; font-family: system-ui, sans-serif; }",
                ".line { margin: 4px 0; }",
                ".line-highlighted { background: #fef08a; color: #111827; padding: 2px 4px; border-radius: 2px; }",
                ".line-number { color: #d1d5db; font-size: 10px; font-family: monospace; user-select: none; margin-right: 8px; }",
                "</style></head><body>",
            ]

            current_section = None
            for ln, text, section, block_type in rows:
                if section and section != current_section:
                    current_section = section
                    parts.append(
                        f'<div class="section-header">{_html.escape(section)}</div>'
                    )

                display_ln = ln + 1
                escaped = _html.escape(text or "")
                if ln == db_line:
                    parts.append(
                        f'<div class="line line-highlighted" id="highlight-target"><span class="line-number">L{display_ln}</span>{escaped}</div>'
                    )
                else:
                    parts.append(
                        f'<div class="line"><span class="line-number">L{display_ln}</span>{escaped}</div>'
                    )

            parts.append("</body></html>")
            return HTMLResponse(content="\n".join(parts))

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error rendering citation context: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/supplements/{document_id}/{filename}/raw")
    async def get_supplement_raw(document_id: str, filename: str):
        """Serve raw supplement file (PDF, image) from GCS for embedded viewing."""
        import mimetypes

        from fastapi.responses import Response

        resolved_id = resolve(document_id)
        if not papers_module:
            raise HTTPException(status_code=503, detail="Papers module not available")

        try:
            conn = _get_db_connection()
            is_pmc = bool(re.match(r"^PMC\d+$", resolved_id, re.IGNORECASE))

            if is_pmc:
                gcs_path = f"pmc/articles/{resolved_id}/{filename}"
            else:
                month = papers_module._get_month_for_doc(resolved_id, conn)
                source_prefix = (
                    "biorxiv"
                    if papers_module._is_biorxiv(resolved_id, conn)
                    else "medrxiv"
                )
                gcs_path = f"{source_prefix}_extracted/{month}/{resolved_id}/content/supplements/{filename}"

            gcs_bucket = _get_gcs_bucket()
            blob = gcs_bucket.blob(gcs_path)

            if not blob.exists():
                raise HTTPException(
                    status_code=404, detail=f"Supplement file not found: {filename}"
                )

            data = blob.download_as_bytes()
            content_type = (
                mimetypes.guess_type(filename)[0] or "application/octet-stream"
            )

            return Response(
                content=data,
                media_type=content_type,
                headers={"Content-Disposition": f'inline; filename="{filename}"'},
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error serving supplement {filename}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/papers/{document_id}/file/{filename:path}")
    async def get_paper_file(document_id: str, filename: str):
        """Serve a single file from a paper directly from GCS."""
        import mimetypes

        from fastapi.responses import Response

        resolved_id = resolve(document_id)
        is_pmc = bool(re.match(r"^PMC\d+$", resolved_id, re.IGNORECASE))

        try:
            gcs_bucket = _get_gcs_bucket()
            if is_pmc:
                gcs_path = f"pmc/articles/{resolved_id}/{filename}"
            else:
                conn = _get_db_connection()
                month = papers_module._get_month_for_doc(resolved_id, conn)
                source_prefix = (
                    "biorxiv"
                    if papers_module._is_biorxiv(resolved_id, conn)
                    else "medrxiv"
                )
                gcs_path = f"{source_prefix}_extracted/{month}/{resolved_id}/content/{filename}"

            blob = gcs_bucket.blob(gcs_path)
            if not blob.exists():
                raise HTTPException(
                    status_code=404, detail=f"File not found: {filename}"
                )

            data = blob.download_as_bytes()
            content_type = (
                mimetypes.guess_type(filename)[0] or "application/octet-stream"
            )
            return Response(
                content=data,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=\"{filename.split('/')[-1]}\"",
                    "Content-Length": str(len(data)),
                },
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error serving file {document_id}/{filename}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ── REST ask-image endpoints (delegate to module._ask_image) ──

    def _authenticate_ask_image(request: Request, document_id: str):
        from shared.core.auth import validate_api_key

        api_key = request.headers.get("X-API-Key") or request.headers.get("x-api-key")
        if not validate_api_key(api_key, f"ask-image:{document_id}"):
            raise HTTPException(status_code=403, detail="Invalid API key")

    @app.post("/papers/{document_id}/ask-image")
    async def ask_image_endpoint(document_id: str, request: Request):
        """Analyze a figure/image via Gemini vision.

        Body: {filename, question?, fn?}
        fn values: describe, extract-data, ocr, compare, methods, summarize
        """
        _authenticate_ask_image(request, document_id)
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        filename = (body.get("filename") or "").strip()
        if not filename:
            raise HTTPException(status_code=400, detail="'filename' is required")
        if ".." in filename or filename.startswith("/") or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        _IMAGE_EXTS = {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".tif",
            ".tiff",
            ".svg",
            ".bmp",
            ".webp",
        }
        ext = ("." + filename.rsplit(".", 1)[-1]).lower() if "." in filename else ""
        if ext not in _IMAGE_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"Not an image file ({ext}). Supported: {', '.join(sorted(_IMAGE_EXTS))}",
            )

        resolved_id = resolve(document_id)
        result = await papers_module._ask_image(
            document_id=resolved_id,
            figure_id=filename,
            question=body.get("question", ""),
            fn=body.get("fn", ""),
        )

        if "error" in result:
            status = 404 if "not found" in result["error"].lower() else 502
            raise HTTPException(status_code=status, detail=result["error"])

        result["document_id"] = resolved_id
        return result

    @app.get("/papers/{document_id}/ask-image/functions")
    async def list_ask_image_functions(document_id: str):
        """List available built-in image analysis functions."""
        return {
            "functions": {
                k: v[:100] + "..."
                for k, v in papers_module._ASK_IMAGE_FUNCTIONS.items()
            },
            "model": papers_module._ASK_IMAGE_MODEL,
        }

    @app.get("/papers/{document_id}/download")
    async def download_paper(document_id: str):
        """Compile and stream a paper's full folder as a zip archive.

        Includes: meta.json, content.lines, original PDF/XML, figures,
        and supplement files — all pulled from DB + GCS.
        """
        import io
        import json
        import zipfile

        from fastapi.responses import StreamingResponse

        resolved_id = resolve(document_id)
        if not papers_module:
            raise HTTPException(status_code=503, detail="Module not available")

        try:
            conn = _get_db_connection()
            is_pmc = bool(re.match(r"^PMC\d+$", resolved_id, re.IGNORECASE))

            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                # --- meta.json ---
                if is_pmc:
                    module = _get_papers_module()
                    pmc_conn = module._get_pmc_db_connection() if module else None
                    if pmc_conn:
                        with pmc_conn.cursor() as cur:
                            cur.execute(
                                """SELECT pmc_id, title, doi, authors, journal_title,
                                          pub_year, pub_date, article_type, source,
                                          abstract_text, keywords, categories
                                   FROM documents WHERE pmc_id = %s""",
                                (resolved_id,),
                            )
                            row = cur.fetchone()
                        if row:
                            meta = {
                                "pmc_id": row[0],
                                "title": row[1],
                                "doi": row[2],
                                "authors": row[3],
                                "journal": row[4],
                                "pub_year": row[5],
                                "pub_date": str(row[6]) if row[6] else None,
                                "article_type": row[7],
                                "source": row[8],
                                "abstract": row[9],
                                "keywords": row[10],
                                "categories": row[11],
                            }
                            zf.writestr(
                                "meta.json", json.dumps(meta, indent=2, default=str)
                            )
                else:
                    with conn.cursor() as cur:
                        cur.execute(
                            """SELECT document_id::text, title, doi, source, authors,
                                      pub_date, abstract_text
                               FROM documents WHERE document_id::text = %s""",
                            (resolved_id,),
                        )
                        row = cur.fetchone()
                    if row:
                        meta = {
                            "document_id": row[0],
                            "title": row[1],
                            "doi": row[2],
                            "source": row[3],
                            "authors": row[4],
                            "pub_date": str(row[5]) if row[5] else None,
                            "abstract": row[6],
                        }
                        zf.writestr(
                            "meta.json", json.dumps(meta, indent=2, default=str)
                        )

                # --- content.lines (from content_blocks) ---
                if is_pmc:
                    module = _get_papers_module()
                    db_conn = module._get_pmc_db_connection() if module else None
                    id_col = "pmc_id"
                else:
                    db_conn = conn
                    id_col = "document_id::text"

                lines = []
                if db_conn:
                    with db_conn.cursor() as cur:
                        cur.execute(
                            f"""SELECT content FROM content_blocks
                                WHERE {id_col} = %s ORDER BY line_number""",
                            (resolved_id,),
                        )
                        lines = [r[0] for r in cur.fetchall()]
                if lines:
                    zf.writestr("content.lines", "\n".join(lines))

                # --- GCS files (PDF, XML, figures, supplements) ---
                gcs_bucket = _get_gcs_bucket()
                if is_pmc:
                    gcs_prefix = f"pmc/articles/{resolved_id}/"
                else:
                    month = papers_module._get_month_for_doc(resolved_id, conn)
                    source_prefix = (
                        "biorxiv"
                        if papers_module._is_biorxiv(resolved_id, conn)
                        else "medrxiv"
                    )
                    gcs_prefix = f"{source_prefix}_extracted/{month}/{resolved_id}/"

                for blob in gcs_bucket.list_blobs(prefix=gcs_prefix, max_results=500):
                    rel_path = blob.name[len(gcs_prefix) :]
                    if not rel_path or rel_path.startswith("_processed/"):
                        continue
                    try:
                        data = blob.download_as_bytes()
                        zf.writestr(rel_path, data)
                    except Exception as e:
                        logger.warning(f"Skipping {rel_path}: {e}")

            buf.seek(0)
            fname = resolved_id.replace("/", "_")
            return StreamingResponse(
                buf,
                media_type="application/zip",
                headers={"Content-Disposition": f'attachment; filename="{fname}.zip"'},
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error compiling paper download {document_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))


class PapersServer(MCPServer):
    """Papers MCP Server with preprint filesystem capabilities."""

    def __init__(self):
        super().__init__("papers-server")

        # Initialize configuration and managers
        self.config = ServerConfig()
        self.session_manager = SessionManager()
        self.response_manager = ResponseManager(self.session_manager)

        # Register modules
        self._register_modules()

        # Setup MCP handlers
        self.setup_handlers()

        logger.info("PapersServer initialized with preprint tools")

    async def on_startup(self):
        """Called by the HTTP transport lifespan once the event loop is running."""
        _start_es_keepalive()
        logger.info("[startup] ES keep-alive background task scheduled")

    def _register_modules(self):
        """Register all available modules."""
        modules_to_register = [
            ("papers", PapersModule),
        ]

        for module_name, module_class in modules_to_register:
            try:
                module = module_class()
                module.initialize(self.config, self.session_manager)
                self.register_module(module)
                logger.info(
                    f"Registered {module_name} module with {len(module.get_tools())} tools"
                )
            except Exception as e:
                logger.error(f"Failed to register {module_name} module: {e}")

        # Pre-warm database connection to avoid cold start latency.
        # Use a thread with a hard timeout because psycopg2 connect via
        # Cloud SQL Unix socket can block indefinitely during startup race.
        import threading

        def _prewarm_db():
            conn = _get_db_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")

        try:
            t = threading.Thread(target=_prewarm_db, daemon=True)
            t.start()
            t.join(timeout=15)
            if t.is_alive():
                logger.warning("Pre-warm Papers DB connection timed out (15s)")
            else:
                logger.info("Pre-warmed Papers database connection")
        except Exception as e:
            logger.warning(f"Failed to pre-warm Papers DB connection: {e}")

        # Pre-load short ID mappings so first search isn't slow
        # Run in background thread to avoid blocking server startup
        def _bg_load_cache():
            try:
                from modules.papers.short_ids import _ensure_cache
                _ensure_cache()
                logger.info("Short ID cache loaded (background)")
            except Exception as e:
                logger.warning(f"Failed to pre-load short ID cache: {e}")

        threading.Thread(target=_bg_load_cache, daemon=True).start()

        try:
            _prewarm_es()
            logger.info("Pre-warmed Papers Elasticsearch connection")
        except Exception as e:
            logger.warning(f"Failed to pre-warm Papers Elasticsearch: {e}")


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main entry point for the Papers MCP server."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Papers MCP Server - Unified transport (stdio/http)"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default=os.getenv("TRANSPORT_MODE", "stdio"),
        help="Transport mode: stdio for local MCP clients, http for Cloud Run (default: stdio or TRANSPORT_MODE env var)",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("HOST", "0.0.0.0"),
        help="HTTP server host (default: 0.0.0.0 or HOST env var)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", 8083)),
        help="HTTP server port (default: 8083 or PORT env var)",
    )
    parser.add_argument(
        "--path-prefix",
        default=os.getenv("PATH_PREFIX", ""),
        help="URL path prefix, e.g. /paperclip (default: none or PATH_PREFIX env var)",
    )
    parser.add_argument(
        "--sources",
        default=os.getenv("ENABLED_SOURCES"),
        help="Comma-separated data sources to enable (e.g. 'pmc' or 'biorxiv,medrxiv,pmc'). "
        "Overrides ENABLED_SOURCES env var and YAML config.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Run in local mode (accepted for consistency with other servers)",
    )

    args = parser.parse_args()

    # Override enabled sources if specified via CLI
    if args.sources:
        global ENABLED_SOURCES
        ENABLED_SOURCES = {s.strip() for s in args.sources.split(",") if s.strip()}

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 60, file=sys.stderr, flush=True)
    source_label = ", ".join(sorted(ENABLED_SOURCES)) if ENABLED_SOURCES else "all"
    print(
        f"📚 Papers MCP Server Starting (sources: {source_label})",
        file=sys.stderr,
        flush=True,
    )
    print(f"Python version: {sys.version}", file=sys.stderr, flush=True)
    print(f"Working directory: {os.getcwd()}", file=sys.stderr, flush=True)
    print(f"Transport mode: {args.transport}", file=sys.stderr, flush=True)
    print("=" * 60, file=sys.stderr, flush=True)

    try:
        # Load environment variables
        load_environment()

        # Log environment at startup
        try:
            from shared.core.environment import get_environment

            env = get_environment()
            logger.info(f"Starting in {env.value.upper()} environment")
        except Exception as e:
            logger.debug(f"Could not determine environment: {e}")

        # Initialize the core MCP server
        logger.info("🔧 Initializing PapersServer...")
        server = PapersServer()
        logger.info("✓ PapersServer initialized successfully")

        # Create and run the appropriate transport
        if args.transport == "http":
            transport = HTTPTransport(
                server,
                host=args.host,
                port=args.port,
                server_name="Papers MCP Server",
                server_version="1.0.0",
                custom_routes_builder=build_citation_routes,
                path_prefix=args.path_prefix,
            )
        else:  # stdio
            transport = StdioTransport(server)

        # Run the server
        asyncio.run(transport.run())

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ FATAL: Server failed to start: {e}")
        traceback.print_exc()
        sys.exit(1)


def main_pmc():
    """Entry point for PMC-only MCP server (port 8084 by default)."""
    os.environ.setdefault("PORT", "8084")
    sys.argv = (
        [sys.argv[0]] + ["--sources", "pmc", "--transport", "http"] + sys.argv[1:]
    )
    main()


if __name__ == "__main__":
    main()
