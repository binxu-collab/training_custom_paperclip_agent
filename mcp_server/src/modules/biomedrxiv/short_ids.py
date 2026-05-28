"""Short ID translation layer for biomedRxiv documents.

Maps between full UUIDs (stored in Cloud SQL / GCS / Elasticsearch) and
short user-facing IDs like ``bio_4f78753a6feb`` / ``med_d7148a086f6a``.

The entire mapping (~460K entries, ~40MB) is loaded into memory on first
use and never hits the database again during normal operation.  New papers
added via the DB trigger are picked up on the next cache refresh (every
10 minutes, or on cache miss).

Usage::

    from modules.biomedrxiv.short_ids import resolve, shorten

    uuid = resolve("bio_4f78753a6feb")   # -> "4f78753a-6feb-1014-ac6d-..."
    sid  = shorten("4f78753a-6feb-1014-ac6d-...", "biorxiv")  # -> "bio_4f78753a6feb"

    # UUIDs pass through unchanged:
    resolve("4f78753a-6feb-1014-ac6d-9262620f3a5f")  # -> same string

    # PMC IDs pass through unchanged:
    resolve("PMC7194329")  # -> "PMC7194329"
"""

import logging
import os
import re
import time
import threading

logger = logging.getLogger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_SHORT_RE = re.compile(r"^(bio|med)_[0-9a-f]{12}$", re.IGNORECASE)
_PMC_RE = re.compile(r"^PMC\d+$", re.IGNORECASE)

_PREFIX_MAP = {"biorxiv": "bio_", "medrxiv": "med_"}

_short_to_uuid: dict[str, str] = {}
_uuid_to_short: dict[str, str] = {}
_loaded = False
_load_lock = threading.Lock()
_last_load: float = 0
_CACHE_TTL = 600  # refresh every 10 min


def _load_cache():
    """Populate the in-memory bidirectional cache from Cloud SQL."""
    global _short_to_uuid, _uuid_to_short, _loaded, _last_load

    try:
        import psycopg2

        db_url = os.getenv("BIOMEDRXIV_DB_URL")
        if not db_url:
            logger.warning("[short_ids] BIOMEDRXIV_DB_URL not set, cache empty")
            _loaded = True
            return

        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT short_id, uuid FROM document_ids")
            rows = cur.fetchall()
        conn.close()

        new_s2u: dict[str, str] = {}
        new_u2s: dict[str, str] = {}
        for short_id, uuid in rows:
            new_s2u[short_id] = uuid
            new_u2s[uuid] = short_id

        _short_to_uuid = new_s2u
        _uuid_to_short = new_u2s
        _loaded = True
        _last_load = time.time()
        logger.info("[short_ids] Loaded %d ID mappings", len(new_s2u))

    except Exception as e:
        logger.error("[short_ids] Failed to load cache: %s", e)
        _loaded = True  # don't retry on every call


def _ensure_cache():
    """Load cache if not loaded or stale."""
    if _loaded and (time.time() - _last_load) < _CACHE_TTL:
        return
    with _load_lock:
        if _loaded and (time.time() - _last_load) < _CACHE_TTL:
            return
        _load_cache()


def resolve(identifier: str) -> str:
    """Translate a short ID or UUID to the full UUID for database queries.

    - Short IDs (``bio_4f78753a6feb``) are looked up in the cache.
    - Full UUIDs pass through unchanged.
    - PMC IDs (``PMC7194329``) pass through unchanged.
    - Truncated UUIDs (8-char hex) are tried as prefix matches.
    """
    if not identifier:
        return identifier

    if _UUID_RE.match(identifier):
        return identifier

    if _PMC_RE.match(identifier):
        return identifier

    _ensure_cache()

    if _SHORT_RE.match(identifier):
        uuid = _short_to_uuid.get(identifier)
        if uuid:
            return uuid
        # Cache miss — try refreshing
        _load_cache()
        return _short_to_uuid.get(identifier, identifier)

    # Truncated UUID prefix (e.g. "4f78753a" from LLM output)
    if re.match(r"^[0-9a-f]{7,16}$", identifier, re.IGNORECASE):
        for uuid in _uuid_to_short:
            if uuid.startswith(identifier):
                return uuid
        # Not found — return as-is, let the SQL query fail gracefully
        return identifier

    return identifier


def shorten(uuid: str, source: str | None = None) -> str:
    """Translate a full UUID to its short ID for user-facing display.

    If not in cache, generates deterministically from the UUID + source.
    """
    if not uuid:
        return uuid

    if _PMC_RE.match(uuid):
        return uuid

    if _SHORT_RE.match(uuid):
        return uuid  # already short

    _ensure_cache()

    cached = _uuid_to_short.get(uuid)
    if cached:
        return cached

    # Generate deterministically
    if source:
        prefix = _PREFIX_MAP.get(source, f"{source[:3]}_")
    else:
        prefix = "bio_"
    return prefix + uuid.replace("-", "")[:12]


def shorten_result(result: dict) -> dict:
    """Shorten document_id in a query result dict (in-place)."""
    doc_id = result.get("document_id")
    if doc_id and _UUID_RE.match(doc_id):
        result["document_id"] = shorten(doc_id, result.get("source"))
    return result


def shorten_results(results: list[dict]) -> list[dict]:
    """Shorten document_id in a list of query result dicts (in-place)."""
    for r in results:
        shorten_result(r)
    return results


def is_short_id(identifier: str) -> bool:
    """Check if a string is a valid short ID."""
    return bool(_SHORT_RE.match(identifier))


def is_uuid(identifier: str) -> bool:
    """Check if a string is a full UUID."""
    return bool(_UUID_RE.match(identifier))


def get_source_from_short_id(short_id: str) -> str | None:
    """Extract source from short ID prefix."""
    if short_id.startswith("bio_"):
        return "biorxiv"
    if short_id.startswith("med_"):
        return "medrxiv"
    return None
