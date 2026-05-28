"""Short ID translation layer for biomedRxiv documents.

Maps between full UUIDs (stored in Cloud SQL / GCS / Elasticsearch) and
short user-facing IDs like ``bio_4f78753a6feb`` / ``med_d7148a086f6a``.

The entire mapping (~460K entries, ~40MB) is loaded into memory on first
use and never hits the database again during normal operation.  New papers
added via the DB trigger are picked up on the next cache refresh (every
10 minutes, or on cache miss).

Usage::

    from modules.papers.short_ids import resolve, shorten

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
# OpenAlex / arxiv IDs pass through unchanged. OpenAlex uses numeric ids
# (``oa_1234567890``) and arxiv uses its own paper code (``arxiv_2410.12345`` or
# ``arx_<slug>``). Both are already short and globally unique.
_OA_RE = re.compile(r"^oa_[A-Za-z0-9._-]+$", re.IGNORECASE)
_ARXIV_RE = re.compile(r"^(arxiv|arx)_[A-Za-z0-9._-]+$", re.IGNORECASE)
_BARE_ARXIV_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
# Legacy arxiv IDs look like ``cond-mat/0211069``, ``hep-ex/0001001``,
# ``quant-ph/0204087`` — a lowercase category, a slash, and exactly 7
# digits (YYMMNNN).  The slash is stripped in the shortened form
# (``arx_cond-mat0211069``) so the ID is path-safe, and restored by
# :func:`bare_arxiv_id` / :func:`resolve` before hitting the DB.
_LEGACY_ARXIV_BARE_RE = re.compile(r"^[a-z][a-z-]*/\d{7}$")
_LEGACY_ARXIV_FLAT_RE = re.compile(r"^([a-z][a-z-]*)(\d{7})$")

_PREFIX_MAP = {"biorxiv": "bio_", "medrxiv": "med_", "arxiv": "arx_"}

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

    # OpenAlex IDs pass through unchanged.
    if _OA_RE.match(identifier):
        return identifier

    # arxiv short IDs (arx_2402.02008, arx_quant-ph0204087) → strip prefix and
    # restore the legacy-format slash for DB queries.
    if _ARXIV_RE.match(identifier):
        bare = re.sub(r"^(arxiv|arx)_", "", identifier)
        m = _LEGACY_ARXIV_FLAT_RE.match(bare)
        if m:
            return f"{m.group(1)}/{m.group(2)}"
        return bare

    # Bare legacy arxiv IDs (``quant-ph/0204087``) pass through unchanged —
    # they're already the canonical form used in the ``arxiv`` database.
    if _LEGACY_ARXIV_BARE_RE.match(identifier):
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

    if _OA_RE.match(uuid) or _ARXIV_RE.match(uuid):
        return uuid  # already prefixed

    # Bare modern arxiv IDs (2402.02008) → arx_2402.02008
    if _BARE_ARXIV_RE.match(uuid):
        return f"arx_{uuid}"

    # Legacy arxiv IDs (``quant-ph/0204087``) → ``arx_quant-ph0204087``
    # (strip the slash so the result is path-safe — round-tripped by
    # :func:`bare_arxiv_id` / :func:`resolve`).
    if _LEGACY_ARXIV_BARE_RE.match(uuid):
        return f"arx_{uuid.replace('/', '')}"

    # ``source='arxiv'`` rows that didn't match either arxiv regex above are
    # anomalies; fall through to the category-aware prefix so we don't mislabel
    # them as ``bio_``.
    if source == "arxiv":
        return f"arx_{uuid.replace('/', '')}"

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
    if not doc_id:
        return result
    source = result.get("source")
    if (
        _UUID_RE.match(doc_id)
        or _BARE_ARXIV_RE.match(doc_id)
        or _LEGACY_ARXIV_BARE_RE.match(doc_id)
        or source == "arxiv"
    ):
        result["document_id"] = shorten(doc_id, source)
    return result


def shorten_results(results: list[dict]) -> list[dict]:
    """Shorten document_id in a list of query result dicts (in-place)."""
    for r in results:
        shorten_result(r)
    return results


def is_arxiv_id(identifier: str) -> bool:
    """Check if a string is an arxiv ID (bare or prefixed, modern or legacy)."""
    return bool(
        _BARE_ARXIV_RE.match(identifier)
        or _LEGACY_ARXIV_BARE_RE.match(identifier)
        or _ARXIV_RE.match(identifier)
    )


def bare_arxiv_id(identifier: str) -> str:
    """Strip ``arx_``/``arxiv_`` prefix and restore legacy-style slashes.

    Modern IDs round-trip verbatim (``arx_2402.02008`` → ``2402.02008``).
    Legacy IDs restore the slash stripped by :func:`shorten`
    (``arx_quant-ph0204087`` → ``quant-ph/0204087``).  Accepts bare IDs too,
    passing them through unchanged.
    """
    stripped = re.sub(r"^(arxiv|arx)_", "", identifier)
    m = _LEGACY_ARXIV_FLAT_RE.match(stripped)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return stripped


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
    if short_id.startswith("arx_") or short_id.startswith("arxiv_"):
        return "arxiv"
    if short_id.startswith("oa_"):
        return "openalex"
    return None
