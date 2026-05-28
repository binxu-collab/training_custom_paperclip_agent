"""Python client for the bitmap-grep query engine service.

Routes queries across per-corpus slab-grep servers (grep-arxiv,
grep-biomedrxiv, grep-openalex, grep-pmc) based on source filter or
doc_id prefix.

Env vars read:
    SLAB_GREP_URL_ARXIV        grep-arxiv service
    SLAB_GREP_URL_BIOMEDRXIV   grep-biomedrxiv (biorxiv + medrxiv) service
    SLAB_GREP_URL_OPENALEX     grep-openalex (abstract-only) service
    SLAB_GREP_URL_PMC          grep-pmc service
    SLAB_GREP_TIMEOUT          per-request HTTP timeout (seconds)
"""

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)

SLAB_GREP_TIMEOUT = int(os.environ.get("SLAB_GREP_TIMEOUT", "30"))

CORPORA = ("arxiv", "biomedrxiv", "openalex", "pmc")

CORPUS_URLS: dict[str, str] = {
    "arxiv":      os.environ.get("SLAB_GREP_URL_ARXIV",      "http://localhost:8090"),
    "biomedrxiv": os.environ.get("SLAB_GREP_URL_BIOMEDRXIV", "http://localhost:8091"),
    "openalex":   os.environ.get("SLAB_GREP_URL_OPENALEX",   "http://localhost:8092"),
    "pmc":        os.environ.get("SLAB_GREP_URL_PMC",        "http://localhost:8093"),
}


def _derive_cat_url(grep_url: str) -> str:
    """Derive the cat service URL from the grep URL.

    In production, each grep VM runs cat on port 8088.  In local-dev the
    cat ports are tunnelled separately (env vars ``SLAB_CAT_URL_*``), so
    this fallback should rarely be hit.  We use the grep URL itself as
    fallback because some deployments serve /cat on the same port as /grep.
    """
    return grep_url


CORPUS_CAT_URLS: dict[str, str] = {
    "arxiv":      os.environ.get("SLAB_CAT_URL_ARXIV",      _derive_cat_url(CORPUS_URLS["arxiv"])),
    "biomedrxiv": os.environ.get("SLAB_CAT_URL_BIOMEDRXIV", _derive_cat_url(CORPUS_URLS["biomedrxiv"])),
    "openalex":   os.environ.get("SLAB_CAT_URL_OPENALEX",   _derive_cat_url(CORPUS_URLS["openalex"])),
    "pmc":        os.environ.get("SLAB_CAT_URL_PMC",        _derive_cat_url(CORPUS_URLS["pmc"])),
}

_SOURCE_TO_CORPUS = {
    "arxiv":      "arxiv",
    "biomedrxiv": "biomedrxiv",
    "biorxiv":    "biomedrxiv",
    "medrxiv":    "biomedrxiv",
    "openalex":   "openalex",
    "abstracts":  "openalex",
    "pmc":        "pmc",
}

SLAB_GREP_URL = CORPUS_URLS["biomedrxiv"]  # back-compat alias

_ARXIV_NEW_ID = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$", re.IGNORECASE)
_ARXIV_LEGACY = re.compile(r"^[a-z\-]+/\d{7}$", re.IGNORECASE)  # cond-mat/0211069
_PMC_ID = re.compile(r"^PMC\d+$", re.IGNORECASE)
_OA_ID = re.compile(r"^oa_W\d+$")
_BIO_ID = re.compile(r"^(bio|med)_[0-9a-f]+$", re.IGNORECASE)


def corpus_for_doc_id(doc_id: str) -> str:
    """Infer the corpus ('arxiv', 'biomedrxiv', 'openalex', 'pmc') from a doc_id.

    Defaults to 'biomedrxiv' for unrecognized IDs (raw UUIDs without a
    prefix are always biomedrxiv — the only corpus that stores bare UUIDs
    as document IDs).
    """
    if not doc_id:
        return "biomedrxiv"
    if _PMC_ID.match(doc_id):
        return "pmc"
    if _OA_ID.match(doc_id):
        return "openalex"
    if _BIO_ID.match(doc_id):
        return "biomedrxiv"
    if doc_id.startswith("arxiv_") or _ARXIV_NEW_ID.match(doc_id) or _ARXIV_LEGACY.match(doc_id):
        return "arxiv"
    return "biomedrxiv"


def corpus_for_source(source_filter: Optional[str]) -> Optional[str]:
    if source_filter is None:
        return None
    s = source_filter.strip().lower()
    if s in ("", "all"):
        return None
    return _SOURCE_TO_CORPUS.get(s)


def _url_for(corpus: str) -> str:
    return CORPUS_URLS.get(corpus, CORPUS_URLS["biomedrxiv"])


def _partition_doc_ids(doc_ids: Iterable[str]) -> dict[Optional[str], list[str]]:
    out: dict[Optional[str], list[str]] = {}
    for d in doc_ids:
        c = corpus_for_doc_id(d)
        out.setdefault(c, []).append(d)
    return out


def slab_grep_available(corpus: Optional[str] = None) -> bool:
    """Return True if the slab-grep service for ``corpus`` is reachable.

    If ``corpus`` is None, checks the default (legacy) URL.
    """
    url = _url_for(corpus)
    try:
        req = Request(f"{url}/health", method="GET")
        resp = urlopen(req, timeout=3)
        data = json.loads(resp.read())
        return data.get("status") == "ok"
    except Exception:
        return False


def slab_grep_stats(corpus: Optional[str] = None) -> Optional[dict]:
    """Return /stats for one corpus's slab-grep service, or the legacy one."""
    url = _url_for(corpus)
    try:
        req = Request(f"{url}/stats", method="GET")
        resp = urlopen(req, timeout=5)
        return json.loads(resp.read())
    except Exception:
        return None


def _post_grep(
    url: str,
    regex: str,
    limit: int,
    case_insensitive: bool,
    timeout: int,
    doc_ids: Optional[list],
    source_filter: Optional[str],
) -> Optional[dict]:
    body: dict = {
        "regex": regex,
        "limit": limit,
        "case_insensitive": case_insensitive,
    }
    if doc_ids is not None:
        body["doc_ids"] = doc_ids
    if source_filter is not None:
        body["source_filter"] = source_filter
    payload = json.dumps(body).encode("utf-8")
    try:
        req = Request(
            f"{url}/grep",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urlopen(req, timeout=timeout)
        return json.loads(resp.read())
    except URLError as e:
        logger.warning(f"slab-grep service error ({url}): {e}")
        return None
    except Exception as e:
        logger.warning(f"slab-grep client error ({url}): {e}")
        return None


def _merge_grep_results(results: list[Optional[dict]], limit: int) -> dict:
    """Merge multiple /grep responses into a single response.

    Aggregates ``matches`` capped at ``limit``, sums hit counters, keeps
    max elapsed_ms (parallel wall time), and concatenates strategy strings.
    """
    matches: list[dict] = []
    elapsed_ms = 0.0
    bitmap_ms = 0.0
    verify_ms = 0.0
    candidate_docs = 0
    total_hits = 0
    unique_docs = 0
    strategies: list[str] = []
    for r in results:
        if r is None:
            continue
        matches.extend(r.get("matches", []))
        elapsed_ms = max(elapsed_ms, float(r.get("elapsed_ms", 0) or 0))
        bitmap_ms = max(bitmap_ms, float(r.get("bitmap_intersect_ms", 0) or 0))
        verify_ms = max(verify_ms, float(r.get("verify_ms", 0) or 0))
        candidate_docs += int(r.get("candidate_docs", 0) or 0)
        total_hits += int(r.get("total_hits", 0) or 0)
        unique_docs += int(r.get("unique_docs", 0) or 0)
        if r.get("strategy"):
            strategies.append(r["strategy"])
    matches = matches[:limit]
    return {
        "matches": matches,
        "elapsed_ms": elapsed_ms,
        "bitmap_intersect_ms": bitmap_ms,
        "verify_ms": verify_ms,
        "candidate_docs": candidate_docs,
        "total_hits": total_hits,
        "unique_docs": unique_docs,
        "strategy": " | ".join(strategies) if strategies else "fanout",
    }


def slab_grep_search(
    regex: str,
    limit: int = 50,
    case_insensitive: bool = True,
    timeout: Optional[int] = None,
    doc_ids: Optional[list] = None,
    source_filter: Optional[str] = None,
) -> Optional[dict]:
    """Execute a grep query against one or more per-corpus slab-grep services.

    Routing rules (in priority order):
      1. If ``source_filter`` maps to a single corpus → route to that service.
      2. If ``doc_ids`` provided → group by doc_id prefix and fan out to each
         corpus's service in parallel, then merge.
      3. Otherwise → fan out to all 4 corpus services in parallel, merge.

    Returns the merged response dict, or None if every targeted service is
    unavailable.
    """
    t = timeout or SLAB_GREP_TIMEOUT

    # Case 1: explicit source filter -> single service
    corpus = corpus_for_source(source_filter)
    if corpus is not None:
        return _post_grep(
            _url_for(corpus), regex, limit, case_insensitive, t,
            doc_ids=doc_ids, source_filter=source_filter,
        )

    # Case 2: doc_ids provided -> partition by corpus and fan out only to needed
    if doc_ids:
        partitions = _partition_doc_ids(doc_ids)
        targets: list[tuple[str, list[str]]] = [
            (_url_for(c), ids) for c, ids in partitions.items()
        ]
        if not targets:
            return None
        if len(targets) == 1:
            url, ids = targets[0]
            return _post_grep(url, regex, limit, case_insensitive, t,
                              doc_ids=ids, source_filter=None)
        results = _parallel(
            [(url, dict(regex=regex, limit=limit, case_insensitive=case_insensitive,
                        timeout=t, doc_ids=ids, source_filter=None))
             for url, ids in targets]
        )
        return _merge_grep_results(results, limit)

    # Case 3: full corpus fan-out -> dedupe by URL in case multiple corpora
    # share the same server
    url_to_payload: dict[str, dict] = {}
    for c in CORPORA:
        url = _url_for(c)
        url_to_payload.setdefault(url, dict(
            regex=regex, limit=limit, case_insensitive=case_insensitive,
            timeout=t, doc_ids=None, source_filter=None,
        ))
    results = _parallel(list(url_to_payload.items()))
    return _merge_grep_results(results, limit)


def _parallel(jobs: list[tuple[str, dict]]) -> list[Optional[dict]]:
    """Run /grep POSTs in parallel across (url, kwargs) pairs."""
    out: list[Optional[dict]] = [None] * len(jobs)

    def _run(idx: int, url: str, kwargs: dict) -> tuple[int, Optional[dict]]:
        return idx, _post_grep(
            url,
            regex=kwargs["regex"],
            limit=kwargs["limit"],
            case_insensitive=kwargs["case_insensitive"],
            timeout=kwargs["timeout"],
            doc_ids=kwargs.get("doc_ids"),
            source_filter=kwargs.get("source_filter"),
        )

    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
        futures = [pool.submit(_run, i, url, kw) for i, (url, kw) in enumerate(jobs)]
        for f in as_completed(futures):
            idx, result = f.result()
            out[idx] = result
    return out


def slab_grep_parallel(
    queries: list[dict],
    timeout: Optional[int] = None,
) -> list[Optional[dict]]:
    """Execute multiple grep queries in parallel.

    Each query dict is passed through ``slab_grep_search`` (which itself may
    fan out to multiple corpora internally). Returns results in input order.
    """
    results: list[Optional[dict]] = [None] * len(queries)

    def _run(idx: int, q: dict) -> tuple[int, Optional[dict]]:
        return idx, slab_grep_search(
            regex=q["regex"],
            limit=q.get("limit", 50),
            case_insensitive=q.get("case_insensitive", True),
            timeout=timeout or q.get("timeout"),
            doc_ids=q.get("doc_ids"),
            source_filter=q.get("source_filter"),
        )

    with ThreadPoolExecutor(max_workers=max(1, len(queries))) as pool:
        futures = [pool.submit(_run, i, q) for i, q in enumerate(queries)]
        for f in as_completed(futures):
            idx, result = f.result()
            results[idx] = result
    return results


def slab_grep_intersect(
    regexes: list[str],
    case_insensitive: bool = True,
    timeout: Optional[int] = None,
    source_filter: Optional[str] = None,
) -> dict:
    """Grep multiple patterns in parallel and return the intersection of matching doc_ids.

    Fires all regex queries concurrently (each of which may itself fan out to
    all corpus services), collects doc_ids, returns their intersection. Wall
    time ≈ max(individual query times).
    """
    t0 = time.time()

    queries = [
        {"regex": r, "limit": 999999, "case_insensitive": case_insensitive,
         "source_filter": source_filter}
        for r in regexes
    ]
    results = slab_grep_parallel(queries, timeout=timeout)

    doc_id_sets = []
    per_query = []
    for i, (r, res) in enumerate(zip(regexes, results)):
        if res is None:
            per_query.append({"regex": r, "count": 0, "elapsed_ms": 0, "error": True})
            doc_id_sets.append(set())
        else:
            ids = {m["doc_id"] for m in res["matches"]}
            doc_id_sets.append(ids)
            per_query.append({"regex": r, "count": len(ids), "elapsed_ms": res["elapsed_ms"]})

    if doc_id_sets:
        intersection = doc_id_sets[0]
        for s in doc_id_sets[1:]:
            intersection &= s
    else:
        intersection = set()

    return {
        "doc_ids": sorted(intersection),
        "count": len(intersection),
        "per_query": per_query,
        "total_wall_ms": (time.time() - t0) * 1000,
    }


def slab_grep_cat(
    doc_id: str,
    offset: Optional[int] = None,
    length: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Optional[dict]:
    """Read a document's full text from the slab cat service.

    Routes to the per-corpus cat service based on doc_id prefix.
    Returns None if the service is unavailable, so the caller can
    fall back to SQL.
    """
    corpus = corpus_for_doc_id(doc_id)
    url = CORPUS_CAT_URLS.get(corpus, _derive_cat_url(_url_for(corpus)))

    body: dict = {"doc_id": doc_id}
    if offset is not None:
        body["offset"] = offset
    if length is not None:
        body["length"] = length
    payload = json.dumps(body).encode("utf-8")

    try:
        req = Request(
            f"{url}/cat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urlopen(req, timeout=timeout or SLAB_GREP_TIMEOUT)
        return json.loads(resp.read())
    except Exception as e:
        logger.debug(f"slab-grep cat ({url}): {e}")
        return None
