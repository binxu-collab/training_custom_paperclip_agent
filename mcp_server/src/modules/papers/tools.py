"""
BioMedRxiv Database module implementation for MCP server.

This module provides tools for querying the biomedrxiv CloudSQL database
containing 453K+ preprint articles from bioRxiv and medRxiv.

Database: gxl-prod:us-central1:biomedrxiv
Tables:
  - documents: Article metadata (453K rows)
  - content_blocks: Full-text content (~70M rows)
"""

import base64
import json
import logging
import os

# re module removed - no longer used after removing text-search fallback
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psycopg2
from lxml import etree
from mcp.types import ImageContent, TextContent, Tool

from ..base import ToolModule

logger = logging.getLogger(__name__)

# =========================================================================
# OpenSearch Client (lazy-loaded via search_backends)
# =========================================================================
_es_client = None


def _get_es_client():
    """Get OpenSearch client for full-text content search."""
    global _es_client
    if _es_client is None:
        from mcps.papers.servers.search_backends import get_opensearch_client

        _es_client = get_opensearch_client()
        if _es_client is not None:
            logger.info("Papers tools using OpenSearch client from search_backends")
        else:
            logger.warning("OpenSearch not configured (set OPENSEARCH_URL)")

    return _es_client


OS_INDEX_PREPRINTS = "preprints"
OS_INDEX_PMC = "pmc"
OS_INDEX_NAME = "preprints,pmc"

# Note: GCS bucket/paths are stored in the database (citation_info->>'source_path')
# so no bucket env var is needed - paths come from DB like:
#   gs://gxl-collections/biorxiv_extracted/...
#   gs://gxl-collections/medrxiv_extracted/...

# Lazy-loaded GCS client
_gcs_client = None
_gcs_client_failed = False  # Track if initialization failed (don't retry every call)


def _get_gcs_client(force_refresh: bool = False):
    """Get GCS client (lazy loaded with refresh capability).

    Uses Application Default Credentials or GOOGLE_APPLICATION_CREDENTIALS env var.
    For local dev: run `gcloud auth application-default login`

    Args:
        force_refresh: If True, recreate the client (useful after auth refresh)
    """
    global _gcs_client, _gcs_client_failed

    if force_refresh:
        _gcs_client = None
        _gcs_client_failed = False

    if _gcs_client_failed:
        return None

    if _gcs_client is None:
        try:
            from google.cloud import storage

            _gcs_client = storage.Client()
            logger.info(f"✅ Initialized GCS client (project: {_gcs_client.project})")
        except ImportError:
            logger.warning("google-cloud-storage not installed")
            _gcs_client_failed = True
            return None
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            _gcs_client_failed = True
            return None
    return _gcs_client


def reset_gcs_client():
    """Reset the GCS client to force re-initialization.

    Call this after running `gcloud auth application-default login` or
    when credentials have been refreshed.
    """
    global _gcs_client, _gcs_client_failed
    _gcs_client = None
    _gcs_client_failed = False
    logger.info("GCS client reset - will reinitialize on next use")


def get_biomedrxiv_connection():
    """Get a connection to the biomedrxiv database.
    
    Requires environment variables:
    - BIOMEDRXIV_DB_URL (connection string), OR:
    - BIOMEDRXIV_DB_HOST, BIOMEDRXIV_DB_PASSWORD (required)
    - BIOMEDRXIV_DB_PORT (default: 5432)
    - BIOMEDRXIV_DB_NAME (default: biomedrxiv)
    - BIOMEDRXIV_DB_USER (default: postgres)
    """
    if url := os.getenv("BIOMEDRXIV_DB_URL"):
        logger.info("Using BIOMEDRXIV_DB_URL for connection")
        conn = psycopg2.connect(url)
        conn.autocommit = True
        return conn

    # Direct connection - requires host and password
    host = os.getenv("BIOMEDRXIV_DB_HOST")
    password = os.getenv("BIOMEDRXIV_DB_PASSWORD")
    
    if not host or not password:
        raise ValueError(
            "Database not configured. Set BIOMEDRXIV_DB_URL or BIOMEDRXIV_DB_HOST + BIOMEDRXIV_DB_PASSWORD."
        )
    
    logger.info("Using direct biomedrxiv connection")
    conn = psycopg2.connect(
        host=host,
        port=int(os.getenv("BIOMEDRXIV_DB_PORT", "5432")),
        database=os.getenv("BIOMEDRXIV_DB_NAME", "biomedrxiv"),
        user=os.getenv("BIOMEDRXIV_DB_USER", "postgres"),
        password=password,
    )
    conn.autocommit = True
    return conn


# LRU cache for XML content to avoid repeated GCS downloads
import time as profile_time
import hashlib

_xml_cache: dict[str, str] = {}
_xml_cache_max_size = 50

# Disk cache directory for persistence across restarts
_DISK_CACHE_DIR = Path("/tmp/biomedrxiv_xml_cache")
_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_disk_cache_path(gcs_path: str) -> Path:
    """Get disk cache path for a GCS path."""
    cache_key = hashlib.md5(gcs_path.encode()).hexdigest()
    return _DISK_CACHE_DIR / f"{cache_key}.xml"


def download_xml_from_gcs(gcs_path: str) -> str | None:
    """Download XML content from GCS with multi-level caching and profiling.

    Cache hierarchy:
    1. In-memory cache (fastest, up to 50 XMLs)
    2. Disk cache (persists across restarts)
    3. GCS download (slowest, ~1-3s)

    Args:
        gcs_path: GCS path like gs://bucket/path/to/file.xml

    Returns:
        XML content as string, or None if download failed
    """
    timings = {}
    t_start = profile_time.perf_counter()

    if not gcs_path or not gcs_path.startswith("gs://"):
        logger.error(f"Invalid GCS path: {gcs_path}")
        return None

    # Level 1: Check in-memory cache
    if gcs_path in _xml_cache:
        logger.debug(f"[GCS CACHE] Memory HIT for {gcs_path}")
        return _xml_cache[gcs_path]
    
    # Level 2: Check disk cache
    disk_cache_path = _get_disk_cache_path(gcs_path)
    if disk_cache_path.exists():
        try:
            content = disk_cache_path.read_text()
            # Promote to memory cache
            if len(_xml_cache) >= _xml_cache_max_size:
                oldest_key = next(iter(_xml_cache))
                del _xml_cache[oldest_key]
            _xml_cache[gcs_path] = content
            logger.debug(f"[GCS CACHE] Disk HIT for {gcs_path}")
            return content
        except Exception as e:
            logger.warning(f"Disk cache read failed: {e}")

    path_without_prefix = gcs_path[5:]
    parts = path_without_prefix.split("/", 1)
    if len(parts) != 2:
        logger.error(f"Cannot parse GCS path: {gcs_path}")
        return None

    bucket_name, blob_path = parts

    # Get GCS client
    t0 = profile_time.perf_counter()
    client = _get_gcs_client()
    timings["get_client"] = (profile_time.perf_counter() - t0) * 1000

    if not client:
        return None

    try:
        # Get bucket reference
        t0 = profile_time.perf_counter()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        timings["get_blob_ref"] = (profile_time.perf_counter() - t0) * 1000

        # Download content directly (skip exists() check - it's slow!)
        # If blob doesn't exist, download_as_text will raise NotFound
        t0 = profile_time.perf_counter()
        content = blob.download_as_text()
        timings["download"] = (profile_time.perf_counter() - t0) * 1000
        timings["size_kb"] = len(content) / 1024

        # Cache the result in memory (with simple LRU eviction)
        if len(_xml_cache) >= _xml_cache_max_size:
            # Remove oldest entry (first key)
            oldest_key = next(iter(_xml_cache))
            del _xml_cache[oldest_key]
        _xml_cache[gcs_path] = content
        
        # Also cache to disk for persistence
        try:
            disk_cache_path.write_text(content)
        except Exception as e:
            logger.warning(f"Disk cache write failed: {e}")

        timings["total"] = (profile_time.perf_counter() - t_start) * 1000
        logger.info(f"[GCS CACHE] MISS - Downloaded {gcs_path}: {timings}")

        return content
    except Exception as e:
        logger.error(f"Failed to download XML: {e}")
        return None


def download_image_from_gcs(gcs_path: str) -> bytes | None:
    """Download image content from GCS.

    Args:
        gcs_path: GCS path like gs://bucket/path/to/file.png

    Returns:
        Image bytes, or None if download failed
    """
    if not gcs_path or not gcs_path.startswith("gs://"):
        logger.error(f"Invalid GCS path: {gcs_path}")
        return None

    path_without_prefix = gcs_path[5:]
    parts = path_without_prefix.split("/", 1)
    if len(parts) != 2:
        logger.error(f"Cannot parse GCS path: {gcs_path}")
        return None

    bucket_name, blob_path = parts

    client = _get_gcs_client()
    if not client:
        logger.error(
            "GCS client not available. For local dev, run: gcloud auth application-default login"
        )
        return None

    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        if not blob.exists():
            logger.warning(f"Image blob not found: {blob_path} in bucket {bucket_name}")
            return None

        return blob.download_as_bytes()
    except Exception as e:
        error_msg = str(e)
        if "Reauthentication" in error_msg or "credentials" in error_msg.lower() or "refresh" in error_msg.lower():
            # Auth failed - try once more with a fresh client
            logger.warning(f"GCS auth failed, retrying with fresh client: {e}")
            client = _get_gcs_client(force_refresh=True)
            if client:
                try:
                    bucket = client.bucket(bucket_name)
                    blob = bucket.blob(blob_path)
                    if blob.exists():
                        return blob.download_as_bytes()
                except Exception as retry_error:
                    logger.error(f"GCS retry also failed: {retry_error}")
            
            logger.error(
                f"GCS authentication failed: {e}\n"
                "Fix for local dev: run `gcloud auth application-default login`\n"
                "Fix for Cloud Run: ensure service account has storage.objects.get permission"
            )
        else:
            logger.error(f"Failed to download image from {gcs_path}: {e}")
        return None


def generate_signed_download_url(gcs_path: str, expiry_minutes: int = 5) -> str | None:
    """Generate a short-lived signed URL for downloading a GCS object.

    Works for any blob (images, PDFs, etc.) without streaming bytes through
    our servers. The URL is valid for *expiry_minutes* (default 5).

    The default is intentionally short: callers (e.g. ``paperclip cat > file``)
    consume the URL immediately. A tighter TTL limits the blast radius if a
    URL leaks via an error message, a proxy log, or shell history.

    On Cloud Run the default metadata-server credentials can't sign URLs
    locally (no private key). When that fails we fall back to IAM API's
    signBlob by passing service_account_email + access_token, which works
    as long as the runtime SA has roles/iam.serviceAccountTokenCreator on
    itself.
    """
    import datetime

    if not gcs_path or not gcs_path.startswith("gs://"):
        logger.error(f"Invalid GCS path for signed URL: {gcs_path}")
        return None

    path_without_prefix = gcs_path[5:]
    parts = path_without_prefix.split("/", 1)
    if len(parts) != 2:
        return None

    bucket_name, blob_path = parts
    client = _get_gcs_client()
    if not client:
        return None

    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        if not blob.exists():
            logger.warning(f"Blob not found for signed URL: {blob_path}")
            return None
        expiration = datetime.timedelta(minutes=expiry_minutes)
        signed_url: str | None = None
        signing_mode: str
        try:
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=expiration,
                method="GET",
            )
            signing_mode = "direct"
        except Exception as direct_err:
            # Fall back to IAM signBlob path when the runtime credentials
            # can't sign locally (typical on Cloud Run).
            try:
                import google.auth
                import google.auth.transport.requests

                credentials, _ = google.auth.default()
                auth_request = google.auth.transport.requests.Request()
                credentials.refresh(auth_request)
                sa_email = getattr(credentials, "service_account_email", None)
                token = getattr(credentials, "token", None)
                if not sa_email or not token:
                    raise direct_err
                signed_url = blob.generate_signed_url(
                    version="v4",
                    expiration=expiration,
                    method="GET",
                    service_account_email=sa_email,
                    access_token=token,
                )
                signing_mode = "iam_signblob"
            except Exception as iam_err:
                logger.warning(
                    f"Failed to generate signed URL for {gcs_path}: "
                    f"direct={direct_err!r}; iam={iam_err!r}"
                )
                return None

        # M3: structured audit log of every signed-URL issuance. Paired with
        # per-request logs in the CLI router so (user, object, issued_at) is
        # reconstructable from log aggregation.
        logger.info(
            "signed_url_issued",
            extra={
                "gcs_path": gcs_path,
                "expiry_minutes": expiry_minutes,
                "method": "GET",
                "signing_mode": signing_mode,
            },
        )
        return signed_url
    except Exception as e:
        logger.warning(f"Failed to generate signed URL for {gcs_path}: {e}")
        return None


def download_json_from_gcs(gcs_path: str) -> dict | None:
    """Download JSON content from GCS.

    Args:
        gcs_path: GCS path like gs://bucket/path/to/file.json

    Returns:
        Parsed JSON as dict, or None if download failed
    """
    import json
    
    if not gcs_path or not gcs_path.startswith("gs://"):
        logger.error(f"Invalid GCS path: {gcs_path}")
        return None

    path_without_prefix = gcs_path[5:]
    parts = path_without_prefix.split("/", 1)
    if len(parts) != 2:
        logger.error(f"Cannot parse GCS path: {gcs_path}")
        return None

    bucket_name, blob_path = parts

    client = _get_gcs_client()
    if not client:
        logger.error(
            "GCS client not available. For local dev, run: gcloud auth application-default login"
        )
        return None

    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        if not blob.exists():
            logger.warning(f"JSON blob not found: {blob_path} in bucket {bucket_name}")
            return None

        content = blob.download_as_text()
        return json.loads(content)
    except Exception as e:
        error_msg = str(e)
        if "Reauthentication" in error_msg or "credentials" in error_msg.lower() or "refresh" in error_msg.lower():
            # Auth failed - try once more with a fresh client
            logger.warning(f"GCS auth failed, retrying with fresh client: {e}")
            client = _get_gcs_client(force_refresh=True)
            if client:
                try:
                    bucket = client.bucket(bucket_name)
                    blob = bucket.blob(blob_path)
                    if blob.exists():
                        content = blob.download_as_text()
                        return json.loads(content)
                except Exception as retry_error:
                    logger.error(f"GCS retry also failed: {retry_error}")
            
            logger.error(
                f"GCS authentication failed: {e}\n"
                "Fix for local dev: run `gcloud auth application-default login`\n"
                "Fix for Cloud Run: ensure service account has storage.objects.get permission"
            )
        else:
            logger.error(f"Failed to download JSON from {gcs_path}: {e}")
        return None


def render_xml_to_html(
    xml_content: str,
    highlight_xpath: str | None = None,
    highlight_xml_id: str | None = None,
    highlight_text: str | None = None,
    highlight_section: str | None = None,
    image_base_url: str | None = None,
    image_url_params: str | None = None,
) -> str:
    """Render JATS XML to readable HTML with optional highlighting.

    Clean, fast rendering inspired by academic paper layouts.

    Args:
        xml_content: Raw JATS XML string
        highlight_xpath: Optional XPath to highlight in the rendered HTML
        highlight_xml_id: Optional xml:id to highlight (faster than xpath)
        highlight_text: Optional text content to highlight (fallback if xpath/xml_id don't match)
        highlight_section: Optional section name to highlight (fallback when xml_id unavailable)
        image_base_url: URL prefix for figure images (e.g., /mcp/biomedrxiv/images/{doc_id}/).
                        Combined with each figure's href to produce the full image src.
        image_url_params: Query string appended after the href (e.g., "?b=Z3M6...").
                          Use this to carry routing metadata without extra DB lookups.

    Returns:
        HTML string with rendered article
    """
    import html as html_lib
    import re
    from difflib import SequenceMatcher

    def normalize_text(text: str) -> str:
        """Normalize text for matching - normalize dashes, whitespace, quotes, etc."""
        # Normalize various dash characters to regular hyphen
        dash_chars = "\u002d\u2010\u2011\u2012\u2013\u2014\u2212"
        for dash in dash_chars:
            text = text.replace(dash, "-")
        # Normalize quotes
        text = text.replace(""", '"').replace(""", '"')
        text = text.replace("'", "'").replace("'", "'")
        # Normalize tilde/approximately signs to ~
        text = text.replace(chr(0x223c), "~").replace(chr(0x2248), "~").replace(chr(0x02dc), "~")
        # Normalize multiplication/times signs to x
        text = text.replace(chr(0x00d7), "x").replace(chr(0x2715), "x")
        # Normalize whitespace
        text = " ".join(text.split())
        return text

    def fuzzy_find_substring(
        needle: str, haystack: str, threshold: float = 0.90
    ) -> tuple[int, int] | None:
        """Find a substring in text using fuzzy matching.

        Args:
            needle: The text to search for
            haystack: The text to search in
            threshold: Minimum similarity ratio (0.0 to 1.0, default 0.90 = 90%)

        Returns:
            Tuple of (start, end) positions in original haystack, or None if not found
        """
        logger.info(f"[FUZZY MATCH] needle ({len(needle)} chars): {needle[:80]}...")
        logger.info(f"[FUZZY MATCH] haystack ({len(haystack)} chars)")

        needle = needle.strip()
        if not needle or not haystack:
            logger.info("[FUZZY MATCH] Empty needle or haystack")
            return None

        # Normalize both for comparison
        normalized_needle = normalize_text(needle).lower()
        normalized_haystack = normalize_text(haystack).lower()

        # Build a mapping from normalized positions to original positions
        # We need to map between the normalized string and the original
        # so that match positions can be translated back.
        norm_to_orig = []
        norm_haystack_chars = []
        for i, char in enumerate(haystack):
            if char.isspace():
                # Collapse whitespace: add a single space if prev wasn't space
                if norm_haystack_chars and norm_haystack_chars[-1] != " ":
                    norm_to_orig.append(i)
                    norm_haystack_chars.append(" ")
            else:
                # Normalize the character (dashes, quotes, etc.)
                nc = normalize_text(char).lower()
                if nc:
                    norm_to_orig.append(i)
                    norm_haystack_chars.append(nc)

        normalized_haystack_rebuilt = "".join(norm_haystack_chars)

        # Try exact match on normalized text
        escaped = re.escape(normalized_needle)
        match = re.search(escaped, normalized_haystack_rebuilt, flags=re.IGNORECASE)
        logger.info(f"[FUZZY MATCH] exact match: {match is not None}")
        if match:
            # Map back to original positions
            if match.start() < len(norm_to_orig) and match.end() <= len(norm_to_orig):
                orig_start = norm_to_orig[match.start()]
                # Find end position - need to include the full matched character
                orig_end = (
                    norm_to_orig[match.end() - 1] + 1 if match.end() > 0 else orig_start
                )
                # Extend to include trailing characters that might be part of the match
                while (
                    orig_end < len(haystack)
                    and haystack[orig_end - 1 : orig_end].isalnum()
                ):
                    if orig_end < len(haystack) and not haystack[orig_end].isalnum():
                        break
                    orig_end += 1
                return (orig_start, orig_end)

        # Skip expensive fuzzy sliding window - it's O(n²) and rarely succeeds
        # If exact match fails, just return None and let the paragraph-level highlight work
        logger.info("[FUZZY MATCH] exact match failed, skipping slow fuzzy search")
        return None

    def get_text(elem):
        """Extract clean text from element (no inline formatting)."""
        if elem is None:
            return ""
        return " ".join(elem.itertext()).strip()

    def render_inline(elem, skip_tags: set = None) -> str:
        """Render element with inline formatting preserved (xref, sup, sub, italic, bold).

        Args:
            elem: The XML element to render
            skip_tags: Set of tag names to skip (e.g., {"label"} to skip label elements)
        """
        if elem is None:
            return ""

        skip_tags = skip_tags or set()
        parts = []

        # Add element's direct text
        if elem.text:
            parts.append(html_lib.escape(elem.text))

        # Process child elements
        for child in elem:
            # Skip specified tags
            if child.tag in skip_tags:
                # But still add the tail text
                if child.tail:
                    parts.append(html_lib.escape(child.tail))
                continue
            tag = child.tag

            if tag == "xref":
                # Cross-reference (citation numbers like [1], [2], Figure 1, etc.)
                ref_type = child.get("ref-type", "")
                rid = child.get("rid", "")  # Target element ID
                text = get_text(child)
                if ref_type == "bibr":
                    # Bibliography reference - just show as styled text (no link)
                    parts.append(
                        f'<span class="xref-bibr">{html_lib.escape(text)}</span>'
                    )
                elif ref_type == "aff":
                    # Affiliation reference - superscript
                    parts.append(f'<sup class="xref-aff">{html_lib.escape(text)}</sup>')
                elif ref_type == "fig":
                    # Figure reference - clickable link to figure
                    if rid:
                        onclick = f"document.getElementById('{rid}')" \
                            "?.scrollIntoView({behavior:'smooth',block:'center',container:'nearest'})"
                        parts.append(
                            f'<span class="xref-fig" onclick="{onclick}">'
                            f'{html_lib.escape(text)}</span>'
                        )
                    else:
                        parts.append(
                            f'<span class="xref-fig">{html_lib.escape(text)}</span>'
                        )
                elif ref_type == "table":
                    # Table reference - clickable link to table
                    if rid:
                        onclick = f"document.getElementById('{rid}')" \
                            "?.scrollIntoView({behavior:'smooth',block:'center',container:'nearest'})"
                        parts.append(
                            f'<span class="xref-table" onclick="{onclick}">'
                            f'{html_lib.escape(text)}</span>'
                        )
                    else:
                        parts.append(
                            f'<span class="xref-table">{html_lib.escape(text)}</span>'
                        )
                else:
                    parts.append(f'<span class="xref">{html_lib.escape(text)}</span>')
            elif tag == "sup":
                parts.append(f"<sup>{render_inline(child)}</sup>")
            elif tag == "sub":
                parts.append(f"<sub>{render_inline(child)}</sub>")
            elif tag == "italic":
                parts.append(f"<em>{render_inline(child)}</em>")
            elif tag == "bold":
                parts.append(f"<strong>{render_inline(child)}</strong>")
            elif tag == "underline":
                parts.append(f"<u>{render_inline(child)}</u>")
            elif tag == "sc":  # small caps
                parts.append(
                    f'<span style="font-variant:small-caps">{render_inline(child)}</span>'
                )
            elif tag == "monospace":
                parts.append(f"<code>{render_inline(child)}</code>")
            elif tag == "ext-link":
                href = child.get("{http://www.w3.org/1999/xlink}href", "#")
                parts.append(
                    f'<a href="{html_lib.escape(href)}" target="_blank">{render_inline(child)}</a>'
                )
            elif tag in ("inline-formula", "disp-formula"):
                # Math formulas - just show as monospace for now
                parts.append(f'<code class="formula">{get_text(child)}</code>')
            else:
                # Unknown inline element - render its content
                parts.append(render_inline(child))

            # Add tail text (text after the child element)
            if child.tail:
                parts.append(html_lib.escape(child.tail))

        return "".join(parts)

    try:
        render_timings = {}
        t_render_start = profile_time.perf_counter()

        # Parse XML with recovery for malformed content
        t0 = profile_time.perf_counter()
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(xml_content.encode(), parser)
        render_timings["xml_parse"] = (profile_time.perf_counter() - t0) * 1000

        # Build HTML with clean styling
        # Collect sections for TOC as we build HTML
        toc_entries = []

        html_parts = [
            "<!DOCTYPE html>",
            '<html><head><meta charset="utf-8">',
            "<style>",
            "* { box-sizing: border-box; }",
            # Layout with sidebar
            "html { scroll-behavior: smooth; }",
            "body { font-family: Georgia, \"Times New Roman\", serif; margin: 0; padding: 0; line-height: 1.7; color: #222; background: #fff; display: flex; }",
            # TOC sidebar styles
            ".toc-sidebar { position: sticky; top: 0; left: 0; width: 200px; min-width: 200px; height: 100vh; overflow-y: auto; background: #f8f9fa; border-right: 1px solid #e9ecef; padding: 16px 12px; font-size: 12px; }",
            ".toc-sidebar::-webkit-scrollbar { width: 4px; }",
            ".toc-sidebar::-webkit-scrollbar-thumb { background: #ccc; border-radius: 2px; }",
            ".toc-title { font-size: 11px; font-weight: 600; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #ddd; }",
            ".toc-item { display: block; padding: 4px 0; color: #555; text-decoration: none; line-height: 1.4; border-left: 2px solid transparent; padding-left: 8px; margin-left: -8px; transition: all 0.15s ease; cursor: pointer; }",
            ".toc-item:hover { color: #2c3e50; background: #e9ecef; border-left-color: #3498db; }",
            ".toc-item.active { color: #2c3e50; font-weight: 600; border-left-color: #3498db; }",
            # Main content area
            ".main-content { flex: 1; max-width: 800px; padding: 40px 24px; margin: 0 auto; }",
            "h1 { font-size: 26px; font-weight: 600; color: #1a1a1a; margin: 0 0 16px; line-height: 1.3; }",
            ".authors { font-size: 14px; color: #555; margin-bottom: 12px; }",
            ".affiliation { font-size: 12px; color: #777; margin-bottom: 6px; padding-left: 16px; }",
            ".doi { font-size: 12px; color: #3498db; margin: 16px 0 24px; }",
            ".section-header { font-size: 18px; font-weight: 600; color: #2c3e50; margin: 32px 0 16px; padding-bottom: 8px; border-bottom: 2px solid #eee; scroll-margin-top: 20px; }",
            ".section-header.level-2 { font-size: 15px; margin: 24px 0 12px; border-bottom: 1px solid #eee; }",
            ".section-header.level-3 { font-size: 13px; margin: 20px 0 10px; border-bottom: none; color: #444; }",
            ".toc-item.toc-level-2 { padding-left: 20px; font-size: 11px; }",
            ".abstract { background: #f8f9fa; padding: 20px 24px; border-left: 4px solid #3498db; margin: 24px 0; }",
            ".abstract-label { font-weight: 600; color: #2c3e50; margin-bottom: 8px; }",
            "p { font-size: 14px; line-height: 1.7; color: #333; margin-bottom: 14px; text-align: justify; }",
            ".figure { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; padding: 16px; margin: 20px 0; scroll-margin-top: 20px; }",
            ".figure img { max-width: 100%; height: auto; display: block; margin: 12px auto; }",
            ".figure-caption { font-size: 12px; color: #555; margin-top: 12px; line-height: 1.5; }",
            # Improved table styles - smaller fonts
            ".table-wrap { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; padding: 12px; margin: 20px 0; overflow-x: auto; scroll-margin-top: 20px; }",
            "table { border-collapse: collapse; width: 100%; font-size: 11px; }",
            "th, td { border: 1px solid #ddd; padding: 4px 8px; text-align: left; max-width: 200px; overflow: hidden; text-overflow: ellipsis; }",
            "th { background: #f0f0f0; font-weight: 600; font-size: 10px; text-transform: uppercase; letter-spacing: 0.3px; }",
            "td { vertical-align: top; line-height: 1.4; }",
            ".supplement { background: #e8f4fc; border: 1px solid #b8daef; border-radius: 6px; padding: 12px 16px; margin: 16px 0; font-size: 13px; }",
            ".reference { font-size: 12px; color: #555; margin-bottom: 8px; padding-left: 24px; text-indent: -24px; line-height: 1.5; }",
            ".ack { background: #f8f9fa; padding: 16px 20px; border-radius: 6px; margin-top: 32px; font-size: 13px; }",
            # Clickable figure/table references
            ".xref-bibr { color: #3498db; }",
            ".xref-aff { color: #e74c3c; font-size: 0.75em; vertical-align: super; margin-left: 1px; }",
            ".xref-fig, .xref-table { color: #27ae60; cursor: pointer; text-decoration: underline; text-decoration-style: dotted; text-underline-offset: 2px; }",
            ".xref-fig:hover, .xref-table:hover { color: #1e8449; text-decoration-style: solid; }",
            ".xref { color: #666; }",
            "sup { font-size: 0.75em; vertical-align: super; }",
            "sub { font-size: 0.75em; vertical-align: sub; }",
            ".formula { background: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-family: monospace; }",
            # Highlighting
            ".highlighted { background: #fff3cd !important; box-shadow: 0 0 0 4px #fff3cd; border-radius: 4px; }",
            "#highlight-target { background: #fff3cd !important; box-shadow: 0 0 0 4px #fff3cd; border-radius: 4px; }",
            # Responsive: hide TOC on small screens
            "@media (max-width: 900px) { .toc-sidebar { display: none; } .main-content { max-width: 100%; } }",
            "</style>",
            "</head><body>",
            # TOC placeholder - will be filled in at the end
            '<nav class="toc-sidebar" id="toc-nav"></nav>',
            '<main class="main-content">',
        ]

        # Track elements by xml:id for fast highlighting
        id_to_index = {}
        element_counter = [0]  # Use list to allow modification in nested function

        # Resolve highlight_xpath to get the target element BEFORE rendering
        # This allows precise element matching by reference, not text search
        highlight_target_elem = None
        if highlight_xpath and not highlight_xml_id:
            try:
                elements = root.xpath(highlight_xpath)
                if not elements and not highlight_xpath.startswith("//"):
                    # Try with relative xpath
                    elements = root.xpath("//" + highlight_xpath.lstrip("/"))
                if elements:
                    highlight_target_elem = elements[0]
            except Exception:
                pass  # Invalid xpath, will fall back to text search

        def add_element(
            tag: str,
            content: str,
            xml_id: str = "",
            extra_attrs: str = "",
            wrapper: str = "",
            source_elem=None,  # The source XML element for precise matching
        ) -> str:
            """Add HTML element with data attributes for highlighting."""
            element_counter[0] += 1
            idx = element_counter[0]
            if xml_id:
                id_to_index[xml_id] = idx

            # Check if this element should be highlighted:
            # 1. By xml_id match (if element has an ID)
            # 2. By element reference match (if we resolved xpath to target element)
            should_highlight = (highlight_xml_id and xml_id == highlight_xml_id) or (
                highlight_target_elem is not None
                and source_elem is highlight_target_elem
            )
            highlight_attr = ' id="highlight-target"' if should_highlight else ""
            highlight_class = " highlighted" if should_highlight else ""

            if should_highlight:
                highlight_found[0] = True

            data_attrs = f'data-idx="{idx}"'
            if xml_id:
                data_attrs += f' data-xmlid="{xml_id}"'

            if wrapper:
                return f'<{wrapper} class="{tag}{highlight_class}"{highlight_attr} {data_attrs} {extra_attrs}>{content}</{wrapper}>'
            return f'<{tag}{highlight_class}"{highlight_attr} {data_attrs} {extra_attrs}>{content}</{tag}>'

        # Track if we've already highlighted something (used later for body sections)
        highlight_found = [False]

        # Title
        title_elem = root.find(".//article-title")
        if title_elem is not None:
            title_html = render_inline(title_elem)
            xml_id = title_elem.get("id", "")
            # Check if Title section should be highlighted
            title_highlight = (highlight_xml_id and xml_id == highlight_xml_id) or (
                highlight_section
                and highlight_section.lower() == "title"
                and not highlight_found[0]
            )
            if title_highlight:
                highlight_found[0] = True
            title_attr = (
                ' id="highlight-target" class="highlighted"' if title_highlight else ""
            )
            html_parts.append(
                f'<h1{title_attr} data-xmlid="{xml_id}">{title_html}</h1>'
            )

        # Authors (with affiliation superscripts)
        authors = []
        for contrib in root.iter("contrib"):
            if contrib.get("contrib-type") == "author":
                name_elem = contrib.find(".//name")
                if name_elem is not None:
                    surname = get_text(name_elem.find("surname"))
                    given = get_text(name_elem.find("given-names"))
                    if surname or given:
                        author_name = f"{given} {surname}".strip()

                        # Get affiliation references (superscript numbers)
                        # Try multiple approaches since JATS structure varies
                        aff_refs = []

                        # Method 1: xref elements with ref-type="aff"
                        for xref in contrib.findall(".//xref[@ref-type='aff']"):
                            ref_text = get_text(xref)
                            if ref_text:
                                aff_refs.append(ref_text)

                        # Method 2: xref elements without ref-type but with rid starting with "aff"
                        if not aff_refs:
                            for xref in contrib.findall(".//xref"):
                                rid = xref.get("rid", "")
                                if rid.startswith("aff") or rid.startswith("AF"):
                                    ref_text = get_text(xref)
                                    if ref_text:
                                        aff_refs.append(ref_text)

                        # Method 3: Look for sup elements containing numbers
                        if not aff_refs:
                            for sup in contrib.findall(".//sup"):
                                sup_text = get_text(sup)
                                if sup_text and (
                                    sup_text.isdigit() or sup_text in "†‡§¶*"
                                ):
                                    aff_refs.append(sup_text)

                        if aff_refs:
                            author_name += (
                                f'<sup class="xref-aff">{",".join(aff_refs)}</sup>'
                            )

                        if contrib.get("corresp") == "yes":
                            author_name += '<sup class="xref-aff">*</sup>'
                        authors.append(author_name)
        if authors:
            # Check if Authors section should be highlighted
            authors_highlight = (
                highlight_section
                and highlight_section.lower() == "authors"
                and not highlight_found[0]
            )
            if authors_highlight:
                highlight_found[0] = True
            authors_attr = (
                ' id="highlight-target" class="highlighted"'
                if authors_highlight
                else ""
            )
            html_parts.append(
                f'<div class="authors"{authors_attr}>{", ".join(authors)}</div>'
            )

        # Affiliations (with label numbers like "1", "2")
        for aff in root.iter("aff"):
            aff_id = aff.get("id", "")
            label_elem = aff.find("label")
            label = get_text(label_elem) if label_elem is not None else ""
            # Get the rest of the affiliation text, skipping the label element to avoid duplication
            aff_text = render_inline(aff, skip_tags={"label"})
            if aff_text.strip():
                label_html = (
                    f'<sup class="xref-aff">{html_lib.escape(label)}</sup> '
                    if label
                    else ""
                )
                html_parts.append(
                    f'<div class="affiliation" data-xmlid="{aff_id}" id="{aff_id}">{label_html}{aff_text}</div>'
                )

        # DOI
        for aid in root.iter("article-id"):
            if aid.get("pub-id-type") == "doi" and aid.text:
                html_parts.append(
                    f'<div class="doi">DOI: {html_lib.escape(aid.text)}</div>'
                )
                break

        # Abstract
        abstract = root.find(".//abstract")
        if abstract is not None:
            abs_id = abstract.get("id", "")

            # Check if we have a specific xpath pointing to a paragraph INSIDE the abstract
            # If so, don't highlight the abstract div - let the paragraph get highlighted instead
            target_is_inside_abstract = highlight_target_elem is not None and any(
                p is highlight_target_elem for p in abstract.findall(".//p")
            )

            # Check if Abstract div should be highlighted by xml_id
            # Don't highlight by section name if we have a specific target inside
            should_highlight = (
                highlight_xml_id
                and abs_id == highlight_xml_id
                and not highlight_found[0]
            ) and not target_is_inside_abstract

            if should_highlight:
                highlight_found[0] = True
            highlight_attr = ' id="highlight-target"' if should_highlight else ""
            highlight_class = " highlighted" if should_highlight else ""

            toc_entries.append(("abstract-section", "Abstract", 1))
            html_parts.append(
                f'<div id="abstract-section" class="abstract{highlight_class}"{highlight_attr} data-xmlid="{abs_id}">'
            )
            html_parts.append('<div class="abstract-label">Abstract</div>')
            for p in abstract.findall(".//p"):
                p_id = p.get("id", "")
                p_html = render_inline(p)
                if p_html.strip():
                    # Check if this paragraph should be highlighted:
                    # 1. By xml_id match
                    # 2. By element reference (if we resolved xpath to this element)
                    p_highlight = (
                        (highlight_xml_id and p_id == highlight_xml_id)
                        or (
                            highlight_target_elem is not None
                            and p is highlight_target_elem
                        )
                    ) and not highlight_found[0]

                    if p_highlight:
                        highlight_found[0] = True

                    p_attr = ""
                    # If highlighted and we have specific text, try substring highlighting
                    if p_highlight and highlight_text:
                        match_result = fuzzy_find_substring(
                            highlight_text, p_html, threshold=0.90
                        )
                        if match_result:
                            start, end = match_result
                            # Found substring - highlight just that part
                            p_html = (
                                p_html[:start]
                                + '<span id="highlight-target" class="highlighted">'
                                + p_html[start:end]
                                + "</span>"
                                + p_html[end:]
                            )
                        else:
                            # Text match failed but xpath was correct — highlight whole paragraph
                            p_attr = ' id="highlight-target" class="highlighted"'
                    elif p_highlight:
                        # No specific text provided - highlight whole paragraph
                        p_attr = ' id="highlight-target" class="highlighted"'

                    html_parts.append(f'<p{p_attr} data-xmlid="{p_id}">{p_html}</p>')
            html_parts.append("</div>")

        def should_highlight_section(sec_id: str, title_text: str) -> bool:
            """Check if this section should be highlighted.

            Only highlight section headers by xml:id match - never by section name.
            Section name matching was too eager and would highlight "Discussion" header
            when we wanted to highlight a paragraph within Discussion.
            """
            if highlight_found[0]:
                return False  # Only highlight once
            # Only highlight section headers by exact xml:id match
            if highlight_xml_id and sec_id == highlight_xml_id:
                highlight_found[0] = True
                return True
            # NOTE: Removed section name matching - it was highlighting headers
            # instead of the actual cited paragraphs. Use text-based fallback instead.
            return False

        # Body sections
        t0 = profile_time.perf_counter()
        body = root.find(".//body")
        section_counter = [0]  # Use list for mutable counter in nested scope
        if body is not None:
            for sec in body.iter("sec"):
                sec_id = sec.get("id", "")
                title = sec.find("title")

                if title is not None:
                    title_text = get_text(title)
                    # Generate stable section ID for TOC
                    section_counter[0] += 1
                    toc_id = sec_id if sec_id else f"section-{section_counter[0]}"

                    # Determine nesting level
                    parent = sec.getparent()
                    is_top_level = parent == body
                    is_level_2 = not is_top_level and parent is not None and parent.getparent() == body
                    level = 1 if is_top_level else (2 if is_level_2 else 3)

                    # Add to TOC (top-level and level-2 subsections)
                    if level <= 2:
                        toc_entries.append((toc_id, title_text, level))

                    level_class = f" level-{level}" if level > 1 else ""
                    sec_highlight = should_highlight_section(sec_id, title_text)
                    sec_attr = (
                        f' id="highlight-target" class="section-header{level_class} highlighted" data-toc-id="{toc_id}"'
                        if sec_highlight
                        else f' id="{toc_id}" class="section-header{level_class}" data-toc-id="{toc_id}"'
                    )
                    html_parts.append(
                        f'<div{sec_attr} data-xmlid="{sec_id}">{html_lib.escape(title_text)}</div>'
                    )

                for child in sec:
                    child_id = child.get("id", "")
                    # Check if this element should be highlighted:
                    # 1. By xml_id match
                    # 2. By element reference (if we resolved xpath)
                    child_highlight = (
                        highlight_xml_id
                        and child_id == highlight_xml_id
                        and not highlight_found[0]
                    ) or (
                        highlight_target_elem is not None
                        and child is highlight_target_elem
                        and not highlight_found[0]
                    )
                    if child_highlight:
                        highlight_found[0] = True
                    child_attr = (
                        ' id="highlight-target" class="highlighted"'
                        if child_highlight
                        else ""
                    )

                    if child.tag == "p":
                        p_html = render_inline(child)
                        if p_html.strip():
                            # If this is the highlighted element and we have specific text to highlight,
                            # try to find and highlight just that substring within the paragraph
                            if child_highlight and highlight_text:
                                match_result = fuzzy_find_substring(
                                    highlight_text, p_html, threshold=0.90
                                )
                                if match_result:
                                    start, end = match_result
                                    p_html = (
                                        p_html[:start]
                                        + '<span id="highlight-target" class="highlighted">'
                                        + p_html[start:end]
                                        + "</span>"
                                        + p_html[end:]
                                    )
                                    child_attr = ""
                                # else: text match failed but xpath was correct — keep child_attr to highlight whole paragraph
                            html_parts.append(
                                f'<p{child_attr} data-xmlid="{child_id}">{p_html}</p>'
                            )

                    elif child.tag == "fig":
                        label = get_text(child.find("label"))
                        caption_elem = child.find(".//caption")
                        caption_html = (
                            render_inline(caption_elem)
                            if caption_elem is not None
                            else ""
                        )
                        graphic = child.find(".//graphic")

                        fig_class = (
                            "figure highlighted" if child_highlight else "figure"
                        )
                        fig_id_attr = (
                            ' id="highlight-target"' if child_highlight else ""
                        )
                        html_parts.append(
                            f'<div class="{fig_class}"{fig_id_attr} data-xmlid="{child_id}" id="{child_id}">'
                        )
                        html_parts.append(f"<strong>{html_lib.escape(label)}</strong>")

                        # Add image if available
                        if graphic is not None and image_base_url:
                            href = graphic.get("{http://www.w3.org/1999/xlink}href")
                            if href:
                                img_url = (
                                    f"{image_base_url}{href}{image_url_params or ''}"
                                )
                                html_parts.append(
                                    f'<img src="{img_url}" alt="{html_lib.escape(label)}" loading="lazy" />'
                                )

                        if caption_html:
                            html_parts.append(
                                f'<div class="figure-caption">{caption_html}</div>'
                            )
                        html_parts.append("</div>")

                    elif child.tag == "table-wrap":
                        label = get_text(child.find("label"))
                        caption_elem = child.find(".//caption")
                        caption_html = (
                            render_inline(caption_elem)
                            if caption_elem is not None
                            else ""
                        )
                        tab_class = (
                            "table-wrap highlighted"
                            if child_highlight
                            else "table-wrap"
                        )
                        tab_id_attr = (
                            ' id="highlight-target"' if child_highlight else ""
                        )
                        html_parts.append(
                            f'<div class="{tab_class}"{tab_id_attr} data-xmlid="{child_id}" id="{child_id}">'
                        )
                        html_parts.append(f"<strong>{html_lib.escape(label)}</strong>")
                        if caption_html:
                            html_parts.append(f" {caption_html}")
                        html_parts.append("</div>")

        # Supplementary materials
        for supp in root.findall(".//supplementary-material"):
            supp_id = supp.get("id", "")
            label = get_text(supp.find("label"))
            caption = get_text(supp.find(".//caption"))
            supp_highlight = highlight_xml_id and supp_id == highlight_xml_id
            supp_class = "supplement highlighted" if supp_highlight else "supplement"
            supp_attr = ' id="highlight-target"' if supp_highlight else ""
            html_parts.append(
                f'<div class="{supp_class}"{supp_attr} data-xmlid="{supp_id}">'
            )
            html_parts.append(
                f"<strong>{html_lib.escape(label)}</strong> {html_lib.escape(caption)}"
            )
            html_parts.append("</div>")
        render_timings["body_sections"] = (profile_time.perf_counter() - t0) * 1000

        # References
        t0 = profile_time.perf_counter()
        refs = root.findall(".//ref-list//ref")
        if refs:
            html_parts.append('<div class="section-header">References</div>')
            for ref in refs:
                ref_id = ref.get("id", "")
                label = get_text(ref.find("label"))
                cit = ref.find(".//mixed-citation")
                if cit is None:
                    cit = ref.find(".//element-citation")
                if cit is None:
                    cit = ref
                # Render citation with italics, links, etc.
                cit_html = render_inline(cit)
                ref_highlight = highlight_xml_id and ref_id == highlight_xml_id
                ref_class = "reference highlighted" if ref_highlight else "reference"
                ref_attr = ' id="highlight-target"' if ref_highlight else ""
                html_parts.append(
                    f'<div class="{ref_class}"{ref_attr} data-xmlid="{ref_id}" id="{ref_id}">{html_lib.escape(label)} {cit_html}</div>'
                )

        # Acknowledgments
        for ack in root.iter("ack"):
            ack_id = ack.get("id", "")
            ack_highlight = highlight_xml_id and ack_id == highlight_xml_id
            ack_class = "ack highlighted" if ack_highlight else "ack"
            ack_attr = ' id="highlight-target"' if ack_highlight else ""
            html_parts.append(
                f'<div class="{ack_class}"{ack_attr} data-xmlid="{ack_id}">'
            )
            html_parts.append(
                '<div class="section-header" style="margin-top:0">Acknowledgments</div>'
            )
            for p in ack.findall(".//p"):
                p_html = render_inline(p)
                if p_html.strip():
                    html_parts.append(f"<p>{p_html}</p>")
            html_parts.append("</div>")

        # Close main content area
        html_parts.append("</main>")

        # Generate TOC JavaScript
        toc_items_js = ",".join(
            f'{{id:"{html_lib.escape(tid)}",title:"{html_lib.escape(ttl)}",level:{lvl}}}'
            for tid, ttl, lvl in toc_entries
        )
        html_parts.append("<script>")
        html_parts.append("(function() {")
        html_parts.append(f"  var tocItems = [{toc_items_js}];")
        html_parts.append("  var nav = document.getElementById('toc-nav');")
        html_parts.append("  if (nav && tocItems.length > 0) {")
        html_parts.append("    var html = '<div class=\"toc-title\">Contents</div>';")
        html_parts.append("    tocItems.forEach(function(item) {")
        html_parts.append("      var cls = 'toc-item' + (item.level > 1 ? ' toc-level-' + item.level : '');")
        html_parts.append("      html += '<a class=\"' + cls + '\" data-target=\"' + item.id + '\">' + item.title + '</a>';")
        html_parts.append("    });")
        # Use click handlers with scrollIntoView instead of href anchors (avoids iframe navigation issues)
        html_parts.append("    nav.addEventListener('click', function(e) {")
        html_parts.append("      var link = e.target.closest('.toc-item');")
        html_parts.append("      if (link) {")
        html_parts.append("        e.preventDefault();")
        html_parts.append("        var target = document.getElementById(link.getAttribute('data-target'));")
        html_parts.append("        if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start', container: 'nearest' });")
        html_parts.append("      }")
        html_parts.append("    });")
        html_parts.append("    nav.innerHTML = html;")
        html_parts.append("  } else if (nav) { nav.style.display = 'none'; }")
        # Scroll to highlight target
        html_parts.append("  var h = document.getElementById('highlight-target') || document.querySelector('.highlighted');")
        html_parts.append("  if (h) h.scrollIntoView({ behavior: 'instant', block: 'center', container: 'nearest' });")
        # Highlight active TOC item on scroll
        html_parts.append("  var sections = document.querySelectorAll('.section-header[data-toc-id]');")
        html_parts.append("  var tocLinks = document.querySelectorAll('.toc-item');")
        html_parts.append("  if (sections.length > 0 && tocLinks.length > 0) {")
        html_parts.append("    var observer = new IntersectionObserver(function(entries) {")
        html_parts.append("      entries.forEach(function(entry) {")
        html_parts.append("        if (entry.isIntersecting) {")
        html_parts.append("          var id = entry.target.getAttribute('data-toc-id');")
        html_parts.append("          tocLinks.forEach(function(link) {")
        html_parts.append("            link.classList.toggle('active', link.getAttribute('data-target') === id);")
        html_parts.append("          });")
        html_parts.append("        }")
        html_parts.append("      });")
        html_parts.append("    }, { rootMargin: '-20% 0px -70% 0px' });")
        html_parts.append("    sections.forEach(function(sec) { observer.observe(sec); });")
        html_parts.append("  }")
        html_parts.append("})();")
        html_parts.append("</script>")

        html_parts.append("</body></html>")
        render_timings["references"] = (profile_time.perf_counter() - t0) * 1000

        t0 = profile_time.perf_counter()
        html = "\n".join(html_parts)
        render_timings["join_html"] = (profile_time.perf_counter() - t0) * 1000

        # Fallback: if highlight_text was provided but not found at the expected
        # location, scan ALL paragraphs in the rendered HTML for the substring.
        if highlight_text and not highlight_found[0]:
            logger.info("[HIGHLIGHT FALLBACK] Target not found at xpath, scanning full document")
            # Search through <p ...>...</p> tags for the highlight_text
            p_pattern = re.compile(r'(<p[^>]*>)(.*?)(</p>)', re.DOTALL)
            def _try_highlight_in_p(m):
                if highlight_found[0]:
                    return m.group(0)  # Already found, skip
                tag_open, inner, tag_close = m.group(1), m.group(2), m.group(3)
                match_result = fuzzy_find_substring(highlight_text, inner, threshold=0.85)
                if match_result:
                    highlight_found[0] = True
                    start, end = match_result
                    new_inner = (
                        inner[:start]
                        + '<span id="highlight-target" class="highlighted">'
                        + inner[start:end]
                        + '</span>'
                        + inner[end:]
                    )
                    # Add highlight-target to the <p> tag for scrolling
                    return tag_open + new_inner + tag_close
                return m.group(0)
            html = p_pattern.sub(_try_highlight_in_p, html)
            if highlight_found[0]:
                logger.info("[HIGHLIGHT FALLBACK] Found match in full document scan")

        render_timings["total"] = (profile_time.perf_counter() - t_render_start) * 1000
        logger.info(f"[XML RENDER PROFILE] Timings (ms): {render_timings}")

        # NOTE: xpath highlighting is now handled during rendering by resolving
        # the xpath to an element reference BEFORE rendering starts. This is more
        # precise than text searching after rendering.

        # NOTE: Text-based fallback removed - it was a form of guessing/fuzzy matching.
        # Highlighting should only happen via:
        # 1. xml_id match (precise element ID)
        # 2. xpath match (precise element path resolved before rendering)

        return html

    except Exception as e:
        logger.error(f"Failed to render XML to HTML: {e}")
        return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
        <style>body {{ font-family: sans-serif; padding: 40px; color: #c00; }}</style>
        </head><body><h1>Error rendering XML</h1><p>{html_lib.escape(str(e))}</p></body></html>"""


class PapersToolModule(ToolModule):
    """Papers database tools module (legacy ToolModule interface).

    Provides access to 453K+ preprint articles from bioRxiv and medRxiv with
    ~70M content blocks for full-text search and analysis.

    Key tables:
    - documents: Article metadata (document_id, source, doi, title, authors, month_year)
    - content_blocks: Full-text content with citation_info (xpath, xml_id, source_path)
    """

    def __init__(self):
        super().__init__()
        self._tools = []
        self._handlers = {}
        self._conn = None
        self._extract_tools()

    def get_name(self) -> str:
        return "biomedrxiv"

    def get_description(self) -> str:
        return "BioMedRxiv database with 453K+ preprint articles"

    def _get_connection(self):
        """Get or create database connection."""
        if self._conn is None or self._conn.closed:
            try:
                self._conn = get_biomedrxiv_connection()
                logger.info("Connected to biomedrxiv database")
            except Exception as e:
                logger.error(f"Failed to connect to biomedrxiv: {e}")
                raise
        return self._conn

    def _reset_connection(self):
        """Reset connection after error."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None

    def _extract_tools(self):
        """Define and register all tools."""
        logger.info("Extracting BioMedRxiv tools...")

        tools_def = [
            {
                "name": "biomedrxiv_find_documents",
                "description": """Find relevant documents in the bioRxiv/medRxiv database.

⚠️ IMPORTANT: This database has 453K+ documents and 70M+ content blocks.
ALWAYS start by finding relevant documents first, then drill down.

SEARCH STRATEGIES (in order of efficiency):
1. DOI lookup (fastest): If you have a DOI, use it
2. Title search: Search by exact or partial title
3. Author search: Find papers by author name
4. Keyword search: Full-text search across titles (use sparingly)

FILTERS:
- source: 'biorxiv' or 'medrxiv' (default: both)
- month_year: e.g., 'August_2025', 'January_2024'

RETURNS:
- document_id (UUID), doi, title, authors, source, month_year
- Use document_id to get full content via biomedrxiv_get_content

Example queries:
- Find by DOI: doi="10.1101/2024.01.15.123456"
- Find by title: title_contains="CRISPR cancer therapy"
- Find by author: author_contains="Smith"
- Combined: title_contains="COVID", source="medrxiv", limit=20""",
                "function": self._find_documents,
                "schema": {
                    "type": "object",
                    "properties": {
                        "doi": {
                            "type": "string",
                            "description": "Exact DOI to look up (fastest)",
                        },
                        "title_contains": {
                            "type": "string",
                            "description": "Search for documents with title containing this text (case-insensitive)",
                        },
                        "author_contains": {
                            "type": "string",
                            "description": "Search for documents with author containing this name",
                        },
                        "source": {
                            "type": "string",
                            "enum": ["biorxiv", "medrxiv"],
                            "description": "Filter by source (default: both)",
                        },
                        "month_year": {
                            "type": "string",
                            "description": "Filter by month_year (e.g., 'August_2025')",
                        },
                        "limit": {
                            "type": "number",
                            "description": "Max results (default: 20, max: 100)",
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "biomedrxiv_search_documents",
                "description": """Document search across titles and abstracts.

Searches 453K documents using relevance ranking.

HOW IT WORKS:
- Combines title + abstract for ranking
- Returns documents ordered by relevance score

USE THIS WHEN:
- You need to find papers on a topic
- You want relevance-ranked results
- You're searching for general concepts or techniques

RETURNS:
- document_id, title, doi, source, relevance score
- Highlighted snippet from abstract

Example: biomedrxiv_search_documents(query="protein stability prediction machine learning", limit=20)""",
                "function": self._search_documents_ranked,
                "schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query",
                        },
                        "source": {
                            "type": "string",
                            "enum": ["biorxiv", "medrxiv"],
                            "description": "Filter by source (optional)",
                        },
                        "limit": {
                            "type": "number",
                            "description": "Max results (default: 20, max: 50)",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "biomedrxiv_get_content",
                "description": """Get full content blocks for a specific document.

After finding documents with biomedrxiv_find_documents, use this to get the
full text content with citation metadata.

RETURNS content blocks with:
- content: The actual text
- block_type: 'title', 'paragraph', 'figure', 'table', 'section_header', etc.
- section: Section name (e.g., 'Abstract', 'Introduction', 'Methods')
- citation_info: Metadata for citing (xpath, source_path, xml_id)

FILTERING:
- section_contains: Filter by section name (e.g., 'Methods', 'Results')
- block_type: Filter by block type. Use 'abstract' to get abstract paragraphs.
- content_contains: Search within document content (regex supported)

The document URL is: https://www.biorxiv.org/content/{doi} or
https://www.medrxiv.org/content/{doi}""",
                "function": self._get_content,
                "schema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "Document UUID from biomedrxiv_find_documents",
                        },
                        "section_contains": {
                            "type": "string",
                            "description": "Filter by section name pattern",
                        },
                        "block_type": {
                            "type": "string",
                            "enum": [
                                "abstract",
                                "title",
                                "authors",
                                "affiliation",
                                "paragraph",
                                "figure",
                                "table",
                                "section_header",
                                "reference",
                                "metadata",
                                "category",
                                "supplement",
                                "date",
                                "correspondence",
                                "competing_interests",
                            ],
                            "description": "Filter by block type. Note: 'abstract' filters by section name, not block_type.",
                        },
                        "content_contains": {
                            "type": "string",
                            "description": "Search for content containing this pattern (regex)",
                        },
                        "limit": {
                            "type": "number",
                            "description": "Max blocks to return (default: 50)",
                        },
                    },
                    "required": ["document_id"],
                },
            },
            {
                "name": "biomedrxiv_search_content",
                "description": """Full-text search across all content blocks.

⚠️ CAUTION: This searches 70M+ rows. Use specific queries and limits!

BEST PRACTICES:
1. First use biomedrxiv_find_documents to identify relevant papers
2. Then use biomedrxiv_get_content for those specific documents
3. Only use this tool for cross-document searches when necessary

The search uses PostgreSQL full-text search (tsquery).

FILTERS:
- source: Limit to 'biorxiv' or 'medrxiv'
- block_type: Limit to specific block types

Returns matching blocks with document context.""",
                "function": self._search_content,
                "schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (will be converted to tsquery)",
                        },
                        "source": {
                            "type": "string",
                            "enum": ["biorxiv", "medrxiv"],
                            "description": "Filter by source",
                        },
                        "block_type": {
                            "type": "string",
                            "description": "Filter by block type",
                        },
                        "limit": {
                            "type": "number",
                            "description": "Max results (default: 20, max: 50)",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "biomedrxiv_search_abstracts",
                "description": """Search across paper abstracts using pattern matching.

🚀 FASTER than biomedrxiv_search_content - only searches abstract sections.
Uses case-insensitive pattern matching (ILIKE) - good for exact terms, 
acronyms, and hyphenated words like "CT-FFR", "CRISPR", "COVID-19".

Use this to find papers discussing specific topics, methods, or findings 
mentioned in their abstracts.

Returns matching abstract blocks with document context.""",
                "function": self._search_abstracts,
                "schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search pattern (exact text, acronyms, or phrase - case insensitive)",
                        },
                        "source": {
                            "type": "string",
                            "enum": ["biorxiv", "medrxiv"],
                            "description": "Filter by source (optional)",
                        },
                        "limit": {
                            "type": "number",
                            "description": "Max results (default: 20, max: 50)",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "biomedrxiv_search_methods",
                "description": """Search across Methods sections using pattern matching.

🚀 FASTER than biomedrxiv_search_content - only searches Methods sections.
Uses case-insensitive pattern matching (ILIKE) - good for exact terms,
acronyms, and hyphenated words like "CT-FFR", "CRISPR", "PyMOL".

Use this to find papers using specific:
- Software packages and tools (exact names)
- Experimental techniques
- Datasets and databases
- Computational methods

Returns matching Methods blocks with document context.""",
                "function": self._search_methods,
                "schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search pattern (exact text, acronyms, or phrase - case insensitive)",
                        },
                        "source": {
                            "type": "string",
                            "enum": ["biorxiv", "medrxiv"],
                            "description": "Filter by source (optional)",
                        },
                        "limit": {
                            "type": "number",
                            "description": "Max results (default: 20, max: 50)",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "biomedrxiv_search_results",
                "description": """Search across Results sections using pattern matching.

🚀 FASTER than biomedrxiv_search_content - only searches Results sections.
Uses case-insensitive pattern matching (ILIKE) - good for exact terms,
specific measurements, or findings.

Use this to find papers with specific findings, measurements, or outcomes.

Returns matching Results blocks with document context.""",
                "function": self._search_results,
                "schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search pattern (exact text or phrase - case insensitive)",
                        },
                        "source": {
                            "type": "string",
                            "enum": ["biorxiv", "medrxiv"],
                            "description": "Filter by source (optional)",
                        },
                        "limit": {
                            "type": "number",
                            "description": "Max results (default: 20, max: 50)",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "biomedrxiv_get_document_url",
                "description": """Get the direct URL to view a document on bioRxiv/medRxiv.

Returns the canonical URL for citing or viewing the paper in a browser.""",
                "function": self._get_document_url,
                "schema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "Document UUID",
                        },
                    },
                    "required": ["document_id"],
                },
            },
            {
                "name": "biomedrxiv_render_citation",
                "description": """Render a content block citation with source context.

Takes a content block's citation_info and renders:
- For XML sources: Highlights the specific element in the rendered article
- For PDF sources: Returns the page and bounding box info

Use this to verify citations and show users the source context.""",
                "function": self._render_citation,
                "schema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "Document UUID",
                        },
                        "line_number": {
                            "type": "number",
                            "description": "Line number of the content block",
                        },
                    },
                    "required": ["document_id", "line_number"],
                },
            },
            {
                "name": "biomedrxiv_stats",
                "description": """Get database statistics.

Returns counts of documents and content blocks by source.""",
                "function": self._get_stats,
                "schema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "biomedrxiv_es_search",
                "description": """Full-text search using Elasticsearch.

Fast full-text search across 70M+ content blocks with relevance scoring.

PERFORMANCE:
- Typical query: 50-200ms
- Handles complex boolean queries efficiently
- No size limits on content

QUERY SYNTAX:
- Simple: "protein folding" (matches both words)
- Phrase: '"machine learning"' (exact phrase with quotes)
- Boolean: "cancer AND therapy NOT chemotherapy"
- Wildcards: "neuro*" (matches neuroscience, neurology, etc.)
- Fuzzy: "protien~" (matches "protein" despite typo)

FILTERS:
- source: 'biorxiv' or 'medrxiv'
- section: Filter by section name (e.g., "Methods", "Results")
- block_type: Filter by type ("paragraph", "figure", "table", etc.)

Returns ranked results with relevance scores and highlighted snippets.""",
                "function": self._es_search,
                "schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query. Supports boolean operators (AND, OR, NOT), phrases in quotes, wildcards (*), and fuzzy matching (~).",
                        },
                        "source": {
                            "type": "string",
                            "enum": ["biorxiv", "medrxiv"],
                            "description": "Filter by source (optional)",
                        },
                        "section": {
                            "type": "string",
                            "description": "Filter by section (e.g., 'Abstract', 'Methods', 'Results')",
                        },
                        "block_type": {
                            "type": "string",
                            "description": "Filter by block type",
                        },
                        "limit": {
                            "type": "number",
                            "description": "Max results (default: 20, max: 100)",
                        },
                    },
                    "required": ["query"],
                },
            },
        ]

        # Create tools and handlers
        for tool_def in tools_def:
            try:
                tool = Tool(
                    name=tool_def["name"],
                    description=tool_def["description"],
                    inputSchema=tool_def["schema"],
                )
                tool._meta = {"service": "default", "async": False}

                self._tools.append(tool)
                self._handlers[tool_def["name"]] = self._create_handler(
                    tool_def["function"]
                )

                logger.debug(f"Registered tool: {tool_def['name']}")
            except Exception as e:
                logger.error(f"Error registering tool {tool_def['name']}: {e}")

        logger.info(f"Extracted {len(self._tools)} BioMedRxiv tools")

    def _create_handler(self, func: Callable) -> Callable:
        """Create async handler wrapper."""

        async def handler(
            arguments: dict[str, Any],
            session_id: str = "default",
            api_key: str | None = None,
            agent_id: str | None = None,
        ):
            try:
                arguments["session_id"] = session_id
                arguments["api_key"] = api_key
                arguments["agent_id"] = agent_id

                result = await func(**arguments)

                if isinstance(result, list) and all(
                    isinstance(item, (TextContent, ImageContent)) for item in result
                ):
                    return result

                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                error_result = {
                    "status": "error",
                    "message": f"Error: {str(e)}",
                }
                return [
                    TextContent(type="text", text=json.dumps(error_result, indent=2))
                ]

        return handler

    def get_tools(self) -> list[Tool]:
        return self._tools

    def get_handlers(self) -> dict[str, Callable]:
        return self._handlers

    def validate_config(self) -> list[str]:
        errors = super().validate_config()
        if not os.getenv("BIOMEDRXIV_DB_URL") and not os.getenv(
            "BIOMEDRXIV_DB_PASSWORD"
        ):
            errors.append("Neither BIOMEDRXIV_DB_URL nor BIOMEDRXIV_DB_PASSWORD is set")
        return errors

    # =========================================================================
    # Tool Implementations
    # =========================================================================

    async def _find_documents(
        self,
        doi: str | None = None,
        title_contains: str | None = None,
        author_contains: str | None = None,
        source: str | None = None,
        month_year: str | None = None,
        limit: int = 20,
        session_id: str = "default",
        api_key: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Find documents by various criteria."""
        start_time = time.perf_counter()
        limit = min(limit or 20, 100)

        conditions = []
        params = []

        if doi:
            conditions.append("doi = %s")
            params.append(doi)

        if title_contains:
            conditions.append("title ILIKE %s")
            params.append(f"%{title_contains}%")

        if author_contains:
            conditions.append("authors ILIKE %s")
            params.append(f"%{author_contains}%")

        if source:
            conditions.append("source = %s")
            params.append(source)

        if month_year:
            conditions.append("month_year = %s")
            params.append(month_year)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        params.append(limit)

        query = f"""
            SELECT document_id, source, month_year, doi, title, authors, total_blocks
            FROM documents
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT %s
        """

        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            documents = []
            for row in rows:
                doc_id, src, my, d, t, a, tb = row
                documents.append(
                    {
                        "document_id": str(doc_id),
                        "source": src,
                        "month_year": my,
                        "doi": d,
                        "title": t,
                        "authors": a[:200] + "..." if a and len(a) > 200 else a,
                        "total_blocks": tb,
                        "url": f"https://www.{src}.org/content/{d}" if d else None,
                    }
                )

            return {
                "status": "success",
                "count": len(documents),
                "query_time_ms": round(elapsed_ms, 1),
                "documents": documents,
                "hint": "Use document_id with biomedrxiv_get_content to get full text",
            }

        except Exception as e:
            self._reset_connection()
            return {"status": "error", "message": str(e)}

    async def _search_documents_ranked(
        self,
        query: str,
        source: str | None = None,
        limit: int = 20,
        session_id: str = "default",
        api_key: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Ranked document search across titles and abstracts."""
        start_time = time.perf_counter()
        limit = min(limit or 20, 50)

        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'documents' AND column_name = 'search_vector'
                """
                )
                has_search_vector = cur.fetchone() is not None

            if not has_search_vector:
                return await self._find_documents(
                    title_contains=query,
                    source=source,
                    limit=limit,
                    session_id=session_id,
                    api_key=api_key,
                    agent_id=agent_id,
                )

            # Build query with optional source filter
            # Use websearch_to_tsquery for better handling of:
            # - Hyphens (e.g., "GLP-1" stays together)
            # - Quoted phrases (e.g., "machine learning")
            # - OR operator (e.g., "cancer or tumor")
            if source:
                sql = """
                    SELECT 
                        document_id,
                        title,
                        doi,
                        source,
                        authors,
                        ts_rank_cd(search_vector, query) AS rank,
                        ts_headline('english', COALESCE(abstract_text, title), query,
                            'MaxFragments=2, MaxWords=40, MinWords=15') AS snippet
                    FROM documents, websearch_to_tsquery('english', %s) AS query
                    WHERE search_vector @@ query AND source = %s
                    ORDER BY rank DESC
                    LIMIT %s
                """
                params = [query, source, limit]
            else:
                sql = """
                    SELECT 
                        document_id,
                        title,
                        doi,
                        source,
                        authors,
                        ts_rank_cd(search_vector, query) AS rank,
                        ts_headline('english', COALESCE(abstract_text, title), query,
                            'MaxFragments=2, MaxWords=40, MinWords=15') AS snippet
                    FROM documents, websearch_to_tsquery('english', %s) AS query
                    WHERE search_vector @@ query
                    ORDER BY rank DESC
                    LIMIT %s
                """
                params = [query, limit]

            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            documents = []
            for row in rows:
                doc_id, title, doi, src, authors, rank, snippet = row
                documents.append(
                    {
                        "document_id": str(doc_id),
                        "title": title,
                        "doi": doi,
                        "source": src,
                        "authors": (
                            authors[:150] + "..."
                            if authors and len(authors) > 150
                            else authors
                        ),
                        "relevance_score": round(float(rank), 4),
                        "snippet": snippet,
                        "url": f"https://www.{src}.org/content/{doi}" if doi else None,
                    }
                )

            return {
                "status": "success",
                "query": query,
                "count": len(documents),
                "query_time_ms": round(elapsed_ms, 1),
                "documents": documents,
                "hint": "Use document_id with biomedrxiv_get_content to get full text",
            }

        except Exception as e:
            self._reset_connection()
            return {"status": "error", "message": str(e)}

    async def _get_content(
        self,
        document_id: str,
        section_contains: str | None = None,
        block_type: str | None = None,
        content_contains: str | None = None,
        limit: int = 50,
        session_id: str = "default",
        api_key: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Get content blocks for a document."""
        start_time = time.perf_counter()
        limit = min(limit or 50, 200)

        conditions = ["document_id = %s"]
        params = [document_id]

        if section_contains:
            conditions.append("section ILIKE %s")
            params.append(f"%{section_contains}%")

        if block_type:
            # Special case: 'abstract' is a section, not a block_type
            if block_type.lower() == "abstract":
                conditions.append("section ILIKE %s")
                params.append("%abstract%")
            else:
                conditions.append("block_type = %s")
                params.append(block_type)

        if content_contains:
            conditions.append("content ~* %s")
            params.append(content_contains)

        params.append(limit)
        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT line_number, content, block_type, section, citation_info
            FROM content_blocks
            WHERE {where_clause}
            ORDER BY line_number
            LIMIT %s
        """

        try:
            conn = self._get_connection()

            # First get document metadata
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT source, doi, title FROM documents WHERE document_id = %s",
                    (document_id,),
                )
                doc_row = cur.fetchone()

            if not doc_row:
                return {
                    "status": "error",
                    "message": f"Document {document_id} not found",
                }

            source, doi, title = doc_row

            # Get content blocks
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            blocks = []
            for row in rows:
                line_num, content, btype, section, citation_info = row
                blocks.append(
                    {
                        "line_number": line_num,
                        "content": content,
                        "block_type": btype,
                        "section": section,
                        "citation_info": citation_info,
                    }
                )

            return {
                "status": "success",
                "document_id": document_id,
                "source": source,
                "doi": doi,
                "title": title,
                "url": f"https://www.{source}.org/content/{doi}" if doi else None,
                "block_count": len(blocks),
                "query_time_ms": round(elapsed_ms, 1),
                "blocks": blocks,
            }

        except Exception as e:
            self._reset_connection()
            return {"status": "error", "message": str(e)}

    async def _search_content(
        self,
        query: str,
        source: str | None = None,
        block_type: str | None = None,
        limit: int = 20,
        session_id: str = "default",
        api_key: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Full-text search across content blocks."""
        start_time = time.perf_counter()
        limit = min(limit or 20, 50)

        # Convert query to tsquery format
        terms = query.split()
        ts_query = " & ".join(terms)

        # Use pre-computed content_tsv if available (much faster with GIN index)
        # Falls back to on-the-fly tsvector if column doesn't exist
        # Also filter out oversized content (PostgreSQL tsvector limit is ~1MB)
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'content_blocks' AND column_name = 'content_tsv'
                """
                )
                has_tsv_column = cur.fetchone() is not None
        except Exception:
            has_tsv_column = False

        if has_tsv_column:
            # Fast path: use pre-computed tsvector with GIN index
            conditions = ["content_tsv @@ to_tsquery('english', %s)"]
        else:
            # Slow path: compute tsvector on the fly, skip oversized content
            conditions = [
                "LENGTH(content) < 1000000",
                "to_tsvector('english', content) @@ to_tsquery('english', %s)",
            ]
        params = [ts_query]

        if source:
            # Need to join with documents table
            join_clause = "JOIN documents d ON cb.document_id = d.document_id::text"
            conditions.append("d.source = %s")
            params.append(source)
        else:
            join_clause = ""

        if block_type:
            conditions.append("cb.block_type = %s")
            params.append(block_type)

        params.append(limit)
        where_clause = " AND ".join(conditions)

        # Note: This query may be slow on first run before indexes are created
        sql = f"""
            SELECT cb.document_id, cb.line_number, cb.content, cb.block_type, 
                   cb.section, cb.citation_info
            FROM content_blocks cb
            {join_clause}
            WHERE {where_clause}
            LIMIT %s
        """

        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            results = []
            for row in rows:
                doc_id, line_num, content, btype, section, citation_info = row
                results.append(
                    {
                        "document_id": doc_id,
                        "line_number": line_num,
                        "content": (
                            content[:500] + "..." if len(content) > 500 else content
                        ),
                        "block_type": btype,
                        "section": section,
                    }
                )

            return {
                "status": "success",
                "query": query,
                "count": len(results),
                "query_time_ms": round(elapsed_ms, 1),
                "results": results,
                "hint": "Use biomedrxiv_get_content with document_id for full context",
                "warning": (
                    "This is a slow query - consider using section-specific searches instead"
                    if elapsed_ms > 5000
                    else None
                ),
            }

        except Exception as e:
            self._reset_connection()
            return {"status": "error", "message": str(e)}

    async def _search_section(
        self,
        query: str,
        section_pattern: str,
        source: str | None = None,
        limit: int = 20,
        session_id: str = "default",
        api_key: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Internal method: Search within a specific section type using regex/ILIKE."""
        start_time = time.perf_counter()
        limit = min(limit or 20, 50)

        # Use ILIKE for case-insensitive pattern matching (better for acronyms, hyphenated terms)
        # The query is used as a pattern - spaces become wildcards for flexibility
        search_pattern = f"%{query}%"

        # Build conditions using ILIKE for content (regex-like pattern matching)
        conditions = ["cb.content ILIKE %s", "cb.section ILIKE %s"]
        params = [search_pattern, f"%{section_pattern}%"]

        join_clause = "JOIN documents d ON cb.document_id = d.document_id::text"

        if source:
            conditions.append("d.source = %s")
            params.append(source)

        params.append(limit)
        where_clause = " AND ".join(conditions)

        sql = f"""
            SELECT cb.document_id, cb.line_number, cb.content, cb.block_type, 
                   cb.section, cb.citation_info, d.doi, d.title, d.source, d.authors, d.month_year
            FROM content_blocks cb
            {join_clause}
            WHERE {where_clause}
            LIMIT %s
        """

        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            results = []
            for row in rows:
                (
                    doc_id,
                    line_num,
                    content,
                    btype,
                    section,
                    citation_info,
                    doi,
                    title,
                    src,
                    authors,
                    month_year,
                ) = row
                # Format authors as "FirstAuthor et al." for brevity
                short_authors = None
                if authors:
                    author_list = [a.strip() for a in authors.split(",")]
                    if len(author_list) > 1:
                        short_authors = f"{author_list[0]}, et al."
                    else:
                        short_authors = author_list[0]

                results.append(
                    {
                        "document_id": doc_id,
                        "line_number": line_num,
                        "content": (
                            content[:500] + "..." if len(content) > 500 else content
                        ),
                        "block_type": btype,
                        "section": section,
                        "citation_info": citation_info,
                        "doi": doi,
                        "title": (
                            title[:100] + "..." if title and len(title) > 100 else title
                        ),
                        "source": src,
                        "authors": short_authors,
                        "month_year": month_year,
                        "url": f"https://www.{src}.org/content/{doi}" if doi else None,
                    }
                )

            return {
                "status": "success",
                "query": query,
                "section_filter": section_pattern,
                "count": len(results),
                "query_time_ms": round(elapsed_ms, 1),
                "results": results,
            }

        except Exception as e:
            self._reset_connection()
            return {"status": "error", "message": str(e)}

    async def _search_abstracts(
        self,
        query: str,
        source: str | None = None,
        limit: int = 20,
        session_id: str = "default",
        api_key: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Search across abstract sections only."""
        return await self._search_section(
            query=query,
            section_pattern="abstract",
            source=source,
            limit=limit,
            session_id=session_id,
            api_key=api_key,
            agent_id=agent_id,
        )

    async def _search_methods(
        self,
        query: str,
        source: str | None = None,
        limit: int = 20,
        session_id: str = "default",
        api_key: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Search across Methods sections only."""
        return await self._search_section(
            query=query,
            section_pattern="method",
            source=source,
            limit=limit,
            session_id=session_id,
            api_key=api_key,
            agent_id=agent_id,
        )

    async def _search_results(
        self,
        query: str,
        source: str | None = None,
        limit: int = 20,
        session_id: str = "default",
        api_key: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Search across Results sections only."""
        return await self._search_section(
            query=query,
            section_pattern="result",
            source=source,
            limit=limit,
            session_id=session_id,
            api_key=api_key,
            agent_id=agent_id,
        )

    async def _get_document_url(
        self,
        document_id: str,
        session_id: str = "default",
        api_key: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Get the URL to view the document."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT source, doi FROM documents WHERE document_id = %s",
                    (document_id,),
                )
                row = cur.fetchone()

            if not row:
                return {
                    "status": "error",
                    "message": f"Document {document_id} not found",
                }

            source, doi = row

            if doi:
                url = f"https://www.{source}.org/content/{doi}"
                return {
                    "status": "success",
                    "document_id": document_id,
                    "url": url,
                    "source": source,
                    "doi": doi,
                }
            else:
                return {
                    "status": "error",
                    "message": "Document has no DOI",
                    "document_id": document_id,
                }

        except Exception as e:
            self._reset_connection()
            return {"status": "error", "message": str(e)}

    async def _render_citation(
        self,
        document_id: str,
        line_number: int,
        session_id: str = "default",
        api_key: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Render citation with source context."""
        try:
            conn = self._get_connection()

            # Get the content block
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT content, block_type, section, citation_info
                       FROM content_blocks
                       WHERE document_id = %s AND line_number = %s""",
                    (document_id, line_number),
                )
                row = cur.fetchone()

            if not row:
                return {
                    "status": "error",
                    "message": f"Content block not found: doc={document_id}, line={line_number}",
                }

            content, block_type, section, citation_info = row
            citation_info = citation_info or {}

            source_type = citation_info.get("source_type", "")
            source_path = citation_info.get("source_path", "")

            result = {
                "status": "success",
                "document_id": document_id,
                "line_number": line_number,
                "content": content,
                "block_type": block_type,
                "section": section,
                "citation_info": citation_info,
            }

            # For XML sources, render the document with highlighting
            if "xml" in source_type and source_path:
                xpath = citation_info.get("xpath")

                # Download and render XML
                xml_content = download_xml_from_gcs(source_path)
                if xml_content:
                    html = render_xml_to_html(xml_content, highlight_xpath=xpath)
                    result["rendered_html_preview"] = (
                        html[:2000] + "..." if len(html) > 2000 else html
                    )
                    result["source_format"] = "xml"
                else:
                    result["warning"] = "Could not download source XML"

            # For PDF sources, provide page/bbox info
            elif "pdf" in source_type:
                result["source_format"] = "pdf"
                result["page"] = citation_info.get("page")
                result["bbox"] = citation_info.get("bbox")
                result["hint"] = "Use page and bbox to locate content in the PDF"

            # Get document URL
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT source, doi FROM documents WHERE document_id = %s",
                    (document_id,),
                )
                doc_row = cur.fetchone()

            if doc_row:
                source, doi = doc_row
                if doi:
                    result["document_url"] = f"https://www.{source}.org/content/{doi}"

            return result

        except Exception as e:
            self._reset_connection()
            return {"status": "error", "message": str(e)}

    async def _get_stats(
        self,
        session_id: str = "default",
        api_key: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Get database statistics."""
        try:
            conn = self._get_connection()

            stats = {}

            # Document counts
            with conn.cursor() as cur:
                cur.execute("SELECT source, COUNT(*) FROM documents GROUP BY source")
                for row in cur.fetchall():
                    stats[f"{row[0]}_documents"] = row[1]

                cur.execute("SELECT COUNT(*) FROM documents")
                stats["total_documents"] = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM content_blocks")
                stats["total_content_blocks"] = cur.fetchone()[0]

            os_client = _get_es_client()
            if os_client:
                try:
                    os_count = os_client.count(index=OS_INDEX_NAME)
                    stats["opensearch_documents"] = os_count.get("count", 0)
                    stats["opensearch_available"] = True
                except Exception:
                    stats["opensearch_available"] = False
            else:
                stats["opensearch_available"] = False

            return {
                "status": "success",
                "statistics": stats,
            }

        except Exception as e:
            self._reset_connection()
            return {"status": "error", "message": str(e)}

    async def _es_search(
        self,
        query: str,
        source: str | None = None,
        section: str | None = None,
        block_type: str | None = None,
        limit: int = 20,
        session_id: str = "default",
        api_key: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Full-text search using OpenSearch."""
        start_time = time.perf_counter()
        limit = min(limit or 20, 100)

        es = _get_es_client()
        if not es:
            logger.warning("OpenSearch not available, falling back to PostgreSQL")
            return await self._search_content(
                query=query,
                source=source,
                block_type=block_type,
                limit=limit,
                session_id=session_id,
                api_key=api_key,
                agent_id=agent_id,
            )

        # Build Elasticsearch query
        must_clauses = [
            {
                "query_string": {
                    "query": query,
                    "default_field": "content",
                    "default_operator": "AND",
                    "analyze_wildcard": True,
                    "allow_leading_wildcard": False,
                }
            }
        ]

        filter_clauses = []
        if source:
            filter_clauses.append({"term": {"source": source}})
        if section:
            filter_clauses.append({"match": {"section": section}})
        if block_type:
            filter_clauses.append({"term": {"block_type": block_type}})

        es_query = {
            "bool": {
                "must": must_clauses,
                "filter": filter_clauses if filter_clauses else None,
            }
        }

        # Remove None filter
        if es_query["bool"]["filter"] is None:
            del es_query["bool"]["filter"]

        try:
            response = es.search(
                index=OS_INDEX_NAME,
                body={
                    "query": es_query,
                    "highlight": {
                        "fields": {
                            "content": {
                                "fragment_size": 200,
                                "number_of_fragments": 2,
                                "pre_tags": ["<mark>"],
                                "post_tags": ["</mark>"],
                            }
                        }
                    },
                    "size": limit,
                    "_source": [
                        "document_id",
                        "line_number",
                        "content",
                        "block_type",
                        "section",
                        "source",
                        "title",
                        "doi",
                        "authors",
                    ],
                },
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            results = []
            for hit in response["hits"]["hits"]:
                src = hit["_source"]
                highlight = hit.get("highlight", {}).get("content", [])

                # Truncate content for display
                content = src.get("content", "")
                if len(content) > 500:
                    content = content[:500] + "..."

                results.append(
                    {
                        "document_id": src.get("document_id"),
                        "line_number": src.get("line_number"),
                        "content": content,
                        "highlight": highlight[0] if highlight else None,
                        "block_type": src.get("block_type"),
                        "section": src.get("section"),
                        "source": src.get("source"),
                        "title": src.get("title"),
                        "doi": src.get("doi"),
                        "authors": (
                            src.get("authors", "")[:100] + "..."
                            if src.get("authors") and len(src.get("authors", "")) > 100
                            else src.get("authors")
                        ),
                        "score": round(hit["_score"], 4),
                        "url": (
                            f"https://www.{src.get('source')}.org/content/{src.get('doi')}"
                            if src.get("doi")
                            else None
                        ),
                    }
                )

            total_hits = response["hits"]["total"]
            total_value = (
                total_hits["value"] if isinstance(total_hits, dict) else total_hits
            )

            return {
                "status": "success",
                "query": query,
                "search_engine": "opensearch",
                "count": len(results),
                "total_hits": total_value,
                "query_time_ms": round(elapsed_ms, 1),
                "es_took_ms": response.get("took", 0),
                "results": results,
                "hint": "Use document_id with biomedrxiv_get_content for full context",
            }

        except Exception as e:
            logger.error(f"OpenSearch search failed: {e}")
            # Fallback to PostgreSQL
            return await self._search_content(
                query=query,
                source=source,
                block_type=block_type,
                limit=limit,
                session_id=session_id,
                api_key=api_key,
                agent_id=agent_id,
            )

    def cleanup(self) -> None:
        """Clean up database connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None
        logger.info("Papers tool module cleaned up")


