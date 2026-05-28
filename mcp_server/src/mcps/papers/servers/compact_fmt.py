"""
Compact output formatters for BioMedRxiv MCP tool responses.

Replaces verbose JSON with token-efficient plain-text formats.
Each formatter takes a dict (the tool's return value) and returns a compact string.

Design principles (following FDA compact_fmt pattern):
- stdout IS the response — emit it directly, no key name
- stderr only when non-empty, prefixed with ERR:
- exit_code only when non-zero (success is implied)
- cwd as a compact @/path/ trailer (always useful for orientation)
- Drop prompt — the LLM never needs the shell prompt string
- Drop time_ms — the LLM never acts on execution time
- No JSON structural overhead — no braces, quotes, commas, indentation
"""

from typing import Any

# ── Shell formatter ─────────────────────────────────────────────────


def fmt_shell(result: dict) -> str:
    """Compact formatter for bash responses.

    Typical savings: 35-66% token reduction for small commands,
    7-15% for large content payloads (where stdout dominates).
    Serialization is ~25x faster (string concat vs json.dumps).
    """
    parts: list[str] = []

    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    exit_code = result.get("exit_code", 0)
    cwd = result.get("cwd", "")

    # stdout is the primary payload — emit directly
    if stdout:
        parts.append(stdout.rstrip("\n"))

    # stderr only if non-empty
    if stderr:
        parts.append(f"ERR: {stderr.rstrip()}")

    # exit_code only if non-zero (success is implied)
    if exit_code != 0:
        parts.append(f"[exit {exit_code}]")

    # cwd as a compact trailer — always useful for the LLM to know
    # where it is after the command
    if cwd:
        parts.append(f"@{cwd}")

    return "\n".join(parts)


# ── Stat formatter ──────────────────────────────────────────────────


def fmt_stat(result: dict) -> str:
    """Compact formatter for papers_stat responses."""
    if "error" in result:
        return f"ERROR: {result['error']}"

    parts: list[str] = []

    # Path
    path = result.get("path", "")
    if path:
        parts.append(path)

    # Key metadata on one line
    meta_parts: list[str] = []
    for key in ("document_id", "title", "doi", "source", "lines", "sections"):
        val = result.get(key)
        if val:
            meta_parts.append(f"{key}={val}")
    if meta_parts:
        parts.append("|".join(meta_parts))

    # Authors
    authors = result.get("authors", "")
    if authors:
        parts.append(authors)

    # Month/year
    month_year = result.get("month_year", "")
    if month_year:
        parts.append(month_year)

    return "\n".join(parts) if parts else "ok"


# ── Citation formatter ──────────────────────────────────────────────


def fmt_citation(result: dict) -> str:
    """Compact formatter for papers_get_citation responses."""
    if "error" in result:
        hint = result.get("hint", "")
        err_parts = [f"ERROR: {result['error']}"]
        if hint:
            err_parts.append(hint)
        return "\n".join(err_parts)

    parts: list[str] = []
    for key in (
        "document_id",
        "source_type",
        "block_id",
        "line_number",
        "page",
        "section",
        "block_type",
        "doi",
        "title",
        "authors",
    ):
        val = result.get(key)
        if val is not None and val != "":
            parts.append(f"{key}: {val}")

    content = result.get("content", "")
    if content:
        parts.append(f"---\n{content}")

    return "\n".join(parts) if parts else "ok"


# ── search_and_filter formatter ──────────────────────────────────────


def fmt_search_and_filter(result: dict) -> str:
    """Compact formatter for search_and_filter responses."""
    if "error" in result:
        return f"ERROR: {result['error']}"

    results_id = result.get("results_id", "")
    total_searched = result.get("total_searched", 0)
    total_relevant = result.get("total_relevant", 0)
    total_returned = result.get("total_returned", 0)
    n_requested = result.get("n_requested", 0)
    search_ms = result.get("search_ms", 0)
    filter_ms = result.get("filter_ms", 0)
    total_ms = result.get("total_ms", 0)

    parts = [
        f"search_and_filter: {total_searched} searched → {total_relevant} relevant → {total_returned} returned (of {n_requested} requested)",
        f"results_id: {results_id}  |  search: {search_ms}ms  filter: {filter_ms}ms  total: {total_ms}ms",
    ]

    papers = result.get("papers", [])
    for i, p in enumerate(papers[:30], 1):
        title = p.get("title", "Untitled")[:80]
        doc_id = p.get("document_id", "?")
        score = p.get("relevance_score", 0)
        authors = p.get("authors", "")[:40]
        month = p.get("month_year", "")
        line = f"  {i}. [{score}/10] {title}"
        line += f"\n     doc_id: {doc_id}"
        if authors:
            line += f"  {authors}"
        if month:
            line += f"  ({month})"
        parts.append(line)

    if len(papers) > 30:
        parts.append(f"  ... and {len(papers) - 30} more")

    if results_id:
        parts.append(
            f'Cite: {{{{"artifact": {{{{"artifact_id": "{results_id}", "type": "table", '
            f'"source_count": {total_returned}, "description": "YOUR_ONE_SENTENCE_SUMMARY"}}}}}}}}'
        )

    return "\n".join(parts)


# ── Registry ────────────────────────────────────────────────────────
# Maps internal function names (func.__name__) to compact formatters.
# Functions not in this dict fall back to json.dumps(result, indent=2).

FORMATTERS: dict[str, Any] = {
    "_shell": fmt_shell,
    "_paperclip": fmt_shell,
    "_stat": fmt_stat,
    "_get_citation": fmt_citation,
    "_search_and_filter": fmt_search_and_filter,
}
