"""Encode/decode numeric block_ids to/from letter-only strings.

Biomedrxiv block IDs are PostgreSQL SERIAL integers (up to ~2.1B) encoded
as 7-character lowercase letter strings (base-26).

PMC block IDs are derived from (pmc_id_num, line_number):
    pmc_int = pmc_id_num * PMC_LINE_RANGE + line_number
This fits in ~13 characters.  The decoder distinguishes them from biomedrxiv
IDs because decoded value ≥ PMC_INT_THRESHOLD.

    biomedrxiv 58234571       →  "dkwpqrt"       (7 chars)
    PMC7194329 line 28        →  "bhizsmbaaa"    (10 chars)

Both are bijective (no collisions, no hashing).
"""

_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
_BASE = len(_ALPHABET)  # 26
_LENGTH = 7  # 26^7 covers biomedrxiv SERIAL max ~2.1B

# PMC encoding constants
# PMC IDs go up to ~PMC13_999_999 today; 100M gives headroom for decades.
# Line numbers per paper rarely exceed 10_000; 1_000_000 gives ample room.
PMC_LINE_RANGE = 1_000_000
PMC_INT_THRESHOLD = 1_000_000 * PMC_LINE_RANGE  # 10^12 — well above biomedrxiv range


def encode_block_id(block_id: int) -> str:
    """Encode a numeric block_id to a variable-length lowercase letter string.

    Values < PMC_INT_THRESHOLD → 7-char (biomedrxiv range).
    Values ≥ PMC_INT_THRESHOLD → 10-char (PMC range).
    """
    if block_id < 0:
        raise ValueError(f"block_id must be non-negative: {block_id}")
    chars = []
    n = block_id
    # Emit at least _LENGTH digits; keep going if n still has bits
    min_len = _LENGTH
    emitted = 0
    while emitted < min_len or n > 0:
        chars.append(_ALPHABET[n % _BASE])
        n //= _BASE
        emitted += 1
    return "".join(reversed(chars))


def decode_block_id(encoded: str) -> int:
    """Decode a letter string back to a numeric block_id (variable length)."""
    encoded = encoded.lower().strip()
    if not encoded or not encoded.isalpha():
        raise ValueError(f"Invalid encoded block_id: {encoded!r}")
    n = 0
    for c in encoded:
        n = n * _BASE + _ALPHABET.index(c)
    return n


def is_encoded_block_id(s: str) -> bool:
    """True for any all-lowercase-letter string of length 7–13 (bio or PMC)."""
    return 7 <= len(s) <= 13 and s.isalpha() and s.islower()


# ── PMC-specific helpers ──────────────────────────────────────────────────────

def encode_pmc_block_id(pmc_id: str, line_number: int) -> str:
    """Encode (pmc_id, line_number) into a letter-only block_id string.

    pmc_id must start with 'PMC' followed by digits.
    """
    num = int(pmc_id[3:])  # strip 'PMC' prefix
    combined = num * PMC_LINE_RANGE + line_number
    return encode_block_id(combined)


def decode_pmc_block_id(encoded: str) -> tuple[str, int]:
    """Decode a PMC block_id string back to (pmc_id, line_number).

    Raises ValueError if the decoded value is in the biomedrxiv range.
    """
    n = decode_block_id(encoded)
    if n < PMC_INT_THRESHOLD:
        raise ValueError(f"{encoded!r} is a biomedrxiv block_id, not PMC")
    pmc_num = n // PMC_LINE_RANGE
    line_num = n % PMC_LINE_RANGE
    return f"PMC{pmc_num}", line_num


def is_pmc_block_id(encoded: str) -> bool:
    """True if the block_id encodes a PMC (pmc_id, line_number) pair."""
    try:
        n = decode_block_id(encoded)
        return n >= PMC_INT_THRESHOLD
    except (ValueError, IndexError):
        return False
