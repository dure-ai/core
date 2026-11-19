"""Text helpers for claim matching in the ledger engine."""

import hashlib
import re

_TOKEN = re.compile(r"[a-z0-9_]+|[가-힣]+")
_HANGUL = re.compile(r"[가-힣]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; Hangul runs are followed by their character bigrams."""
    tokens: list[str] = []
    for piece in _TOKEN.findall(text.lower()):
        tokens.append(piece)
        if _HANGUL.fullmatch(piece):
            tokens.extend(piece[i : i + 2] for i in range(len(piece) - 1))
    return tokens


def jaccard(a: str, b: str) -> float:
    """Jaccard overlap of the token sets of two texts (0.0 when both are empty)."""
    left, right = set(tokenize(a)), set(tokenize(b))
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def signature(text: str) -> str:
    """Stable 16-hex-char fingerprint of a text's token sequence."""
    return hashlib.sha256(" ".join(tokenize(text)).encode("utf-8")).hexdigest()[:16]
