"""Text helpers for claim matching in the ledger engine."""

import hashlib
import re

# Scripts written without spaces between words: Hangul syllables, Hiragana,
# Katakana and CJK ideographs. Their runs also yield character bigrams.
_RUNS = "぀-ヿ㐀-䶿一-鿿豈-﫿가-힣"
_TOKEN = re.compile(rf"[{_RUNS}]+|[^\W{_RUNS}]+")
_RUN = re.compile(rf"[{_RUNS}]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens in any script (letters, digits and `_`); a run
    of a script written without spaces is followed by its character bigrams."""
    tokens: list[str] = []
    for piece in _TOKEN.findall(text.lower()):
        tokens.append(piece)
        if _RUN.fullmatch(piece):
            tokens.extend(piece[i : i + 2] for i in range(len(piece) - 1))
    return tokens


def jaccard(a: str, b: str) -> float:
    """Jaccard overlap of the token sets of two texts (0.0 when both are empty)."""
    left, right = set(tokenize(a)), set(tokenize(b))
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def signature(text: str) -> str | None:
    """Stable 16-hex-char fingerprint of a text's token sequence, or None for
    a text with no tokens (emoji or punctuation only): such texts would all
    share one fingerprint."""
    tokens = tokenize(text)
    if not tokens:
        return None
    return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()[:16]


def contains_term(text: str, term: str) -> bool:
    """Whether `term` occurs in `text` (both compared lowercase) as whole
    words: a letter, digit or `_` may not touch either end of it. Inside a
    script written without spaces any occurrence counts, since particles
    attach to words there (`비밀번호를` contains `비밀번호`)."""
    term = term.strip().lower()
    if not term:
        return False
    edge = rf"[^\W{_RUNS}]"
    pattern = rf"(?<!{edge}){re.escape(term)}(?!{edge})"
    return re.search(pattern, text.lower()) is not None
