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


def token_set(text: str) -> frozenset[str]:
    """The set of a text's tokens, for comparing one text with many."""
    return frozenset(tokenize(text))


def overlap(left: frozenset[str], right: frozenset[str]) -> float:
    """Jaccard overlap of two token sets (0.0 when both are empty)."""
    union = len(left | right)
    if not union:
        return 0.0
    return len(left & right) / union


def jaccard(a: str, b: str) -> float:
    """Jaccard overlap of the token sets of two texts (0.0 when both are empty)."""
    return overlap(token_set(a), token_set(b))


def signature(text: str) -> str | None:
    """Stable 16-hex-char fingerprint of a text's token sequence, or None for
    a text with no tokens (emoji or punctuation only): such texts would all
    share one fingerprint."""
    tokens = tokenize(text)
    if not tokens:
        return None
    return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()[:16]


def contains_term(text: str, term: str) -> bool:
    """Whether `term` occurs in `text` (both compared lowercase), unless a
    letter of a spaced script sits right before it (`key` is not in
    `monkey`). It errs toward a match, since a miss skips a human check:
    anything may follow the term (`passwords`), and a digit, `_` or a letter
    of a script written without spaces may precede it (`DB_PASSWORD`,
    `새비밀번호`)."""
    term = term.strip().lower()
    if not term:
        return False
    letter = rf"[^\W\d_{_RUNS}]"
    return re.search(rf"(?<!{letter}){re.escape(term)}", text.lower()) is not None
