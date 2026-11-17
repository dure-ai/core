"""Tokenizer, visibility SQL and parameter rewriting shared by the stores."""

import re

_WORD = re.compile(r"\w+")
_HANGUL = re.compile(r"[가-힣]")
_NAMED_PARAM = re.compile(r"(?<!:):([A-Za-z_][A-Za-z0-9_]*)")

EVENT_VISIBLE_SQL = (
    "(e.user = :user OR (:scope IN ('team', 'org') AND e.scope = 'team' AND e.team = :team)"
    " OR (:scope = 'org' AND e.scope = 'org'))"
)
"""WHERE fragment limiting events to those the caller may see."""

KNOWLEDGE_VISIBLE_SQL = (
    "((k.scope = 'personal' AND k.owner = :user)"
    " OR (k.scope = 'team' AND :scope IN ('team', 'org') AND k.owner = :team)"
    " OR (k.scope = 'org' AND :scope = 'org'))"
)
"""WHERE fragment limiting knowledge to what the caller may see."""


def words(text: str) -> list[str]:
    """Lowercased word tokens of the text."""
    return _WORD.findall(text.lower())


def bigrams(text: str) -> list[str]:
    """Character bigrams of the Hangul words in the text."""
    grams = []
    for word in words(text):
        if _HANGUL.search(word):
            grams += [word[i : i + 2] for i in range(len(word) - 1)]
    return grams


def pg_params(sql: str) -> str:
    """Rewrite :name parameters to psycopg's %(name)s, leaving :: casts alone."""
    return _NAMED_PARAM.sub(r"%(\1)s", sql)
