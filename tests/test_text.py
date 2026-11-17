from cortex_mem.store.text import (
    EVENT_VISIBLE_SQL,
    KNOWLEDGE_VISIBLE_SQL,
    bigrams,
    pg_params,
    words,
)


def test_words_lowercases_and_splits():
    assert words("Run API_v2!") == ["run", "api_v2"]


def test_bigrams_only_for_hangul_words():
    assert bigrams("배포가 done") == ["배포", "포가"]


def test_pg_params_rewrites_named_parameters():
    sql = "e.user = :user AND :scope IN ('team')"
    assert pg_params(sql) == "e.user = %(user)s AND %(scope)s IN ('team')"


def test_pg_params_keeps_casts():
    assert pg_params("emb <=> :q::vector") == "emb <=> %(q)s::vector"


def test_visibility_sql_is_unchanged():
    assert EVENT_VISIBLE_SQL == (
        "(e.user = :user OR (:scope IN ('team', 'org') AND e.scope = 'team' AND e.team = :team)"
        " OR (:scope = 'org' AND e.scope = 'org'))"
    )
    assert KNOWLEDGE_VISIBLE_SQL == (
        "((k.scope = 'personal' AND k.owner = :user)"
        " OR (k.scope = 'team' AND :scope IN ('team', 'org') AND k.owner = :team)"
        " OR (k.scope = 'org' AND :scope = 'org'))"
    )
