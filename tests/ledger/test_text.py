from cortex_mem.engines.ledger.text import jaccard, signature, tokenize


def test_tokenize_lowercases_and_splits_words():
    assert tokenize("Run `pytest -x` in API_v2!") == ["run", "pytest", "x", "in", "api_v2"]


def test_tokenize_keeps_a_hangul_run_and_adds_its_bigrams():
    assert tokenize("테스트를 돌린다") == ["테스트를", "테스", "스트", "트를", "돌린다", "돌린", "린다"]


def test_jaccard_is_token_set_overlap():
    assert jaccard("use vendor v2", "use vendor v1") == 0.5
    assert jaccard("a b", "b a") == 1.0
    assert jaccard("", "") == 0.0


def test_signature_ignores_case_spacing_and_punctuation():
    assert signature("Use  vendor v2.") == signature("use vendor V2")
    assert signature("use vendor v2") != signature("use vendor v1")
    assert len(signature("x")) == 16
