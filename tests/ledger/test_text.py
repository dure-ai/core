from cortex_mem.engines.ledger.text import contains_term, jaccard, signature, tokenize


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
    sig = signature("x")
    assert sig is not None and len(sig) == 16


def test_tokenize_keeps_words_in_other_scripts():
    assert tokenize("Порт 8080, café!") == ["порт", "8080", "café"]


def test_tokenize_adds_bigrams_for_every_run_written_without_spaces():
    assert tokenize("配置文件") == ["配置文件", "配置", "置文", "文件"]
    assert tokenize("ポートは8080") == ["ポートは", "ポー", "ート", "トは", "8080"]


def test_signature_is_none_for_text_without_tokens():
    assert signature("🚀 !!") is None
    assert signature("") is None
    assert signature("配置文件在这里") != signature("数据库密码已更换")
    assert signature("Порт 8080") != signature("ポートは8080です")


def test_contains_term_needs_word_edges_except_inside_runs_without_spaces():
    assert contains_term("rotate the api key monthly", "key")
    assert not contains_term("the monkey patch", "key")
    assert not contains_term("see form.py", "rm")
    assert contains_term("never rm -rf / here", "rm -rf")
    assert contains_term("비밀번호를 바꿨다", "비밀번호")
    assert contains_term("修改密码", "密码")
