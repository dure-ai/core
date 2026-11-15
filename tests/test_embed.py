import hashlib
import math

import pytest

from cortex_mem.embed import HashEmbedder, cosine, text_digest


def e(text, dim=256, kind="passage"):
    return HashEmbedder(dim).embed([text], kind)[0]


def test_cosine_of_identical_vectors_is_one():
    assert cosine([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine([1, 0], [0, 1]) == 0.0


def test_cosine_with_a_zero_vector_is_zero():
    assert cosine([0, 0], [1, 1]) == 0.0


def test_text_digest_is_sha256_hex():
    assert text_digest("a") == hashlib.sha256(b"a").hexdigest()


def test_hash_embedder_is_deterministic():
    emb = HashEmbedder()
    first = emb.embed(["deploy failed"], "passage")[0]
    assert emb.embed(["deploy failed"], "passage")[0] == first
    assert HashEmbedder().embed(["deploy failed"], "passage")[0] == first


def test_hash_embedder_vectors_are_unit_length():
    v = e("use vendor v2 for payments")
    assert math.fsum(x * x for x in v) == pytest.approx(1.0, abs=1e-12)


def test_hash_embedder_empty_text_is_the_zero_vector():
    assert e("") == [0.0] * 256
    assert e("!!!") == [0.0] * 256


def test_shared_words_score_higher_than_unrelated():
    base = e("use vendor v2 for payments")
    assert cosine(base, e("payments use vendor v2")) > cosine(base, e("rotate the staging keys"))


def test_korean_particles_still_overlap():
    assert cosine(e("배포가 실패했다"), e("배포 실패")) > 0


def test_kind_does_not_change_hash_vectors():
    assert e("same text", kind="passage") == e("same text", kind="query")


def test_dim_is_respected():
    emb = HashEmbedder(dim=64)
    assert len(emb.embed(["hello world"], "passage")[0]) == 64
    assert emb.model == "hash-64"


def test_embed_keeps_input_order():
    emb = HashEmbedder()
    both = emb.embed(["a b", "c d"], "passage")
    assert both[0] == emb.embed(["a b"], "passage")[0]
    assert both[1] == emb.embed(["c d"], "passage")[0]
