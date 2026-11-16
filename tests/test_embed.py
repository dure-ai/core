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


def test_cosine_of_identical_vectors_is_exactly_one():
    import random

    rng = random.Random(20261115)
    for _ in range(300):
        v = [rng.gauss(0, 1) for _ in range(1024)]
        assert cosine(v, list(v)) == 1.0
        scaled = [x * 3.0 for x in v]
        assert cosine(scaled, scaled) == 1.0


def test_cosine_rejects_vectors_of_different_lengths():
    with pytest.raises(ValueError, match="length"):
        cosine([1.0, 0.0, 0.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="length"):
        cosine([0.0], [0.0, 0.0])  # even when either is a zero vector


def test_cosine_survives_a_product_of_norms_out_of_float_range():
    # Each squared norm fits in a float64; their product does not.
    assert cosine([1e-100, 0.0], [1e-100, 1e-100]) == pytest.approx(2**-0.5)
    assert cosine([1e100, 0.0], [1e100, 1e100]) == pytest.approx(2**-0.5)


def test_cosine_keeps_precision_when_the_norm_product_is_subnormal():
    assert cosine([1e-160, 0], [1e-160, 1e-160]) == pytest.approx(0.70711, rel=1e-4)
    assert cosine([1e-160, 0], [1e-160, 1e-160]) == pytest.approx(math.sqrt(0.5), rel=1e-9)
