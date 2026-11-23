import hashlib
import math
import random

import pytest

from cortex_mem.embed import HashEmbedder, cosine, float32, text_digest, top_by_cosine


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


def test_float32_rounds_finite_values():
    assert float32([0.5, -2.0, 1e-50]) == [0.5, -2.0, 0.0]
    assert float32((1, 2)) == [1.0, 2.0]


@pytest.mark.parametrize(
    "bad", [[float("nan")], [1.0, float("inf")], [float("-inf")], [1e39], [-3.5e38]],
    ids=["nan", "inf", "-inf", "overflows", "overflows-negative"],
)
def test_float32_refuses_non_finite_values(bad):
    with pytest.raises(ValueError, match="finite"):
        float32(bad)


def _fsum_cosine(a, b):
    """The reference: every sum correctly rounded by math.fsum."""
    na2 = math.fsum(x * x for x in a)
    nb2 = math.fsum(y * y for y in b)
    return math.fsum(x * y for x, y in zip(a, b)) / math.sqrt(na2 * nb2)


@pytest.mark.parametrize("dim", [3, 256, 4096])
def test_cosine_of_float32_vectors_is_bit_for_bit_the_fsum_formula(dim):
    # The stores keep float32, whose products are exact in a double: any exact
    # summation (fsum, sumprod) then gives the same float, so the Python stores
    # rank exactly as before however the sums are computed.
    rng = random.Random(dim)
    query = float32([rng.gauss(0, 1) for _ in range(dim)])
    vectors = [float32([rng.gauss(0, 1) for _ in range(dim)]) for _ in range(200)]
    assert [cosine(query, v) for v in vectors] == [_fsum_cosine(query, v) for v in vectors]
    assert [s for _, s in top_by_cosine(query, enumerate(vectors), 200)] == sorted(
        (s for s in (_fsum_cosine(query, v) for v in vectors) if s > 0), reverse=True
    )


# ── OutageBreaker ───────────────────────────────────────────────────────────


class _Backend:
    """HashEmbedder that fails while `down` (an outage) or refuses `refuse` (one text)."""

    def __init__(self):
        self.inner = HashEmbedder()
        self.model, self.dim = self.inner.model, self.inner.dim
        self.down = False
        self.refuse = None
        self.calls = 0

    def embed(self, texts, kind):
        from cortex_mem.embed import EmbeddingError, EmbeddingRejected

        self.calls += 1
        if self.down:
            raise EmbeddingError("backend is down")
        if self.refuse in texts:
            raise EmbeddingRejected("refused")
        return self.inner.embed(texts, kind)


def _breaker(backend, now):
    from cortex_mem.embed import OutageBreaker

    return OutageBreaker(backend, first=5.0, maximum=20.0, clock=lambda: now[0])


def test_breaker_fails_fast_after_an_outage_and_retries_after_the_cooldown():
    from cortex_mem.embed import EmbeddingError

    backend, now = _Backend(), [100.0]
    emb = _breaker(backend, now)
    backend.down = True
    with pytest.raises(EmbeddingError, match="backend is down"):
        emb.embed(["a"], "passage")
    with pytest.raises(EmbeddingError, match="unavailable"):
        emb.embed(["b"], "query")  # within the cooldown: no request
    assert backend.calls == 1

    now[0] = 105.0  # the cooldown is over: one request finds it still down, and the wait doubles
    with pytest.raises(EmbeddingError, match="backend is down"):
        emb.embed(["a"], "passage")
    now[0] = 114.0
    with pytest.raises(EmbeddingError, match="unavailable"):
        emb.embed(["a"], "passage")
    assert backend.calls == 2

    backend.down = False
    now[0] = 115.0
    assert emb.embed(["a"], "passage") == HashEmbedder().embed(["a"], "passage")
    backend.down = True  # a success resets the wait to the first one
    with pytest.raises(EmbeddingError):
        emb.embed(["a"], "passage")
    now[0] = 120.0
    with pytest.raises(EmbeddingError, match="backend is down"):
        emb.embed(["a"], "passage")
    assert backend.calls == 5


def test_breaker_caps_the_wait_and_ignores_refused_texts():
    from cortex_mem.embed import EmbeddingError, EmbeddingRejected

    backend, now = _Backend(), [0.0]
    emb = _breaker(backend, now)
    backend.refuse = "too long"
    with pytest.raises(EmbeddingRejected):
        emb.embed(["too long"], "passage")
    assert emb.embed(["fine"], "passage")  # a refused text is not an outage
    backend.down = True
    for _ in range(6):  # waits 5, 10, 20, 20, 20, 20
        with pytest.raises(EmbeddingError, match="backend is down"):
            emb.embed(["a"], "passage")
        now[0] += 20.0
    assert backend.calls == 8
    assert (emb.model, emb.dim) == (backend.model, backend.dim)


def test_breaker_takes_a_refused_text_as_an_answer_and_starts_the_next_wait_over():
    from cortex_mem.embed import EmbeddingError, EmbeddingRejected

    backend, now = _Backend(), [0.0]
    emb = _breaker(backend, now)
    backend.down = True
    with pytest.raises(EmbeddingError, match="backend is down"):
        emb.embed(["a"], "passage")  # waits 5
    now[0] = 6.0
    with pytest.raises(EmbeddingError, match="backend is down"):
        emb.embed(["a"], "passage")  # waits 10
    backend.down, backend.refuse = False, "too long"
    now[0] = 17.0
    with pytest.raises(EmbeddingRejected):
        emb.embed(["too long"], "passage")  # the backend answered
    backend.down = True
    now[0] = 18.0
    with pytest.raises(EmbeddingError, match="backend is down"):
        emb.embed(["a"], "passage")
    now[0] = 23.0  # the new outage waits 5 again, not 20
    with pytest.raises(EmbeddingError, match="backend is down"):
        emb.embed(["a"], "passage")
    assert backend.calls == 5
