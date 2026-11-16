import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from cortex_mem.embed import EmbeddingError, UpstageEmbedder

KEY = "sk-test-secret"


class Fake:
    def __init__(self):
        self.requests = []
        self.status = 200
        self.body = None  # callable(request_json) -> bytes/obj, or None for default
        self.delay = 0.0
        self.base_url = ""


def _default(req):
    return {
        "data": [{"index": i, "embedding": [float(len(t)), 1.0, 0.0]} for i, t in enumerate(req["input"])]
    }


@pytest.fixture
def fake():
    state = Fake()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            req = json.loads(raw)
            state.requests.append({"path": self.path, "headers": dict(self.headers), "json": req})
            if state.delay:
                time.sleep(state.delay)
            body = state.body(req) if state.body else _default(req)
            payload = body if isinstance(body, bytes) else json.dumps(body).encode()
            try:
                self.send_response(state.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except OSError:
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    state.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    yield state
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def make(fake, **kw):
    return UpstageEmbedder(KEY, "p-model", "q-model", base_url=fake.base_url, **kw)


def test_posts_the_texts_to_the_embeddings_endpoint(fake):
    make(fake).embed(["a", "b"], "passage")
    (r,) = fake.requests
    assert r["path"] == "/embeddings"
    headers = {k.lower(): v for k, v in r["headers"].items()}
    assert headers["authorization"] == f"Bearer {KEY}"
    assert headers["content-type"] == "application/json"
    assert r["json"] == {"model": "p-model", "input": ["a", "b"]}


def test_query_kind_uses_the_query_model(fake):
    make(fake).embed(["a"], "query")
    assert fake.requests[0]["json"]["model"] == "q-model"


def test_results_follow_the_index_field(fake):
    fake.body = lambda req: {"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]}
    assert make(fake).embed(["a", "b"], "passage") == [[1.0], [2.0]]


def test_batches_large_inputs(fake):
    out = make(fake, batch=2).embed(["a", "bb", "ccc", "dddd", "eeeee"], "passage")
    assert len(fake.requests) == 3
    assert [v[0] for v in out] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_dim_is_known_after_the_first_call(fake):
    emb = make(fake)
    with pytest.raises(ValueError):
        _ = emb.dim
    emb.embed(["a"], "passage")
    assert emb.dim == 3


def test_http_error_raises_embedding_error_without_the_key(fake):
    fake.status = 500
    fake.body = lambda req: {"error": "boom"}
    emb = make(fake)
    with pytest.raises(EmbeddingError) as info:
        emb.embed(["a"], "passage")
    assert "500" in str(info.value)
    assert KEY not in str(info.value)
    assert KEY not in repr(emb)
    assert repr(emb) == "UpstageEmbedder(passage_model='p-model', query_model='q-model')"


def test_count_mismatch_raises(fake):
    fake.body = lambda req: {"data": [{"index": 0, "embedding": [1.0]}]}
    with pytest.raises(EmbeddingError):
        make(fake).embed(["a", "b"], "passage")


def test_timeout_raises_embedding_error(fake):
    fake.delay = 1.0
    with pytest.raises(EmbeddingError):
        make(fake, timeout=0.2).embed(["a"], "passage")


def test_empty_input_makes_no_request(fake):
    assert make(fake).embed([], "passage") == []
    assert fake.requests == []


def test_truncated_body_raises_embedding_error(fake):
    import socket

    # Raw server that promises more bytes than it sends, then closes.
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def serve():
        conn, _ = srv.accept()
        conn.recv(65536)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 500\r\n\r\n{\"data\"")
        conn.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    emb = UpstageEmbedder(KEY, "p", "q", base_url=f"http://127.0.0.1:{srv.getsockname()[1]}")
    try:
        with pytest.raises(EmbeddingError) as info:
            emb.embed(["a"], "passage")
    finally:
        srv.close()
        t.join(timeout=5)
    assert KEY not in str(info.value)


def test_duplicate_index_raises(fake):
    fake.body = lambda req: {"data": [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}]}
    with pytest.raises(EmbeddingError):
        make(fake).embed(["a", "b"], "passage")


def test_out_of_range_index_raises(fake):
    fake.body = lambda req: {"data": [{"index": 0, "embedding": [1.0]}, {"index": 5, "embedding": [2.0]}]}
    with pytest.raises(EmbeddingError):
        make(fake).embed(["a", "b"], "passage")


def test_empty_embedding_raises_and_dim_stays_unknown(fake):
    fake.body = lambda req: {"data": [{"index": 0, "embedding": []}]}
    emb = make(fake)
    with pytest.raises(EmbeddingError):
        emb.embed(["a"], "passage")
    with pytest.raises(ValueError):
        _ = emb.dim


def test_inconsistent_lengths_within_a_response_raise(fake):
    fake.body = lambda req: {"data": [{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": [1.0, 2.0]}]}
    emb = make(fake)
    with pytest.raises(EmbeddingError):
        emb.embed(["a", "b"], "passage")
    with pytest.raises(ValueError):
        _ = emb.dim


def test_length_change_on_a_later_call_raises(fake):
    emb = make(fake)
    emb.embed(["a"], "passage")
    fake.body = lambda req: {"data": [{"index": 0, "embedding": [1.0, 2.0]}]}
    with pytest.raises(EmbeddingError):
        emb.embed(["a"], "passage")
    assert emb.dim == 3


# ── keys that cannot go into a header ───────────────────────────────────────


@pytest.fixture
def offline(monkeypatch):
    """Fail the test if anything opens a connection."""
    import http.client

    def connect(self):
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(http.client.HTTPConnection, "connect", connect)


def _everything(exc):
    import traceback

    return "".join(traceback.format_exception(exc)) + repr(exc) + repr(exc.__context__) + repr(exc.__cause__)


BAD_KEYS = ["sk-line\r\nX-Injected: 1", "sk-bare\n", "sk-tab\tkey", "sk-한글-key", "sk-nul\x00key", "sk-del\x7fkey"]


@pytest.mark.parametrize("key", BAD_KEYS + [""])
def test_constructor_rejects_a_key_that_cannot_go_into_a_header(key, offline):
    with pytest.raises(ValueError) as info:
        UpstageEmbedder(key, "p", "q")
    if key:
        assert key not in _everything(info.value)
        assert key.strip() not in str(info.value)


@pytest.mark.parametrize("key", ["sk-line\r\nX-Injected: 1", "sk-한글-key"])
def test_a_bad_key_that_reaches_the_request_becomes_an_embedding_error(key, offline):
    emb = UpstageEmbedder(KEY, "p", "q", base_url="http://127.0.0.1:9")
    emb._api_key = key  # past the constructor check, as a subclass or a later change could do
    with pytest.raises(EmbeddingError) as info:
        emb.embed(["a"], "passage")
    assert key not in _everything(info.value)
    assert info.value.__context__ is None and info.value.__cause__ is None
