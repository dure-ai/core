"""scripts/sqlite_to_pg.py: move a SQLite store into PostgreSQL (spec §9)."""

import importlib.util
import sqlite3
import subprocess
import sys
from array import array
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from cortex_mem.contract import Caller, Event, Knowledge
from cortex_mem.embed import HashEmbedder, text_digest
from cortex_mem.index import Index
from cortex_mem.runtime import Runtime
from cortex_mem.store.postgres import PostgresDatabase
from cortex_mem.store.sqlite import SqliteDatabase

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sqlite_to_pg.py"
T0 = datetime(2026, 11, 15, tzinfo=UTC)
TEXTS = [
    "deploy uses vendor v2",
    "payments run on stripe",
    "deploy tags are signed",
    "the cache is warmed first",
    "payments retry on timeout",
]


def _load():
    spec = importlib.util.spec_from_file_location("sqlite_to_pg", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sqlite_to_pg = _load()


class CountingEmbedder:
    """HashEmbedder that remembers every text it was asked to embed."""

    def __init__(self):
        self.inner = HashEmbedder()
        self.model = self.inner.model
        self.dim = self.inner.dim
        self.calls = []

    def embed(self, texts, kind):
        self.calls.extend(texts)
        return self.inner.embed(texts, kind)


class CountEngine:
    state_version = "count-1"

    def observe(self, event, ctx):
        ctx.state.put("n", (ctx.state.get("n") or 0) + 1)
        return []

    def tick(self, now, ctx):
        return []

    def handoff(self, query, caller, scope, ctx, k=6):
        return []


def ev(i, text):
    return Event(
        id=f"e{i}", kind="record", at=T0, user="u1", team="t1", session="s1", scope="personal", text=text
    )


def runtime(db, embedder):
    index = Index(db.vectors, embedder, 0.9)
    return Runtime(CountEngine(), db.events, db.knowledge, db.state("count"), db.transaction, index=index)


def build_source(path):
    """5 events, 2 knowledge items, one engine's state/cursor/version, a full cache."""
    db = SqliteDatabase(path)
    for i in (1, 2):
        db.knowledge.put(
            Knowledge(
                id=f"k{i}",
                kind="knowledge",
                title=f"title {i}",
                body=TEXTS[i],
                scope="personal",
                owner="u1",
                sources=(f"e{i}",),
            )
        )
    rt = runtime(db, HashEmbedder())
    for i, text in enumerate(TEXTS, 1):
        rt.append(ev(i, text))
    db.con.close()


def run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=ROOT, check=False
    )


def caller():
    return Caller(user="u1", team="t1", session="s2", at=T0)


def test_moves_everything_and_the_first_sync_embeds_nothing(tmp_path, pg_schema):
    path = tmp_path / "core.db"
    build_source(path)
    result = run(str(path), pg_schema)
    assert result.returncode == 0, result.stderr
    assert pg_schema not in result.stdout + result.stderr
    for table in ("events", "knowledge", "engine_state", "engine_cursor", "embed_cache"):
        assert table in result.stdout

    src, pg = SqliteDatabase(path), PostgresDatabase(pg_schema)
    try:
        assert list(pg.events.since(0)) == list(src.events.since(0))
        assert [seq for seq, _ in pg.events.since(0)] == [1, 2, 3, 4, 5]
        assert sorted(pg.knowledge.all(), key=lambda k: k.id) == sorted(src.knowledge.all(), key=lambda k: k.id)
        s, p = src.state("count"), pg.state("count")
        assert dict(p.scan()) == dict(s.scan())
        assert p.cursor() == s.cursor() == 5
        assert p.version() == s.version() == "count-1"
        digest = text_digest(TEXTS[0])
        model = HashEmbedder().model
        raw = src.query("SELECT vec FROM embed_cache WHERE model = ? AND digest = ?", (model, digest))[0][0]
        expected = array("f")
        expected.frombytes(raw)
        assert pg.vectors.cached(model, digest) == expected.tolist()
        # Nothing derived was moved.
        assert pg.query("SELECT count(*) FROM event_vectors")[0][0] == 0
        assert pg.vectors.meta() is None
        # Text search ranks the same ids in the same order.
        for q in ("deploy", "payments", "cache warmed"):
            want = [e.id for e, _ in src.events.search(q, caller(), "personal")]
            got = [e.id for e, _ in pg.events.search(q, caller(), "personal")]
            assert got == want and want

        counter = CountingEmbedder()
        rt = runtime(pg, counter)
        rt.sync()
        assert counter.calls == []
        assert pg.vectors.meta()["store"] == "postgres"
        assert pg.vectors.indexed() == 5
    finally:
        src.con.close()
        pg.close()


def test_a_target_with_data_exits_2_and_is_left_alone(tmp_path, pg_schema):
    path = tmp_path / "core.db"
    build_source(path)
    assert run(str(path), pg_schema).returncode == 0
    again = run(str(path), pg_schema)
    assert again.returncode == 2
    assert "empty" in again.stderr
    pg = PostgresDatabase(pg_schema)
    try:
        assert pg.query("SELECT count(*) FROM events")[0][0] == 5
    finally:
        pg.close()


def test_a_source_that_fails_midway_leaves_the_target_empty(tmp_path, pg_schema):
    path = tmp_path / "core.db"
    build_source(path)
    con = sqlite3.connect(path)
    con.execute("UPDATE embed_cache SET vec = x'0102030405'")  # not a multiple of 4 bytes
    con.commit()
    con.close()
    result = run(str(path), pg_schema)
    assert result.returncode == 2
    pg = PostgresDatabase(pg_schema)
    try:
        for table in ("events", "knowledge", "engine_state", "engine_cursor", "embed_cache"):
            assert pg.query(f"SELECT count(*) FROM {table}")[0][0] == 0
    finally:
        pg.close()


def test_a_missing_sqlite_file_exits_2(tmp_path):
    missing = tmp_path / "nope.db"
    result = run(str(missing), "postgresql://u:hunter2pw@127.0.0.1:1/db")
    assert result.returncode == 2
    assert "nope.db" in result.stderr
    assert not missing.exists()


def test_a_file_that_is_not_sqlite_exits_2(tmp_path):
    path = tmp_path / "junk.db"
    path.write_text("this is not a database" * 50, encoding="utf-8")
    result = run(str(path), "postgresql://u:hunter2pw@127.0.0.1:1/db")
    assert result.returncode == 2


def test_a_failed_connection_exits_2_without_the_password(tmp_path):
    path = tmp_path / "core.db"
    SqliteDatabase(path).con.close()
    url = "postgresql://someone:hunter2pw@127.0.0.1:1/db"
    result = run(str(path), url)
    assert result.returncode == 2
    assert "hunter2pw" not in result.stdout + result.stderr
    assert url not in result.stdout + result.stderr


def test_the_password_never_appears_in_the_output(tmp_path, pg_schema):
    path = tmp_path / "core.db"
    build_source(path)
    password = urlsplit(pg_schema).password
    result = run(str(path), pg_schema)
    assert password and password not in result.stdout + result.stderr
