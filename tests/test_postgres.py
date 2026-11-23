"""PostgreSQL database: connection, the one advisory-locked transaction, and the schema.

Every test that needs a server uses the ``pg_schema`` fixture, so those skip without
``CORTEX_TEST_PG_URL``; the pgvector version checks run without one.
"""

import math
import random
import re
import threading
from contextlib import closing
from datetime import datetime, timedelta, timezone

import pytest

from cortex_mem.contract import Event, Knowledge
from cortex_mem.embed import cosine
from cortex_mem.runtime import Runtime
from cortex_mem.store.base import CursorConflict
from cortex_mem.store.postgres import (
    SCHEMA_VERSION,
    PostgresDatabase,
    pgvector_version,
    require_hnsw_support,
    require_vector_type,
)
from cortex_mem.testing import (
    SCORE_TOLERANCE,
    DatabaseContract,
    EngineStateBackendContract,
    EventStoreContract,
    KnowledgeStoreContract,
    VectorBackendContract,
    caller,
    ev,
    f32,
    item,
)

TABLES = {
    "cortex_meta", "events", "event_terms", "event_stats", "knowledge", "knowledge_terms",
    "knowledge_stats", "engine_state", "engine_cursor", "event_vectors", "vector_groups",
    "knowledge_vectors", "embed_cache", "vector_meta",
}  # fmt: skip
WAIT = 5.0  # generous upper bound for things that should happen at once


def _tables(db: PostgresDatabase) -> set[str]:
    rows = db.query(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()"
    )
    return {r[0] for r in rows}


def _column_type(db: PostgresDatabase, table: str, column: str) -> str | None:
    rows = db.query(
        "SELECT udt_name FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = %s AND column_name = %s",
        (table, column),
    )
    return rows[0][0] if rows else None


def test_fresh_schema_is_created_and_versioned(pg_schema):
    assert SCHEMA_VERSION == 2
    with closing(PostgresDatabase(pg_schema)) as db:
        assert db.kind == "postgres"
        assert _tables(db) == TABLES
        assert db.query("SELECT value FROM cortex_meta WHERE key = 'schema_version'") == [("2",)]
        assert _column_type(db, "events", "user") == "text"
        assert _column_type(db, "events", "seq") == "int8"
        assert _column_type(db, "events", "text_redacted") == "bool"
        for table in ("event_vectors", "knowledge_vectors", "embed_cache"):
            assert _column_type(db, table, "emb") == "vector"
    # A second start on the same schema finds it and keeps it.
    with closing(PostgresDatabase(pg_schema)) as db:
        assert _tables(db) == TABLES
        assert db.query("SELECT count(*) FROM cortex_meta") == [(1,)]
    with pytest.raises(ValueError, match="vector_index"):
        PostgresDatabase(pg_schema, vector_index="ivfflat")


def test_generation_survives_reopening(pg_schema):
    meta = {"model": "m", "group_threshold": 0.5}
    with closing(PostgresDatabase(pg_schema)) as db:
        assert db.vectors.generation() == 0
        db.vectors.reset(meta)
        db.vectors.reset(meta)
        db.vectors.put_vector("knowledge", "k1", [1.0, 0.0], digest="dk")
    with closing(PostgresDatabase(pg_schema)) as db:
        assert db.vectors.generation() == 2
        assert db.vectors.meta() == meta
        assert db.vectors.knowledge_digests() == {"k1": "dk"}
        with pytest.raises(CursorConflict):
            db.vectors.set_indexed(0, 1, 1)
        db.vectors.set_indexed(0, 1, 2)
    with closing(PostgresDatabase(pg_schema)) as db:
        assert db.vectors.indexed() == 1


def test_unknown_schema_version_refuses_to_start(pg_schema):
    with closing(PostgresDatabase(pg_schema)) as db:
        db.query("UPDATE cortex_meta SET value = '99' WHERE key = 'schema_version'")
    with pytest.raises(RuntimeError, match=r"99.*\b2\b"):
        PostgresDatabase(pg_schema)


def test_a_schema_1_database_is_refused(pg_schema):
    """Schema 1 stored search terms from the old tokens, with a Latin word kept whole when a Korean
    particle followed it; new queries would not match them, so it must be started again."""
    with closing(PostgresDatabase(pg_schema)) as db:
        db.query("UPDATE cortex_meta SET value = '1' WHERE key = 'schema_version'")
    with pytest.raises(RuntimeError, match=r"schema 1 is not 2"):
        PostgresDatabase(pg_schema)


@pytest.mark.parametrize(
    "table",
    [
        "CREATE TABLE engine_cursor (engine text PRIMARY KEY, cursor bigint, version text)",
        "CREATE TABLE knowledge_vectors (id text PRIMARY KEY, emb vector)",
    ],
)
def test_a_schema_from_before_the_epoch_and_digest_columns_is_refused(pg_schema, table):
    import psycopg

    with psycopg.connect(pg_schema, autocommit=True) as con:
        con.execute(table)

    with pytest.raises(RuntimeError, match="drop") as raised:
        PostgresDatabase(pg_schema)
    assert "postgresql://" not in str(raised.value) and "cortex@" not in str(raised.value)


def test_missing_vector_extension_is_reported(pg_schema):
    import psycopg

    with closing(PostgresDatabase(pg_schema)) as db:
        schema = db.query("SELECT current_schema()")[0][0]
    # The extension lives in public; without public on the search path the type does not
    # resolve, which is what a database without the extension looks like to the check.
    with psycopg.connect(pg_schema, autocommit=True) as conn:
        require_vector_type(conn)  # found through public
        conn.execute(f'SET search_path = "{schema}"')
        with pytest.raises(RuntimeError, match="pgvector") as info:
            require_vector_type(conn)
    message = str(info.value)
    assert "CREATE EXTENSION vector" in message
    assert "postgresql://" not in message


class _Answers:
    """A connection stand-in whose every query returns ``row``."""

    def __init__(self, row):
        self.row = row

    def execute(self, sql, params=()):
        return self

    def fetchone(self):
        return self.row


@pytest.mark.parametrize(
    ("version", "ok"),
    [("0.7.4", False), ("0.5.1", False), ("0.8.0", True), ("0.8.1", True), ("0.10.1", True), ("1.0", True)],
)
def test_hnsw_needs_pgvector_0_8(version, ok):
    # Numbers, not strings: "0.10.1" < "0.8" as text, but it is the newer release.
    if ok:
        require_hnsw_support(_Answers((version,)))
        return
    with pytest.raises(RuntimeError, match=rf"0\.8.*{re.escape(version)}") as info:
        require_hnsw_support(_Answers((version,)))
    assert "postgresql://" not in str(info.value)


def test_hnsw_refuses_when_the_pgvector_version_is_unknown():
    with pytest.raises(RuntimeError, match="0.8"):
        require_hnsw_support(_Answers(None))


def test_opening_with_hnsw_checks_the_server_version(pg_schema):
    with closing(PostgresDatabase(pg_schema)) as db:
        found = db.query("SELECT extversion FROM pg_extension WHERE extname = 'vector'")[0][0]
    with closing(PostgresDatabase(pg_schema, vector_index="hnsw")) as db:  # the test server has 0.8+
        require_hnsw_support(db.con)
    assert pgvector_version(found) >= (0, 8)


def test_nested_transaction_rolls_back_everything(pg_schema):
    with closing(PostgresDatabase(pg_schema)) as db:
        with pytest.raises(KeyError), db.transaction():
            db.query("INSERT INTO vector_meta (key, value) VALUES ('outer', '1')")
            with db.transaction():
                db.query("INSERT INTO vector_meta (key, value) VALUES ('inner', '2')")
            raise KeyError("boom")
        assert db.query("SELECT key FROM vector_meta") == []
        # The connection is usable again and a later (nested) transaction commits.
        with db.transaction(), db.transaction():
            db.query("INSERT INTO vector_meta (key, value) VALUES ('kept', '3')")
        assert db.query("SELECT key FROM vector_meta") == [("kept",)]
        # An exception raised only in the inner block still rolls back the outer writes.
        with pytest.raises(KeyError), db.transaction():
            db.query("INSERT INTO vector_meta (key, value) VALUES ('outer2', '4')")
            with db.transaction():
                raise KeyError("inner")
        assert db.query("SELECT key FROM vector_meta") == [("kept",)]


def _hold(db: PostgresDatabase, holding: threading.Event, release: threading.Event) -> None:
    with db.transaction():
        holding.set()
        release.wait(WAIT)


def test_transaction_serializes_two_connections(pg_schema):
    with closing(PostgresDatabase(pg_schema)) as a, closing(PostgresDatabase(pg_schema)) as b:
        holding, release, entered = threading.Event(), threading.Event(), threading.Event()

        def enter() -> None:
            with b.transaction():
                entered.set()

        holder = threading.Thread(target=_hold, args=(a, holding, release))
        waiter = threading.Thread(target=enter)
        holder.start()
        try:
            assert holding.wait(WAIT)
            waiter.start()
            assert not entered.wait(0.3), "B entered while A held the transaction"
            release.set()
            assert entered.wait(WAIT), "B did not enter after A finished"
        finally:
            release.set()
            holder.join(WAIT)
            if waiter.is_alive() or waiter.ident is not None:
                waiter.join(WAIT)


def test_lock_timeout_raises_a_clear_error(pg_schema):
    with closing(PostgresDatabase(pg_schema)) as a, closing(
        PostgresDatabase(pg_schema, lock_timeout="200ms")
    ) as b:
        holding, release = threading.Event(), threading.Event()
        holder = threading.Thread(target=_hold, args=(a, holding, release))
        holder.start()
        try:
            assert holding.wait(WAIT)
            with (
                pytest.raises(TimeoutError, match="CORTEX_DB_LOCK_TIMEOUT") as info,
                b.transaction(),
            ):
                pass
        finally:
            release.set()
            holder.join(WAIT)
        assert "postgresql://" not in str(info.value)
        assert "200ms" in str(info.value)
        # B recovers: once A is done, its next transaction goes through.
        with b.transaction():
            b.query("INSERT INTO vector_meta (key, value) VALUES ('after', '1')")
        assert b.query("SELECT key FROM vector_meta") == [("after",)]


def test_password_is_not_in_errors_or_repr(pg_schema):
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    params = conninfo_to_dict(pg_schema)
    wrong = "Wr0ng-pa55word-x7q"
    with pytest.raises(ConnectionError) as info:
        PostgresDatabase(make_conninfo(pg_schema, password=wrong))
    message = str(info.value)
    assert wrong not in message
    assert info.value.__cause__ is None and info.value.__suppress_context__
    assert str(params.get("port", "")) in message

    with closing(PostgresDatabase(pg_schema)) as db:
        text = repr(db)
    assert text == (
        f"PostgresDatabase(host={params.get('host')}, port={params.get('port')}, "
        f"dbname={params.get('dbname')})"
    )
    if params.get("password"):
        assert params["password"] not in text


def _lock_is_free(url: str) -> None:
    """Take the write lock from another connection; TimeoutError if someone still holds it."""
    with closing(PostgresDatabase(url, lock_timeout="200ms")) as other, other.transaction():
        pass


def test_caught_inner_failure_is_not_committed_silently(pg_schema):
    import psycopg

    with closing(PostgresDatabase(pg_schema)) as db:
        with pytest.raises(RuntimeError, match="rolled back"), db.transaction():
            db.query("INSERT INTO vector_meta (key, value) VALUES ('outer', '1')")
            # The caller catches a failed statement; PostgreSQL has aborted the transaction.
            with pytest.raises(psycopg.errors.UniqueViolation), db.transaction():
                db.query("INSERT INTO vector_meta (key, value) VALUES ('outer', '2')")
        assert db.query("SELECT key FROM vector_meta") == []
        _lock_is_free(pg_schema)
        with db.transaction():
            db.query("INSERT INTO vector_meta (key, value) VALUES ('later', '1')")
        assert db.query("SELECT key FROM vector_meta") == [("later",)]


def test_rollback_keeps_the_original_error_when_the_connection_is_gone(pg_schema):
    import psycopg

    # Closed by this process.
    db = PostgresDatabase(pg_schema)
    with pytest.raises(KeyError, match="original"), db.transaction():
        db.con.close()
        raise KeyError("original")
    # Ended by the server; the client has not noticed yet.
    with closing(PostgresDatabase(pg_schema)) as db:
        pid = db.query("SELECT pg_backend_pid()")[0][0]
        with pytest.raises(KeyError, match="original"), db.transaction():
            with psycopg.connect(pg_schema, autocommit=True) as admin:
                admin.execute("SELECT pg_terminate_backend(%s)", (pid,))
            raise KeyError("original")
    _lock_is_free(pg_schema)


class _CommitFails:
    """Wrap a connection so COMMIT fails on the client and never reaches the server."""

    def __init__(self, con):
        self._con = con

    def __getattr__(self, name):
        return getattr(self._con, name)

    def execute(self, sql, *args, **kwargs):
        if sql == "COMMIT":
            import psycopg

            raise psycopg.OperationalError("simulated commit failure")
        return self._con.execute(sql, *args, **kwargs)


def test_failed_commit_releases_the_lock(pg_schema):
    import psycopg

    with closing(PostgresDatabase(pg_schema)) as db:
        # A real COMMIT failure: a deferred constraint is checked at COMMIT.
        db.query("CREATE TABLE deferred (k text UNIQUE DEFERRABLE INITIALLY DEFERRED)")
        with pytest.raises(psycopg.errors.UniqueViolation), db.transaction():
            db.query("INSERT INTO deferred VALUES ('a'), ('a')")
        _lock_is_free(pg_schema)
        assert db.query("SELECT count(*) FROM deferred") == [(0,)]
        # A COMMIT that fails before the server ends the transaction.
        real = db.con
        db.con = _CommitFails(real)
        try:
            with pytest.raises(psycopg.OperationalError, match="simulated"), db.transaction():
                db.query("INSERT INTO vector_meta (key, value) VALUES ('lost', '1')")
        finally:
            db.con = real
        _lock_is_free(pg_schema)
        assert db.query("SELECT key FROM vector_meta") == []


def test_lock_timeout_is_validated_at_construction(pg_schema):
    for bad in ("bogus", "30 seconds", "-1s", "", "9999999999999d"):
        with pytest.raises(ValueError, match="CORTEX_DB_LOCK_TIMEOUT"):
            PostgresDatabase(pg_schema, lock_timeout=bad)
    for good in ("30s", "500ms", "2min", "0", "1.5s"):
        with closing(PostgresDatabase(pg_schema, lock_timeout=good)) as db, db.transaction():
            pass


class TestPgEvents(EventStoreContract):
    """The shared event-store contract on PostgreSQL, plus what only PostgreSQL has to prove."""

    @pytest.fixture(autouse=True)
    def _databases(self, pg_schema):
        self.url = pg_schema
        self.dbs: list[PostgresDatabase] = []
        yield
        for db in self.dbs:
            db.close()

    def open(self) -> PostgresDatabase:
        db = PostgresDatabase(self.url)
        self.dbs.append(db)
        return db

    def make(self):
        return self.open().events

    def test_append_positions_follow_commit_order(self):
        first, second = self.open(), self.open()
        for i in range(6):
            (first if i % 2 == 0 else second).events.append(ev(f"e{i}", "x"))

        rows = list(first.events.since(0))
        assert [seq for seq, _ in rows] == [1, 2, 3, 4, 5, 6]
        assert [e.id for _, e in rows] == [f"e{i}" for i in range(6)]

    def test_rolled_back_append_leaves_no_gap(self):
        db = self.open()
        with pytest.raises(RuntimeError, match="boom"), db.transaction():
            db.events.append(ev("lost", "x"))
            raise RuntimeError("boom")

        db.events.append(ev("kept", "x"))

        assert [(seq, e.id) for seq, e in db.events.since(0)] == [(1, "kept")]

    def test_duplicate_id_leaves_the_transaction_usable(self):
        db = self.open()
        db.events.append(ev("e1", "a"))
        with db.transaction():
            with pytest.raises(ValueError, match="duplicate event id: e1"):
                db.events.append(ev("e1", "b"))
            db.events.append(ev("e2", "c"))

        assert [e.id for _, e in db.events.since(0)] == ["e1", "e2"]

    def test_round_trip_keeps_timezone_microseconds_and_json_key_order(self):
        at = datetime(2026, 11, 16, 9, 30, 15, 123456, tzinfo=timezone(timedelta(hours=9)))
        original = Event(
            id="e1", kind="record", at=at, user="u1", team="t1", session="s1", scope="team",
            text="x", conditions={"zeta": 1, "alpha": {"y": 2, "b": 3}, "mid": [3, 1]}, refs=("h2", "h1"),
        )  # fmt: skip
        store = self.make()
        store.append(original)

        (got,) = store.get(["e1"])
        assert got == original
        assert got.at.utcoffset() == timedelta(hours=9) and got.at.microsecond == 123456
        assert list(got.conditions) == ["zeta", "alpha", "mid"]
        assert list(got.conditions["alpha"]) == ["y", "b"]
        assert got.refs == ("h2", "h1")

    def test_hidden_events_do_not_change_visible_scores(self):
        store = self.make()
        store.append(ev("mine", "release notes", user="u1"))
        store.append(ev("other", "canary notes", user="u1"))
        before = store.search("release canary", caller(), "personal")

        store.append(ev("hidden1", "release release release", user="u2"))
        store.append(ev("hidden2", "release canary", user="u2", team="t2", scope="team"))

        after = store.search("release canary", caller(), "personal")
        assert [(e.id, score) for e, score in after] == [(e.id, score) for e, score in before]
        assert len(after) == 2

    def test_search_ignores_empty_query_and_non_positive_k(self):
        store = self.make()
        store.append(ev("e1", "deploy"))

        assert store.search("  ", caller(), "personal") == []
        assert store.search("deploy", caller(), "personal", k=0) == []

    def test_empty_text_is_stored_but_matches_nothing(self):
        store = self.make()
        store.append(ev("e1", ""))

        assert [e.id for _, e in store.since(0)] == ["e1"]
        assert store.search("deploy", caller(), "personal") == []


class TestPgKnowledge(KnowledgeStoreContract):
    """The shared knowledge-store contract on PostgreSQL, plus what only PostgreSQL has to prove."""

    @pytest.fixture(autouse=True)
    def _databases(self, pg_schema):
        self.url = pg_schema
        self.dbs: list[PostgresDatabase] = []
        yield
        for db in self.dbs:
            db.close()

    def open(self) -> PostgresDatabase:
        db = PostgresDatabase(self.url)
        self.dbs.append(db)
        return db

    def make(self):
        return self.open().knowledge

    def test_knowledge_is_created_once(self):
        db = self.open()

        assert db.knowledge is db.knowledge

    def test_put_replaces_terms(self, kb):
        kb.put(item("k1", "canary rollout"))
        kb.put(item("k1", "blue green"))

        assert kb.search("canary", caller(), "team") == []
        assert [i.id for i, _ in kb.search("green", caller(), "team")] == ["k1"]
        db = kb.db
        assert db.query("SELECT count(*) FROM knowledge_stats WHERE id = 'k1'") == [(1,)]
        assert db.query("SELECT count(*) FROM knowledge_terms WHERE term = 'canary'") == [(0,)]

    def test_remove_drops_terms_and_stats(self, kb):
        kb.put(item("k1", "canary rollout"))
        kb.remove("k1")

        assert kb.db.query("SELECT count(*) FROM knowledge_terms") == [(0,)]
        assert kb.db.query("SELECT count(*) FROM knowledge_stats") == [(0,)]
        assert kb.search("canary", caller(), "team") == []

    def test_get_keeps_requested_order_and_skips_unknown(self, kb):
        for id in ["a", "b", "c"]:
            kb.put(item(id))

        assert [i.id for i in kb.get(["c", "x", "a"])] == ["c", "a"]

    def test_ids_order_by_bytes_not_locale(self, kb):
        for id in ["b", "B", "a", "_"]:
            kb.put(item(id))

        assert [i.id for i in kb.all()] == ["B", "_", "a", "b"]
        assert [i.id for i, _ in kb.search("deploy", caller(), "team")] == ["B", "_", "a", "b"]

    def test_json_fields_round_trip_exactly(self, kb):
        original = Knowledge(
            id="k1", kind="knowledge", title="t", body="deploy", scope="team", owner="t1",
            sources=("e2", "e1"), conditions={"z": "1", "a": "2", "m": "3"},
        )  # fmt: skip
        kb.put(original)

        (got,) = kb.get(["k1"])
        assert list(got.conditions) == ["z", "a", "m"]
        assert got.sources == ("e2", "e1")
        assert got == original

    def test_hidden_items_do_not_change_a_visible_score(self, kb):
        kb.put(item("mine", "canary rollout"))
        kb.put(item("other", "plain text"))
        before = kb.search("canary", caller(), "team")

        kb.put(item("hidden1", "canary canary canary", scope="team", owner="t2"))
        kb.put(item("hidden2", "canary " * 40, scope="personal", owner="u2"))

        assert kb.search("canary", caller(), "team") == before


class TestPgState(EngineStateBackendContract):
    """The shared engine-state contract on PostgreSQL, plus key order and engine isolation."""

    @pytest.fixture(autouse=True)
    def _databases(self, pg_schema):
        self.url = pg_schema
        self.dbs: list[PostgresDatabase] = []
        yield
        for db in self.dbs:
            db.close()

    def open(self) -> PostgresDatabase:
        db = PostgresDatabase(self.url)
        self.dbs.append(db)
        return db

    def make(self):
        return self.open().state("engine")

    def test_values_keep_key_order(self):
        state = self.make()
        state.commit(0, 1, {"k": {"b": 1, "a": 2}}, set(), epoch=state.epoch())

        assert list(state.get("k")) == ["b", "a"]
        (_, value), = state.scan()
        assert list(value) == ["b", "a"]

    def test_two_engines_do_not_share_state(self):
        db = self.open()
        a, b = db.state("a"), db.state("b")
        a.commit(0, 2, {"k": "from a", "only-a": 1}, set(), epoch=a.epoch())
        b.commit(0, 5, {"k": "from b"}, set(), epoch=b.epoch())

        assert (a.get("k"), a.cursor()) == ("from a", 2)
        assert (b.get("k"), b.cursor()) == ("from b", 5)
        assert list(b.scan()) == [("k", "from b")]
        b.reset("9")
        assert (a.get("k"), a.cursor(), a.version()) == ("from a", 2, None)
        assert list(a.scan()) == [("k", "from a"), ("only-a", 1)]


def _hnsw_indexes(db: PostgresDatabase) -> set[str]:
    rows = db.query(
        "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema() "
        "AND tablename IN ('event_vectors', 'knowledge_vectors') AND indexdef LIKE '%%hnsw%%'"
    )
    return {r[0] for r in rows}


class TestPgVectors(VectorBackendContract):
    """The shared vector contract on PostgreSQL (pgvector), plus dimensions and HNSW."""

    @pytest.fixture(autouse=True)
    def _databases(self, pg_schema):
        self.url = pg_schema
        self.dbs: list[PostgresDatabase] = []
        yield
        for db in self.dbs:
            db.close()

    def open(self, vector_index: str = "none") -> PostgresDatabase:
        db = PostgresDatabase(self.url, vector_index=vector_index)
        self.dbs.append(db)
        return db

    def make(self):
        return self.open()

    def test_scores_match_python_cosine_at_1024_dimensions(self, db, vb):
        # The error grows with the dimension (8e-7 measured at 4096); 1024 is a real model's size.
        rng = random.Random(1024)
        vb.reset({"model": "m1", "group_threshold": 0.8})
        query = [rng.gauss(0, 1) for _ in range(1024)]
        stored = {}
        for i in range(40):
            noise = 0.1 + i * 0.1  # from near duplicates to almost unrelated
            vector = [x + rng.gauss(0, noise) for x in query]
            self.index_event(db, ev(f"e{i}", "x"), vector, i + 1)
            stored[f"e{i}"] = f32(vector)

        got = vb.search_events(query, caller(), "personal", 40)

        assert len(got) == 40
        for id, score in got:
            assert score == pytest.approx(cosine(stored[id], f32(query)), abs=SCORE_TOLERANCE)

    def test_vectors_is_one_backend_per_database(self, db):
        assert db.vectors is db.vectors
        assert db.vectors.kind == "postgres"

    def test_mixed_dimensions_in_one_index_version_are_refused(self, db, vb):
        vb.reset({"model": "m1", "group_threshold": 0.8})
        vb.put_vector("event", "e1", [1.0, 0.0])
        with pytest.raises(ValueError, match="dimension"):
            vb.put_vector("event", "e2", [1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="dimension"):
            vb.put_vector("knowledge", "k1", [1.0])
        with pytest.raises(ValueError, match="dimension"):
            vb.put_vector("event", "e1", [1.0])  # replacing does not change it either
        with pytest.raises(ValueError):
            vb.put_vector("event", "e3", [])
        # Refused before anything is written, so a surrounding transaction still commits.
        with db.transaction():
            with pytest.raises(ValueError):
                vb.put_vector("event", "e4", [1.0, 0.0, 0.0])
            vb.put_vector("event", "e5", [0.0, 1.0])
        assert [i for i, _ in vb.vectors("event")] == ["e1", "e5"]
        assert list(vb.vectors("knowledge")) == []
        # The cache is not an index: other models may have other dimensions.
        vb.cache("m2", "d1", [1.0, 2.0, 3.0])
        assert vb.cached("m2", "d1") == [1.0, 2.0, 3.0]
        # A query of another dimension is refused too, rather than failing in pgvector.
        with pytest.raises(ValueError, match="dimension"):
            vb.search_events([1.0], caller(), "org", 5)
        # A new index version may have a new dimension.
        vb.reset({"model": "m2", "group_threshold": 0.8})
        vb.put_vector("event", "e1", [1.0, 0.0, 0.0])
        assert vb.vector("event", "e1") == [1.0, 0.0, 0.0]

    def test_non_finite_vectors_are_refused(self, db, vb):
        for bad in ([math.nan, 0.0], [math.inf, 0.0], [1e39, 0.0]):
            with pytest.raises(ValueError):
                vb.put_vector("event", "e1", bad)
            with pytest.raises(ValueError):
                vb.put_vector("knowledge", "k1", bad)
            with pytest.raises(ValueError):
                vb.cache("m1", "d1", bad)
            with pytest.raises(ValueError):
                vb.search_events(bad, caller(), "org", 5)
            with pytest.raises(ValueError):
                vb.search_knowledge(bad, caller(), "org", 5)
            with pytest.raises(ValueError):
                vb.nearest_member("team", "t1", bad, 10)
        assert list(vb.vectors("event")) == list(vb.vectors("knowledge")) == []
        assert vb.cached("m1", "d1") is None

    def test_zero_vectors_score_zero_as_in_python(self, db, vb):
        # pgvector gives NaN for the cosine of a zero vector; the other stores give 0.0.
        rows = [("a", [0.0, 0.0]), ("b", [0.0, 1.0]), ("c", [1.0, 0.0])]
        for seq, (id, vector) in enumerate(rows, start=1):
            self.index_event(db, ev(id, scope="team"), vector, seq)
        assert vb.nearest_member("team", "t1", [0.0, 0.0], 10) == ("a", 0.0)
        assert vb.nearest_member("team", "t1", [-1.0, 0.0], 10) == ("a", 0.0)
        assert [i for i, _ in vb.search_events([1.0, 0.0], caller(), "team", 5)] == ["c"]
        assert vb.search_events([0.0, 0.0], caller(), "team", 5) == []

    def test_reset_drops_vectors_and_the_hnsw_index_but_keeps_the_cache(self):
        db = self.open("hnsw")
        vb = db.vectors
        vb.reset({"model": "m1", "group_threshold": 0.8})
        vb.cache("m1", "d1", [0.5, 0.5])
        vb.put_vector("event", "e1", [1.0, 0.0])
        vb.put_vector("knowledge", "k1", [1.0, 0.0])
        vb.put_group("e1", "g1", "team", "t1", 1)
        vb.set_indexed(0, 3, vb.generation())
        assert _hnsw_indexes(db) == {"event_vectors_hnsw_d2", "knowledge_vectors_hnsw_d2"}

        vb.reset({"model": "m2", "group_threshold": 0.9})

        assert _hnsw_indexes(db) == set()
        assert list(vb.vectors("event")) == list(vb.vectors("knowledge")) == []
        assert vb.group_of("e1") is None
        assert vb.indexed() == 0
        assert vb.cached("m1", "d1") == [0.5, 0.5]
        # The next index version builds its own index for its own dimension.
        vb.put_vector("event", "e1", [1.0, 0.0, 0.0])
        assert _hnsw_indexes(db) == {"event_vectors_hnsw_d3", "knowledge_vectors_hnsw_d3"}
        # A database without HNSW still drops an index another process left behind.
        self.open("none").vectors.reset({"model": "m3", "group_threshold": 0.9})
        assert _hnsw_indexes(db) == set()

    def test_hnsw_indexes_come_back_after_a_rollback_or_another_process_reset(self):
        db = self.open("hnsw")
        vb = db.vectors
        vb.reset({"model": "m1", "group_threshold": 0.8})
        with pytest.raises(RuntimeError, match="later failure"), db.transaction():
            vb.put_vector("event", "e1", [1.0, 0.0])  # builds the indexes, then rolls back
            raise RuntimeError("later failure")
        assert _hnsw_indexes(db) == set()
        vb.put_vector("event", "e1", [1.0, 0.0])
        assert _hnsw_indexes(db) == {"event_vectors_hnsw_d2", "knowledge_vectors_hnsw_d2"}

        self.open("hnsw").vectors.reset({"model": "m2", "group_threshold": 0.8})  # another process
        assert _hnsw_indexes(db) == set()
        vb.put_vector("event", "e2", [0.0, 1.0])  # same D as before, in the new generation
        assert _hnsw_indexes(db) == {"event_vectors_hnsw_d2", "knowledge_vectors_hnsw_d2"}

    def test_a_zero_vector_scores_0_as_in_python(self, db, vb):
        vb.reset({"model": "m1", "group_threshold": 0.0})
        vb.put_vector("event", "e1", [0.0, 0.0])
        vb.put_group("e1", "e1", "team", "t1", 1)
        assert vb.nearest_member("team", "t1", [1.0, 0.0], 1) == ("e1", 0.0)
        assert vb.nearest_member("team", "t1", [0.0, 0.0], 1) == ("e1", 0.0)

    def test_no_hnsw_index_without_the_setting(self, db, vb):
        vb.reset({"model": "m1", "group_threshold": 0.8})
        vb.put_vector("event", "e1", [1.0, 0.0])
        assert _hnsw_indexes(db) == set()

    def test_hnsw_search_finds_the_exact_top_k_on_a_small_set(self):
        exact, hnsw = self.open("none"), self.open("hnsw")
        hnsw.vectors.reset({"model": "m1", "group_threshold": 0.8})
        rng = random.Random(7)
        dim = 8
        owners = {"personal": "u1", "team": "t1", "org": "*"}
        for seq in range(1, 51):
            scope = ("personal", "team", "org")[seq % 3]
            kind = "recall" if seq % 5 == 0 else "record"
            vector = [rng.uniform(-1, 1) for _ in range(dim)]
            self.index_event(hnsw, ev(f"e{seq:02d}", user=("u1", "u2")[seq % 2], scope=scope, kind=kind), vector, seq)
            hnsw.knowledge.put(item(f"k{seq:02d}", scope=scope, owner=owners[scope]))
            hnsw.vectors.put_vector("knowledge", f"k{seq:02d}", vector)
        assert _hnsw_indexes(hnsw) == {f"event_vectors_hnsw_d{dim}", f"knowledge_vectors_hnsw_d{dim}"}

        found = 0
        for _ in range(10):
            q = [rng.uniform(-1, 1) for _ in range(dim)]
            for who, scope, k in ((("u1", "t1"), "org", 5), (("u2", "t1"), "team", 3), (("u1", "t1"), "personal", 10)):
                me = caller(*who)
                for kw in ({}, {"kinds": {"record"}, "upto": 40, "exclude": "e03"}):
                    want = exact.vectors.search_events(q, me, scope, k, **kw)
                    self.scores_match(hnsw.vectors.search_events(q, me, scope, k, **kw), want)
                    found += len(want)
                want = exact.vectors.search_knowledge(q, me, scope, k)
                self.scores_match(hnsw.vectors.search_knowledge(q, me, scope, k), want)
                found += len(want)
        assert found > 100  # the comparison is not between empty lists

    def test_too_many_dimensions_for_hnsw_warn_and_search_exactly(self, caplog):
        db = self.open("hnsw")
        vb = db.vectors
        vb.reset({"model": "big", "group_threshold": 0.8})
        self.index_event(db, ev("e1"), [1.0] * 4001, 1)
        assert _hnsw_indexes(db) == set()
        assert "4001" in caplog.text
        assert vb.search_events([1.0] * 4001, caller(), "personal", 5) == [("e1", pytest.approx(1.0))]

    def test_hnsw_turned_on_for_an_existing_index_version_builds_its_index(self):
        before = self.open("none")
        before.vectors.reset({"model": "m1", "group_threshold": 0.8})
        rng = random.Random(3)
        for seq in range(1, 31):
            vector = [rng.uniform(-1, 1) for _ in range(4)]
            self.index_event(before, ev(f"e{seq:02d}", scope="org"), vector, seq)
            before.knowledge.put(item(f"k{seq:02d}", scope="org", owner="*"))
            before.vectors.put_vector("knowledge", f"k{seq:02d}", vector)
        assert _hnsw_indexes(before) == set()

        hnsw = self.open("hnsw")  # opening learns D from the index version and builds the index
        assert _hnsw_indexes(hnsw) == {"event_vectors_hnsw_d4", "knowledge_vectors_hnsw_d4"}
        q, me = [0.3, -0.2, 0.9, 0.1], caller()
        self.scores_match(
            hnsw.vectors.search_events(q, me, "org", 7), before.vectors.search_events(q, me, "org", 7)
        )
        self.scores_match(
            hnsw.vectors.search_knowledge(q, me, "org", 7), before.vectors.search_knowledge(q, me, "org", 7)
        )
        # Without its index a search is exact rather than a halfvec scan.
        hnsw.query("DROP INDEX event_vectors_hnsw_d4")
        assert hnsw.vectors.search_events(q, me, "org", 30) == before.vectors.search_events(q, me, "org", 30)

    def test_halfvec_overflow_is_refused_with_hnsw_on(self, caplog):
        db = self.open("hnsw")
        vb = db.vectors
        vb.reset({"model": "m1", "group_threshold": 0.8})
        vb.put_vector("event", "e1", [1.0, 0.0])
        for big in ([70000.0, 0.0], [0.0, -65505.0]):
            with pytest.raises(ValueError, match="halfvec"):
                vb.put_vector("event", "e2", big)
            with pytest.raises(ValueError, match="halfvec"):
                vb.put_vector("knowledge", "k1", big)
            with pytest.raises(ValueError, match="halfvec"):
                vb.search_events(big, caller(), "org", 5)
            with pytest.raises(ValueError, match="halfvec"):
                vb.search_knowledge(big, caller(), "org", 5)
        # Refused before anything is written, so a surrounding transaction still commits.
        with db.transaction():
            with pytest.raises(ValueError):
                vb.put_vector("event", "e3", [70000.0, 0.0])
            vb.put_vector("event", "e4", [65504.0, -65504.0])  # the largest halfvec
        assert [i for i, _ in vb.vectors("event")] == ["e1", "e4"]

        # Without HNSW such vectors are fine; turning HNSW on then skips that table's index
        # (building it would fail) and searches it exactly.
        plain = self.open("none")
        plain.vectors.reset({"model": "m2", "group_threshold": 0.8})
        self.index_event(plain, ev("big"), [70000.0, 1.0], 1)
        reopened = self.open("hnsw")
        assert _hnsw_indexes(reopened) == {"knowledge_vectors_hnsw_d2"}
        assert "halfvec" in caplog.text
        got = reopened.vectors.search_events([1.0, 0.0], caller(), "personal", 5)
        self.scores_match(got, [("big", 1.0)])


class TestPgDatabase(DatabaseContract):
    """The shared multi-process contract: every opener call is a new connection to one schema."""

    @pytest.fixture(autouse=True)
    def _databases(self, pg_schema):
        self.url = pg_schema
        self.dbs: list[PostgresDatabase] = []
        yield
        for db in self.dbs:
            db.close()

    def make(self, tmp_path):
        def opener():
            db = PostgresDatabase(self.url)
            self.dbs.append(db)
            return db

        return opener

    def test_poison_event_with_a_db_error_is_skipped(self, open_db):
        """observe runs outside the write transaction, so a failed statement
        aborts no transaction: the event is counted, skipped after max_failures
        without being indexed again, and the next event is delivered."""
        import psycopg

        from cortex_mem.embed import HashEmbedder
        from cortex_mem.index import Index

        db = open_db()

        class Poison:
            def __init__(self):
                self.seen = []

            def observe(self, event, ctx):
                self.seen.append(event.id)
                ctx.state.put(event.id, True)
                if event.id == "e1":
                    db.query("SELECT 1/0")
                return []

            def tick(self, now, ctx):
                return []

            def handoff(self, query, caller, scope, ctx, k=6):
                return []

        engine = Poison()
        index = Index(db.vectors, HashEmbedder(), 0.9, transaction=db.transaction)
        rt = Runtime(engine, db.events, db.knowledge, db.state("poison"), db.transaction, max_failures=2,
                     index=index)

        with pytest.raises(psycopg.errors.DivisionByZero):
            rt.append(ev("e1", "x"))
        rt.append(ev("e2", "y"))  # the second attempt on e1 gives up, then e2 is delivered

        assert engine.seen == ["e1", "e1", "e2"]
        assert [(s.seq, s.event_id, s.attempts) for s in rt.skipped] == [(1, "e1", 2)]
        assert rt.skipped[0].error.startswith("DivisionByZero")
        assert rt.state.cursor() == 2 and rt.pending() == 0
        assert [k for k, _ in rt.state.scan()] == ["e2"]
        assert db.vectors.indexed() >= rt.state.cursor()
        assert rt._failures == {}

        later = Poison()
        other = open_db()
        Runtime(later, other.events, other.knowledge, other.state("poison"), other.transaction,
                max_failures=2, index=Index(other.vectors, HashEmbedder(), 0.9,
                                            transaction=other.transaction)).sync()
        assert later.seen == []


# ── reconnect ───────────────────────────────────────────────────────────────


def _kill(db: PostgresDatabase) -> None:
    """End db's server session from another connection, as a restart or an idle timeout would."""
    import psycopg

    pid = db.con.info.backend_pid
    with psycopg.connect(db._url, autocommit=True) as admin:  # pyright: ignore[reportPrivateUsage]
        assert admin.execute("SELECT pg_terminate_backend(%s, 5000)", (pid,)).fetchone() == (True,)


def test_a_read_after_the_session_is_lost_reconnects(pg_schema):
    with closing(PostgresDatabase(pg_schema)) as db:
        db.events.append(ev("e1", "deploy uses tags"))
        before = db.con.info.backend_pid
        _kill(db)
        assert db.events.count_since(0) == 1
        assert db.con.info.backend_pid != before


def test_a_write_after_the_session_is_lost_reconnects(pg_schema):
    with closing(PostgresDatabase(pg_schema)) as db:
        db.events.append(ev("e1", "deploy uses tags"))
        _kill(db)
        db.events.append(ev("e2", "payments retry"))
        assert [e.id for _, e in db.events.since(0)] == ["e1", "e2"]


def test_a_session_lost_inside_a_transaction_is_not_retried(pg_schema):
    """A retry inside a transaction could repeat writes the server already dropped; the error is the
    caller's to see, and the next operation reconnects."""
    import psycopg

    with closing(PostgresDatabase(pg_schema)) as db:
        with pytest.raises(psycopg.OperationalError), db.transaction():
            db.query("INSERT INTO vector_meta (key, value) VALUES ('lost', '1')")
            _kill(db)
            db.query("INSERT INTO vector_meta (key, value) VALUES ('after', '1')")
        assert db.query("SELECT key FROM vector_meta") == []
        db.events.append(ev("e1", "deploy uses tags"))
        assert db.events.count_since(0) == 1


def test_a_new_session_keeps_the_lock_timeout(pg_schema):
    with closing(PostgresDatabase(pg_schema, lock_timeout="2s")) as db:
        _kill(db)
        assert db.query("SELECT current_setting('lock_timeout')") == [("2s",)]


def test_reconnecting_backs_off_while_the_server_stays_down(pg_schema):
    """Each failed attempt doubles the wait before the next, up to a cap, and calls in between fail at
    once without trying; a successful attempt starts over from the first wait."""
    now = [100.0]
    with closing(PostgresDatabase(pg_schema, clock=lambda: now[0])) as db:
        real_open = db._open  # pyright: ignore[reportPrivateUsage]
        attempts: list[float] = []

        def down() -> object:
            attempts.append(now[0])
            raise ConnectionError("cannot connect to PostgreSQL (host=x): server is down")

        db._open = down  # pyright: ignore[reportPrivateUsage]
        _kill(db)
        with pytest.raises(ConnectionError, match="server is down"):
            db.events.count_since(0)
        for _ in range(3):
            with pytest.raises(ConnectionError, match="unavailable"):
                db.events.count_since(0)  # within the first wait: no attempt
        assert attempts == [100.0]

        waits = []
        for _ in range(8):
            last = attempts[-1]
            while len(attempts) == len(waits) + 1:
                now[0] += 0.5
                with pytest.raises(ConnectionError):
                    db.events.count_since(0)
            waits.append(attempts[-1] - last)
        assert waits == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0]

        db._open = real_open  # pyright: ignore[reportPrivateUsage]
        now[0] += 30.0
        assert db.events.count_since(0) == 0  # the server is back
        db._open = down  # pyright: ignore[reportPrivateUsage]
        _kill(db)
        with pytest.raises(ConnectionError, match="server is down"):
            db.events.count_since(0)
        now[0] += 1.0
        with pytest.raises(ConnectionError, match="server is down"):
            db.events.count_since(0)  # the wait started over at 1 s


def test_an_hnsw_search_after_the_session_is_lost_reconnects(pg_schema):
    """HNSW searches run in a read-only transaction of their own, outside the write path."""
    with closing(PostgresDatabase(pg_schema, vector_index="hnsw")) as db:
        db.vectors.reset({"model": "m1", "group_threshold": 0.8})
        db.knowledge.put(item("k1", scope="team", owner="t1"))
        db.vectors.put_vector("knowledge", "k1", [1.0, 0.0])
        _kill(db)
        assert [id for id, _ in db.vectors.search_knowledge([1.0, 0.0], caller("u1", "t1"), "team", 5)] == ["k1"]
