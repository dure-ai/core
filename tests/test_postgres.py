"""PostgreSQL database: connection, the one advisory-locked transaction, and the schema.

Every test uses the ``pg_schema`` fixture, so all of them skip without ``CORTEX_TEST_PG_URL``.
"""

import threading
from contextlib import closing

import pytest

from cortex_mem.store.postgres import (
    SCHEMA_VERSION,
    PostgresDatabase,
    require_vector_type,
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
    assert SCHEMA_VERSION == 1
    with closing(PostgresDatabase(pg_schema)) as db:
        assert db.kind == "postgres"
        assert _tables(db) == TABLES
        assert db.query("SELECT value FROM cortex_meta WHERE key = 'schema_version'") == [("1",)]
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


def test_unknown_schema_version_refuses_to_start(pg_schema):
    with closing(PostgresDatabase(pg_schema)) as db:
        db.query("UPDATE cortex_meta SET value = '99' WHERE key = 'schema_version'")
    with pytest.raises(RuntimeError, match=r"99.*\b1\b"):
        PostgresDatabase(pg_schema)


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
