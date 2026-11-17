"""PostgreSQL + pgvector store: one connection, one advisory-locked write transaction.

This is the only module that may import ``psycopg``, and only inside functions, so core
imports without the driver. Install it with the ``postgres`` extra.

transaction() is SQLite's ``BEGIN IMMEDIATE`` rebuilt on PostgreSQL: the outermost call
begins a transaction and takes one database-wide advisory lock (``TRANSACTION_LOCK``), so
writes from every process serialize and event positions commit in order (spec §4). Reads
take no lock and see committed rows.

Tables are created in the connection's current schema (the first one on ``search_path``),
never a hard-coded one.

The event, knowledge, state and vector stores land on top of this in later steps; until
then ``events``, ``knowledge``, ``vectors`` and ``state()`` raise ``NotImplementedError``.
"""

import hashlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

SCHEMA_VERSION = 1
VECTOR_INDEXES = ("none", "hnsw")
LOCK_TIMEOUT_SETTING = "CORTEX_DB_LOCK_TIMEOUT"

# The one key every cortex process locks in transaction(): the first 8 bytes of
# sha256("cortex-mem/store/transaction") as a signed 64-bit integer, which is what
# pg_advisory_xact_lock(bigint) takes. A fixed string keeps it stable across versions.
TRANSACTION_LOCK = int.from_bytes(
    hashlib.sha256(b"cortex-mem/store/transaction").digest()[:8], "big", signed=True
)

_SCHEMA = (
    (
        "CREATE TABLE IF NOT EXISTS events (seq bigint PRIMARY KEY, id text NOT NULL UNIQUE, "
        'kind text, at text, "user" text, team text, session text, scope text, text text, '
        "agent text, signal text, outcome text, conditions text, refs text, repo text, "
        "rev text, text_redacted boolean NOT NULL DEFAULT false)"
    ),
    "CREATE TABLE IF NOT EXISTS event_terms (seq bigint, term text, tf int, PRIMARY KEY (term, seq))",
    "CREATE TABLE IF NOT EXISTS event_stats (seq bigint PRIMARY KEY, len int)",
    (
        "CREATE TABLE IF NOT EXISTS knowledge (id text PRIMARY KEY, kind text, title text, "
        "body text, scope text, owner text, sources text, conditions text)"
    ),
    "CREATE TABLE IF NOT EXISTS knowledge_terms (id text, term text, tf int, PRIMARY KEY (term, id))",
    "CREATE TABLE IF NOT EXISTS knowledge_stats (id text PRIMARY KEY, len int)",
    "CREATE TABLE IF NOT EXISTS engine_state (engine text, key text, value text, PRIMARY KEY (engine, key))",
    "CREATE TABLE IF NOT EXISTS engine_cursor (engine text PRIMARY KEY, cursor bigint, version text)",
    "CREATE TABLE IF NOT EXISTS event_vectors (event_id text PRIMARY KEY, ord bigint, emb vector)",
    (
        "CREATE TABLE IF NOT EXISTS vector_groups (event_id text PRIMARY KEY, group_id text, "
        "scope text, owner text, seq bigint)"
    ),
    "CREATE INDEX IF NOT EXISTS vector_groups_partition ON vector_groups (scope, owner, seq)",
    "CREATE INDEX IF NOT EXISTS vector_groups_group ON vector_groups (group_id, seq)",
    "CREATE TABLE IF NOT EXISTS knowledge_vectors (id text PRIMARY KEY, emb vector)",
    (
        "CREATE TABLE IF NOT EXISTS embed_cache (model text, digest text, emb vector, "
        "PRIMARY KEY (model, digest))"
    ),
    "CREATE TABLE IF NOT EXISTS vector_meta (key text PRIMARY KEY, value text)",
)


def require_vector_type(conn: Any) -> None:
    """Raise RuntimeError unless the pgvector ``vector`` type resolves on ``conn``."""
    if conn.execute("SELECT to_regtype('vector')").fetchone()[0] is None:
        raise RuntimeError(
            "the pgvector extension is missing: the 'vector' type does not resolve on this "
            "connection. Install pgvector on the server and run CREATE EXTENSION vector in "
            "this database (or add the extension's schema to search_path)."
        )


def _describe(url: str) -> tuple[str, str | None]:
    """Return ("host=…, port=…, dbname=…", password) for ``url``; never the URL itself.

    The password is "" when the URL has none and None when the URL does not parse.
    """
    from psycopg import Error
    from psycopg.conninfo import conninfo_to_dict

    try:
        params = conninfo_to_dict(url)
    except Error:
        return "unparseable connection string", None
    where = f"host={params.get('host')}, port={params.get('port')}, dbname={params.get('dbname')}"
    return where, str(params.get("password") or "")


def _redact(text: str, *secrets: str) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


class PostgresDatabase:
    kind = "postgres"

    def __init__(self, url: str, lock_timeout: str = "30s", vector_index: str = "none") -> None:
        if vector_index not in VECTOR_INDEXES:
            raise ValueError(f"vector_index must be one of {VECTOR_INDEXES}, not {vector_index!r}")
        import psycopg  # fails early with a clear error if the extra is missing

        self._where, password = _describe(url)
        self.lock_timeout = lock_timeout
        self.vector_index = vector_index  # read when the vector store builds its indexes
        try:
            self.con = psycopg.connect(url, autocommit=True)
        except psycopg.Error as exc:
            # The driver's message names host and user, not the password; redact anyway.
            # A parse error may quote any fragment of the string, so it is not repeated.
            reason = (
                _redact(str(exc).strip(), url, password)
                if password is not None
                else "the connection string could not be parsed"
            )
            raise ConnectionError(f"cannot connect to PostgreSQL ({self._where}): {reason}") from None
        # MCP servers may call tools from worker threads; one lock serializes them.
        self._lock = threading.RLock()
        self._depth = 0
        try:
            require_vector_type(self.con)
            self._create()
        except BaseException:
            self.con.close()
            raise

    def __repr__(self) -> str:
        return f"PostgresDatabase({self._where})"

    def _create(self) -> None:
        # Under the write lock, so two processes starting on an empty schema do not race.
        with self.transaction():
            self.con.execute("CREATE TABLE IF NOT EXISTS cortex_meta (key text PRIMARY KEY, value text)")
            row = self.con.execute("SELECT value FROM cortex_meta WHERE key = 'schema_version'").fetchone()
            if row is not None and row[0] != str(SCHEMA_VERSION):
                raise RuntimeError(
                    f"database schema {row[0]} is not {SCHEMA_VERSION}; use a new database or schema"
                )
            for statement in _SCHEMA:
                self.con.execute(statement)
            if row is None:
                self.con.execute(
                    "INSERT INTO cortex_meta (key, value) VALUES ('schema_version', %s)",
                    (str(SCHEMA_VERSION),),
                )

    @property
    def events(self) -> Any:
        raise NotImplementedError("the PostgreSQL event store is not implemented yet")

    @property
    def knowledge(self) -> Any:
        raise NotImplementedError("the PostgreSQL knowledge store is not implemented yet")

    @property
    def vectors(self) -> Any:
        raise NotImplementedError("the PostgreSQL vector store is not implemented yet")

    def state(self, engine: str) -> Any:
        raise NotImplementedError("the PostgreSQL state store is not implemented yet")

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """One write transaction under the database-wide lock; nested calls join the outer one."""
        with self._lock:
            outer = self._depth == 0
            if outer:
                self._begin()
            self._depth += 1
            try:
                yield
            except BaseException:
                self._depth -= 1
                if outer:
                    self._rollback()
                raise
            self._depth -= 1
            if outer:
                self.con.execute("COMMIT")

    def _begin(self) -> None:
        from psycopg.errors import LockNotAvailable

        self.con.execute("BEGIN")
        try:
            # set_config(..., true) is SET LOCAL with a bound value.
            self.con.execute("SELECT set_config('lock_timeout', %s, true)", (self.lock_timeout,))
            self.con.execute("SELECT pg_advisory_xact_lock(%s)", (TRANSACTION_LOCK,))
        except LockNotAvailable:
            self._rollback()
            raise TimeoutError(
                f"waited {self.lock_timeout} for the cortex write lock, which another process "
                f"still holds; raise {LOCK_TIMEOUT_SETTING} to wait longer"
            ) from None
        except BaseException:
            self._rollback()
            raise

    def _rollback(self) -> None:
        # A broken connection has no transaction left to end; keep the original error.
        if not self.con.broken:
            self.con.execute("ROLLBACK")

    def query(self, sql: str, params: Any = ()) -> list[tuple]:
        with self._lock:
            cur = self.con.execute(sql, params)
            return cur.fetchall() if cur.description is not None else []

    def close(self) -> None:
        self.con.close()
