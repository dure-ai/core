"""PostgreSQL + pgvector store: one connection, one advisory-locked write transaction.

This is the only module that may import ``psycopg``, and only inside functions, so core
imports without the driver. Install it with the ``postgres`` extra.

transaction() is SQLite's ``BEGIN IMMEDIATE`` rebuilt on PostgreSQL: the outermost call
begins a transaction and takes one database-wide advisory lock (``TRANSACTION_LOCK``), so
writes from every process serialize and event positions commit in order (spec §4). Reads
take no lock and see committed rows. A transaction PostgreSQL aborted is never committed
silently: see transaction().

Tables are created in the connection's current schema (the first one on ``search_path``),
never a hard-coded one.

The event, knowledge, state and vector stores land on top of this in later steps; until
then ``events``, ``knowledge``, ``vectors`` and ``state()`` raise ``NotImplementedError``.
"""

import hashlib
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

SCHEMA_VERSION = 1
VECTOR_INDEXES = ("none", "hnsw")
LOCK_TIMEOUT_SETTING = "CORTEX_DB_LOCK_TIMEOUT"
# A PostgreSQL duration as lock_timeout takes it: a number with an optional unit (no unit
# means milliseconds). PostgreSQL still checks the range when transaction() applies it.
_DURATION = re.compile(r"\d+(\.\d+)?(us|ms|s|min|h|d)?")

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


class TransactionAborted(RuntimeError):
    """The outermost transaction() ended after PostgreSQL had aborted it; it was rolled back."""


def _bad_lock_timeout(value: str) -> ValueError:
    return ValueError(
        f"lock timeout {value!r} is not a PostgreSQL duration; set {LOCK_TIMEOUT_SETTING} "
        "to a value such as 30s, 500ms or 2min (0 waits forever)"
    )


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
        if not _DURATION.fullmatch(lock_timeout):
            raise _bad_lock_timeout(lock_timeout)
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
        """One write transaction under the database-wide lock; nested calls join the outer one.

        As in SQLite, an exception that leaves the outermost block rolls everything back,
        and an exception the caller catches inside it does not by itself. One difference:
        a failed statement makes PostgreSQL abort the whole transaction. If the caller
        catches that error and carries on, the outermost block rolls back at its end and
        raises TransactionAborted instead of committing, so no writes are lost silently.
        """
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
                self._commit()

    def _begin(self) -> None:
        from psycopg.errors import InvalidParameterValue, LockNotAvailable

        self.con.execute("BEGIN")
        try:
            # set_config(..., true) is SET LOCAL with a bound value.
            try:
                self.con.execute("SELECT set_config('lock_timeout', %s, true)", (self.lock_timeout,))
            except InvalidParameterValue:  # matches the pattern but PostgreSQL refuses it
                raise _bad_lock_timeout(self.lock_timeout) from None
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

    def _commit(self) -> None:
        from psycopg.pq import TransactionStatus

        if self.con.info.transaction_status == TransactionStatus.INERROR:
            # COMMIT would answer ROLLBACK without raising; say so instead.
            self._rollback()
            raise TransactionAborted(
                "a statement inside this transaction failed and the error was caught; "
                "PostgreSQL aborted the transaction, so it was rolled back and nothing "
                "in it was committed"
            )
        try:
            self.con.execute("COMMIT")
        except BaseException:
            # The server usually ends a failed COMMIT itself; if the transaction is still
            # open (the COMMIT never arrived), end it so the advisory lock is released.
            self._rollback()
            raise

    def _rollback(self) -> None:
        """End the open transaction if there is one; never mask the error being raised."""
        from psycopg import Error
        from psycopg.pq import TransactionStatus

        con = self.con
        if con.closed or con.broken or con.info.transaction_status == TransactionStatus.IDLE:
            return  # nothing left to end: the server drops a lost session's transaction
        try:
            con.execute("ROLLBACK")
        except Error:
            pass  # the connection died meanwhile; the caller's error is the one to see

    def query(self, sql: str, params: Any = ()) -> list[tuple]:
        """Run one statement and return its rows ([] when it returns none).

        Takes the in-process lock, so a call from another thread waits while this object
        has a transaction open. It takes no database lock: outside a transaction it sees
        committed rows, inside one it also sees that transaction's own writes.
        """
        with self._lock:
            cur = self.con.execute(sql, params)
            return cur.fetchall() if cur.description is not None else []

    def close(self) -> None:
        self.con.close()
