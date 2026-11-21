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

``PgEventStore`` (``db.events``) is the append-only event log and ``PgKnowledgeStore``
(``db.knowledge``) the knowledge base, both with BM25 text search. ``PgStateBackend``
(``db.state(name)``) keeps engine state, and ``PgVectorBackend`` (``db.vectors``) keeps
pgvector vectors and scores cosine similarity in SQL (spec §6).

Shared SQL fragments are written with ``:name`` parameters and turned into psycopg's
``%(name)s`` by ``pg_params``. That rewrite does not parse string literals, so it would
change a ``:name`` inside one, and a literal ``%`` in any statement sent through psycopg
must be written ``%%``. The shared fragments contain neither.
"""

import hashlib
import json
import logging
import re
import threading
from array import array
from collections import Counter
from collections.abc import Collection, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from contextlib import contextmanager
from datetime import datetime
from typing import Any, LiteralString, cast

from ..contract import Caller, Event, Kind, Knowledge, Scope
from ..embed import float32
from .base import CursorConflict
from .text import EVENT_VISIBLE_SQL, KNOWLEDGE_VISIBLE_SQL, bigrams, pg_params, words

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
    (
        "CREATE TABLE IF NOT EXISTS engine_cursor (engine text PRIMARY KEY, cursor bigint, version text, "
        "epoch bigint NOT NULL DEFAULT 0)"
    ),
    "CREATE TABLE IF NOT EXISTS event_vectors (event_id text PRIMARY KEY, ord bigint, emb vector)",
    (
        "CREATE TABLE IF NOT EXISTS vector_groups (event_id text PRIMARY KEY, group_id text, "
        "scope text, owner text, seq bigint)"
    ),
    "CREATE INDEX IF NOT EXISTS vector_groups_partition ON vector_groups (scope, owner, seq)",
    "CREATE INDEX IF NOT EXISTS vector_groups_group ON vector_groups (group_id, seq)",
    "CREATE TABLE IF NOT EXISTS knowledge_vectors (id text PRIMARY KEY, emb vector, digest text)",
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


HNSW_MIN_PGVECTOR = (0, 8)  # hnsw.iterative_scan, set before every HNSW search, came in 0.8


def pgvector_version(text: str) -> tuple[int, ...]:
    """``"0.8.1"`` -> ``(0, 8, 1)``, so versions compare as numbers ("0.10" is after "0.8").
    A part that does not start with a digit ends the version."""
    parts = []
    for part in text.split("."):
        digits = re.match(r"\d+", part)
        if digits is None:
            break
        parts.append(int(digits.group()))
    return tuple(parts)


def require_hnsw_support(conn: Any) -> None:
    """Raise RuntimeError unless the installed pgvector is new enough for ``vector_index="hnsw"``."""
    row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    found = None if row is None else str(row[0])
    if found is None or pgvector_version(found) < HNSW_MIN_PGVECTOR:
        need = ".".join(map(str, HNSW_MIN_PGVECTOR))
        raise RuntimeError(
            f"vector_index='hnsw' needs pgvector {need} or later, but this database has "
            f"{'no vector extension version' if found is None else found}; upgrade pgvector "
            "(ALTER EXTENSION vector UPDATE) or use vector_index='none' (exact search, pgvector 0.7+)"
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
    """The ValueError for a lock timeout that is not a PostgreSQL duration,
    naming the setting to change."""
    return ValueError(
        f"lock timeout {value!r} is not a PostgreSQL duration; set {LOCK_TIMEOUT_SETTING} "
        "to a value such as 30s, 500ms or 2min (0 waits forever)"
    )


def check_lock_timeout(raw: str) -> str:
    """Return ``raw`` if it is a PostgreSQL duration, else raise ValueError naming the setting."""
    if not _DURATION.fullmatch(raw):
        raise _bad_lock_timeout(raw)
    return raw


def _text_ids(ids: list[Any]) -> list[str]:
    """The string ids in ``ids``: a text column compared with another type is an error in
    PostgreSQL (one that aborts the transaction), and such an id matches nothing anyway."""
    return [i for i in ids if isinstance(i, str)]


def _redact(text: str, *secrets: str) -> str:
    """text with every non-empty secret replaced by ***."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


class PostgresDatabase:
    """All stores in the connection's current schema, behind one autocommit
    connection. Writes go through transaction(), which serializes writers
    in every process with one advisory lock. Needs the pgvector extension."""
    kind = "postgres"

    def __init__(self, url: str, lock_timeout: str = "30s", vector_index: str = "none") -> None:
        """url: a libpq connection string or URL. lock_timeout: how long
        transaction() waits for the write lock, as a PostgreSQL duration
        ("0" waits forever). vector_index: "none" (exact search) or "hnsw"
        (see PgVectorBackend). Creates the tables if missing.

        Raises ValueError for a bad setting, ConnectionError (password
        redacted) when it cannot connect, and RuntimeError when pgvector (or,
        for HNSW, a recent enough pgvector) is missing or the schema is of
        another version."""
        if vector_index not in VECTOR_INDEXES:
            raise ValueError(f"vector_index must be one of {VECTOR_INDEXES}, not {vector_index!r}")
        check_lock_timeout(lock_timeout)
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
        try:
            # Once per connection rather than SET LOCAL in every transaction: it only
            # bounds lock waits, and this connection takes no lock but the advisory one
            # (and DDL in _create). set_config binds the value.
            self.con.execute("SELECT set_config('lock_timeout', %s, false)", (lock_timeout,))
        except psycopg.errors.InvalidParameterValue:  # matches the pattern but PostgreSQL refuses it
            self.con.close()
            raise _bad_lock_timeout(lock_timeout) from None
        # MCP servers may call tools from worker threads; one lock serializes them.
        self._lock = threading.RLock()
        self._depth = 0
        self._events = PgEventStore(self)
        self._knowledge = PgKnowledgeStore(self)
        self._vectors = PgVectorBackend(self)
        try:
            require_vector_type(self.con)
            if vector_index == "hnsw":
                require_hnsw_support(self.con)
            self._create()
            self._vectors.load()
        except BaseException:
            self.con.close()
            raise

    def __repr__(self) -> str:
        """Host, port and database only, never the password."""
        return f"PostgresDatabase({self._where})"

    def _create(self) -> None:
        """Create the tables if missing, under the write lock so two processes
        starting on an empty schema do not race. RuntimeError for a schema of
        another version, or one from before the epoch and digest columns."""
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
            # Columns added on this branch without a version bump. PostgreSQL is
            # only in development until it ships, so an older schema is not migrated.
            missing = [
                f"{table}.{column}"
                for table, column in (("engine_cursor", "epoch"), ("knowledge_vectors", "digest"))
                if not self.con.execute(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_schema = current_schema() AND table_name = %s AND column_name = %s",
                    (table, column),
                ).fetchone()
            ]
            if missing:
                raise RuntimeError(
                    f"the cortex schema in this database predates {', '.join(missing)}; "
                    "drop that schema (or use a new one) and start again"
                )
            if row is None:
                self.con.execute(
                    "INSERT INTO cortex_meta (key, value) VALUES ('schema_version', %s)",
                    (str(SCHEMA_VERSION),),
                )

    @property
    def events(self) -> "PgEventStore":
        return self._events

    @property
    def knowledge(self) -> "PgKnowledgeStore":
        return self._knowledge

    @property
    def vectors(self) -> "PgVectorBackend":
        return self._vectors

    def state(self, engine: str) -> "PgStateBackend":
        return PgStateBackend(self, engine)

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
                    self._vectors.forget_hnsw()  # indexes built inside it are gone
                raise
            self._depth -= 1
            if outer:
                try:
                    self._commit()
                except BaseException:
                    self._vectors.forget_hnsw()
                    raise

    def _begin(self) -> None:
        """BEGIN and take the database-wide advisory lock, waiting at most the
        connection's lock_timeout. TimeoutError when the lock stays taken past
        the timeout; on any failure the transaction is rolled back."""
        from psycopg.errors import LockNotAvailable

        self.con.execute("BEGIN")
        try:
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
        """COMMIT, unless PostgreSQL aborted the transaction: then roll back and
        raise TransactionAborted. A failed COMMIT is rolled back too, so the
        advisory lock is released."""
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

    def query(self, sql: str, params: Any = ()) -> list[tuple[Any, ...]]:
        """Run one statement and return its rows ([] when it returns none).

        Takes the in-process lock, so a call from another thread waits while this object
        has a transaction open. It takes no database lock: outside a transaction it sees
        committed rows, inside one it also sees that transaction's own writes.
        """
        with self._lock:
            cur = self.con.execute(_sql(sql), params)
            return cur.fetchall() if cur.description is not None else []

    def close(self) -> None:
        """Close the connection."""
        self.con.close()


_EVENT_COLUMNS = (
    'id, kind, at, "user", team, session, scope, text, agent, signal, outcome, '
    "conditions, refs, repo, rev, text_redacted"
)
BM25_K1 = 1.2
BM25_B = 0.75

# BM25 over event_terms. N, average length and df count only the events the caller may
# see: corpus-wide statistics would let a score reveal hidden events' words. kinds is a
# final filter and does not change them. Placeholders are psycopg's.
_EVENT_COLS = ", ".join("e." + c for c in _EVENT_COLUMNS.split(", "))
_SEARCH_SQL = f"""
WITH vis AS MATERIALIZED (
    SELECT e.seq FROM events e WHERE {pg_params(EVENT_VISIBLE_SQL)}
), hits AS MATERIALIZED (
    -- Only the query terms' postings, found from the term index; every CTE is computed
    -- once, so the cost is one pass over the visible events plus the postings.
    SELECT t.seq, t.term, t.tf
    FROM event_terms t JOIN vis ON vis.seq = t.seq
    WHERE t.term = ANY(%(terms)s)
), df AS (
    SELECT term, count(*)::float8 AS df FROM hits GROUP BY term
), corpus AS MATERIALIZED (
    SELECT count(*)::float8 AS n, avg(s.len)::float8 AS avgdl
    FROM vis JOIN event_stats s ON s.seq = vis.seq
)
SELECT {_EVENT_COLS},
       sum(ln((corpus.n - df.df + 0.5) / (df.df + 0.5) + 1)
           * h.tf * ({BM25_K1} + 1) / (h.tf + {BM25_K1} * (1 - {BM25_B} + {BM25_B} * s.len / corpus.avgdl))) AS score
FROM hits h
JOIN df ON df.term = h.term
JOIN events e ON e.seq = h.seq
JOIN event_stats s ON s.seq = h.seq
CROSS JOIN corpus
"""


def _row_to_event(row: tuple[Any, ...]) -> Event:
    """An Event from a row of _EVENT_COLUMNS."""
    id, kind, at, user, team, session, scope, text, agent, signal, outcome, conditions, refs, repo, rev, redacted = row
    return Event(
        id=id, kind=kind, at=datetime.fromisoformat(at), user=user, team=team, session=session,
        scope=scope, text=text, agent=agent, signal=signal, outcome=outcome,
        conditions=json.loads(conditions), refs=tuple(json.loads(refs)), repo=repo, rev=rev,
        text_redacted=bool(redacted),
    )  # fmt: skip


class PgEventStore:
    """The event log: events in append order (seq), with each event's term
    counts (event_terms, event_stats) for BM25 search whose statistics
    count only the events the caller may see."""
    def __init__(self, db: PostgresDatabase) -> None:
        self.db = db

    def append(self, event: Event) -> str:
        e = event
        tokens = Counter(words(e.text) + bigrams(e.text))
        with self.db.transaction():
            # Checked, not caught: a failed INSERT would abort the whole transaction.
            if self.db.con.execute("SELECT 1 FROM events WHERE id = %s", (e.id,)).fetchone():
                raise ValueError(f"duplicate event id: {e.id}")
            # Writers hold the database-wide lock, so max+1 is gap-free and in commit order.
            seq = _scalar(self.db.con.execute("SELECT coalesce(max(seq), 0) + 1 FROM events"))
            self.db.con.execute(
                _sql(f"INSERT INTO events (seq, {_EVENT_COLUMNS}) VALUES ({', '.join(['%s'] * 17)})"),
                (
                    seq, e.id, e.kind, e.at.isoformat(), e.user, e.team, e.session, e.scope, e.text,
                    e.agent, e.signal, e.outcome, json.dumps(dict(e.conditions)),
                    json.dumps(list(e.refs)), e.repo, e.rev, e.text_redacted,
                ),
            )  # fmt: skip
            if tokens:
                with self.db.con.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO event_terms (seq, term, tf) VALUES (%s, %s, %s)",
                        [(seq, term, tf) for term, tf in tokens.items()],
                    )
            self.db.con.execute(
                "INSERT INTO event_stats (seq, len) VALUES (%s, %s)", (seq, sum(tokens.values()))
            )
        return e.id

    def since(self, seq: int, page: int = 256) -> Iterator[tuple[int, Event]]:
        while True:
            rows = self.db.query(
                f"SELECT seq, {_EVENT_COLUMNS} FROM events WHERE seq > %s ORDER BY seq LIMIT %s", (seq, page)
            )
            for row in rows:
                seq = row[0]
                yield seq, _row_to_event(row[1:])
            if len(rows) < page:
                return

    def get(self, ids: Collection[str]) -> list[Event]:
        ids = list(ids)
        rows = self.db.query(f"SELECT {_EVENT_COLUMNS} FROM events WHERE id = ANY(%s)", (_text_ids(ids),))
        found = {row[0]: _row_to_event(row) for row in rows}
        return [found[i] for i in ids if i in found]

    def search(
        self,
        query: str,
        caller: Caller,
        scope: Scope,
        k: int = 20,
        kinds: Collection[Kind] | None = None,
    ) -> list[tuple[Event, float]]:
        terms = list(dict.fromkeys(words(query) + bigrams(query)))
        if not terms or k <= 0:
            return []
        sql = _SEARCH_SQL
        params: dict[str, Any] = {
            "terms": terms, "user": caller.user, "team": caller.team, "scope": scope, "k": k,
        }  # fmt: skip
        if kinds is not None:
            sql += " WHERE e.kind = ANY(%(kinds)s)"
            params["kinds"] = list(kinds)
        sql += " GROUP BY e.seq ORDER BY score DESC, e.seq LIMIT %(k)s"
        return [(_row_to_event(row[:-1]), row[-1]) for row in self.db.query(sql, params)]


_KNOWLEDGE_COLUMNS = "id, kind, title, body, scope, owner, sources, conditions"

# The same BM25 over knowledge_terms, but df, N and the average length count only the items
# the caller may see: corpus-wide statistics would let a score reveal hidden items' words.
_KNOWLEDGE_SEARCH_SQL = f"""
WITH vis AS (
    SELECT k.* FROM knowledge k WHERE {pg_params(KNOWLEDGE_VISIBLE_SQL)}
), df AS (
    SELECT t.term, count(*)::float8 AS df
    FROM knowledge_terms t JOIN vis ON vis.id = t.id WHERE t.term = ANY(%(terms)s) GROUP BY t.term
), corpus AS (
    SELECT count(*)::float8 AS n, avg(s.len)::float8 AS avgdl
    FROM vis JOIN knowledge_stats s ON s.id = vis.id
)
SELECT {", ".join("k." + c for c in _KNOWLEDGE_COLUMNS.split(", "))},
       sum(ln((corpus.n - df.df + 0.5) / (df.df + 0.5) + 1)
           * t.tf * ({BM25_K1} + 1) / (t.tf + {BM25_K1} * (1 - {BM25_B} + {BM25_B} * s.len / corpus.avgdl))) AS score
FROM knowledge_terms t
JOIN df ON df.term = t.term
JOIN vis k ON k.id = t.id
JOIN knowledge_stats s ON s.id = t.id
CROSS JOIN corpus
WHERE t.term = ANY(%(terms)s)
GROUP BY k.id, {", ".join("k." + c for c in _KNOWLEDGE_COLUMNS.split(", ")[1:])}
ORDER BY score DESC, k.id COLLATE "C" LIMIT %(k)s
"""


def _row_to_knowledge(row: tuple[Any, ...]) -> Knowledge:
    """A Knowledge item from a row of _KNOWLEDGE_COLUMNS."""
    id, kind, title, body, scope, owner, sources, conditions = row
    return Knowledge(
        id=id, kind=kind, title=title, body=body, scope=scope, owner=owner,
        sources=tuple(json.loads(sources)), conditions=json.loads(conditions),
    )  # fmt: skip


class PgKnowledgeStore:
    """The knowledge base, with each item's term counts (knowledge_terms,
    knowledge_stats) for BM25 search over what the caller may see."""
    def __init__(self, db: PostgresDatabase) -> None:
        self.db = db

    def put(self, item: Knowledge) -> None:
        tokens = Counter(words(f"{item.title} {item.body}") + bigrams(f"{item.title} {item.body}"))
        con = self.db.con
        with self.db.transaction():
            con.execute(
                _sql(
                    f"INSERT INTO knowledge ({_KNOWLEDGE_COLUMNS}) VALUES ({', '.join(['%s'] * 8)}) "
                    "ON CONFLICT (id) DO UPDATE SET kind = excluded.kind, title = excluded.title, "
                    "body = excluded.body, scope = excluded.scope, owner = excluded.owner, "
                    "sources = excluded.sources, conditions = excluded.conditions"
                ),
                (
                    item.id, item.kind, item.title, item.body, item.scope, item.owner,
                    json.dumps(list(item.sources)), json.dumps(dict(item.conditions)),
                ),
            )  # fmt: skip
            con.execute("DELETE FROM knowledge_terms WHERE id = %s", (item.id,))
            con.execute("DELETE FROM knowledge_stats WHERE id = %s", (item.id,))
            if tokens:
                with con.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO knowledge_terms (id, term, tf) VALUES (%s, %s, %s)",
                        [(item.id, term, tf) for term, tf in tokens.items()],
                    )
            con.execute(
                "INSERT INTO knowledge_stats (id, len) VALUES (%s, %s)", (item.id, sum(tokens.values()))
            )

    def remove(self, id: str) -> None:
        with self.db.transaction():
            for table in ("knowledge", "knowledge_terms", "knowledge_stats"):
                self.db.con.execute(f"DELETE FROM {table} WHERE id = %s", (id,))

    def clear(self) -> None:
        with self.db.transaction():
            for table in ("knowledge", "knowledge_terms", "knowledge_stats"):
                self.db.con.execute(f"DELETE FROM {table}")

    def get(self, ids: Collection[str]) -> list[Knowledge]:
        ids = list(ids)
        rows = self.db.query(
            f"SELECT {_KNOWLEDGE_COLUMNS} FROM knowledge WHERE id = ANY(%s)", (_text_ids(ids),)
        )
        found = {row[0]: _row_to_knowledge(row) for row in rows}
        return [found[i] for i in ids if i in found]

    def all(self) -> list[Knowledge]:
        rows = self.db.query(f'SELECT {_KNOWLEDGE_COLUMNS} FROM knowledge ORDER BY id COLLATE "C"')
        return [_row_to_knowledge(row) for row in rows]

    def search(
        self, query: str, caller: Caller, scope: Scope, k: int = 20
    ) -> list[tuple[Knowledge, float]]:
        terms = list(dict.fromkeys(words(query) + bigrams(query)))
        if not terms or k <= 0:
            return []
        params: dict[str, Any] = {
            "terms": terms, "user": caller.user, "team": caller.team, "scope": scope, "k": k,
        }  # fmt: skip
        return [(_row_to_knowledge(row[:-1]), row[-1]) for row in self.db.query(_KNOWLEDGE_SEARCH_SQL, params)]


class PgStateBackend:
    """One engine's key/value state and delivery cursor, as SQLite's backend does it.

    Values are stored as JSON text (not jsonb) so key order survives a round trip.
    """

    def __init__(self, db: PostgresDatabase, engine: str) -> None:
        self.db = db
        self.engine = engine

    def get(self, key: str) -> Any | None:
        rows = self.db.query(
            "SELECT value FROM engine_state WHERE engine = %s AND key = %s", (self.engine, key)
        )
        return json.loads(rows[0][0]) if rows else None

    def scan(self, prefix: str = "") -> Iterator[tuple[str, Any]]:
        # left() takes the prefix as data, so no LIKE escaping is needed.
        rows = self.db.query(
            'SELECT key, value FROM engine_state WHERE engine = %s AND left(key, length(%s)) = %s '
            'ORDER BY key COLLATE "C"',
            (self.engine, prefix, prefix),
        )
        for key, value in rows:
            yield key, json.loads(value)

    def cursor(self) -> int:
        rows = self.db.query("SELECT cursor FROM engine_cursor WHERE engine = %s", (self.engine,))
        return rows[0][0] if rows else 0

    def epoch(self) -> int:
        rows = self.db.query("SELECT epoch FROM engine_cursor WHERE engine = %s", (self.engine,))
        return rows[0][0] if rows else 0

    def commit(
        self,
        expected: int,
        cursor: int,
        writes: Mapping[str, Any],
        deletes: AbstractSet[str],
        *,
        epoch: int,
    ) -> None:
        # Serialize before any write, so a value that is not JSON leaves nothing half-written.
        encoded = [(self.engine, key, json.dumps(value)) for key, value in writes.items()]
        with self.db.transaction():
            rows = self.db.query("SELECT cursor, epoch FROM engine_cursor WHERE engine = %s", (self.engine,))
            current, current_epoch = rows[0] if rows else (0, 0)
            if current != expected or current_epoch != epoch:
                raise CursorConflict(
                    f"cursor is {current} at epoch {current_epoch}, expected {expected} at epoch {epoch}"
                )
            con = self.db.con
            with con.cursor() as cur:
                cur.executemany(
                    "DELETE FROM engine_state WHERE engine = %s AND key = %s",
                    [(self.engine, key) for key in deletes],
                )
                cur.executemany(
                    "INSERT INTO engine_state (engine, key, value) VALUES (%s, %s, %s) "
                    "ON CONFLICT (engine, key) DO UPDATE SET value = EXCLUDED.value",
                    encoded,
                )
            con.execute(
                "INSERT INTO engine_cursor (engine, cursor, epoch) VALUES (%s, %s, %s) "
                "ON CONFLICT (engine) DO UPDATE SET cursor = EXCLUDED.cursor, epoch = EXCLUDED.epoch",
                (self.engine, cursor, epoch + 1),
            )

    def version(self) -> str | None:
        rows = self.db.query("SELECT version FROM engine_cursor WHERE engine = %s", (self.engine,))
        return rows[0][0] if rows else None

    def position(self) -> tuple[int, int, str | None]:
        rows = self.db.query("SELECT epoch, cursor, version FROM engine_cursor WHERE engine = %s", (self.engine,))
        return (rows[0][0], rows[0][1], rows[0][2]) if rows else (0, 0, None)

    def reset(self, version: str) -> None:
        with self.db.transaction():
            con = self.db.con
            con.execute("DELETE FROM engine_state WHERE engine = %s", (self.engine,))
            con.execute(
                "INSERT INTO engine_cursor (engine, cursor, version, epoch) VALUES (%s, 0, %s, 1) "
                "ON CONFLICT (engine) DO UPDATE SET cursor = 0, version = EXCLUDED.version, "
                "epoch = engine_cursor.epoch + 1",
                (self.engine, version),
            )


log = logging.getLogger(__name__)

HNSW_MAX_DIM = 4000  # halfvec's HNSW limit; larger vectors are searched exactly
HALFVEC_MAX = 65504.0  # largest finite half-precision value; beyond it the halfvec cast fails
_HNSW_TABLES = ("event_vectors", "knowledge_vectors")
_KINDS = {"event": "event_vectors", "knowledge": "knowledge_vectors"}


def _sql(text: str) -> LiteralString:
    """Mark a statement built in this module as one psycopg may send as is.

    psycopg's types accept only literal strings, so that values cannot be pasted into SQL.
    The statements here interpolate only this module's own names (tables, columns, index
    names it created), integers and placeholder lists; every value goes in as a parameter.
    """
    return cast(LiteralString, text)


def _scalar(cursor: Any) -> Any:
    """The first column of a statement that always returns exactly one row (an aggregate, EXISTS)."""
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError("expected one row from an aggregate query")
    return row[0]


def _literal(vector: list[float]) -> str:
    """pgvector's text form. repr() of a float32-rounded value parses back to that same
    float32 (pgvector reads components with strtof)."""
    return "[" + ",".join(repr(x) for x in vector) + "]"


def _as_float32(values: list[float]) -> list[float]:
    """emb::real[] as Python floats of exactly the stored float32 values, whatever text
    form the driver parsed them from."""
    return array("f", values).tolist()


def _hnsw_name(table: str, dim: int) -> str:
    """Name of the HNSW index on `table` for dimension `dim`, such as
    event_vectors_hnsw_d256."""
    return f"{table}_hnsw_d{dim}"


# Cosine distance to the query, with pgvector's NaN for a zero vector read as distance 1
# (score 0), which is what cortex_mem.embed.cosine gives. PostgreSQL sorts NaN above every
# number and NaN > 0 is true there, so it must not reach ORDER BY or the score filter.
_DIST = "coalesce(nullif(v.emb <=> %(q)s::vector, 'NaN'), 1)"
_EVENT_FROM = (
    "FROM vector_groups g JOIN event_vectors v ON v.event_id = g.event_id "
    "JOIN events e ON e.id = g.event_id"
)
_KNOWLEDGE_FROM = "FROM knowledge_vectors v JOIN knowledge k ON k.id = v.id"


class PgVectorBackend:
    """Vectors as pgvector ``vector`` (float32), cosine computed in SQL: score =
    1 - (emb <=> q). The searches join events and knowledge under the shared visibility
    SQL, so one statement picks, scores and ranks.

    Every vector of one index version has one dimension D, kept in vector_meta as 'dim'
    from the first put after reset(); a vector or query of another dimension raises
    ValueError before anything is written. The embedding cache holds any dimension.

    Exact search is the default and the reference: measure with it. With
    ``vector_index="hnsw"`` the backend builds an HNSW index on emb::halfvec(D) for each
    vector table once it learns D (on the first put of an index version, or on opening a
    database that already has one; D <= 4000), and search_events / search_knowledge
    (never nearest_member) take candidates from it and rank them by the exact score.
    Those results are approximate: halfvec precision can put a slightly farther vector
    among the candidates instead of the nearest one, and under a selective visibility
    filter the iterative scan can stop at ``hnsw.max_scan_tuples`` and return fewer than
    k matches although more exist. A table without its index (D > 4000, or vectors
    written beyond halfvec range before HNSW was turned on) is searched exactly. With
    HNSW on, a vector or query component beyond halfvec range (|x| > 65504) raises
    ValueError before anything is written, so a caller's transaction is never aborted
    by the cast.
    """

    kind = "postgres"

    def __init__(self, db: PostgresDatabase) -> None:
        self.db = db
        # (generation, D) this backend last built or checked the indexes for. Another
        # generation (a reset by any process drops them) or a rollback makes it stale.
        self._hnsw_for: tuple[int, int] | None = None

    def load(self) -> None:
        """On opening the database: with HNSW on, build the indexes of an index version
        that already has a dimension (vectors written while HNSW was off)."""
        if self.db.vector_index != "hnsw":
            return
        with self.db.transaction():
            dim = self._dim()
            if dim is not None:
                self._ensure_hnsw(dim, self.generation())

    def forget_hnsw(self) -> None:
        """Called when a write transaction rolls back: indexes built inside it
        are gone, so the next put checks them again."""
        self._hnsw_for = None

    # Index state

    def _get_meta(self, key: str) -> str | None:
        """A vector_meta value ("meta", "generation", "indexed" or "dim"), or None."""
        rows = self.db.query("SELECT value FROM vector_meta WHERE key = %s", (key,))
        return rows[0][0] if rows else None

    def _set_meta(self, key: str, value: str) -> None:
        """Write a vector_meta value; call it inside a write transaction."""
        self.db.con.execute(
            "INSERT INTO vector_meta (key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            (key, value),
        )

    def meta(self) -> dict[str, Any] | None:
        value = self._get_meta("meta")
        return json.loads(value) if value is not None else None

    def reset(self, meta: dict[str, Any]) -> None:
        encoded = json.dumps(meta)
        with self.db.transaction():
            con = self.db.con
            for name in self._hnsw_indexes():
                con.execute(_sql(f'DROP INDEX IF EXISTS "{name}"'))
            for table in ("event_vectors", "knowledge_vectors", "vector_groups"):
                con.execute(f"DELETE FROM {table}")
            con.execute("DELETE FROM vector_meta WHERE key IN ('indexed', 'dim')")
            self._set_meta("meta", encoded)
            self._set_meta("generation", str(self.generation() + 1))
            self._hnsw_for = None

    def generation(self) -> int:
        value = self._get_meta("generation")
        return int(value) if value is not None else 0

    def indexed(self) -> int:
        value = self._get_meta("indexed")
        return int(value) if value is not None else 0

    def position(self) -> tuple[dict[str, Any] | None, int, int]:
        found = dict(self.db.query("SELECT key, value FROM vector_meta WHERE key IN ('meta', 'generation', 'indexed')"))
        meta = found.get("meta")
        return (
            json.loads(meta) if meta is not None else None,
            int(found.get("generation", 0)),
            int(found.get("indexed", 0)),
        )

    def set_indexed(self, expected: int, seq: int, generation: int) -> None:
        with self.db.transaction():
            _, now, current = self.position()
            if current != expected or now != generation:
                raise CursorConflict(
                    f"index at {current} (generation {now}), not {expected} (generation {generation})"
                )
            self._set_meta("indexed", str(seq))

    # Dimension and the HNSW index

    def _dim(self) -> int | None:
        """D of this index version: recorded, or else taken from a stored vector."""
        value = self._get_meta("dim")
        if value is not None:
            return int(value)
        rows = self.db.query(
            "SELECT vector_dims(emb) FROM event_vectors UNION ALL "
            "SELECT vector_dims(emb) FROM knowledge_vectors LIMIT 1"
        )
        return rows[0][0] if rows else None

    def _check_dim(self, vector: list[float], dim: int | None) -> None:
        """ValueError for an empty vector, or one whose length is not `dim`, the
        index version's dimension (None: not known yet, anything goes)."""
        if not vector:
            raise ValueError("a vector must have at least one dimension")
        if dim is not None and len(vector) != dim:
            raise ValueError(
                f"vector dimension {len(vector)} differs from this index version's dimension {dim}; "
                "reset the index to change models"
            )

    def _check_halfvec(self, vector: list[float]) -> None:
        """With HNSW on, refuse what the emb::halfvec(D) cast would fail on."""
        if self.db.vector_index == "hnsw" and any(abs(x) > HALFVEC_MAX for x in vector):
            raise ValueError(
                f"with the HNSW index on, vector components must be within halfvec range "
                f"(|x| <= {HALFVEC_MAX:g})"
            )

    def _hnsw_dim(self, table: str, dim: int | None) -> int | None:
        """D when searches of ``table`` go through its HNSW index (it exists), else None."""
        if self.db.vector_index != "hnsw" or dim is None or dim > HNSW_MAX_DIM:
            return None
        rows = self.db.query("SELECT to_regclass(%s) IS NOT NULL", (_hnsw_name(table, dim),))
        return dim if rows[0][0] else None

    def _hnsw_indexes(self) -> list[str]:
        """Names of the HNSW indexes on the vector tables of the current schema."""
        rows = self.db.query(
            "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema() "
            "AND tablename = ANY(%s) AND indexname ~ '_hnsw_d[0-9]+$'",
            (list(_HNSW_TABLES),),
        )
        return [r[0] for r in rows]

    def _ensure_hnsw(self, dim: int, generation: int) -> None:
        """Inside a write transaction, with HNSW on: build the missing indexes for D, once
        per D and index generation per backend (until a rollback). A table holding a
        component beyond halfvec range (written while HNSW was off) gets no index, since
        building it would fail; it is searched exactly."""
        self._hnsw_for = (generation, dim)
        if dim > HNSW_MAX_DIM:
            log.warning(
                "vectors have %d dimensions, more than HNSW's %d for halfvec; searching exactly",
                dim, HNSW_MAX_DIM,
            )  # fmt: skip
            return
        have = set(self._hnsw_indexes())
        for table in _HNSW_TABLES:
            name = _hnsw_name(table, dim)
            if name in have:
                continue
            overflow = _scalar(
                self.db.con.execute(
                    f"SELECT EXISTS (SELECT 1 FROM {table} v, unnest(v.emb::real[]) x WHERE abs(x) > %s)",
                    (HALFVEC_MAX,),
                )
            )
            if overflow:
                log.warning(
                    "%s holds components beyond halfvec range (|x| > %g); no HNSW index, "
                    "searching it exactly until the index is reset",
                    table, HALFVEC_MAX,
                )  # fmt: skip
                continue
            self.db.con.execute(
                _sql(f"CREATE INDEX IF NOT EXISTS {name} ON {table} USING hnsw ((emb::halfvec({dim})) halfvec_cosine_ops)")
            )

    # Vectors

    def put_vector(
        self, kind: str, id: str, vector: Sequence[float], digest: str | None = None
    ) -> None:
        table = _KINDS[kind]
        values = float32(vector)
        self._check_halfvec(values)  # before any write, so the transaction stays usable
        with self.db.transaction():
            recorded = self._get_meta("dim")
            dim = int(recorded) if recorded is not None else self._dim()
            self._check_dim(values, dim)  # likewise
            if recorded is None:
                self._set_meta("dim", str(len(values)))
            if self.db.vector_index == "hnsw":
                generation = self.generation()
                if self._hnsw_for != (generation, len(values)):
                    self._ensure_hnsw(len(values), generation)
            if kind == "event":
                # ord is assigned on first insert; replacing keeps it (and the position).
                self.db.con.execute(
                    "INSERT INTO event_vectors (event_id, ord, emb) VALUES "
                    "(%s, (SELECT coalesce(max(ord), 0) + 1 FROM event_vectors), %s::vector) "
                    "ON CONFLICT (event_id) DO UPDATE SET emb = EXCLUDED.emb",
                    (id, _literal(values)),
                )
            else:
                self.db.con.execute(
                    _sql(
                        f"INSERT INTO {table} (id, emb, digest) VALUES (%s, %s::vector, %s) "
                        "ON CONFLICT (id) DO UPDATE SET emb = EXCLUDED.emb, digest = EXCLUDED.digest"
                    ),
                    (id, _literal(values), digest),
                )

    def knowledge_digests(self, ids: Collection[str] | None = None) -> dict[str, str]:
        sql = "SELECT id, digest FROM knowledge_vectors WHERE digest IS NOT NULL"
        if ids is None:
            return dict(self.db.query(sql))
        return dict(self.db.query(sql + " AND id = ANY(%s)", (_text_ids(list(ids)),)))

    def vector(self, kind: str, id: str) -> list[float] | None:
        key = "event_id" if kind == "event" else "id"
        rows = self.db.query(f"SELECT emb::real[] FROM {_KINDS[kind]} WHERE {key} = %s", (id,))
        return _as_float32(rows[0][0]) if rows else None

    def vectors(self, kind: str, upto: int | None = None) -> Iterator[tuple[str, list[float]]]:
        if kind == "knowledge":
            if upto is not None:
                raise ValueError("upto bounds event vectors only")
            rows = self.db.query('SELECT id, emb::real[] FROM knowledge_vectors ORDER BY id COLLATE "C"')
        elif upto is None:
            rows = self.db.query("SELECT event_id, emb::real[] FROM event_vectors ORDER BY ord")
        else:
            rows = self.db.query(
                "SELECT v.event_id, v.emb::real[] FROM event_vectors v "
                "JOIN vector_groups g ON g.event_id = v.event_id WHERE g.seq <= %s ORDER BY v.ord",
                (upto,),
            )
        for id, values in rows:
            yield id, _as_float32(values)

    def delete_vectors(self, kind: str, ids: Collection[str]) -> None:
        key = "event_id" if kind == "event" else "id"
        with self.db.transaction():
            self.db.con.execute(_sql(f"DELETE FROM {_KINDS[kind]} WHERE {key} = ANY(%s)"), (list(ids),))

    # Groups

    def put_group(self, event_id: str, group_id: str, scope: Scope, owner: str, seq: int) -> None:
        with self.db.transaction():
            self.db.con.execute(
                "INSERT INTO vector_groups (event_id, group_id, scope, owner, seq) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (event_id) DO UPDATE SET group_id = EXCLUDED.group_id, scope = EXCLUDED.scope, "
                "owner = EXCLUDED.owner, seq = EXCLUDED.seq",
                (event_id, group_id, scope, owner, seq),
            )

    def group_of(self, event_id: str) -> str | None:
        rows = self.db.query("SELECT group_id FROM vector_groups WHERE event_id = %s", (event_id,))
        return rows[0][0] if rows else None

    def members(self, group_id: str, upto: int | None = None) -> list[str]:
        sql = "SELECT event_id FROM vector_groups WHERE group_id = %(group)s"
        if upto is not None:
            sql += " AND seq <= %(upto)s"
        rows = self.db.query(sql + ' ORDER BY seq, event_id COLLATE "C"', {"group": group_id, "upto": upto})
        return [r[0] for r in rows]

    def partition(self, scope: Scope, owner: str) -> list[str]:
        rows = self.db.query(
            "SELECT event_id FROM vector_groups WHERE scope = %s AND owner = %s "
            'ORDER BY seq, event_id COLLATE "C"',
            (scope, owner),
        )
        return [r[0] for r in rows]

    # Embedding cache: any dimension, since models differ.

    def cached(self, model: str, digest: str) -> list[float] | None:
        rows = self.db.query(
            "SELECT emb::real[] FROM embed_cache WHERE model = %s AND digest = %s", (model, digest)
        )
        return _as_float32(rows[0][0]) if rows else None

    def cache(self, model: str, digest: str, vector: Sequence[float]) -> None:
        values = float32(vector)
        if not values:
            raise ValueError("a vector must have at least one dimension")
        with self.db.transaction():
            self.db.con.execute(
                "INSERT INTO embed_cache (model, digest, emb) VALUES (%s, %s, %s::vector) "
                "ON CONFLICT (model, digest) DO UPDATE SET emb = EXCLUDED.emb",
                (model, digest, _literal(values)),
            )

    # Similarity

    def _query(self, vector: Sequence[float], halfvec: bool = False) -> tuple[str, int | None]:
        """The query as a pgvector literal, and D; ValueError on a bad query. halfvec: the
        query may be cast to halfvec (the searches, with HNSW on)."""
        values = float32(vector)
        if halfvec:
            self._check_halfvec(values)
        dim = self._dim()
        self._check_dim(values, dim)
        return _literal(values), dim

    def nearest_member(
        self, scope: Scope, owner: str, vector: Sequence[float], upto: int
    ) -> tuple[str, float] | None:
        q, _ = self._query(vector)
        rows = self.db.query(
            f"SELECT g.event_id, 1 - {_DIST} AS score "
            "FROM vector_groups g JOIN event_vectors v ON v.event_id = g.event_id "
            "WHERE g.scope = %(scope)s AND g.owner = %(owner)s AND g.seq <= %(upto)s "
            f'ORDER BY {_DIST}, g.seq, g.event_id COLLATE "C" LIMIT 1',
            {"q": q, "scope": scope, "owner": owner, "upto": upto},
        )
        return (rows[0][0], rows[0][1]) if rows else None

    def _ranked(
        self, table: str, candidates: str, ties: str, params: dict[str, Any], k: int, dim: int | None
    ) -> list[tuple[Any, ...]]:
        """(id, score) of the best k rows of ``candidates``, a SELECT of ``id``, its tie
        columns and ``dist`` (already limited to dist < 1, i.e. score > 0) whose FROM has
        ``table`` as ``v``. Exact unless ``table`` has its HNSW index: then the index picks
        candidates by halfvec distance and they are ranked here by the exact one, which is
        approximate (see the class docstring)."""
        params = {**params, "k": k}
        rank = f"ORDER BY dist, {ties} LIMIT %(k)s"
        hnsw = self._hnsw_dim(table, dim)
        if hnsw is None:
            return self.db.query(f"SELECT id, 1 - dist FROM ({candidates}) c {rank}", params)
        params["n"] = max(4 * k, 40)
        sql = (
            f"SELECT id, 1 - dist FROM ({candidates} "
            f"ORDER BY (v.emb::halfvec({hnsw})) <=> %(q)s::halfvec({hnsw}) LIMIT %(n)s) c {rank}"
        )
        with self.db._lock:
            if self.db._depth > 0:  # inside a write transaction: SET LOCAL lasts until it ends
                self.db.con.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
                return self.db.con.execute(_sql(sql), params).fetchall()
            # A read-only transaction of its own, so SET LOCAL has one to live in; it takes
            # no advisory lock, so a search never waits for writers.
            self.db.con.execute("BEGIN READ ONLY")
            try:
                self.db.con.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
                rows = self.db.con.execute(_sql(sql), params).fetchall()
            finally:
                self.db._rollback()
            return rows

    def search_events(
        self,
        vector: Sequence[float],
        caller: Caller,
        scope: Scope,
        k: int,
        kinds: Collection[Kind] | None = None,
        upto: int | None = None,
        exclude: str | None = None,
    ) -> list[tuple[str, float]]:
        """Exact, or approximate with HNSW on: then possibly fewer than k (class docstring)."""
        if k <= 0 or (kinds is not None and not kinds):
            return []
        q, dim = self._query(vector, halfvec=True)
        sql = (
            f"SELECT g.event_id AS id, g.seq AS seq, {_DIST} AS dist {_EVENT_FROM} "
            f"WHERE {pg_params(EVENT_VISIBLE_SQL)} AND {_DIST} < 1"
        )
        params: dict[str, Any] = {"q": q, "user": caller.user, "team": caller.team, "scope": scope}
        if upto is not None:
            sql += " AND g.seq <= %(upto)s"
            params["upto"] = upto
        if kinds is not None:
            sql += " AND e.kind = ANY(%(kinds)s)"
            params["kinds"] = list(kinds)
        if exclude is not None:
            sql += " AND g.event_id <> %(exclude)s"
            params["exclude"] = exclude
        return self._ranked("event_vectors", sql, 'seq, id COLLATE "C"', params, k, dim)

    def search_knowledge(
        self, vector: Sequence[float], caller: Caller, scope: Scope, k: int
    ) -> list[tuple[str, float]]:
        """Exact, or approximate with HNSW on: then possibly fewer than k (class docstring)."""
        if k <= 0:
            return []
        q, dim = self._query(vector, halfvec=True)
        sql = (
            f"SELECT v.id AS id, {_DIST} AS dist {_KNOWLEDGE_FROM} "
            f"WHERE {pg_params(KNOWLEDGE_VISIBLE_SQL)} AND {_DIST} < 1"
        )
        params = {"q": q, "user": caller.user, "team": caller.team, "scope": scope}
        return self._ranked("knowledge_vectors", sql, 'id COLLATE "C"', params, k, dim)
