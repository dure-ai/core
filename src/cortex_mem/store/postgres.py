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
from typing import Any

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
        self._events = PgEventStore(self)
        self._knowledge = PgKnowledgeStore(self)
        self._vectors = PgVectorBackend(self)
        try:
            require_vector_type(self.con)
            self._create()
            self._vectors.load()
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
WITH vis AS (
    SELECT e.seq FROM events e WHERE {pg_params(EVENT_VISIBLE_SQL)}
), df AS (
    SELECT t.term, count(*)::float8 AS df
    FROM event_terms t JOIN vis ON vis.seq = t.seq WHERE t.term = ANY(%(terms)s) GROUP BY t.term
), corpus AS (
    SELECT count(*)::float8 AS n, avg(s.len)::float8 AS avgdl
    FROM vis JOIN event_stats s ON s.seq = vis.seq
)
SELECT {_EVENT_COLS},
       sum(ln((corpus.n - df.df + 0.5) / (df.df + 0.5) + 1)
           * t.tf * ({BM25_K1} + 1) / (t.tf + {BM25_K1} * (1 - {BM25_B} + {BM25_B} * s.len / corpus.avgdl))) AS score
FROM event_terms t
JOIN df ON df.term = t.term
JOIN vis ON vis.seq = t.seq
JOIN events e ON e.seq = t.seq
JOIN event_stats s ON s.seq = t.seq
CROSS JOIN corpus
"""


def _row_to_event(row: tuple) -> Event:
    id, kind, at, user, team, session, scope, text, agent, signal, outcome, conditions, refs, repo, rev, redacted = row
    return Event(
        id=id, kind=kind, at=datetime.fromisoformat(at), user=user, team=team, session=session,
        scope=scope, text=text, agent=agent, signal=signal, outcome=outcome,
        conditions=json.loads(conditions), refs=tuple(json.loads(refs)), repo=repo, rev=rev,
        text_redacted=bool(redacted),
    )  # fmt: skip


class PgEventStore:
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
            seq = self.db.con.execute("SELECT coalesce(max(seq), 0) + 1 FROM events").fetchone()[0]
            self.db.con.execute(
                f"INSERT INTO events (seq, {_EVENT_COLUMNS}) VALUES ({', '.join(['%s'] * 17)})",
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
        rows = self.db.query(f"SELECT {_EVENT_COLUMNS} FROM events WHERE id = ANY(%s)", (ids,))
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


def _row_to_knowledge(row: tuple) -> Knowledge:
    id, kind, title, body, scope, owner, sources, conditions = row
    return Knowledge(
        id=id, kind=kind, title=title, body=body, scope=scope, owner=owner,
        sources=tuple(json.loads(sources)), conditions=json.loads(conditions),
    )  # fmt: skip


class PgKnowledgeStore:
    def __init__(self, db: PostgresDatabase) -> None:
        self.db = db

    def put(self, item: Knowledge) -> None:
        tokens = Counter(words(f"{item.title} {item.body}") + bigrams(f"{item.title} {item.body}"))
        con = self.db.con
        with self.db.transaction():
            con.execute(
                f"INSERT INTO knowledge ({_KNOWLEDGE_COLUMNS}) VALUES ({', '.join(['%s'] * 8)}) "
                "ON CONFLICT (id) DO UPDATE SET kind = excluded.kind, title = excluded.title, "
                "body = excluded.body, scope = excluded.scope, owner = excluded.owner, "
                "sources = excluded.sources, conditions = excluded.conditions",
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

    def get(self, ids: Collection[str]) -> list[Knowledge]:
        ids = list(ids)
        rows = self.db.query(f"SELECT {_KNOWLEDGE_COLUMNS} FROM knowledge WHERE id = ANY(%s)", (ids,))
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

    def commit(
        self, expected: int, cursor: int, writes: Mapping[str, Any], deletes: AbstractSet[str]
    ) -> None:
        # Serialize before any write, so a value that is not JSON leaves nothing half-written.
        encoded = [(self.engine, key, json.dumps(value)) for key, value in writes.items()]
        with self.db.transaction():
            current = self.cursor()
            if current != expected:
                raise CursorConflict(f"cursor is {current}, expected {expected}")
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
                "INSERT INTO engine_cursor (engine, cursor) VALUES (%s, %s) "
                "ON CONFLICT (engine) DO UPDATE SET cursor = EXCLUDED.cursor",
                (self.engine, cursor),
            )

    def version(self) -> str | None:
        rows = self.db.query("SELECT version FROM engine_cursor WHERE engine = %s", (self.engine,))
        return rows[0][0] if rows else None

    def reset(self, version: str) -> None:
        with self.db.transaction():
            con = self.db.con
            con.execute("DELETE FROM engine_state WHERE engine = %s", (self.engine,))
            con.execute(
                "INSERT INTO engine_cursor (engine, cursor, version) VALUES (%s, 0, %s) "
                "ON CONFLICT (engine) DO UPDATE SET cursor = 0, version = EXCLUDED.version",
                (self.engine, version),
            )


log = logging.getLogger(__name__)

HNSW_MAX_DIM = 4000  # halfvec's HNSW limit; larger vectors are searched exactly
HALFVEC_MAX = 65504.0  # largest finite half-precision value; beyond it the halfvec cast fails
_HNSW_TABLES = ("event_vectors", "knowledge_vectors")
_KINDS = {"event": "event_vectors", "knowledge": "knowledge_vectors"}


def _literal(vector: list[float]) -> str:
    """pgvector's text form. repr() of a float32-rounded value parses back to that same
    float32 (pgvector reads components with strtof)."""
    return "[" + ",".join(repr(x) for x in vector) + "]"


def _as_float32(values: list[float]) -> list[float]:
    """emb::real[] as Python floats of exactly the stored float32 values, whatever text
    form the driver parsed them from."""
    return array("f", values).tolist()


def _hnsw_name(table: str, dim: int) -> str:
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
        self._hnsw_for: int | None = None  # the D this backend last built (or checked) indexes for

    def load(self) -> None:
        """On opening the database: with HNSW on, build the indexes of an index version
        that already has a dimension (vectors written while HNSW was off)."""
        if self.db.vector_index != "hnsw":
            return
        with self.db.transaction():
            dim = self._dim()
            if dim is not None:
                self._ensure_hnsw(dim)

    # Index state

    def _get_meta(self, key: str) -> str | None:
        rows = self.db.query("SELECT value FROM vector_meta WHERE key = %s", (key,))
        return rows[0][0] if rows else None

    def _set_meta(self, key: str, value: str) -> None:
        self.db.con.execute(
            "INSERT INTO vector_meta (key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            (key, value),
        )

    def meta(self) -> dict | None:
        value = self._get_meta("meta")
        return json.loads(value) if value is not None else None

    def reset(self, meta: dict) -> None:
        encoded = json.dumps(meta)
        with self.db.transaction():
            con = self.db.con
            for name in self._hnsw_indexes():
                con.execute(f'DROP INDEX IF EXISTS "{name}"')
            for table in ("event_vectors", "knowledge_vectors", "vector_groups"):
                con.execute(f"DELETE FROM {table}")
            con.execute("DELETE FROM vector_meta WHERE key IN ('indexed', 'dim')")
            self._set_meta("meta", encoded)
            self._hnsw_for = None

    def indexed(self) -> int:
        value = self._get_meta("indexed")
        return int(value) if value is not None else 0

    def set_indexed(self, seq: int) -> None:
        with self.db.transaction():
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
        rows = self.db.query(
            "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema() "
            "AND tablename = ANY(%s) AND indexname ~ '_hnsw_d[0-9]+$'",
            (list(_HNSW_TABLES),),
        )
        return [r[0] for r in rows]

    def _ensure_hnsw(self, dim: int) -> None:
        """Inside a write transaction, with HNSW on: build the missing indexes for D, once
        per D per backend. A table holding a component beyond halfvec range (written while
        HNSW was off) gets no index, since building it would fail; it is searched exactly."""
        self._hnsw_for = dim
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
            overflow = self.db.con.execute(
                f"SELECT EXISTS (SELECT 1 FROM {table} v, unnest(v.emb::real[]) x WHERE abs(x) > %s)",
                (HALFVEC_MAX,),
            ).fetchone()[0]
            if overflow:
                log.warning(
                    "%s holds components beyond halfvec range (|x| > %g); no HNSW index, "
                    "searching it exactly until the index is reset",
                    table, HALFVEC_MAX,
                )  # fmt: skip
                continue
            self.db.con.execute(
                f"CREATE INDEX IF NOT EXISTS {name} ON {table} "
                f"USING hnsw ((emb::halfvec({dim})) halfvec_cosine_ops)"
            )

    # Vectors

    def put_vector(self, kind: str, id: str, vector: Sequence[float]) -> None:
        table = _KINDS[kind]
        values = float32(vector)
        self._check_halfvec(values)  # before any write, so the transaction stays usable
        with self.db.transaction():
            dim = self._dim()
            self._check_dim(values, dim)  # likewise
            if self._get_meta("dim") is None:
                self._set_meta("dim", str(len(values)))
            if self.db.vector_index == "hnsw" and self._hnsw_for != len(values):
                self._ensure_hnsw(len(values))
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
                    f"INSERT INTO {table} (id, emb) VALUES (%s, %s::vector) "
                    "ON CONFLICT (id) DO UPDATE SET emb = EXCLUDED.emb",
                    (id, _literal(values)),
                )

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
            self.db.con.execute(f"DELETE FROM {_KINDS[kind]} WHERE {key} = ANY(%s)", (list(ids),))

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
    ) -> list:
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
                return self.db.con.execute(sql, params).fetchall()
            # A read-only transaction of its own, so SET LOCAL has one to live in; it takes
            # no advisory lock, so a search never waits for writers.
            self.db.con.execute("BEGIN READ ONLY")
            try:
                self.db.con.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
                rows = self.db.con.execute(sql, params).fetchall()
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
