"""SQLite stores: event log, knowledge base and engine state in one file.

Korean attaches particles to words ("배포가"), so whole-word matching misses
them. Hangul tokens are also indexed as character bigrams (the pg_bigm idea),
which lets "배포" match "배포가". Score is -bm25 (higher = better); ties keep
append order for events and id order for knowledge.

Several server processes (one per stdio session) may share the file.
transaction() takes SQLite's write lock (BEGIN IMMEDIATE), so the runtime
delivers each event exactly once across processes.

Standard library only, so the core package stays dependency-free. Temporary
until the PostgreSQL store, which passes the same suites (cortex_mem.testing).
"""

import json
import re
import sqlite3
import threading
from array import array
from collections.abc import Collection, Iterator, Mapping, Sequence, Set
from contextlib import contextmanager
from datetime import datetime
from os import PathLike
from typing import Any

from ..contract import Caller, Event, Kind, Knowledge, Scope
from .base import CursorConflict

SCHEMA_VERSION = 4
BUSY_TIMEOUT_MS = 5000  # how long a connection waits for another's write lock

_WORD = re.compile(r"\w+")
_HANGUL = re.compile(r"[가-힣]")
_EVENT_COLUMNS = (
    "id, kind, at, user, team, session, scope, text, agent, signal, outcome, "
    "conditions, refs, repo, rev, text_redacted"
)
_KNOWLEDGE_COLUMNS = "id, kind, title, body, scope, owner, sources, conditions"
_EVENT_VISIBLE = (
    "(e.user = :user OR (:scope IN ('team', 'org') AND e.scope = 'team' AND e.team = :team)"
    " OR (:scope = 'org' AND e.scope = 'org'))"
)
_KNOWLEDGE_VISIBLE = (
    "((k.scope = 'personal' AND k.owner = :user)"
    " OR (k.scope = 'team' AND :scope IN ('team', 'org') AND k.owner = :team)"
    " OR (k.scope = 'org' AND :scope = 'org'))"
)


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _bigrams(text: str) -> list[str]:
    grams = []
    for word in _words(text):
        if _HANGUL.search(word):
            grams += [word[i : i + 2] for i in range(len(word) - 1)]
    return grams


def _match(query: str) -> str | None:
    terms = [f'words:"{w}"' for w in _words(query)] + [f'grams:"{g}"' for g in _bigrams(query)]
    return " OR ".join(terms) or None


def _row_to_event(row: tuple) -> Event:
    id, kind, at, user, team, session, scope, text, agent, signal, outcome, conditions, refs, repo, rev, redacted = row
    return Event(
        id=id, kind=kind, at=datetime.fromisoformat(at), user=user, team=team, session=session,
        scope=scope, text=text, agent=agent, signal=signal, outcome=outcome,
        conditions=json.loads(conditions), refs=tuple(json.loads(refs)), repo=repo, rev=rev,
        text_redacted=bool(redacted),
    )


def _row_to_knowledge(row: tuple) -> Knowledge:
    id, kind, title, body, scope, owner, sources, conditions = row
    return Knowledge(
        id=id, kind=kind, title=title, body=body, scope=scope, owner=owner,
        sources=tuple(json.loads(sources)), conditions=json.loads(conditions),
    )


class SqliteDatabase:
    def __init__(self, path: str | PathLike[str] = ":memory:") -> None:
        # MCP servers may call tools from worker threads; one lock serializes them.
        self.con = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._lock = threading.RLock()
        self._depth = 0
        self.con.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        self._create()
        self.events = SqliteEventStore(self)
        self.knowledge = SqliteKnowledgeBase(self)
        self.vectors = SqliteVectorBackend(self)

    def _create(self) -> None:
        version = self.con.execute("PRAGMA user_version").fetchone()[0]
        has_tables = self.con.execute("SELECT count(*) FROM sqlite_master WHERE name = 'events'").fetchone()[0]
        if has_tables and version == 2:
            with self.transaction():  # 2 -> 3: events.text_redacted
                self.con.execute("ALTER TABLE events ADD COLUMN text_redacted INTEGER NOT NULL DEFAULT 0")
                self.con.execute("PRAGMA user_version = 3")
            version = 3
        # 3 -> 4 only adds the vector tables, which the CREATE IF NOT EXISTS below makes.
        if has_tables and version not in (3, SCHEMA_VERSION):
            raise RuntimeError(f"database schema {version} is not {SCHEMA_VERSION}; start from a new file")
        with self.transaction():
            self.con.execute(
                f"CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, {_EVENT_COLUMNS}, UNIQUE (id))"
            )
            self.con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(words, grams)")
            self.con.execute(
                "CREATE TABLE IF NOT EXISTS knowledge (id TEXT PRIMARY KEY, kind, title, body, scope, owner, "
                "sources, conditions)"
            )
            self.con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(words, grams)")
            self.con.execute(
                "CREATE TABLE IF NOT EXISTS engine_state (engine TEXT, key TEXT, value TEXT, "
                "PRIMARY KEY (engine, key))"
            )
            self.con.execute(
                "CREATE TABLE IF NOT EXISTS engine_cursor (engine TEXT PRIMARY KEY, cursor INTEGER, version TEXT)"
            )
            self.con.execute(
                "CREATE TABLE IF NOT EXISTS vectors (kind TEXT, id TEXT, seq INTEGER, vec BLOB, "
                "PRIMARY KEY (kind, id))"
            )
            self.con.execute(
                "CREATE TABLE IF NOT EXISTS vector_groups (event_id TEXT PRIMARY KEY, group_id TEXT, "
                "scope TEXT, owner TEXT, seq INTEGER)"
            )
            self.con.execute(
                "CREATE INDEX IF NOT EXISTS vector_groups_partition ON vector_groups (scope, owner, seq)"
            )
            self.con.execute("CREATE INDEX IF NOT EXISTS vector_groups_group ON vector_groups (group_id, seq)")
            self.con.execute(
                "CREATE TABLE IF NOT EXISTS embed_cache (model TEXT, digest TEXT, vec BLOB, "
                "PRIMARY KEY (model, digest))"
            )
            self.con.execute("CREATE TABLE IF NOT EXISTS vector_meta (key TEXT PRIMARY KEY, value TEXT)")
            self.con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def state(self, engine: str) -> "SqliteStateBackend":
        return SqliteStateBackend(self, engine)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """One write transaction; nested calls join the outer one."""
        with self._lock:
            outer = self._depth == 0
            if outer:
                self.con.execute("BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield
            except BaseException:
                self._depth -= 1
                if outer:
                    self.con.execute("ROLLBACK")
                raise
            self._depth -= 1
            if outer:
                self.con.execute("COMMIT")

    def query(self, sql: str, params: Any = ()) -> list[tuple]:
        with self._lock:
            return self.con.execute(sql, params).fetchall()


class SqliteEventStore:
    def __init__(self, db: SqliteDatabase) -> None:
        self.db = db

    def append(self, event: Event) -> str:
        e = event
        try:
            with self.db.transaction():
                cur = self.db.con.execute(
                    f"INSERT INTO events ({_EVENT_COLUMNS}) VALUES ({', '.join('?' * 16)})",
                    (
                        e.id, e.kind, e.at.isoformat(), e.user, e.team, e.session, e.scope, e.text,
                        e.agent, e.signal, e.outcome, json.dumps(dict(e.conditions)),
                        json.dumps(list(e.refs)), e.repo, e.rev, int(e.text_redacted),
                    ),
                )
                self.db.con.execute(
                    "INSERT INTO events_fts (rowid, words, grams) VALUES (?, ?, ?)",
                    (cur.lastrowid, " ".join(_words(e.text)), " ".join(_bigrams(e.text))),
                )
        except sqlite3.IntegrityError:
            raise ValueError(f"duplicate event id: {e.id}") from None
        return e.id

    def since(self, seq: int, page: int = 256) -> Iterator[tuple[int, Event]]:
        # Paged: the runtime usually takes only the next event.
        while True:
            rows = self.db.query(
                f"SELECT seq, {_EVENT_COLUMNS} FROM events WHERE seq > ? ORDER BY seq LIMIT ?", (seq, page)
            )
            for row in rows:
                seq = row[0]
                yield seq, _row_to_event(row[1:])
            if len(rows) < page:
                return

    def get(self, ids: Collection[str]) -> list[Event]:
        ids = list(ids)
        rows = self.db.query(f"SELECT {_EVENT_COLUMNS} FROM events WHERE id IN ({', '.join('?' * len(ids))})", ids)
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
        match = _match(query)
        if match is None:
            return []
        columns = ", ".join("e." + c for c in _EVENT_COLUMNS.split(", "))
        sql = (
            f"SELECT {columns}, -bm25(events_fts) AS score "
            "FROM events_fts JOIN events e ON e.seq = events_fts.rowid "
            f"WHERE events_fts MATCH :match AND {_EVENT_VISIBLE}"
        )
        params: dict[str, Any] = {"match": match, "user": caller.user, "team": caller.team, "scope": scope}
        if kinds is not None:
            names = [f"kind{i}" for i in range(len(kinds))]
            sql += f" AND e.kind IN ({', '.join(':' + n for n in names)})"
            params.update(zip(names, kinds))
        sql += " ORDER BY score DESC, e.seq LIMIT :k"
        params["k"] = k
        return [(_row_to_event(row[:-1]), row[-1]) for row in self.db.query(sql, params)]


class SqliteKnowledgeBase:
    def __init__(self, db: SqliteDatabase) -> None:
        self.db = db

    def put(self, item: Knowledge) -> None:
        values = (
            item.kind, item.title, item.body, item.scope, item.owner,
            json.dumps(list(item.sources)), json.dumps(dict(item.conditions)),
        )
        text = f"{item.title} {item.body}"
        with self.db.transaction():
            row = self.db.con.execute("SELECT rowid FROM knowledge WHERE id = ?", (item.id,)).fetchone()
            if row is None:
                cur = self.db.con.execute(
                    f"INSERT INTO knowledge ({_KNOWLEDGE_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (item.id, *values)
                )
                rowid = cur.lastrowid
            else:
                rowid = row[0]
                self.db.con.execute(
                    "UPDATE knowledge SET kind = ?, title = ?, body = ?, scope = ?, owner = ?, sources = ?, "
                    "conditions = ? WHERE rowid = ?",
                    (*values, rowid),
                )
                self.db.con.execute("DELETE FROM knowledge_fts WHERE rowid = ?", (rowid,))
            self.db.con.execute(
                "INSERT INTO knowledge_fts (rowid, words, grams) VALUES (?, ?, ?)",
                (rowid, " ".join(_words(text)), " ".join(_bigrams(text))),
            )

    def remove(self, id: str) -> None:
        with self.db.transaction():
            row = self.db.con.execute("SELECT rowid FROM knowledge WHERE id = ?", (id,)).fetchone()
            if row is not None:
                self.db.con.execute("DELETE FROM knowledge_fts WHERE rowid = ?", (row[0],))
                self.db.con.execute("DELETE FROM knowledge WHERE rowid = ?", (row[0],))

    def get(self, ids: Collection[str]) -> list[Knowledge]:
        ids = list(ids)
        rows = self.db.query(
            f"SELECT {_KNOWLEDGE_COLUMNS} FROM knowledge WHERE id IN ({', '.join('?' * len(ids))})", ids
        )
        found = {row[0]: _row_to_knowledge(row) for row in rows}
        return [found[i] for i in ids if i in found]

    def all(self) -> list[Knowledge]:
        rows = self.db.query(f"SELECT {_KNOWLEDGE_COLUMNS} FROM knowledge ORDER BY id")
        return [_row_to_knowledge(row) for row in rows]

    def search(
        self, query: str, caller: Caller, scope: Scope, k: int = 20
    ) -> list[tuple[Knowledge, float]]:
        match = _match(query)
        if match is None:
            return []
        columns = ", ".join("k." + c for c in _KNOWLEDGE_COLUMNS.split(", "))
        sql = (
            f"SELECT {columns}, -bm25(knowledge_fts) AS score "
            "FROM knowledge_fts JOIN knowledge k ON k.rowid = knowledge_fts.rowid "
            f"WHERE knowledge_fts MATCH :match AND {_KNOWLEDGE_VISIBLE} "
            "ORDER BY score DESC, k.id LIMIT :k"
        )
        params = {"match": match, "user": caller.user, "team": caller.team, "scope": scope, "k": k}
        return [(_row_to_knowledge(row[:-1]), row[-1]) for row in self.db.query(sql, params)]


class SqliteStateBackend:
    def __init__(self, db: SqliteDatabase, engine: str) -> None:
        self.db = db
        self.engine = engine

    def get(self, key: str) -> Any | None:
        rows = self.db.query("SELECT value FROM engine_state WHERE engine = ? AND key = ?", (self.engine, key))
        return json.loads(rows[0][0]) if rows else None

    def scan(self, prefix: str = "") -> Iterator[tuple[str, Any]]:
        rows = self.db.query(
            "SELECT key, value FROM engine_state WHERE engine = ? AND substr(key, 1, ?) = ? ORDER BY key",
            (self.engine, len(prefix), prefix),
        )
        for key, value in rows:
            yield key, json.loads(value)

    def cursor(self) -> int:
        rows = self.db.query("SELECT cursor FROM engine_cursor WHERE engine = ?", (self.engine,))
        return rows[0][0] if rows else 0

    def commit(
        self, expected: int, cursor: int, writes: Mapping[str, Any], deletes: Set[str]
    ) -> None:
        encoded = {key: json.dumps(value) for key, value in writes.items()}
        with self.db.transaction():
            current = self.cursor()
            if current != expected:
                raise CursorConflict(f"cursor is {current}, expected {expected}")
            self.db.con.executemany(
                "DELETE FROM engine_state WHERE engine = ? AND key = ?", [(self.engine, k) for k in deletes]
            )
            self.db.con.executemany(
                "INSERT INTO engine_state (engine, key, value) VALUES (?, ?, ?) "
                "ON CONFLICT (engine, key) DO UPDATE SET value = excluded.value",
                [(self.engine, k, v) for k, v in encoded.items()],
            )
            self.db.con.execute(
                "INSERT INTO engine_cursor (engine, cursor) VALUES (?, ?) "
                "ON CONFLICT (engine) DO UPDATE SET cursor = excluded.cursor",
                (self.engine, cursor),
            )

    def version(self) -> str | None:
        rows = self.db.query("SELECT version FROM engine_cursor WHERE engine = ?", (self.engine,))
        return rows[0][0] if rows else None

    def reset(self, version: str) -> None:
        with self.db.transaction():
            self.db.con.execute("DELETE FROM engine_state WHERE engine = ?", (self.engine,))
            self.db.con.execute(
                "INSERT INTO engine_cursor (engine, cursor, version) VALUES (?, 0, ?) "
                "ON CONFLICT (engine) DO UPDATE SET cursor = 0, version = excluded.version",
                (self.engine, version),
            )


def _pack(vector: Sequence[float]) -> bytes:
    return array("d", vector).tobytes()


def _unpack(blob: bytes) -> list[float]:
    values = array("d")
    values.frombytes(blob)
    return values.tolist()


class SqliteVectorBackend:
    def __init__(self, db: SqliteDatabase) -> None:
        self.db = db

    def meta(self) -> dict | None:
        rows = self.db.query("SELECT value FROM vector_meta WHERE key = 'meta'")
        return json.loads(rows[0][0]) if rows else None

    def reset(self, meta: dict) -> None:
        with self.db.transaction():
            self.db.con.execute("DELETE FROM vectors")
            self.db.con.execute("DELETE FROM vector_groups")
            self.db.con.execute("DELETE FROM vector_meta WHERE key = 'indexed'")
            self.db.con.execute(
                "INSERT INTO vector_meta (key, value) VALUES ('meta', ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (json.dumps(meta),),
            )

    def indexed(self) -> int:
        rows = self.db.query("SELECT value FROM vector_meta WHERE key = 'indexed'")
        return int(rows[0][0]) if rows else 0

    def set_indexed(self, seq: int) -> None:
        with self.db.transaction():
            self.db.con.execute(
                "INSERT INTO vector_meta (key, value) VALUES ('indexed', ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (str(seq),),
            )

    def put_vector(self, kind: str, id: str, vector: Sequence[float]) -> None:
        with self.db.transaction():
            # seq is assigned on first insert; replacing keeps it (and the position).
            self.db.con.execute(
                "INSERT INTO vectors (kind, id, seq, vec) VALUES "
                "(?, ?, (SELECT COALESCE(MAX(seq), 0) + 1 FROM vectors WHERE kind = ?), ?) "
                "ON CONFLICT (kind, id) DO UPDATE SET vec = excluded.vec",
                (kind, id, kind, _pack(vector)),
            )

    def vector(self, kind: str, id: str) -> list[float] | None:
        rows = self.db.query("SELECT vec FROM vectors WHERE kind = ? AND id = ?", (kind, id))
        return _unpack(rows[0][0]) if rows else None

    def vectors(self, kind: str, upto: int | None = None) -> Iterator[tuple[str, list[float]]]:
        if upto is None:
            order = "id" if kind == "knowledge" else "seq"
            rows = self.db.query(f"SELECT id, vec FROM vectors WHERE kind = ? ORDER BY {order}", (kind,))
        elif kind != "event":
            raise ValueError("upto bounds event vectors only")
        else:
            rows = self.db.query(
                "SELECT v.id, v.vec FROM vectors v JOIN vector_groups g ON g.event_id = v.id "
                "WHERE v.kind = 'event' AND g.seq <= ? ORDER BY v.seq",
                (upto,),
            )
        for id, blob in rows:
            yield id, _unpack(blob)

    def delete_vectors(self, kind: str, ids: Collection[str]) -> None:
        with self.db.transaction():
            self.db.con.executemany("DELETE FROM vectors WHERE kind = ? AND id = ?", [(kind, i) for i in ids])

    def put_group(self, event_id: str, group_id: str, scope: Scope, owner: str, seq: int) -> None:
        with self.db.transaction():
            self.db.con.execute(
                "INSERT INTO vector_groups (event_id, group_id, scope, owner, seq) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (event_id) DO UPDATE SET group_id = excluded.group_id, scope = excluded.scope, "
                "owner = excluded.owner, seq = excluded.seq",
                (event_id, group_id, scope, owner, seq),
            )

    def group_of(self, event_id: str) -> str | None:
        rows = self.db.query("SELECT group_id FROM vector_groups WHERE event_id = ?", (event_id,))
        return rows[0][0] if rows else None

    def members(self, group_id: str, upto: int | None = None) -> list[str]:
        if upto is None:
            rows = self.db.query(
                "SELECT event_id FROM vector_groups WHERE group_id = ? ORDER BY seq, event_id", (group_id,)
            )
        else:
            rows = self.db.query(
                "SELECT event_id FROM vector_groups WHERE group_id = ? AND seq <= ? ORDER BY seq, event_id",
                (group_id, upto),
            )
        return [r[0] for r in rows]

    def partition(self, scope: Scope, owner: str) -> list[str]:
        rows = self.db.query(
            "SELECT event_id FROM vector_groups WHERE scope = ? AND owner = ? ORDER BY seq, event_id",
            (scope, owner),
        )
        return [r[0] for r in rows]

    def cached(self, model: str, digest: str) -> list[float] | None:
        rows = self.db.query("SELECT vec FROM embed_cache WHERE model = ? AND digest = ?", (model, digest))
        return _unpack(rows[0][0]) if rows else None

    def cache(self, model: str, digest: str, vector: Sequence[float]) -> None:
        with self.db.transaction():
            self.db.con.execute(
                "INSERT INTO embed_cache (model, digest, vec) VALUES (?, ?, ?) "
                "ON CONFLICT (model, digest) DO UPDATE SET vec = excluded.vec",
                (model, digest, _pack(vector)),
            )
