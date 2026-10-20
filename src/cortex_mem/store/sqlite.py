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
from collections.abc import Collection, Iterator, Mapping, Set
from contextlib import contextmanager
from datetime import datetime
from os import PathLike
from typing import Any

from ..contract import Caller, Event, Kind, Knowledge, Scope
from .base import CursorConflict

SCHEMA_VERSION = 2

_WORD = re.compile(r"\w+")
_HANGUL = re.compile(r"[가-힣]")
_EVENT_COLUMNS = (
    "id, kind, at, user, team, session, scope, text, agent, signal, outcome, "
    "conditions, refs, repo, rev"
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
    id, kind, at, user, team, session, scope, text, agent, signal, outcome, conditions, refs, repo, rev = row
    return Event(
        id=id, kind=kind, at=datetime.fromisoformat(at), user=user, team=team, session=session,
        scope=scope, text=text, agent=agent, signal=signal, outcome=outcome,
        conditions=json.loads(conditions), refs=tuple(json.loads(refs)), repo=repo, rev=rev,
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
        self.con.execute("PRAGMA busy_timeout = 5000")
        self._create()
        self.events = SqliteEventStore(self)
        self.knowledge = SqliteKnowledgeBase(self)

    def _create(self) -> None:
        version = self.con.execute("PRAGMA user_version").fetchone()[0]
        has_tables = self.con.execute("SELECT count(*) FROM sqlite_master WHERE name = 'events'").fetchone()[0]
        if has_tables and version != SCHEMA_VERSION:
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
                    f"INSERT INTO events ({_EVENT_COLUMNS}) VALUES ({', '.join('?' * 15)})",
                    (
                        e.id, e.kind, e.at.isoformat(), e.user, e.team, e.session, e.scope, e.text,
                        e.agent, e.signal, e.outcome, json.dumps(dict(e.conditions)),
                        json.dumps(list(e.refs)), e.repo, e.rev,
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
