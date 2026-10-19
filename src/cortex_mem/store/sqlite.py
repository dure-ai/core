"""SQLite EpisodeStore: append-only events table plus an FTS5 index.

Korean attaches particles to words ("배포가"), so whole-word matching misses
them. Hangul tokens are also indexed as character bigrams (the pg_bigm idea),
which lets "배포" match "배포가". Score is -bm25 (higher = better); ties keep
append order.
"""

import json
import re
import sqlite3
from collections.abc import Collection, Iterator
from datetime import datetime
from os import PathLike

from ..contract import Event, Kind, QueryContext

_WORD = re.compile(r"\w+")
_HANGUL = re.compile(r"[가-힣]")
_COLUMNS = (
    "id, kind, at, user, session, scope, text, agent, signal, outcome, "
    "conditions, refs, repo, rev"
)


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _bigrams(text: str) -> list[str]:
    grams = []
    for word in _words(text):
        if _HANGUL.search(word):
            grams += [word[i : i + 2] for i in range(len(word) - 1)]
    return grams


def _row_to_event(row: tuple) -> Event:
    id, kind, at, user, session, scope, text, agent, signal, outcome, conditions, refs, repo, rev = row
    return Event(
        id=id, kind=kind, at=datetime.fromisoformat(at), user=user, session=session,
        scope=scope, text=text, agent=agent, signal=signal, outcome=outcome,
        conditions=json.loads(conditions), refs=tuple(json.loads(refs)), repo=repo, rev=rev,
    )


class SqliteEpisodeStore:
    def __init__(self, path: str | PathLike[str] = ":memory:") -> None:
        # MCP servers may call tools from worker threads.
        self.con = sqlite3.connect(path, check_same_thread=False)
        with self.con:
            self.con.execute(
                f"CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, {_COLUMNS}, "
                "UNIQUE (id))"
            )
            self.con.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(words, grams)"
            )

    def append(self, event: Event) -> str:
        e = event
        try:
            with self.con:
                cur = self.con.execute(
                    f"INSERT INTO events ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        e.id, e.kind, e.at.isoformat(), e.user, e.session, e.scope, e.text,
                        e.agent, e.signal, e.outcome, json.dumps(dict(e.conditions)),
                        json.dumps(list(e.refs)), e.repo, e.rev,
                    ),
                )
                self.con.execute(
                    "INSERT INTO events_fts (rowid, words, grams) VALUES (?, ?, ?)",
                    (cur.lastrowid, " ".join(_words(e.text)), " ".join(_bigrams(e.text))),
                )
        except sqlite3.IntegrityError:
            raise ValueError(f"duplicate event id: {e.id}") from None
        return e.id

    def get(self, ids: Collection[str]) -> list[Event]:
        ids = list(ids)
        marks = ", ".join("?" * len(ids))
        rows = self.con.execute(f"SELECT {_COLUMNS} FROM events WHERE id IN ({marks})", ids)
        found = {row[0]: _row_to_event(row) for row in rows}
        return [found[i] for i in ids if i in found]

    def events(self) -> Iterator[Event]:
        """Every event in append order, for rebuilding engine state on startup."""
        for row in self.con.execute(f"SELECT {_COLUMNS} FROM events ORDER BY seq"):
            yield _row_to_event(row)

    def search(
        self,
        query: str,
        ctx: QueryContext,
        k: int = 20,
        kinds: Collection[Kind] | None = None,
    ) -> list[tuple[Event, float]]:
        terms = [f'words:"{w}"' for w in _words(query)] + [f'grams:"{g}"' for g in _bigrams(query)]
        if not terms:
            return []
        sql = (
            f"SELECT {', '.join('e.' + c for c in _COLUMNS.split(', '))}, -bm25(events_fts) AS score "
            "FROM events_fts JOIN events e ON e.seq = events_fts.rowid "
            "WHERE events_fts MATCH ? AND (e.user = ? OR (? = 'team' AND e.scope = 'team'))"
        )
        params: list = [" OR ".join(terms), ctx.user, ctx.scope]
        if kinds is not None:
            kinds = list(kinds)
            sql += f" AND e.kind IN ({', '.join('?' * len(kinds))})"
            params += kinds
        sql += " ORDER BY score DESC, e.seq LIMIT ?"
        params.append(k)
        return [(_row_to_event(row[:-1]), row[-1]) for row in self.con.execute(sql, params)]
