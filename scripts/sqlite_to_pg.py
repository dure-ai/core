"""Move a SQLite store into an empty PostgreSQL database.

Moves events (same seq), knowledge, every engine's state, cursor and version, and the
embedding cache. Vectors, groups and the index version are not moved: the first sync on
PostgreSQL rebuilds the index (its version names the store) and finds every embedding in
the cache. Everything happens in one transaction, so a failure leaves the target empty.

Usage:
    python scripts/sqlite_to_pg.py SQLITE_PATH PG_URL
"""

from __future__ import annotations

import argparse
import sys
from array import array
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TABLES = ("events", "knowledge", "engine_state", "engine_cursor", "embed_cache")


class MigrationError(Exception):
    """A reason to stop; the message is safe to print (it never holds the URL)."""


def _count(db, table: str) -> int:
    """Row count of `table`."""
    return db.query(f"SELECT count(*) FROM {table}")[0][0]


def _check_empty(pg) -> None:
    """MigrationError when any table the migration writes already has rows."""
    full = [t for t in TABLES if _count(pg, t)]
    if full:
        raise MigrationError(f"the target must be empty, but it already has rows in: {', '.join(full)}")


def _copy_events(src, pg) -> None:
    """Append every source event in order. MigrationError when the source seq
    has a gap or an event lands at another seq in the target."""
    for expected, (seq, event) in enumerate(src.events.since(0), 1):
        if seq != expected:
            raise MigrationError(f"the source events have a gap: expected seq {expected}, found {seq}")
        pg.events.append(event)
        landed = pg.query("SELECT max(seq) FROM events")[0][0]
        if landed != seq:
            raise MigrationError(f"event {event.id} is seq {seq} in the source but {landed} in the target")


def _copy_state(src, pg) -> None:
    """Copy engine_state, and each engine's cursor and version, as they are.
    The target's epoch starts at 0."""
    # Written directly: the state backend's commit needs the cursor it expects, and a
    # migration has no engine to ask. This is the one place that bypasses it.
    con = pg.con
    for engine, key, value in src.query("SELECT engine, key, value FROM engine_state ORDER BY engine, key"):
        con.execute(
            "INSERT INTO engine_state (engine, key, value) VALUES (%s, %s, %s)", (engine, key, value)
        )
    for engine, cursor, version in src.query("SELECT engine, cursor, version FROM engine_cursor"):
        con.execute(
            "INSERT INTO engine_cursor (engine, cursor, version) VALUES (%s, %s, %s)",
            (engine, cursor, version),
        )


def _copy_cache(src, pg) -> None:
    """Copy the embedding cache. MigrationError for a row that is not a
    float32 vector or that PostgreSQL refuses."""
    for model, digest, blob in src.query("SELECT model, digest, vec FROM embed_cache ORDER BY model, digest"):
        if len(blob) == 0 or len(blob) % 4:
            raise MigrationError(f"embedding cache row {digest[:12]} is not a float32 vector")
        values = array("f")
        values.frombytes(blob)
        try:
            pg.vectors.cache(model, digest, values.tolist())
        except ValueError as exc:
            raise MigrationError(f"embedding cache row {digest[:12]} cannot be stored: {exc}") from None


def migrate(src, pg) -> list[tuple[str, int, int]]:
    """Copy src (a SqliteDatabase) into pg (an empty PostgresDatabase), all or nothing.

    Returns (table, source rows, target rows) for each table."""
    with pg.transaction():
        _check_empty(pg)
        _copy_events(src, pg)
        for item in src.knowledge.all():
            pg.knowledge.put(item)
        _copy_state(src, pg)
        _copy_cache(src, pg)
        rows = [(t, _count(src, t), _count(pg, t)) for t in TABLES]
        bad = [t for t, a, b in rows if a != b]
        if bad:
            raise MigrationError(f"row counts differ after the move: {', '.join(bad)}")
    return rows


def _snapshot(path: Path, into: Path) -> Path:
    """Copy the source with SQLite's backup API, read-only, and return the copy's path.

    Raises MigrationError when the file is not a SQLite store with an events table."""
    import sqlite3

    try:
        # as_uri() percent-encodes the path, so ?, # and % in it stay part of the file name.
        ro = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise MigrationError(f"cannot read {path}: {exc}") from None
    try:
        try:
            has_events = ro.execute("SELECT 1 FROM sqlite_master WHERE name = 'events'").fetchone()
            version = ro.execute("PRAGMA user_version").fetchone()[0]
            if not has_events:
                raise MigrationError(f"{path} has no events table; it is not a cortex store")
            copy = into / "source.db"
            dest = sqlite3.connect(copy)
            try:
                ro.backup(dest)
            finally:
                dest.close()
        except sqlite3.Error as exc:
            raise MigrationError(f"cannot read {path}: {exc}") from None
    finally:
        ro.close()
    if version == 4:
        print(
            "warning: the source is schema 4, whose embedding cache is not usable; "
            "nothing is cached and the first sync will re-embed",
            file=sys.stderr,
        )
    return copy


def main(argv: Sequence[str] | None = None) -> int:
    """Migrate, print the row counts per table and return 0. Returns 2 with a
    one-line error, never holding the URL, when anything fails."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sqlite_path", type=Path, help="the SQLite store to read (never changed)")
    parser.add_argument("pg_url", help="PostgreSQL URL of an empty target (never printed)")
    args = parser.parse_args(argv)

    if not args.sqlite_path.is_file():
        print(f"error: {args.sqlite_path} is not a readable SQLite file", file=sys.stderr)
        return 2
    try:
        rows = _run(args.sqlite_path, args.pg_url)
    except MigrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001  one line, no traceback, and never the URL
        message = " ".join(str(exc).replace(args.pg_url, "<url>").split())
        print(f"error: {type(exc).__name__}: {message}", file=sys.stderr)
        return 2

    print(f"{'table':<16}{'source':>8}{'target':>8}")
    for table, a, b in rows:
        print(f"{table:<16}{a:>8}{b:>8}")
    print("moved; the first sync on PostgreSQL rebuilds the index from the embedding cache")
    return 0


def _run(path: Path, url: str) -> list[tuple[str, int, int]]:
    """Snapshot the source into a temporary directory, open both stores and
    migrate; a failed connection to PostgreSQL becomes a MigrationError."""
    import tempfile

    from cortex_mem.store.postgres import PostgresDatabase
    from cortex_mem.store.sqlite import SqliteDatabase

    with tempfile.TemporaryDirectory() as tmp:
        src = SqliteDatabase(_snapshot(path, Path(tmp)))
        try:
            try:
                pg = PostgresDatabase(url)
            except ConnectionError as exc:
                raise MigrationError(str(exc)) from None  # worded without the password
            try:
                return migrate(src, pg)
            finally:
                pg.close()
        finally:
            src.con.close()


if __name__ == "__main__":
    sys.exit(main())
