"""Measure delivery cost: write transactions per event, and how long a slow engine blocks other writes.

Usage:
    python scripts/bench_delivery.py --store sqlite --label baseline
    CORTEX_BENCH_PG_URL=postgresql://... python scripts/bench_delivery.py --store pg --label baseline

Measurement A appends --events events through a Runtime with an Index and counts the
outermost write transactions. Measurement B holds a slow `observe` for --hold seconds
on one connection and times a raw `events.append` and a `Runtime.append` on a second
connection to the same store. It uses the public API only, so it runs before and after
the delivery redesign.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Sequence
from contextlib import contextmanager, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cortex_mem.contract import Event, Knowledge, Promote
from cortex_mem.embed import HashEmbedder
from cortex_mem.index import Index
from cortex_mem.runtime import Runtime

ENV = "CORTEX_BENCH_PG_URL"
REPS = 5
SLOW_ID = "slow"
START_WAIT_S = 0.3
_AT0 = datetime(2026, 1, 1, tzinfo=UTC)


# ── store setup ──────────────────────────────────────────────────────────────


def url_with_search_path(url: str, schema: str) -> str:
    """``url`` with ``-csearch_path=<schema>,public`` in its libpq ``options`` parameter."""
    parts = urlsplit(url)
    query = parse_qsl(parts.query, keep_blank_values=True)
    setting = f"-csearch_path={schema},public"
    merged = []
    found = False
    for key, value in query:
        if key == "options":
            value = f"{value} {setting}".strip()
            found = True
        merged.append((key, value))
    if not found:
        merged.append(("options", setting))
    return urlunsplit(parts._replace(query=urlencode(merged, quote_via=quote)))


@contextmanager
def fresh_store(store: str, pg_url: str | None):
    """Yield a function that opens a new connection to one empty store; clean up after."""
    if store == "sqlite":
        from cortex_mem.store.sqlite import SqliteDatabase

        tmp = tempfile.mkdtemp(prefix="cortex-bench-")
        opened: list[Any] = []

        def open_db() -> Any:
            db = SqliteDatabase(Path(tmp) / "bench.db")
            opened.append(db)
            return db

        try:
            yield open_db
        finally:
            for db in opened:
                db.con.close()
            shutil.rmtree(tmp, ignore_errors=True)
        return

    import psycopg

    from cortex_mem.store.postgres import PostgresDatabase

    assert pg_url is not None
    with psycopg.connect(pg_url, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    schema = f"bench_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(pg_url, autocommit=True) as conn:
        conn.execute(f'CREATE SCHEMA "{schema}"')
    url = url_with_search_path(pg_url, schema)
    opened = []

    def open_pg() -> Any:
        db = PostgresDatabase(url)
        opened.append(db)
        return db

    try:
        yield open_pg
    finally:
        for db in opened:
            with suppress(Exception):  # cleanup must reach the DROP
                db.con.close()
        with psycopg.connect(pg_url, autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


# ── building blocks ──────────────────────────────────────────────────────────


class CountingEmbedder:
    """HashEmbedder that counts `embed` calls (no network)."""

    def __init__(self) -> None:
        self._inner = HashEmbedder()
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        """model, dim and the rest come from the wrapped HashEmbedder."""
        return getattr(self._inner, name)

    def embed(self, texts: Sequence[str], kind: Any) -> list[list[float]]:
        """Count the call, then embed with the HashEmbedder."""
        self.calls += 1
        return self._inner.embed(texts, kind)


class CountingTransaction:
    """Wraps `db.transaction` and counts only the outermost entries (a depth counter)."""

    def __init__(self, db: Any) -> None:
        """Replace db.transaction with this wrapper. Build the Index and Runtime
        afterwards, so they take the wrapper."""
        self._inner = db.transaction
        self._lock = threading.Lock()
        self._depth = 0
        self.outer = 0
        db.transaction = self  # Index and Runtime are built after this, so they see the wrapper

    @contextmanager
    def __call__(self):
        """db's own transaction, counted in `outer` when it is the outermost."""
        with self._lock:
            if self._depth == 0:
                self.outer += 1
            self._depth += 1
        try:
            with self._inner():
                yield
        finally:
            with self._lock:
                self._depth -= 1


class BenchEngine:
    """Passes events through. `observe` sleeps `hold` seconds on the event with id `slow_id`."""

    def __init__(self, promote: bool, slow_id: str | None = None, hold: float = 0.0) -> None:
        """promote: return a Promote for every event, so delivery writes
        knowledge. slow_id and hold: the event whose observe sleeps, and for
        how many seconds; `started` is set when that sleep begins."""
        self.promote = promote
        self.slow_id = slow_id
        self.hold = hold
        self.started = threading.Event()

    def observe(self, event: Event, ctx: Any) -> list[Any]:
        """Sleep on the slow event, then promote the event's text as personal
        knowledge when `promote` is set."""
        if self.slow_id is not None and event.id == self.slow_id:
            self.started.set()
            time.sleep(self.hold)
        if not self.promote:
            return []
        item = Knowledge(
            id=f"k-{event.id}",
            kind="knowledge",
            title=f"note {event.id}",
            body=event.text,
            scope="personal",
            owner=event.user,
            sources=(event.id,),
        )
        return [Promote(item)]

    def tick(self, now: datetime, ctx: Any) -> list[Any]:
        """No decisions."""
        return []

    def handoff(self, query: str, caller: Any, scope: Any, ctx: Any, k: int = 6) -> list[Any]:
        """No items."""
        return []


def make_event(event_id: str, i: int) -> Event:
    """Personal record number i, i seconds after _AT0; its text shares topic
    and component words with other events."""
    return Event(
        id=event_id,
        kind="record",
        at=_AT0 + timedelta(seconds=i),
        user="u",
        team="t",
        session="s",
        scope="personal",
        text=f"bench event {i} about topic {i % 17} and component {i % 5}",
    )


def make_runtime(db: Any, engine: BenchEngine, name: str, embedder: CountingEmbedder) -> Runtime:
    """Runtime with an Index, built with `transaction=` when this Index accepts it."""
    kwargs: dict[str, Any] = {}
    if "transaction" in inspect.signature(Index.__init__).parameters:
        kwargs["transaction"] = db.transaction
    index = Index(db.vectors, embedder, 0.9, **kwargs)
    return Runtime(engine, db.events, db.knowledge, db.state(name), db.transaction, index=index)


def summary(values: list[float]) -> dict[str, float]:
    """min, median and max of the values (all 0.0 when there are none)."""
    if not values:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


# ── measurements ─────────────────────────────────────────────────────────────


def measure_write_transactions(store: str, pg_url: str | None, events: int) -> dict[str, Any]:
    """Measurement A: append `events` records through a Runtime with an Index
    on a fresh store. Returns the outermost write transactions (in total
    and per event) and the number of embed calls."""
    with fresh_store(store, pg_url) as open_db:
        db = open_db()
        counter = CountingTransaction(db)
        embedder = CountingEmbedder()
        runtime = make_runtime(db, BenchEngine(promote=True), "bench", embedder)
        for i in range(events):
            runtime.append(make_event(f"e{i}", i))
        total = counter.outer
        return {
            "write_transactions": {"total": total, "per_event": total / events if events else 0.0},
            "embedder_calls": embedder.calls,
        }


def measure_append_while_observe_holds(store: str, pg_url: str | None, hold: float, reps: int) -> dict[str, Any]:
    """Measurement B, `reps` times on a fresh store each: while one
    connection's engine sleeps `hold` seconds in observe, time a raw
    events.append and a Runtime.append on a second connection. Returns
    min, median and max of each; an append that fails, a slow engine that
    never starts, or a failure on the slow side counts in `errors`."""
    raw: list[float] = []
    via_runtime: list[float] = []
    errors = 0
    for rep in range(reps):
        with fresh_store(store, pg_url) as open_db:
            db1, db2 = open_db(), open_db()
            slow = BenchEngine(promote=False, slow_id=SLOW_ID, hold=hold)
            runtime1 = make_runtime(db1, slow, "slow", CountingEmbedder())
            runtime2 = make_runtime(db2, BenchEngine(promote=False), "fast", CountingEmbedder())
            failures: list[BaseException] = []

            def run_slow(runtime: Runtime, failed: list[BaseException]) -> None:
                try:
                    runtime.append(make_event(SLOW_ID, 0))
                except BaseException as exc:  # noqa: BLE001 - counted in `errors`
                    failed.append(exc)

            thread = threading.Thread(target=run_slow, args=(runtime1, failures))
            thread.start()
            if not slow.started.wait(timeout=hold + 10):
                errors += 1  # the engine never started; nothing was held
                thread.join()
                continue
            time.sleep(START_WAIT_S)
            try:
                t0 = time.perf_counter()
                db2.events.append(make_event(f"raw{rep}", 1))
                raw.append(time.perf_counter() - t0)
            except Exception:  # noqa: BLE001 - e.g. a lock error: counted, not a delay
                errors += 1
            try:
                t0 = time.perf_counter()
                runtime2.append(make_event(f"rt{rep}", 2))
                via_runtime.append(time.perf_counter() - t0)
            except Exception:  # noqa: BLE001
                errors += 1
            thread.join()
            errors += len(failures)
    return {
        "hold_s": hold,
        "raw_append_s": summary(raw),
        "runtime_append_s": summary(via_runtime),
        "errors": errors,
    }


def git_commit() -> str:
    """Short hash of the checked-out commit, or "unknown"."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def run_bench(
    store: str, label: str, events: int = 200, hold: float = 3.0, reps: int = REPS, pg_url: str | None = None
) -> dict[str, Any]:
    """Both measurements on one store, with the label and commit: the JSON
    result."""
    a = measure_write_transactions(store, pg_url, events)
    b = measure_append_while_observe_holds(store, pg_url, hold, reps)
    return {
        "label": label,
        "store": store,
        "commit": git_commit(),
        "events": events,
        "write_transactions": a["write_transactions"],
        "embedder_calls": a["embedder_calls"],
        "append_while_observe_holds": b,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark, print the JSON result, and write it to --out when
    given. Returns the exit code."""
    parser = argparse.ArgumentParser(description="Measure write transactions per event and append latency.")
    parser.add_argument("--store", required=True, choices=["sqlite", "pg"])
    parser.add_argument("--label", required=True, help="names the run in the result")
    parser.add_argument("--events", type=int, default=200)
    parser.add_argument("--hold", type=float, default=3.0, help="seconds the slow engine holds (default 3.0)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.events < 1:
        parser.error("--events must be at least 1")
    if args.hold <= 0:
        parser.error("--hold must be positive")
    pg_url = os.environ.get(ENV) or None
    if args.store == "pg" and pg_url is None:
        parser.error(f"--store pg needs {ENV} (a PostgreSQL URL)")
    result = run_bench(args.store, args.label, args.events, args.hold, pg_url=pg_url)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
