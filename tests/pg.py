"""PostgreSQL test base: the URL switch and a per-test empty schema.

The tests that need PostgreSQL run only when ``CORTEX_TEST_PG_URL`` is set, for example
``postgresql://postgres:cortex@localhost:54329/postgres`` against ``docker compose up -d pg``.
"""

from __future__ import annotations

import os
import uuid
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import pytest

ENV = "CORTEX_TEST_PG_URL"
_EXTENSION_LOCK = 724_001_001  # arbitrary advisory-lock key for test setup


def pg_url() -> str | None:
    """The base PostgreSQL URL for tests, or None when none is configured."""
    return os.environ.get(ENV) or None


def url_with_search_path(url: str, schema: str) -> str:
    """Return ``url`` with ``-csearch_path=<schema>`` in its libpq ``options`` parameter.

    libpq applies ``options`` to every connection opened with this URL, so each new
    connection to it sees the same schema without a ``SET`` per connection. ``public`` is
    kept second so extension objects installed there (the ``vector`` type) still resolve.
    An existing ``options`` parameter is kept and the setting is appended to it.
    """
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


_extension_ready: set[str] = set()


def _ensure_vector_extension(base: str) -> None:
    """Install the ``vector`` extension once per test session and database.

    The advisory lock serializes concurrent sessions (parallel runs), since
    ``CREATE EXTENSION IF NOT EXISTS`` can still race on the catalog.
    """
    if base in _extension_ready:
        return
    import psycopg

    with psycopg.connect(base, autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (_EXTENSION_LOCK,))
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_EXTENSION_LOCK,))
    _extension_ready.add(base)


@pytest.fixture
def pg_schemas():
    """Yield a function that makes a fresh, empty schema and returns its URL.

    Every schema it made is dropped at teardown, so a test can ask for as many as it needs.
    """
    base = pg_url()
    if base is None:
        pytest.skip(f"{ENV} is not set")
    import psycopg

    _ensure_vector_extension(base)
    made: list[str] = []

    def new_schema() -> str:
        schema = f"t_{uuid.uuid4().hex[:8]}"
        with psycopg.connect(base, autocommit=True) as conn:
            conn.execute(f'CREATE SCHEMA "{schema}"')
        made.append(schema)
        return url_with_search_path(base, schema)

    try:
        yield new_schema
    finally:
        with psycopg.connect(base, autocommit=True) as conn:
            for schema in made:
                conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


@pytest.fixture
def pg_schema(pg_schemas):
    """Yield a URL whose connections land in a fresh, empty schema; drop it afterwards."""
    return pg_schemas()
