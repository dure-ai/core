"""PostgreSQL test base: the URL switch and a per-test empty schema.

The tests that need PostgreSQL run only when ``CORTEX_TEST_PG_URL`` is set, for example
``postgresql://postgres:cortex@localhost:54329/postgres`` against ``docker compose up -d pg``.
"""

from __future__ import annotations

import os
import uuid
from urllib.parse import quote

import pytest

ENV = "CORTEX_TEST_PG_URL"


def pg_url() -> str | None:
    """The base PostgreSQL URL for tests, or None when none is configured."""
    return os.environ.get(ENV) or None


def url_with_search_path(url: str, schema: str) -> str:
    """Return ``url`` with a libpq ``options=-csearch_path=<schema>`` parameter.

    libpq applies ``options`` to every connection opened with this URL, so each new
    connection to it sees the same schema without a ``SET`` per connection. ``public`` is
    kept second so extension objects installed there (the ``vector`` type) still resolve.
    """
    options = quote(f"-csearch_path={schema},public", safe="")
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}options={options}"


@pytest.fixture
def pg_schema():
    """Yield a URL whose connections land in a fresh, empty schema; drop it afterwards."""
    base = pg_url()
    if base is None:
        pytest.skip(f"{ENV} is not set")
    import psycopg

    schema = f"t_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(base, autocommit=True) as conn:
        # Once per database, not per schema: the extension is database-wide.
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(f'CREATE SCHEMA "{schema}"')
    try:
        yield url_with_search_path(base, schema)
    finally:
        with psycopg.connect(base, autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
