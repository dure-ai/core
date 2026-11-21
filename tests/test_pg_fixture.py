"""Packaging guard and the PostgreSQL test fixture."""

import subprocess
import sys

import pytest
from pg import pg_url

_BLOCKED_IMPORT = (
    "import sys; sys.modules['psycopg'] = None; "
    "import cortex_mem, cortex_mem.store, cortex_mem.testing, cortex_mem.store.postgres"
)


def test_core_imports_without_psycopg():
    result = subprocess.run(
        [sys.executable, "-c", _BLOCKED_IMPORT], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_pg_fixture_gives_an_empty_schema(pg_schema):
    import psycopg

    with psycopg.connect(pg_schema, autocommit=True) as conn:
        schema = conn.execute("SELECT current_schema()").fetchone()[0]
        assert schema.startswith("t_")
        tables = conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema = %s",
            (schema,),
        ).fetchone()[0]
        assert tables == 0
        # The vector extension is installed database-wide, so its type resolves here.
        assert conn.execute("SELECT '[1,2]'::vector").fetchone() is not None


def test_pg_url_reads_the_environment(monkeypatch):
    monkeypatch.delenv("CORTEX_TEST_PG_URL", raising=False)
    assert pg_url() is None
    monkeypatch.setenv("CORTEX_TEST_PG_URL", "postgresql://x/y")
    assert pg_url() == "postgresql://x/y"


@pytest.mark.skipif(pg_url() is not None, reason="only meaningful when PG is absent")
def test_fixture_skips_without_url(pg_schema):
    raise AssertionError("fixture should have skipped")
