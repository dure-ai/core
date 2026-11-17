import sys
from pathlib import Path

# Reuse the PostgreSQL fixtures of the core tests (tests/pg.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
from pg import pg_schema, pg_schemas  # noqa: F401  (registers the fixtures)
