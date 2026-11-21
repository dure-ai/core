"""Shared fixtures: registers the PostgreSQL schema fixtures from tests/pg.py."""
from pg import pg_schema, pg_schemas  # noqa: F401  (registers the fixtures)
