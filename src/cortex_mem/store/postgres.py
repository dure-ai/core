"""PostgreSQL + pgvector store (placeholder until the store lands).

This is the only module that may import ``psycopg``, and only inside functions, so core
imports without the driver. Install it with the ``postgres`` extra.
"""


class PostgresDatabase:
    def __init__(self, url: str) -> None:
        import psycopg  # noqa: F401  (fails early with a clear error if the extra is missing)

        raise NotImplementedError("PostgreSQL store is not implemented yet")
