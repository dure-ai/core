import pytest

from cortex_mem.store.sqlite import SqliteDatabase


@pytest.fixture(autouse=True)
def _close_sqlite_databases(monkeypatch):
    """Close every SqliteDatabase a test opened.

    Python 3.13+ warns about connections left to the garbage collector, and CI runs
    pytest with -W error. Tracking here keeps each test free of close() calls.
    """
    opened: list[SqliteDatabase] = []
    init = SqliteDatabase.__init__

    def tracking_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        opened.append(self)

    monkeypatch.setattr(SqliteDatabase, "__init__", tracking_init)
    yield
    for db in opened:
        db.close()
