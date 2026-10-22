"""The in-memory stores against the shared conformance suites."""

from cortex_mem.store.memory import (
    InMemoryDatabase,
    InMemoryEventStore,
    InMemoryKnowledgeBase,
    InMemoryStateBackend,
)
from cortex_mem.testing import (
    DatabaseContract,
    EngineStateBackendContract,
    EventStoreContract,
    KnowledgeStoreContract,
)


class TestInMemoryEventStore(EventStoreContract):
    def make(self):
        return InMemoryEventStore()


class TestInMemoryKnowledgeBase(KnowledgeStoreContract):
    def make(self):
        return InMemoryKnowledgeBase()


class TestInMemoryStateBackend(EngineStateBackendContract):
    def make(self):
        return InMemoryStateBackend()


class TestInMemoryDatabase(DatabaseContract):
    def make(self, tmp_path):
        db = InMemoryDatabase()
        return lambda: db
