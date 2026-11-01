"""Real Chroma controls for update/delete races and ambiguous read-back."""
import asyncio

import pytest

from mem0.memory.main import AsyncMemory, Memory
from mem0.memory.storage import SQLiteManager
from mem0.vector_stores.chroma import ChromaDB


@pytest.fixture
def actual_chroma_memory(tmp_path):
    store = ChromaDB(collection_name="update_race_fixture", path=str(tmp_path / "chroma"))
    history = SQLiteManager(str(tmp_path / "history.sqlite"))
    store.insert(vectors=[[0.1, 0.2, 0.3, 0.4]], payloads=[{
        "data": "old fixture", "user_id": "fixture", "created_at": "2026-11-01T00:00:00Z",
    }], ids=["fixture-id"])
    try:
        yield store, history
    finally:
        history.close()
        store.client.close()


def _memory(memory_class, store, history):
    memory = memory_class.__new__(memory_class)
    memory.vector_store = store
    memory.db = history
    if memory_class is AsyncMemory:
        async def noop(*_args, **_kwargs):
            return None
    else:
        def noop(*_args, **_kwargs):
            return None
    memory._remove_memory_from_entity_store = noop
    memory._link_entities_for_memory = noop
    return memory


@pytest.mark.parametrize("memory_class", [Memory, AsyncMemory])
def test_actual_chroma_delete_between_read_and_update_fails_without_history(actual_chroma_memory, memory_class):
    store, history = actual_chroma_memory
    memory = _memory(memory_class, store, history)
    original = store.update
    def delete_then_update(**kwargs):
        store.delete(vector_id="fixture-id")
        return original(**kwargs)
    store.update = delete_then_update
    with pytest.raises(ValueError, match="not found"):
        result = memory._update_memory("fixture-id", "new fixture", {"new fixture": [0.4, 0.3, 0.2, 0.1]})
        if memory_class is AsyncMemory:
            asyncio.run(result)
    assert store.get("fixture-id") is None
    assert not any(row["event"] == "UPDATE" for row in history.get_history("fixture-id"))


def test_actual_chroma_successful_update_records_history(actual_chroma_memory):
    store, history = actual_chroma_memory
    memory = _memory(Memory, store, history)
    assert memory._update_memory("fixture-id", "new fixture", {"new fixture": [0.4, 0.3, 0.2, 0.1]}) == "fixture-id"
    assert store.get("fixture-id").payload["data"] == "new fixture"
    assert [row["event"] for row in history.get_history("fixture-id")] == ["UPDATE"]


def test_actual_chroma_readback_error_is_not_reported_as_success(actual_chroma_memory):
    store, history = actual_chroma_memory
    memory = _memory(Memory, store, history)
    actual = store.collection
    class ReadbackFailure:
        written = False
        def get(self, **kwargs):
            if self.written:
                raise ConnectionError("fixture confirmation failed")
            return actual.get(**kwargs)
        def update(self, **kwargs):
            actual.update(**kwargs)
            self.written = True
    store.collection = ReadbackFailure()
    with pytest.raises(ConnectionError, match="fixture confirmation failed"):
        memory._update_memory("fixture-id", "new fixture", {"new fixture": [0.4, 0.3, 0.2, 0.1]})
    assert actual.get(ids=["fixture-id"])["metadatas"][0]["data"] == "new fixture"
    assert history.get_history("fixture-id") == []


def test_actual_chroma_public_update_racing_delete_reports_missing(actual_chroma_memory):
    store, history = actual_chroma_memory
    memory = _memory(Memory, store, history)
    class Embedder:
        def embed(self, _text, _action):
            return [0.4, 0.3, 0.2, 0.1]
    memory.embedding_model = Embedder()
    original = store.update
    def delete_then_update(**kwargs):
        store.delete(vector_id="fixture-id")
        return original(**kwargs)
    store.update = delete_then_update
    with pytest.raises(ValueError, match="not found"):
        memory.update("fixture-id", text="new fixture")
    assert store.get("fixture-id") is None
    assert not any(row["event"] == "UPDATE" for row in history.get_history("fixture-id"))
