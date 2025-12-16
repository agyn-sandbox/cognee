import uuid

import pytest

from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter
from cognee.infrastructure.engine import DataPoint


class _StubEmbeddingEngine:
    def __init__(self):
        self._vector_size = 3
        self.last_payload = None

    async def embed_text(self, texts: list[str]):
        self.last_payload = texts
        return [[1.0] * self._vector_size for _ in texts]

    def get_vector_size(self) -> int:
        return self._vector_size

    def get_batch_size(self) -> int:
        return 16


class _StubCollection:
    def __init__(self):
        self.inserted_rows = None

    def merge_insert(self, *args, **kwargs):
        return self

    def when_matched_update_all(self):
        return self

    def when_not_matched_insert_all(self):
        return self

    async def execute(self, rows):
        self.inserted_rows = rows


class _SamplePoint(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


@pytest.mark.asyncio
async def test_create_data_points_handles_empty_text(monkeypatch):
    embedding_engine = _StubEmbeddingEngine()
    adapter = LanceDBAdapter(url=None, api_key=None, embedding_engine=embedding_engine)

    async def _fake_has_collection(collection_name: str) -> bool:
        return True

    async def _fake_get_collection(collection_name: str):
        return _StubCollection()

    monkeypatch.setattr(adapter, "has_collection", _fake_has_collection)
    collection_stub = _StubCollection()

    async def _fake_get_collection(collection_name: str):
        return collection_stub

    monkeypatch.setattr(adapter, "get_collection", _fake_get_collection)

    empty_point = _SamplePoint(id=uuid.uuid4(), text="    ")
    normal_point = _SamplePoint(id=uuid.uuid4(), text="Meaningful text")

    await adapter.create_data_points("sample", [empty_point, normal_point])

    assert embedding_engine.last_payload == ["Meaningful text"]
    assert collection_stub.inserted_rows is not None

    empty_vector = collection_stub.inserted_rows[0].vector
    filled_vector = collection_stub.inserted_rows[1].vector

    assert all(value == 0.0 for value in empty_vector)
    assert filled_vector == [1.0, 1.0, 1.0]
