"""向量库全量重建回归：空集合时维度自检不得因 numpy 空数组的布尔歧义而崩。

背景（端到端实测）：`aclear()` 清空集合后全量入库，chromadb 的 `get(include=["embeddings"])`
返回 numpy 空数组；旧写法 `(probe or {}).get("embeddings") or []` 对它求布尔值，抛
`ValueError: The truth value of an empty array is ambiguous`，于是每个文件都入库失败，
全量重建静默失效，索引长期停留在旧嵌入模型生成的向量上（查询与库内向量近乎正交，
检索置信度恒为 low）。
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from langchain_core.documents import Document

from infrastructure.vector_store.async_chroma_vector import ChromaVector


class _FakeEmbeddings:
    def __init__(self, dim: int):
        self.dim = dim

    def embed_documents(self, texts):
        return [[0.1] * self.dim for _ in texts]


class _FakeCollection:
    def __init__(self, stored):
        self.stored = stored
        self.upserted = None

    def get(self, limit=1, include=None):
        return {"embeddings": self.stored}

    def upsert(self, ids, embeddings, metadatas, documents):
        self.upserted = {"ids": ids, "embeddings": embeddings}


class _FakeStore:
    def __init__(self, collection):
        self._collection = collection


def _make_store(stored, dim=4):
    store = ChromaVector.__new__(ChromaVector)   # 绕开真实 Chroma/云端 embedding
    store.embedding_model = _FakeEmbeddings(dim)
    store._executor = ThreadPoolExecutor(max_workers=1)
    store.vector_store = _FakeStore(_FakeCollection(stored))
    store._bump_index_version = lambda: None
    return store


def _docs(n=2):
    return [Document(page_content=f"块{i}", metadata={"id": f"doc-{i}"}) for i in range(n)]


def test_aadd_documents_survives_empty_collection():
    """空集合（numpy 空数组）时应正常入库，不得抛布尔歧义异常。"""
    store = _make_store(np.array([]))
    asyncio.run(store.aadd_documents(_docs(2)))
    assert store.vector_store._collection.upserted["ids"] == ["doc-0", "doc-1"]


def test_aadd_documents_still_rejects_dimension_mismatch():
    """空集合判断修好之后，维度不一致的显式报错仍必须保留。"""
    store = _make_store(np.array([[0.1] * 8]), dim=4)
    with pytest.raises(ValueError, match="嵌入维度不一致"):
        asyncio.run(store.aadd_documents(_docs(1)))
    assert store.vector_store._collection.upserted is None
