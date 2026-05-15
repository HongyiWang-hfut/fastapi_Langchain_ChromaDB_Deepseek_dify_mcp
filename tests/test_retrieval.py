"""RAG 检索测试：文档切分、演示嵌入、向量检索。"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("DASHSCOPE_API_KEY", "")

from document_processing import load_campus_documents, split_campus_documents, load_and_split_campus_documents
from vector_store import DemoEmbeddings, retrieve_top_chunks, build_campus_chroma


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


class TestDocumentProcessing:
    def test_load_campus_documents_returns_documents(self):
        docs = load_campus_documents(DATA_DIR)
        assert len(docs) > 0
        assert all(hasattr(d, "page_content") for d in docs)

    def test_split_campus_documents_creates_chunks(self):
        docs = load_campus_documents(DATA_DIR)
        chunks = split_campus_documents(docs, chunk_size=700, chunk_overlap=150)
        assert len(chunks) >= len(docs)
        assert all(len(c.page_content) > 0 for c in chunks)

    def test_load_and_split_end_to_end(self):
        docs, chunks = load_and_split_campus_documents(
            DATA_DIR, chunk_size=700, chunk_overlap=150, verbose=False
        )
        assert len(docs) > 0
        assert len(chunks) > 0
        assert all(c.metadata for c in chunks)

    def test_load_nonexistent_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            load_campus_documents("/nonexistent/path")


class TestVectorStore:
    @pytest.fixture
    def demo_embeddings(self):
        return DemoEmbeddings(dimension=128)

    def test_demo_embedding_dimension(self, demo_embeddings):
        vec = demo_embeddings.embed_query("测试")
        assert len(vec) == 128

    def test_demo_embedding_deterministic(self, demo_embeddings):
        vec1 = demo_embeddings.embed_query("相同的文本")
        vec2 = demo_embeddings.embed_query("相同的文本")
        assert vec1 == vec2

    def test_demo_embedding_different_texts(self, demo_embeddings):
        vec1 = demo_embeddings.embed_query("文本A")
        vec2 = demo_embeddings.embed_query("文本B")
        assert vec1 != vec2

    def test_build_and_retrieve(self, demo_embeddings, tmp_path):
        from langchain_core.documents import Document

        chunks = [
            Document(page_content="图书馆开放时间：周一至周日 7:30-22:30", metadata={"source": "test"}),
            Document(page_content="借阅规则：本科生可借图书15册，借期30天", metadata={"source": "test"}),
            Document(page_content="校园网SSID：Campus-WiFi，全校覆盖", metadata={"source": "test"}),
        ]
        persist_dir = tmp_path / "chroma_test"
        vs = build_campus_chroma(
            chunks,
            persist_directory=str(persist_dir),
            collection_name="test_collection",
            embeddings=demo_embeddings,
            reset=True,
        )
        assert vs._collection.count() == 3

        hits = retrieve_top_chunks(vs, "图书馆", k=2, max_margin_from_best=0.5)
        assert len(hits) <= 2
        assert all("content" in h for h in hits)
        assert all("distance" in h for h in hits)

    def test_build_empty_raises_value_error(self, demo_embeddings, tmp_path):
        persist_dir = tmp_path / "empty_test"
        with pytest.raises(ValueError, match="non-empty"):
            build_campus_chroma(
                [],
                persist_directory=str(persist_dir),
                collection_name="empty_test",
                embeddings=demo_embeddings,
                reset=True,
            )

    def test_retrieve_from_new_collection_returns_empty(self, demo_embeddings, tmp_path):
        from langchain_core.documents import Document

        persist_dir = tmp_path / "retrieve_test"
        vs = build_campus_chroma(
            [Document(page_content="唯一文档", metadata={"source": "test"})],
            persist_directory=str(persist_dir),
            collection_name="retrieve_test",
            embeddings=demo_embeddings,
            reset=True,
        )
        hits = retrieve_top_chunks(vs, "完全不相关的内容__xyz__", k=3, max_distance=0.1)
        assert hits == []
