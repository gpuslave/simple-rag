from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from rag.domain import Chunk
from rag.infrastructure.qdrant import LangChainQdrantChunkIndex
from rag.infrastructure.retrieval import QdrantDenseRetriever
from rag.ports import RetrievalError


class RetrievalEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if text == "best" else [0.0, 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [1.0, 0.0]


def chunk(chunk_id: str, text: str, page: int) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="document-1",
        source_path=Path("/corpus/lesson.pdf"),
        filename="lesson.pdf",
        content_hash="hash",
        viewer_page=page,
        page_label="L1" if page == 1 else None,
        chunk_index=page - 1,
        text=text,
    )


def test_dense_retrieval_embeds_query_and_returns_ranked_payloads(
    tmp_path: Path,
) -> None:
    embeddings = RetrievalEmbeddings()
    chunks = (
        chunk("00000000-0000-0000-0000-000000000001", "other", 2),
        chunk("00000000-0000-0000-0000-000000000002", "best", 1),
    )
    LangChainQdrantChunkIndex(tmp_path, embeddings).index(chunks)

    results = QdrantDenseRetriever(tmp_path, embeddings).retrieve("question", limit=2)

    assert embeddings.query_calls == ["question"]
    assert [result.chunk.text for result in results] == ["best", "other"]
    assert results[0].score > results[1].score
    assert results[0].chunk.page_label == "L1"
    assert results[0].source_id is None


def test_dense_retrieval_requires_an_index(tmp_path: Path) -> None:
    with pytest.raises(RetrievalError, match="run rag sync"):
        QdrantDenseRetriever(tmp_path, RetrievalEmbeddings()).retrieve(
            "question", limit=30
        )
