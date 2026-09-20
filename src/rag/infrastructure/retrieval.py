from __future__ import annotations

from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from rag.domain import Chunk, RetrievedChunk
from rag.ports import QueryEmbeddingProvider, RetrievalError

from .qdrant import COLLECTION_NAME


class QdrantDenseRetriever:
    def __init__(self, state_path: Path, embeddings: QueryEmbeddingProvider) -> None:
        self._path = state_path / "qdrant"
        self._embeddings = embeddings

    def retrieve(self, query: str, *, limit: int) -> tuple[RetrievedChunk, ...]:
        if not self._path.exists():
            raise RetrievalError("local index is missing; run rag sync first")
        try:
            vector = list(self._embeddings.embed_query(query))
        except Exception as exc:
            raise RetrievalError("unable to embed the question") from exc
        if not vector:
            raise RetrievalError("question embedding was empty")

        client = QdrantClient(path=str(self._path))
        try:
            if not client.collection_exists(COLLECTION_NAME):
                raise RetrievalError("local index is missing; run rag sync first")
            points = client.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                using="",
                limit=limit,
                with_payload=True,
                with_vectors=False,
            ).points
            return tuple(self._to_retrieved(point) for point in points)
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError("unable to search the local index") from exc
        finally:
            client.close()

    @classmethod
    def _to_retrieved(cls, point: Any) -> RetrievedChunk:
        payload = point.payload
        if not isinstance(payload, dict):
            raise RetrievalError("local index contains invalid chunk metadata")
        metadata = payload.get("metadata")
        text = payload.get("text")
        if not isinstance(metadata, dict) or not isinstance(text, str):
            raise RetrievalError("local index contains invalid chunk metadata")

        page_label = metadata.get("page_label")
        if page_label is not None and not isinstance(page_label, str):
            raise RetrievalError("local index contains invalid chunk metadata")
        viewer_page = cls._integer(metadata, "viewer_page")
        chunk_index = cls._integer(metadata, "chunk_index")
        return RetrievedChunk(
            chunk=Chunk(
                id=cls._string(metadata, "chunk_id"),
                document_id=cls._string(metadata, "document_id"),
                source_path=Path(cls._string(metadata, "source_path")),
                filename=cls._string(metadata, "filename"),
                content_hash=cls._string(metadata, "content_hash"),
                viewer_page=viewer_page,
                page_label=page_label,
                chunk_index=chunk_index,
                text=text,
            ),
            score=float(point.score),
        )

    @staticmethod
    def _string(metadata: dict[str, Any], key: str) -> str:
        value = metadata.get(key)
        if not isinstance(value, str):
            raise RetrievalError("local index contains invalid chunk metadata")
        return value

    @staticmethod
    def _integer(metadata: dict[str, Any], key: str) -> int:
        value = metadata.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise RetrievalError("local index contains invalid chunk metadata")
        return value
