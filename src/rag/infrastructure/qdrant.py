from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.documents import Document as LangChainDocument
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from rag.domain import Chunk, SyncStage
from rag.infrastructure.embeddings import EmbeddingRequestError
from rag.ports import SyncOperationError

COLLECTION_NAME = "chunks"


class LangChainQdrantChunkIndex:
    def __init__(self, state_path: Path, embeddings: Embeddings) -> None:
        self._path = state_path / "qdrant"
        self._embeddings = embeddings

    def contains_all(self, chunk_ids: Sequence[str]) -> bool:
        if not chunk_ids or not self._path.exists():
            return False
        client = QdrantClient(path=str(self._path))
        try:
            if not client.collection_exists(COLLECTION_NAME):
                return False
            records = client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=list(chunk_ids),
                with_payload=False,
                with_vectors=False,
            )
            found = {str(record.id) for record in records}
            return found == set(chunk_ids)
        except Exception as exc:
            raise SyncOperationError(
                SyncStage.STORAGE, "unable to inspect the local Qdrant index"
            ) from exc
        finally:
            client.close()

    def index(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        documents = [self._to_langchain_document(chunk) for chunk in chunks]
        ids = [chunk.id for chunk in chunks]

        try:
            if self._collection_exists():
                self._add_to_existing(documents, ids)
            else:
                store = QdrantVectorStore.from_documents(
                    documents=documents,
                    embedding=self._embeddings,
                    ids=ids,
                    path=str(self._path),
                    collection_name=COLLECTION_NAME,
                    content_payload_key="text",
                    metadata_payload_key="metadata",
                )
                store.client.close()
        except EmbeddingRequestError as exc:
            raise SyncOperationError(SyncStage.EMBEDDING, str(exc)) from exc
        except Exception as exc:
            raise SyncOperationError(
                SyncStage.STORAGE, "unable to update the local Qdrant index"
            ) from exc

    def _collection_exists(self) -> bool:
        if not self._path.exists():
            return False
        client = QdrantClient(path=str(self._path))
        try:
            return client.collection_exists(COLLECTION_NAME)
        finally:
            client.close()

    def _add_to_existing(
        self, documents: list[LangChainDocument], ids: list[str]
    ) -> None:
        client = QdrantClient(path=str(self._path))
        try:
            store = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=self._embeddings,
                content_payload_key="text",
                metadata_payload_key="metadata",
            )
            store.add_documents(documents=documents, ids=ids)
        finally:
            client.close()

    @staticmethod
    def _to_langchain_document(chunk: Chunk) -> LangChainDocument:
        metadata: dict[str, Any] = {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "source_path": str(chunk.source_path),
            "filename": chunk.filename,
            "content_hash": chunk.content_hash,
            "viewer_page": chunk.viewer_page,
            "page_label": chunk.page_label,
            "chunk_index": chunk.chunk_index,
        }
        return LangChainDocument(
            id=chunk.id,
            page_content=chunk.text,
            metadata=metadata,
        )
