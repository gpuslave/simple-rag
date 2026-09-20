from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

from rag.domain import Chunk, IndexedDocumentVersion, SyncStage
from rag.infrastructure.embeddings import EmbeddingRequestError
from rag.ports import SyncOperationError

COLLECTION_NAME = "chunks"


class LangChainQdrantChunkIndex:
    def __init__(self, state_path: Path, embeddings: Embeddings) -> None:
        self._path = state_path / "qdrant"
        self._embeddings = embeddings

    def inventory(self) -> tuple[IndexedDocumentVersion, ...]:
        if not self._path.exists():
            return ()
        client = QdrantClient(path=str(self._path))
        try:
            if not client.collection_exists(COLLECTION_NAME):
                return ()
            grouped: dict[tuple[str, str], dict[str, Any]] = {}
            offset: Any = None
            while True:
                records, offset = client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for record in records:
                    payload = record.payload or {}
                    metadata = payload.get("metadata")
                    if not isinstance(metadata, dict):
                        continue
                    document_id = metadata.get("document_id")
                    content_hash = metadata.get("content_hash")
                    source_path = metadata.get("source_path")
                    filename = metadata.get("filename")
                    if not isinstance(document_id, str):
                        continue
                    if not isinstance(content_hash, str):
                        continue
                    if not isinstance(source_path, str):
                        continue
                    if not isinstance(filename, str):
                        continue
                    key = (document_id, content_hash)
                    group = grouped.setdefault(
                        key,
                        {
                            "source_path": source_path,
                            "filename": filename,
                            "chunk_ids": [],
                        },
                    )
                    group["chunk_ids"].append(str(record.id))
                if offset is None:
                    break
            return tuple(
                IndexedDocumentVersion(
                    document_id=document_id,
                    source_path=Path(str(values["source_path"])),
                    filename=str(values["filename"]),
                    content_hash=content_hash,
                    chunk_ids=tuple(sorted(values["chunk_ids"])),
                )
                for (document_id, content_hash), values in sorted(grouped.items())
            )
        except Exception as exc:
            raise SyncOperationError(
                SyncStage.STORAGE, "unable to inspect the local Qdrant index"
            ) from exc
        finally:
            client.close()

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
        try:
            vectors = self._embeddings.embed_documents([chunk.text for chunk in chunks])
            if not vectors or any(len(vector) != len(vectors[0]) for vector in vectors):
                raise ValueError("embedding response has inconsistent dimensions")
            client = QdrantClient(path=str(self._path))
            try:
                if not client.collection_exists(COLLECTION_NAME):
                    client.create_collection(
                        collection_name=COLLECTION_NAME,
                        vectors_config={
                            "": qdrant_models.VectorParams(
                                size=len(vectors[0]),
                                distance=qdrant_models.Distance.COSINE,
                            )
                        },
                    )
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[
                        qdrant_models.PointStruct(
                            id=chunk.id,
                            vector={"": vector},
                            payload=self._to_payload(chunk),
                        )
                        for chunk, vector in zip(chunks, vectors, strict=True)
                    ],
                    wait=True,
                )
            finally:
                client.close()
        except EmbeddingRequestError as exc:
            raise SyncOperationError(SyncStage.EMBEDDING, str(exc)) from exc
        except Exception as exc:
            raise SyncOperationError(
                SyncStage.STORAGE, "unable to update the local Qdrant index"
            ) from exc

    def delete_version(self, document_id: str, content_hash: str) -> None:
        self._delete(self._version_filter(document_id, content_hash))

    def delete_other_versions(self, document_id: str, keep_hash: str) -> None:
        self._delete(
            qdrant_models.Filter(
                must=[self._match("metadata.document_id", document_id)],
                must_not=[self._match("metadata.content_hash", keep_hash)],
            )
        )

    def delete_document(self, document_id: str) -> None:
        self._delete(
            qdrant_models.Filter(
                must=[self._match("metadata.document_id", document_id)]
            )
        )

    def _delete(self, points_filter: qdrant_models.Filter) -> None:
        if not self._path.exists():
            return
        client = QdrantClient(path=str(self._path))
        try:
            if not client.collection_exists(COLLECTION_NAME):
                return
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=qdrant_models.FilterSelector(filter=points_filter),
                wait=True,
            )
        except Exception as exc:
            raise SyncOperationError(
                SyncStage.STORAGE, "unable to remove data from the local Qdrant index"
            ) from exc
        finally:
            client.close()

    @classmethod
    def _version_filter(
        cls, document_id: str, content_hash: str
    ) -> qdrant_models.Filter:
        return qdrant_models.Filter(
            must=[
                cls._match("metadata.document_id", document_id),
                cls._match("metadata.content_hash", content_hash),
            ]
        )

    @staticmethod
    def _match(key: str, value: str) -> qdrant_models.FieldCondition:
        return qdrant_models.FieldCondition(
            key=key,
            match=qdrant_models.MatchValue(value=value),
        )

    @staticmethod
    def _to_payload(chunk: Chunk) -> dict[str, Any]:
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
        return {"text": chunk.text, "metadata": metadata}
