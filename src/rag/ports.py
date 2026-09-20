from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rag.domain import (
    Chunk,
    Document,
    GeneratedAnswer,
    IndexedDocumentVersion,
    Page,
    RetrievedChunk,
    SyncReport,
    SyncStage,
)


@dataclass(frozen=True)
class GatewayProbe:
    vector_dimension: int
    structured_output_endpoints: tuple[str, ...]


class ModelGatewayDiagnostics(Protocol):
    def probe(self) -> GatewayProbe: ...


class CorpusSource(Protocol):
    def discover(self) -> Sequence[Document]: ...


class PdfExtractor(Protocol):
    def extract(self, document: Document) -> Sequence[Page]: ...


class PageChunker(Protocol):
    def split(self, document: Document, pages: Sequence[Page]) -> Sequence[Chunk]: ...


class QueryEmbeddingProvider(Protocol):
    def embed_query(self, text: str) -> Sequence[float]: ...


class EmbeddingProvider(QueryEmbeddingProvider, Protocol):
    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class Retriever(Protocol):
    def retrieve(self, query: str, *, limit: int) -> Sequence[RetrievedChunk]: ...


class AnswerGenerator(Protocol):
    def generate(
        self, question: str, evidence: Sequence[RetrievedChunk]
    ) -> GeneratedAnswer: ...


class CorpusSynchronizer(Protocol):
    def synchronize(self, *, dry_run: bool = False) -> SyncReport: ...


class ChunkIndex(Protocol):
    def inventory(self) -> Sequence[IndexedDocumentVersion]: ...

    def contains_all(self, chunk_ids: Sequence[str]) -> bool: ...

    def index(self, chunks: Sequence[Chunk]) -> None: ...

    def delete_version(self, document_id: str, content_hash: str) -> None: ...

    def delete_other_versions(self, document_id: str, keep_hash: str) -> None: ...

    def delete_document(self, document_id: str) -> None: ...


class SyncOperationError(RuntimeError):
    def __init__(self, stage: SyncStage, message: str):
        super().__init__(message)
        self.stage = stage


class CorpusSourceError(RuntimeError):
    """A safe-to-display corpus discovery failure."""


class RetrievalError(RuntimeError):
    """A safe-to-display retrieval failure."""


class AnswerGenerationError(RuntimeError):
    """A safe-to-display generation failure."""
