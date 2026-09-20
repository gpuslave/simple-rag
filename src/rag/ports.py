from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rag.domain import AnswerResult, Chunk, Page, RetrievedChunk, SyncReport


@dataclass(frozen=True)
class GatewayProbe:
    vector_dimension: int
    structured_output_endpoints: tuple[str, ...]


class ModelGatewayDiagnostics(Protocol):
    def probe(self) -> GatewayProbe: ...


class PdfExtractor(Protocol):
    def extract(self, source_path: Path) -> Sequence[Page]: ...


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

    def embed_query(self, text: str) -> Sequence[float]: ...


class Retriever(Protocol):
    def retrieve(self, query: str, *, limit: int) -> Sequence[RetrievedChunk]: ...


class AnswerGenerator(Protocol):
    def generate(
        self, question: str, evidence: Sequence[RetrievedChunk]
    ) -> AnswerResult: ...


class CorpusSynchronizer(Protocol):
    def synchronize(self, *, dry_run: bool = False) -> SyncReport: ...


class ChunkStore(Protocol):
    def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> None: ...
