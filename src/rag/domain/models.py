from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DomainModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class Document(DomainModel):
    id: str
    source_path: Path
    filename: str
    content_hash: str


class Page(DomainModel):
    id: str
    document_id: str
    viewer_page: int = Field(ge=1)
    label: str | None = None
    text: str


class Chunk(DomainModel):
    id: str
    document_id: str
    source_path: Path
    filename: str
    content_hash: str
    viewer_page: int = Field(ge=1)
    page_label: str | None = None
    chunk_index: int = Field(ge=0)
    text: str


class RetrievedChunk(DomainModel):
    chunk: Chunk
    score: float
    source_id: str | None = None


class Claim(DomainModel):
    text: str
    source_ids: tuple[str, ...]


class Citation(DomainModel):
    source_id: str
    source_path: Path
    filename: str
    viewer_page: int = Field(ge=1)
    page_label: str | None = None
    score: float
    excerpt: str


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class AnswerResult(DomainModel):
    status: AnswerStatus
    claims: tuple[Claim, ...] = ()
    citations: tuple[Citation, ...] = ()

    @model_validator(mode="after")
    def status_matches_claims(self) -> AnswerResult:
        if self.status is AnswerStatus.ANSWERED and not self.claims:
            raise ValueError("answered results require at least one claim")
        if self.status is AnswerStatus.INSUFFICIENT_EVIDENCE and self.claims:
            raise ValueError("insufficient evidence results cannot contain claims")
        return self


class SyncStage(StrEnum):
    EXTRACTION = "extraction"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORAGE = "storage"


class SkippedPage(DomainModel):
    source_path: Path
    viewer_page: int = Field(ge=1)
    page_label: str | None = None
    reason: str


class SyncFailure(DomainModel):
    source_path: Path
    stage: SyncStage
    message: str


class SyncAction(StrEnum):
    ADD = "add"
    UPDATE = "update"
    SKIP = "skip"
    REMOVE = "remove"


class SyncOutcome(StrEnum):
    PLANNED = "planned"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    DEFERRED = "deferred"
    FAILED = "failed"


class SyncChange(DomainModel):
    action: SyncAction
    outcome: SyncOutcome
    source_path: Path
    content_hash: str | None = None
    previous_content_hash: str | None = None


class IndexedDocumentVersion(DomainModel):
    document_id: str
    source_path: Path
    filename: str
    content_hash: str
    chunk_ids: tuple[str, ...]


class SyncReport(DomainModel):
    dry_run: bool = False
    indexed_documents: int = Field(default=0, ge=0)
    unchanged_documents: int = Field(default=0, ge=0)
    indexed_pages: int = Field(default=0, ge=0)
    indexed_chunks: int = Field(default=0, ge=0)
    skipped_pages: tuple[SkippedPage, ...] = ()
    changes: tuple[SyncChange, ...] = ()
    warnings: tuple[str, ...] = ()
    failures: tuple[SyncFailure, ...] = ()


class IndexFingerprint(DomainModel):
    schema_version: int = Field(ge=1)
    corpus_root: Path
    collection_name: str
    embedding_model: str
    vector_dimension: int = Field(gt=0)
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    created_at: datetime | None = None
