from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from rag.application.sync import SyncCorpus
from rag.domain import (
    Chunk,
    Document,
    IndexedDocumentVersion,
    Page,
    SyncAction,
    SyncOutcome,
    SyncStage,
)
from rag.domain.identity import chunk_id, document_id, page_id
from rag.infrastructure.embeddings import EmbeddingRequestError
from rag.infrastructure.qdrant import COLLECTION_NAME, LangChainQdrantChunkIndex
from rag.ports import ChunkIndex, SyncOperationError


class CountingEmbeddings(Embeddings):
    def __init__(self, *, fail: bool = False) -> None:
        self.document_calls = 0
        self.fail = fail

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        if self.fail:
            raise EmbeddingRequestError("RouterAI document embedding failed")
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [float(len(text)), float(text.count("alpha")), 1.0]


class StaticSource:
    def __init__(self, document: Document) -> None:
        self.document = document

    def discover(self) -> Sequence[Document]:
        return [self.document]


class StaticExtractor:
    def __init__(self, pages: Sequence[Page]) -> None:
        self.pages = pages

    def extract(self, document: Document) -> Sequence[Page]:
        return self.pages


class StaticChunker:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = chunks

    def split(self, document: Document, pages: Sequence[Page]) -> Sequence[Chunk]:
        return self.chunks


class MutableSource:
    def __init__(self, documents: Sequence[Document]) -> None:
        self.documents = documents

    def discover(self) -> Sequence[Document]:
        return self.documents


class DocumentExtractor:
    def __init__(self, *, fail_hash: str | None = None) -> None:
        self.fail_hash = fail_hash

    def extract(self, document: Document) -> Sequence[Page]:
        if document.content_hash == self.fail_hash:
            raise SyncOperationError(
                SyncStage.EXTRACTION, f"unable to extract {document.filename}"
            )
        text = f"evidence {document.content_hash[0]}"
        return (
            Page(
                id=page_id(document, 1),
                document_id=document.id,
                viewer_page=1,
                text=text,
            ),
        )


class DocumentChunker:
    def __init__(self, *, chunks_per_page: int = 1) -> None:
        self.chunks_per_page = chunks_per_page

    def split(self, document: Document, pages: Sequence[Page]) -> Sequence[Chunk]:
        return tuple(
            Chunk(
                id=chunk_id(page, index, f"{page.text} {index}"),
                document_id=document.id,
                source_path=document.source_path,
                filename=document.filename,
                content_hash=document.content_hash,
                viewer_page=page.viewer_page,
                chunk_index=index,
                text=f"{page.text} {index}",
            )
            for page in pages
            for index in range(self.chunks_per_page)
        )


class PartialUploadIndex:
    def __init__(self, delegate: ChunkIndex) -> None:
        self.delegate = delegate

    def inventory(self) -> Sequence[IndexedDocumentVersion]:
        return self.delegate.inventory()

    def contains_all(self, chunk_ids: Sequence[str]) -> bool:
        return self.delegate.contains_all(chunk_ids)

    def index(self, chunks: Sequence[Chunk]) -> None:
        self.delegate.index(chunks[:1])
        raise SyncOperationError(SyncStage.STORAGE, "simulated partial upload")

    def delete_version(self, document_id: str, content_hash: str) -> None:
        self.delegate.delete_version(document_id, content_hash)

    def delete_other_versions(self, document_id: str, keep_hash: str) -> None:
        self.delegate.delete_other_versions(document_id, keep_hash)

    def delete_document(self, document_id: str) -> None:
        self.delegate.delete_document(document_id)


def version(path: Path, marker: str) -> Document:
    canonical = path.resolve()
    return Document(
        id=document_id(canonical),
        source_path=canonical,
        filename=canonical.name,
        content_hash=marker * 64,
    )


def content(tmp_path: Path) -> tuple[Document, tuple[Page, ...], tuple[Chunk, ...]]:
    source_path = (tmp_path / "source.pdf").resolve()
    document = Document(
        id=document_id(source_path),
        source_path=source_path,
        filename=source_path.name,
        content_hash="a" * 64,
    )
    page = Page(
        id=page_id(document, 1),
        document_id=document.id,
        viewer_page=1,
        text="alpha evidence",
    )
    empty_page = Page(
        id=page_id(document, 2),
        document_id=document.id,
        viewer_page=2,
        text="  ",
    )
    chunks = (
        Chunk(
            id=chunk_id(page, 0, page.text),
            document_id=document.id,
            source_path=source_path,
            filename=source_path.name,
            content_hash=document.content_hash,
            viewer_page=1,
            chunk_index=0,
            text=page.text,
        ),
    )
    return document, (page, empty_page), chunks


def test_persists_retrievable_chunks_and_skips_unchanged_version(
    tmp_path: Path,
) -> None:
    document, pages, chunks = content(tmp_path)
    embeddings = CountingEmbeddings()
    index = LangChainQdrantChunkIndex(tmp_path / "state", embeddings)
    sync = SyncCorpus(
        StaticSource(document), StaticExtractor(pages), StaticChunker(chunks), index
    )

    first = sync.synchronize()
    calls_after_first = embeddings.document_calls
    second = sync.synchronize()

    assert first.indexed_documents == 1
    assert first.indexed_pages == 1
    assert first.indexed_chunks == 1
    assert first.skipped_pages[0].viewer_page == 2
    assert first.warnings == ("textless page skipped: source.pdf, PDF p. 2",)
    assert second.unchanged_documents == 1
    assert embeddings.document_calls == calls_after_first

    client = QdrantClient(path=str(tmp_path / "state" / "qdrant"))
    try:
        collection = client.get_collection(COLLECTION_NAME)
        vectors = collection.config.params.vectors
        assert isinstance(vectors, dict)
        assert vectors[""].size == 3
        assert client.count(COLLECTION_NAME, exact=True).count == 1
        records, _ = client.scroll(
            COLLECTION_NAME, limit=10, with_payload=True, with_vectors=False
        )
        assert str(records[0].id) == chunks[0].id
        assert records[0].payload is not None
        assert records[0].payload["text"] == "alpha evidence"
        metadata = records[0].payload["metadata"]
        assert metadata["document_id"] == document.id
        assert metadata["source_path"] == str(document.source_path)
        assert metadata["filename"] == document.filename
        assert metadata["chunk_index"] == 0
        store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
            content_payload_key="text",
            metadata_payload_key="metadata",
        )
        results = store.similarity_search("alpha", k=1)
        assert results[0].page_content == "alpha evidence"
        assert results[0].metadata["viewer_page"] == 1
        assert results[0].metadata["content_hash"] == document.content_hash
    finally:
        client.close()


def test_embedding_failure_is_reported_without_exposing_details(tmp_path: Path) -> None:
    document, pages, chunks = content(tmp_path)
    index = LangChainQdrantChunkIndex(tmp_path / "state", CountingEmbeddings(fail=True))
    sync = SyncCorpus(
        StaticSource(document), StaticExtractor(pages), StaticChunker(chunks), index
    )

    report = sync.synchronize()

    assert report.indexed_documents == 0
    assert report.failures[0].stage is SyncStage.EMBEDDING
    assert report.failures[0].message == "RouterAI document embedding failed"


def test_partial_chunk_set_is_not_treated_as_unchanged(tmp_path: Path) -> None:
    document, pages, chunks = content(tmp_path)
    page = pages[0]
    second_text = "beta evidence"
    second = chunks[0].model_copy(
        update={
            "id": chunk_id(page, 1, second_text),
            "chunk_index": 1,
            "text": second_text,
        }
    )
    embeddings = CountingEmbeddings()
    index = LangChainQdrantChunkIndex(tmp_path / "state", embeddings)
    index.index(chunks)
    sync = SyncCorpus(
        StaticSource(document),
        StaticExtractor(pages),
        StaticChunker((*chunks, second)),
        index,
    )

    report = sync.synchronize()

    assert report.indexed_documents == 1
    client = QdrantClient(path=str(tmp_path / "state" / "qdrant"))
    try:
        assert client.count(COLLECTION_NAME, exact=True).count == 2
    finally:
        client.close()


def test_changed_document_replaces_previous_version_after_complete_upload(
    tmp_path: Path,
) -> None:
    old = version(tmp_path / "source.pdf", "a")
    changed = version(tmp_path / "source.pdf", "b")
    source = MutableSource([old])
    embeddings = CountingEmbeddings()
    index = LangChainQdrantChunkIndex(tmp_path / "state", embeddings)
    sync = SyncCorpus(source, DocumentExtractor(), DocumentChunker(), index)
    sync.synchronize()

    source.documents = [changed]
    report = sync.synchronize()

    assert report.indexed_documents == 1
    assert report.changes[0].action is SyncAction.UPDATE
    assert report.changes[0].outcome is SyncOutcome.COMPLETED
    assert report.changes[0].previous_content_hash == old.content_hash
    inventory = index.inventory()
    assert [item.content_hash for item in inventory] == [changed.content_hash]


def test_embedding_failure_preserves_previous_usable_version(tmp_path: Path) -> None:
    old = version(tmp_path / "source.pdf", "a")
    changed = version(tmp_path / "source.pdf", "b")
    source = MutableSource([old])
    embeddings = CountingEmbeddings()
    index = LangChainQdrantChunkIndex(tmp_path / "state", embeddings)
    sync = SyncCorpus(source, DocumentExtractor(), DocumentChunker(), index)
    sync.synchronize()

    embeddings.fail = True
    source.documents = [changed]
    report = sync.synchronize()

    assert report.failures[0].stage is SyncStage.EMBEDDING
    assert report.changes[0].outcome is SyncOutcome.FAILED
    assert [item.content_hash for item in index.inventory()] == [old.content_hash]


def test_partial_upload_is_cleaned_and_previous_version_is_preserved(
    tmp_path: Path,
) -> None:
    old = version(tmp_path / "source.pdf", "a")
    changed = version(tmp_path / "source.pdf", "b")
    source = MutableSource([old])
    embeddings = CountingEmbeddings()
    index = LangChainQdrantChunkIndex(tmp_path / "state", embeddings)
    chunker = DocumentChunker(chunks_per_page=2)
    SyncCorpus(source, DocumentExtractor(), chunker, index).synchronize()

    source.documents = [changed]
    report = SyncCorpus(
        source, DocumentExtractor(), chunker, PartialUploadIndex(index)
    ).synchronize()

    assert report.failures[0].message == "simulated partial upload"
    inventory = index.inventory()
    assert [(item.content_hash, len(item.chunk_ids)) for item in inventory] == [
        (old.content_hash, 2)
    ]


def test_dry_run_reports_exact_changes_without_writes_or_embeddings(
    tmp_path: Path,
) -> None:
    old = version(tmp_path / "updated.pdf", "a")
    missing = version(tmp_path / "missing.pdf", "m")
    changed = version(tmp_path / "updated.pdf", "b")
    added = version(tmp_path / "added.pdf", "c")
    source = MutableSource([old, missing])
    embeddings = CountingEmbeddings()
    index = LangChainQdrantChunkIndex(tmp_path / "state", embeddings)
    sync = SyncCorpus(source, DocumentExtractor(), DocumentChunker(), index)
    sync.synchronize()
    inventory_before = index.inventory()
    embedding_calls_before = embeddings.document_calls

    source.documents = [changed, added]
    report = sync.synchronize(dry_run=True)

    assert report.dry_run is True
    assert [(change.action, change.outcome) for change in report.changes] == [
        (SyncAction.UPDATE, SyncOutcome.PLANNED),
        (SyncAction.ADD, SyncOutcome.PLANNED),
        (SyncAction.REMOVE, SyncOutcome.PLANNED),
    ]
    assert index.inventory() == inventory_before
    assert embeddings.document_calls == embedding_calls_before


def test_rename_is_applied_as_addition_and_removal(tmp_path: Path) -> None:
    old = version(tmp_path / "old.pdf", "a")
    renamed = version(tmp_path / "renamed.pdf", "a")
    source = MutableSource([old])
    index = LangChainQdrantChunkIndex(tmp_path / "state", CountingEmbeddings())
    sync = SyncCorpus(source, DocumentExtractor(), DocumentChunker(), index)
    sync.synchronize()

    source.documents = [renamed]
    report = sync.synchronize()

    assert [(change.action, change.outcome) for change in report.changes] == [
        (SyncAction.ADD, SyncOutcome.COMPLETED),
        (SyncAction.REMOVE, SyncOutcome.COMPLETED),
    ]
    assert {item.document_id for item in index.inventory()} == {renamed.id}


def test_missing_document_deletion_is_deferred_when_present_document_fails(
    tmp_path: Path,
) -> None:
    missing = version(tmp_path / "missing.pdf", "a")
    broken = version(tmp_path / "broken.pdf", "b")
    source = MutableSource([missing])
    index = LangChainQdrantChunkIndex(tmp_path / "state", CountingEmbeddings())
    SyncCorpus(source, DocumentExtractor(), DocumentChunker(), index).synchronize()

    source.documents = [broken]
    report = SyncCorpus(
        source,
        DocumentExtractor(fail_hash=broken.content_hash),
        DocumentChunker(),
        index,
    ).synchronize()

    assert report.failures[0].stage is SyncStage.EXTRACTION
    assert [(change.action, change.outcome) for change in report.changes] == [
        (SyncAction.ADD, SyncOutcome.FAILED),
        (SyncAction.REMOVE, SyncOutcome.DEFERRED),
    ]
    assert {item.document_id for item in index.inventory()} == {missing.id}


def test_successful_documents_advance_while_missing_deletion_is_deferred(
    tmp_path: Path,
) -> None:
    old = version(tmp_path / "updated.pdf", "a")
    changed = version(tmp_path / "updated.pdf", "c")
    missing = version(tmp_path / "missing.pdf", "m")
    broken = version(tmp_path / "broken.pdf", "b")
    source = MutableSource([old, missing])
    index = LangChainQdrantChunkIndex(tmp_path / "state", CountingEmbeddings())
    SyncCorpus(source, DocumentExtractor(), DocumentChunker(), index).synchronize()

    source.documents = [changed, broken]
    report = SyncCorpus(
        source,
        DocumentExtractor(fail_hash=broken.content_hash),
        DocumentChunker(),
        index,
    ).synchronize()

    assert [(change.action, change.outcome) for change in report.changes] == [
        (SyncAction.UPDATE, SyncOutcome.COMPLETED),
        (SyncAction.ADD, SyncOutcome.FAILED),
        (SyncAction.REMOVE, SyncOutcome.DEFERRED),
    ]
    assert {(item.document_id, item.content_hash) for item in index.inventory()} == {
        (changed.id, changed.content_hash),
        (missing.id, missing.content_hash),
    }


def test_failure_free_sync_removes_missing_documents(tmp_path: Path) -> None:
    removed = version(tmp_path / "removed.pdf", "a")
    source = MutableSource([removed])
    index = LangChainQdrantChunkIndex(tmp_path / "state", CountingEmbeddings())
    sync = SyncCorpus(source, DocumentExtractor(), DocumentChunker(), index)
    sync.synchronize()

    source.documents = []
    report = sync.synchronize()

    assert report.failures == ()
    assert report.changes[0].action is SyncAction.REMOVE
    assert report.changes[0].outcome is SyncOutcome.COMPLETED
    assert index.inventory() == ()
