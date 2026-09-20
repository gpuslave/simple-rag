from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from rag.domain import (
    Chunk,
    Document,
    IndexedDocumentVersion,
    Page,
    SkippedPage,
    SyncAction,
    SyncChange,
    SyncFailure,
    SyncOutcome,
    SyncReport,
    SyncStage,
)
from rag.ports import (
    ChunkIndex,
    CorpusSource,
    CorpusSourceError,
    PageChunker,
    PdfExtractor,
    SyncOperationError,
)


class SyncExecutionError(RuntimeError):
    """A safe-to-display synchronization failure."""


class SyncCorpus:
    def __init__(
        self,
        source: CorpusSource,
        extractor: PdfExtractor,
        chunker: PageChunker,
        index: ChunkIndex,
    ) -> None:
        self._source = source
        self._extractor = extractor
        self._chunker = chunker
        self._index = index

    def synchronize(self, *, dry_run: bool = False) -> SyncReport:
        try:
            documents = tuple(self._source.discover())
            inventory = tuple(self._index.inventory())
        except (CorpusSourceError, SyncOperationError) as exc:
            raise SyncExecutionError(str(exc)) from exc

        versions_by_document: dict[str, list[IndexedDocumentVersion]] = defaultdict(
            list
        )
        for version in inventory:
            versions_by_document[version.document_id].append(version)

        indexed_documents = 0
        unchanged_documents = 0
        indexed_pages = 0
        indexed_chunks = 0
        skipped_pages: list[SkippedPage] = []
        changes: list[SyncChange] = []
        warnings: list[str] = []
        failures: list[SyncFailure] = []

        for document in documents:
            versions = versions_by_document.get(document.id, [])
            previous_hash = self._previous_hash(versions, document.content_hash)
            action = SyncAction.UPDATE if versions else SyncAction.ADD
            try:
                pages = tuple(self._extractor.extract(document))
                usable_pages = self._usable_pages(
                    document, pages, skipped_pages, warnings
                )
                chunks = tuple(self._chunker.split(document, usable_pages))
                if not chunks:
                    raise SyncOperationError(
                        SyncStage.CHUNKING,
                        f"no indexable text found in {document.filename}",
                    )

                matching = next(
                    (
                        version
                        for version in versions
                        if version.content_hash == document.content_hash
                    ),
                    None,
                )
                if matching is not None and self._index.contains_all(
                    [chunk.id for chunk in chunks]
                ):
                    unchanged_documents += 1
                    changes.append(
                        self._change(
                            document,
                            SyncAction.SKIP,
                            SyncOutcome.SKIPPED,
                            matching.content_hash,
                        )
                    )
                    continue

                if dry_run:
                    changes.append(
                        self._change(
                            document, action, SyncOutcome.PLANNED, previous_hash
                        )
                    )
                    continue

                self._stage_version(document, chunks, warnings)
                self._index.delete_other_versions(document.id, document.content_hash)
                indexed_documents += 1
                indexed_pages += len(usable_pages)
                indexed_chunks += len(chunks)
                changes.append(
                    self._change(document, action, SyncOutcome.COMPLETED, previous_hash)
                )
            except SyncOperationError as exc:
                failures.append(
                    SyncFailure(
                        source_path=document.source_path,
                        stage=exc.stage,
                        message=str(exc),
                    )
                )
                changes.append(
                    self._change(document, action, SyncOutcome.FAILED, previous_hash)
                )

        present_ids = {document.id for document in documents}
        missing = self._missing_documents(inventory, present_ids)
        if failures:
            for version in missing:
                changes.append(self._removal_change(version, SyncOutcome.DEFERRED))
        elif dry_run:
            for version in missing:
                changes.append(self._removal_change(version, SyncOutcome.PLANNED))
        else:
            for version in missing:
                try:
                    self._index.delete_document(version.document_id)
                    changes.append(self._removal_change(version, SyncOutcome.COMPLETED))
                except SyncOperationError as exc:
                    failures.append(
                        SyncFailure(
                            source_path=version.source_path,
                            stage=exc.stage,
                            message=str(exc),
                        )
                    )
                    changes.append(self._removal_change(version, SyncOutcome.FAILED))

        return SyncReport(
            dry_run=dry_run,
            indexed_documents=indexed_documents,
            unchanged_documents=unchanged_documents,
            indexed_pages=indexed_pages,
            indexed_chunks=indexed_chunks,
            skipped_pages=tuple(skipped_pages),
            changes=tuple(changes),
            warnings=tuple(warnings),
            failures=tuple(failures),
        )

    def _stage_version(
        self, document: Document, chunks: Sequence[Chunk], warnings: list[str]
    ) -> None:
        try:
            self._index.index(chunks)
            if not self._index.contains_all([chunk.id for chunk in chunks]):
                raise SyncOperationError(
                    SyncStage.STORAGE,
                    f"incomplete upload for {document.filename}",
                )
        except SyncOperationError:
            try:
                self._index.delete_version(document.id, document.content_hash)
            except SyncOperationError as cleanup_error:
                warnings.append(
                    f"partial upload cleanup failed for {document.filename}: "
                    f"{cleanup_error}"
                )
            raise

    @staticmethod
    def _usable_pages(
        document: Document,
        pages: Sequence[Page],
        skipped_pages: list[SkippedPage],
        warnings: list[str],
    ) -> tuple[Page, ...]:
        usable: list[Page] = []
        for page in pages:
            if page.text.strip():
                usable.append(page)
                continue
            skipped_pages.append(
                SkippedPage(
                    source_path=document.source_path,
                    viewer_page=page.viewer_page,
                    page_label=page.label,
                    reason="textless page",
                )
            )
            warnings.append(
                f"textless page skipped: {document.filename}, PDF p. {page.viewer_page}"
            )
        return tuple(usable)

    @staticmethod
    def _previous_hash(
        versions: Sequence[IndexedDocumentVersion], current_hash: str
    ) -> str | None:
        return next(
            (
                version.content_hash
                for version in versions
                if version.content_hash != current_hash
            ),
            versions[0].content_hash if versions else None,
        )

    @staticmethod
    def _missing_documents(
        inventory: Sequence[IndexedDocumentVersion], present_ids: set[str]
    ) -> tuple[IndexedDocumentVersion, ...]:
        missing: dict[str, IndexedDocumentVersion] = {}
        for version in inventory:
            if version.document_id not in present_ids:
                missing.setdefault(version.document_id, version)
        return tuple(sorted(missing.values(), key=lambda item: str(item.source_path)))

    @staticmethod
    def _change(
        document: Document,
        action: SyncAction,
        outcome: SyncOutcome,
        previous_hash: str | None,
    ) -> SyncChange:
        return SyncChange(
            action=action,
            outcome=outcome,
            source_path=document.source_path,
            content_hash=document.content_hash,
            previous_content_hash=previous_hash,
        )

    @staticmethod
    def _removal_change(
        version: IndexedDocumentVersion, outcome: SyncOutcome
    ) -> SyncChange:
        return SyncChange(
            action=SyncAction.REMOVE,
            outcome=outcome,
            source_path=version.source_path,
            previous_content_hash=version.content_hash,
        )
