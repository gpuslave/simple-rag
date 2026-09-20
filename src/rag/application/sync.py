from __future__ import annotations

from rag.domain import SkippedPage, SyncFailure, SyncReport
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
        if dry_run:
            raise SyncExecutionError("dry-run synchronization arrives in ticket 03")

        try:
            documents = self._source.discover()
        except CorpusSourceError as exc:
            raise SyncExecutionError(str(exc)) from exc
        if not documents:
            raise SyncExecutionError("no PDF files found in the configured corpus")

        indexed_documents = 0
        unchanged_documents = 0
        indexed_pages = 0
        indexed_chunks = 0
        skipped_pages: list[SkippedPage] = []
        warnings: list[str] = []
        failures: list[SyncFailure] = []

        for document in documents:
            try:
                pages = self._extractor.extract(document)
                usable_pages = [page for page in pages if page.text.strip()]
                for page in pages:
                    if page.text.strip():
                        continue
                    skipped = SkippedPage(
                        source_path=document.source_path,
                        viewer_page=page.viewer_page,
                        page_label=page.label,
                        reason="textless page",
                    )
                    skipped_pages.append(skipped)
                    warnings.append(
                        f"textless page skipped: {document.filename}, "
                        f"PDF p. {page.viewer_page}"
                    )

                chunks = tuple(self._chunker.split(document, usable_pages))
                if not chunks:
                    continue
                if self._index.contains_all([chunk.id for chunk in chunks]):
                    unchanged_documents += 1
                    continue

                self._index.index(chunks)
                indexed_documents += 1
                indexed_pages += len(usable_pages)
                indexed_chunks += len(chunks)
            except SyncOperationError as exc:
                failures.append(
                    SyncFailure(
                        source_path=document.source_path,
                        stage=exc.stage,
                        message=str(exc),
                    )
                )

        return SyncReport(
            indexed_documents=indexed_documents,
            unchanged_documents=unchanged_documents,
            indexed_pages=indexed_pages,
            indexed_chunks=indexed_chunks,
            skipped_pages=tuple(skipped_pages),
            warnings=tuple(warnings),
            failures=tuple(failures),
        )
