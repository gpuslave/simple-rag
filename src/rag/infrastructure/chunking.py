from __future__ import annotations

from collections.abc import Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.domain import Chunk, Document, Page, SyncStage
from rag.domain.identity import chunk_id
from rag.ports import SyncOperationError


class LangChainPageChunker:
    def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, document: Document, pages: Sequence[Page]) -> tuple[Chunk, ...]:
        try:
            chunks = []
            for page in pages:
                texts = self._splitter.split_text(page.text)
                for chunk_index, text in enumerate(texts):
                    normalized = text.strip()
                    if not normalized:
                        continue
                    chunks.append(
                        Chunk(
                            id=chunk_id(page, chunk_index, normalized),
                            document_id=document.id,
                            source_path=document.source_path,
                            filename=document.filename,
                            content_hash=document.content_hash,
                            viewer_page=page.viewer_page,
                            page_label=page.label,
                            chunk_index=chunk_index,
                            text=normalized,
                        )
                    )
            return tuple(chunks)
        except Exception as exc:
            raise SyncOperationError(
                SyncStage.CHUNKING,
                f"unable to split {document.filename}",
            ) from exc
