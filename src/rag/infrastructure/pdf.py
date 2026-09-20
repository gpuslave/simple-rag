from __future__ import annotations

import pymupdf

from rag.domain import Document, Page, SyncStage
from rag.domain.identity import page_id
from rag.ports import SyncOperationError


class PyMuPdfExtractor:
    def extract(self, document: Document) -> tuple[Page, ...]:
        try:
            with pymupdf.open(document.source_path) as pdf:  # type: ignore[no-untyped-call]
                has_explicit_labels = bool(pdf.get_page_labels())
                pages = []
                for offset, pdf_page in enumerate(pdf):
                    viewer_page = offset + 1
                    raw_label = pdf_page.get_label() if has_explicit_labels else ""
                    label = raw_label.strip() or None
                    pages.append(
                        Page(
                            id=page_id(document, viewer_page),
                            document_id=document.id,
                            viewer_page=viewer_page,
                            label=label,
                            text=pdf_page.get_text("text", sort=True).strip(),
                        )
                    )
                return tuple(pages)
        except Exception as exc:
            raise SyncOperationError(
                SyncStage.EXTRACTION,
                f"unable to extract {document.filename}",
            ) from exc
