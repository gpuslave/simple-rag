from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from rag.domain.models import Document, Page


def document_id(source_path: Path) -> str:
    return str(uuid5(NAMESPACE_URL, source_path.as_posix()))


def page_id(document: Document, viewer_page: int) -> str:
    return str(uuid5(UUID(document.id), f"{document.content_hash}:page:{viewer_page}"))


def chunk_id(page: Page, chunk_index: int, text: str) -> str:
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return str(uuid5(UUID(page.id), f"chunk:{chunk_index}:{text_hash}"))
