from __future__ import annotations

import hashlib
from pathlib import Path

from rag.domain import Document
from rag.domain.identity import document_id
from rag.ports import CorpusSourceError


class LocalCorpusSource:
    def __init__(self, corpus_path: Path) -> None:
        self._corpus_path = corpus_path

    def discover(self) -> tuple[Document, ...]:
        try:
            candidates = sorted(
                (
                    path
                    for path in self._corpus_path.iterdir()
                    if path.is_file()
                    and not path.is_symlink()
                    and path.suffix.lower() == ".pdf"
                ),
                key=lambda path: path.name.casefold(),
            )
            documents = []
            for candidate in candidates:
                canonical = candidate.resolve(strict=True)
                documents.append(
                    Document(
                        id=document_id(canonical),
                        source_path=canonical,
                        filename=canonical.name,
                        content_hash=_content_sha256(canonical),
                    )
                )
            return tuple(documents)
        except OSError as exc:
            raise CorpusSourceError("cannot read the configured corpus") from exc


def _content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
