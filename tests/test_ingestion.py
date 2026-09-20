from pathlib import Path

import pymupdf
import pytest

from rag.domain import Document, Page
from rag.domain.identity import chunk_id, document_id, page_id
from rag.infrastructure.chunking import LangChainPageChunker
from rag.infrastructure.filesystem import LocalCorpusSource
from rag.infrastructure.pdf import PyMuPdfExtractor
from rag.ports import SyncOperationError


def make_pdf(path: Path) -> None:
    pdf = pymupdf.open()  # type: ignore[no-untyped-call]
    first = pdf.new_page()
    first.insert_text((72, 72), "First page text")
    pdf.new_page()
    pdf.set_page_labels(  # type: ignore[no-untyped-call]
        [{"startpage": 0, "prefix": "A-", "style": "D", "firstpagenum": 1}]
    )
    pdf.save(path)  # type: ignore[no-untyped-call]
    pdf.close()  # type: ignore[no-untyped-call]


def test_discovers_pdfs_recursively_case_insensitively_without_symlinks(
    tmp_path: Path,
) -> None:
    upper = tmp_path / "B.PDF"
    lower = tmp_path / "a.pdf"
    upper.write_bytes(b"upper")
    lower.write_bytes(b"lower")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "nested.pdf").write_bytes(b"nested")
    (tmp_path / "linked.pdf").symlink_to(lower)
    linked_directory = tmp_path / "linked-directory"
    linked_directory.symlink_to(nested, target_is_directory=True)

    documents = LocalCorpusSource(tmp_path).discover()

    assert [document.filename for document in documents] == [
        "a.pdf",
        "B.PDF",
        "nested.pdf",
    ]
    assert documents[0].source_path == lower.resolve()
    assert documents[0].id == document_id(lower.resolve())
    assert documents[0].content_hash != documents[1].content_hash


def test_extracts_ordered_pages_labels_and_textless_pages(tmp_path: Path) -> None:
    source_path = tmp_path / "labels.pdf"
    make_pdf(source_path)
    document = LocalCorpusSource(tmp_path).discover()[0]

    pages = PyMuPdfExtractor().extract(document)

    assert len(pages) == 2
    assert pages[0].viewer_page == 1
    assert pages[0].label == "A-1"
    assert "First page text" in pages[0].text
    assert pages[1].viewer_page == 2
    assert pages[1].label == "A-2"
    assert pages[1].text == ""
    assert pages[0].id == page_id(document, 1)


def test_corrupt_pdf_reports_safe_extraction_failure(tmp_path: Path) -> None:
    source_path = tmp_path / "broken.pdf"
    source_path.write_bytes(b"not a PDF")
    document = LocalCorpusSource(tmp_path).discover()[0]

    with pytest.raises(SyncOperationError, match=r"unable to extract broken\.pdf"):
        PyMuPdfExtractor().extract(document)


def test_page_bounded_chunks_and_identities_are_deterministic(tmp_path: Path) -> None:
    source_path = (tmp_path / "one.pdf").resolve()
    document = Document(
        id=document_id(source_path),
        source_path=source_path,
        filename=source_path.name,
        content_hash="a" * 64,
    )
    pages = (
        Page(
            id=page_id(document, 1),
            document_id=document.id,
            viewer_page=1,
            text="alpha " * 30,
        ),
        Page(
            id=page_id(document, 2),
            document_id=document.id,
            viewer_page=2,
            text="beta " * 30,
        ),
    )
    chunker = LangChainPageChunker(chunk_size=12, chunk_overlap=3)

    first = chunker.split(document, pages)
    second = chunker.split(document, pages)

    assert len(first) > 2
    assert first == second
    assert {chunk.viewer_page for chunk in first} == {1, 2}
    assert all(not ("alpha" in chunk.text and "beta" in chunk.text) for chunk in first)
    assert first[0].id == chunk_id(pages[0], 0, first[0].text)

    changed = document.model_copy(update={"content_hash": "b" * 64})
    changed_page = pages[0].model_copy(update={"id": page_id(changed, 1)})
    changed_chunks = chunker.split(changed, [changed_page])
    assert changed_chunks[0].id != first[0].id
