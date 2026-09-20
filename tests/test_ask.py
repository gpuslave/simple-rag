from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from rag.application.ask import AnswerQuestion, AnswerValidationError
from rag.domain import (
    AnswerStatus,
    Chunk,
    Claim,
    GeneratedAnswer,
    RetrievedChunk,
)


def retrieved(
    chunk_id: str,
    *,
    page: int,
    text: str,
    score: float = 1.0,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id=chunk_id,
            document_id="document-1",
            source_path=Path("/corpus/lesson.pdf"),
            filename="lesson.pdf",
            content_hash="hash",
            viewer_page=page,
            page_label="A" if page == 1 else None,
            chunk_index=0,
            text=text,
        ),
        score=score,
    )


class StubRetriever:
    def __init__(self, results: Sequence[RetrievedChunk]) -> None:
        self.results = results
        self.query = ""
        self.limit = 0

    def retrieve(self, query: str, *, limit: int) -> Sequence[RetrievedChunk]:
        self.query = query
        self.limit = limit
        return self.results


class StubGenerator:
    def __init__(self, result: GeneratedAnswer) -> None:
        self.result = result
        self.question = ""
        self.evidence: Sequence[RetrievedChunk] = ()

    def generate(
        self, question: str, evidence: Sequence[RetrievedChunk]
    ) -> GeneratedAnswer:
        self.question = question
        self.evidence = evidence
        return self.result


def test_selects_ranked_unique_evidence_with_page_cap_and_stable_ids() -> None:
    first = retrieved("a", page=1, text="English evidence", score=0.9)
    candidates = (
        first,
        retrieved("b", page=1, text="Русское доказательство", score=0.8),
        first,
        retrieved("c", page=1, text="capped", score=0.7),
        retrieved("d", page=2, text="another page", score=0.6),
    )
    retriever = StubRetriever(candidates)
    generator = StubGenerator(
        GeneratedAnswer(
            status=AnswerStatus.ANSWERED,
            claims=(Claim(text="Подтверждённый ответ.", source_ids=("S1", "S3")),),
        )
    )
    answerer = AnswerQuestion(
        retriever,
        generator,
        candidate_limit=30,
        max_chunks_per_page=2,
        evidence_limit=3,
    )

    result = answerer.answer("  question  ")

    assert retriever.query == "question"
    assert retriever.limit == 30
    assert [item.chunk.id for item in generator.evidence] == ["a", "b", "d"]
    assert [item.source_id for item in generator.evidence] == ["S1", "S2", "S3"]
    assert [citation.source_id for citation in result.citations] == ["S1", "S3"]
    assert result.citations[0].page_label == "A"
    assert result.citations[1].viewer_page == 2


def test_empty_retrieval_refuses_without_calling_generation() -> None:
    retriever = StubRetriever(())
    generator = StubGenerator(GeneratedAnswer(status=AnswerStatus.ANSWERED, claims=()))

    result = AnswerQuestion(retriever, generator).answer("question")

    assert result.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert result.claims == ()
    assert generator.question == ""


@pytest.mark.parametrize(
    ("generated", "message"),
    [
        (
            GeneratedAnswer(status=AnswerStatus.ANSWERED, claims=()),
            "no cited claims",
        ),
        (
            GeneratedAnswer(
                status=AnswerStatus.ANSWERED,
                claims=(Claim(text=" ", source_ids=("S1",)),),
            ),
            "empty claim",
        ),
        (
            GeneratedAnswer(
                status=AnswerStatus.ANSWERED,
                claims=(Claim(text="Факт", source_ids=()),),
            ),
            "without citations",
        ),
        (
            GeneratedAnswer(
                status=AnswerStatus.ANSWERED,
                claims=(Claim(text="Факт", source_ids=("S1", "S1")),),
            ),
            "duplicate citations",
        ),
        (
            GeneratedAnswer(
                status=AnswerStatus.ANSWERED,
                claims=(Claim(text="Факт", source_ids=("S9",)),),
            ),
            "unknown evidence",
        ),
        (
            GeneratedAnswer(
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                claims=(Claim(text="Факт", source_ids=("S1",)),),
            ),
            "claims with insufficient",
        ),
    ],
)
def test_rejects_unvalidated_generated_answers(
    generated: GeneratedAnswer, message: str
) -> None:
    answerer = AnswerQuestion(
        StubRetriever((retrieved("a", page=1, text="evidence"),)),
        StubGenerator(generated),
    )

    with pytest.raises(AnswerValidationError, match=message):
        answerer.answer("question")


def test_rejects_empty_question_before_retrieval() -> None:
    answerer = AnswerQuestion(
        StubRetriever(()),
        StubGenerator(GeneratedAnswer(status=AnswerStatus.ANSWERED, claims=())),
    )

    with pytest.raises(AnswerValidationError, match="question cannot be empty"):
        answerer.answer("   ")
