from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from rag.domain import (
    AnswerResult,
    AnswerStatus,
    Citation,
    GeneratedAnswer,
    RetrievedChunk,
)
from rag.ports import AnswerGenerator, Retriever


class AnswerValidationError(ValueError):
    """A safe-to-display invalid-answer failure."""


@dataclass(frozen=True)
class AnswerQuestion:
    retriever: Retriever
    generator: AnswerGenerator
    candidate_limit: int = 30
    max_chunks_per_page: int = 3
    evidence_limit: int = 15

    def answer(self, question: str) -> AnswerResult:
        normalized_question = question.strip()
        if not normalized_question:
            raise AnswerValidationError("question cannot be empty")

        candidates = self.retriever.retrieve(
            normalized_question,
            limit=self.candidate_limit,
        )
        evidence = self._select_evidence(candidates)
        if not evidence:
            return AnswerResult(status=AnswerStatus.INSUFFICIENT_EVIDENCE)

        generated = self.generator.generate(normalized_question, evidence)
        return self._validate(generated, evidence)

    def _select_evidence(
        self, candidates: Sequence[RetrievedChunk]
    ) -> tuple[RetrievedChunk, ...]:
        selected: list[RetrievedChunk] = []
        seen_chunks: set[str] = set()
        page_counts: defaultdict[tuple[str, int], int] = defaultdict(int)

        for candidate in candidates:
            chunk = candidate.chunk
            page_key = (chunk.document_id, chunk.viewer_page)
            if chunk.id in seen_chunks:
                continue
            if page_counts[page_key] >= self.max_chunks_per_page:
                continue
            selected.append(
                candidate.model_copy(update={"source_id": f"S{len(selected) + 1}"})
            )
            seen_chunks.add(chunk.id)
            page_counts[page_key] += 1
            if len(selected) >= self.evidence_limit:
                break

        return tuple(selected)

    @staticmethod
    def _validate(
        generated: GeneratedAnswer,
        evidence: tuple[RetrievedChunk, ...],
    ) -> AnswerResult:
        if generated.status is AnswerStatus.INSUFFICIENT_EVIDENCE:
            if generated.claims:
                raise AnswerValidationError(
                    "generation model returned claims with insufficient evidence"
                )
            return AnswerResult(status=AnswerStatus.INSUFFICIENT_EVIDENCE)

        if not generated.claims:
            raise AnswerValidationError("generation model returned no cited claims")

        available = {
            item.source_id: item for item in evidence if item.source_id is not None
        }
        citation_ids: list[str] = []
        for claim in generated.claims:
            if not claim.text.strip():
                raise AnswerValidationError("generation model returned an empty claim")
            if not claim.source_ids:
                raise AnswerValidationError(
                    "generation model returned a claim without citations"
                )
            if len(set(claim.source_ids)) != len(claim.source_ids):
                raise AnswerValidationError(
                    "generation model returned duplicate citations for a claim"
                )
            for source_id in claim.source_ids:
                if source_id not in available:
                    raise AnswerValidationError(
                        "generation model cited an unknown evidence source"
                    )
                if source_id not in citation_ids:
                    citation_ids.append(source_id)

        citations = tuple(
            Citation(
                source_id=source_id,
                source_path=available[source_id].chunk.source_path,
                filename=available[source_id].chunk.filename,
                viewer_page=available[source_id].chunk.viewer_page,
                page_label=available[source_id].chunk.page_label,
                score=available[source_id].score,
                excerpt=available[source_id].chunk.text,
            )
            for source_id in citation_ids
        )
        return AnswerResult(
            status=AnswerStatus.ANSWERED,
            claims=generated.claims,
            citations=citations,
        )
