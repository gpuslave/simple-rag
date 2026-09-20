from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rag.config import AppConfig, ProviderRouting
from rag.domain import AnswerStatus, Chunk, Claim, GeneratedAnswer, RetrievedChunk
from rag.infrastructure.generation import RouterAIAnswerGenerator
from rag.ports import AnswerGenerationError


def evidence() -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id="chunk-1",
            document_id="document-1",
            source_path=Path("/corpus/lesson.pdf"),
            filename="lesson.pdf",
            content_hash="hash",
            viewer_page=2,
            page_label="ii",
            chunk_index=0,
            text="Untrusted English evidence",
        ),
        score=0.75,
        source_id="S1",
    )


def config() -> AppConfig:
    return AppConfig(
        corpus_path=Path("corpus"),
        generation_model="vendor/generation",
        embedding_model="vendor/embedding",
        api_key="secret",
        provider_routing=ProviderRouting(only=("provider-a",), allow_fallbacks=False),
    )


class FakeRunnable:
    def __init__(self, result: GeneratedAnswer | Exception) -> None:
        self.result = result
        self.input: Any = None

    def invoke(self, value: Any) -> GeneratedAnswer:
        self.input = value
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_generation_uses_strict_schema_and_sends_only_question_and_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = GeneratedAnswer(
        status=AnswerStatus.ANSWERED,
        claims=(Claim(text="Факт.", source_ids=("S1",)),),
    )
    runnable = FakeRunnable(result)
    captured: dict[str, Any] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured["kwargs"] = kwargs

        def with_structured_output(self, schema: Any, **kwargs: Any) -> FakeRunnable:
            captured["schema"] = schema
            captured["structured_kwargs"] = kwargs
            return runnable

    monkeypatch.setattr("rag.infrastructure.generation.ChatOpenAI", FakeChatOpenAI)

    actual = RouterAIAnswerGenerator(config()).generate("What?", (evidence(),))

    assert actual == result
    assert captured["structured_kwargs"] == {"method": "json_schema", "strict": True}
    assert captured["kwargs"]["extra_body"] == {
        "provider": {"only": ["provider-a"], "allow_fallbacks": False}
    }
    messages = runnable.input
    assert "outside knowledge" in messages[0][1]
    request = json.loads(messages[1][1])
    assert request == {
        "question": "What?",
        "evidence": [
            {
                "source_id": "S1",
                "filename": "lesson.pdf",
                "viewer_page": 2,
                "page_label": "ii",
                "score": 0.75,
                "text": "Untrusted English evidence",
            }
        ],
    }


def test_generation_wraps_malformed_model_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runnable = FakeRunnable(ValueError("malformed secret response"))

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def with_structured_output(self, schema: Any, **kwargs: Any) -> FakeRunnable:
            return runnable

    monkeypatch.setattr("rag.infrastructure.generation.ChatOpenAI", FakeChatOpenAI)
    generator = RouterAIAnswerGenerator(config())

    with pytest.raises(AnswerGenerationError, match="valid structured answer") as error:
        generator.generate("What?", (evidence(),))

    assert "malformed secret response" not in str(error.value)
