from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from rag.config import AppConfig


class EmbeddingRequestError(RuntimeError):
    """A safe-to-display embedding request failure."""


class RouterAIEmbeddings(Embeddings):
    def __init__(self, config: AppConfig) -> None:
        self._delegate = OpenAIEmbeddings(
            model=config.embedding_model,
            base_url=config.gateway_base_url,
            api_key=SecretStr(config.api_key),
            check_embedding_ctx_length=False,
            model_kwargs={"encoding_format": "float"},
            timeout=30.0,
            max_retries=2,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._delegate.embed_documents(texts)
        except Exception as exc:
            raise EmbeddingRequestError("RouterAI document embedding failed") from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._delegate.embed_query(text)
        except Exception as exc:
            raise EmbeddingRequestError("RouterAI query embedding failed") from exc
