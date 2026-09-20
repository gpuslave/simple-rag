from pathlib import Path

from rag.config import AppConfig
from rag.infrastructure.embeddings import RouterAIEmbeddings


def test_routerai_embeddings_send_raw_strings() -> None:
    embeddings = RouterAIEmbeddings(
        AppConfig(
            corpus_path=Path("examples"),
            generation_model="vendor/generation",
            embedding_model="vendor/embedding",
            api_key="secret",
        )
    )

    assert embeddings._delegate.check_embedding_ctx_length is False
    assert embeddings._delegate.model_kwargs["encoding_format"] == "float"
