from __future__ import annotations

import os
from pathlib import Path

import pytest

from rag.application.sync import SyncCorpus
from rag.config import AppConfig, GenerationConfig
from rag.infrastructure.chunking import LangChainPageChunker
from rag.infrastructure.embeddings import RouterAIEmbeddings
from rag.infrastructure.filesystem import LocalCorpusSource
from rag.infrastructure.pdf import PyMuPdfExtractor
from rag.infrastructure.qdrant import LangChainQdrantChunkIndex


@pytest.mark.live
def test_live_routerai_sync_persists_sample_and_skips_second_embedding(
    tmp_path: Path,
) -> None:
    api_key = os.environ.get("ROUTERAI_API_KEY")
    if not api_key:
        pytest.skip("ROUTERAI_API_KEY is not configured")

    config = AppConfig(
        corpus_path=Path("examples"),
        generation_model="z-ai/glm-5.3-flash",
        embedding_model="baai/bge-m3",
        state_path=tmp_path / "state",
        generation=GenerationConfig(
            reasoning_effort="low", temperature=0.2, max_tokens=8192
        ),
        api_key=api_key,
    )
    synchronizer = SyncCorpus(
        source=LocalCorpusSource(config.corpus_path),
        extractor=PyMuPdfExtractor(),
        chunker=LangChainPageChunker(
            chunk_size=config.chunking.size,
            chunk_overlap=config.chunking.overlap,
        ),
        index=LangChainQdrantChunkIndex(
            config.state_path,
            RouterAIEmbeddings(config),
        ),
    )

    first = synchronizer.synchronize()
    second = synchronizer.synchronize()

    assert first.failures == ()
    assert first.indexed_documents == 1
    assert first.indexed_pages == 4
    assert first.indexed_chunks > 0
    assert first.skipped_pages == ()
    assert second.unchanged_documents == 1
    assert second.indexed_documents == 0
