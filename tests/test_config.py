from pathlib import Path

import pytest

from rag.config import DEFAULT_GATEWAY_URL, ConfigurationError, load_config


def write_config(path: Path, extra: str = "") -> None:
    path.write_text(
        """
[corpus]
path = "documents"
[models]
generation = "vendor/generation"
embedding = "vendor/embedding"
"""
        + extra,
        encoding="utf-8",
    )


def test_loads_required_settings_and_defaults(tmp_path: Path) -> None:
    path = tmp_path / "rag.toml"
    write_config(path)

    config = load_config(path, {"ROUTERAI_API_KEY": "secret"})

    assert config.corpus_path == Path("documents")
    assert config.gateway_base_url == DEFAULT_GATEWAY_URL
    assert config.api_key == "secret"
    assert config.provider_routing is None
    assert config.chunking.size == 800
    assert config.chunking.overlap == 100
    assert config.retrieval.candidate_limit == 30
    assert config.retrieval.max_chunks_per_page == 3
    assert config.retrieval.evidence_limit == 15


def test_loads_optional_provider_routing(tmp_path: Path) -> None:
    path = tmp_path / "rag.toml"
    write_config(
        path,
        """
[generation.provider]
only = ["provider-a"]
allow_fallbacks = false
country = "ru"
""",
    )

    config = load_config(path, {"ROUTERAI_API_KEY": "secret"})

    assert config.provider_routing is not None
    assert config.provider_routing.only == ("provider-a",)
    assert config.provider_routing.allow_fallbacks is False


def test_missing_secret_is_actionable_and_never_prints_other_environment(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rag.toml"
    write_config(path)

    with pytest.raises(
        ConfigurationError, match="ROUTERAI_API_KEY is required"
    ) as error:
        load_config(path, {"UNRELATED_SECRET": "must-not-leak"})

    assert "must-not-leak" not in str(error.value)


def test_missing_model_is_actionable(tmp_path: Path) -> None:
    path = tmp_path / "rag.toml"
    path.write_text('[corpus]\npath = "documents"\n', encoding="utf-8")

    with pytest.raises(ConfigurationError, match="generation_model"):
        load_config(path, {"ROUTERAI_API_KEY": "secret"})


def test_loads_chunking_settings(tmp_path: Path) -> None:
    path = tmp_path / "rag.toml"
    write_config(path, "\n[chunking]\nsize = 40\noverlap = 5\n")

    config = load_config(path, {"ROUTERAI_API_KEY": "secret"})

    assert config.chunking.size == 40
    assert config.chunking.overlap == 5


def test_rejects_chunk_overlap_equal_to_size(tmp_path: Path) -> None:
    path = tmp_path / "rag.toml"
    write_config(path, "\n[chunking]\nsize = 10\noverlap = 10\n")

    with pytest.raises(ConfigurationError, match="overlap must be smaller"):
        load_config(path, {"ROUTERAI_API_KEY": "secret"})


def test_loads_retrieval_settings(tmp_path: Path) -> None:
    path = tmp_path / "rag.toml"
    write_config(
        path,
        "\n[retrieval]\ncandidate_limit = 20\nmax_chunks_per_page = 2\n"
        "evidence_limit = 10\n",
    )

    config = load_config(path, {"ROUTERAI_API_KEY": "secret"})

    assert config.retrieval.candidate_limit == 20
    assert config.retrieval.max_chunks_per_page == 2
    assert config.retrieval.evidence_limit == 10


@pytest.mark.parametrize(
    "settings",
    [
        "candidate_limit = 2\nevidence_limit = 3",
        "max_chunks_per_page = 4\nevidence_limit = 3",
    ],
)
def test_rejects_inconsistent_retrieval_limits(tmp_path: Path, settings: str) -> None:
    path = tmp_path / "rag.toml"
    write_config(path, f"\n[retrieval]\n{settings}\n")

    with pytest.raises(ConfigurationError, match="cannot exceed"):
        load_config(path, {"ROUTERAI_API_KEY": "secret"})
