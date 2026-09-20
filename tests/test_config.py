from pathlib import Path

import pytest

from rag.config import ConfigurationError, DEFAULT_GATEWAY_URL, load_config


def write_config(path: Path, extra: str = "") -> None:
    path.write_text(
        """
[corpus]
path = "documents"
[models]
generation = "vendor/generation"
embedding = "vendor/embedding"
""" + extra,
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

    with pytest.raises(ConfigurationError, match="ROUTERAI_API_KEY is required") as error:
        load_config(path, {"UNRELATED_SECRET": "must-not-leak"})

    assert "must-not-leak" not in str(error.value)


def test_missing_model_is_actionable(tmp_path: Path) -> None:
    path = tmp_path / "rag.toml"
    path.write_text('[corpus]\npath = "documents"\n', encoding="utf-8")

    with pytest.raises(ConfigurationError, match="generation_model"):
        load_config(path, {"ROUTERAI_API_KEY": "secret"})

