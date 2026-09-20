from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DEFAULT_GATEWAY_URL = "https://routerai.ru/api/v1"


class ConfigurationError(ValueError):
    """A safe-to-display configuration error."""


class ProviderRouting(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    order: tuple[str, ...] | None = None
    only: tuple[str, ...] | None = None
    ignore: tuple[str, ...] | None = None
    allow_fallbacks: bool = True
    country: str | None = Field(default=None, pattern=r"^[a-zA-Z]{2}$")

    @field_validator("order", "only", "ignore")
    @classmethod
    def non_empty_provider_lists(
        cls, value: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        if value is not None and not value:
            raise ValueError("provider lists cannot be empty")
        return value


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    corpus_path: Path
    generation_model: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    gateway_base_url: str = DEFAULT_GATEWAY_URL
    state_path: Path = Path(".rag-state")
    provider_routing: ProviderRouting | None = None
    api_key: str = Field(repr=False, min_length=1)

    @field_validator("gateway_base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        if not value.startswith(("https://", "http://")):
            raise ValueError("must start with http:// or https://")
        return value.rstrip("/")


def _table(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigurationError(f"[{name}] must be a TOML table")
    return value


def load_config(path: Path, environ: dict[str, str] | None = None) -> AppConfig:
    env = os.environ if environ is None else environ
    try:
        with path.open("rb") as config_file:
            data = tomllib.load(config_file)
    except FileNotFoundError as exc:
        raise ConfigurationError(
            f"configuration file not found: {path}; copy rag.toml.example to rag.toml"
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(f"invalid TOML in {path}: {exc}") from exc

    corpus = _table(data, "corpus")
    models = _table(data, "models")
    gateway = _table(data, "gateway")
    state = _table(data, "state")
    generation = _table(data, "generation")

    allowed = {"corpus", "models", "gateway", "state", "generation"}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigurationError(f"unknown top-level setting: {unknown[0]}")

    api_key = env.get("ROUTERAI_API_KEY")
    if not api_key:
        raise ConfigurationError(
            "ROUTERAI_API_KEY is required; export it in the environment"
        )

    raw = {
        "corpus_path": corpus.get("path"),
        "generation_model": models.get("generation"),
        "embedding_model": models.get("embedding"),
        "gateway_base_url": gateway.get("base_url", DEFAULT_GATEWAY_URL),
        "state_path": state.get("path", ".rag-state"),
        "provider_routing": generation.get("provider"),
        "api_key": api_key,
    }
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors(include_url=False, include_input=False)[0]
        location = ".".join(str(part) for part in first["loc"])
        raise ConfigurationError(f"invalid setting {location}: {first['msg']}") from exc

