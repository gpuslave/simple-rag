from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

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


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reasoning_effort: Literal["low", "high", "max"]
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0)
    provider: ProviderRouting | None = None


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    size: int = Field(default=800, gt=0)
    overlap: int = Field(default=100, ge=0)

    @model_validator(mode="after")
    def overlap_is_smaller_than_size(self) -> ChunkingConfig:
        if self.overlap >= self.size:
            raise ValueError("overlap must be smaller than size")
        return self


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_limit: int = Field(default=30, gt=0)
    max_chunks_per_page: int = Field(default=3, gt=0)
    evidence_limit: int = Field(default=15, gt=0)

    @model_validator(mode="after")
    def limits_are_consistent(self) -> RetrievalConfig:
        if self.evidence_limit > self.candidate_limit:
            raise ValueError("evidence_limit cannot exceed candidate_limit")
        if self.max_chunks_per_page > self.evidence_limit:
            raise ValueError("max_chunks_per_page cannot exceed evidence_limit")
        return self


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    corpus_path: Path
    generation_model: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    gateway_base_url: str = DEFAULT_GATEWAY_URL
    state_path: Path = Path(".rag-state")
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig
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
    chunking = _table(data, "chunking")
    retrieval = _table(data, "retrieval")
    generation = _table(data, "generation")

    allowed = {
        "corpus",
        "models",
        "gateway",
        "state",
        "chunking",
        "retrieval",
        "generation",
    }
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
        "chunking": chunking,
        "retrieval": retrieval,
        "generation": generation,
        "api_key": api_key,
    }
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors(include_url=False, include_input=False)[0]
        location = ".".join(str(part) for part in first["loc"])
        raise ConfigurationError(f"invalid setting {location}: {first['msg']}") from exc
