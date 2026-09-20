from __future__ import annotations

import tempfile
from dataclasses import dataclass

from rag.config import AppConfig
from rag.ports import GatewayProbe, ModelGatewayDiagnostics


class LocalDiagnosticError(RuntimeError):
    pass


@dataclass(frozen=True)
class DoctorReport:
    corpus_path: str
    state_path: str
    generation_model: str
    embedding_model: str
    vector_dimension: int
    structured_output_endpoints: tuple[str, ...]
    reasoning_effort: str
    temperature: float
    max_tokens: int


def check_local_paths(config: AppConfig) -> None:
    if not config.corpus_path.is_dir():
        raise LocalDiagnosticError(
            f"corpus directory does not exist or is not a directory: {config.corpus_path}"
        )
    try:
        config.state_path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=config.state_path):
            pass
    except OSError as exc:
        raise LocalDiagnosticError(
            f"local state directory is not writable: {config.state_path}"
        ) from exc


def run_doctor(config: AppConfig, gateway: ModelGatewayDiagnostics) -> DoctorReport:
    check_local_paths(config)
    probe: GatewayProbe = gateway.probe()
    return DoctorReport(
        corpus_path=str(config.corpus_path.resolve()),
        state_path=str(config.state_path.resolve()),
        generation_model=config.generation_model,
        embedding_model=config.embedding_model,
        vector_dimension=probe.vector_dimension,
        structured_output_endpoints=probe.structured_output_endpoints,
        reasoning_effort=config.generation.reasoning_effort,
        temperature=config.generation.temperature,
        max_tokens=config.generation.max_tokens,
    )
