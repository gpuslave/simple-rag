from pathlib import Path

import httpx
import pytest

from rag.application.doctor import LocalDiagnosticError, check_local_paths, run_doctor
from rag.config import AppConfig, ProviderRouting
from rag.infrastructure.routerai import GatewayDiagnosticError, RouterAIDiagnostics
from rag.ports import GatewayProbe


def config(tmp_path: Path, routing: ProviderRouting | None = None) -> AppConfig:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    return AppConfig(
        corpus_path=corpus,
        generation_model="vendor/generator",
        embedding_model="vendor/embedder",
        state_path=tmp_path / "state",
        provider_routing=routing,
        api_key="secret",
    )


def successful_transport(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/endpoints"):
        return httpx.Response(
            200,
            json={
                "data": {
                    "endpoints": [
                        {
                            "tag": "good",
                            "country": "ru",
                            "supported_parameters": ["structured_outputs"],
                        }
                    ]
                }
            },
        )
    if request.url.path.endswith("/embeddings"):
        return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2, 0.3]}]})
    return httpx.Response(404)


def test_application_diagnostics_depend_on_port(tmp_path: Path) -> None:
    class FakeGatewayDiagnostics:
        def probe(self) -> GatewayProbe:
            return GatewayProbe(
                vector_dimension=7,
                structured_output_endpoints=("fake",),
            )

    settings = config(tmp_path)

    report = run_doctor(settings, FakeGatewayDiagnostics())

    assert report.vector_dimension == 7
    assert report.structured_output_endpoints == ("fake",)


def test_successful_mocked_diagnostics(tmp_path: Path) -> None:
    settings = config(tmp_path)
    with RouterAIDiagnostics(
        settings, transport=httpx.MockTransport(successful_transport)
    ) as gateway:
        report = run_doctor(settings, gateway)

    assert report.vector_dimension == 3
    assert report.structured_output_endpoints == ("good",)
    assert settings.state_path.is_dir()


def test_rejects_generation_model_without_strict_output(tmp_path: Path) -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": {"endpoints": [{"tag": "weak", "supported_parameters": []}]}},
        )

    settings = config(tmp_path)
    with RouterAIDiagnostics(
        settings, transport=httpx.MockTransport(transport)
    ) as gateway:
        with pytest.raises(GatewayDiagnosticError, match="strict structured outputs"):
            gateway.probe()


def test_routing_filters_capability_endpoints(tmp_path: Path) -> None:
    routing = ProviderRouting(only=("blocked",), allow_fallbacks=False)
    settings = config(tmp_path, routing)
    with RouterAIDiagnostics(
        settings, transport=httpx.MockTransport(successful_transport)
    ) as gateway:
        with pytest.raises(GatewayDiagnosticError, match="strict structured outputs"):
            gateway.probe()


def test_capability_failure_is_actionable(tmp_path: Path) -> None:
    settings = config(tmp_path)
    with RouterAIDiagnostics(
        settings,
        transport=httpx.MockTransport(lambda _: httpx.Response(503)),
    ) as gateway:
        with pytest.raises(GatewayDiagnosticError, match="HTTP 503"):
            gateway.probe()


def test_invalid_model_is_actionable(tmp_path: Path) -> None:
    settings = config(tmp_path)
    with RouterAIDiagnostics(
        settings,
        transport=httpx.MockTransport(lambda _: httpx.Response(404)),
    ) as gateway:
        with pytest.raises(GatewayDiagnosticError, match="model or endpoint was not found"):
            gateway.probe()


def test_invalid_api_key_does_not_leak_secret(tmp_path: Path) -> None:
    settings = config(tmp_path)
    with RouterAIDiagnostics(
        settings,
        transport=httpx.MockTransport(lambda _: httpx.Response(401)),
    ) as gateway:
        with pytest.raises(GatewayDiagnosticError, match="rejected ROUTERAI_API_KEY") as error:
            gateway.probe()
    assert "secret" not in str(error.value)


def test_missing_corpus_fails_before_remote_probe(tmp_path: Path) -> None:
    settings = AppConfig(
        corpus_path=tmp_path / "missing",
        generation_model="vendor/generator",
        embedding_model="vendor/embedder",
        state_path=tmp_path / "state",
        api_key="secret",
    )
    with pytest.raises(LocalDiagnosticError, match="corpus directory"):
        check_local_paths(settings)
