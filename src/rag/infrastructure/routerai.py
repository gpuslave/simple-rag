from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from urllib.parse import quote

import httpx

from rag.config import AppConfig, ProviderRouting
from rag.ports import GatewayProbe


class GatewayDiagnosticError(RuntimeError):
    """A safe-to-display gateway diagnostic failure."""


class RouterAIDiagnostics:
    def __init__(
        self, config: AppConfig, *, transport: httpx.BaseTransport | None = None
    ):
        self._config = config
        self._client = httpx.Client(
            base_url=config.gateway_base_url + "/",
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=20.0,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> RouterAIDiagnostics:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def probe(self) -> GatewayProbe:
        endpoints = self._model_endpoints(self._config.generation_model)
        capable = self._structured_output_endpoints(
            endpoints, self._config.provider_routing
        )
        if not capable:
            raise GatewayDiagnosticError(
                "generation model has no eligible endpoint advertising strict structured outputs"
            )
        dimension = self._embedding_dimension(self._config.embedding_model)
        return GatewayProbe(
            vector_dimension=dimension,
            structured_output_endpoints=tuple(capable),
        )

    def _model_endpoints(self, model: str) -> list[dict[str, Any]]:
        encoded_model = "/".join(quote(part, safe="") for part in model.split("/"))
        response = self._request("GET", f"models/{encoded_model}/endpoints")
        payload = self._json(response, "model capability response")
        endpoints = payload.get("data", {}).get("endpoints")
        if not isinstance(endpoints, list):
            raise GatewayDiagnosticError(
                "model capability response has no endpoint list"
            )
        return [item for item in endpoints if isinstance(item, dict)]

    def _embedding_dimension(self, model: str) -> int:
        response = self._request(
            "POST", "embeddings", json={"model": model, "input": ["dimension probe"]}
        )
        payload = self._json(response, "embedding response")
        try:
            vector = payload["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GatewayDiagnosticError("embedding response has no vector") from exc
        if not isinstance(vector, list) or not vector:
            raise GatewayDiagnosticError("embedding response contains an empty vector")
        if not all(isinstance(value, (int, float)) for value in vector):
            raise GatewayDiagnosticError(
                "embedding response contains a non-numeric vector"
            )
        return len(vector)

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._client.request(method, path, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in (401, 403):
                message = "RouterAI rejected ROUTERAI_API_KEY"
            elif status == 404:
                message = "configured RouterAI model or endpoint was not found"
            elif status == 429:
                message = "RouterAI rate limit prevented diagnostics"
            else:
                message = f"RouterAI diagnostics failed with HTTP {status}"
            raise GatewayDiagnosticError(message) from exc
        except httpx.RequestError as exc:
            raise GatewayDiagnosticError(
                f"cannot connect to Model Gateway: {exc.__class__.__name__}"
            ) from exc

    @staticmethod
    def _json(response: httpx.Response, label: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise GatewayDiagnosticError(f"{label} is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise GatewayDiagnosticError(f"{label} is not a JSON object")
        return payload

    @staticmethod
    def _structured_output_endpoints(
        endpoints: Sequence[dict[str, Any]], routing: ProviderRouting | None
    ) -> list[str]:
        eligible = list(endpoints)
        if routing:
            if routing.only:
                allowed = set(routing.only)
                eligible = [item for item in eligible if item.get("tag") in allowed]
            if routing.ignore:
                ignored = set(routing.ignore)
                eligible = [item for item in eligible if item.get("tag") not in ignored]
            if routing.country:
                country = routing.country.lower()
                eligible = [
                    item
                    for item in eligible
                    if str(item.get("country", "")).lower() == country
                ]

        capable: list[str] = []
        for item in eligible:
            parameters = item.get("supported_parameters", [])
            if isinstance(parameters, list) and "structured_outputs" in parameters:
                capable.append(str(item.get("tag") or item.get("name") or "unknown"))
        return capable
