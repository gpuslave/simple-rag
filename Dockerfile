# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.17@sha256:10787c682e4184e4f290de1171fd4703dc63de99221f10fe1c99002ce7fa9acc

FROM ${UV_IMAGE} AS uv
FROM ${PYTHON_IMAGE} AS builder

COPY --from=uv /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    TIKTOKEN_CACHE_DIR=/app/tiktoken-cache

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

RUN /app/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

FROM ${PYTHON_IMAGE} AS runtime

ARG VERSION=0.1.0
ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="Page-cited RAG CLI" \
      org.opencontainers.image.description="Local PDF RAG CLI with validated page-level citations" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/gpuslave/simple-rag" \
      org.opencontainers.image.licenses="MIT"

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TIKTOKEN_CACHE_DIR=/app/tiktoken-cache

RUN groupadd --gid 10001 rag \
    && useradd --uid 10001 --gid rag --no-create-home --home-dir /nonexistent rag \
    && mkdir -p /app /data/corpus /data/state \
    && chown -R rag:rag /app /data

WORKDIR /app

COPY --from=builder --chown=rag:rag /app/.venv /app/.venv
COPY --from=builder --chown=rag:rag /app/tiktoken-cache /app/tiktoken-cache
COPY --chown=rag:rag container/rag.toml /app/rag.toml

USER rag:rag

VOLUME ["/data/state"]

ENTRYPOINT ["rag"]
CMD ["--help"]
