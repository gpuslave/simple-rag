set dotenv-filename := ".env.example"
set shell := ["bash", "-euo", "pipefail", "-c"]

# List available commands when `just` is run without arguments.
default:
    @just --list

# Install the exact locked dependencies.
setup:
    uv sync --locked

# Run the offline test suite.
test:
    uv run --locked pytest

# Run live local and RouterAI readiness checks.
doctor config="rag.toml.example":
    uv run --locked rag doctor --config "{{ config }}"

# Verify the lockfile and run all offline checks.
check:
    uv lock --check
    uv run --locked pytest
    uv run --locked rag --help > /dev/null

# Build the source distribution and wheel.
build:
    uv build
