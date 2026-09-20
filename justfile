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
    uv run --locked pytest -m "not live"

# Run opt-in tests that call RouterAI and require local credentials.
test-live:
    uv run --locked pytest -m "live"

# Run offline tests with coverage and enforce the minimum threshold.
coverage:
    uv run --locked pytest -m "not live" --cov=rag --cov-report=term-missing --cov-report=xml:coverage.xml --cov-fail-under=85

# Check Python lint rules and formatting without changing files.
lint:
    uv run --locked ruff check src tests
    uv run --locked ruff format --check src tests

# Apply safe lint fixes and format Python files.
format:
    uv run --locked ruff check --fix src tests
    uv run --locked ruff format src tests

# Run strict static type checking.
typecheck:
    uv run --locked mypy src tests

# Run live local and RouterAI readiness checks.
doctor config="rag.toml.example":
    uv run --locked rag doctor --config "{{ config }}"

# Synchronize the authoritative PDF corpus into local Qdrant.
sync config="rag.toml.example" *args:
    uv run --locked rag sync --config "{{ config }}" {{ args }}

# Verify the lockfile and run all offline checks.
check:
    uv lock --check
    uv run --locked ruff check src tests
    uv run --locked ruff format --check src tests
    uv run --locked mypy src tests
    uv run --locked pytest -m "not live" --cov=rag --cov-report=term-missing --cov-report=xml:coverage.xml --cov-fail-under=85
    uv run --locked rag --help > /dev/null

# Build the source distribution and wheel.
build:
    uv build
