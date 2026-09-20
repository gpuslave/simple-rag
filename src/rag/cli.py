from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rag.application.doctor import LocalDiagnosticError, run_doctor
from rag.config import ConfigurationError, load_config
from rag.infrastructure.routerai import GatewayDiagnosticError, RouterAIDiagnostics

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Local PDF RAG with validated page-level citations.",
)


@app.callback()
def root() -> None:
    """Run a RAG operation."""


@app.command()
def doctor(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Path to the TOML configuration file."),
    ] = Path("rag.toml"),
) -> None:
    """Verify local paths, credentials, and configured model capabilities."""
    try:
        config = load_config(config_path)
        with RouterAIDiagnostics(config) as gateway:
            report = run_doctor(config, gateway)
    except (ConfigurationError, LocalDiagnosticError, GatewayDiagnosticError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from None

    typer.echo("ready")
    typer.echo(f"corpus: {report.corpus_path}")
    typer.echo(f"state: {report.state_path}")
    typer.echo(
        f"generation model: {report.generation_model} (strict structured output)"
    )
    typer.echo(
        f"embedding model: {report.embedding_model} ({report.vector_dimension} dimensions)"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
