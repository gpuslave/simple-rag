from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from rag.application.doctor import LocalDiagnosticError, run_doctor
from rag.application.sync import SyncCorpus, SyncExecutionError
from rag.config import AppConfig, ConfigurationError, load_config
from rag.domain import SyncReport
from rag.infrastructure.chunking import LangChainPageChunker
from rag.infrastructure.embeddings import RouterAIEmbeddings
from rag.infrastructure.filesystem import LocalCorpusSource
from rag.infrastructure.pdf import PyMuPdfExtractor
from rag.infrastructure.qdrant import LangChainQdrantChunkIndex
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


def _build_synchronizer(config: AppConfig) -> SyncCorpus:
    return SyncCorpus(
        source=LocalCorpusSource(config.corpus_path),
        extractor=PyMuPdfExtractor(),
        chunker=LangChainPageChunker(
            chunk_size=config.chunking.size,
            chunk_overlap=config.chunking.overlap,
        ),
        index=LangChainQdrantChunkIndex(
            config.state_path,
            RouterAIEmbeddings(config),
        ),
    )


def _render_sync_report(report: SyncReport) -> None:
    typer.echo(f"mode: {'dry-run' if report.dry_run else 'apply'}")
    typer.echo(f"indexed documents: {report.indexed_documents}")
    typer.echo(f"unchanged documents: {report.unchanged_documents}")
    typer.echo(f"indexed pages: {report.indexed_pages}")
    typer.echo(f"indexed chunks: {report.indexed_chunks}")
    typer.echo(f"skipped pages: {len(report.skipped_pages)}")
    typer.echo(f"failures: {len(report.failures)}")
    for change in report.changes:
        typer.echo(f"{change.outcome}: {change.action}: {change.source_path}")
    for warning in report.warnings:
        typer.echo(f"warning: {warning}", err=True)
    for failure in report.failures:
        typer.echo(
            f"error: {failure.source_path}: {failure.stage}: {failure.message}",
            err=True,
        )


@app.command("sync")
def sync_command(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Path to the TOML configuration file."),
    ] = Path("rag.toml"),
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the synchronization report as JSON."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Preview changes without modifying Qdrant."),
    ] = False,
) -> None:
    """Synchronize PDFs from the authoritative corpus into local Qdrant."""
    try:
        config = load_config(config_path)
        report = _build_synchronizer(config).synchronize(dry_run=dry_run)
    except (ConfigurationError, SyncExecutionError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from None

    if json_output:
        typer.echo(json.dumps(report.model_dump(mode="json"), ensure_ascii=False))
    else:
        _render_sync_report(report)
    if report.failures:
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
