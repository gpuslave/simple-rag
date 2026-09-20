from pathlib import Path

from typer.testing import CliRunner

from rag.cli import app
from rag.domain import SyncFailure, SyncReport, SyncStage

runner = CliRunner()


def test_cli_reports_missing_configuration() -> None:
    result = runner.invoke(app, ["doctor", "--config", "absent.toml"], env={})

    assert result.exit_code == 1
    assert "configuration file not found" in result.stderr


def test_cli_never_echoes_api_key(tmp_path: Path) -> None:
    config = tmp_path / "rag.toml"
    config.write_text(
        '[corpus]\npath = "missing"\n[models]\ngeneration = "a/b"\nembedding = "c/d"\n',
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["doctor", "--config", str(config)],
        env={"ROUTERAI_API_KEY": "highly-secret-value"},
    )

    assert result.exit_code == 1
    assert "highly-secret-value" not in result.output


def config_file(tmp_path: Path) -> Path:
    config = tmp_path / "rag.toml"
    config.write_text(
        '[corpus]\npath = "."\n[models]\ngeneration = "a/b"\nembedding = "c/d"\n',
        encoding="utf-8",
    )
    return config


class StubSynchronizer:
    def __init__(self, report: SyncReport) -> None:
        self.report = report
        self.dry_run = False

    def synchronize(self, *, dry_run: bool = False) -> SyncReport:
        self.dry_run = dry_run
        return self.report


def test_sync_json_output(monkeypatch: object, tmp_path: Path) -> None:
    config = config_file(tmp_path)
    report = SyncReport(indexed_documents=1, indexed_pages=2, indexed_chunks=3)
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "rag.cli._build_synchronizer", lambda _: StubSynchronizer(report)
    )

    result = runner.invoke(
        app,
        ["sync", "--config", str(config), "--json"],
        env={"ROUTERAI_API_KEY": "secret"},
    )

    assert result.exit_code == 0
    assert '"indexed_documents": 1' in result.stdout
    assert result.stderr == ""


def test_sync_failure_is_rendered_and_exits_nonzero(
    monkeypatch: object, tmp_path: Path
) -> None:
    config = config_file(tmp_path)
    report = SyncReport(
        failures=(
            SyncFailure(
                source_path=Path("broken.pdf"),
                stage=SyncStage.EXTRACTION,
                message="unable to extract broken.pdf",
            ),
        )
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "rag.cli._build_synchronizer", lambda _: StubSynchronizer(report)
    )

    result = runner.invoke(
        app,
        ["sync", "--config", str(config)],
        env={"ROUTERAI_API_KEY": "secret"},
    )

    assert result.exit_code == 1
    assert "failures: 1" in result.stdout
    assert "unable to extract broken.pdf" in result.stderr


def test_sync_dry_run_is_forwarded_and_rendered(
    monkeypatch: object, tmp_path: Path
) -> None:
    config = config_file(tmp_path)
    synchronizer = StubSynchronizer(SyncReport(dry_run=True))
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "rag.cli._build_synchronizer", lambda _: synchronizer
    )

    result = runner.invoke(
        app,
        ["sync", "--config", str(config), "--dry-run"],
        env={"ROUTERAI_API_KEY": "secret"},
    )

    assert result.exit_code == 0
    assert synchronizer.dry_run is True
    assert "mode: dry-run" in result.stdout
