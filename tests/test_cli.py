from pathlib import Path

from typer.testing import CliRunner

from rag.application.ask import AnswerValidationError
from rag.cli import app
from rag.domain import (
    AnswerResult,
    AnswerStatus,
    Citation,
    Claim,
    SyncFailure,
    SyncReport,
    SyncStage,
)

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


class StubAnswerer:
    def __init__(self, result: AnswerResult | Exception) -> None:
        self.result = result
        self.question = ""

    def answer(self, question: str) -> AnswerResult:
        self.question = question
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


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


def cited_answer() -> AnswerResult:
    return AnswerResult(
        status=AnswerStatus.ANSWERED,
        claims=(Claim(text="Первый факт.", source_ids=("S1", "S2")),),
        citations=(
            Citation(
                source_id="S1",
                source_path=Path("/corpus/lesson.pdf"),
                filename="lesson.pdf",
                viewer_page=2,
                page_label="ii",
                score=0.9,
                excerpt="first excerpt",
            ),
            Citation(
                source_id="S2",
                source_path=Path("/corpus/appendix.pdf"),
                filename="appendix.pdf",
                viewer_page=4,
                score=0.8,
                excerpt="second excerpt",
            ),
        ),
    )


def test_ask_renders_claim_level_page_citations(
    monkeypatch: object, tmp_path: Path
) -> None:
    config = config_file(tmp_path)
    answerer = StubAnswerer(cited_answer())
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "rag.cli._build_answerer", lambda _: answerer
    )

    result = runner.invoke(
        app,
        ["ask", "What?", "--config", str(config)],
        env={"ROUTERAI_API_KEY": "secret"},
    )

    assert result.exit_code == 0
    assert answerer.question == "What?"
    assert result.stdout == (
        "Первый факт. [lesson.pdf, PDF p. 2, label ii] [appendix.pdf, PDF p. 4]\n"
    )
    assert result.stderr == ""


def test_ask_json_contains_citations_and_models(
    monkeypatch: object, tmp_path: Path
) -> None:
    config = config_file(tmp_path)
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "rag.cli._build_answerer", lambda _: StubAnswerer(cited_answer())
    )

    result = runner.invoke(
        app,
        ["ask", "What?", "--config", str(config), "--json"],
        env={"ROUTERAI_API_KEY": "secret"},
    )

    assert result.exit_code == 0
    assert '"source_path": "/corpus/lesson.pdf"' in result.stdout
    assert '"viewer_page": 2' in result.stdout
    assert '"page_label": "ii"' in result.stdout
    assert '"score": 0.9' in result.stdout
    assert '"excerpt": "first excerpt"' in result.stdout
    assert '"generation": "a/b"' in result.stdout
    assert '"embedding": "c/d"' in result.stdout


def test_ask_renders_fixed_insufficient_evidence_message(
    monkeypatch: object, tmp_path: Path
) -> None:
    config = config_file(tmp_path)
    refusal = AnswerResult(status=AnswerStatus.INSUFFICIENT_EVIDENCE)
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "rag.cli._build_answerer", lambda _: StubAnswerer(refusal)
    )

    result = runner.invoke(
        app,
        ["ask", "What?", "--config", str(config)],
        env={"ROUTERAI_API_KEY": "secret"},
    )

    assert result.exit_code == 0
    assert result.stdout == (
        "Недостаточно данных в проиндексированных документах для ответа.\n"
    )


def test_ask_never_prints_invalid_answer(monkeypatch: object, tmp_path: Path) -> None:
    config = config_file(tmp_path)
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "rag.cli._build_answerer",
        lambda _: StubAnswerer(AnswerValidationError("unknown evidence source")),
    )

    result = runner.invoke(
        app,
        ["ask", "What?", "--config", str(config)],
        env={"ROUTERAI_API_KEY": "secret"},
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "unknown evidence source" in result.stderr
