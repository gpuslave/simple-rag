from pathlib import Path

from typer.testing import CliRunner

from rag.cli import app

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
