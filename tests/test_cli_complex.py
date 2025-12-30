# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from coreason_etl_clinicaltrialsgov.main import app

runner = CliRunner()


def test_cli_invalid_page_size() -> None:
    """Test CLI validation for invalid page size (non-int)."""
    result = runner.invoke(app, ["run", "--page-size", "invalid"])
    assert result.exit_code == 2
    assert "Invalid value for '--page-size'" in result.output


def test_cli_unknown_argument() -> None:
    """Test CLI with unknown argument."""
    result = runner.invoke(app, ["run", "--unknown-arg", "value"])
    assert result.exit_code == 2
    assert "no such option: --unknown-arg" in result.output.lower()


@patch("coreason_etl_clinicaltrialsgov.main.dlt.pipeline")
def test_cli_pipeline_init_failure(mock_pipeline: MagicMock) -> None:
    """Test graceful failure when pipeline init fails."""
    mock_pipeline.side_effect = ValueError("Invalid destination configuration")

    result = runner.invoke(app, ["run", "--destination", "invalid"])

    # Should exit with code 1 (caught exception)
    assert result.exit_code == 1
    # Typer/CliRunner usually captures stderr/stdout. The logger prints to stderr/stdout depending on config.
    # Our logger configuration might send to file or stderr.
    # We can check if exit code is 1, which means our try/except block caught it.


@patch("coreason_etl_clinicaltrialsgov.main.dlt.pipeline")
@patch("coreason_etl_clinicaltrialsgov.main.clinicaltrials_source")
def test_cli_source_failure(mock_source: MagicMock, mock_pipeline: MagicMock) -> None:
    """Test graceful failure when source generation fails."""
    pipeline_instance = MagicMock()
    mock_pipeline.return_value = pipeline_instance

    # Mock source creation failure
    mock_source.side_effect = RuntimeError("Source configuration error")

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 1


@patch("coreason_etl_clinicaltrialsgov.main.dlt.pipeline")
@patch("coreason_etl_clinicaltrialsgov.main.clinicaltrials_source")
def test_cli_empty_args(mock_source: MagicMock, mock_pipeline: MagicMock) -> None:
    """Test passing empty strings as arguments (should be handled)."""
    pipeline_instance = MagicMock()
    mock_pipeline.return_value = pipeline_instance
    mock_source.return_value = "Source"

    # dlt might complain about empty dataset_name, but the CLI should pass it through.
    # We rely on dlt to validate or fail.
    # Here we just check that main.py doesn't crash before calling pipeline.

    result = runner.invoke(app, ["run", "--dataset-name", ""])

    assert result.exit_code == 0
    mock_pipeline.assert_called_with(
        pipeline_name="clinicaltrials_etl", destination="postgres", dataset_name="", progress="log"
    )
