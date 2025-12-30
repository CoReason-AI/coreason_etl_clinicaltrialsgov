# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

import re
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from coreason_etl_clinicaltrialsgov.main import app

runner = CliRunner()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def test_cli_invalid_page_size() -> None:
    """Test CLI validation for invalid page size (non-int)."""
    # Force NO_COLOR to avoid rich formatting issues in assertion
    result = runner.invoke(app, ["run", "--page-size", "invalid"], env={"NO_COLOR": "1"})

    clean_output = strip_ansi(result.output)
    assert result.exit_code == 2
    # Expect "Invalid value for '--page-size': 'invalid' is not a valid integer"
    # We check for key parts to be robust
    assert "Invalid value" in clean_output
    assert "--page-size" in clean_output


def test_cli_unknown_argument() -> None:
    """Test CLI with unknown argument."""
    result = runner.invoke(app, ["run", "--unknown-arg", "value"], env={"NO_COLOR": "1"})

    clean_output = strip_ansi(result.output)
    assert result.exit_code == 2
    # Typer/Click usually reports "No such option: --unknown-arg"
    assert "no such option" in clean_output.lower()


@patch("coreason_etl_clinicaltrialsgov.main.dlt.pipeline")
def test_cli_pipeline_init_failure(mock_pipeline: MagicMock) -> None:
    """Test graceful failure when pipeline init fails."""
    mock_pipeline.side_effect = ValueError("Invalid destination configuration")

    result = runner.invoke(app, ["run", "--destination", "invalid"], env={"NO_COLOR": "1"})

    assert result.exit_code == 1


@patch("coreason_etl_clinicaltrialsgov.main.dlt.pipeline")
@patch("coreason_etl_clinicaltrialsgov.main.clinicaltrials_source")
def test_cli_source_failure(mock_source: MagicMock, mock_pipeline: MagicMock) -> None:
    """Test graceful failure when source generation fails."""
    pipeline_instance = MagicMock()
    mock_pipeline.return_value = pipeline_instance

    # Mock source creation failure
    mock_source.side_effect = RuntimeError("Source configuration error")

    result = runner.invoke(app, ["run"], env={"NO_COLOR": "1"})

    assert result.exit_code == 1


@patch("coreason_etl_clinicaltrialsgov.main.dlt.pipeline")
@patch("coreason_etl_clinicaltrialsgov.main.clinicaltrials_source")
def test_cli_empty_args(mock_source: MagicMock, mock_pipeline: MagicMock) -> None:
    """Test passing empty strings as arguments (should be handled)."""
    pipeline_instance = MagicMock()
    mock_pipeline.return_value = pipeline_instance
    mock_source.return_value = "Source"

    result = runner.invoke(app, ["run", "--dataset-name", ""], env={"NO_COLOR": "1"})

    assert result.exit_code == 0
    mock_pipeline.assert_called_with(
        pipeline_name="clinicaltrials_etl", destination="postgres", dataset_name="", progress="log"
    )
