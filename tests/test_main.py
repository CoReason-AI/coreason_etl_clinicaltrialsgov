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


@patch("coreason_etl_clinicaltrialsgov.main.dlt.pipeline")
@patch("coreason_etl_clinicaltrialsgov.main.clinicaltrials_source")
def test_run_command_defaults(mock_source: MagicMock, mock_pipeline: MagicMock) -> None:
    """Test running the CLI with default arguments."""
    pipeline_instance = MagicMock()
    mock_pipeline.return_value = pipeline_instance
    pipeline_instance.run.return_value = "LoadInfo"
    mock_source.return_value = "Source"

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 0
    mock_pipeline.assert_called_once_with(
        pipeline_name="clinicaltrials_etl",
        destination="postgres",
        dataset_name=None,
        progress="log",
    )
    mock_source.assert_called_once_with(page_size=100, query_term=None)
    pipeline_instance.run.assert_called_once_with("Source")


@patch("coreason_etl_clinicaltrialsgov.main.dlt.pipeline")
@patch("coreason_etl_clinicaltrialsgov.main.clinicaltrials_source")
def test_run_command_custom_args(mock_source: MagicMock, mock_pipeline: MagicMock) -> None:
    """Test running the CLI with custom arguments."""
    pipeline_instance = MagicMock()
    mock_pipeline.return_value = pipeline_instance
    pipeline_instance.run.return_value = "LoadInfo"
    mock_source.return_value = "Source"

    result = runner.invoke(
        app,
        [
            "run",
            "--page-size",
            "50",
            "--query-term",
            "heart attack",
            "--destination",
            "duckdb",
            "--pipeline-name",
            "custom_pipe",
            "--dataset-name",
            "custom_ds",
        ],
    )

    assert result.exit_code == 0
    mock_pipeline.assert_called_once_with(
        pipeline_name="custom_pipe", destination="duckdb", dataset_name="custom_ds", progress="log"
    )
    mock_source.assert_called_once_with(page_size=50, query_term="heart attack")
    pipeline_instance.run.assert_called_once_with("Source")


@patch("coreason_etl_clinicaltrialsgov.main.dlt.pipeline")
def test_run_command_failure(mock_pipeline: MagicMock) -> None:
    """Test that the CLI exits with 1 on exception."""
    mock_pipeline.side_effect = Exception("Boom")

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 1
    # Verify strict error handling is logged (covered by logger.exception but nice to check exit code)
