# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

from typing import Any
from unittest.mock import MagicMock, patch

from coreason_etl_clinicaltrialsgov.main import run_pipeline


@patch("coreason_etl_clinicaltrialsgov.main.dlt.pipeline")
@patch("coreason_etl_clinicaltrialsgov.main.clinicaltrials_source")
def test_run_pipeline(mock_source: Any, mock_pipeline: Any) -> None:
    # Setup mocks
    pipeline_instance = MagicMock()
    mock_pipeline.return_value = pipeline_instance
    pipeline_instance.run.return_value = "LoadInfo"

    mock_source.return_value = "Source"

    run_pipeline()

    # Verify calls
    mock_pipeline.assert_called_once_with(
        pipeline_name="clinicaltrials_etl", destination="postgres", dataset_name="clinical_trials_data", progress="log"
    )
    mock_source.assert_called_once()
    pipeline_instance.run.assert_called_once_with("Source")
