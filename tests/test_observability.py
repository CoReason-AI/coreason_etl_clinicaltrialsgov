# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

from unittest.mock import patch

import pytest

from coreason_etl_clinicaltrialsgov.client import ClinicalTrialsClient
from coreason_etl_clinicaltrialsgov.extractors import clinicaltrials_source


def test_observability_logs(caplog: pytest.LogCaptureFixture) -> None:
    """Test that Pages Fetched and Records Extracted are logged."""

    # Mock data
    mock_study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT12345678"},
            "statusModule": {"lastUpdatePostDateStruct": {"date": "2023-01-01"}},
        }
    }

    # Mock client response
    mock_response = {"studies": [mock_study] * 5, "nextPageToken": None}

    with patch("coreason_etl_clinicaltrialsgov.client.ClinicalTrialsClient.fetch_studies") as mock_fetch:
        mock_fetch.return_value = mock_response

        # Run source
        source = clinicaltrials_source(page_size=2)
        resource = source.resources["studies_stream"]

        # Iterate to trigger execution
        list(resource)

        # Check logs

        # 1. Check "Pages Fetched" from client
        # We expect one fetch call returning 5 studies
        assert "Pages Fetched: Retrieved 5 studies" in caplog.text

        # 2. Check "Records Extracted" from extractor
        # page_size=2, total 5 items.
        # Batches: 2, 2, 1
        # Expect logs for each batch
        assert "Records Extracted: Processing batch of 2. Total so far: 2" in caplog.text
        assert "Records Extracted: Processing batch of 2. Total so far: 4" in caplog.text
        assert "Records Extracted: Processing final batch of 1. Total so far: 5" in caplog.text


def test_client_observability_pagination(caplog: pytest.LogCaptureFixture) -> None:
    """Test client logging across pages."""
    client = ClinicalTrialsClient()

    with patch.object(client, "fetch_studies") as mock_fetch:
        mock_fetch.side_effect = [
            {"studies": [{"id": 1}], "nextPageToken": "token1"},
            {"studies": [{"id": 2}], "nextPageToken": None},
        ]

        list(client.list_studies(page_size=1))

        assert "Pages Fetched: Retrieved 1 studies. Next Token: token1" in caplog.text
        assert "Pages Fetched: Retrieved 1 studies. Next Token: None" in caplog.text
