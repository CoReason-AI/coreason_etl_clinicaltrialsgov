# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from coreason_etl_clinicaltrialsgov.extractors import clinicaltrials_source


@pytest.fixture
def mock_client_class() -> Generator[MagicMock, None, None]:
    with patch("coreason_etl_clinicaltrialsgov.extractors.ClinicalTrialsClient") as mock:
        yield mock


@pytest.fixture
def mock_transform_study() -> Generator[MagicMock, None, None]:
    with patch("coreason_etl_clinicaltrialsgov.extractors.transform_study") as mock:
        yield mock


@pytest.fixture
def mock_transform_gold() -> Generator[MagicMock, None, None]:
    with patch("coreason_etl_clinicaltrialsgov.extractors.transform_gold") as mock:
        yield mock


def test_clinicaltrials_source_structure() -> None:
    source = clinicaltrials_source()
    # It returns a DltSource object
    assert source.name == "clinicaltrials"
    # Check resources
    assert "studies_stream" in source.resources


def test_studies_generator_flow(
    mock_client_class: MagicMock, mock_transform_study: MagicMock, mock_transform_gold: MagicMock
) -> None:
    # Setup mocks
    client_instance = mock_client_class.return_value

    # Mock data
    raw_study = {"protocolSection": {"identificationModule": {"nctId": "NCT001"}}}
    client_instance.list_studies.return_value = iter([raw_study])

    # Mock transforms
    mock_transform_study.return_value = {
        "silver_studies": [{"source_id": "NCT001", "overall_status": "RECRUITING"}],
        "silver_locations": [{"city": "Boston"}],
    }
    mock_transform_gold.return_value = {"gold_field": "val"}

    # Run generator directly
    source = clinicaltrials_source()
    resource = source.resources["studies_stream"]

    # We can iterate the resource
    items = list(resource)

    # Expected items:
    # 1. Bronze record (dict with table name mark)
    # 2. Silver study
    # 3. Silver locations
    # 4. Gold record

    # dlt resources yield TDataItems which can be dicts or lists of dicts
    # Verify we got items
    assert len(items) >= 3

    # Check Bronze
    # Note: accessing yielded items from dlt resource directly might require handling DltResource specifics,
    # but for a generator resource, iterating it yields the data items.

    bronze = next(i for i in items if isinstance(i, dict) and i.get("raw_payload") == raw_study)
    assert bronze["source_id"] == "NCT001"

    # Check Silver
    silver_study = next(i for i in items if isinstance(i, dict) and i.get("overall_status") == "RECRUITING")
    assert silver_study["source_id"] == "NCT001"

    silver_loc = next(i for i in items if isinstance(i, dict) and i.get("city") == "Boston")
    assert silver_loc is not None

    # Check Gold
    gold = next(i for i in items if isinstance(i, dict) and i.get("gold_field") == "val")
    assert gold is not None


def test_studies_generator_skip_no_nct(mock_client_class: MagicMock) -> None:
    client_instance = mock_client_class.return_value
    # Study with no nctId
    client_instance.list_studies.return_value = iter([{"protocolSection": {}}])

    source = clinicaltrials_source()
    resource = source.resources["studies_stream"]
    items = list(resource)

    assert len(items) == 0


def test_studies_generator_gold_skip(
    mock_client_class: MagicMock, mock_transform_study: MagicMock, mock_transform_gold: MagicMock
) -> None:
    client_instance = mock_client_class.return_value
    raw_study = {"protocolSection": {"identificationModule": {"nctId": "NCT001"}}}
    client_instance.list_studies.return_value = iter([raw_study])

    mock_transform_study.return_value = {
        "silver_studies": [{"source_id": "NCT001"}],
        "silver_locations": [],
    }
    # Gold returns None (filtered out)
    mock_transform_gold.return_value = None

    source = clinicaltrials_source()
    resource = source.resources["studies_stream"]
    items = list(resource)

    # Should contain Bronze + Silver, but no Gold
    assert any(i.get("source_id") == "NCT001" for i in items)
    # No item with table name 'gold_studies' logic check is hard without checking dlt marks,
    # but we can check the return values.
    # Since we mocked transform_gold to return None, the generator logic `if gold_record: yield` won't yield it.

    # Verify mock call
    mock_transform_gold.assert_called_once()
