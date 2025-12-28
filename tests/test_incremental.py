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


@pytest.fixture  # type: ignore[misc]
def mock_client_class() -> Generator[MagicMock, None, None]:
    with patch("coreason_etl_clinicaltrialsgov.extractors.ClinicalTrialsClient") as mock:
        yield mock


@pytest.fixture  # type: ignore[misc]
def mock_dlt_state() -> Generator[dict[str, str], None, None]:
    # Mock dlt.current.source_state()
    # It returns a dict-like object that persists changes
    state: dict[str, str] = {}
    with patch("dlt.current.source_state", return_value=state):
        yield state


def test_incremental_load_logic(mock_client_class: MagicMock, mock_dlt_state: dict[str, str]) -> None:
    # Setup initial state
    mock_dlt_state["last_updated_date"] = "2023-01-01"

    client_instance = mock_client_class.return_value
    # Mock studies with newer date
    client_instance.list_studies.return_value = iter(
        [
            {
                "protocolSection": {
                    "identificationModule": {"nctId": "NCT001"},
                    "statusModule": {"lastUpdatePostDateStruct": {"date": "2023-02-01"}},
                }
            }
        ]
    )

    source = clinicaltrials_source()
    resource = source.resources["studies_stream"]

    # Run extraction
    list(resource)

    # Verify query term used
    client_instance.list_studies.assert_called_once()
    call_kwargs = client_instance.list_studies.call_args[1]
    assert call_kwargs["query_term"] == "AREA[LastUpdatePostDate]RANGE[2023-01-01,MAX]"

    # Verify state update
    assert mock_dlt_state["last_updated_date"] == "2023-02-01"


def test_initial_load_logic(mock_client_class: MagicMock, mock_dlt_state: dict[str, str]) -> None:
    # Empty state
    mock_dlt_state.clear()

    client_instance = mock_client_class.return_value
    client_instance.list_studies.return_value = iter(
        [
            {
                "protocolSection": {
                    "identificationModule": {"nctId": "NCT001"},
                    "statusModule": {"lastUpdatePostDateStruct": {"date": "2023-02-01"}},
                }
            }
        ]
    )

    source = clinicaltrials_source()
    resource = source.resources["studies_stream"]
    list(resource)

    # Verify no query term
    call_kwargs = client_instance.list_studies.call_args[1]
    assert call_kwargs["query_term"] is None

    # Verify state updated (first run sets baseline)
    assert mock_dlt_state["last_updated_date"] == "2023-02-01"


def test_custom_query_override(mock_client_class: MagicMock, mock_dlt_state: dict[str, str]) -> None:
    mock_dlt_state["last_updated_date"] = "2023-01-01"

    client_instance = mock_client_class.return_value
    client_instance.list_studies.return_value = iter([])

    custom_term = "Heart Attack"
    source = clinicaltrials_source(query_term=custom_term)
    resource = source.resources["studies_stream"]
    list(resource)

    # Verify custom term used, ignoring state logic for query construction
    call_kwargs = client_instance.list_studies.call_args[1]
    assert call_kwargs["query_term"] == custom_term


def test_state_updates_max_seen(mock_client_class: MagicMock, mock_dlt_state: dict[str, str]) -> None:
    mock_dlt_state["last_updated_date"] = "2023-01-01"

    client_instance = mock_client_class.return_value
    # Mixed dates, some older, some newer, random order
    studies = [
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT1"},
                "statusModule": {"lastUpdatePostDateStruct": {"date": "2023-01-05"}},
            }
        },
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT2"},
                "statusModule": {"lastUpdatePostDateStruct": {"date": "2022-01-01"}},
            }
        },  # Old
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT3"},
                "statusModule": {"lastUpdatePostDateStruct": {"date": "2023-01-10"}},
            }
        },  # New Max
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT4"},
                "statusModule": {"lastUpdatePostDateStruct": {"date": "2023-01-02"}},
            }
        },
    ]
    client_instance.list_studies.return_value = iter(studies)

    source = clinicaltrials_source()
    resource = source.resources["studies_stream"]
    list(resource)

    assert mock_dlt_state["last_updated_date"] == "2023-01-10"
