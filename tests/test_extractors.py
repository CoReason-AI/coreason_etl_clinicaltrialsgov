# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

import pytest
import dlt
from unittest.mock import MagicMock, patch
from coreason_etl_clinicaltrialsgov.extractors import clinicaltrials_source

@pytest.fixture
def mock_client_class():
    with patch("coreason_etl_clinicaltrialsgov.extractors.ClinicalTrialsClient") as mock:
        yield mock

def test_clinicaltrials_source_yields_tables(mock_client_class):
    # Setup mock data
    mock_instance = mock_client_class.return_value

    raw_study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT123", "briefTitle": "Test"},
            "statusModule": {"overallStatus": "RECRUITING"},
            "designModule": {"phases": ["PHASE1"]},
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Sponsor1"}},
            "contactsLocationsModule": {"locations": [{"country": "USA"}]}
        },
        "resultsSection": {}
    }

    mock_instance.list_studies.return_value = iter([raw_study])

    source = clinicaltrials_source()
    # Correct way to access resource from source
    resource = source.resources["studies_stream"]

    # Iterate the resource to get items
    items = list(resource)

    # We expect Bronze, Silver (studies, sponsors, locations), Gold
    # Count: 5 items

    assert len(items) >= 4

    # Categorize by keys since we can't easily rely on _dlt_meta in mocked unit test context
    # unless we verify dlt.mark behavior.

    items_by_type = {
        "bronze": [],
        "silver_study": [],
        "silver_sponsor": [],
        "silver_location": [],
        "gold": []
    }

    for item in items:
        if "raw_payload" in item:
            items_by_type["bronze"].append(item)
        elif "title" in item and "enrollment_bucket" not in item:
            items_by_type["silver_study"].append(item)
        elif "role" in item:
            items_by_type["silver_sponsor"].append(item)
        elif "city" in item: # Location keys
            items_by_type["silver_location"].append(item)
        elif "enrollment_bucket" in item:
            items_by_type["gold"].append(item)

    assert len(items_by_type["bronze"]) == 1
    assert items_by_type["bronze"][0]["source_id"] == "NCT123"

    assert len(items_by_type["silver_study"]) == 1
    assert items_by_type["silver_study"][0]["title"] == "Test"

    assert len(items_by_type["silver_sponsor"]) == 1
    assert items_by_type["silver_sponsor"][0]["name"] == "Sponsor1"

    assert len(items_by_type["gold"]) == 1
    assert items_by_type["gold"][0]["overall_status"] == "RECRUITING"

def test_clinicaltrials_source_skips_invalid(mock_client_class):
    mock_instance = mock_client_class.return_value
    # No NCT ID
    mock_instance.list_studies.return_value = iter([{}])

    source = clinicaltrials_source()
    resource = source.resources["studies_stream"]
    items = list(resource)
    assert len(items) == 0

def test_clinicaltrials_source_filters_gold(mock_client_class):
    mock_instance = mock_client_class.return_value

    raw_study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT123"},
            "statusModule": {"overallStatus": "WITHDRAWN"}, # Invalid for Gold
        }
    }

    mock_instance.list_studies.return_value = iter([raw_study])

    source = clinicaltrials_source()
    resource = source.resources["studies_stream"]
    items = list(resource)

    has_gold = False
    for item in items:
        if "enrollment_bucket" in item:
            has_gold = True

    assert not has_gold
