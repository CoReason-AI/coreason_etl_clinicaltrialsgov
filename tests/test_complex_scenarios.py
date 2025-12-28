# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

from datetime import date
from typing import Any, Generator
from unittest.mock import patch

import pytest
from coreason_etl_clinicaltrialsgov.extractors import clinicaltrials_source
from coreason_etl_clinicaltrialsgov.transformers import (
    get_enrollment_bucket,
    normalize_age,
    transform_gold,
    transform_study,
)


def test_normalize_age_complex() -> None:
    # Decimals
    assert normalize_age("1.5 Years") == 1.5
    assert normalize_age("0.5 Months") == 0.5 / 12

    # Case insensitivity mixed
    assert normalize_age("10 wEeKs") == 10.0 / 52

    # White space handling
    assert normalize_age("  5   Days  ") == 5.0 / 365

    # Zero
    assert normalize_age("0 Years") == 0.0


def test_enrollment_bucket_boundaries() -> None:
    # Small < 100
    assert get_enrollment_bucket(0) == "Small"
    assert get_enrollment_bucket(99) == "Small"

    # Medium 100 <= x < 1000
    assert get_enrollment_bucket(100) == "Medium"
    assert get_enrollment_bucket(999) == "Medium"

    # Large >= 1000
    assert get_enrollment_bucket(1000) == "Large"
    assert get_enrollment_bucket(100000) == "Large"


def test_transform_study_minimal_valid() -> None:
    # Minimal payload that yields a valid ID
    raw: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_MINIMAL"},
        }
    }

    result = transform_study(raw)

    assert result["silver_studies"][0]["source_id"] == "NCT_MINIMAL"
    # Verify strict robustness: fields should be None, not raise KeyError
    study = result["silver_studies"][0]
    assert study["title"] is None
    assert study["start_date"] is None
    assert study["phases"] is None

    # Lists should be empty
    assert result["silver_sponsors"] == []
    assert result["silver_locations"] == []


def test_transform_study_nested_missing_fields() -> None:
    # Payload with structure but missing leaf nodes
    raw: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_MISSING"},
            "sponsorCollaboratorsModule": {
                # leadSponsor present but missing name
                "leadSponsor": {"class": "INDUSTRY"}
            },
            "contactsLocationsModule": {
                # Location present but empty
                "locations": [{}]
            },
        }
    }

    result = transform_study(raw)

    sponsors = result["silver_sponsors"]
    assert len(sponsors) == 1
    assert sponsors[0]["name"] is None
    assert sponsors[0]["agency_class"] == "INDUSTRY"

    locations = result["silver_locations"]
    assert len(locations) == 1
    assert locations[0]["country"] is None


def test_transform_gold_complex_logic() -> None:
    # Test years_active calculation logic carefully
    silver = {
        "source_id": "NCT_GOLD",
        "coreason_id": "UUID",
        "overall_status": "COMPLETED",
        "start_date": date(2020, 1, 1),
        "completion_date": date(2021, 1, 1),  # 366 days (leap year 2020)
        "enrollment_count": 100,
    }
    raw: dict[str, Any] = {"resultsSection": {}}
    locations: list[dict[str, Any]] = []

    gold = transform_gold(raw, silver, locations)

    assert gold is not None
    # 2020 is a leap year. 2020-01-01 to 2021-01-01 is 366 days.
    # Logic: delta.days / 365.25
    expected_years = 366 / 365.25
    assert gold["years_active"] == expected_years


def test_transform_gold_case_insensitive_status() -> None:
    # Status mixed case
    silver = {
        "source_id": "NCT_CASE",
        "overall_status": "Recruiting",  # Mixed case
        "enrollment_count": 10,
    }
    raw: dict[str, Any] = {}
    locations: list[dict[str, Any]] = []

    gold = transform_gold(raw, silver, locations)
    assert gold is not None
    assert gold["overall_status"] == "Recruiting"
    assert gold["enrollment_bucket"] == "Small"


# --- Extractor Complex Tests ---


@pytest.fixture  # type: ignore[misc]
def mock_client_class() -> Generator[Any, None, None]:
    with patch("coreason_etl_clinicaltrialsgov.extractors.ClinicalTrialsClient") as mock:
        yield mock


def test_extractor_pagination_gap(mock_client_class: Any) -> None:
    mock_instance = mock_client_class.return_value

    # Page 1: Empty studies, but has next token.
    # We simulate the Client behavior by just returning the eventual study found in Page 2.
    page2_study = {"protocolSection": {"identificationModule": {"nctId": "NCT_PAGE2"}}}

    # We test the source behavior when list_studies yields successfully.
    mock_instance.list_studies.return_value = iter([page2_study])

    source = clinicaltrials_source()
    items = list(source.resources["studies_stream"])

    assert len(items) > 0
    assert items[0]["source_id"] == "NCT_PAGE2"


def test_extractor_ignores_study_without_nctid(mock_client_class: Any) -> None:
    mock_instance = mock_client_class.return_value

    # Study 1: Valid
    valid = {"protocolSection": {"identificationModule": {"nctId": "NCT_VALID"}}}
    # Study 2: Invalid (missing ID)
    invalid: dict[str, Any] = {"protocolSection": {"identificationModule": {}}}

    mock_instance.list_studies.return_value = iter([valid, invalid])

    source = clinicaltrials_source()
    items = list(source.resources["studies_stream"])

    # Should only have records for valid
    nct_ids = set()
    for item in items:
        if "source_id" in item:
            nct_ids.add(item["source_id"])

    assert "NCT_VALID" in nct_ids
    assert None not in nct_ids
    assert len(nct_ids) == 1
