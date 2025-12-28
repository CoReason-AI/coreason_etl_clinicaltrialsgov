# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

import uuid
from datetime import date
from typing import Any

import pytest

from coreason_etl_clinicaltrialsgov.transformers import (
    flatten_phases,
    generate_coreason_id,
    get_enrollment_bucket,
    normalize_age,
    parse_date,
    transform_gold,
    transform_study,
)

# --- Helper Tests ---


@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("2023-10-01", date(2023, 10, 1)),
        ("2023-10", date(2023, 10, 1)),
        ("2023", date(2023, 1, 1)),
        (None, None),
        ("invalid", None),
        ("2023-13-01", None),  # Invalid month
    ],
)
def test_parse_date(input_str: str | None, expected: date | None) -> None:
    assert parse_date(input_str) == expected


@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("18 Years", 18.0),
        ("24 Months", 2.0),
        ("52 Weeks", 1.0),
        ("365 Days", 1.0),
        ("18", 18.0),  # No unit
        (None, None),
        ("invalid", None),
        ("Year", None),  # No number
    ],
)
def test_normalize_age(input_str: str | None, expected: float | None) -> None:
    if expected is None:
        assert normalize_age(input_str) is None
    else:
        assert normalize_age(input_str) == pytest.approx(expected)


def test_generate_coreason_id() -> None:
    nct_id = "NCT123"
    date_str = "2023-01-01"
    expected_seed = f"clinicaltrials.gov/{nct_id}/{date_str}"
    expected_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, expected_seed))

    assert generate_coreason_id(nct_id, date_str) == expected_uuid


def test_flatten_phases() -> None:
    assert flatten_phases(["PHASE2", "PHASE1"]) == "PHASE1|PHASE2"
    assert flatten_phases(None) is None
    assert flatten_phases([]) is None


@pytest.mark.parametrize(
    "count, expected",
    [
        (50, "Small"),
        (500, "Medium"),
        (1500, "Large"),
        (None, None),
    ],
)
def test_get_enrollment_bucket(count: int | None, expected: str | None) -> None:
    assert get_enrollment_bucket(count) == expected


# --- Transformation Logic Tests ---


@pytest.fixture
def sample_raw_study() -> dict[str, Any]:
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT00000001",
                "briefTitle": "Brief Title",
                "officialTitle": "Official Title",
                "orgStudyIdInfo": {"id": "ORG123"},
            },
            "statusModule": {
                "overallStatus": "RECRUITING",
                "startDateStruct": {"date": "2023-01"},
                "completionDateStruct": {"date": "2024-01"},
                "studyFirstPostDateStruct": {"date": "2022-12-01"},
            },
            "designModule": {
                "phases": ["PHASE1", "PHASE2"],
                "studyType": "INTERVENTIONAL",
                "enrollmentInfo": {"count": 100, "type": "ESTIMATED"},
            },
            "eligibilityModule": {
                "minimumAge": "18 Years",
                "maximumAge": "65 Years",
                "sex": "ALL",
                "healthyVolunteers": True,
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "Lead Corp", "class": "INDUSTRY"},
                "collaborators": [{"name": "Uni Lab", "class": "OTHER"}],
            },
            "contactsLocationsModule": {
                "locations": [
                    {
                        "facility": "Hospital A",
                        "city": "New York",
                        "state": "NY",
                        "zip": "10001",
                        "country": "United States",
                        "status": "RECRUITING",
                    }
                ]
            },
            "armsInterventionsModule": {
                "interventions": [
                    {
                        "type": "DRUG",
                        "name": "Drug X",
                        "description": "5mg daily",
                        "otherNames": ["BrandX"],
                    }
                ]
            },
            "outcomesModule": {
                "primaryOutcomes": [{"measure": "Pain score", "timeFrame": "1 year"}],
                "secondaryOutcomes": [{"measure": "Survival", "description": "Overall survival"}],
            },
            "referencesModule": {
                "references": [{"pmid": "12345", "citation": "Author et al."}],
                "seeAlsoLinks": [{"label": "Link 1", "url": "http://example.com"}],
            },
        },
        "resultsSection": {},  # Indicating results exist
    }


def test_transform_study(sample_raw_study: dict[str, Any]) -> None:
    result = transform_study(sample_raw_study)

    # Check Studies Table
    assert len(result["silver_studies"]) == 1
    study = result["silver_studies"][0]
    assert study["source_id"] == "NCT00000001"
    assert study["title"] == "Brief Title"
    assert study["overall_status"] == "RECRUITING"
    assert study["phases"] == "PHASE1|PHASE2"
    assert study["min_age"] == 18.0

    # Check Sponsors
    sponsors = result["silver_sponsors"]
    assert len(sponsors) == 2
    assert sponsors[0]["role"] == "LEAD"
    assert sponsors[0]["name"] == "Lead Corp"
    assert sponsors[1]["role"] == "COLLABORATOR"
    assert sponsors[1]["name"] == "Uni Lab"

    # Check Locations
    locations = result["silver_locations"]
    assert len(locations) == 1
    assert locations[0]["city"] == "New York"

    # Check Interventions
    interventions = result["silver_interventions"]
    assert len(interventions) == 1
    assert interventions[0]["name"] == "Drug X"

    # Check Outcomes
    outcomes = result["silver_outcomes"]
    assert len(outcomes) == 2
    assert outcomes[0]["outcome_type"] == "PRIMARY"
    assert outcomes[1]["outcome_type"] == "SECONDARY"

    # Check References
    refs = result["silver_references"]
    assert len(refs) == 2
    assert refs[0]["type"] == "REFERENCE"
    assert refs[0]["pmid"] == "12345"
    assert refs[1]["type"] == "LINK"
    assert refs[1]["url"] == "http://example.com"


def test_transform_study_empty_nct() -> None:
    # If no nctId, returns empty dict
    res = transform_study({})
    assert res == {}


def test_transform_gold(sample_raw_study: dict[str, Any]) -> None:
    silver_study = transform_study(sample_raw_study)["silver_studies"][0]
    locations = transform_study(sample_raw_study)["silver_locations"]

    gold = transform_gold(sample_raw_study, silver_study, locations)

    assert gold is not None
    assert gold["source_id"] == "NCT00000001"
    assert gold["overall_status"] == "RECRUITING"
    assert gold["enrollment_bucket"] == "Medium"  # 100
    assert gold["has_results"] is True
    assert "United States" in gold["geo_countries"]
    # 2023-01-01 to 2024-01-01 is 1 year (approx 365 days)
    # 365 / 365.25 ~= 0.999
    assert gold["years_active"] == pytest.approx(1.0, rel=0.01)


def test_transform_gold_filtered_status(sample_raw_study: dict[str, Any]) -> None:
    silver_study = transform_study(sample_raw_study)["silver_studies"][0]
    locations = transform_study(sample_raw_study)["silver_locations"]

    # Change status to invalid
    silver_study["overall_status"] = "WITHDRAWN"

    gold = transform_gold(sample_raw_study, silver_study, locations)
    assert gold is None
