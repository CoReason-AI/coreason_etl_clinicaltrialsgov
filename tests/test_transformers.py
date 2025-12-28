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
from typing import Any

from coreason_etl_clinicaltrialsgov.transformers import (
    flatten_phases,
    generate_coreason_id,
    get_enrollment_bucket,
    normalize_age,
    parse_date,
    transform_gold,
    transform_study,
)


def test_parse_date() -> None:
    assert parse_date("2023-01-15") == date(2023, 1, 15)
    assert parse_date("2023-01") == date(2023, 1, 1)
    assert parse_date("2023") == date(2023, 1, 1)
    assert parse_date(None) is None
    assert parse_date("invalid") is None
    # Test fallthrough (too many parts)
    assert parse_date("2023-01-01-01") is None


def test_normalize_age() -> None:
    assert normalize_age("18 Years") == 18.0
    assert normalize_age("24 Months") == 2.0
    assert normalize_age("52 Weeks") == 1.0
    assert normalize_age("365 Days") == 1.0
    assert normalize_age("18") == 18.0  # Fallback
    assert normalize_age("Unknown") is None
    assert normalize_age(None) is None
    # Test invalid number format
    assert normalize_age("Years") is None
    assert normalize_age("Unknown Years") is None
    # Test unknown unit
    assert normalize_age("18 Centuries") == 18.0


def test_flatten_phases() -> None:
    assert flatten_phases(["PHASE1", "PHASE2"]) == "PHASE1|PHASE2"
    assert flatten_phases(["PHASE2", "PHASE1"]) == "PHASE1|PHASE2"
    assert flatten_phases(None) is None


def test_generate_coreason_id() -> None:
    uuid_1 = generate_coreason_id("NCT001", "2023-01-01")
    uuid_2 = generate_coreason_id("NCT001", "2023-01-01")
    uuid_3 = generate_coreason_id("NCT002", "2023-01-01")

    assert uuid_1 == uuid_2
    assert uuid_1 != uuid_3


def test_transform_study_basic() -> None:
    raw: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT123", "briefTitle": "Test Study"},
            "statusModule": {
                "startDateStruct": {"date": "2023-01-01"},
                "studyFirstPostDateStruct": {"date": "2022-01-01"},
            },
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "PharmaCorp", "class": "INDUSTRY"}},
            "designModule": {"phases": ["PHASE1"]},
        }
    }

    result = transform_study(raw)

    study = result["silver_studies"][0]
    assert study["source_id"] == "NCT123"
    assert study["title"] == "Test Study"
    assert study["start_date"] == date(2023, 1, 1)
    assert study["phases"] == "PHASE1"

    sponsors = result["silver_sponsors"]
    assert len(sponsors) == 1
    assert sponsors[0]["name"] == "PharmaCorp"
    assert sponsors[0]["role"] == "LEAD"


def test_transform_study_full() -> None:
    raw: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT999"},
            "statusModule": {},
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Lead"}, "collaborators": [{"name": "Collab"}]},
            "contactsLocationsModule": {"locations": [{"city": "New York", "country": "USA"}]},
            "armsInterventionsModule": {"interventions": [{"type": "DRUG", "name": "Aspirin"}]},
            "outcomesModule": {"primaryOutcomes": [{"measure": "Survival"}]},
            "referencesModule": {
                "references": [{"pmid": "123", "citation": "Cit"}],
                "seeAlsoLinks": [{"url": "http://example.com"}],
            },
        }
    }

    result = transform_study(raw)

    assert len(result["silver_sponsors"]) == 2
    assert result["silver_sponsors"][1]["role"] == "COLLABORATOR"

    assert len(result["silver_locations"]) == 1
    assert result["silver_locations"][0]["city"] == "New York"

    assert len(result["silver_interventions"]) == 1
    assert result["silver_interventions"][0]["name"] == "Aspirin"

    assert len(result["silver_outcomes"]) == 1
    assert result["silver_outcomes"][0]["measure"] == "Survival"
    assert result["silver_outcomes"][0]["outcome_type"] == "PRIMARY"

    assert len(result["silver_references"]) == 2
    assert result["silver_references"][0]["type"] == "REFERENCE"
    assert result["silver_references"][1]["type"] == "LINK"


def test_transform_study_empty() -> None:
    result = transform_study({})
    assert result == {}


def test_get_enrollment_bucket() -> None:
    assert get_enrollment_bucket(None) is None
    assert get_enrollment_bucket(50) == "Small"
    assert get_enrollment_bucket(200) == "Medium"
    assert get_enrollment_bucket(2000) == "Large"


def test_transform_gold_valid() -> None:
    raw: dict[str, Any] = {"resultsSection": {}}
    silver = {
        "source_id": "NCT001",
        "coreason_id": "UUID",
        "title": "Title",
        "overall_status": "RECRUITING",
        "start_date": date(2020, 1, 1),
        "completion_date": date(2021, 1, 1),
        "enrollment_count": 50,
    }
    locations: list[dict[str, Any]] = [{"country": "USA"}, {"country": "Canada"}, {"country": "USA"}]

    gold = transform_gold(raw, silver, locations)

    assert gold is not None
    assert gold["overall_status"] == "RECRUITING"
    assert gold["enrollment_bucket"] == "Small"
    assert abs(gold["years_active"] - 1.0) < 0.01
    assert gold["has_results"] is True
    assert set(gold["geo_countries"]) == {"USA", "Canada"}


def test_transform_gold_filtered() -> None:
    raw: dict[str, Any] = {}
    silver = {
        "overall_status": "WITHDRAWN"  # Not in valid set
    }
    locations: list[dict[str, Any]] = []

    gold = transform_gold(raw, silver, locations)
    assert gold is None


def test_transform_gold_missing_dates() -> None:
    raw: dict[str, Any] = {}
    silver = {"overall_status": "COMPLETED", "enrollment_count": None}
    locations: list[dict[str, Any]] = []

    gold = transform_gold(raw, silver, locations)
    assert gold is not None
    assert gold["years_active"] is None
    assert gold["enrollment_bucket"] is None
    assert gold["has_results"] is False
