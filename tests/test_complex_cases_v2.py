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

from coreason_etl_clinicaltrialsgov.transformers import transform_gold, transform_study


def test_transform_study_multiple_collaborators_and_locations() -> None:
    raw: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_MULTI"},
            "sponsorCollaboratorsModule": {
                "collaborators": [
                    {"name": "Collab 1", "class": "OTHER"},
                    {"name": "Collab 2", "class": "INDUSTRY"},
                ]
            },
            "contactsLocationsModule": {
                "locations": [
                    {"facility": "Fac 1", "country": "United States"},
                    {"facility": "Fac 2", "country": "Canada"},
                    {"facility": "Fac 3", "country": "United States"},  # Duplicate country
                ]
            },
        }
    }

    result = transform_study(raw)

    sponsors = result["silver_sponsors"]
    # 0 lead, 2 collaborators
    assert len(sponsors) == 2
    assert sponsors[0]["name"] == "Collab 1"
    assert sponsors[0]["role"] == "COLLABORATOR"
    assert sponsors[1]["name"] == "Collab 2"

    locations = result["silver_locations"]
    assert len(locations) == 3


def test_transform_gold_geo_countries_deduplication() -> None:
    # Gold layer should deduplicate countries list
    silver_study = {
        "source_id": "NCT_MULTI",
        "coreason_id": "uuid",
        "overall_status": "RECRUITING",
        "enrollment_count": 100,
    }
    locations: list[dict[str, Any]] = [
        {"country": "United States"},
        {"country": "Canada"},
        {"country": "United States"},
        {"country": None},  # Should be ignored
    ]

    gold = transform_gold({"resultsSection": {}}, silver_study, locations)
    assert gold is not None
    countries = gold["geo_countries"]
    assert len(countries) == 2
    assert "United States" in countries
    assert "Canada" in countries


def test_transform_study_malformed_structs() -> None:
    # Test when struct exists but inner fields are unexpected
    raw: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_MALFORMED"},
            "statusModule": {
                # startDateStruct exists but is empty
                "startDateStruct": {},
                # completionDateStruct exists but date is None
                "completionDateStruct": {"date": None},
            },
        }
    }

    result = transform_study(raw)
    study = result["silver_studies"][0]
    assert study["start_date"] is None
    assert study["completion_date"] is None


def test_transform_study_results_section_presence() -> None:
    # Case 1: resultsSection present (dict)
    silver_study = {
        "source_id": "NCT_RESULTS",
        "coreason_id": "UUID-123",
        "overall_status": "COMPLETED",
        "enrollment_count": 10,
    }
    locations_empty: list[dict[str, Any]] = []
    gold_yes = transform_gold({"resultsSection": {}}, silver_study, locations_empty)
    assert gold_yes is not None
    assert gold_yes["has_results"] is True

    # Case 2: resultsSection absent
    locations: list[dict[str, Any]] = []
    gold_no = transform_gold({}, silver_study, locations)
    assert gold_no is not None
    assert gold_no["has_results"] is False


def test_string_coercion_for_enrollment() -> None:
    # enrollmentInfo count as string "100" -> should coerce to int 100 by Pydantic
    raw: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_COERCE"},
            "designModule": {"enrollmentInfo": {"count": "100", "type": "ACTUAL"}},
        }
    }

    result = transform_study(raw)
    study = result["silver_studies"][0]
    assert study["enrollment_count"] == 100
    assert isinstance(study["enrollment_count"], int)
