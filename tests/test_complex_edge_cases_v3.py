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

from coreason_etl_clinicaltrialsgov.transformers import transform_gold, transform_study


def test_transform_gold_negative_years_active() -> None:
    # Scenario: Start date is AFTER completion date (data error in source)
    silver_study = {
        "source_id": "NCT_NEG_DATE",
        "overall_status": "COMPLETED",
        "start_date": date(2025, 1, 1),
        "completion_date": date(2020, 1, 1),
        "enrollment_count": 100,
    }
    locations: list[dict[str, Any]] = []

    gold = transform_gold({}, silver_study, locations)
    assert gold is not None
    assert gold["years_active"] is not None
    # Should be approx -5 years
    assert gold["years_active"] < 0
    assert abs(gold["years_active"] + 5.0) < 0.1


def test_transform_study_duplicate_sponsors() -> None:
    # Scenario: Same entity listed as Lead and Collaborator
    raw: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_DUP_SPONSOR"},
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "Pharma Big", "class": "INDUSTRY"},
                "collaborators": [
                    {"name": "Pharma Big", "class": "INDUSTRY"},
                    {"name": "Uni Small", "class": "OTHER"},
                ],
            },
        }
    }

    result = transform_study(raw)
    sponsors = result["silver_sponsors"]

    # Current logic treats them as separate entries in the list
    # The FRD says: "Wrap leadSponsor in a list and Union with collaborators."
    # It does NOT explicitly say to deduplicate by name.
    # So we expect 3 entries (1 lead + 2 collaborators).
    assert len(sponsors) == 3
    names = [s["name"] for s in sponsors]
    assert names.count("Pharma Big") == 2
    roles = [s["role"] for s in sponsors]
    assert "LEAD" in roles
    assert "COLLABORATOR" in roles


def test_transform_study_unicode_characters() -> None:
    # Scenario: Fields with special characters
    title = "Étude sur la maladie de Ménière"
    sponsor_name = "Hôpital Universitaire"

    raw: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT_UNICODE",
                "briefTitle": title,
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": sponsor_name, "class": "OTHER"},
            },
        }
    }

    result = transform_study(raw)

    study = result["silver_studies"][0]
    assert study["title"] == title

    sponsors = result["silver_sponsors"]
    assert sponsors[0]["name"] == sponsor_name


def test_transform_study_extreme_dates() -> None:
    # Scenario: Very old or future dates
    # API sends dates as strings "YYYY-MM-DD"
    raw: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_EXTREME_DATE"},
            "statusModule": {
                "startDateStruct": {"date": "1900-01-01"},
                "completionDateStruct": {"date": "2100-12-31"},
            },
        }
    }

    result = transform_study(raw)
    study = result["silver_studies"][0]

    assert study["start_date"] == date(1900, 1, 1)
    assert study["completion_date"] == date(2100, 12, 31)


def test_transform_study_empty_lists() -> None:
    # Scenario: Modules exist but lists are empty
    raw: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_EMPTY_LISTS"},
            "armsInterventionsModule": {"interventions": []},
            "outcomesModule": {"primaryOutcomes": [], "secondaryOutcomes": []},
            "contactsLocationsModule": {"locations": []},
        }
    }

    result = transform_study(raw)

    assert result["silver_interventions"] == []
    assert result["silver_outcomes"] == []
    assert result["silver_locations"] == []


def test_transform_gold_large_enrollment() -> None:
    # Scenario: Very large enrollment number
    silver_study = {
        "source_id": "NCT_LARGE",
        "overall_status": "COMPLETED",
        "enrollment_count": 1000000,  # 1 Million
    }

    gold = transform_gold({}, silver_study, [])
    assert gold is not None
    assert gold["enrollment_bucket"] == "Large"
