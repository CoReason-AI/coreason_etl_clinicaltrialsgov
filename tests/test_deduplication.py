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

from coreason_etl_clinicaltrialsgov.transformers import transform_study


def test_deduplication_sponsors() -> None:
    """Verify that identical sponsors are deduplicated."""
    raw_study: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_DUP_SPONSOR"},
            "sponsorCollaboratorsModule": {
                # Note: leadSponsor is single, collaborators is list
                "leadSponsor": {"name": "Pharma Big", "class": "INDUSTRY"},
                "collaborators": [
                    {"name": "Uni Small", "class": "OTHER"},
                    {"name": "Uni Small", "class": "OTHER"},  # Duplicate
                    {"name": "Uni Small", "class": "DIFFERENT_CLASS"},  # Same Key (Name+Role), Diff Attr
                ],
            },
        }
    }

    result = transform_study(raw_study)
    sponsors = result["silver_sponsors"]

    # Expected:
    # 1. Lead: Pharma Big
    # 2. Collab: Uni Small (First one wins)
    # The third one has same Name+Role ("COLLABORATOR" + "Uni Small"), so it should be skipped
    # based on the "seen" logic using the surrogate key.

    assert len(sponsors) == 2

    names = sorted([s["name"] for s in sponsors])
    assert names == ["Pharma Big", "Uni Small"]

    # Verify First Write Wins for the duplicate (agency_class should be OTHER, not DIFFERENT_CLASS)
    uni_sponsor = next(s for s in sponsors if s["name"] == "Uni Small")
    assert uni_sponsor["agency_class"] == "OTHER"


def test_deduplication_locations() -> None:
    """Verify that identical locations are deduplicated."""
    raw_study: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_DUP_LOC"},
            "contactsLocationsModule": {
                "locations": [
                    {
                        "facility": "General Hospital",
                        "city": "New York",
                        "state": "NY",
                        "country": "USA",
                        "status": "RECRUITING",
                    },
                    {
                        "facility": "General Hospital",
                        "city": "New York",
                        "state": "NY",
                        "country": "USA",
                        "status": "SUSPENDED",  # Different non-key attr
                    },
                ]
            },
        }
    }

    result = transform_study(raw_study)
    locations = result["silver_locations"]

    # Key = facility + city + state + country
    assert len(locations) == 1
    assert locations[0]["status"] == "RECRUITING"  # First write wins


def test_deduplication_interventions() -> None:
    """Verify that identical interventions are deduplicated."""
    raw_study: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_DUP_INT"},
            "armsInterventionsModule": {
                "interventions": [
                    {"type": "DRUG", "name": "Aspirin", "description": "Desc A"},
                    {"type": "DRUG", "name": "Aspirin", "description": "Desc B"},
                ]
            },
        }
    }

    result = transform_study(raw_study)
    interventions = result["silver_interventions"]

    # Key = type + name
    assert len(interventions) == 1
    assert interventions[0]["description"] == "Desc A"


def test_deduplication_outcomes() -> None:
    """Verify that identical outcomes are deduplicated."""
    raw_study: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_DUP_OUT"},
            "outcomesModule": {
                "primaryOutcomes": [
                    {"measure": "Death", "timeFrame": "1 Year", "description": "D1"},
                    {"measure": "Death", "timeFrame": "1 Year", "description": "D2"},
                ]
            },
        }
    }

    result = transform_study(raw_study)
    outcomes = result["silver_outcomes"]

    # Key = type (PRIMARY) + measure + timeFrame
    assert len(outcomes) == 1
    assert outcomes[0]["description"] == "D1"
