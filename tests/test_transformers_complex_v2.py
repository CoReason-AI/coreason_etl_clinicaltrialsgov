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

import pytest

from coreason_etl_clinicaltrialsgov.transformers import normalize_age, parse_date, transform_study

# --- Date Parsing Tests ---


@pytest.mark.parametrize(
    "input_date,expected",
    [
        ("2023-01-01", date(2023, 1, 1)),
        ("2023-01", date(2023, 1, 1)),
        ("2023", date(2023, 1, 1)),
        ("2023-02-28", date(2023, 2, 28)),
        # Invalid dates should return None
        ("2023-02-30", None),
        ("2023-13-01", None),
        ("0000-00-00", None),
        ("9999-99-99", None),
        ("Invalid", None),
        ("", None),
        (None, None),
        ("2023/01/01", None),  # Wrong separator
    ],
)
def test_parse_date_complex(input_date: str | None, expected: date | None) -> None:
    assert parse_date(input_date) == expected


# --- Age Normalization Tests ---


@pytest.mark.parametrize(
    "input_age,expected",
    [
        ("18 Years", 18.0),
        ("18 years", 18.0),
        ("18", 18.0),  # Default to years
        ("24 Months", 2.0),
        ("104 Weeks", 2.0),
        ("730 Days", 2.0),
        ("18.5 Years", 18.5),
        # Edge cases
        ("N/A", None),
        ("Unknown", None),
        ("> 18 Years", None),  # Modifiers not handled currently
        ("", None),
        (None, None),
    ],
)
def test_normalize_age_complex(input_age: str | None, expected: float | None) -> None:
    result = normalize_age(input_age)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected, 0.001)


# --- Deduplication Tests ---


def test_transform_study_deduplicate_sponsors() -> None:
    """Test that sponsors with same name/role are deduplicated."""
    raw_study: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000001"},
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "Pharma Corp", "class": "INDUSTRY"},
                "collaborators": [
                    {"name": "Lab A", "class": "OTHER"},
                    {"name": "Lab A", "class": "OTHER"},  # Duplicate
                    {"name": "Lab B", "class": "OTHER"},
                ],
            },
        }
    }

    result = transform_study(raw_study)
    sponsors = result["silver_sponsors"]

    # Expect: 1 Lead (Pharma Corp) + 1 Lab A + 1 Lab B = 3 total
    assert len(sponsors) == 3

    names = sorted([s["name"] for s in sponsors])
    assert names == ["Lab A", "Lab B", "Pharma Corp"]


def test_transform_study_deduplicate_locations() -> None:
    """Test that locations with same keys are deduplicated."""
    raw_study: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000002"},
            "contactsLocationsModule": {
                "locations": [
                    {
                        "facility": "Hospital A",
                        "city": "New York",
                        "state": "NY",
                        "country": "USA",
                        "status": "RECRUITING",
                    },
                    {
                        "facility": "Hospital A",
                        "city": "New York",
                        "state": "NY",
                        "country": "USA",
                        "status": "SUSPENDED",  # Different status shouldn't affect ID (not part of key)
                    },
                    {"facility": "Hospital B", "city": "Boston", "state": "MA", "country": "USA"},
                ]
            },
        }
    }

    result = transform_study(raw_study)
    locations = result["silver_locations"]

    # Expect: Hospital A (deduped) + Hospital B = 2 total
    assert len(locations) == 2

    facilities = sorted([loc["facility"] for loc in locations])
    assert facilities == ["Hospital A", "Hospital B"]


def test_transform_study_deduplicate_interventions() -> None:
    """Test that interventions are deduplicated."""
    raw_study: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000003"},
            "armsInterventionsModule": {
                "interventions": [
                    {"type": "DRUG", "name": "Aspirin", "description": "100mg"},
                    {"type": "DRUG", "name": "Aspirin", "description": "200mg"},  # Duplicate key
                    {"type": "DEVICE", "name": "Stent"},
                ]
            },
        }
    }

    result = transform_study(raw_study)
    interventions = result["silver_interventions"]

    assert len(interventions) == 2
    names = sorted([i["name"] for i in interventions])
    assert names == ["Aspirin", "Stent"]


def test_transform_study_deduplicate_outcomes() -> None:
    """Test that outcomes are deduplicated."""
    raw_study: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000004"},
            "outcomesModule": {
                "primaryOutcomes": [
                    {"measure": "Survival", "timeFrame": "1 Year"},
                    {"measure": "Survival", "timeFrame": "1 Year", "description": "Detailed"},  # Duplicate
                ],
                "secondaryOutcomes": [{"measure": "Pain", "timeFrame": "1 Month"}],
            },
        }
    }

    result = transform_study(raw_study)
    outcomes = result["silver_outcomes"]

    assert len(outcomes) == 2
    measures = sorted([o["measure"] for o in outcomes])
    assert measures == ["Pain", "Survival"]
