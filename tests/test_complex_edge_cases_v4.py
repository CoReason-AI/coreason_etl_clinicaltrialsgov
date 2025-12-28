# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

import sys
from datetime import date
from typing import Any

from coreason_etl_clinicaltrialsgov.transformers import (
    normalize_age,
    parse_date,
    transform_study,
)

# --- Malformed Date Tests ---


def test_malformed_dates() -> None:
    # Invalid formats that should result in None
    assert parse_date("2023/01/01") is None
    assert parse_date("01-2023") is None
    assert parse_date("2023-13-01") is None  # Invalid month
    assert parse_date("2023-00-01") is None  # Invalid month
    assert parse_date("2023-01-32") is None  # Invalid day
    assert parse_date("NotADate") is None
    assert parse_date("") is None


# --- Unicode and Emoji Tests ---


def test_unicode_text_handling() -> None:
    # Text with emojis, non-latin characters
    raw_study: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT_UNICODE",
                "briefTitle": "Study of ❤️ & 💊 in 日本",
                "officialTitle": "Official 🔬 Study",
            },
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "München Pharma", "class": "INDUSTRY"}},
        }
    }

    result = transform_study(raw_study)
    study = result["silver_studies"][0]

    assert study["title"] == "Study of ❤️ & 💊 in 日本"
    assert study["official_title"] == "Official 🔬 Study"

    sponsor = result["silver_sponsors"][0]
    assert sponsor["name"] == "München Pharma"


# --- Extreme Numeric Values ---


def test_extreme_numeric_values() -> None:
    # Max integer for enrollment
    max_int = sys.maxsize
    raw_study: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_EXTREME"},
            "designModule": {"enrollmentInfo": {"count": max_int}},
            "eligibilityModule": {"minimumAge": "150 Years"},
        }
    }

    result = transform_study(raw_study)
    study = result["silver_studies"][0]

    assert study["enrollment_count"] == max_int
    assert study["min_age"] == 150.0

    # Very small age (premature babies?)
    assert normalize_age("0.1 Hours") == 0.1  # Fallback to value
    # But current logic only handles Month/Week/Day/Year units specially?
    # normalize_age logic: "Hour" -> value (0.1)
    # Let's verify that assumption
    assert normalize_age("2 Days") == 2.0 / 365.0


# --- Partial Objects ---


def test_partial_objects_transformation() -> None:
    # Sponsor with no name
    raw_study: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_PARTIAL"},
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"class": "INDUSTRY"}  # No name
            },
            "contactsLocationsModule": {
                "locations": [
                    {"city": "Paris"}  # No facility, no country
                ]
            },
        }
    }

    result = transform_study(raw_study)

    # Check Sponsor
    sponsor = result["silver_sponsors"][0]
    assert sponsor["name"] is None
    assert sponsor["agency_class"] == "INDUSTRY"

    # Check Location
    loc = result["silver_locations"][0]
    assert loc["facility"] is None
    assert loc["city"] == "Paris"
    assert loc["country"] is None


# --- Empty Lists ---


def test_empty_lists_transformation() -> None:
    # Explicit empty lists in source JSON
    raw_study: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_EMPTY_LISTS"},
            "sponsorCollaboratorsModule": {"collaborators": []},
            "contactsLocationsModule": {"locations": []},
            "armsInterventionsModule": {"interventions": []},
            "outcomesModule": {"primaryOutcomes": []},
        }
    }

    result = transform_study(raw_study)

    assert result["silver_sponsors"] == []  # No lead, empty collaborators
    assert result["silver_locations"] == []
    assert result["silver_interventions"] == []
    assert result["silver_outcomes"] == []


# --- Date Logic Edge Cases ---


def test_date_parsing_edge_cases() -> None:
    # Year only
    assert parse_date("2024") == date(2024, 1, 1)
    # Year-Month only
    assert parse_date("2024-02") == date(2024, 2, 1)
    # Full date
    assert parse_date("2024-02-29") == date(2024, 2, 29)  # Leap day
