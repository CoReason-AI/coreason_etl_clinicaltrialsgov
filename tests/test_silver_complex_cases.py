# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

from coreason_etl_clinicaltrialsgov.transformers import generate_surrogate_key, parse_date, transform_study


def test_surrogate_key_unicode() -> None:
    """Verify robust handling of Unicode and special characters in keys."""
    # Japanese characters
    k1 = generate_surrogate_key("NCT123", "株式会社", "München")
    # Same input must produce same key
    k2 = generate_surrogate_key("NCT123", "株式会社", "München")
    assert k1 == k2

    # Different character should differ
    k3 = generate_surrogate_key("NCT123", "株式会社", "Munchen")
    assert k1 != k3


def test_surrogate_key_special_chars() -> None:
    """Verify handling of delimiters and special chars in input."""
    # We now sanitize the input by replacing '|' with '_' to avoid collisions.
    # Therefore ("A", "B") -> "parent|A|B"
    # But ("A|B") -> "parent|A_B"
    # These should NOT collide anymore.

    k_split = generate_surrogate_key("NCT", "A", "B")
    k_joined = generate_surrogate_key("NCT", "A|B")

    assert k_split != k_joined


def test_duplicate_deduplication_in_transform() -> None:
    """Verify that identical source records are deduplicated in transform."""
    raw_study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_DUP_TEST"},
            "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
            "sponsorCollaboratorsModule": {
                "collaborators": [
                    {"name": "Same Lab", "class": "OTHER"},
                    {"name": "Same Lab", "class": "OTHER"},
                ]
            },
        }
    }

    result = transform_study(raw_study)
    sponsors = result["silver_sponsors"]

    # Deduplication enabled: should have 1 record
    assert len(sponsors) == 1

    # Check content
    assert sponsors[0]["name"] == "Same Lab"


def test_large_list_performance() -> None:
    """Verify handling of large 1:N lists."""
    # Create 1000 locations
    locations = []
    for i in range(1000):
        locations.append({"facility": f"Facility {i}", "city": "City", "country": "Country"})

    raw_study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_LARGE"},
            "contactsLocationsModule": {"locations": locations},
        }
    }

    result = transform_study(raw_study)
    silver_locs = result["silver_locations"]

    assert len(silver_locs) == 1000
    # verify unique IDs for distinct facilities
    ids = {loc["id"] for loc in silver_locs}
    assert len(ids) == 1000


def test_null_vs_empty_string_collision() -> None:
    """Verify None and Empty string might collide if not handled carefully."""
    # Current impl: (p or "") -> None becomes ""
    k1 = generate_surrogate_key("NCT", "A", None)
    k2 = generate_surrogate_key("NCT", "A", "")

    assert k1 == k2

    # This is likely acceptable/desired (None implies missing, Empty string implies missing)
    # But let's verify " " (space) differs
    k3 = generate_surrogate_key("NCT", "A", " ")
    assert k1 != k3


def test_date_boundary_conditions() -> None:
    """Test invalid or boundary date formats."""
    # Valid
    assert parse_date("2023-01-01") is not None

    # Invalid boundary dates
    assert parse_date("0000-00-00") is None
    assert parse_date("9999-99-99") is None
    assert parse_date("2023-13-01") is None  # Invalid month
    assert parse_date("2023-00-01") is None  # Invalid month
    assert parse_date("2023-01-32") is None  # Invalid day

    # Malformed strings
    assert parse_date("not-a-date") is None
    assert parse_date("") is None
    assert parse_date(None) is None
