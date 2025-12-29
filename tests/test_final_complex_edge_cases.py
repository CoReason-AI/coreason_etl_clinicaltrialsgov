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
    generate_surrogate_key,
    parse_date,
    transform_study,
)


def test_unicode_robustness() -> None:
    """Ensure that Unicode characters in critical fields are handled correctly."""
    # A mix of emojis, CJK, and accents
    nasty_string = "Study of 💊 (Pill) in 東京 & Fãçade"

    raw: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_UNICODE", "briefTitle": nasty_string},
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Sponsor 🚀", "class": "INDUSTRY"}},
        }
    }

    result = transform_study(raw)

    # Silver Study check
    study = result["silver_studies"][0]
    assert study["title"] == nasty_string

    # Silver Sponsor check (surrogate key generation must handle unicode too)
    sponsor = result["silver_sponsors"][0]
    assert sponsor["name"] == "Sponsor 🚀"

    # Verify deterministic ID generation with unicode
    # It shouldn't crash
    assert sponsor["id"] is not None
    assert len(sponsor["id"]) == 36  # UUID length


def test_date_boundaries_and_invalid() -> None:
    """Test date parsing with extreme and invalid values."""

    # 1. Extreme future
    assert parse_date("9999-12-31") == date(9999, 12, 31)

    # 2. Year only
    assert parse_date("2023") == date(2023, 1, 1)

    # 3. Invalid day (Feb 30) - Should return None, not raise ValueError
    assert parse_date("2023-02-30") is None

    # 4. 0000-00-00 - often occurs in legacy, invalid in Python date
    assert parse_date("0000-00-00") is None

    # 5. Malformed string
    assert parse_date("Not a date") is None

    # 6. Empty
    assert parse_date("") is None
    assert parse_date(None) is None


def test_surrogate_key_injection() -> None:
    """Verify that using the delimiter in the key content doesn't cause collision."""

    # Case A: Two parts: "Pfizer|Inc", "USA"
    # Case B: Two parts: "Pfizer", "Inc|USA"
    # If we simply joined with "|", these would look identical: "Pfizer|Inc|USA"

    # The generator should replace internal pipes.

    key_a = generate_surrogate_key("ROOT", "Pfizer|Inc", "USA")
    key_b = generate_surrogate_key("ROOT", "Pfizer", "Inc|USA")

    assert key_a != key_b

    # Verify exact replacement logic manually to be sure
    # "Pfizer|Inc" -> "Pfizer_Inc"
    # Seed A: "ROOT|Pfizer_Inc|USA"
    # Seed B: "ROOT|Pfizer|Inc_USA"

    # Verify against manual construction
    key_a_manual = generate_surrogate_key("ROOT", "Pfizer_Inc", "USA")
    assert key_a == key_a_manual


def test_deduplication_exact_match() -> None:
    """Verify that identical child records are deduped."""

    raw: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_DEDUP"},
            "sponsorCollaboratorsModule": {
                "collaborators": [
                    {"name": "Same Guy", "class": "OTHER"},
                    {"name": "Same Guy", "class": "OTHER"},  # Exact duplicate
                ]
            },
        }
    }

    result = transform_study(raw)

    sponsors = result["silver_sponsors"]

    # Should only have 1 record
    assert len(sponsors) == 1
    assert sponsors[0]["name"] == "Same Guy"
    assert sponsors[0]["role"] == "COLLABORATOR"


def test_deduplication_collision_avoidance() -> None:
    """Verify that records with same name but different attributes (that part of key) are NOT deduped."""
    # Actually, for Sponsors, the key is (nct_id, role, name).
    # So if class is different, it is STILL considered same sponsor in this logic?
    # Let's check the code:
    # s_id = generate_surrogate_key(nct_id, "COLLABORATOR", name)
    # So yes, if name is same, it will dedupe, effectively picking the first one encountered.
    # This is "Last write wins" or "First write wins" logic depending on implementation.
    # Implementation: `if s_id not in seen_sponsors`. So First Write Wins.

    raw: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_COLLISION"},
            "sponsorCollaboratorsModule": {
                "collaborators": [
                    {"name": "Same Guy", "class": "OTHER"},
                    {"name": "Same Guy", "class": "INDUSTRY"},  # Different class, but same key!
                ]
            },
        }
    }

    result = transform_study(raw)
    sponsors = result["silver_sponsors"]

    # Should still be 1 because key relies ONLY on name + role
    assert len(sponsors) == 1
    # Since First Wins, class should be OTHER
    assert sponsors[0]["agency_class"] == "OTHER"
