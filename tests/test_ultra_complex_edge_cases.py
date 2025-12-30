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
    normalize_age,
    parse_date,
    transform_study,
)


def test_surrogate_key_unicode() -> None:
    """Test surrogate key generation with Unicode characters (Emoji, CJK)."""
    # 1. Emoji in name
    key_emoji = generate_surrogate_key("NCT123", "LEAD", "Pharma 💊 Corp")
    # 2. CJK in name
    key_cjk = generate_surrogate_key("NCT123", "LEAD", "中国製薬")

    assert key_emoji
    assert key_cjk
    assert key_emoji != key_cjk
    # Deterministic check
    assert key_emoji == generate_surrogate_key("NCT123", "LEAD", "Pharma 💊 Corp")


def test_surrogate_key_delimiter_injection() -> None:
    """Test handling of internal delimiter injection in input strings."""
    # The generator uses "|" as delimiter.
    # Case A: "Type", "Name"
    key_a = generate_surrogate_key("NCT", "Type", "Name")

    # Case B: "Type|Name", "" (malicious input trying to mimic Case A if naive join used)
    # If implemented correctly, input "Type|Name" should be sanitized (e.g. replaced with _)
    # resulting in seed "NCT|Type_Name|" vs "NCT|Type|Name"
    key_b = generate_surrogate_key("NCT", "Type|Name", "")

    assert key_a != key_b


def test_parse_date_historical() -> None:
    """Test parsing of dates before 1900."""
    # Python's fromisoformat supports year 0001+, but strict formats are key.
    d = parse_date("1850-01-01")
    assert d == date(1850, 1, 1)

    d2 = parse_date("0001-01-01")
    assert d2 == date(1, 1, 1)


def test_normalize_age_boundaries() -> None:
    """Test boundary conditions for age normalization."""
    # 0 Days
    assert normalize_age("0 Days") == 0.0

    # Very small fraction
    # 1 Minute is not standard, but let's check parsing of "0.5 Days"
    val = normalize_age("0.5 Days")
    assert val is not None
    assert val == 0.5 / 365.0

    # Whitespace handling
    assert normalize_age("  10   Years  ") == 10.0


def test_transform_study_empty_strings() -> None:
    """Test transformation with empty strings in key fields."""
    raw_study: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_EMPTY"},
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "", "class": "INDUSTRY"},  # Empty name
            },
            "armsInterventionsModule": {
                "interventions": [
                    {"type": "DRUG", "name": "  ", "description": "Whitespace name"},
                ]
            },
        }
    }

    result = transform_study(raw_study)

    # Check Sponsors
    sponsors = result["silver_sponsors"]
    # Should still create a record, key will rely on empty string/whitespace
    assert len(sponsors) == 1
    assert sponsors[0]["name"] == ""

    # Check Interventions
    interventions = result["silver_interventions"]
    assert len(interventions) == 1
    assert interventions[0]["name"] == "  "


def test_silver_reference_excluded_fields() -> None:
    """Verify SilverReference strictly excludes banned fields."""
    # Note: Pydantic model definition test.
    # The schema SilverReference should NOT have seeAlsoLinks, label, url.
    from coreason_etl_clinicaltrialsgov.schemas import SilverReference

    fields = SilverReference.model_fields.keys()
    assert "seeAlsoLinks" not in fields
    assert "label" not in fields
    assert "url" not in fields

    # Verify extra fields are ignored during instantiation if data is passed
    data = {
        "id": "123",
        "source_id": "src",
        "coreason_id": "core",
        "type": "REF",
        "seeAlsoLinks": ["http://bad.com"],
        "label": "Bad Label",
    }
    ref = SilverReference(**data)
    # Dump should not contain them
    dump = ref.model_dump()
    assert "seeAlsoLinks" not in dump
