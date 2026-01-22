# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

import polars as pl

from coreason_etl_clinicaltrialsgov.transformers_polars import transform_to_silver_sponsors


def test_sponsor_deduplication_whitespace() -> None:
    """Verify that sponsors with same name differing only by whitespace are deduplicated."""
    data = [
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT001"},
                "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "Big Pharma", "class": "INDUSTRY"},
                    "collaborators": [
                        {"name": "  Small Biotech ", "class": "OTHER"},
                        {"name": "Small Biotech", "class": "OTHER_GOV"},  # Duplicate name, different class
                    ],
                },
            }
        }
    ]

    lf = pl.DataFrame(data).lazy()
    df = transform_to_silver_sponsors(lf)

    records = df.to_dicts()

    # We expect:
    # 1. Big Pharma (LEAD)
    # 2. Small Biotech (COLLABORATOR) - only one entry

    assert len(records) == 2

    names = {r["name"] for r in records}
    assert "Big Pharma" in names
    # The name in the output preserves the input of the kept record (usually first)
    # OR we might want to normalize the name in the output too?
    # The current logic only normalizes for ID generation.
    # So "  Small Biotech " (first) is kept.

    # Let's check IDs
    ids = {r["id"] for r in records}
    assert len(ids) == 2

    # Check that we kept the first one ("  Small Biotech ") or normalized?
    # The transformer does not normalize the 'name' column, only the 'id' generation uses normalization.
    # So we expect the original name of the first record.

    collabs = [r for r in records if r["role"] == "COLLABORATOR"]
    assert len(collabs) == 1
    # Name should now be stripped
    assert collabs[0]["name"] == "Small Biotech"
    assert collabs[0]["agency_class"] == "OTHER"  # First one wins


def test_sponsor_deduplication_exact_match() -> None:
    """Verify standard deduplication for exact matches."""
    data = [
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT002"},
                "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "Lead", "class": "IND"},
                    "collaborators": [
                        {"name": "Collab A", "class": "OTHER"},
                        {"name": "Collab A", "class": "OTHER"},
                    ],
                },
            }
        }
    ]

    lf = pl.DataFrame(data).lazy()
    df = transform_to_silver_sponsors(lf)
    records = df.to_dicts()

    assert len(records) == 2  # Lead + 1 Collab
