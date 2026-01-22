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


def test_sponsor_extended_whitespace_and_types() -> None:
    """
    Test extended edge cases for sponsor deduplication:
    1. Vertical whitespace (newlines, tabs).
    2. Non-breaking spaces.
    3. Type coercion (Int to String) and subsequent whitespace stripping.
    4. Empty string vs None handling.
    """
    data = [
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT_EXTENDED"},
                "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "Lead Entity", "class": "IND"},
                    "collaborators": [
                        # 1. Newlines and Tabs
                        {"name": "\n\tTabbed Corp\r\n", "class": "OTHER"},
                        {"name": "Tabbed Corp", "class": "OTHER"},  # Should dedup with above
                        # 2. Non-breaking space \u00A0
                        {
                            "name": "NBSP\u00a0Inc",
                            "class": "OTHER",
                        },  # Inner NBSP - strip chars default doesn't touch inner
                        {"name": "\u00a0Outer NBSP\u00a0", "class": "OTHER"},
                        {"name": "Outer NBSP", "class": "OTHER"},  # Should dedup with above if \u00A0 is stripped
                        # 3. Numeric Name (Type Coercion)
                        {"name": 12345, "class": "OTHER"},
                        {"name": " 12345 ", "class": "OTHER"},  # Should dedup with above
                        # 4. Empty/None
                        {"name": "", "class": "OTHER"},
                        {"name": "   ", "class": "OTHER"},  # Should dedup with empty
                    ],
                },
            }
        }
    ]

    # We rely on Polars to infer a common schema (likely String because of mixed Int/String)
    lf = pl.DataFrame(data, strict=False).lazy()
    df = transform_to_silver_sponsors(lf)
    results = df.to_dicts()

    # Analyze

    # 1. Tabbed Corp
    tabbed = [r for r in results if r["name"] and "Tabbed Corp" in r["name"]]
    assert len(tabbed) == 1
    # Verify strict stripping
    assert tabbed[0]["name"] == "Tabbed Corp"

    # 2. NBSP
    # Outer NBSP (stripped)
    outer = [r for r in results if r["name"] and "Outer NBSP" in r["name"]]
    assert len(outer) == 1
    assert outer[0]["name"] == "Outer NBSP"

    # Inner NBSP (preserved)
    inner = [r for r in results if r["name"] and "NBSP\u00a0Inc" in r["name"]]
    assert len(inner) == 1
    assert inner[0]["name"] == "NBSP\u00a0Inc"

    # 3. Numeric
    # 12345 should be coerced to string "12345" and match " 12345 " stripped
    nums = [r for r in results if r["name"] and "12345" in r["name"]]
    assert len(nums) == 1
    assert nums[0]["name"] == "12345"

    # 4. Empty/Whitespace
    # "" and "   " should collapse to ""
    # Filter for empty names (but not None)
    empties = [r for r in results if r["name"] == ""]
    assert len(empties) == 1

    # Verify total count
    # Lead (1) + Tabbed (1) + Outer (1) + Inner (1) + Num (1) + Empty (1) = 6
    assert len(results) == 6
