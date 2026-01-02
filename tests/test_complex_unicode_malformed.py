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

import polars as pl

from coreason_etl_clinicaltrialsgov.transformers_polars import (
    transform_to_silver_sponsors,
    transform_to_silver_studies,
)


def test_unicode_id_stability() -> None:
    """
    Verify that surrogate key generation is deterministic and stable
    for complex Unicode inputs (Emojis, CJK, RTL).
    """
    # 1. Emoji Sponsor Name
    emoji_name = "Sponsor 💊🧬"
    # 2. RTL Sponsor Name (Arabic)
    rtl_name = "شركة الأدوية"
    # 3. CJK Sponsor Name (Chinese)
    cjk_name = "医药公司"

    data: list[dict[str, Any]] = [
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT_UNI_1"},
                "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": emoji_name, "class": "IND"},
                    "collaborators": [
                        {"name": rtl_name, "class": "OTHER"},
                        {"name": cjk_name, "class": "OTHER"},
                    ],
                },
            }
        }
    ]

    lf = pl.DataFrame(data).lazy()
    df = transform_to_silver_sponsors(lf)
    results = df.sort("name").to_dicts()

    assert len(results) == 3

    # Check that IDs are generated (not null) and look like UUIDs
    for res in results:
        assert res["id"] is not None
        assert len(res["id"]) == 36  # UUID length
        assert res["coreason_id"] is not None

    # Verify distinctness
    ids = {r["id"] for r in results}
    assert len(ids) == 3

    # Verify deterministic behavior (run again with same input)
    df2 = transform_to_silver_sponsors(pl.DataFrame(data).lazy())
    results2 = df2.sort("name").to_dicts()
    ids2 = {r["id"] for r in results2}

    assert ids == ids2


def test_malformed_json_structure() -> None:
    """
    Verify that the transformers handle unexpected JSON structures gracefully.
    Polars `struct.field` might fail if the column type inferred is not Struct.
    However, we usually force inference or `safe_get_field` logic needs to handle mismatch.

    Scenario: `identificationModule` is a List instead of a Struct.
    """
    data = [
        {
            "protocolSection": {
                # MALFORMED: Should be a dict, but is a list here
                "identificationModule": [{"nctId": "NCT_BAD"}],
                "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
            }
        }
    ]

    # When creating DataFrame from list of dicts, Polars infers schema.
    # Here it will infer identificationModule as List(Struct).
    lf = pl.DataFrame(data).lazy()

    # The transformer expects identificationModule to be a Struct.
    # _safe_get_field checks schema.get(root_col).
    # If it is not a Struct, it should return null logic?
    # Let's see if _safe_get_field handles non-Struct root types when path traversal starts.

    # In transform_to_silver_studies:
    # get(["identificationModule", "nctId"], "nct_id")
    # -> _safe_get_field(lf, "protocolSection", ...)
    # -> protocolSection is Struct.
    # -> field "identificationModule" found? Yes.
    # -> is it Struct? No, it's List.

    # We expect the transformer NOT to crash, but to return nulls for those fields.
    df = transform_to_silver_studies(lf)
    res = df.to_dicts()

    # Since identificationModule was malformed (List instead of Struct),
    # nct_id extraction fails (returns None).
    # The transformer filters `nct_id` is_not_null(), so the record is dropped.
    assert len(res) == 0


def test_malformed_deep_structure() -> None:
    """
    Test where a deep field is the wrong type.
    """
    data = [
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT_OK"},
                "statusModule": {
                    "studyFirstPostDateStruct": {"date": "2023-01-01"},
                    "startDateStruct": "INVALID_STRING_NOT_STRUCT",  # Should be struct
                },
            }
        }
    ]

    lf = pl.DataFrame(data).lazy()
    df = transform_to_silver_studies(lf)
    res = df.to_dicts()

    assert len(res) == 1
    row = res[0]
    assert row["source_id"] == "NCT_OK"
    # start_date should be None because startDateStruct was not a struct
    assert row["start_date"] is None
