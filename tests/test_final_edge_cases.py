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

import polars as pl

from coreason_etl_clinicaltrialsgov.transformers import transform_gold
from coreason_etl_clinicaltrialsgov.transformers_polars import transform_to_silver_sponsors


def test_surrogate_key_collision_delimiter_handling() -> None:
    """
    Test that delimiter sanitization (replacing '|' with '_') results in key collision
    for specifically crafted inputs, and that 'First Write Wins' logic prevails.

    Scenario:
    Record 1: Name="ACME_INC" (Sanitized -> "ACME_INC")
    Record 2: Name="ACME|INC" (Sanitized -> "ACME_INC")

    These two should collide. The first one encountered should be kept.
    """
    data = [
        # Record 1
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT_COLLISION"},
                "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
                "sponsorCollaboratorsModule": {"leadSponsor": {"name": "ACME_INC", "class": "INDUSTRY"}},
            }
        },
        # Record 2 (Same NCT ID to force same batch processing)
        # Note: transformers_polars flattens sponsors. To simulate collision in the SAME batch,
        # we can use the 'collaborators' list which is flattened.
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT_COLLISION_2"},
                "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
                "sponsorCollaboratorsModule": {
                    "collaborators": [
                        {"name": "ACME_INC", "class": "A"},
                        {"name": "ACME|INC", "class": "B"},  # Should collide with above
                    ]
                },
            }
        },
    ]

    lf = pl.DataFrame(data).lazy()
    df = transform_to_silver_sponsors(lf)
    results = df.filter(pl.col("source_id") == "NCT_COLLISION_2").to_dicts()

    # We expect strictly 1 record because keys are identical after sanitization
    # ID generation: uuid5(seed)
    # Seed 1: "NCT_COLLISION_2|COLLABORATOR|ACME_INC"
    # Seed 2: "NCT_COLLISION_2|COLLABORATOR|ACME_INC" (Because | replaced by _)

    assert len(results) == 1

    # "First Write Wins" means the first one in the list order is kept.
    # In the list, "ACME_INC" (class A) came before "ACME|INC" (class B).
    assert results[0]["agency_class"] == "A"


def test_gold_negative_dates() -> None:
    """
    Test logic when completion_date is earlier than start_date.
    The code calculates (end - start).days / 365.25.
    If end < start, this should yield a negative float.
    """
    silver_study = {
        "source_id": "NCT_NEG",
        "coreason_id": "UUID-NEG",
        "overall_status": "COMPLETED",
        "start_date": date(2023, 1, 1),
        "completion_date": date(2020, 1, 1),  # 3 years earlier
        "enrollment_count": 100,
    }

    gold = transform_gold({"resultsSection": {}}, silver_study, [])
    assert gold is not None

    # Expected: approx -3.0
    # (2020-01-01) - (2023-01-01) = -1096 days (includes leap year 2020)
    # -1096 / 365.25 = -3.00068

    assert gold["years_active"] < 0
    assert gold["years_active"] == -1096 / 365.25
