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

import polars as pl
import pytest
from pydantic import ValidationError

from coreason_etl_clinicaltrialsgov.schemas import SilverStudy
from coreason_etl_clinicaltrialsgov.transformers import transform_gold
from coreason_etl_clinicaltrialsgov.transformers_polars import transform_to_silver_studies


def test_gold_negative_duration() -> None:
    """Verify years_active handles negative duration (completion < start)."""
    silver_study: dict[str, Any] = {
        "source_id": "NCT_NEG",
        "coreason_id": "UUID-NEG",
        "overall_status": "COMPLETED",
        "start_date": date(2023, 1, 1),
        "completion_date": date(2022, 1, 1),  # Ends before it starts
        "enrollment_count": 100,
    }
    gold = transform_gold({"resultsSection": {}}, silver_study, [])

    assert gold is not None
    assert gold["years_active"] is not None
    # Difference is -365 days. -365 / 365.25 ~= -0.999
    assert gold["years_active"] < 0
    assert gold["years_active"] == pytest.approx(-1.0, rel=0.01)


def test_gold_missing_dates() -> None:
    """Verify years_active is None if dates are missing."""
    # 1. Missing start
    s1: dict[str, Any] = {
        "source_id": "NCT_M1",
        "coreason_id": "U1",
        "overall_status": "COMPLETED",
        "completion_date": date(2023, 1, 1),
    }
    g1 = transform_gold({}, s1, [])
    assert g1 is not None
    assert g1["years_active"] is None

    # 2. Missing completion
    s2: dict[str, Any] = {
        "source_id": "NCT_M2",
        "coreason_id": "U2",
        "overall_status": "COMPLETED",
        "start_date": date(2023, 1, 1),
    }
    g2 = transform_gold({}, s2, [])
    assert g2 is not None
    assert g2["years_active"] is None


def test_silver_schema_evolution() -> None:
    """Verify pipeline ignores extra/new fields in source JSON."""
    data = [
        {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT_EXTRA",
                    "newFieldThatDidNotExist": "SomeValue",  # Extra field
                },
                "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
            }
        }
    ]
    lf = pl.DataFrame(data).lazy()
    df = transform_to_silver_studies(lf)
    assert df.height == 1
    # Check that Pydantic validation passes (it ignores extra fields by default in our config)
    record = df.to_dicts()[0]
    model = SilverStudy.model_validate(record)
    assert model.source_id == "NCT_EXTRA"
    # Ensure no crash


def test_silver_type_mismatch_coercion() -> None:
    """Verify handling of type mismatches (String passed to Int field)."""
    # Case 1: Coercible String ("100" -> 100)
    # Note: Polars transform explicitly maps to pl.Int64() for enrollment_count
    # If source is string "100", Polars should cast it if possible.
    # Let's test the behavior.
    data = [
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT_COERCE"},
                "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
                "designModule": {
                    "enrollmentInfo": {"count": "100"}  # String, expected Int
                },
            }
        }
    ]
    lf = pl.DataFrame(data).lazy()
    # In `transform_to_silver_studies`, we do:
    # get(..., "enrollment_count", pl.Int64())
    # This uses pl.lit(None, pl.Int64) IF the field is missing.
    # But if field exists, it does: pl.col("count").
    # If pl.col("count") is String, it stays String in the LazyFrame expression unless cast.
    # The return_dtype in `_safe_get_field` is only used for the DEFAULT (missing) value.
    # So `enrollment_count` will be String "100".
    # Pydantic `enrollment_count: Optional[int]` SHOULD coerce "100" to 100.

    df = transform_to_silver_studies(lf)
    record = df.to_dicts()[0]
    assert record["enrollment_count"] == "100"  # It remains string in DF

    # Validate with Pydantic
    model = SilverStudy.model_validate(record)
    assert model.enrollment_count == 100  # Pydantic coerces it!


def test_silver_type_mismatch_failure() -> None:
    """Verify handling when type coercion fails (e.g. 'Ten' -> Int)."""
    data = [
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT_FAIL"},
                "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
                "designModule": {
                    "enrollmentInfo": {"count": "Ten"}  # Not an int
                },
            }
        }
    ]
    lf = pl.DataFrame(data).lazy()
    df = transform_to_silver_studies(lf)
    record = df.to_dicts()[0]
    assert record["enrollment_count"] == "Ten"

    # Pydantic should raise ValidationError
    with pytest.raises(ValidationError) as excinfo:
        SilverStudy.model_validate(record)

    assert "enrollment_count" in str(excinfo.value)
    assert "Input should be a valid integer" in str(excinfo.value)
