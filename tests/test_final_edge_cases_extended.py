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
from typer.testing import CliRunner

from coreason_etl_clinicaltrialsgov.main import app
from coreason_etl_clinicaltrialsgov.transformers_polars import (
    _normalize_age_udf,
    transform_to_silver_studies,
)

runner = CliRunner()


def test_age_parsing_singular_and_boundary() -> None:
    """
    Test singular units, boundary values, and case insensitivity for age parsing.
    """
    # Direct UDF testing for granular verification
    assert _normalize_age_udf("1 Year") == 1.0
    assert _normalize_age_udf("1 year") == 1.0
    assert _normalize_age_udf("1 YEAR") == 1.0

    assert _normalize_age_udf("1 Month") == 1.0 / 12.0
    assert _normalize_age_udf("1 month") == 1.0 / 12.0

    assert _normalize_age_udf("1 Week") == 1.0 / 52.0
    assert _normalize_age_udf("1 week") == 1.0 / 52.0

    assert _normalize_age_udf("365 Day") == 1.0
    assert _normalize_age_udf("365 day") == 1.0

    # Boundary
    assert _normalize_age_udf("0 Years") == 0.0
    # Negative values technically parse as floats, even if logically weird for age
    assert _normalize_age_udf("-5 Years") == -5.0


def test_phase_flattening_order() -> None:
    """
    Test that phases are flattened and sorted alphabetically.
    """
    data = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_PHASES"},
            "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
            "designModule": {"phases": ["Phase 2", "Phase 1", "Early Phase 1"]},
        }
    }

    lf = pl.DataFrame([data]).lazy()
    df = transform_to_silver_studies(lf)
    res = df.to_dicts()[0]

    # Expected sort: "Early Phase 1", "Phase 1", "Phase 2"
    expected = "Early Phase 1|Phase 1|Phase 2"
    assert res["phases"] == expected


def test_cli_negative_page_size() -> None:
    """
    Test that the CLI rejects negative page sizes.
    """
    result = runner.invoke(app, ["run", "--page-size", "-10"], env={"NO_COLOR": "1"})

    # Current implementation allows int, but API logic handles it or it passes through.
    # We want to enforce strictness here.
    # If the app doesn't validate, this test will FAIL (exit code 0 or 1 depending on downstream),
    # prompting us to implement validation.

    # We expect a validation error from Typer/Callback or custom logic
    assert result.exit_code != 0
    assert "Invalid value" in result.output or "must be positive" in result.output
