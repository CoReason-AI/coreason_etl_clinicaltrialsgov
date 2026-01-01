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

from coreason_etl_clinicaltrialsgov.transformers_polars import (
    transform_to_silver_outcomes,
    transform_to_silver_references,
    transform_to_silver_studies,
)


def test_silver_studies_date_edge_cases() -> None:
    """
    Test that invalid dates are handled gracefully (become null)
    and partial dates are parsed correctly.
    """
    data = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_DATES"},
            "statusModule": {
                "studyFirstPostDateStruct": {"date": "2023-01-01"},
                "startDateStruct": {"date": "2023-02-30"},  # Invalid Date
                "completionDateStruct": {"date": "2023-13-01"},  # Invalid Month
            },
        }
    }

    lf = pl.DataFrame([data]).lazy()
    df = transform_to_silver_studies(lf)
    result = df.to_dicts()[0]

    # Invalid dates should result in None
    assert result["start_date"] is None
    assert result["completion_date"] is None

    # Verify partial date handling (YYYY-MM -> YYYY-MM-01)
    data_partial = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_PARTIAL"},
            "statusModule": {
                "studyFirstPostDateStruct": {"date": "2023-01"},
                "startDateStruct": {"date": "2023-05"},
                "completionDateStruct": {"date": "2023"},
            },
        }
    }
    lf_p = pl.DataFrame([data_partial]).lazy()
    df_p = transform_to_silver_studies(lf_p)
    res_p = df_p.to_dicts()[0]

    assert res_p["start_date"] == date(2023, 5, 1)
    assert res_p["completion_date"] == date(2023, 1, 1)


def test_silver_studies_age_parsing_complex() -> None:
    """
    Test various formats of age strings.
    """
    # Note: _normalize_age_udf logic:
    # "18 Years" -> 18.0
    # "18" -> 18.0 (default no unit check logic, just split[0])
    # "Months" -> error/None?
    # "1.5 Years" -> 1.5

    data: list[dict[str, Any]] = [
        {"raw": "18 Years", "exp": 18.0},
        {"raw": "1.5 Years", "exp": 1.5},
        {"raw": "24 Months", "exp": 2.0},  # 24/12
        {"raw": "52 Weeks", "exp": 1.0},  # 52/52
        {"raw": "365 Days", "exp": 1.0},  # 365/365
        {"raw": "18", "exp": 18.0},  # Fallback to value
        {"raw": "Invalid", "exp": None},  # Parsing fail
        {"raw": "Years", "exp": None},  # Parsing fail
    ]

    rows = []
    for i, d in enumerate(data):
        rows.append(
            {
                "protocolSection": {
                    "identificationModule": {"nctId": f"NCT_{i}"},
                    "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
                    "eligibilityModule": {"minimumAge": d["raw"]},
                }
            }
        )

    lf = pl.DataFrame(rows).lazy()
    df = transform_to_silver_studies(lf)
    results = df.sort("source_id").to_dicts()

    for i, res in enumerate(results):
        expected = data[i]["exp"]
        assert res["min_age"] == expected, f"Failed for {data[i]['raw']}"


def test_silver_outcomes_sparse_data() -> None:
    """
    Test generation of silver outcomes with sparse/missing fields.
    """
    data = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_OUT"},
            "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
            "outcomesModule": {
                "primaryOutcomes": [
                    {
                        "measure": "Measure A",
                        # Missing timeFrame, description
                    }
                ],
                "secondaryOutcomes": [
                    {
                        # Missing measure
                        "timeFrame": "1 Year"
                    }
                ],
            },
        }
    }

    lf = pl.DataFrame([data]).lazy()
    df = transform_to_silver_outcomes(lf)
    results = df.to_dicts()

    # Should have 2 outcomes
    assert len(results) == 2

    primary = next(r for r in results if r["outcome_type"] == "PRIMARY")
    assert primary["measure"] == "Measure A"
    assert primary["time_frame"] is None

    secondary = next(r for r in results if r["outcome_type"] == "SECONDARY")
    assert secondary["measure"] is None
    assert secondary["time_frame"] == "1 Year"

    # IDs should be generated without error (handling Nones)
    assert primary["id"] is not None
    assert secondary["id"] is not None


def test_silver_references_retraction() -> None:
    """
    Test extraction of nested retraction struct in references.
    """
    data = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_REF"},
            "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
            "referencesModule": {
                "references": [
                    {
                        "pmid": "12345",
                        "citation": "Cit A",
                        "retraction": {
                            "retraction": True,
                            "ignored_field": "Should not be picked up by pl.Struct definition?",
                        },
                    },
                    {
                        "pmid": 67890,  # Int pmid
                        "citation": "Cit B",
                    },
                ]
            },
        }
    }

    # strict=False allows mixed types (e.g. String vs Int) to be upcast (usually to String)
    lf = pl.DataFrame([data], strict=False).lazy()
    df = transform_to_silver_references(lf)
    results = df.to_dicts()

    assert len(results) == 2

    ref_a = next(r for r in results if r["citation"] == "Cit A")
    # Verify Retraction
    # Since we define schema in pl.Struct, Polars might enforce it strictly or allow extra?
    # Logic: pl.Struct([pl.Field("retraction", pl.Boolean)])
    # If JSON has extra, Polars usually ignores extra if reading from JSON,
    # but here we are reading from a Dictionary/Struct column.

    # The extraction preserves the full dictionary for the 'retraction' field
    # because it is defined as dict[str, Any] and Polars struct.field extraction
    # returns the underlying struct value as-is.
    assert ref_a["retraction"] == {
        "retraction": True,
        "ignored_field": "Should not be picked up by pl.Struct definition?",
    }
    assert ref_a["pmid"] == "12345"

    ref_b = next(r for r in results if r["citation"] == "Cit B")
    # Int PMID should be cast to String
    assert ref_b["pmid"] == "67890"
