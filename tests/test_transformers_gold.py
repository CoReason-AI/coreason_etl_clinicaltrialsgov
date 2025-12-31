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

import pytest

from coreason_etl_clinicaltrialsgov.transformers import get_enrollment_bucket, transform_gold


@pytest.mark.parametrize(
    "count, expected",
    [
        (50, "Small"),
        (500, "Medium"),
        (1500, "Large"),
        (None, None),
    ],
)
def test_get_enrollment_bucket(count: int | None, expected: str | None) -> None:
    assert get_enrollment_bucket(count) == expected


def test_transform_gold_success() -> None:
    raw_study: dict[str, Any] = {"resultsSection": {}}  # Indicates has_results=True

    silver_study: dict[str, Any] = {
        "source_id": "NCT00000001",
        "coreason_id": "uuid-123",
        "title": "Test Study",
        "overall_status": "RECRUITING",
        "start_date": date(2023, 1, 1),
        "completion_date": date(2024, 1, 1),
        "enrollment_count": 100,
    }

    locations: list[dict[str, Any]] = [
        {"country": "United States"},
        {"country": "Canada"},
    ]

    gold = transform_gold(raw_study, silver_study, locations)

    assert gold is not None
    assert gold["source_id"] == "NCT00000001"
    assert gold["overall_status"] == "RECRUITING"
    assert gold["enrollment_bucket"] == "Medium"  # 100
    assert gold["has_results"] is True
    assert set(gold["geo_countries"]) == {"United States", "Canada"}
    # 2023-01-01 to 2024-01-01 is 1 year (approx 365 days)
    # 365 / 365.25 ~= 0.999
    assert gold["years_active"] == pytest.approx(1.0, rel=0.01)


def test_transform_gold_filtered_status() -> None:
    raw_study: dict[str, Any] = {}
    silver_study: dict[str, Any] = {
        "source_id": "NCT00000001",
        "overall_status": "WITHDRAWN",  # Invalid status
    }
    locations: list[dict[str, Any]] = []

    gold = transform_gold(raw_study, silver_study, locations)
    assert gold is None


def test_transform_gold_string_dates() -> None:
    # Verify it handles string dates as input

    raw_study: dict[str, Any] = {}
    silver_study: dict[str, Any] = {
        "source_id": "NCT00000001",
        "coreason_id": "uuid-123",
        "overall_status": "COMPLETED",
        "start_date": "2023-01-01",
        "completion_date": "2024-01-01",
        "enrollment_count": 50,
    }
    locations: list[dict[str, Any]] = []

    gold = transform_gold(raw_study, silver_study, locations)
    assert gold is not None
    assert gold["years_active"] == pytest.approx(1.0, rel=0.01)
