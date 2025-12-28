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
from coreason_etl_clinicaltrialsgov.transformers import transform_gold


def test_transform_gold_string_date_handling() -> None:
    # Simulate a scenario where silver_study has strings instead of dates (e.g. after round trip serialization)
    silver_study = {
        "source_id": "1",
        "coreason_id": "uuid",
        "title": "Title",
        "overall_status": "RECRUITING",
        "start_date": "2023-01-01",
        "completion_date": "2024-01-01",
        "enrollment_count": 50,
    }
    locations: list[dict[str, Any]] = []

    # This should trigger the isinstance(..., str) check in transform_gold
    gold = transform_gold({"resultsSection": {}}, silver_study, locations)
    assert gold is not None
    assert gold["years_active"] == pytest.approx(1.0, rel=0.01)


def test_transform_gold_date_object_handling() -> None:
    # Simulate correct behavior where objects are preserved
    silver_study = {
        "source_id": "1",
        "coreason_id": "uuid",
        "title": "Title",
        "overall_status": "RECRUITING",
        "start_date": date(2023, 1, 1),
        "completion_date": date(2024, 1, 1),
        "enrollment_count": 50,
    }
    locations: list[dict[str, Any]] = []

    gold = transform_gold({"resultsSection": {}}, silver_study, locations)
    assert gold is not None
    assert gold["years_active"] == pytest.approx(1.0, rel=0.01)
