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

from coreason_etl_clinicaltrialsgov.schemas import GoldStudy
from coreason_etl_clinicaltrialsgov.transformers import transform_gold


def test_gold_study_schema_validation() -> None:
    """Verify that GoldStudy schema enforces types correctly."""
    silver_study = {
        "source_id": "NCT123",
        "coreason_id": "UUID-123",
        "title": "Title",
        "overall_status": "RECRUITING",
        "start_date": date(2023, 1, 1),
        "completion_date": date(2024, 1, 1),
        "enrollment_count": 50,
    }
    locations: list[dict[str, Any]] = [{"country": "United States"}]
    raw_study: dict[str, Any] = {"resultsSection": {}}

    result = transform_gold(raw_study, silver_study, locations)
    assert result is not None

    # Validate with Pydantic model again to ensure strict compliance
    model = GoldStudy(**result)
    assert model.source_id == "NCT123"
    assert model.geo_countries == ["United States"]
    assert model.years_active is not None
    assert model.enrollment_bucket == "Small"
    assert model.has_results is True


def test_gold_study_schema_invalid_status() -> None:
    """Verify filtering still works."""
    silver_study = {
        "source_id": "NCT123",
        "coreason_id": "UUID-123",
        "title": "Title",
        "overall_status": "WITHDRAWN",  # Invalid for Gold
    }
    locations: list[dict[str, Any]] = []
    raw_study: dict[str, Any] = {}

    result = transform_gold(raw_study, silver_study, locations)
    assert result is None
