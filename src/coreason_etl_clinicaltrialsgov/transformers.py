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
from typing import Any, Optional

from coreason_etl_clinicaltrialsgov.schemas import GoldStudy

# --- Helpers ---


def get_enrollment_bucket(count: Optional[int]) -> Optional[str]:
    """Categorize enrollment count."""
    if count is None:
        return None
    if count < 100:
        return "Small"
    elif count < 1000:
        return "Medium"
    else:
        return "Large"


# --- Transformation Logic ---


def transform_gold(
    raw_study: dict[str, Any], silver_study: dict[str, Any], locations: list[dict[str, Any]]
) -> Optional[dict[str, Any]]:
    """Transform to Gold layer. Returns None if filtered out."""

    valid_statuses = {"RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED", "SUSPENDED", "TERMINATED"}
    overall_status = silver_study.get("overall_status")

    # Filter
    if not overall_status or overall_status.upper() not in valid_statuses:
        return None

    # Derived Columns
    start = silver_study.get("start_date")
    end = silver_study.get("completion_date")
    years_active = None
    if start and end:
        # Calculate years as float
        if isinstance(start, str):
            start = date.fromisoformat(start)
        if isinstance(end, str):
            end = date.fromisoformat(end)

        delta = end - start
        years_active = delta.days / 365.25

    enrollment = silver_study.get("enrollment_count")
    bucket = get_enrollment_bucket(enrollment)

    # Check raw for results section presence
    has_results = "resultsSection" in raw_study

    # Geo countries
    geo_countries = list({loc.get("country") for loc in locations if loc.get("country")})

    # Create Gold Model
    gold_model = GoldStudy(
        source_id=silver_study.get("source_id"),
        coreason_id=silver_study.get("coreason_id"),
        title=silver_study.get("title"),
        overall_status=overall_status,
        enrollment_bucket=bucket,
        years_active=years_active,
        has_results=has_results,
        geo_countries=geo_countries,
    )

    return gold_model.model_dump()
