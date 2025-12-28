# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov


from coreason_etl_clinicaltrialsgov.transformers import normalize_age, transform_gold


def test_normalize_age_edge_cases():
    # Test "X Year" singular
    assert normalize_age("1 Year") == 1.0
    # Test just number string
    assert normalize_age("10") == 10.0
    # Test weird spacing
    assert normalize_age("  20   Years ") == 20.0
    # Test failing float conversion
    assert normalize_age("Ten Years") is None
    # Test empty string
    assert normalize_age("") is None
    # Test single part non-number
    assert normalize_age("Years") is None


def test_transform_gold_no_status():
    # If overall_status is missing
    silver_study = {"source_id": "1", "overall_status": None}
    assert transform_gold({}, silver_study, []) is None


def test_transform_gold_partial_dates():
    # If start or end date is missing, years_active is None
    silver_study = {"source_id": "1", "overall_status": "RECRUITING", "start_date": None, "completion_date": None}
    gold = transform_gold({"resultsSection": {}}, silver_study, [])
    assert gold["years_active"] is None
