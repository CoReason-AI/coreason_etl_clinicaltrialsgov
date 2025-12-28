# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

from coreason_etl_clinicaltrialsgov.transformers import parse_date


def test_parse_date_coverage() -> None:
    # Hit line 41: except ValueError: return None (outer try)
    # The outer try block wraps `parts = date_str.split("-")` and the if/elif logic.
    # To hit ValueError here, one of the `date.fromisoformat` calls must fail.

    # 3 parts, but invalid numbers
    assert parse_date("2023-13-32") is None

    # 2 parts, but invalid month
    assert parse_date("2023-13") is None

    # 1 part, but invalid year
    assert parse_date("NotAYear") is None

    # 4 parts, falls through if/elif chain and hits final return None
    assert parse_date("2023-01-01-01") is None
