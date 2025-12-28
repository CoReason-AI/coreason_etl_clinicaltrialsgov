# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov


from coreason_etl_clinicaltrialsgov.transformers import normalize_age, parse_date


def test_normalize_age_unreachable_cases():
    # Attempt to hit the "except ValueError" block for split parts
    # normalize_age logic:
    # 1. parts = age_str.strip().lower().split()
    # 2. if not parts or len(parts) < 2:
    #       try: return float(parts[0]) except ...

    # We already tested "Ten Years" -> None (hits the len(parts) >= 2 path second try block)
    # We tested "10" -> 10.0 (hits the len(parts) < 2 path)
    # We tested "Years" -> None (hits the len(parts) < 2 path exception)

    # To hit the `except (ValueError, IndexError): return None` inside `if len(parts) < 2`:
    # We need parts[0] to fail float conversion.
    assert normalize_age("Invalid") is None

    # To hit `except ValueError: return None` after `len(parts) >= 2`:
    # We need parts[0] to fail float conversion.
    assert normalize_age("Ten Years") is None

    # To hit the final `return value` fallback (unknown unit)
    # "10 Decades" -> "decade" not in unit list
    # The code returns value (10.0)
    assert normalize_age("10 Decades") == 10.0

    # To hit line 32 in parse_date, we need to provide a string that raises ValueError.
    # parse_date handles this, but maybe we missed a case?
    # line 32 is `except ValueError: return None`

    # invalid month
    assert parse_date("2023-13-01") is None

    # invalid format that splits correctly but fails parsing
    # 2 parts
    assert parse_date("2023-13") is None
    # 1 part
    assert parse_date("Invalid") is None

    # line 66: IndexError in normalize_age?
    # parts[0] failing? `try: return float(parts[0]) except (ValueError, IndexError): return None`
    # parts is `age_str.strip().lower().split()`
    # If age_str is "", parts is empty list. `parts[0]` raises IndexError.
    assert normalize_age("") is None
    assert normalize_age("   ") is None

    # How to trigger ValueError on float(parts[0]) when len(parts) < 2?
    # "ABC" -> parts=["abc"], len=1. float("abc") -> ValueError.
    assert normalize_age("ABC") is None
