# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

from coreason_etl_clinicaltrialsgov.transformers import generate_surrogate_key


def test_generate_surrogate_key_unsafe_parent_id() -> None:
    """Test that parent_id containing the delimiter is sanitized."""
    # If parent_id is "A|B" and part is "C", it should produce "A_B|C", not "A|B|C"
    # "A|B|C" would be indistinguishable from parent="A", parts="B", "C".

    # Case 1: Unsafe parent ID
    unsafe_key = generate_surrogate_key("A|B", "C")

    # Case 2: Safe equivalent
    # If we sanitize, "A|B" becomes "A_B".
    # So generate_surrogate_key("A|B", "C") should match generate_surrogate_key("A_B", "C")

    safe_key = generate_surrogate_key("A_B", "C")

    # Should match if sanitization works
    assert unsafe_key == safe_key

    # Verify logic explicitly: "A|B" -> "A_B"
    # Hash seed should be "A_B|C"

    # Check that it DOES NOT match the injection case
    # Injection case: parent="A", parts="B", "C" -> seed "A|B|C"
    injection_key = generate_surrogate_key("A", "B", "C")

    assert unsafe_key != injection_key


def test_generate_surrogate_key_all_none() -> None:
    """Test behavior when all parts are None."""
    # parent_id must be str, but parts can be None
    key = generate_surrogate_key("parent")
    # Seed: "parent"

    key_none = generate_surrogate_key("parent", None)
    # Seed: "parent|"

    assert key != key_none
