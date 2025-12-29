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


def test_generate_surrogate_key_determinism() -> None:
    # Same inputs -> same key
    k1 = generate_surrogate_key("NCT123", "role", "name")
    k2 = generate_surrogate_key("NCT123", "role", "name")
    assert k1 == k2


def test_generate_surrogate_key_uniqueness() -> None:
    # Different inputs -> different key
    k1 = generate_surrogate_key("NCT123", "role", "name1")
    k2 = generate_surrogate_key("NCT123", "role", "name2")
    assert k1 != k2


def test_generate_surrogate_key_none_handling() -> None:
    # Handling None
    k1 = generate_surrogate_key("NCT123", "role", None)
    k2 = generate_surrogate_key("NCT123", "role", "")
    # Should treat None as empty string for stability
    # Our implementation: seed_parts.append(p or "")
    assert k1 == k2


def test_generate_surrogate_key_ordering() -> None:
    # Order matters
    k1 = generate_surrogate_key("NCT123", "A", "B")
    k2 = generate_surrogate_key("NCT123", "B", "A")
    assert k1 != k2
