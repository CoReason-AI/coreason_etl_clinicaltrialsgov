# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

from typing import Any

import pytest
from pydantic import ValidationError

from coreason_etl_clinicaltrialsgov.schemas import (
    SilverIntervention,
    SilverLocation,
    SilverStudy,
)


def test_schema_evolution_extra_fields() -> None:
    """
    Test that extra fields (representing API schema evolution) are ignored
    and do not cause validation errors.
    """
    data: dict[str, Any] = {
        "source_id": "NCT001",
        "coreason_id": "CID1",
        "title": "Test Study",
        "new_api_field": "ShouldBeIgnored",
        "another_extra": 123,
    }

    # Should not raise ValidationError
    model = SilverStudy.model_validate(data)
    dumped = model.model_dump()

    # Verify extra fields are NOT in the output
    assert "new_api_field" not in dumped
    assert "another_extra" not in dumped
    # Verify core fields match
    assert dumped["source_id"] == "NCT001"


def test_nested_validation_geopoint() -> None:
    """
    Test validation failures in nested structures (GeoPoint).
    """
    # 1. Invalid type for nested field
    data: dict[str, Any] = {
        "id": "L1",
        "source_id": "NCT001",
        "coreason_id": "CID1",
        "geo_point": {"lat": "NotANumber", "lon": -74.0},
    }

    with pytest.raises(ValidationError) as exc:
        SilverLocation.model_validate(data)

    # Check that error path points to geo_point -> lat
    err_str = str(exc.value)
    assert "geo_point" in err_str
    assert "lat" in err_str


def test_list_validation_interventions() -> None:
    """
    Test validation in lists (SilverIntervention.other_names).
    """
    # 1. List containing invalid type (e.g., None if not allowed, or complex obj)
    # List[str] usually coerces ints/floats to strings.
    # Pass a dict instead of string to force error.
    data: dict[str, Any] = {
        "id": "I1",
        "source_id": "NCT001",
        "coreason_id": "CID1",
        "name": "Drug X",
        "other_names": ["Brand A", {"invalid": "obj"}],
    }

    with pytest.raises(ValidationError) as exc:
        SilverIntervention.model_validate(data)

    assert "other_names" in str(exc.value)


def test_type_coercion_limits() -> None:
    """
    Verify type coercion behavior boundaries.
    """
    # 1. Numeric string to int (Should work)
    data_valid: dict[str, Any] = {
        "source_id": "NCT001",
        "coreason_id": "CID1",
        "enrollment_count": "100",
    }
    m = SilverStudy.model_validate(data_valid)
    assert m.enrollment_count == 100

    # 2. Boolean string (Pydantic V2 is strict on some bools, usually accepts 'true', '1')
    data_bool: dict[str, Any] = {
        "source_id": "NCT001",
        "coreason_id": "CID1",
        "accepted_healthy_volunteers": "true",
    }
    m2 = SilverStudy.model_validate(data_bool)
    assert m2.accepted_healthy_volunteers is True

    # 3. Invalid int string
    data_invalid: dict[str, Any] = {
        "source_id": "NCT001",
        "coreason_id": "CID1",
        "enrollment_count": "100.5",  # Float string to int often fails in strict mode or even default
    }
    with pytest.raises(ValidationError):
        SilverStudy.model_validate(data_invalid)


def test_validation_unicode_safety() -> None:
    """
    Test complex Unicode handling in validated fields.
    """
    unicode_str = "Étude de cas: 💊 & 🧬 for 100% efficiency"
    data: dict[str, Any] = {
        "source_id": "NCT_UNI",
        "coreason_id": "CID_UNI",
        "title": unicode_str,
    }

    model = SilverStudy.model_validate(data)
    assert model.title == unicode_str
