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

from coreason_etl_clinicaltrialsgov.transformers import transform_study


@pytest.fixture
def base_study() -> dict[str, Any]:
    return {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT001"},
            "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
            "sponsorCollaboratorsModule": {},
            "contactsLocationsModule": {},
            "armsInterventionsModule": {},
            "outcomesModule": {},
            "referencesModule": {},
        }
    }


def test_sponsor_identity_stability(base_study: dict[str, Any]) -> None:
    """Test that changing non-key fields does not change the ID."""
    # Base case
    base_study["protocolSection"]["sponsorCollaboratorsModule"] = {
        "leadSponsor": {"name": "Pharma Corp", "class": "INDUSTRY"}
    }

    result1 = transform_study(base_study)
    sponsors1 = result1["silver_sponsors"]
    assert len(sponsors1) == 1
    id1 = sponsors1[0]["id"]

    # Modify non-key field (class)
    base_study["protocolSection"]["sponsorCollaboratorsModule"]["leadSponsor"]["class"] = "OTHER"

    result2 = transform_study(base_study)
    sponsors2 = result2["silver_sponsors"]
    assert len(sponsors2) == 1
    id2 = sponsors2[0]["id"]

    assert id1 == id2, "ID should be stable when agency_class changes"


def test_sponsor_identity_uniqueness(base_study: dict[str, Any]) -> None:
    """Test that changing key fields changes the ID."""
    base_study["protocolSection"]["sponsorCollaboratorsModule"] = {
        "leadSponsor": {"name": "Pharma Corp", "class": "INDUSTRY"}
    }
    result1 = transform_study(base_study)
    id1 = result1["silver_sponsors"][0]["id"]

    # Modify key field (name)
    base_study["protocolSection"]["sponsorCollaboratorsModule"]["leadSponsor"]["name"] = "Pharma Inc"
    result2 = transform_study(base_study)
    id2 = result2["silver_sponsors"][0]["id"]

    assert id1 != id2, "ID should change when name changes"


def test_location_identity_stability(base_study: dict[str, Any]) -> None:
    """Test location ID stability (Keys: Facility, City, State, Country)."""
    loc = {
        "facility": "General Hospital",
        "city": "Boston",
        "state": "MA",
        "country": "USA",
        "zip": "02114",
        "status": "RECRUITING",
    }
    base_study["protocolSection"]["contactsLocationsModule"] = {"locations": [loc]}

    result1 = transform_study(base_study)
    id1 = result1["silver_locations"][0]["id"]

    # Modify non-key field (zip, status)
    loc2 = loc.copy()
    loc2["zip"] = "99999"
    loc2["status"] = "COMPLETED"
    base_study["protocolSection"]["contactsLocationsModule"] = {"locations": [loc2]}

    result2 = transform_study(base_study)
    id2 = result2["silver_locations"][0]["id"]

    assert id1 == id2, "Location ID should be stable when zip/status changes"


def test_intervention_identity_stability(base_study: dict[str, Any]) -> None:
    """Test intervention ID stability (Keys: Type, Name)."""
    interv = {"type": "DRUG", "name": "Aspirin", "description": "100mg", "otherNames": ["BrandA"]}
    base_study["protocolSection"]["armsInterventionsModule"] = {"interventions": [interv]}

    result1 = transform_study(base_study)
    id1 = result1["silver_interventions"][0]["id"]

    # Modify non-key fields
    interv2 = interv.copy()
    interv2["description"] = "200mg"
    interv2["otherNames"] = ["BrandB"]
    base_study["protocolSection"]["armsInterventionsModule"] = {"interventions": [interv2]}

    result2 = transform_study(base_study)
    id2 = result2["silver_interventions"][0]["id"]

    assert id1 == id2, "Intervention ID should be stable when description/otherNames changes"


def test_outcome_identity_stability(base_study: dict[str, Any]) -> None:
    """Test outcome ID stability (Keys: Type, Measure, TimeFrame)."""
    out = {"measure": "Survival", "timeFrame": "1 year", "description": "Overall survival rate"}
    # Primary Outcome
    base_study["protocolSection"]["outcomesModule"] = {"primaryOutcomes": [out]}

    result1 = transform_study(base_study)
    id1 = result1["silver_outcomes"][0]["id"]

    # Modify non-key field
    out2 = out.copy()
    out2["description"] = "Changed description"
    base_study["protocolSection"]["outcomesModule"] = {"primaryOutcomes": [out2]}

    result2 = transform_study(base_study)
    id2 = result2["silver_outcomes"][0]["id"]

    assert id1 == id2, "Outcome ID should be stable when description changes"


def test_reference_identity_stability(base_study: dict[str, Any]) -> None:
    """Test reference ID stability (Keys: PMID, Citation)."""
    ref = {"pmid": "12345", "citation": "Smith et al.", "retraction": {"pmid": "999"}}
    base_study["protocolSection"]["referencesModule"] = {"references": [ref]}

    result1 = transform_study(base_study)
    id1 = result1["silver_references"][0]["id"]

    # Modify non-key field
    ref2 = ref.copy()
    ref2["retraction"] = {"pmid": "888"}
    base_study["protocolSection"]["referencesModule"] = {"references": [ref2]}

    result2 = transform_study(base_study)
    id2 = result2["silver_references"][0]["id"]

    assert id1 == id2, "Reference ID should be stable when retraction info changes"


def test_child_deduplication(base_study: dict[str, Any]) -> None:
    """Test that identical child records are deduplicated."""
    interv = {"type": "DRUG", "name": "Aspirin", "description": "100mg"}
    # Add same intervention twice
    base_study["protocolSection"]["armsInterventionsModule"] = {"interventions": [interv, interv]}

    result = transform_study(base_study)
    interventions = result["silver_interventions"]

    assert len(interventions) == 1, "Should deduplicate identical interventions"


def test_missing_keys_collision(base_study: dict[str, Any]) -> None:
    """Test behavior when keys are missing (None)."""
    # Two locations with no details
    loc1 = {"city": None}
    loc2 = {"city": None}

    base_study["protocolSection"]["contactsLocationsModule"] = {"locations": [loc1, loc2]}

    result = transform_study(base_study)
    assert len(result["silver_locations"]) == 1


def test_key_type_safety(base_study: dict[str, Any]) -> None:
    """Test that non-string key components (e.g. integer PMID) do not crash the hash generation."""
    ref = {
        "pmid": 12345,  # Integer instead of string
        "citation": "Some citation",
    }
    base_study["protocolSection"]["referencesModule"] = {"references": [ref]}

    # This should not crash
    result = transform_study(base_study)
    assert len(result["silver_references"]) == 1
    # We cast to string in the model creation, so it should be a string
    assert result["silver_references"][0]["pmid"] == "12345"
