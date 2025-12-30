# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

from coreason_etl_clinicaltrialsgov.transformers import generate_surrogate_key, transform_study


def test_surrogate_key_emojis() -> None:
    """Verify robust handling of Emojis in surrogate keys."""
    # Emojis (Medical symbol, Pill, Hospital)
    k1 = generate_surrogate_key("NCT_EMOJI", "⚕️", "💊", "🏥")
    k2 = generate_surrogate_key("NCT_EMOJI", "⚕️", "💊", "🏥")

    assert k1 == k2

    # Verify it produces a valid string UUID
    assert isinstance(k1, str)
    assert len(k1) == 36

    # Changing an emoji changes the key
    k3 = generate_surrogate_key("NCT_EMOJI", "⚕️", "💊", "🚑")
    assert k1 != k3


def test_surrogate_key_ignore_non_identifying_fields_interventions() -> None:
    """Verify that SilverIntervention ID ignores non-identifying fields like description."""

    base_raw = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_ID_STABILITY"},
            "armsInterventionsModule": {
                "interventions": [
                    {"type": "DRUG", "name": "Drug A", "description": "Original Description", "otherNames": ["Alias 1"]}
                ]
            },
        }
    }

    # Run first transform
    result1 = transform_study(base_raw)
    # The result contains list of dicts. We assert it's a list first.
    assert isinstance(result1["silver_interventions"], list)
    id1 = result1["silver_interventions"][0]["id"]

    # Modify description and otherNames (non-identifying fields)
    # We must help mypy know this structure is mutable and has these keys
    interventions = base_raw["protocolSection"]["armsInterventionsModule"]["interventions"]  # type: ignore
    interventions[0]["description"] = "New Description"
    interventions[0]["otherNames"] = ["Alias 2"]

    # Run second transform
    result2 = transform_study(base_raw)
    assert isinstance(result2["silver_interventions"], list)
    id2 = result2["silver_interventions"][0]["id"]

    # IDs should match because Business Key is (nct_id + type + name)
    assert id1 == id2

    # Check that contents actually updated
    assert result2["silver_interventions"][0]["description"] == "New Description"


def test_surrogate_key_ignore_non_identifying_fields_outcomes() -> None:
    """Verify that SilverOutcome ID ignores non-identifying fields like description."""

    base_raw = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_ID_STABILITY_OUT"},
            "outcomesModule": {
                "primaryOutcomes": [
                    {"measure": "Survival", "timeFrame": "1 year", "description": "Original Description"}
                ]
            },
        }
    }

    # Run first transform
    result1 = transform_study(base_raw)
    assert isinstance(result1["silver_outcomes"], list)
    id1 = result1["silver_outcomes"][0]["id"]

    # Modify description
    outcomes = base_raw["protocolSection"]["outcomesModule"]["primaryOutcomes"]  # type: ignore
    outcomes[0]["description"] = "New Description"

    # Run second transform
    result2 = transform_study(base_raw)
    assert isinstance(result2["silver_outcomes"], list)
    id2 = result2["silver_outcomes"][0]["id"]

    # IDs should match because Business Key is (nct_id + type + measure + time_frame)
    assert id1 == id2


def test_surrogate_key_ignore_non_identifying_fields_locations() -> None:
    """Verify that SilverLocation ID ignores non-identifying fields like status and zip."""

    base_raw = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_ID_STABILITY_LOC"},
            "contactsLocationsModule": {
                "locations": [
                    {
                        "facility": "General Hospital",
                        "city": "Boston",
                        "state": "MA",
                        "country": "USA",
                        "zip": "02114",
                        "status": "RECRUITING",
                    }
                ]
            },
        }
    }

    # Run first transform
    result1 = transform_study(base_raw)
    assert isinstance(result1["silver_locations"], list)
    id1 = result1["silver_locations"][0]["id"]

    # Modify status and zip
    locations = base_raw["protocolSection"]["contactsLocationsModule"]["locations"]  # type: ignore
    locations[0]["status"] = "COMPLETED"
    locations[0]["zip"] = "99999"

    # Run second transform
    result2 = transform_study(base_raw)
    assert isinstance(result2["silver_locations"], list)
    id2 = result2["silver_locations"][0]["id"]

    # IDs should match because Business Key is (nct_id + facility + city + state + country)
    assert id1 == id2


def test_surrogate_key_none_vs_empty_string_complex() -> None:
    """Detailed verification of None vs Empty String handling in keys."""
    # (A, None, B) vs (A, "", B)
    # Implementation treats None as "" so they might collide if not handled.
    # Looking at implementation:
    # if p is None: append("")
    # else: append(str(p))
    # So None -> "" and "" -> "". They WILL collide.
    # The requirement is that they *should* be treated consistently or distinct if business logic requires.
    # Typically None and "" are treated as "missing" in these ETLs unless specified otherwise.
    # Let's verify they DO collide, consistent with current implementation.

    k1 = generate_surrogate_key("ROOT", "A", None, "B")
    k2 = generate_surrogate_key("ROOT", "A", "", "B")
    assert k1 == k2

    # Verify that "0" (string) does not look like None/Empty
    k3 = generate_surrogate_key("ROOT", "A", "0", "B")
    assert k1 != k3


def test_future_dates_parsing() -> None:
    """Verify parsing of far future dates."""
    # API V2 can return dates like "2099-12-31"
    raw_study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_FUTURE"},
            "statusModule": {"startDateStruct": {"date": "2099-12-31"}, "completionDateStruct": {"date": "3000-01-01"}},
        }
    }

    result = transform_study(raw_study)
    assert isinstance(result["silver_studies"], list)
    study = result["silver_studies"][0]

    assert str(study["start_date"]) == "2099-12-31"
    assert str(study["completion_date"]) == "3000-01-01"
