# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

from coreason_etl_clinicaltrialsgov.transformers import transform_study


def test_explicit_deduplication_sponsors() -> None:
    """Verify that identical sponsors are explicitly deduplicated."""
    raw_study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_DEDUP_001"},
            "sponsorCollaboratorsModule": {
                "collaborators": [
                    {"name": "Lab A", "class": "OTHER"},
                    {"name": "Lab A", "class": "OTHER"},  # Duplicate
                    {"name": "Lab B", "class": "OTHER"},
                ]
            },
        }
    }

    result = transform_study(raw_study)
    sponsors = result["silver_sponsors"]

    # Should be 2 unique sponsors (Lab A, Lab B), not 3
    assert len(sponsors) == 2
    names = {s["name"] for s in sponsors}
    assert names == {"Lab A", "Lab B"}


def test_explicit_deduplication_locations() -> None:
    """Verify that identical locations are explicitly deduplicated."""
    loc = {"facility": "Fac X", "city": "City Y", "country": "Country Z"}
    raw_study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_DEDUP_002"},
            "contactsLocationsModule": {
                "locations": [loc, loc, loc]  # 3 identical
            },
        }
    }

    result = transform_study(raw_study)
    locations = result["silver_locations"]

    assert len(locations) == 1
    assert locations[0]["facility"] == "Fac X"


def test_unicode_emoji_handling() -> None:
    """Verify handling of Emoji and wide Unicode characters."""
    title_with_emoji = "Study of 💊 and 💉 in Patients"
    facility_jp = "病院"  # Hospital

    raw_study = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT_UNICODE",
                "briefTitle": title_with_emoji
            },
            "contactsLocationsModule": {
                "locations": [{"facility": facility_jp, "city": "Tokyo", "country": "Japan"}]
            }
        }
    }

    result = transform_study(raw_study)

    study = result["silver_studies"][0]
    assert study["title"] == title_with_emoji

    loc = result["silver_locations"][0]
    assert loc["facility"] == facility_jp
    # Ensure ID generation worked (didn't crash on unicode)
    assert loc["id"] is not None


def test_deduplication_interventions() -> None:
    """Verify deduplication of interventions."""
    interv = {"type": "DRUG", "name": "Aspirin"}
    raw_study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_DEDUP_003"},
            "armsInterventionsModule": {
                "interventions": [interv, interv]
            }
        }
    }
    result = transform_study(raw_study)
    assert len(result["silver_interventions"]) == 1
