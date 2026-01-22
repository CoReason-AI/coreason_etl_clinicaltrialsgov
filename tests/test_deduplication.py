import polars as pl

from coreason_etl_clinicaltrialsgov.transformers_polars import transform_to_silver_sponsors


def test_silver_sponsors_deduplication() -> None:
    """Test that duplicate sponsors are removed."""
    data = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000001"},
            "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "Pharma Corp", "class": "INDUSTRY"},
                "collaborators": [
                    {"name": "Duplicate Partner", "class": "OTHER"},
                    {"name": "Duplicate Partner", "class": "OTHER"},
                    {"name": "Unique Partner", "class": "OTHER"},
                ],
            },
        }
    }

    lf = pl.DataFrame([data]).lazy()
    result_df = transform_to_silver_sponsors(lf)
    results = result_df.to_dicts()

    # Expect 3 unique entries
    assert len(results) == 3
    names = sorted([r["name"] for r in results])
    assert names == ["Duplicate Partner", "Pharma Corp", "Unique Partner"]

    ids = [r["id"] for r in results]
    assert len(ids) == len(set(ids))


def test_silver_sponsors_complex_edge_cases() -> None:
    """Test mixed roles, unicode, case sensitivity, and special characters."""
    data = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_EDGE_CASE"},
            "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "Global Corp", "class": "INDUSTRY"},
                "collaborators": [
                    # Case 1: Same name as lead, but role is Collaborator
                    # Should be kept (different role = different ID)
                    {"name": "Global Corp", "class": "INDUSTRY"},
                    # Case 2: Unicode characters
                    {"name": "München Universität", "class": "OTHER"},
                    # Case 3: Case Sensitivity (Strict equality expected)
                    {"name": "global corp", "class": "OTHER"},
                    # Case 4: Whitespace (Strict equality expected)
                    {"name": "Global Corp ", "class": "OTHER"},
                    # Case 5: Special characters including delimiter '|'
                    {"name": "Special|Name_With_Pipe", "class": "OTHER"},
                    # Case 6: Special characters resulting in potential collision?
                    # "A|B" vs "A_B". The internal logic replaces | with _.
                    # If we have "A|B", it becomes "A_B" in seed.
                    # If we have "A_B", it becomes "A_B" in seed.
                    # They will collide. This verifies that behavior.
                    {"name": "Collision|Candidate", "class": "OTHER"},
                    {"name": "Collision_Candidate", "class": "OTHER"},
                ],
            },
        }
    }

    lf = pl.DataFrame([data]).lazy()
    result_df = transform_to_silver_sponsors(lf)
    results = result_df.to_dicts()

    # Analyze results

    # 1. Lead "Global Corp"
    lead = next(r for r in results if r["role"] == "LEAD")
    assert lead["name"] == "Global Corp"

    # 2. Collaborator "Global Corp"
    collab_same_name = next(r for r in results if r["role"] == "COLLABORATOR" and r["name"] == "Global Corp")
    assert collab_same_name["name"] == "Global Corp"
    # Ensure IDs are different due to role
    assert lead["id"] != collab_same_name["id"]

    # 3. Unicode
    unicode_entry = next(r for r in results if r["name"] == "München Universität")
    assert unicode_entry["role"] == "COLLABORATOR"

    # 4. Case Sensitivity ("global corp" vs "Global Corp")
    lower_entry = next(r for r in results if r["name"] == "global corp")
    assert lower_entry["id"] != collab_same_name["id"]

    # 5. Whitespace ("Global Corp " vs "Global Corp")
    # UPDATED BEHAVIOR: We now strip whitespace in ID generation.
    # So "Global Corp " becomes "Global Corp" in ID generation.
    # It collides with "Global Corp" (collab_same_name).
    # Since "Global Corp" (collab_same_name) appeared earlier in the list (Case 1),
    # "Global Corp " (Case 4) is dropped as a duplicate.

    # Assert that "Global Corp " is NOT present in the results
    ws_entries = [r for r in results if r["name"] == "Global Corp "]
    assert len(ws_entries) == 0

    # 6. Pipe Handling
    # "Collision|Candidate" vs "Collision_Candidate"
    # Logic: replace("|", "_") -> both become "Collision_Candidate" in the seed.
    # Therefore, they will generate the same ID.
    # Since we dedup on ID, only ONE should remain (First Write Wins).
    # In the list, "Collision|Candidate" comes first.

    collision_candidates = [r for r in results if "Collision" in r["name"]]
    # We expect strictly 1 record if they collide, or 2 if the logic handles it differently.
    # Based on _generate_surrogate_key_udf logic:
    # parent_id + role + name
    # name "Collision|Candidate" -> "Collision_Candidate"
    # name "Collision_Candidate" -> "Collision_Candidate"
    # They ARE collisions.
    assert len(collision_candidates) == 1
    # First one wins
    assert collision_candidates[0]["name"] == "Collision|Candidate"
