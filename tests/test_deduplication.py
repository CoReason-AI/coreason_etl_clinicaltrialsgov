import polars as pl

from coreason_etl_clinicaltrialsgov.transformers_polars import transform_to_silver_sponsors


def test_silver_sponsors_deduplication() -> None:
    """Test that duplicate sponsors are removed."""
    # Create a raw payload with duplicate sponsors
    # We simulate this by creating a LazyFrame that mimics the structure passed to the transformer
    # The transformer expects a LazyFrame with specific nested structure

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

    # Wrap in a list to create DataFrame
    lf = pl.DataFrame([data]).lazy()

    # Run transformation
    result_df = transform_to_silver_sponsors(lf)

    # Convert to list of dicts for assertion
    results = result_df.to_dicts()

    # We expect:
    # 1. Lead Sponsor: Pharma Corp (LEAD)
    # 2. Duplicate Partner (COLLABORATOR) - only once
    # 3. Unique Partner (COLLABORATOR)
    # Total = 3

    assert len(results) == 3

    names = sorted([r["name"] for r in results])
    assert names == ["Duplicate Partner", "Pharma Corp", "Unique Partner"]

    # Verify strict deduplication on the ID
    ids = [r["id"] for r in results]
    assert len(ids) == len(set(ids))
