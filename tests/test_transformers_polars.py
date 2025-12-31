from datetime import date

import polars as pl
import pytest

from coreason_etl_clinicaltrialsgov.transformers_polars import (
    _generate_coreason_id_udf,
    _generate_surrogate_key_udf,
    _normalize_age_udf,
    _parse_date_udf,
    _safe_get_field,
    transform_to_silver_interventions,
    transform_to_silver_locations,
    transform_to_silver_outcomes,
    transform_to_silver_references,
    transform_to_silver_sponsors,
    transform_to_silver_studies,
)

# Sample study with some missing fields in child records to test robustness
SAMPLE_STUDY = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT00000001",
            "briefTitle": "Test Study",
            "officialTitle": "Official Test Study Title",
            "orgStudyIdInfo": {"id": "ORG-001"},
        },
        "statusModule": {
            "overallStatus": "RECRUITING",
            "startDateStruct": {"date": "2023-01-01"},
            "completionDateStruct": {"date": "2024-01-01"},
            "studyFirstPostDateStruct": {"date": "2022-12-01"},
        },
        "designModule": {
            "phases": ["PHASE1", "PHASE2"],
            "studyType": "INTERVENTIONAL",
            "enrollmentInfo": {"count": 50, "type": "ESTIMATED"},
        },
        "eligibilityModule": {
            "minimumAge": "18 Years",
            "maximumAge": "65 Years",
            "sex": "ALL",
            "healthyVolunteers": True,
        },
        "sponsorCollaboratorsModule": {
            "leadSponsor": {"name": "Big Pharma", "class": "INDUSTRY"},
            "collaborators": [{"name": "University X", "class": "OTHER"}],
        },
        "contactsLocationsModule": {
            "locations": [
                {
                    "facility": "Hospital A",
                    "city": "New York",
                    "state": "NY",
                    "country": "United States",
                    "zip": "10001",
                    "status": "RECRUITING",
                    "geoPoint": {"lat": 40.71, "lon": -74.00},
                }
            ]
        },
        "armsInterventionsModule": {
            "interventions": [
                {
                    "type": "DRUG",
                    "name": "Drug X",
                    "description": "Daily dose",
                    "otherNames": ["Brand X"],
                }
            ]
        },
        "outcomesModule": {
            "primaryOutcomes": [
                {"measure": "Survival", "timeFrame": "1 year", "description": "Overall survival"}
            ],
            "secondaryOutcomes": [
                {"measure": "Safety", "timeFrame": "1 year", "description": "Adverse events"}
            ],
        },
        "referencesModule": {
            "references": [{"pmid": "123456", "citation": "Author et al. 2023", "retraction": None}]
        },
    }
}

# Add a second study with missing optional fields to test safe extraction
SAMPLE_STUDY_MISSING = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT00000002",
            "briefTitle": "Partial Study",
        },
        "statusModule": {
            "overallStatus": "COMPLETED",
            "studyFirstPostDateStruct": {"date": "2022-12-01"},
        },
        # Missing designModule, etc.
        # Minimal modules
    }
}

# Empty study (no child records)
SAMPLE_STUDY_EMPTY = {
    "protocolSection": {
        "identificationModule": {"nctId": "NCT00000003"},
        "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
    }
}


@pytest.fixture
def lf_sample() -> pl.LazyFrame:
    # Use both studies to force schema inference to see both structures (though List[Dict] usually unions keys)
    # Actually, Polars.read_json or from_dicts might union keys if they differ.
    return pl.DataFrame([SAMPLE_STUDY, SAMPLE_STUDY_MISSING]).lazy()

@pytest.fixture
def lf_empty() -> pl.LazyFrame:
    return pl.DataFrame([SAMPLE_STUDY_EMPTY]).lazy()

def test_silver_studies(lf_sample: pl.LazyFrame) -> None:
    df = transform_to_silver_studies(lf_sample)
    assert df.height == 2
    rows = sorted(df.to_dicts(), key=lambda x: x["source_id"])

    # Study 1
    row = rows[0]
    assert row["source_id"] == "NCT00000001"
    assert row["title"] == "Test Study"
    assert row["min_age"] == 18.0
    assert row["phases"] == "PHASE1|PHASE2"
    assert row["start_date"] == date(2023, 1, 1)

    # Study 2
    row2 = rows[1]
    assert row2["source_id"] == "NCT00000002"
    assert row2["title"] == "Partial Study"
    assert row2["min_age"] is None
    assert row2["phases"] is None


def test_silver_sponsors(lf_sample: pl.LazyFrame, lf_empty: pl.LazyFrame) -> None:
    df = transform_to_silver_sponsors(lf_sample)
    # Study 1 has 2 sponsors. Study 2 has None.
    rows = df.to_dicts()
    assert len(rows) == 2
    names = {r["name"] for r in rows}
    assert "Big Pharma" in names
    assert "University X" in names

    lead = next(r for r in rows if r["role"] == "LEAD")
    assert lead["agency_class"] == "INDUSTRY"

    # Empty
    df_e = transform_to_silver_sponsors(lf_empty)
    assert df_e.height == 0
    assert "agency_class" in df_e.columns

def test_silver_locations(lf_sample: pl.LazyFrame, lf_empty: pl.LazyFrame) -> None:
    df = transform_to_silver_locations(lf_sample)
    rows = df.to_dicts()
    assert len(rows) == 1
    row = rows[0]
    assert row["facility"] == "Hospital A"
    assert row["city"] == "New York"
    assert row["zip"] == "10001"
    # geo_point might be Struct or Object depending on inference.
    # With explicit struct in test data, it should be Struct
    assert row["geo_point"]["lat"] == 40.71

    # Empty
    df_e = transform_to_silver_locations(lf_empty)
    assert df_e.height == 0
    assert "facility" in df_e.columns

def test_silver_interventions(lf_sample: pl.LazyFrame, lf_empty: pl.LazyFrame) -> None:
    df = transform_to_silver_interventions(lf_sample)
    rows = df.to_dicts()
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "Drug X"
    assert row["other_names"][0] == "Brand X"

    # Empty
    df_e = transform_to_silver_interventions(lf_empty)
    assert df_e.height == 0
    assert "other_names" in df_e.columns

def test_silver_outcomes(lf_sample: pl.LazyFrame, lf_empty: pl.LazyFrame) -> None:
    df = transform_to_silver_outcomes(lf_sample)
    rows = df.to_dicts()
    assert len(rows) == 2
    types = {r["outcome_type"] for r in rows}
    assert "PRIMARY" in types
    assert "SECONDARY" in types

    # Empty
    df_e = transform_to_silver_outcomes(lf_empty)
    assert df_e.height == 0
    assert "outcome_type" in df_e.columns

def test_silver_references(lf_sample: pl.LazyFrame, lf_empty: pl.LazyFrame) -> None:
    df = transform_to_silver_references(lf_sample)
    rows = df.to_dicts()
    assert len(rows) == 1
    row = rows[0]
    assert row["pmid"] == "123456"
    assert row["citation"] == "Author et al. 2023"

    # Empty
    df_e = transform_to_silver_references(lf_empty)
    assert df_e.height == 0
    assert "citation" in df_e.columns

def test_all_missing_structure() -> None:
    # Test completely empty dicts to ensure schema inspection safely returns None literals
    data = [
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT000"},
                "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
            }
        }
    ]
    lf = pl.DataFrame(data).lazy()

    # Just ensure they don't crash and return empty DFs or valid DFs with Nulls
    transform_to_silver_studies(lf)
    transform_to_silver_sponsors(lf)
    transform_to_silver_locations(lf)
    transform_to_silver_interventions(lf)
    transform_to_silver_outcomes(lf)
    transform_to_silver_references(lf)

# --- UDF Coverage Tests ---

def test_parse_date_udf() -> None:
    assert _parse_date_udf("2023-01-01") == "2023-01-01"
    assert _parse_date_udf("2023-01") == "2023-01-01"
    assert _parse_date_udf("2023") == "2023-01-01"
    assert _parse_date_udf(None) is None
    # Invalid strings are just padded by the UDF, Polars handles parsing failure
    # Updated to reflect implementation behavior (no validation inside UDF)
    assert _parse_date_udf("invalid") == "invalid-01-01"

def test_normalize_age_udf() -> None:
    assert _normalize_age_udf("18 Years") == 18.0
    assert _normalize_age_udf("18") == 18.0
    assert _normalize_age_udf("24 Months") == 2.0
    assert _normalize_age_udf("52 Weeks") == 1.0
    assert _normalize_age_udf("365 Days") == 1.0
    assert _normalize_age_udf(None) is None
    assert _normalize_age_udf("Invalid") is None
    assert _normalize_age_udf(" Years") is None
    # Coverage for empty string (splits to empty list)
    assert _normalize_age_udf("   ") is None
    # Coverage for unknown unit (returns raw value)
    assert _normalize_age_udf("18 Decades") == 18.0

def test_generate_coreason_id_udf() -> None:
    uid = _generate_coreason_id_udf("NCT123", "2023-01-01")
    assert isinstance(uid, str)
    assert len(uid) > 0

def test_generate_surrogate_key_udf() -> None:
    key = _generate_surrogate_key_udf("NCT123", "PART1", None, "PART2")
    assert isinstance(key, str)
    assert len(key) > 0

# --- Safe Get Field Coverage Tests ---

def test_safe_get_field_internals() -> None:
    # 1. Test root column missing
    lf = pl.DataFrame([{"otherCol": "val"}]).lazy()
    expr = _safe_get_field(lf, "protocolSection", ["id"], "id")
    res = lf.select(expr).collect()
    assert res.item(0, 0) is None

    # 2. Test root column exists but path missing
    lf2 = pl.DataFrame([{"protocolSection": {"id": "123"}}]).lazy()
    expr2 = _safe_get_field(lf2, "protocolSection", ["missingModule", "field"], "field")
    res2 = lf2.select(expr2).collect()
    assert res2.item(0, 0) is None

    # 3. Test path exists
    expr3 = _safe_get_field(lf2, "protocolSection", ["id"], "id")
    res3 = lf2.select(expr3).collect()
    assert res3.item(0, 0) == "123"

    # 4. Test path collision with non-struct (nested field on non-struct)
    lf4 = pl.DataFrame([{"protocolSection": {"id": "123"}}]).lazy()
    # 'id' is String. Try to access 'id.subfield'.
    expr4 = _safe_get_field(lf4, "protocolSection", ["id", "subfield"], "field")
    res4 = lf4.select(expr4).collect()
    assert res4.item(0, 0) is None

# --- Branch Coverage for Transformers (Empty Result Logic) ---

def test_sponsors_structure_exists_values_null() -> None:
    schema = {
        "protocolSection": pl.Struct([
            pl.Field("identificationModule", pl.Struct([pl.Field("nctId", pl.String)])),
            pl.Field("statusModule", pl.Struct([pl.Field("studyFirstPostDateStruct", pl.Struct([pl.Field("date", pl.String)]))])),
            pl.Field("sponsorCollaboratorsModule", pl.Struct([
                pl.Field("leadSponsor", pl.Struct([pl.Field("name", pl.String), pl.Field("class", pl.String)])),
                pl.Field("collaborators", pl.List(pl.Struct([pl.Field("name", pl.String), pl.Field("class", pl.String)])))
            ]))
        ])
    }
    data = [{"protocolSection": {"identificationModule": {"nctId": "NCT999"}, "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}}, "sponsorCollaboratorsModule": {"leadSponsor": None, "collaborators": None}}}]
    lf = pl.DataFrame(data).lazy().with_columns(pl.col("protocolSection").cast(schema["protocolSection"]))
    df = transform_to_silver_sponsors(lf)
    assert df.height == 0

def test_locations_structure_exists_values_null() -> None:
    schema = {
        "protocolSection": pl.Struct([
            pl.Field("identificationModule", pl.Struct([pl.Field("nctId", pl.String)])),
            pl.Field("statusModule", pl.Struct([pl.Field("studyFirstPostDateStruct", pl.Struct([pl.Field("date", pl.String)]))])),
            pl.Field("contactsLocationsModule", pl.Struct([
                pl.Field("locations", pl.List(pl.Struct([
                    pl.Field("facility", pl.String), pl.Field("city", pl.String), pl.Field("state", pl.String),
                    pl.Field("country", pl.String), pl.Field("zip", pl.String), pl.Field("status", pl.String),
                    pl.Field("geoPoint", pl.Struct([pl.Field("lat", pl.Float64), pl.Field("lon", pl.Float64)]))
                ])))
            ]))
        ])
    }
    data = [{"protocolSection": {"identificationModule": {"nctId": "NCT999"}, "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}}, "contactsLocationsModule": {"locations": None}}}]
    lf = pl.DataFrame(data).lazy().with_columns(pl.col("protocolSection").cast(schema["protocolSection"]))
    df = transform_to_silver_locations(lf)
    assert df.height == 0

def test_interventions_structure_exists_values_null() -> None:
    schema = {
        "protocolSection": pl.Struct([
            pl.Field("identificationModule", pl.Struct([pl.Field("nctId", pl.String)])),
            pl.Field("statusModule", pl.Struct([pl.Field("studyFirstPostDateStruct", pl.Struct([pl.Field("date", pl.String)]))])),
            pl.Field("armsInterventionsModule", pl.Struct([
                pl.Field("interventions", pl.List(pl.Struct([
                    pl.Field("type", pl.String), pl.Field("name", pl.String),
                    pl.Field("description", pl.String), pl.Field("otherNames", pl.List(pl.String))
                ])))
            ]))
        ])
    }
    data = [{"protocolSection": {"identificationModule": {"nctId": "NCT999"}, "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}}, "armsInterventionsModule": {"interventions": None}}}]
    lf = pl.DataFrame(data).lazy().with_columns(pl.col("protocolSection").cast(schema["protocolSection"]))
    df = transform_to_silver_interventions(lf)
    assert df.height == 0

def test_outcomes_structure_exists_values_null() -> None:
    # Define minimal outcomes struct
    o_struct = pl.List(pl.Struct([pl.Field("measure", pl.String), pl.Field("timeFrame", pl.String), pl.Field("description", pl.String)]))
    schema = {
        "protocolSection": pl.Struct([
            pl.Field("identificationModule", pl.Struct([pl.Field("nctId", pl.String)])),
            pl.Field("statusModule", pl.Struct([pl.Field("studyFirstPostDateStruct", pl.Struct([pl.Field("date", pl.String)]))])),
            pl.Field("outcomesModule", pl.Struct([
                pl.Field("primaryOutcomes", o_struct),
                pl.Field("secondaryOutcomes", o_struct),
                pl.Field("otherOutcomes", o_struct)
            ]))
        ])
    }
    data = [{"protocolSection": {"identificationModule": {"nctId": "NCT999"}, "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}}, "outcomesModule": {"primaryOutcomes": None, "secondaryOutcomes": None, "otherOutcomes": None}}}]
    lf = pl.DataFrame(data).lazy().with_columns(pl.col("protocolSection").cast(schema["protocolSection"]))
    df = transform_to_silver_outcomes(lf)
    assert df.height == 0

def test_references_structure_exists_values_null() -> None:
    schema = {
        "protocolSection": pl.Struct([
            pl.Field("identificationModule", pl.Struct([pl.Field("nctId", pl.String)])),
            pl.Field("statusModule", pl.Struct([pl.Field("studyFirstPostDateStruct", pl.Struct([pl.Field("date", pl.String)]))])),
            pl.Field("referencesModule", pl.Struct([
                pl.Field("references", pl.List(pl.Struct([
                    pl.Field("pmid", pl.String), pl.Field("citation", pl.String), pl.Field("retraction", pl.Struct([pl.Field("retraction", pl.Boolean)]))
                ])))
            ]))
        ])
    }
    data = [{"protocolSection": {"identificationModule": {"nctId": "NCT999"}, "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}}, "referencesModule": {"references": None}}}]
    lf = pl.DataFrame(data).lazy().with_columns(pl.col("protocolSection").cast(schema["protocolSection"]))
    df = transform_to_silver_references(lf)
    assert df.height == 0
