# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

import uuid
from typing import Any, Optional

import polars as pl

# --- Constants ---

DEFAULT_STRING_TYPE = pl.String()

# --- Helpers ---


def _parse_date_udf(date_str: Optional[str]) -> Optional[str]:
    """Parse partial dates into a date string YYYY-MM-DD."""
    if not date_str:
        return None

    parts = date_str.split("-")
    if len(parts) == 3:
        return date_str
    elif len(parts) == 2:
        return f"{date_str}-01"

    return f"{date_str}-01-01"


def _normalize_age_udf(age_str: Optional[str]) -> Optional[float]:
    """Normalize age string to years (float)."""
    if not age_str:
        return None

    parts = age_str.strip().lower().split()
    if not parts:
        return None

    try:
        val_str = parts[0]
        value = float(val_str)
    except (ValueError, IndexError):
        return None

    if len(parts) < 2:
        return value

    unit = parts[1]
    if "month" in unit:
        return value / 12.0
    elif "week" in unit:
        return value / 52.0
    elif "day" in unit:
        return value / 365.0
    elif "year" in unit:
        return value
    return value


def _generate_coreason_id_udf(nct_id: str, first_received_date: Optional[str]) -> str:
    """Generate deterministic UUID for study."""
    seed = f"clinicaltrials.gov/{nct_id}/{first_received_date or ''}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))


def _generate_surrogate_key_udf(parent_id: str, *parts: Any) -> str:
    """Generate deterministic UUID for child records."""
    seed_parts = [parent_id.replace("|", "_")]
    for p in parts:
        if p is None:
            seed_parts.append("")
        else:
            sanitized = str(p).replace("|", "_")
            seed_parts.append(sanitized)
    seed = "|".join(seed_parts)
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))


def _safe_get_field(
    lf: pl.LazyFrame, root_col: str, path: list[str], alias: str, dtype: pl.DataType = DEFAULT_STRING_TYPE
) -> pl.Expr:
    """
    Checks if the path exists in the LazyFrame schema. If so, returns the field expression.
    If not, returns a null literal with the alias.
    """
    schema = lf.collect_schema()

    # Using Any to avoid complex union types with Polars internals (DataType vs DataTypeClass)
    curr_type: Any = schema.get(root_col)

    if curr_type is None:
        return pl.lit(None, dtype=dtype).alias(alias)

    expr = pl.col(root_col)

    for p in path:
        if isinstance(curr_type, pl.Struct):
            found = False
            for field in curr_type.fields:
                if field.name == p:
                    expr = expr.struct.field(p)
                    curr_type = field.dtype
                    found = True
                    break

            if not found:
                return pl.lit(None, dtype=dtype).alias(alias)
        else:
            return pl.lit(None, dtype=dtype).alias(alias)

    return expr.alias(alias)


# --- Transformers ---


def transform_to_silver_studies(lf: pl.LazyFrame) -> pl.DataFrame:
    """Transform to Silver Studies."""

    def get(path: list[str], alias: str, dtype: pl.DataType = DEFAULT_STRING_TYPE) -> pl.Expr:
        return _safe_get_field(lf, "protocolSection", path, alias, dtype)

    base = lf.select([
        get(["identificationModule", "nctId"], "nct_id"),
        get(["identificationModule", "briefTitle"], "title"),
        get(["identificationModule", "officialTitle"], "official_title"),
        get(["identificationModule", "orgStudyIdInfo", "id"], "org_study_id"),
        get(["statusModule", "overallStatus"], "overall_status"),
        get(["statusModule", "startDateStruct", "date"], "start_date_raw"),
        get(["statusModule", "completionDateStruct", "date"], "completion_date_raw"),
        get(["statusModule", "studyFirstPostDateStruct", "date"], "first_received_date"),
        get(["designModule", "phases"], "phases_list", pl.List(pl.String())),
        get(["designModule", "studyType"], "study_type"),
        get(["designModule", "enrollmentInfo", "count"], "enrollment_count", pl.Int64()),
        get(["designModule", "enrollmentInfo", "type"], "enrollment_type"),
        get(["eligibilityModule", "minimumAge"], "min_age_raw"),
        get(["eligibilityModule", "maximumAge"], "max_age_raw"),
        get(["eligibilityModule", "sex"], "sex"),
        get(["eligibilityModule", "healthyVolunteers"], "accepted_healthy_volunteers", pl.Boolean()),
    ])

    return (
        base.filter(pl.col("nct_id").is_not_null())
        .with_columns([
            pl.col("start_date_raw")
            .map_elements(_parse_date_udf, return_dtype=pl.String)
            .str.to_date(strict=False)
            .alias("start_date"),
            pl.col("completion_date_raw")
            .map_elements(_parse_date_udf, return_dtype=pl.String)
            .str.to_date(strict=False)
            .alias("completion_date"),
            pl.col("phases_list").list.sort().list.join("|").alias("phases"),
            pl.col("min_age_raw").map_elements(_normalize_age_udf, return_dtype=pl.Float64).alias("min_age"),
            pl.col("max_age_raw").map_elements(_normalize_age_udf, return_dtype=pl.Float64).alias("max_age"),
            pl.struct(["nct_id", "first_received_date"])
            .map_elements(
                lambda x: _generate_coreason_id_udf(x["nct_id"], x["first_received_date"]), return_dtype=pl.String
            )
            .alias("coreason_id"),
        ])
        .select([
            pl.col("nct_id").alias("source_id"),
            pl.col("coreason_id"),
            pl.col("title"),
            pl.col("official_title"),
            pl.col("org_study_id"),
            pl.col("overall_status"),
            pl.col("start_date"),
            pl.col("completion_date"),
            pl.col("phases"),
            pl.col("study_type"),
            pl.col("enrollment_count"),
            pl.col("enrollment_type"),
            pl.col("min_age"),
            pl.col("max_age"),
            pl.col("sex"),
            pl.col("accepted_healthy_volunteers"),
        ])
        .collect()
    )


def transform_to_silver_sponsors(lf: pl.LazyFrame) -> pl.DataFrame:
    def get(path: list[str], alias: str, dtype: pl.DataType = DEFAULT_STRING_TYPE) -> pl.Expr:
        return _safe_get_field(lf, "protocolSection", path, alias, dtype)

    base = lf.select([
        get(["identificationModule", "nctId"], "nct_id"),
        get(["statusModule", "studyFirstPostDateStruct", "date"], "first_received_date"),
        _safe_get_field(
            lf,
            "protocolSection",
            ["sponsorCollaboratorsModule"],
            "sponsors_mod",
            pl.Struct([
                pl.Field("leadSponsor", pl.Struct([pl.Field("name", pl.String), pl.Field("class", pl.String)])),
                pl.Field(
                    "collaborators",
                    pl.List(pl.Struct([pl.Field("name", pl.String), pl.Field("class", pl.String)])),
                ),
            ]),
        ),
    ]).filter(pl.col("nct_id").is_not_null())

    base = base.with_columns(
        pl.struct(["nct_id", "first_received_date"])
        .map_elements(
            lambda x: _generate_coreason_id_udf(x["nct_id"], x["first_received_date"]), return_dtype=pl.String
        )
        .alias("coreason_id")
    )

    base_schema = base.collect_schema()
    s_mod_dtype = base_schema.get("sponsors_mod")

    has_lead = False
    has_collab = False

    if isinstance(s_mod_dtype, pl.Struct):
        field_names = {f.name for f in s_mod_dtype.fields}
        has_lead = "leadSponsor" in field_names
        has_collab = "collaborators" in field_names

    leads = []
    if has_lead:
        lead_df = (
            base.filter(pl.col("sponsors_mod").is_not_null())
            .select([
                pl.col("nct_id"),
                pl.col("coreason_id"),
                pl.col("sponsors_mod").struct.field("leadSponsor").alias("lead"),
            ])
            .filter(pl.col("lead").is_not_null())
            .select([
                pl.col("nct_id"),
                pl.col("coreason_id"),
                pl.col("lead").struct.field("name").alias("name"),
                pl.col("lead").struct.field("class").alias("agency_class"),
                pl.lit("LEAD").alias("role"),
            ])
        )
        leads.append(lead_df)

    collabs = []
    if has_collab:
        collab_df = (
            base.filter(pl.col("sponsors_mod").is_not_null())
            .select([
                pl.col("nct_id"),
                pl.col("coreason_id"),
                pl.col("sponsors_mod").struct.field("collaborators").alias("c_list"),
            ])
            .filter(pl.col("c_list").is_not_null())
            .explode("c_list")
            .select([
                pl.col("nct_id"),
                pl.col("coreason_id"),
                pl.col("c_list").struct.field("name").alias("name"),
                pl.col("c_list").struct.field("class").alias("agency_class"),
                pl.lit("COLLABORATOR").alias("role"),
            ])
        )
        collabs.append(collab_df)

    to_concat = []
    if leads:
        to_concat.append(leads[0].collect())
    if collabs:
        to_concat.append(collabs[0].collect())

    if not to_concat:
        # Fallback schema if completely missing from source
        return pl.DataFrame(
            schema={
                "id": pl.String,
                "source_id": pl.String,
                "coreason_id": pl.String,
                "name": pl.String,
                "agency_class": pl.String,
                "role": pl.String,
            }
        )

    combined = pl.concat(to_concat)

    def _gen_sponsor_id(row: dict[str, Any]) -> str:
        return _generate_surrogate_key_udf(row["nct_id"], row["role"], row["name"])

    return (
        combined.lazy()
        .with_columns(
            pl.struct(["nct_id", "role", "name"]).map_elements(_gen_sponsor_id, return_dtype=pl.String).alias("id")
        )
        .unique(subset=["id"], keep="first")
        .select([
            pl.col("id"),
            pl.col("nct_id").alias("source_id"),
            pl.col("coreason_id"),
            pl.col("name"),
            pl.col("agency_class"),
            pl.col("role"),
        ])
        .collect()
    )


def transform_to_silver_locations(lf: pl.LazyFrame) -> pl.DataFrame:
    base = lf.select([
        _safe_get_field(lf, "protocolSection", ["identificationModule", "nctId"], "nct_id"),
        _safe_get_field(
            lf, "protocolSection", ["statusModule", "studyFirstPostDateStruct", "date"], "first_received_date"
        ),
        _safe_get_field(
            lf,
            "protocolSection",
            ["contactsLocationsModule", "locations"],
            "locations",
            pl.List(
                pl.Struct([
                    pl.Field("facility", pl.String),
                    pl.Field("city", pl.String),
                    pl.Field("state", pl.String),
                    pl.Field("country", pl.String),
                    pl.Field("zip", pl.String),
                    pl.Field("status", pl.String),
                    pl.Field("geoPoint", pl.Struct([pl.Field("lat", pl.Float64), pl.Field("lon", pl.Float64)])),
                ])
            ),
        ),
    ]).filter(pl.col("nct_id").is_not_null())

    base = base.with_columns(
        pl.struct(["nct_id", "first_received_date"])
        .map_elements(
            lambda x: _generate_coreason_id_udf(x["nct_id"], x["first_received_date"]), return_dtype=pl.String
        )
        .alias("coreason_id")
    )

    exploded = base.explode("locations").filter(pl.col("locations").is_not_null())

    ex_schema = exploded.collect_schema()
    loc_dtype = ex_schema.get("locations")

    def safe_extract(
        col_name: str, field: str, alias: str, dtype: pl.DataType = DEFAULT_STRING_TYPE
    ) -> pl.Expr:
        if isinstance(loc_dtype, pl.Struct):
            if any(f.name == field for f in loc_dtype.fields):
                return pl.col(col_name).struct.field(field).alias(alias)
        return pl.lit(None, dtype=dtype).alias(alias)

    df = (
        exploded.select([
            pl.col("nct_id"),
            pl.col("coreason_id"),
            safe_extract("locations", "facility", "facility"),
            safe_extract("locations", "city", "city"),
            safe_extract("locations", "state", "state"),
            safe_extract("locations", "country", "country"),
            safe_extract("locations", "zip", "zip"),
            safe_extract("locations", "status", "status"),
            safe_extract(
                "locations",
                "geoPoint",
                "geo_point",
                pl.Struct([pl.Field("lat", pl.Float64), pl.Field("lon", pl.Float64)]),
            ),
        ])
        .collect()
    )

    def _gen_loc_id(row: dict[str, Any]) -> str:
        return _generate_surrogate_key_udf(row["nct_id"], row["facility"], row["city"], row["state"], row["country"])

    return (
        df.lazy()
        .with_columns(
            pl.struct(["nct_id", "facility", "city", "state", "country"])
            .map_elements(_gen_loc_id, return_dtype=pl.String)
            .alias("id")
        )
        .unique(subset=["id"], keep="first")
        .select([
            pl.col("id"),
            pl.col("nct_id").alias("source_id"),
            pl.col("coreason_id"),
            pl.col("facility"),
            pl.col("city"),
            pl.col("state"),
            pl.col("zip"),
            pl.col("country"),
            pl.col("status"),
            pl.col("geo_point"),
        ])
        .collect()
    )


def transform_to_silver_interventions(lf: pl.LazyFrame) -> pl.DataFrame:
    base = lf.select([
        _safe_get_field(lf, "protocolSection", ["identificationModule", "nctId"], "nct_id"),
        _safe_get_field(
            lf, "protocolSection", ["statusModule", "studyFirstPostDateStruct", "date"], "first_received_date"
        ),
        _safe_get_field(
            lf,
            "protocolSection",
            ["armsInterventionsModule", "interventions"],
            "interventions",
            pl.List(
                pl.Struct([
                    pl.Field("type", pl.String),
                    pl.Field("name", pl.String),
                    pl.Field("description", pl.String),
                    pl.Field("otherNames", pl.List(pl.String)),
                ])
            ),
        ),
    ]).filter(pl.col("nct_id").is_not_null())

    base = base.with_columns(
        pl.struct(["nct_id", "first_received_date"])
        .map_elements(
            lambda x: _generate_coreason_id_udf(x["nct_id"], x["first_received_date"]), return_dtype=pl.String
        )
        .alias("coreason_id")
    )

    exploded = base.explode("interventions").filter(pl.col("interventions").is_not_null())

    ex_schema = exploded.collect_schema()
    int_dtype = ex_schema.get("interventions")

    def safe_extract(field: str, alias: str, dtype: pl.DataType = DEFAULT_STRING_TYPE) -> pl.Expr:
        if isinstance(int_dtype, pl.Struct):
            if any(f.name == field for f in int_dtype.fields):
                return pl.col("interventions").struct.field(field).alias(alias)
        return pl.lit(None, dtype=dtype).alias(alias)

    df = (
        exploded.select([
            pl.col("nct_id"),
            pl.col("coreason_id"),
            safe_extract("type", "type"),
            safe_extract("name", "name"),
            safe_extract("description", "description"),
            safe_extract("otherNames", "other_names", pl.List(pl.String)),
        ])
        .collect()
    )

    def _gen_int_id(row: dict[str, Any]) -> str:
        return _generate_surrogate_key_udf(row["nct_id"], row["type"], row["name"])

    return (
        df.lazy()
        .with_columns(
            pl.struct(["nct_id", "type", "name"]).map_elements(_gen_int_id, return_dtype=pl.String).alias("id")
        )
        .unique(subset=["id"], keep="first")
        .select([
            pl.col("id"),
            pl.col("nct_id").alias("source_id"),
            pl.col("coreason_id"),
            pl.col("type"),
            pl.col("name"),
            pl.col("description"),
            pl.col("other_names"),
        ])
        .collect()
    )


def transform_to_silver_outcomes(lf: pl.LazyFrame) -> pl.DataFrame:
    base = lf.select([
        _safe_get_field(lf, "protocolSection", ["identificationModule", "nctId"], "nct_id"),
        _safe_get_field(
            lf, "protocolSection", ["statusModule", "studyFirstPostDateStruct", "date"], "first_received_date"
        ),
        _safe_get_field(lf, "protocolSection", ["outcomesModule"], "outcomes_mod"),
    ]).filter(pl.col("nct_id").is_not_null())

    base = base.with_columns(
        pl.struct(["nct_id", "first_received_date"])
        .map_elements(
            lambda x: _generate_coreason_id_udf(x["nct_id"], x["first_received_date"]), return_dtype=pl.String
        )
        .alias("coreason_id")
    )

    base_schema = base.collect_schema()
    mod_dtype = base_schema.get("outcomes_mod")

    dfs = []

    def process_outcome_list(field_name: str, outcome_type_label: str) -> Optional[pl.DataFrame]:
        has_field = False
        if isinstance(mod_dtype, pl.Struct):
            if any(f.name == field_name for f in mod_dtype.fields):
                has_field = True

        if not has_field:
            return None

        exploded = (
            base.select([
                pl.col("nct_id"),
                pl.col("coreason_id"),
                pl.col("outcomes_mod").struct.field(field_name).alias("outcomes"),
            ])
            .filter(pl.col("outcomes").is_not_null())
            .explode("outcomes")
        )

        ex_schema = exploded.collect_schema()
        o_dtype = ex_schema.get("outcomes")

        def safe_ex(f: str, alias: str) -> pl.Expr:
            if isinstance(o_dtype, pl.Struct) and any(x.name == f for x in o_dtype.fields):
                return pl.col("outcomes").struct.field(f).alias(alias)
            return pl.lit(None, pl.String).alias(alias)

        return exploded.select([
            pl.col("nct_id"),
            pl.col("coreason_id"),
            pl.lit(outcome_type_label).alias("outcome_type"),
            safe_ex("measure", "measure"),
            safe_ex("timeFrame", "time_frame"),
            safe_ex("description", "description"),
        ]).collect()

    for field, label in [
        ("primaryOutcomes", "PRIMARY"),
        ("secondaryOutcomes", "SECONDARY"),
        ("otherOutcomes", "OTHER"),
    ]:
        res = process_outcome_list(field, label)
        if res is not None and res.height > 0:
            dfs.append(res)

    if not dfs:
        return pl.DataFrame(
            schema={
                "id": pl.String,
                "source_id": pl.String,
                "coreason_id": pl.String,
                "outcome_type": pl.String,
                "measure": pl.String,
                "description": pl.String,
                "time_frame": pl.String,
            }
        )

    combined = pl.concat(dfs)

    def _gen_outcome_id(row: dict[str, Any]) -> str:
        return _generate_surrogate_key_udf(row["nct_id"], row["outcome_type"], row["measure"], row["time_frame"])

    return (
        combined.lazy()
        .with_columns(
            pl.struct(["nct_id", "outcome_type", "measure", "time_frame"])
            .map_elements(_gen_outcome_id, return_dtype=pl.String)
            .alias("id")
        )
        .unique(subset=["id"], keep="first")
        .select([
            pl.col("id"),
            pl.col("nct_id").alias("source_id"),
            pl.col("coreason_id"),
            pl.col("outcome_type"),
            pl.col("measure"),
            pl.col("description"),
            pl.col("time_frame"),
        ])
        .collect()
    )


def transform_to_silver_references(lf: pl.LazyFrame) -> pl.DataFrame:
    base = lf.select([
        _safe_get_field(lf, "protocolSection", ["identificationModule", "nctId"], "nct_id"),
        _safe_get_field(
            lf, "protocolSection", ["statusModule", "studyFirstPostDateStruct", "date"], "first_received_date"
        ),
        # Fixed: pass dtype as List so missing fields return empty list-like null, not string null
        _safe_get_field(
            lf,
            "protocolSection",
            ["referencesModule", "references"],
            "references",
            pl.List(
                pl.Struct([
                    pl.Field("pmid", pl.String),
                    pl.Field("citation", pl.String),
                    pl.Field("retraction", pl.Struct([pl.Field("retraction", pl.Boolean)])),
                ])
            ),
        ),
    ]).filter(pl.col("nct_id").is_not_null())

    base = base.with_columns(
        pl.struct(["nct_id", "first_received_date"])
        .map_elements(
            lambda x: _generate_coreason_id_udf(x["nct_id"], x["first_received_date"]), return_dtype=pl.String
        )
        .alias("coreason_id")
    )

    exploded = base.explode("references").filter(pl.col("references").is_not_null())

    ex_schema = exploded.collect_schema()
    ref_dtype = ex_schema.get("references")

    def safe_ex(f: str, alias: str, dtype: pl.DataType = DEFAULT_STRING_TYPE) -> pl.Expr:
        if isinstance(ref_dtype, pl.Struct) and any(x.name == f for x in ref_dtype.fields):
            return pl.col("references").struct.field(f).alias(alias)
        return pl.lit(None, dtype).alias(alias)

    df = (
        exploded.select([
            pl.col("nct_id"),
            pl.col("coreason_id"),
            safe_ex("pmid", "pmid"),
            safe_ex("citation", "citation"),
            safe_ex("retraction", "retraction", pl.Struct([pl.Field("retraction", pl.Boolean)])),
            pl.lit("REFERENCE").alias("type"),
        ])
        .collect()
    )

    def _gen_ref_id(row: dict[str, Any]) -> str:
        return _generate_surrogate_key_udf(row["nct_id"], row["pmid"], row["citation"])

    return (
        df.lazy()
        .with_columns(pl.col("pmid").cast(pl.String))
        .with_columns(
            pl.struct(["nct_id", "pmid", "citation"]).map_elements(_gen_ref_id, return_dtype=pl.String).alias("id")
        )
        .unique(subset=["id"], keep="first")
        .select([
            pl.col("id"),
            pl.col("nct_id").alias("source_id"),
            pl.col("coreason_id"),
            pl.col("type"),
            pl.col("pmid"),
            pl.col("citation"),
            pl.col("retraction"),
        ])
        .collect()
    )
