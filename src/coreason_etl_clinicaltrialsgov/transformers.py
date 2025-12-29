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
from datetime import date
from typing import Any, Optional

from coreason_etl_clinicaltrialsgov.schemas import (
    GoldStudy,
    SilverIntervention,
    SilverLocation,
    SilverOutcome,
    SilverReference,
    SilverSponsor,
    SilverStudy,
)

# --- Helpers ---


def parse_date(date_str: Optional[str]) -> Optional[date]:
    """Parse partial dates into a date object."""
    if not date_str:
        return None
    try:
        parts = date_str.split("-")
        if len(parts) == 3:
            return date.fromisoformat(date_str)
        elif len(parts) == 2:
            return date.fromisoformat(f"{date_str}-01")
        elif len(parts) == 1:
            return date.fromisoformat(f"{date_str}-01-01")
    except ValueError:
        return None
    return None


def normalize_age(age_str: Optional[str]) -> Optional[float]:
    """Normalize age string to years (float)."""
    if not age_str:
        return None

    parts = age_str.strip().lower().split()
    if not parts or len(parts) < 2:
        try:
            return float(parts[0])
        except (ValueError, IndexError):
            return None

    try:
        value = float(parts[0])
    except ValueError:
        return None

    unit = parts[1]
    if "month" in unit:
        return value / 12
    elif "week" in unit:
        return value / 52
    elif "day" in unit:
        return value / 365
    elif "year" in unit:
        return value
    return value


def generate_coreason_id(nct_id: str, first_received_date: Optional[str]) -> str:
    """Generate deterministic UUID for study."""
    seed = f"clinicaltrials.gov/{nct_id}/{first_received_date or ''}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))


def generate_surrogate_key(parent_id: str, *parts: str | None) -> str:
    """Generate deterministic UUID for child records."""
    # Concatenate all parts to form a unique seed for this record
    seed_parts = [parent_id]
    for p in parts:
        seed_parts.append(p or "")
    seed = "|".join(seed_parts)
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))


def flatten_phases(phases: Optional[list[str]]) -> Optional[str]:
    """Sort and join phases."""
    if not phases:
        return None
    return "|".join(sorted(phases))


def get_enrollment_bucket(count: Optional[int]) -> Optional[str]:
    """Categorize enrollment count."""
    if count is None:
        return None
    if count < 100:
        return "Small"
    elif count < 1000:
        return "Medium"
    else:
        return "Large"


# --- Transformation Logic ---


def transform_study(raw_study: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Transform a raw study dictionary into Silver layer tables."""

    protocol = raw_study.get("protocolSection", {})
    ident = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    design = protocol.get("designModule", {})
    eligibility = protocol.get("eligibilityModule", {})

    nct_id = ident.get("nctId")
    if not nct_id:
        return {}

    first_received = status.get("studyFirstPostDateStruct", {}).get("date")
    c_id = generate_coreason_id(nct_id, first_received)

    # 1. Silver Studies
    silver_study_model = SilverStudy(
        source_id=nct_id,
        coreason_id=c_id,
        title=ident.get("briefTitle"),
        official_title=ident.get("officialTitle"),
        org_study_id=ident.get("orgStudyIdInfo", {}).get("id"),
        overall_status=status.get("overallStatus"),
        start_date=parse_date(status.get("startDateStruct", {}).get("date")),
        completion_date=parse_date(status.get("completionDateStruct", {}).get("date")),
        phases=flatten_phases(design.get("phases")),
        study_type=design.get("studyType"),
        enrollment_count=design.get("enrollmentInfo", {}).get("count"),
        enrollment_type=design.get("enrollmentInfo", {}).get("type"),
        min_age=normalize_age(eligibility.get("minimumAge")),
        max_age=normalize_age(eligibility.get("maximumAge")),
        sex=eligibility.get("sex"),
        accepted_healthy_volunteers=eligibility.get("healthyVolunteers"),
    )

    # 2. Silver Sponsors
    silver_sponsors_list = []
    seen_sponsors = set()

    sponsors_module = protocol.get("sponsorCollaboratorsModule", {})
    lead = sponsors_module.get("leadSponsor")
    if lead:
        # Unique ID: nct_id + role + name
        name = lead.get("name")
        s_id = generate_surrogate_key(nct_id, "LEAD", name)
        if s_id not in seen_sponsors:
            seen_sponsors.add(s_id)
            silver_sponsors_list.append(
                SilverSponsor(
                    id=s_id,
                    source_id=nct_id,
                    coreason_id=c_id,
                    name=name,
                    agency_class=lead.get("class"),
                    role="LEAD",
                )
            )

    collaborators = sponsors_module.get("collaborators", [])
    for collab in collaborators:
        name = collab.get("name")
        s_id = generate_surrogate_key(nct_id, "COLLABORATOR", name)
        if s_id not in seen_sponsors:
            seen_sponsors.add(s_id)
            silver_sponsors_list.append(
                SilverSponsor(
                    id=s_id,
                    source_id=nct_id,
                    coreason_id=c_id,
                    name=name,
                    agency_class=collab.get("class"),
                    role="COLLABORATOR",
                )
            )

    # 3. Silver Locations
    silver_locations_list = []
    seen_locations = set()
    locations_module = protocol.get("contactsLocationsModule", {})
    locations = locations_module.get("locations", [])
    for loc in locations:
        # Unique ID: nct_id + facility + city + country
        # Note: multiple locations could have same facility name? Hopefully distinct enough.
        # Adding geo_point? Maybe not stable if float.
        # Let's use facility, city, state, country.
        facility = loc.get("facility")
        city = loc.get("city")
        state = loc.get("state")
        country = loc.get("country")

        # If all are None, this might duplicate? But locations usually have some info.
        # We'll include them all.
        s_id = generate_surrogate_key(nct_id, facility, city, state, country)

        if s_id not in seen_locations:
            seen_locations.add(s_id)
            silver_locations_list.append(
                SilverLocation(
                    id=s_id,
                    source_id=nct_id,
                    coreason_id=c_id,
                    facility=facility,
                    city=city,
                    state=state,
                    zip=loc.get("zip"),
                    country=country,
                    status=loc.get("status"),
                    geo_point=loc.get("geoPoint"),
                )
            )

    # 4. Silver Interventions
    silver_interventions_list = []
    seen_interventions = set()
    arms_module = protocol.get("armsInterventionsModule", {})
    interventions = arms_module.get("interventions", [])
    for interv in interventions:
        # ID: nct_id + type + name
        i_type = interv.get("type")
        i_name = interv.get("name")
        s_id = generate_surrogate_key(nct_id, i_type, i_name)

        if s_id not in seen_interventions:
            seen_interventions.add(s_id)
            silver_interventions_list.append(
                SilverIntervention(
                    id=s_id,
                    source_id=nct_id,
                    coreason_id=c_id,
                    type=i_type,
                    name=i_name,
                    description=interv.get("description"),
                    other_names=interv.get("otherNames", []),
                )
            )

    # 5. Silver Outcomes
    silver_outcomes_list = []
    seen_outcomes = set()
    outcomes_module = protocol.get("outcomesModule", {})
    for outcome_type in ["primaryOutcomes", "secondaryOutcomes", "otherOutcomes"]:
        outcomes = outcomes_module.get(outcome_type, [])
        normalized_type = outcome_type.replace("Outcomes", "").upper()
        for out in outcomes:
            # ID: nct_id + type + measure + time_frame
            measure = out.get("measure")
            time_frame = out.get("timeFrame")
            s_id = generate_surrogate_key(nct_id, normalized_type, measure, time_frame)

            if s_id not in seen_outcomes:
                seen_outcomes.add(s_id)
                silver_outcomes_list.append(
                    SilverOutcome(
                        id=s_id,
                        source_id=nct_id,
                        coreason_id=c_id,
                        outcome_type=normalized_type,
                        measure=measure,
                        description=out.get("description"),
                        time_frame=time_frame,
                    )
                )

    # 6. Silver References
    silver_references_list = []
    seen_references = set()
    refs_module = protocol.get("referencesModule", {})

    refs = refs_module.get("references", [])
    for ref in refs:
        # ID: nct_id + pmid + citation
        pmid = ref.get("pmid")
        citation = ref.get("citation")
        s_id = generate_surrogate_key(nct_id, "REFERENCE", pmid, citation)

        if s_id not in seen_references:
            seen_references.add(s_id)
            silver_references_list.append(
                SilverReference(
                    id=s_id,
                    source_id=nct_id,
                    coreason_id=c_id,
                    type="REFERENCE",
                    pmid=pmid,
                    citation=citation,
                    retraction=ref.get("retraction"),
                )
            )

    return {
        "silver_studies": [silver_study_model.model_dump()],
        "silver_sponsors": [s.model_dump() for s in silver_sponsors_list],
        "silver_locations": [loc.model_dump() for loc in silver_locations_list],
        "silver_interventions": [i.model_dump() for i in silver_interventions_list],
        "silver_outcomes": [o.model_dump() for o in silver_outcomes_list],
        "silver_references": [r.model_dump() for r in silver_references_list],
    }


def transform_gold(
    raw_study: dict[str, Any], silver_study: dict[str, Any], locations: list[dict[str, Any]]
) -> Optional[dict[str, Any]]:
    """Transform to Gold layer. Returns None if filtered out."""

    valid_statuses = {"RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED", "SUSPENDED", "TERMINATED"}
    overall_status = silver_study.get("overall_status")

    # Filter
    if not overall_status or overall_status.upper() not in valid_statuses:
        return None

    # Derived Columns
    start = silver_study.get("start_date")
    end = silver_study.get("completion_date")
    years_active = None
    if start and end:
        # Calculate years as float
        if isinstance(start, str):
            start = date.fromisoformat(start)
        if isinstance(end, str):
            end = date.fromisoformat(end)

        delta = end - start
        years_active = delta.days / 365.25

    enrollment = silver_study.get("enrollment_count")
    bucket = get_enrollment_bucket(enrollment)

    # Check raw for results section presence
    has_results = "resultsSection" in raw_study

    # Geo countries
    geo_countries = list({loc.get("country") for loc in locations if loc.get("country")})

    # Create Gold Model
    gold_model = GoldStudy(
        source_id=silver_study.get("source_id"),
        coreason_id=silver_study.get("coreason_id"),
        title=silver_study.get("title"),
        overall_status=overall_status,
        enrollment_bucket=bucket,
        years_active=years_active,
        has_results=has_results,
        geo_countries=geo_countries,
    )

    return gold_model.model_dump()
