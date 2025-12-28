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
        # Try to parse just the number if no unit, assume years? Or return None?
        # Example "18 Years".
        # If just "18", return 18.0? The spec examples usually have units.
        # Let's be safe and try to parse the first part as float if possible.
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
    # Hour/Minute? Unlikely for clinical trials eligibility, but treat as 0 or None?
    return value  # Default to value if unit unknown? Or years?

def generate_coreason_id(nct_id: str, first_received_date: Optional[str]) -> str:
    """Generate deterministic UUID."""
    # coreason_id: uuid5(uuid.NAMESPACE_DNS, "clinicaltrials.gov/" + nctId + "/" + firstReceivedDate)
    seed = f"clinicaltrials.gov/{nct_id}/{first_received_date or ''}"
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
        # Should not happen for valid records, but handle gracefully
        return {}

    first_received = status.get("studyFirstPostDateStruct", {}).get("date")
    c_id = generate_coreason_id(nct_id, first_received)

    # 1. Silver Studies
    silver_study = {
        "source_id": nct_id,
        "coreason_id": c_id,
        "title": ident.get("briefTitle"),
        "official_title": ident.get("officialTitle"),
        "org_study_id": ident.get("orgStudyIdInfo", {}).get("id"),
        "overall_status": status.get("overallStatus"),
        "start_date": parse_date(status.get("startDateStruct", {}).get("date")),
        "completion_date": parse_date(status.get("completionDateStruct", {}).get("date")),
        "phases": flatten_phases(design.get("phases")),
        "study_type": design.get("studyType"),
        "enrollment_count": design.get("enrollmentInfo", {}).get("count"),
        "enrollment_type": design.get("enrollmentInfo", {}).get("type"),
        "min_age": normalize_age(eligibility.get("minimumAge")),
        "max_age": normalize_age(eligibility.get("maximumAge")),
        "sex": eligibility.get("sex"),
        "accepted_healthy_volunteers": eligibility.get("healthyVolunteers"),
    }

    # 2. Silver Sponsors
    silver_sponsors = []
    sponsors_module = protocol.get("sponsorCollaboratorsModule", {})
    lead = sponsors_module.get("leadSponsor")
    if lead:
        silver_sponsors.append({
            "source_id": nct_id,
            "coreason_id": c_id,
            "name": lead.get("name"),
            "agency_class": lead.get("class"),
            "role": "LEAD"
        })

    collaborators = sponsors_module.get("collaborators", [])
    for collab in collaborators:
        silver_sponsors.append({
            "source_id": nct_id,
            "coreason_id": c_id,
            "name": collab.get("name"),
            "agency_class": collab.get("class"),
            "role": "COLLABORATOR"
        })

    # 3. Silver Locations
    silver_locations = []
    locations_module = protocol.get("contactsLocationsModule", {})
    locations = locations_module.get("locations", [])
    for loc in locations:
        silver_locations.append({
            "source_id": nct_id,
            "coreason_id": c_id,
            "facility": loc.get("facility"),
            "city": loc.get("city"),
            "state": loc.get("state"),
            "zip": loc.get("zip"),
            "country": loc.get("country"),
            "status": loc.get("status"), # Location status
            "geo_point": loc.get("geoPoint") # If available
        })

    # 4. Silver Interventions
    silver_interventions = []
    arms_module = protocol.get("armsInterventionsModule", {})
    interventions = arms_module.get("interventions", [])
    for interv in interventions:
        silver_interventions.append({
            "source_id": nct_id,
            "coreason_id": c_id,
            "type": interv.get("type"),
            "name": interv.get("name"),
            "description": interv.get("description"),
            "other_names": interv.get("otherNames", [])
        })

    # 5. Silver Outcomes
    silver_outcomes = []
    outcomes_module = protocol.get("outcomesModule", {})
    for outcome_type in ["primaryOutcomes", "secondaryOutcomes", "otherOutcomes"]:
        outcomes = outcomes_module.get(outcome_type, [])
        for out in outcomes:
            silver_outcomes.append({
                "source_id": nct_id,
                "coreason_id": c_id,
                "outcome_type": outcome_type.replace("Outcomes", "").upper(), # PRIMARY, SECONDARY, OTHER
                "measure": out.get("measure"),
                "description": out.get("description"),
                "time_frame": out.get("timeFrame")
            })

    # 6. Silver References
    silver_references = []
    refs_module = protocol.get("referencesModule", {})

    refs = refs_module.get("references", [])
    for ref in refs:
        silver_references.append({
            "source_id": nct_id,
            "coreason_id": c_id,
            "type": "REFERENCE",
            "pmid": ref.get("pmid"),
            "citation": ref.get("citation"),
            "retraction": ref.get("retraction")
        })

    links = refs_module.get("seeAlsoLinks", [])
    for link in links:
        silver_references.append({
             "source_id": nct_id,
             "coreason_id": c_id,
             "type": "LINK",
             "label": link.get("label"),
             "url": link.get("url")
        })

    return {
        "silver_studies": [silver_study],
        "silver_sponsors": silver_sponsors,
        "silver_locations": silver_locations,
        "silver_interventions": silver_interventions,
        "silver_outcomes": silver_outcomes,
        "silver_references": silver_references
    }

def transform_gold(raw_study: dict[str, Any], silver_study: dict[str, Any], locations: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
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
        delta = end - start
        years_active = delta.days / 365.25

    enrollment = silver_study.get("enrollment_count")
    bucket = get_enrollment_bucket(enrollment)

    # Check raw for results section presence
    has_results = "resultsSection" in raw_study

    # Geo countries
    geo_countries = list({loc.get("country") for loc in locations if loc.get("country")})

    return {
        "source_id": silver_study.get("source_id"),
        "coreason_id": silver_study.get("coreason_id"),
        "title": silver_study.get("title"),
        "overall_status": overall_status,
        "enrollment_bucket": bucket,
        "years_active": years_active,
        "has_results": has_results,
        "geo_countries": geo_countries
    }
