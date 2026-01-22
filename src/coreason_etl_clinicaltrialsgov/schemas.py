# File: coreason_etl_clinicaltrialsgov/src/coreason_etl_clinicaltrialsgov/schemas.py

from datetime import date
from typing import Any, Optional, List

from pydantic import BaseModel, ConfigDict


class GeoPoint(BaseModel):
    lat: float
    lon: float
    model_config = ConfigDict(extra="ignore")


class SilverStudy(BaseModel):
    source_id: str
    coreason_id: str
    title: Optional[str] = None
    official_title: Optional[str] = None
    org_study_id: Optional[str] = None
    overall_status: Optional[str] = None
    start_date: Optional[date] = None
    completion_date: Optional[date] = None
    phases: Optional[str] = None
    study_type: Optional[str] = None
    enrollment_count: Optional[int] = None
    enrollment_type: Optional[str] = None
    min_age: Optional[float] = None
    max_age: Optional[float] = None
    sex: Optional[str] = None
    accepted_healthy_volunteers: Optional[bool] = None
    allocation: Optional[str] = None
    intervention_model: Optional[str] = None
    primary_purpose: Optional[str] = None
    study_population: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class SilverSponsor(BaseModel):
    id: str
    source_id: str
    coreason_id: str
    name: Optional[str] = None
    agency_class: Optional[str] = None
    role: str

    model_config = ConfigDict(extra="ignore")


class SilverLocation(BaseModel):
    id: str
    source_id: str
    coreason_id: str
    facility: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    country: Optional[str] = None
    status: Optional[str] = None
    geo_point: Optional[GeoPoint] = None

    model_config = ConfigDict(extra="ignore")


class SilverIntervention(BaseModel):
    id: str
    source_id: str
    coreason_id: str
    type: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    other_names: Optional[list[str]] = None

    model_config = ConfigDict(extra="ignore")


class SilverOutcome(BaseModel):
    id: str
    source_id: str
    coreason_id: str
    outcome_type: str
    measure: Optional[str] = None
    description: Optional[str] = None
    time_frame: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class SilverReference(BaseModel):
    id: str
    source_id: str
    coreason_id: str
    type: str
    pmid: Optional[str] = None
    citation: Optional[str] = None
    retraction: Optional[dict[str, Any]] = None

    model_config = ConfigDict(extra="ignore")


class GoldStudy(BaseModel):
    source_id: str
    coreason_id: str
    title: Optional[str] = None
    
    # REQUIRED: These fields must be here to be saved
    years_active: Optional[float] = None
    eligibility_criteria: Optional[str] = None
    brief_summary: Optional[str] = None
    
    # CHANGED: List[str] -> str to force a single DB column
    geo_countries: Optional[str] = None
    
    # JSON columns
    structural_attributes: dict[str, Any]
    sponsors_details: dict[str, Any]

    model_config = ConfigDict(extra="ignore")
