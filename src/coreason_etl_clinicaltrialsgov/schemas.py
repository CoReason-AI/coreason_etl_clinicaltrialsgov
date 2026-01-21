# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

from datetime import date
from typing import Any, Optional

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


class SilverOfficial(BaseModel):
    id: str
    source_id: str
    coreason_id: str
    name: Optional[str] = None
    role: Optional[str] = None
    affiliation: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class GoldStudy(BaseModel):
    source_id: str
    coreason_id: str
    title: Optional[str] = None
    overall_status: str
    enrollment_bucket: Optional[str] = None
    years_active: Optional[float] = None
    has_results: bool
    geo_countries: list[str]

    model_config = ConfigDict(extra="ignore")
