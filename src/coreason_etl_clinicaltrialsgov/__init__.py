# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

from .client import ClinicalTrialsClient as ClinicalTrialsClient
from .extractors import clinicaltrials_source as clinicaltrials_source
from .main import run_pipeline as run_pipeline
from .transformers import transform_gold as transform_gold
from .transformers import transform_study as transform_study
