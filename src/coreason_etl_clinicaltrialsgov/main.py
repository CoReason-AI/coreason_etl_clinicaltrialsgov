# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov


import dlt

from coreason_etl_clinicaltrialsgov.extractors import clinicaltrials_source
from coreason_etl_clinicaltrialsgov.utils.logger import logger


@logger.catch  # type: ignore
def run_pipeline() -> None:
    """Run the ClinicalTrials.gov ETL pipeline."""

    # Configure pipeline
    # Destination is picked up from env vars or secrets.toml by dlt.
    # Defaulting to postgres if not specified, or duckdb for local?
    # BRD says "Stick to postgres".
    # User said "Stick to postgres".
    # We assume environment is configured.

    pipeline = dlt.pipeline(
        pipeline_name="clinicaltrials_etl", destination="postgres", dataset_name="clinical_trials_data", progress="log"
    )

    # Load data
    logger.info("Starting extraction...")

    # We can pass params from env vars or args if needed
    source = clinicaltrials_source()

    info = pipeline.run(source)

    logger.info(f"Pipeline finished. Load info: {info}")


if __name__ == "__main__":  # pragma: no cover
    run_pipeline()
