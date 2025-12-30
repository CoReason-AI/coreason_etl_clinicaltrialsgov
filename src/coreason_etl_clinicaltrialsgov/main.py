# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

from typing import Optional

import dlt
import typer
from typing_extensions import Annotated

from coreason_etl_clinicaltrialsgov.config import settings
from coreason_etl_clinicaltrialsgov.extractors import clinicaltrials_source
from coreason_etl_clinicaltrialsgov.utils.logger import logger

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.callback()
def main() -> None:
    """ClinicalTrials.gov ETL CLI."""


@app.command()
def run(
    page_size: Annotated[int, typer.Option(help="API page size")] = settings.API_PAGE_SIZE,
    query_term: Annotated[Optional[str], typer.Option(help="Optional query term for filtering")] = None,
    destination: Annotated[str, typer.Option(help="DLT destination")] = "postgres",
    pipeline_name: Annotated[str, typer.Option(help="DLT pipeline name")] = "clinicaltrials_etl",
    dataset_name: Annotated[str, typer.Option(help="DLT dataset name")] = "clinical_trials_data",
) -> None:
    """Run the ClinicalTrials.gov ETL pipeline."""
    try:
        # Configure pipeline
        pipeline = dlt.pipeline(
            pipeline_name=pipeline_name,
            destination=destination,
            dataset_name=dataset_name,
            progress="log",
        )

        logger.info(
            f"Starting extraction with page_size={page_size}, query_term={query_term}, destination={destination}"
        )

        source = clinicaltrials_source(page_size=page_size, query_term=query_term)

        info = pipeline.run(source)

        logger.info(f"Pipeline finished. Load info: {info}")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":  # pragma: no cover
    app()
