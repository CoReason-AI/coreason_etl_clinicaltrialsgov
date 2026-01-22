# File: coreason_etl_clinicaltrialsgov1/src/coreason_etl_clinicaltrialsgov/main.py

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
    # Note: dataset_name argument is removed/ignored here because we enforce schema separation below
) -> None:
    """
    Run the ClinicalTrials.gov ETL pipeline.
    
    This command now enforces data separation into three distinct schemas:
    'bronze', 'silver', and 'gold'.
    """
    if page_size <= 0:
        logger.error(f"Invalid page_size: {page_size}. Must be positive.")
        raise typer.BadParameter("page_size must be positive")

    schemas = ["bronze", "silver", "gold"]
    
    logger.info(f"Starting Multi-Schema ETL load to: {schemas}")

    for schema in schemas:
        try:
            logger.info(f"--- Processing Schema: {schema} ---")
            
            # Configure pipeline specifically for this schema
            # This ensures dataset_name is exactly 'bronze', 'silver', or 'gold'
            pipeline = dlt.pipeline(
                pipeline_name=f"{pipeline_name}_{schema}",
                destination=destination,
                dataset_name=schema, 
                progress="log",
            )

            logger.info(
                f"Extracting for layer='{schema}' with page_size={page_size}, query_term={query_term}"
            )

            # Pass the target schema to the source to filter data
            source = clinicaltrials_source(
                page_size=page_size, 
                query_term=query_term, 
                target_schema=schema
            )

            info = pipeline.run(source)

            logger.info(f"Pipeline finished for schema '{schema}'. Load info: {info}")

        except Exception as e:
            logger.exception(f"Pipeline failed for schema '{schema}': {e}")
            raise typer.Exit(code=1) from e
            
    logger.info("All schemas processed successfully.")


if __name__ == "__main__":  # pragma: no cover
    app()
