# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

from datetime import datetime, timezone
from typing import Any, Iterator, Optional, Type

import dlt
import polars as pl
from dlt.common.typing import TDataItems
from dlt.sources import DltResource
from loguru import logger
from pydantic import BaseModel

from coreason_etl_clinicaltrialsgov.client import ClinicalTrialsClient
from coreason_etl_clinicaltrialsgov.schemas import (
    SilverIntervention,
    SilverLocation,
    SilverOfficial,
    SilverOutcome,
    SilverReference,
    SilverSponsor,
    SilverStudy,
)
from coreason_etl_clinicaltrialsgov.transformers import transform_gold
from coreason_etl_clinicaltrialsgov.transformers_polars import (
    transform_to_silver_interventions,
    transform_to_silver_locations,
    transform_to_silver_officials,
    transform_to_silver_outcomes,
    transform_to_silver_references,
    transform_to_silver_sponsors,
    transform_to_silver_studies,
)


@dlt.source(name="clinicaltrials")
def clinicaltrials_source(page_size: int = 100, query_term: Optional[str] = None) -> Iterator[DltResource]:
    """
    The ClinicalTrials.gov V2 API source.
    Produces Bronze, Silver, and Gold tables.
    """

    @dlt.resource(name="studies_stream", write_disposition="merge", primary_key="source_id")
    def studies_generator() -> Iterator[TDataItems]:
        client = ClinicalTrialsClient()

        # State management for incremental loading
        state = dlt.current.source_state()
        last_date = state.get("last_updated_date")

        current_query_term = query_term
        if not current_query_term and last_date:
            current_query_term = f"AREA[LastUpdatePostDate]RANGE[{last_date},MAX]"
            logger.info(f"Incremental load enabled. Filter: {current_query_term}")
        elif not current_query_term:
            logger.info("Initial load (Full extraction). No filter.")
        else:
            logger.info(f"Custom query term provided: {current_query_term}")

        max_date_seen = last_date

        # Batch accumulation
        batch_size = page_size  # Use page_size as batch size for Polars processing
        current_batch: list[dict[str, Any]] = []
        total_records_extracted = 0

        def process_batch(batch: list[dict[str, Any]]) -> Iterator[TDataItems]:
            now_ts = datetime.now(timezone.utc).isoformat()

            # 1. Yield Bronze
            for raw_study in batch:
                nct_id = raw_study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
                bronze_record = {"source_id": nct_id, "ingestion_ts": now_ts, "raw_payload": raw_study}
                yield dlt.mark.with_hints(
                    dlt.mark.with_table_name(bronze_record, "bronze_clinicaltrials_studies"),
                    dlt.mark.make_hints(write_disposition="merge", primary_key="source_id"),
                )

            # 2. Silver Transformations via Polars
            lf = pl.DataFrame(batch).lazy()

            # Execute transformations
            # Note: We collect immediately because we need to yield rows to dlt
            try:
                df_studies = transform_to_silver_studies(lf)
                df_sponsors = transform_to_silver_sponsors(lf)
                df_locations = transform_to_silver_locations(lf)
                df_interventions = transform_to_silver_interventions(lf)
                df_outcomes = transform_to_silver_outcomes(lf)
                df_references = transform_to_silver_references(lf)
                df_officials = transform_to_silver_officials(lf)
            except Exception as e:
                logger.error(f"Error in Polars transformation: {e}")
                # Fallback or re-raise? Re-raise to ensure integrity.
                raise e

            # Map DataFrames to Table Names and Pydantic Models
            silver_map: dict[str, tuple[pl.DataFrame, Type[BaseModel]]] = {
                "silver_clinicaltrials_studies": (df_studies, SilverStudy),
                "silver_clinicaltrials_sponsors": (df_sponsors, SilverSponsor),
                "silver_clinicaltrials_locations": (df_locations, SilverLocation),
                "silver_clinicaltrials_interventions": (df_interventions, SilverIntervention),
                "silver_clinicaltrials_outcomes": (df_outcomes, SilverOutcome),
                "silver_clinicaltrials_references": (df_references, SilverReference),
                "silver_clinicaltrials_officials": (df_officials, SilverOfficial),
            }

            # Prepare lookup for Gold transformation (need Silver Study + Locations per NCT ID)
            # This requires joining or indexing.
            # Doing it in memory via dictionaries is easiest for 'parity' with existing logic.

            studies_dict = {row["source_id"]: row for row in df_studies.to_dicts()}

            # Collect locations for Gold (grouped by source_id)
            locations_lookup: dict[str, list[dict[str, Any]]] = {}
            for loc in df_locations.to_dicts():
                sid = loc["source_id"]
                if sid not in locations_lookup:
                    locations_lookup[sid] = []
                locations_lookup[sid].append(loc)

            # Yield Silver Records with Strict Validation
            for table_name, (df, model_class) in silver_map.items():
                pk = "source_id" if table_name == "silver_clinicaltrials_studies" else "id"
                for record in df.to_dicts():
                    # Validate via Pydantic
                    # This raises ValidationError if schema is violated
                    validated_model = model_class.model_validate(record)
                    validated_record = validated_model.model_dump()

                    yield dlt.mark.with_hints(
                        dlt.mark.with_table_name(validated_record, table_name),
                        dlt.mark.make_hints(write_disposition="merge", primary_key=pk),
                    )

            # 3. Yield Gold Records
            # Iterating through the batch again to match Gold logic
            for raw_study in batch:
                nct_id = raw_study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
                if not nct_id or nct_id not in studies_dict:
                    continue

                silver_study = studies_dict[nct_id]
                silver_locs = locations_lookup.get(nct_id, [])

                gold_record = transform_gold(raw_study, silver_study, silver_locs)
                if gold_record:
                    yield dlt.mark.with_hints(
                        dlt.mark.with_table_name(gold_record, "gold_clinicaltrials_studies"),
                        dlt.mark.make_hints(write_disposition="merge", primary_key="source_id"),
                    )

        for raw_study in client.list_studies(page_size=page_size, query_term=current_query_term):
            nct_id = raw_study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
            if not nct_id:
                continue

            # Update High Water Mark
            status_mod = raw_study.get("protocolSection", {}).get("statusModule", {})
            study_date_str = status_mod.get("lastUpdatePostDateStruct", {}).get("date")
            if study_date_str:
                if not max_date_seen or study_date_str > max_date_seen:
                    max_date_seen = study_date_str

            current_batch.append(raw_study)
            total_records_extracted += 1

            if len(current_batch) >= batch_size:
                logger.info(
                    f"Records Extracted: Processing batch of {len(current_batch)}. "
                    f"Total so far: {total_records_extracted}"
                )
                yield from process_batch(current_batch)
                current_batch = []

        # Process remaining
        if current_batch:
            logger.info(
                f"Records Extracted: Processing final batch of {len(current_batch)}. "
                f"Total so far: {total_records_extracted}"
            )
            yield from process_batch(current_batch)

        # Save the new high water mark
        if max_date_seen:
            state["last_updated_date"] = max_date_seen
            logger.info(f"Updated high water mark to: {max_date_seen}")

    yield studies_generator
