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
from typing import Iterator, Optional

import dlt
from dlt.common.typing import TDataItems
from dlt.sources import DltResource
from loguru import logger

from coreason_etl_clinicaltrialsgov.client import ClinicalTrialsClient
from coreason_etl_clinicaltrialsgov.transformers import transform_gold, transform_study


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

        for raw_study in client.list_studies(page_size=page_size, query_term=current_query_term):
            nct_id = raw_study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
            if not nct_id:
                continue

            # Update High Water Mark
            # Date format: YYYY-MM-DD
            status_mod = raw_study.get("protocolSection", {}).get("statusModule", {})
            study_date_str = status_mod.get("lastUpdatePostDateStruct", {}).get("date")
            if study_date_str:
                if not max_date_seen or study_date_str > max_date_seen:
                    max_date_seen = study_date_str

            now_ts = datetime.now(timezone.utc).isoformat()

            # 1. Bronze Layer
            bronze_record = {"source_id": nct_id, "ingestion_ts": now_ts, "raw_payload": raw_study}
            yield dlt.mark.with_hints(
                dlt.mark.with_table_name(bronze_record, "bronze_studies"),
                dlt.mark.make_hints(write_disposition="merge", primary_key="source_id"),
            )

            # 2. Silver Layer
            silver_data = transform_study(raw_study)

            silver_study_record = None
            if silver_data.get("silver_studies"):
                silver_study_record = silver_data["silver_studies"][0]

            for table_name, records in silver_data.items():
                if records:
                    for record in records:
                        if table_name == "silver_studies":
                            # Explicitly set merge and PK for main silver table
                            yield dlt.mark.with_hints(
                                dlt.mark.with_table_name(record, table_name),
                                dlt.mark.make_hints(write_disposition="merge", primary_key="source_id"),
                            )
                        else:
                            # Use new 'id' as PK for 1:N tables
                            yield dlt.mark.with_hints(
                                dlt.mark.with_table_name(record, table_name),
                                dlt.mark.make_hints(write_disposition="merge", primary_key="id"),
                            )

            # 3. Gold Layer
            if silver_study_record:
                locations = silver_data.get("silver_locations", [])
                gold_record = transform_gold(raw_study, silver_study_record, locations)
                if gold_record:
                    # Gold uses source_id (NCT ID) as PK
                    yield dlt.mark.with_hints(
                        dlt.mark.with_table_name(gold_record, "gold_studies"),
                        dlt.mark.make_hints(write_disposition="merge", primary_key="source_id"),
                    )

        # Save the new high water mark
        if max_date_seen:
            state["last_updated_date"] = max_date_seen
            logger.info(f"Updated high water mark to: {max_date_seen}")

    yield studies_generator
