# File: coreason_etl_clinicaltrialsgov1/src/coreason_etl_clinicaltrialsgov/extractors.py

from datetime import datetime, timezone
from typing import Any, Iterator, Optional, Type, Literal

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
    SilverOutcome,
    SilverReference,
    SilverSponsor,
    SilverStudy,
)
from coreason_etl_clinicaltrialsgov.transformers import transform_gold
from coreason_etl_clinicaltrialsgov.transformers_polars import (
    transform_to_silver_interventions,
    transform_to_silver_locations,
    transform_to_silver_outcomes,
    transform_to_silver_references,
    transform_to_silver_sponsors,
    transform_to_silver_studies,
)


@dlt.source(name="clinicaltrials")
def clinicaltrials_source(
    page_size: int = 100, 
    query_term: Optional[str] = None,
    target_schema: Literal["bronze", "silver", "gold", "all"] = "all"
) -> Iterator[DltResource]:
    """
    The ClinicalTrials.gov V2 API source.
    Produces Bronze, Silver, and Gold tables.
    Args:
        target_schema: Filters output to a specific schema layer ('bronze', 'silver', 'gold') or 'all'.
    """

    @dlt.resource(name="studies_stream", write_disposition="merge", primary_key="source_id")
    def studies_generator() -> Iterator[TDataItems]:
        client = ClinicalTrialsClient()

        state = dlt.current.source_state()
        last_date = state.get("last_updated_date")

        current_query_term = query_term
        if not current_query_term and last_date:
            current_query_term = f"AREA[LastUpdatePostDate]RANGE[{last_date},MAX]"
            logger.info(f"Incremental load enabled for {target_schema}. Filter: {current_query_term}")
        elif not current_query_term:
            logger.info(f"Initial load (Full extraction) for {target_schema}. No filter.")
        else:
            logger.info(f"Custom query term provided for {target_schema}: {current_query_term}")

        max_date_seen = last_date
        batch_size = page_size
        current_batch: list[dict[str, Any]] = []
        total_records_extracted = 0

        def process_batch(batch: list[dict[str, Any]]) -> Iterator[TDataItems]:
            now_ts = datetime.now(timezone.utc).isoformat()

            # 1. Yield Bronze (Only if target is bronze or all)
            if target_schema in ("bronze", "all"):
                for raw_study in batch:
                    nct_id = raw_study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
                    bronze_record = {"source_id": nct_id, "ingestion_ts": now_ts, "raw_payload": raw_study}
                    
                    yield dlt.mark.with_hints(
                        bronze_record,
                        dlt.mark.make_hints(
                            table_name="bronze_clinicaltrials_studies",
                            write_disposition="merge", 
                            primary_key="source_id"
                        )
                    )

            # Prepare data for Silver/Gold processing (needed even if we only yield Gold)
            lf = pl.DataFrame(batch).lazy()

            # We must compute Silver dataframes to generate Gold records
            try:
                df_studies = transform_to_silver_studies(lf)
                df_sponsors = transform_to_silver_sponsors(lf)
                df_locations = transform_to_silver_locations(lf)
                df_interventions = transform_to_silver_interventions(lf)
                df_outcomes = transform_to_silver_outcomes(lf)
                df_references = transform_to_silver_references(lf)
            except Exception as e:
                logger.error(f"Error in Polars transformation: {e}")
                raise e

            studies_dict = {row["source_id"]: row for row in df_studies.to_dicts()}

            # 2. Yield Silver (Only if target is silver or all)
            if target_schema in ("silver", "all"):
                silver_map: dict[str, tuple[pl.DataFrame, Type[BaseModel]]] = {
                    "silver_clinicaltrials_studies": (df_studies, SilverStudy),
                    "silver_clinicaltrials_sponsors": (df_sponsors, SilverSponsor),
                    "silver_clinicaltrials_locations": (df_locations, SilverLocation),
                    "silver_clinicaltrials_interventions": (df_interventions, SilverIntervention),
                    "silver_clinicaltrials_outcomes": (df_outcomes, SilverOutcome),
                    "silver_clinicaltrials_references": (df_references, SilverReference),
                }

                for table_name, (df, model_class) in silver_map.items():
                    pk = "source_id" if table_name == "silver_clinicaltrials_studies" else "id"
                    
                    for record in df.to_dicts():
                        validated_model = model_class.model_validate(record)
                        validated_record = validated_model.model_dump()

                        yield dlt.mark.with_hints(
                            validated_record,
                            dlt.mark.make_hints(
                                table_name=table_name,
                                write_disposition="merge", 
                                primary_key=pk
                            )
                        )

            # 3. Yield Gold (Only if target is gold or all)
            if target_schema in ("gold", "all"):
                locations_lookup: dict[str, list[dict[str, Any]]] = {}
                for loc in df_locations.to_dicts():
                    sid = loc["source_id"]
                    if sid not in locations_lookup:
                        locations_lookup[sid] = []
                    locations_lookup[sid].append(loc)

                sponsors_lookup: dict[str, list[dict[str, Any]]] = {}
                for sp in df_sponsors.to_dicts():
                    sid = sp["source_id"]
                    if sid not in sponsors_lookup:
                        sponsors_lookup[sid] = []
                    sponsors_lookup[sid].append(sp)

                for raw_study in batch:
                    nct_id = raw_study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
                    if not nct_id or nct_id not in studies_dict:
                        continue

                    silver_study = studies_dict[nct_id]
                    silver_locs = locations_lookup.get(nct_id, [])
                    silver_sponsors = sponsors_lookup.get(nct_id, [])

                    gold_record = transform_gold(raw_study, silver_study, silver_locs, silver_sponsors)
                    
                    if gold_record:
                        hints = dlt.mark.make_hints(
                            table_name="gold_clinicaltrials_studies",
                            write_disposition="merge", 
                            primary_key="source_id"
                        )
                        
                        hints["columns"] = {
                            "structural_attributes": {"data_type": "complex"},
                            "sponsors_details": {"data_type": "complex"}
                        }

                        yield dlt.mark.with_hints(gold_record, hints)

        for raw_study in client.list_studies(page_size=page_size, query_term=current_query_term):
            nct_id = raw_study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
            if not nct_id:
                continue

            status_mod = raw_study.get("protocolSection", {}).get("statusModule", {})
            study_date_str = status_mod.get("lastUpdatePostDateStruct", {}).get("date")
            if study_date_str:
                if not max_date_seen or study_date_str > max_date_seen:
                    max_date_seen = study_date_str

            current_batch.append(raw_study)
            total_records_extracted += 1

            if len(current_batch) >= batch_size:
                logger.info(f"Processing batch for {target_schema}: {len(current_batch)} records.")
                yield from process_batch(current_batch)
                current_batch = []

        if current_batch:
            logger.info(f"Processing final batch for {target_schema}: {len(current_batch)} records.")
            yield from process_batch(current_batch)

        if max_date_seen:
            state["last_updated_date"] = max_date_seen
            logger.info(f"Updated high water mark for {target_schema} to: {max_date_seen}")

    yield studies_generator
