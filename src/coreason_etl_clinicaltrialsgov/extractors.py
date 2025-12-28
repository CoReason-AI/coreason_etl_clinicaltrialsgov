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
from typing import Iterator, Optional, Any

import dlt
from dlt.common.typing import TDataItems
from dlt.sources import DltResource

from coreason_etl_clinicaltrialsgov.client import ClinicalTrialsClient
from coreason_etl_clinicaltrialsgov.transformers import transform_study, transform_gold

@dlt.source(name="clinicaltrials")
def clinicaltrials_source(
    page_size: int = 100,
    query_term: Optional[str] = None
) -> Iterator[DltResource]:
    """
    The ClinicalTrials.gov V2 API source.
    Produces Bronze, Silver, and Gold tables.
    """

    # We define a single resource that emits to multiple tables dynamically.
    # Alternatively, we could define separate resources but that might complicate the single-pass requirement
    # without caching. dlt supports dynamic table routing.

    @dlt.resource(name="studies_stream", write_disposition="merge", primary_key="source_id")
    def studies_generator() -> Iterator[TDataItems]:
        client = ClinicalTrialsClient()

        # We need to track load ID or similar? dlt handles load ids.
        # Bronze requires `_dlt_load_id`? dlt adds `_dlt_load_id` automatically.

        for raw_study in client.list_studies(page_size=page_size, query_term=query_term):
            nct_id = raw_study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
            if not nct_id:
                continue

            now_ts = datetime.now(timezone.utc).isoformat()

            # 1. Bronze Layer
            bronze_record = {
                "source_id": nct_id,
                "ingestion_ts": now_ts,
                "raw_payload": raw_study
            }
            yield dlt.mark.with_table_name(bronze_record, "bronze_studies")

            # 2. Silver Layer
            silver_data = transform_study(raw_study)
            # transform_study returns a dict of lists. It is never empty if nct_id is present.

            # Iterate over silver tables
            # silver_studies is a list of 1 dict usually
            silver_study_record = None
            if silver_data.get("silver_studies"):
                silver_study_record = silver_data["silver_studies"][0]

            for table_name, records in silver_data.items():
                if records:
                    for record in records:
                        yield dlt.mark.with_table_name(record, table_name)

            # 3. Gold Layer
            if silver_study_record:
                # We need locations for gold
                locations = silver_data.get("silver_locations", [])
                gold_record = transform_gold(raw_study, silver_study_record, locations)
                if gold_record:
                    yield dlt.mark.with_table_name(gold_record, "gold_studies")

    yield studies_generator
