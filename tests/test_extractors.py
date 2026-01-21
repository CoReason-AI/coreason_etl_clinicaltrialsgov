# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest
from dlt.extract.exceptions import ResourceExtractionError
from dlt.extract.items import DataItemWithMeta

from coreason_etl_clinicaltrialsgov.extractors import clinicaltrials_source


@pytest.fixture
def mock_client_class() -> Generator[MagicMock, None, None]:
    with patch("coreason_etl_clinicaltrialsgov.extractors.ClinicalTrialsClient") as mock:
        yield mock


@pytest.fixture
def mock_transform_gold() -> Generator[MagicMock, None, None]:
    with patch("coreason_etl_clinicaltrialsgov.extractors.transform_gold") as mock:
        yield mock


# Mock Polars transformers
@pytest.fixture
def mock_polars_transformers() -> Generator[dict[str, MagicMock], None, None]:
    modules = [
        "transform_to_silver_studies",
        "transform_to_silver_sponsors",
        "transform_to_silver_locations",
        "transform_to_silver_interventions",
        "transform_to_silver_outcomes",
        "transform_to_silver_references",
        "transform_to_silver_officials",
    ]
    mocks = {}
    patchers = []

    for mod in modules:
        p = patch(f"coreason_etl_clinicaltrialsgov.extractors.{mod}")
        m = p.start()
        mocks[mod] = m
        patchers.append(p)

    yield mocks

    for p in patchers:
        p.stop()


def test_clinicaltrials_source_structure() -> None:
    source = clinicaltrials_source()
    assert source.name == "clinicaltrials"
    assert "studies_stream" in source.resources


def test_studies_generator_flow(
    mock_client_class: MagicMock, mock_polars_transformers: dict[str, MagicMock], mock_transform_gold: MagicMock
) -> None:
    # Setup mocks
    client_instance = mock_client_class.return_value

    # Mock data
    raw_study = {"protocolSection": {"identificationModule": {"nctId": "NCT001"}}}
    client_instance.list_studies.return_value = iter([raw_study])

    # Helper to create mock DataFrame
    def mock_df(data: list[dict[str, Any]]) -> MagicMock:
        m = MagicMock()
        m.to_dicts.return_value = data
        return m

    # Setup transformer returns
    # Must match Pydantic schemas (SilverStudy requires coreason_id)
    mock_polars_transformers["transform_to_silver_studies"].return_value = mock_df(
        [{"source_id": "NCT001", "coreason_id": "CID1", "overall_status": "RECRUITING"}]
    )
    # SilverSponsor requires id, source_id, coreason_id, role
    mock_polars_transformers["transform_to_silver_sponsors"].return_value = mock_df([])
    # SilverLocation requires id, source_id, coreason_id
    mock_polars_transformers["transform_to_silver_locations"].return_value = mock_df(
        [{"id": "L1", "source_id": "NCT001", "coreason_id": "CID1", "city": "Boston"}]
    )
    mock_polars_transformers["transform_to_silver_interventions"].return_value = mock_df([])
    mock_polars_transformers["transform_to_silver_outcomes"].return_value = mock_df([])
    mock_polars_transformers["transform_to_silver_references"].return_value = mock_df([])
    mock_polars_transformers["transform_to_silver_officials"].return_value = mock_df([])

    mock_transform_gold.return_value = {"gold_field": "val"}

    # Run generator directly
    source = clinicaltrials_source()
    resource = source.resources["studies_stream"]

    # We can iterate the resource
    items = list(resource)

    def extract_data(item: Any) -> Any:
        while isinstance(item, DataItemWithMeta):
            item = item.data
        return item

    # Check Bronze
    bronze_item = next(
        i for i in items if isinstance(extract_data(i), dict) and extract_data(i).get("raw_payload") == raw_study
    )
    bronze = extract_data(bronze_item)
    assert bronze["source_id"] == "NCT001"
    # Verify table name
    if hasattr(bronze_item, "meta"):
        assert bronze_item.meta.table_name == "bronze_clinicaltrials_studies"

    # Check Silver
    silver_item = next(
        i for i in items if isinstance(extract_data(i), dict) and extract_data(i).get("overall_status") == "RECRUITING"
    )
    silver_study = extract_data(silver_item)
    assert silver_study["source_id"] == "NCT001"
    if hasattr(silver_item, "meta"):
        assert silver_item.meta.table_name == "silver_clinicaltrials_studies"

    silver_loc = next(i for i in items if isinstance(extract_data(i), dict) and extract_data(i).get("city") == "Boston")
    assert extract_data(silver_loc) is not None

    # Check Gold
    gold_item = next(
        i for i in items if isinstance(extract_data(i), dict) and extract_data(i).get("gold_field") == "val"
    )
    gold = extract_data(gold_item)
    assert gold is not None
    if hasattr(gold_item, "meta"):
        assert gold_item.meta.table_name == "gold_clinicaltrials_studies"


def test_studies_generator_skip_no_nct(mock_client_class: MagicMock) -> None:
    client_instance = mock_client_class.return_value
    # Study with no nctId
    client_instance.list_studies.return_value = iter([{"protocolSection": {}}])

    source = clinicaltrials_source()
    resource = source.resources["studies_stream"]
    items = list(resource)

    assert len(items) == 0


def test_studies_generator_gold_skip(
    mock_client_class: MagicMock, mock_polars_transformers: dict[str, MagicMock], mock_transform_gold: MagicMock
) -> None:
    client_instance = mock_client_class.return_value
    raw_study = {"protocolSection": {"identificationModule": {"nctId": "NCT001"}}}
    client_instance.list_studies.return_value = iter([raw_study])

    def mock_df(data: list[dict[str, Any]]) -> MagicMock:
        m = MagicMock()
        m.to_dicts.return_value = data
        return m

    # Mock valid study
    mock_polars_transformers["transform_to_silver_studies"].return_value = mock_df(
        [{"source_id": "NCT001", "coreason_id": "CID1"}]
    )
    # Other transformers return empty
    for k, m in mock_polars_transformers.items():
        if k != "transform_to_silver_studies":
            m.return_value = mock_df([])

    # Gold returns None (filtered out)
    mock_transform_gold.return_value = None

    source = clinicaltrials_source()
    resource = source.resources["studies_stream"]
    items = list(resource)

    def extract_data(item: Any) -> Any:
        while isinstance(item, DataItemWithMeta):
            item = item.data
        return item

    unwrapped = [extract_data(i) for i in items]

    # Should contain Bronze + Silver, but no Gold
    assert any(i.get("source_id") == "NCT001" for i in unwrapped if isinstance(i, dict))

    # Verify mock call
    mock_transform_gold.assert_called_once()


def test_studies_generator_batching_and_exception(
    mock_client_class: MagicMock, mock_polars_transformers: dict[str, MagicMock]
) -> None:
    # Test batching logic: page_size=2, total=3 items. Should yield batch of 2 then batch of 1.
    client_instance = mock_client_class.return_value
    studies = [
        {"protocolSection": {"identificationModule": {"nctId": "NCT1"}}},
        {"protocolSection": {"identificationModule": {"nctId": "NCT2"}}},
        {"protocolSection": {"identificationModule": {"nctId": "NCT3"}}},
    ]
    client_instance.list_studies.return_value = iter(studies)

    def mock_df(data: list[dict[str, Any]]) -> MagicMock:
        m = MagicMock()
        m.to_dicts.return_value = data
        return m

    # Return empty valid data for Silver tables to pass validation loop
    for _, m in mock_polars_transformers.items():
        m.return_value = mock_df([])

    # Use page_size=2
    source = clinicaltrials_source(page_size=2)
    resource = source.resources["studies_stream"]
    items = list(resource)

    # 3 studies -> 3 Bronze + Silver overhead
    # We just check we got all 3 bronze
    def extract_data(item: Any) -> Any:
        while isinstance(item, DataItemWithMeta):
            item = item.data
        return item

    unwrapped = [extract_data(i) for i in items]
    bronze_ids = {i["source_id"] for i in unwrapped if isinstance(i, dict) and "raw_payload" in i}
    assert bronze_ids == {"NCT1", "NCT2", "NCT3"}


def test_studies_generator_exception_handling(
    mock_client_class: MagicMock, mock_polars_transformers: dict[str, MagicMock]
) -> None:
    client_instance = mock_client_class.return_value
    studies = [{"protocolSection": {"identificationModule": {"nctId": "NCT1"}}}]
    client_instance.list_studies.return_value = iter(studies)

    # Force exception
    mock_polars_transformers["transform_to_silver_studies"].side_effect = ValueError("Polars Error")

    source = clinicaltrials_source()
    resource = source.resources["studies_stream"]

    # Check for ResourceExtractionError which wraps the ValueError
    with pytest.raises(ResourceExtractionError):
        list(resource)


def test_studies_generator_with_query_term(
    mock_client_class: MagicMock, mock_polars_transformers: dict[str, MagicMock]
) -> None:
    client_instance = mock_client_class.return_value
    client_instance.list_studies.return_value = iter([])

    # Just run to hit the log logic
    source = clinicaltrials_source(query_term="Term")
    list(source.resources["studies_stream"])


def test_studies_generator_high_water_mark(
    mock_client_class: MagicMock, mock_polars_transformers: dict[str, MagicMock]
) -> None:
    # Test high water mark update coverage
    client_instance = mock_client_class.return_value
    # One study with a date
    raw_study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_HWM"},
            "statusModule": {"lastUpdatePostDateStruct": {"date": "2023-12-31"}},
        }
    }
    client_instance.list_studies.return_value = iter([raw_study])

    def mock_df(data: list[dict[str, Any]]) -> MagicMock:
        m = MagicMock()
        m.to_dicts.return_value = data
        return m

    mock_polars_transformers["transform_to_silver_studies"].return_value = mock_df(
        [{"source_id": "NCT_HWM", "coreason_id": "CID"}]
    )
    for k, m in mock_polars_transformers.items():
        if k != "transform_to_silver_studies":
            m.return_value = mock_df([])

    with patch("dlt.current.source_state", return_value={}) as mock_state:
        source = clinicaltrials_source()
        list(source.resources["studies_stream"])
        # Check if state was updated
        assert mock_state.return_value["last_updated_date"] == "2023-12-31"


def test_studies_generator_incremental_auto_filter(
    mock_client_class: MagicMock, mock_polars_transformers: dict[str, MagicMock]
) -> None:
    """Test that query term is automatically generated if state exists."""
    client_instance = mock_client_class.return_value
    client_instance.list_studies.return_value = iter([])

    last_date = "2023-01-01"

    with patch("dlt.current.source_state", return_value={"last_updated_date": last_date}):
        source = clinicaltrials_source()  # No query_term provided
        list(source.resources["studies_stream"])

        # Verify client called with correct filter
        expected_term = f"AREA[LastUpdatePostDate]RANGE[{last_date},MAX]"
        client_instance.list_studies.assert_called_with(page_size=100, query_term=expected_term)
