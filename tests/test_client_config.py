# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

from unittest.mock import Mock

import pytest
import requests

from coreason_etl_clinicaltrialsgov.client import ClinicalTrialsClient, wait_for_retry_after
from coreason_etl_clinicaltrialsgov.config import settings


@pytest.fixture
def mock_session() -> Mock:
    return Mock(spec=requests.Session)


def test_client_uses_default_settings(mock_session: Mock) -> None:
    """Test that the client uses default settings from config."""
    client = ClinicalTrialsClient(session=mock_session)
    mock_response = Mock()
    mock_response.json.return_value = {"studies": []}
    mock_response.status_code = 200
    mock_session.get.return_value = mock_response

    client.fetch_studies()

    mock_session.get.assert_called_once()
    # Check timeout from default settings
    assert mock_session.get.call_args.kwargs["timeout"] == settings.API_TIMEOUT
    # Check page_size
    assert mock_session.get.call_args.kwargs["params"]["pageSize"] == settings.API_PAGE_SIZE


def test_client_propagates_config_changes(mock_session: Mock) -> None:
    """Test that runtime changes to settings affect the client call."""
    # Temporarily modify settings
    original_timeout = settings.API_TIMEOUT
    original_page_size = settings.API_PAGE_SIZE

    try:
        settings.API_TIMEOUT = 1
        settings.API_PAGE_SIZE = 10

        client = ClinicalTrialsClient(session=mock_session)
        mock_response = Mock()
        mock_response.json.return_value = {"studies": []}
        mock_session.get.return_value = mock_response

        client.fetch_studies()

        assert mock_session.get.call_args.kwargs["timeout"] == 1
        assert mock_session.get.call_args.kwargs["params"]["pageSize"] == 10
    finally:
        # Restore settings
        settings.API_TIMEOUT = original_timeout
        settings.API_PAGE_SIZE = original_page_size


def test_custom_wait_strategy_date_parsing() -> None:
    """Test complex edge cases for Retry-After header parsing."""
    # We mock the fallback to return a sentinel value
    fallback = Mock(return_value=1.0)
    wait_strategy = wait_for_retry_after(fallback=fallback)

    # Mock RetryCallState and Exception
    mock_retry_state = Mock()
    mock_exc = requests.HTTPError()
    mock_response = Mock()
    mock_response.status_code = 429

    # Case 1: Integer seconds
    mock_response.headers = {"Retry-After": "120"}
    mock_exc.response = mock_response
    mock_retry_state.outcome.exception.return_value = mock_exc

    assert wait_strategy(mock_retry_state) == 120.0

    # Case 2: HTTP Date (future)
    # We need to mock datetime.now to make this deterministic
    future_date = "Fri, 31 Dec 2099 23:59:59 GMT"
    mock_response.headers = {"Retry-After": future_date}

    # Calculate expected diff roughly (just ensure it's > 0 and huge)
    val = wait_strategy(mock_retry_state)
    assert val > 100000

    # Case 3: HTTP Date (past) -> Should return 0.0 (immediate retry)
    past_date = "Fri, 31 Dec 1999 23:59:59 GMT"
    mock_response.headers = {"Retry-After": past_date}
    assert wait_strategy(mock_retry_state) == 0.0

    # Case 4: Invalid Date -> Fallback
    mock_response.headers = {"Retry-After": "Invalid Date String"}
    assert wait_strategy(mock_retry_state) == 1.0  # Fallback value

    # Case 5: Negative Integer -> Fallback
    mock_response.headers = {"Retry-After": "-50"}
    assert wait_strategy(mock_retry_state) == 1.0
