# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import requests
from coreason_etl_clinicaltrialsgov.client import ClinicalTrialsClient, wait_for_retry_after
from tenacity import RetryCallState


@pytest.fixture
def mock_retry_state() -> RetryCallState:
    state = MagicMock(spec=RetryCallState)
    state.outcome = MagicMock()
    return state


def test_wait_for_retry_after_seconds(mock_retry_state: RetryCallState) -> None:
    # Setup exception with 429 and Retry-After header
    response = requests.Response()
    response.status_code = 429
    response.headers["Retry-After"] = "10"
    exc = requests.HTTPError(response=response)

    # Cast because Mypy complains about 'exception' on Future | None
    cast(MagicMock, mock_retry_state.outcome).exception.return_value = exc

    # Setup fallback mock
    fallback = MagicMock()

    waiter = wait_for_retry_after(fallback=fallback)
    wait_time = waiter(mock_retry_state)

    assert wait_time == 10.0
    fallback.assert_not_called()


def test_wait_for_retry_after_date(mock_retry_state: RetryCallState) -> None:
    # Setup future date (1 minute from now)
    now = datetime.now(timezone.utc)
    future = now + timedelta(seconds=60)
    http_date = format_datetime(future, usegmt=True)

    response = requests.Response()
    response.status_code = 429
    response.headers["Retry-After"] = http_date
    exc = requests.HTTPError(response=response)
    cast(MagicMock, mock_retry_state.outcome).exception.return_value = exc

    fallback = MagicMock()

    waiter = wait_for_retry_after(fallback=fallback)

    # We patch datetime inside the module to ensure consistent comparison
    # but since we calculate delta, checking if it's close to 60 is enough.
    # However, 'now' shifts. Let's patch datetime.now in the client module.

    with patch("coreason_etl_clinicaltrialsgov.client.datetime") as mock_datetime:
        mock_datetime.now.return_value = now
        # Also need email.utils.parsedate_to_datetime to return the future object correctly
        # But real parser should work.

        wait_time = waiter(mock_retry_state)

        # Should be roughly 60
        assert 59.0 <= wait_time <= 61.0
        fallback.assert_not_called()


def test_wait_for_retry_after_fallback_no_header(mock_retry_state: RetryCallState) -> None:
    # 429 but no Retry-After
    response = requests.Response()
    response.status_code = 429
    # No header
    exc = requests.HTTPError(response=response)
    cast(MagicMock, mock_retry_state.outcome).exception.return_value = exc

    fallback = MagicMock(return_value=5.0)

    waiter = wait_for_retry_after(fallback=fallback)
    wait_time = waiter(mock_retry_state)

    assert wait_time == 5.0
    fallback.assert_called_once()


def test_wait_for_retry_after_fallback_other_error(mock_retry_state: RetryCallState) -> None:
    # 500 Error
    response = requests.Response()
    response.status_code = 500
    exc = requests.HTTPError(response=response)
    cast(MagicMock, mock_retry_state.outcome).exception.return_value = exc

    fallback = MagicMock(return_value=2.0)

    waiter = wait_for_retry_after(fallback=fallback)
    wait_time = waiter(mock_retry_state)

    assert wait_time == 2.0
    fallback.assert_called_once()


def test_wait_for_retry_after_parsing_error(mock_retry_state: RetryCallState) -> None:
    # Bad header
    response = requests.Response()
    response.status_code = 429
    response.headers["Retry-After"] = "invalid"
    exc = requests.HTTPError(response=response)
    cast(MagicMock, mock_retry_state.outcome).exception.return_value = exc

    fallback = MagicMock(return_value=3.0)

    waiter = wait_for_retry_after(fallback=fallback)
    wait_time = waiter(mock_retry_state)

    assert wait_time == 3.0
    fallback.assert_called_once()


def test_wait_for_retry_after_exception_in_parsing(mock_retry_state: RetryCallState) -> None:
    # Header exists but triggers exception during parsing flow
    response = requests.Response()
    response.status_code = 429
    response.headers["Retry-After"] = "invalid"
    exc = requests.HTTPError(response=response)
    cast(MagicMock, mock_retry_state.outcome).exception.return_value = exc

    # Mock fallback
    fallback = MagicMock(return_value=1.0)

    waiter = wait_for_retry_after(fallback=fallback)

    # Force exception inside try/except block by mocking email.utils or similar if needed.
    # Actually, ValueError is already caught.
    # We want to trigger the generic Exception catch.

    with patch(
        "coreason_etl_clinicaltrialsgov.client.email.utils.parsedate_to_datetime", side_effect=Exception("Boom")
    ):
        wait_time = waiter(mock_retry_state)

        # Should fallback
        assert wait_time == 1.0
        fallback.assert_called_once()


# Integration test with Client
def test_client_fetch_studies_retry_logic() -> None:
    # We mock requests.Session.get to simulate 429 then success

    client = ClinicalTrialsClient()
    mock_session = MagicMock()
    client.session = mock_session

    # Create responses
    resp_429 = requests.Response()
    resp_429.status_code = 429
    resp_429.headers["Retry-After"] = "0.1"  # Fast retry

    resp_200 = requests.Response()
    resp_200.status_code = 200
    resp_200._content = b'{"studies": []}'

    # Side effect: 429 (raises HTTPError), then 200
    mock_session.get.side_effect = [resp_429, resp_200]

    # We need to ensure tenacity actually waits.
    # Since we set Retry-After to 0.1, it should wait 0.1s.
    # We can patch time.sleep to verify.

    with patch("time.sleep") as mock_sleep:
        client.fetch_studies()

        # Verify it was called twice
        assert mock_session.get.call_count == 2

        # Verify sleep was called with close to 0.1
        # Tenacity calls time.sleep
        mock_sleep.assert_called()
        args, _ = mock_sleep.call_args
        assert 0.09 <= args[0] <= 0.15  # Allow small float error


def test_wait_for_retry_after_date_naive(mock_retry_state: RetryCallState) -> None:
    """Test when parsed date is naive (no tzinfo)."""
    # Setup future date (1 minute from now)
    now = datetime.now(timezone.utc)
    # 60s in future
    future_ts = now.timestamp() + 60
    # Naive datetime from timestamp (local time usually, or utc if we specify)
    # email.utils.parsedate_to_datetime returns aware datetime if timezone in string,
    # or naive if not.
    # Let's mock parsedate_to_datetime to return a naive datetime

    naive_future = datetime.fromtimestamp(future_ts)  # naive

    response = requests.Response()
    response.status_code = 429
    response.headers["Retry-After"] = "Tue, 15 Nov 1994 08:12:31 GMT"  # Dummy string, we mock return
    exc = requests.HTTPError(response=response)
    cast(MagicMock, mock_retry_state.outcome).exception.return_value = exc

    fallback = MagicMock()
    waiter = wait_for_retry_after(fallback=fallback)

    with patch("coreason_etl_clinicaltrialsgov.client.email.utils.parsedate_to_datetime", return_value=naive_future):
        with patch("coreason_etl_clinicaltrialsgov.client.datetime") as mock_datetime:
            # mock now to be 'naive_future' - 60s, but ensure it works with the logic
            # Logic: if parsed_date.tzinfo is None: parsed_date = parsed_date.replace(tzinfo=timezone.utc)
            # So our naive_future will become UTC.
            # We need 'now' to be 60s before that moment.

            # If naive_future is 1000s, it becomes 1000s UTC.
            # We want now to be 940s UTC.

            mock_datetime.now.return_value = naive_future.replace(tzinfo=timezone.utc) - timedelta(seconds=60)

            wait_time = waiter(mock_retry_state)

            assert 59.0 <= wait_time <= 61.0
