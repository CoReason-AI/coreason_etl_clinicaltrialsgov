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
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import requests
from tenacity import RetryCallState

from coreason_etl_clinicaltrialsgov.client import wait_for_retry_after


@pytest.fixture
def mock_retry_state() -> RetryCallState:
    state = MagicMock(spec=RetryCallState)
    state.outcome = MagicMock()
    return state


def test_retry_after_negative_seconds(mock_retry_state: RetryCallState) -> None:
    """Negative seconds should be ignored and fallback used."""
    response = requests.Response()
    response.status_code = 429
    response.headers["Retry-After"] = "-10"
    exc = requests.HTTPError(response=response)
    cast(MagicMock, mock_retry_state.outcome).exception.return_value = exc

    fallback = MagicMock(return_value=5.0)
    waiter = wait_for_retry_after(fallback=fallback)
    wait_time = waiter(mock_retry_state)

    assert wait_time == 5.0
    fallback.assert_called_once()


def test_retry_after_zero_seconds(mock_retry_state: RetryCallState) -> None:
    """Zero seconds should be respected (immediate retry)."""
    response = requests.Response()
    response.status_code = 429
    response.headers["Retry-After"] = "0"
    exc = requests.HTTPError(response=response)
    cast(MagicMock, mock_retry_state.outcome).exception.return_value = exc

    fallback = MagicMock()
    waiter = wait_for_retry_after(fallback=fallback)
    wait_time = waiter(mock_retry_state)

    assert wait_time == 0.0
    fallback.assert_not_called()


def test_retry_after_past_date(mock_retry_state: RetryCallState) -> None:
    """Past date should result in 0.0 wait (immediate retry)."""
    past_date = "Tue, 15 Nov 1994 08:12:31 GMT"
    response = requests.Response()
    response.status_code = 429
    response.headers["Retry-After"] = past_date
    exc = requests.HTTPError(response=response)
    cast(MagicMock, mock_retry_state.outcome).exception.return_value = exc

    fallback = MagicMock()
    waiter = wait_for_retry_after(fallback=fallback)

    # We don't need to patch datetime.now() because 1994 is definitely in the past
    wait_time = waiter(mock_retry_state)

    assert wait_time == 0.0
    fallback.assert_not_called()


def test_retry_after_future_date_long(mock_retry_state: RetryCallState) -> None:
    """Date far in the future should be respected."""
    # 1 year in future
    now = datetime.now(timezone.utc)
    future = now + timedelta(days=365)
    # RFC 1123 format
    future_str = future.strftime("%a, %d %b %Y %H:%M:%S GMT")

    response = requests.Response()
    response.status_code = 429
    response.headers["Retry-After"] = future_str
    exc = requests.HTTPError(response=response)
    cast(MagicMock, mock_retry_state.outcome).exception.return_value = exc

    fallback = MagicMock()
    waiter = wait_for_retry_after(fallback=fallback)

    with patch("coreason_etl_clinicaltrialsgov.client.datetime") as mock_datetime:
        mock_datetime.now.return_value = now
        wait_time = waiter(mock_retry_state)

        # Should be roughly 365 days in seconds
        expected = 365 * 24 * 3600
        assert (expected - 5) <= wait_time <= (expected + 5)
        fallback.assert_not_called()


def test_retry_after_garbage_string(mock_retry_state: RetryCallState) -> None:
    """Garbage string should fallback."""
    response = requests.Response()
    response.status_code = 429
    response.headers["Retry-After"] = "Not a date or number"
    exc = requests.HTTPError(response=response)
    cast(MagicMock, mock_retry_state.outcome).exception.return_value = exc

    fallback = MagicMock(return_value=2.0)
    waiter = wait_for_retry_after(fallback=fallback)
    wait_time = waiter(mock_retry_state)

    assert wait_time == 2.0
    fallback.assert_called_once()


def test_retry_after_empty_string(mock_retry_state: RetryCallState) -> None:
    """Empty string should fallback."""
    response = requests.Response()
    response.status_code = 429
    response.headers["Retry-After"] = ""
    exc = requests.HTTPError(response=response)
    cast(MagicMock, mock_retry_state.outcome).exception.return_value = exc

    fallback = MagicMock(return_value=2.0)
    waiter = wait_for_retry_after(fallback=fallback)
    wait_time = waiter(mock_retry_state)

    assert wait_time == 2.0
    fallback.assert_called_once()
