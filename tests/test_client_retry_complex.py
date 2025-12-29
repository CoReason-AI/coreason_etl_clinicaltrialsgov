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
from tenacity import RetryCallState

from coreason_etl_clinicaltrialsgov.client import wait_for_retry_after


@pytest.fixture
def mock_retry_state() -> RetryCallState:
    state = MagicMock(spec=RetryCallState)
    state.outcome = MagicMock()
    return state


def test_retry_after_negative_integer(mock_retry_state: RetryCallState) -> None:
    """Test negative integer returns fallback."""
    response = requests.Response()
    response.status_code = 429
    response.headers["Retry-After"] = "-1"
    exc = requests.HTTPError(response=response)
    cast(MagicMock, mock_retry_state.outcome).exception.return_value = exc

    fallback = MagicMock(return_value=5.0)
    waiter = wait_for_retry_after(fallback=fallback)

    wait_time = waiter(mock_retry_state)

    assert wait_time == 5.0
    fallback.assert_called_once()


def test_retry_after_zero(mock_retry_state: RetryCallState) -> None:
    """Test zero returns 0.0."""
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


def test_retry_after_whitespace(mock_retry_state: RetryCallState) -> None:
    """Test whitespace is handled correctly."""
    response = requests.Response()
    response.status_code = 429
    response.headers["Retry-After"] = "  10  "
    exc = requests.HTTPError(response=response)
    cast(MagicMock, mock_retry_state.outcome).exception.return_value = exc

    fallback = MagicMock()
    waiter = wait_for_retry_after(fallback=fallback)

    wait_time = waiter(mock_retry_state)

    assert wait_time == 10.0


def test_retry_after_float_string(mock_retry_state: RetryCallState) -> None:
    """Test float string parsing."""
    response = requests.Response()
    response.status_code = 429
    response.headers["Retry-After"] = "2.5"
    exc = requests.HTTPError(response=response)
    cast(MagicMock, mock_retry_state.outcome).exception.return_value = exc

    fallback = MagicMock()
    waiter = wait_for_retry_after(fallback=fallback)

    wait_time = waiter(mock_retry_state)

    assert wait_time == 2.5


def test_retry_after_past_date(mock_retry_state: RetryCallState) -> None:
    """Test past date returns 0.0."""
    now = datetime.now(timezone.utc)
    past = now - timedelta(seconds=60)
    http_date = format_datetime(past, usegmt=True)

    response = requests.Response()
    response.status_code = 429
    response.headers["Retry-After"] = http_date
    exc = requests.HTTPError(response=response)
    cast(MagicMock, mock_retry_state.outcome).exception.return_value = exc

    fallback = MagicMock()
    waiter = wait_for_retry_after(fallback=fallback)

    # We don't need to patch datetime.now because the date is strictly in the past relative to real time
    # provided machine time isn't weird. But safer to patch.

    with patch("coreason_etl_clinicaltrialsgov.client.datetime") as mock_datetime:
        mock_datetime.now.return_value = now
        wait_time = waiter(mock_retry_state)

        assert wait_time == 0.0
        fallback.assert_not_called()


def test_retry_after_far_future_date(mock_retry_state: RetryCallState) -> None:
    """Test far future date."""
    now = datetime.now(timezone.utc)
    # 1 year later
    future = now + timedelta(days=365)
    http_date = format_datetime(future, usegmt=True)

    response = requests.Response()
    response.status_code = 429
    response.headers["Retry-After"] = http_date
    exc = requests.HTTPError(response=response)
    cast(MagicMock, mock_retry_state.outcome).exception.return_value = exc

    fallback = MagicMock()
    waiter = wait_for_retry_after(fallback=fallback)

    with patch("coreason_etl_clinicaltrialsgov.client.datetime") as mock_datetime:
        mock_datetime.now.return_value = now
        wait_time = waiter(mock_retry_state)

        # Approx seconds in year
        expected = 365 * 24 * 3600
        assert (expected - 1) <= wait_time <= (expected + 1)
