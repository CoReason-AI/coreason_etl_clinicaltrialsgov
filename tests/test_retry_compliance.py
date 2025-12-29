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


def test_retry_after_exact_verification(mock_retry_state: RetryCallState) -> None:
    """Strict verification of Retry-After logic as requested."""

    # 1. Past Date -> Immediate (0.0)
    # This is critical to ensure we don't wait forever or crash.
    now = datetime.now(timezone.utc)
    past = now - timedelta(seconds=100)
    http_date = format_datetime(past, usegmt=True)

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
        assert wait_time == 0.0
        fallback.assert_not_called()

    # 2. Future Date -> Delta
    future = now + timedelta(seconds=120)
    http_date_future = format_datetime(future, usegmt=True)
    response.headers["Retry-After"] = http_date_future

    with patch("coreason_etl_clinicaltrialsgov.client.datetime") as mock_datetime:
        mock_datetime.now.return_value = now
        wait_time = waiter(mock_retry_state)
        # Should be exactly 120.0
        assert 119.0 <= wait_time <= 121.0
        fallback.assert_not_called()

    # 3. Integer Seconds
    response.headers["Retry-After"] = "42"
    wait_time = waiter(mock_retry_state)
    assert wait_time == 42.0
    fallback.assert_not_called()

    # 4. Negative Integer -> Fallback (Invalid)
    response.headers["Retry-After"] = "-1"
    fallback.reset_mock()
    fallback.return_value = 5.0
    wait_time = waiter(mock_retry_state)
    assert wait_time == 5.0
    fallback.assert_called()

    # 5. Garbage -> Fallback
    response.headers["Retry-After"] = "NotADate"
    fallback.reset_mock()
    wait_time = waiter(mock_retry_state)
    assert wait_time == 5.0
    fallback.assert_called()

    # 6. Float Seconds (Robustness check)
    # RFC 7231 says non-negative decimal integer, but robustness principle applies.
    # Our code uses float(), so "10.5" should work.
    response.headers["Retry-After"] = "10.5"
    wait_time = waiter(mock_retry_state)
    assert wait_time == 10.5
