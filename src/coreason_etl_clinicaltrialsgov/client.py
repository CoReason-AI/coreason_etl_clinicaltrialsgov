# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

import email.utils
from datetime import datetime, timezone
from typing import Any, Iterator, Optional, cast

import requests
from loguru import logger
from tenacity import RetryCallState, retry, stop_after_attempt, wait_exponential
from tenacity.wait import wait_base


class wait_for_retry_after(wait_base):
    """Wait strategy that respects the Retry-After header."""

    def __init__(self, fallback: wait_base) -> None:
        self.fallback = fallback

    def __call__(self, retry_state: RetryCallState) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None

        if isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code == 429:
            retry_after = exc.response.headers.get("Retry-After")
            if retry_after:
                try:
                    # Try parsing as integer seconds
                    return float(retry_after)
                except ValueError:
                    # Try parsing as HTTP Date
                    try:
                        parsed_date = email.utils.parsedate_to_datetime(retry_after)
                        if parsed_date:
                            now = datetime.now(timezone.utc)
                            # Ensure both are offset-aware or convert if needed
                            if parsed_date.tzinfo is None:
                                # Assume GMT/UTC if not specified in parsing (parsedate_to_datetime handles this usually)
                                parsed_date = parsed_date.replace(tzinfo=timezone.utc)

                            wait_seconds = (parsed_date - now).total_seconds()
                            if wait_seconds > 0:
                                return wait_seconds
                    except Exception as e:
                        logger.warning(f"Failed to parse Retry-After header '{retry_after}': {e}")

        # Fallback to the default strategy
        return self.fallback(retry_state)


class ClinicalTrialsClient:
    """Client for the ClinicalTrials.gov API V2."""

    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        """Initialize the client.

        Args:
            session: Optional requests session to use.
        """
        self.session = session or requests.Session()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_for_retry_after(fallback=wait_exponential(multiplier=1, min=4, max=10)),
        reraise=True,
    )
    def fetch_studies(
        self,
        page_token: Optional[str] = None,
        page_size: int = 100,
        query_term: Optional[str] = None,
    ) -> dict[str, Any]:
        """Fetch a single page of studies.

        Args:
            page_token: Cursor for the next page.
            page_size: Number of records to return.
            query_term: Optional query term for filtering.

        Returns:
            The JSON response body.
        """
        params: dict[str, str | int] = {"pageSize": page_size}
        if page_token:
            params["pageToken"] = page_token
        if query_term:
            params["query.term"] = query_term

        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            return cast(dict[str, Any], response.json())
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                logger.warning(f"Rate limited (429). Retrying... {e}")
                raise  # Tenacity will catch this
            logger.error(f"HTTP Error: {e}")
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

    def list_studies(self, page_size: int = 100, query_term: Optional[str] = None) -> Iterator[dict[str, Any]]:
        """Yield studies from the API, handling pagination.

        Args:
            page_size: Number of records per page.
            query_term: Optional query term for filtering.

        Yields:
            Study dictionaries.
        """
        next_page_token: Optional[str] = None

        while True:
            data = self.fetch_studies(page_token=next_page_token, page_size=page_size, query_term=query_term)
            studies = data.get("studies", [])
            for study in studies:
                yield study

            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break
