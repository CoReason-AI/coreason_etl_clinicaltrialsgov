import logging
from typing import Iterator

import pytest
from loguru import logger


@pytest.fixture(autouse=True)
def caplog_loguru(caplog: pytest.LogCaptureFixture) -> Iterator[None]:
    """Redirect loguru logs to standard logging for caplog capture."""

    class PropagateHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logging.getLogger(record.name).handle(record)

    # Add the sink to loguru
    sink_id = logger.add(PropagateHandler(), format="{message}")

    yield

    # Remove the sink after test
    logger.remove(sink_id)
