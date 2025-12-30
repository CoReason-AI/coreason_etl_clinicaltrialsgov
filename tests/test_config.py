# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

from typing import Any

import pytest
from pydantic import ValidationError

from coreason_etl_clinicaltrialsgov.config import AppSettings


def test_defaults() -> None:
    """Test default values."""
    s = AppSettings()
    assert s.LOG_LEVEL == "INFO"
    assert s.API_PAGE_SIZE == 100
    assert s.API_TIMEOUT == 30
    assert s.API_MAX_RETRIES == 5
    assert s.APP_ENV == "development"


def test_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test overriding defaults with environment variables."""
    monkeypatch.setenv("CLINICALTRIALS_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("CLINICALTRIALS_API_PAGE_SIZE", "50")
    monkeypatch.setenv("CLINICALTRIALS_API_TIMEOUT", "60")

    s = AppSettings()
    assert s.LOG_LEVEL == "DEBUG"
    assert s.API_PAGE_SIZE == 50
    assert s.API_TIMEOUT == 60


def test_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test validation fails for invalid types."""
    monkeypatch.setenv("CLINICALTRIALS_API_PAGE_SIZE", "invalid")

    with pytest.raises(ValidationError):
        AppSettings()


def test_invalid_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test validation fails for invalid log level."""
    monkeypatch.setenv("CLINICALTRIALS_LOG_LEVEL", "INVALID_LEVEL")

    with pytest.raises(ValidationError):
        AppSettings()


def test_case_sensitivity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that env vars matching is case-insensitive."""
    # When case_sensitive=False (default), lowercase env vars should work
    monkeypatch.setenv("clinicaltrials_log_level", "DEBUG")
    s = AppSettings()
    assert s.LOG_LEVEL == "DEBUG"

    # Uppercase should also work
    monkeypatch.setenv("CLINICALTRIALS_LOG_LEVEL", "WARNING")
    s2 = AppSettings()
    assert s2.LOG_LEVEL == "WARNING"


def test_dotenv_loading_tmp(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading from a temporary .env file."""
    env_file = tmp_path / ".env"
    env_file.write_text("CLINICALTRIALS_API_PAGE_SIZE=200\nCLINICALTRIALS_LOG_LEVEL=WARNING", encoding="utf-8")

    # We need to tell the Settings class to look at this file.
    # We can pass _env_file to the constructor.

    s = AppSettings(_env_file=str(env_file))
    assert s.API_PAGE_SIZE == 200
    assert s.LOG_LEVEL == "WARNING"
