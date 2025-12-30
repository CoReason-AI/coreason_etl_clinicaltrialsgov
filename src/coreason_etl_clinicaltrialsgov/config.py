# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Application configuration settings."""

    # Application
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # API Client
    API_PAGE_SIZE: int = 100
    API_TIMEOUT: int = 30
    API_MAX_RETRIES: int = 5
    API_RETRY_MIN_WAIT: int = 4
    API_RETRY_MAX_WAIT: int = 10

    # Environment
    APP_ENV: Literal["development", "testing", "production"] = "development"

    model_config = SettingsConfigDict(
        env_prefix="CLINICALTRIALS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Singleton instance
settings = AppSettings()
