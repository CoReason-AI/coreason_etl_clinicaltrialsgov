# coreason_etl_clinicaltrialsgov

A foundational data pipeline to ingest, normalize, and serve the complete corpus of biomedical literature from the National Library of Medicine (NLM) ClinicalTrials.gov database via its modern V2 API.

## Project Overview

This project implements a **Medallion Architecture** ETL pipeline:

1.  **Bronze (The Lake)**: Lossless ingestion of raw source JSON data.
    *   **Source**: ClinicalTrials.gov API V2.
    *   **Logic**: Incremental loading via `LastUpdatePostDate` and `dlt` state management.
    *   **Schema**: `source_id`, `ingestion_ts`, `raw_payload`.

2.  **Silver (The Refinery)**: Structured, type-safe tables with identity resolution.
    *   **Engine**: Polars for high-performance transformation, Pydantic for strict schema validation.
    *   **Tables**: `silver_studies`, `silver_sponsors`, `silver_locations`, `silver_interventions`, `silver_outcomes`, `silver_references`.
    *   **Features**: Sponsor deduplication, date normalization, age parsing, surrogate key generation.

3.  **Gold (The Product)**: High-value business data.
    *   **Filter**: Active/Recruiting studies only.
    *   **Derived Fields**: `enrollment_bucket`, `years_active`, `has_results`, `geo_countries`.

## Prerequisites

-   **Python**: 3.12+
-   **Poetry**: Dependency management.
-   **PostgreSQL**: Target database (default destination).

## Installation

1.  **Clone the repository**:
    ```sh
    git clone https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov.git
    cd coreason_etl_clinicaltrialsgov
    ```

2.  **Install dependencies**:
    ```sh
    poetry install
    ```

3.  **Setup Environment**:
    Copy the example environment file (if available) or set the following environment variables.

## Configuration

Configuration is managed via environment variables (prefix `CLINICALTRIALS_`). Create a `.env` file or export them directly:

```sh
# Application
CLINICALTRIALS_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# API Client
CLINICALTRIALS_API_PAGE_SIZE=100
CLINICALTRIALS_API_TIMEOUT=30
CLINICALTRIALS_API_MAX_RETRIES=5

# DLT Destination (Postgres)
# DLT expects standard Postgres env vars or secrets.toml
DESTINATION__POSTGRES__CREDENTIALS="postgresql://user:password@localhost:5432/clinical_trials"
```

## Usage

The package provides a CLI entry point `coreason-etl-clinicaltrialsgov`.

### Run the ETL Pipeline

To run the full extraction and loading pipeline:

```sh
poetry run coreason-etl-clinicaltrialsgov run
```

### Options

| Option            | Default                | Description                                      |
| ----------------- | ---------------------- | ------------------------------------------------ |
| `--page-size`     | `100`                  | Number of records per API page.                  |
| `--query-term`    | `None`                 | Optional filter (e.g., `heart attack`).          |
| `--destination`   | `postgres`             | DLT destination (e.g., `postgres`, `duckdb`).    |
| `--pipeline-name` | `clinicaltrials_etl`   | Name of the DLT pipeline.                        |
| `--dataset-name`  | `None`                 | Target dataset name (defaults to pipeline name). |

**Example:**
```sh
poetry run coreason-etl-clinicaltrialsgov run --page-size 50 --query-term "AREA[LocationCountry]RANGE[United States]"
```

## Development

### Running Tests
This project enforces **100% test coverage**.

```sh
poetry run pytest
```

### Linting and Formatting
Strict adherence to `ruff` and `mypy` is required.

```sh
# Format code
poetry run ruff format .

# Check linting errors
poetry run ruff check --fix .

# Static Type Checking
poetry run mypy .

# Run all pre-commit hooks
poetry run pre-commit run --all-files
```

## License

Prosperity Public License 3.0. See [LICENSE](LICENSE) for details.
