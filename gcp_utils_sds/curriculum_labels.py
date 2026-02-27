"""
Append assessment titles from multiple pipelines to one BigQuery table.
Safe for concurrent writes (streaming insert). Vendors can use different
column names; pass column_map to map them to year, title, curriculum.
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict

import pandas as pd
from google.cloud import bigquery
from google.cloud import storage
from google.cloud.exceptions import NotFound

from .buckets import send_to_gcs


# Standard columns in the BigQuery table
STAGING_COLUMNS = ["year", "data_source", "title", "curriculum"]

DEFAULT_DATASET_ID = "views"
DEFAULT_TABLE_NAME = "curriculum_labels"
BACKUP_BUCKET_NAME = "curriculum_labels"


def _ensure_curriculum_table_exists(
    project_id: str,
    dataset_id: str,
    table_name: str,
    bq_client: Optional[bigquery.Client] = None,
) -> None:
    """Create dataset and table if they don't exist."""
    if bq_client is None:
        bq_client = bigquery.Client(project=project_id)
    dataset_ref = f"{project_id}.{dataset_id}"
    table_ref = f"{project_id}.{dataset_id}.{table_name}"

    try:
        bq_client.get_dataset(dataset_ref)
        logging.debug(f"Dataset {dataset_id} already exists")
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        dataset.description = "Dataset for curriculum / assessment title labeling"
        bq_client.create_dataset(dataset, exists_ok=True)
        logging.info(f"Created dataset {dataset_id}")

    try:
        bq_client.get_table(table_ref)
        logging.debug(f"Table {table_name} already exists")
    except Exception:
        schema = [
            bigquery.SchemaField("year", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("data_source", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("title", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("curriculum", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("inserted_at", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("batch_id", "STRING", mode="NULLABLE"),
        ]
        table = bigquery.Table(table_ref, schema=schema)
        table.description = "Staging table for assessment titles; multiple pipelines append via streaming insert"
        bq_client.create_table(table)
        logging.info(f"Created table {dataset_id}.{table_name} with schema")


def _prepare_subset_df(
    frame: pd.DataFrame,
    data_source: str,
    column_map: Optional[Dict[str, str]] = None,
    year: Optional[str] = None,
) -> pd.DataFrame:
    """
    Map vendor columns to standard names and build subset.
    column_map: standard_name -> vendor_column_name, e.g. {"year": "school_year", "title": "assessment_name"}.
    year: Optional constant year to apply to all rows in the frame
        (e.g. "25-26"). When provided, the frame does not need to contain
        a year column; this value is used instead.
    """
    column_map = column_map or {}
    # Resolve actual column name for each standard field (use standard name if not in map)
    year_col = column_map.get("year", "year")
    title_col = column_map.get("title", "title")
    curriculum_col = column_map.get("curriculum", "curriculum")

    # Require title column always; require year column only when no explicit year is provided
    required_vendor_cols = [title_col]
    if year is None:
        required_vendor_cols.append(year_col)

    missing = [c for c in required_vendor_cols if c not in frame.columns]
    if missing:
        raise ValueError(
            f"DataFrame missing required columns. Expected 'title' and (optionally) 'year' "
            f"(or mapped names). "
            f"Missing: {missing}. Columns: {list(frame.columns)}"
        )

    if year is None:
        year_values = frame[year_col]
    else:
        # Broadcast a constant year string across all rows, preserving index
        year_values = pd.Series([year] * len(frame), index=frame.index)

    subset = pd.DataFrame({
        "year": year_values,
        "data_source": data_source,
        "title": frame[title_col],
        "curriculum": frame[curriculum_col] if curriculum_col in frame.columns else pd.NA,
    })
    subset = subset.drop_duplicates(subset=["year", "data_source", "title"], keep="first")
    return subset


def _existing_keys_in_table(
    bq_client: bigquery.Client,
    table_ref: str,
    data_source: str,
) -> set:
    """Return set of (year, title) already in the table for this data_source."""
    query = f"""
    SELECT DISTINCT year, title
    FROM `{table_ref}`
    WHERE data_source = @data_source
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("data_source", "STRING", data_source)]
    )
    result = bq_client.query(query, job_config=job_config).result()
    return {(row.year, row.title) for row in result}


def _ensure_backup_bucket_exists(project_id: str) -> None:
    """Create the curriculum_labels bucket if it does not exist."""
    client = storage.Client(project=project_id)
    try:
        client.get_bucket(BACKUP_BUCKET_NAME)
        logging.debug(f"Bucket {BACKUP_BUCKET_NAME} already exists")
    except NotFound:
        client.create_bucket(BACKUP_BUCKET_NAME, location="US")
        logging.info(f"Created bucket {BACKUP_BUCKET_NAME}")


def _backup_table_to_gcs(
    bq_client: bigquery.Client,
    table_ref: str,
    project_id: str,
) -> None:
    """Export full table to GCS as CSV at bucket root (latest.csv). Rely on GCS versioning for history."""
    _ensure_backup_bucket_exists(project_id)
    query = f"SELECT * FROM `{table_ref}`"
    df = bq_client.query(query).to_dataframe()
    send_to_gcs(
        BACKUP_BUCKET_NAME,
        "",
        df,
        "latest.csv",
        project_id=project_id,
    )
    logging.info(f"Backed up curriculum table to gs://{BACKUP_BUCKET_NAME}/latest.csv")


def append_assessment_titles(
    frame: pd.DataFrame,
    project_id: str,
    data_source: str,
    column_map: Optional[Dict[str, str]] = None,
    year: Optional[str] = None,
    dataset_id: str = DEFAULT_DATASET_ID,
    table_name: str = DEFAULT_TABLE_NAME,
    batch_id: Optional[str] = None,
    create_table_if_missing: bool = True,
    skip_existing: bool = True,
) -> int:
    """
    Append assessment titles to the shared BigQuery table. Safe for concurrent
    writes from multiple pipelines (streaming insert).

    Pass column_map when the vendor uses different column names. Keys are the
    standard names (year, title, curriculum); values are the actual column
    names in your DataFrame.

    Duplicates: We always dedupe within the current batch. If skip_existing is
    True (default), we also query the table for existing (year, data_source, title)
    and only insert rows that are not already present, so re-running the same
    pipeline multiple times in a day does not create duplicates.

    Backup: When this run inserts at least one row, the full table is exported to
    the GCS bucket 'curriculum_labels' at the bucket root as latest.csv (overwritten
    each time). The bucket is created if it does not exist. Enable GCS versioning on
    the bucket to keep history. Use as a safety net if the table is ever refreshed.

    Args:
        frame: DataFrame with at least the columns for year and title (or names
            given in column_map).
        project_id: GCP project ID.
        data_source: Which pipeline/source (e.g. "illuminate", "clever").
        column_map: Optional mapping from standard name -> vendor column name.
            E.g. {"year": "school_year", "title": "assessment_name", "curriculum": "subject"}.
            Omit or use None for columns that already match (year, title, curriculum).
        year: Optional constant year to apply to all rows in this frame
            (e.g. "25-26"). When provided, the frame does not need its own
            year column; this value is used instead.
        dataset_id: BigQuery dataset for the table.
        table_name: BigQuery table name.
        batch_id: Optional run id for traceability.
        create_table_if_missing: Create dataset/table when absent.
        skip_existing: If True (default), do not insert rows whose (year, data_source, title)
            already exist in the table (avoids duplicates when re-running).

    Returns:
        Number of rows appended.
    """
    if frame.empty:
        logging.info("append_assessment_titles: empty frame, nothing to append")
        return 0

    subset = _prepare_subset_df(frame, data_source, column_map, year=year)
    if subset.empty:
        logging.info("append_assessment_titles: no rows after dedup, nothing to append")
        return 0

    bq_client = bigquery.Client(project=project_id)
    if create_table_if_missing:
        _ensure_curriculum_table_exists(project_id, dataset_id, table_name, bq_client)

    table_ref = f"{project_id}.{dataset_id}.{table_name}"

    if skip_existing:
        existing = _existing_keys_in_table(bq_client, table_ref, data_source)
        subset = subset[~subset.apply(lambda r: (r["year"], r["title"]) in existing, axis=1)]
        if subset.empty:
            logging.info("append_assessment_titles: all rows already in table, nothing to append")
            return 0

    now = datetime.utcnow().isoformat()
    rows: List[dict] = []
    for _, row in subset.iterrows():
        record = {
            "year": str(row["year"]) if pd.notna(row["year"]) else None,
            "data_source": str(data_source),
            "title": str(row["title"]) if pd.notna(row["title"]) else None,
            "curriculum": str(row["curriculum"]) if pd.notna(row["curriculum"]) else None,
            "inserted_at": now,
            "batch_id": batch_id,
        }
        rows.append(record)

    errors = bq_client.insert_rows_json(table_ref, rows)
    if errors:
        logging.error(f"BigQuery streaming insert had errors: {errors}")
        raise RuntimeError(f"Failed to append assessment titles: {errors}")

    logging.info(f"Appended {len(rows)} assessment title rows to {table_ref} (data_source={data_source})")

    if len(rows) > 0:
        try:
            _backup_table_to_gcs(
                bq_client,
                table_ref,
                project_id,
            )
        except Exception as e:
            logging.warning(f"Backup to GCS failed (non-fatal): {e}")

    return len(rows)
