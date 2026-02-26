import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from gcp_utils_sds.curriculum_labels import (
    append_assessment_titles,
    _prepare_subset_df,
    STAGING_COLUMNS,
    DEFAULT_DATASET_ID,
    DEFAULT_TABLE_NAME,
)


def test_append_assessment_titles_calls_insert_and_returns_count():
    df = pd.DataFrame({
        "year": ["24-25", "24-25"],
        "title": ["Math Benchmark", "ELA Assessment"],
        "curriculum": [pd.NA, "ELA"],
    })
    mock_bq = MagicMock()
    mock_bq.insert_rows_json.return_value = []  # success
    mock_bq.query.return_value.result.return_value = []  # no existing keys (skip_existing)

    with patch("gcp_utils_sds.curriculum_labels.bigquery.Client", return_value=mock_bq), \
         patch("gcp_utils_sds.curriculum_labels._ensure_curriculum_table_exists"):
        n = append_assessment_titles(
            df,
            project_id="my-project",
            data_source="illuminate",
        )
    assert n == 2
    mock_bq.insert_rows_json.assert_called_once()
    rows = mock_bq.insert_rows_json.call_args[0][1]
    assert len(rows) == 2
    assert rows[0]["year"] == "24-25" and rows[0]["data_source"] == "illuminate"
    assert rows[0]["title"] == "Math Benchmark" and rows[0]["curriculum"] is None
    assert rows[1]["curriculum"] == "ELA"


def test_append_assessment_titles_empty_frame_returns_zero():
    df = pd.DataFrame({"year": [], "title": []})
    with patch("gcp_utils_sds.curriculum_labels.bigquery.Client") as mock_bq:
        n = append_assessment_titles(
            df,
            project_id="my-project",
            data_source="source_a",
        )
    assert n == 0
    mock_bq.insert_rows_json.assert_not_called()


def test_append_assessment_titles_missing_columns_raises():
    df = pd.DataFrame({"title": ["Only title"], "other": [1]})
    with pytest.raises(ValueError, match="year"):
        append_assessment_titles(
            df,
            project_id="my-project",
            data_source="x",
        )


def test_prepare_subset_df_dedupes_and_adds_data_source():
    df = pd.DataFrame({
        "year": ["24-25", "24-25"],
        "title": ["Same Title", "Same Title"],
        "curriculum": [None, None],
    })
    out = _prepare_subset_df(df, "clever")
    assert list(out.columns) == STAGING_COLUMNS
    assert out["data_source"].tolist() == ["clever", "clever"]
    assert len(out) == 1  # dedup on (year, data_source, title)
    assert out["title"].iloc[0] == "Same Title"


def test_prepare_subset_df_with_column_map():
    df = pd.DataFrame({
        "school_year": ["24-25"],
        "assessment_name": ["Math Benchmark"],
        "subject": ["Math"],
    })
    out = _prepare_subset_df(
        df,
        "vendor_a",
        column_map={"year": "school_year", "title": "assessment_name", "curriculum": "subject"},
    )
    assert list(out.columns) == STAGING_COLUMNS
    assert out["year"].iloc[0] == "24-25"
    assert out["title"].iloc[0] == "Math Benchmark"
    assert out["curriculum"].iloc[0] == "Math"
    assert out["data_source"].iloc[0] == "vendor_a"


def test_append_assessment_titles_with_column_map():
    df = pd.DataFrame({
        "school_year": ["24-25"],
        "assessment_name": ["ELA Test"],
    })
    mock_bq = MagicMock()
    mock_bq.insert_rows_json.return_value = []
    mock_bq.query.return_value.result.return_value = []

    with patch("gcp_utils_sds.curriculum_labels.bigquery.Client", return_value=mock_bq), \
         patch("gcp_utils_sds.curriculum_labels._ensure_curriculum_table_exists"):
        n = append_assessment_titles(
            df,
            project_id="my-project",
            data_source="illuminate",
            column_map={"year": "school_year", "title": "assessment_name"},
        )
    assert n == 1
    rows = mock_bq.insert_rows_json.call_args[0][1]
    assert rows[0]["year"] == "24-25" and rows[0]["title"] == "ELA Test"
    assert rows[0]["curriculum"] is None


def test_append_assessment_titles_skip_existing_avoids_duplicates():
    """Re-running with same data: skip_existing=True only inserts rows not already in table."""
    df = pd.DataFrame({
        "year": ["24-25", "24-25"],
        "title": ["Already In Table", "New Title"],
    })
    # Simulate (24-25, Already In Table) already in BQ
    existing_row = MagicMock()
    existing_row.year = "24-25"
    existing_row.title = "Already In Table"
    mock_bq = MagicMock()
    mock_bq.query.return_value.result.return_value = [existing_row]
    mock_bq.insert_rows_json.return_value = []

    with patch("gcp_utils_sds.curriculum_labels.bigquery.Client", return_value=mock_bq), \
         patch("gcp_utils_sds.curriculum_labels._ensure_curriculum_table_exists"):
        n = append_assessment_titles(
            df,
            project_id="my-project",
            data_source="illuminate",
            skip_existing=True,
        )
    assert n == 1
    rows = mock_bq.insert_rows_json.call_args[0][1]
    assert len(rows) == 1
    assert rows[0]["title"] == "New Title"


def test_append_assessment_titles_skip_existing_false_allows_duplicates():
    """With skip_existing=False we do not query and insert all rows (can create duplicates)."""
    df = pd.DataFrame({"year": ["24-25"], "title": ["Any Title"]})
    mock_bq = MagicMock()
    mock_bq.insert_rows_json.return_value = []

    with patch("gcp_utils_sds.curriculum_labels.bigquery.Client", return_value=mock_bq), \
         patch("gcp_utils_sds.curriculum_labels._ensure_curriculum_table_exists"):
        n = append_assessment_titles(
            df,
            project_id="my-project",
            data_source="illuminate",
            skip_existing=False,
        )
    assert n == 1
    mock_bq.query.assert_not_called()


def test_append_assessment_titles_backup_to_gcs_when_inserts():
    """When we insert rows, full table is written to curriculum_labels bucket root as latest.csv."""
    df = pd.DataFrame({"year": ["24-25"], "title": ["Backup Test"]})
    mock_bq = MagicMock()
    mock_bq.insert_rows_json.return_value = []
    mock_bq.query.return_value.result.return_value = []  # skip_existing: no existing keys
    mock_bq.query.return_value.to_dataframe.return_value = pd.DataFrame({
        "year": ["24-25"],
        "data_source": ["illuminate"],
        "title": ["Backup Test"],
        "curriculum": [None],
        "inserted_at": ["2025-02-26 12:00:00"],
        "batch_id": [None],
    })

    with patch("gcp_utils_sds.curriculum_labels.bigquery.Client", return_value=mock_bq), \
         patch("gcp_utils_sds.curriculum_labels._ensure_curriculum_table_exists"), \
         patch("gcp_utils_sds.curriculum_labels._ensure_backup_bucket_exists"), \
         patch("gcp_utils_sds.curriculum_labels.send_to_gcs") as mock_send:
        n = append_assessment_titles(
            df,
            project_id="my-project",
            data_source="illuminate",
        )
    assert n == 1
    mock_send.assert_called_once()
    args = mock_send.call_args[0]
    assert args[0] == "curriculum_labels"  # hardcoded bucket
    assert args[1] == ""  # root of bucket
    assert args[3] == "latest.csv"


def test_append_assessment_titles_no_backup_when_no_inserts():
    """Backup is not run when no rows are inserted (e.g. all skipped as existing)."""
    df = pd.DataFrame({"year": ["24-25"], "title": ["Already Exists"]})
    existing_row = MagicMock()
    existing_row.year = "24-25"
    existing_row.title = "Already Exists"
    mock_bq = MagicMock()
    mock_bq.insert_rows_json.return_value = []
    mock_bq.query.return_value.result.return_value = [existing_row]  # skip_existing: all skipped

    with patch("gcp_utils_sds.curriculum_labels.bigquery.Client", return_value=mock_bq), \
         patch("gcp_utils_sds.curriculum_labels._ensure_curriculum_table_exists"), \
         patch("gcp_utils_sds.curriculum_labels.send_to_gcs") as mock_send:
        n = append_assessment_titles(
            df,
            project_id="my-project",
            data_source="illuminate",
        )
    assert n == 0
    mock_send.assert_not_called()
