import re
import logging
from io import BytesIO
import pandas as pd
import numpy as np
from google.cloud import storage
from google.cloud import bigquery
from typing import List

class YearlyDataAppender:
    def __init__(self, project_id: str, dataset_id: str, bucket_name: str):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.bucket_name = bucket_name

        self.bq_client = bigquery.Client(project=project_id)
        self.gcs_client = storage.Client(project=project_id)

    def _get_bq_schema(self, table_name: str) -> dict:
        """Fetch BigQuery schema as {column_name: BQ_type}"""
        table_ref = f"{self.project_id}.{self.dataset_id}.{table_name}"
        table = self.bq_client.get_table(table_ref)
        return {field.name: field.field_type for field in table.schema}

    def _map_bq_to_pandas(self, schema: dict) -> dict:
        """Convert BQ types to pandas dtypes"""
        type_map = {
            "STRING": "string",
            "INTEGER": "Int64",
            "INT64": "Int64",
            "FLOAT": "Float64",
            "FLOAT64": "Float64",
            "BOOLEAN": "boolean",
            "DATE": "datetime64[ns]",
            "DATETIME": "datetime64[ns]",
            "TIMESTAMP": "datetime64[ns]",
        }
        return {col: type_map.get(bq_type, "object") for col, bq_type in schema.items()}

    def _cast_df_to_bq_types(self, df: pd.DataFrame, dtype_map: dict) -> pd.DataFrame:
        """Safely cast columns to BigQuery-compatible types"""
        for col, dtype in dtype_map.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    logging.warning(f"Column '{col}' could not be cast to '{dtype}': {e}")
        return df

    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to match BQ schema expectations"""
        cleaned_cols = [
            re.sub(r'[^A-Za-z0-9_]', '_', re.sub(r'\(.*?\)', '', col)).lower().strip('_')
            for col in df.columns
        ]
        df.columns = cleaned_cols
        logging.info(f"Cleaned column names: {cleaned_cols}")
        return df

    def _extract_year_from_filename(self, filename: str) -> str:
        """Extract year (like '24-25') from filename"""
        match = re.search(r'(\d{2}-\d{2})(?=\.csv$)', filename)
        if not match:
            raise ValueError(f"Could not extract year from filename: {filename}")
        return match.group(1)

    def _align_columns(self, df1: pd.DataFrame, df2: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """Ensure both DataFrames have the same set of columns (union of all columns)."""
        all_columns = set(df1.columns).union(df2.columns)
        for df in (df1, df2):
            missing_cols = all_columns - set(df.columns)
            for col in missing_cols:
                df[col] = pd.NA
        return df1[list(all_columns)], df2[list(all_columns)]

    def _normalize_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize null-like values to <NA> and cast to pandas extension dtypes."""
        # First, replace all known null-like values with pd.NA
        df = df.replace({None: pd.NA, np.nan: pd.NA, '': pd.NA, 'nan': pd.NA, 'NaN': pd.NA})

        # Now cast to proper nullable types
        for col in df.columns:
            if df[col].dtype.kind in {'O', 'U', 'S'}:
                df[col] = df[col].astype("string")
            elif pd.api.types.is_integer_dtype(df[col]):
                df[col] = df[col].astype("Int64")
            elif pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].astype("Float64")
            elif pd.api.types.is_bool_dtype(df[col]):
                df[col] = df[col].astype("boolean")

        return df

    def load_and_append(self, table_name: str, blob_paths_old: List[str], current_df: pd.DataFrame, drop_duplicate_columns: list = None) -> pd.DataFrame:
        """
        Download multiple historical datasets from GCS, normalize and align both historical and current data, 
        append them, normalize nulls, and optionally deduplicate.
        """
        logging.info(f"Processing historical blobs: {blob_paths_old} from bucket: {self.bucket_name}")

        # Get schema and type mapping for current table
        schema = self._get_bq_schema(table_name)
        dtype_map = self._map_bq_to_pandas(schema)

        # Download and combine all old blobs
        old_dfs = []
        for blob_path_old in blob_paths_old:
            logging.info(f"Downloading: {blob_path_old}")
            blob = self.gcs_client.bucket(self.bucket_name).blob(blob_path_old)
            blob_bytes = blob.download_as_bytes()
            old_df = pd.read_csv(BytesIO(blob_bytes), low_memory=False)
            logging.info(f"Downloaded {blob_path_old} with shape: {old_df.shape}")

            # Extract year and append to the DataFrame
            year_val = self._extract_year_from_filename(blob_path_old)
            old_df["year"] = year_val
            old_dfs.append(old_df)

        # Combine all old DataFrames
        combined_old_df = pd.concat(old_dfs, ignore_index=True)
        logging.info(f"Combined old DataFrames shape: {combined_old_df.shape}")

        # Clean and cast current DataFrame
        current_df = self._clean_column_names(current_df)
        current_df = self._cast_df_to_bq_types(current_df, dtype_map)

        # Align columns
        current_df['new_or_current'] = 'current'
        combined_old_df['new_or_current'] = 'historical'
        current_df, combined_old_df = self._align_columns(current_df, combined_old_df)

        # Combine historical and current
        logging.info(f'Before adding in new data\nNew dataframe shape: {current_df.shape}, Old dataframe shape: {combined_old_df.shape}')
        combined_df = pd.concat([current_df, combined_old_df], ignore_index=True)
        logging.info(f"Total combined DataFrame shape after adding in new data from pipeline run: {combined_df.shape}")

        # Normalize missing values
        combined_df = self._normalize_missing_values(combined_df)

        # Optional deduplication
        if drop_duplicate_columns:
            before_shape = combined_df.shape
            combined_df = combined_df.drop_duplicates(subset=drop_duplicate_columns)
            after_shape = combined_df.shape
            rows_dropped = before_shape[0] - after_shape[0]
            logging.info(f"Dropped duplicates using columns {drop_duplicate_columns}.\nShape before: {before_shape}, after: {after_shape}. Rows dropped: {rows_dropped}")
        else:
            logging.info("No deduplication columns provided; skipping deduplication.")

        logging.info(f"Final combined DataFrame shape: {combined_df.shape}\n\n")
        return combined_df