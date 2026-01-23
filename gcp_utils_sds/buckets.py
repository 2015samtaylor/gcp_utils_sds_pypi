
import pandas as pd
import logging
import os
from datetime import datetime
from google.cloud import storage
from google.cloud import bigquery
from typing import Optional
from io import BytesIO
from io import TextIOWrapper


def _ensure_audit_table_exists(project_id: str, audit_table_name: str) -> None:
    """Create audit table if it doesn't exist. Also creates 'logging' dataset if needed."""
    bq_client = bigquery.Client(project=project_id)
    dataset_id = "logging"
    dataset_ref = f"{project_id}.{dataset_id}"
    table_ref = f"{project_id}.{dataset_id}.{audit_table_name}"
    
    # First, ensure the dataset exists
    try:
        bq_client.get_dataset(dataset_ref)
        logging.debug(f"Dataset {dataset_id} already exists")
    except Exception:
        # Dataset doesn't exist, create it
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"  # Default location, can be customized if needed
        dataset.description = "Dataset for audit and logging tables"
        dataset = bq_client.create_dataset(dataset, exists_ok=True)
        logging.info(f"Created dataset {dataset_id}")
    
    try:
        bq_client.get_table(table_ref)
        logging.debug(f"Audit table {audit_table_name} already exists")
    except Exception:
        # Table doesn't exist, create it
        schema = [
            bigquery.SchemaField("run_date", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("table_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("current_rows_added", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("previous_rows_added", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("row_difference", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("bucket_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("file_name", "STRING", mode="REQUIRED"),
        ]
        
        table = bigquery.Table(table_ref, schema=schema)
        table.description = "Audit table for tracking daily row additions to GCS files"
        table = bq_client.create_table(table)
        logging.info(f"Created audit table {audit_table_name} with schema")


def _get_last_run_stats(project_id: str, audit_table_name: str, table_name: str) -> Optional[dict]:
    """Get statistics from the last run for comparison"""
    try:
        bq_client = bigquery.Client(project=project_id)
        dataset_id = "logging"
        query = f"""
        SELECT 
            current_rows_added,
            run_date
        FROM `{project_id}.{dataset_id}.{audit_table_name}`
        WHERE table_name = @table_name
        ORDER BY run_date DESC
        LIMIT 1
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("table_name", "STRING", table_name)
            ]
        )
        
        result = bq_client.query(query, job_config=job_config).result()
        row = next(result, None)
        
        if row:
            return {
                'current_rows_added': row.current_rows_added,
                'run_date': row.run_date
            }
        return None
    except Exception as e:
        logging.warning(f"Could not fetch last run stats: {e}")
        return None


def _log_gcs_upload_stats(
    project_id: str,
    audit_table_name: str,
    table_name: str,
    current_rows_added: int,
    bucket_name: str,
    file_name: str
) -> None:
    """Log GCS upload statistics to BigQuery audit table"""
    
    # Ensure audit table exists (also creates 'logging' dataset if needed)
    _ensure_audit_table_exists(project_id, audit_table_name)
    
    # Get previous run stats
    last_run = _get_last_run_stats(project_id, audit_table_name, table_name)
    previous_rows_added = last_run['current_rows_added'] if last_run else None
    
    row_difference = (
        current_rows_added - previous_rows_added 
        if previous_rows_added is not None 
        else None
    )
    
    # Prepare audit record
    # Convert datetime to ISO format string for JSON serialization
    audit_record = {
        'run_date': datetime.now().isoformat(),
        'table_name': table_name,
        'current_rows_added': current_rows_added,
        'previous_rows_added': previous_rows_added,
        'row_difference': row_difference,
        'bucket_name': bucket_name,
        'file_name': file_name
    }
    
    # Insert into audit table
    bq_client = bigquery.Client(project=project_id)
    dataset_id = "logging"
    table_ref = f"{project_id}.{dataset_id}.{audit_table_name}"
    errors = bq_client.insert_rows_json(
        table_ref, 
        [audit_record]
    )
    
    if errors:
        logging.error(f"Failed to insert audit record: {errors}")
    else:
        logging.info(f"Logged row stats to {audit_table_name}: {audit_record}")
        
        # Log comparison message
        if previous_rows_added is not None:
            change_pct = ((current_rows_added - previous_rows_added) / previous_rows_added) * 100
            logging.info(
                f"Row count comparison for {table_name}:\n"
                f"  Previous run ({last_run['run_date']}): {previous_rows_added} rows\n"
                f"  Current run: {current_rows_added} rows\n"
                f"  Difference: {row_difference:+d} rows ({change_pct:+.2f}%)"
            )
        else:
            logging.info(f"First run for {table_name}: {current_rows_added} rows uploaded")


def send_to_gcs(
    bucket_name, 
    save_path="",
    frame=None, 
    frame_name=None,
    project_id=None
):
    """
    Uploads a DataFrame as a CSV file to a GCS bucket directly from memory.
    Optionally logs audit statistics to BigQuery.

    Args:
        bucket_name (str): The name of the GCS bucket.
        save_path (str): The path within the bucket where the file will be saved (default: "" for root).
        frame (pd.DataFrame): The DataFrame to upload.
        frame_name (str): The name of the file to save. For audit tracking, frame_name (without extension) is used as the identifier stored in the table_name column of the audit table.
        project_id (str, optional): Project ID for audit logging. If provided, enables audit logging to the 'logging' dataset.
    
    Note: For backward compatibility, you can call this as:
        send_to_gcs(bucket_name, save_path, frame, frame_name)
        or
        send_to_gcs(bucket_name, frame=frame, frame_name=frame_name)  # save_path defaults to ""
    """
    # Validate required parameters
    if frame is None:
        raise ValueError("frame parameter is required")
    if frame_name is None:
        raise ValueError("frame_name parameter is required")
    rows_uploaded = frame.shape[0] if not frame.empty else 0
    
    if not frame.empty:
        client = storage.Client()

        try:
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(os.path.join(save_path, frame_name))
            blob.chunk_size = 5 * 1024 * 1024  # 5MB chunk size for large uploads

            buffer = BytesIO()
            text_buffer = TextIOWrapper(buffer, encoding='utf-8')
            frame.to_csv(text_buffer, index=False)
            text_buffer.flush()
            buffer.seek(0)

            blob.upload_from_file(buffer, content_type='text/csv')
            logging.info(f"{frame_name} uploaded to GCS bucket {bucket_name} at {save_path}/{frame_name}")
            
            # Optional audit logging
            if project_id:
                # Use frame_name without extension as the identifier
                # This is stored in the table_name column of the audit table
                table_identifier = os.path.splitext(frame_name)[0]
                try:
                    _log_gcs_upload_stats(
                        project_id=project_id,
                        audit_table_name="data_pipeline_audit",
                        table_name=table_identifier,
                        current_rows_added=rows_uploaded,
                        bucket_name=bucket_name,
                        file_name=frame_name
                    )
                except Exception as e:
                    logging.warning(f"Could not log audit stats (non-fatal): {e}")
                    
        except Exception as e:
            logging.error(f"Failed to upload {frame_name} to GCS bucket {bucket_name}: {e}")
        finally:
            buffer.close()
    else:
        logging.info(f"No data present in {frame_name} file")
        # Still log audit for empty files if audit is enabled
        if project_id:
            # Use frame_name without extension as the identifier
            # This is stored in the table_name column of the audit table
            table_identifier = os.path.splitext(frame_name)[0]
            try:
                _log_gcs_upload_stats(
                    project_id=project_id,
                    audit_table_name="data_pipeline_audit",
                    table_name=table_identifier,
                    current_rows_added=0,
                    bucket_name=bucket_name,
                    file_name=frame_name
                )
            except Exception as e:
                logging.warning(f"Could not log audit stats for empty file (non-fatal): {e}")




def read_gcs_csv_to_df(gcs_uri, client=None):
  
    if client is None:
        client = storage.Client()

    # Parse bucket and path
    assert gcs_uri.startswith('gs://'), "GCS URI must start with 'gs://'"
    path_parts = gcs_uri[5:].split('/', 1)
    bucket_name = path_parts[0]
    blob_path = path_parts[1]

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    data = blob.download_as_bytes()
    df = pd.read_csv(BytesIO(data))
    return df



