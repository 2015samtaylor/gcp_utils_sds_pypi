import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from gcp_utils_sds.buckets import send_to_gcs

def test_send_to_gcs_uploads_when_not_empty():
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    
    with patch('gcp_utils_sds.buckets.storage.Client', return_value=mock_client):
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        send_to_gcs('test-bucket', 'test/path', df, 'file.csv')
        mock_client.bucket.assert_called_once_with('test-bucket')
        mock_bucket.blob.assert_called_once()
        mock_blob.upload_from_file.assert_called_once()

def test_send_to_gcs_logs_when_empty(caplog):
    caplog.set_level("INFO")
    df = pd.DataFrame()
    with patch('gcp_utils_sds.buckets.storage.Client') as mock_client:
        send_to_gcs('test-bucket', 'test/path', df, 'file.csv')
        assert 'No data present in file.csv file' in caplog.text

def test_send_to_gcs_default_save_path():
    """Test that save_path defaults to empty string (root)"""
    df = pd.DataFrame({'a': [1, 2]})
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    
    with patch('gcp_utils_sds.buckets.storage.Client', return_value=mock_client):
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Call without save_path (should default to "")
        send_to_gcs('test-bucket', frame=df, frame_name='file.csv')
        # Verify blob was created with empty path (root)
        mock_bucket.blob.assert_called_once()
        call_args = mock_bucket.blob.call_args[0][0]
        assert call_args == 'file.csv'  # Should be just filename when save_path is ""

def test_send_to_gcs_no_audit_when_params_missing():
    """Test that audit logging doesn't run when project_id/dataset_id are not provided"""
    df = pd.DataFrame({'a': [1, 2]})
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    
    with patch('gcp_utils_sds.buckets.storage.Client', return_value=mock_client), \
         patch('gcp_utils_sds.buckets.bigquery.Client') as mock_bq_client:
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Call without project_id/dataset_id
        send_to_gcs('test-bucket', '', df, 'file.csv')
        
        # BigQuery client should not be instantiated
        mock_bq_client.assert_not_called()

def test_send_to_gcs_audit_logging_when_params_provided():
    """Test that audit logging runs when project_id and dataset_id are provided"""
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    
    with patch('gcp_utils_sds.buckets.storage.Client', return_value=mock_client), \
         patch('gcp_utils_sds.buckets.bigquery.Client'), \
         patch('gcp_utils_sds.buckets._log_gcs_upload_stats') as mock_log_stats:
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Call with project_id and dataset_id
        send_to_gcs(
            'test-bucket', 
            '', 
            df, 
            'file.csv',
            project_id='test-project',
            dataset_id='test-dataset'
        )
        
        # Verify audit logging was called
        mock_log_stats.assert_called_once()
        call_kwargs = mock_log_stats.call_args[1]
        assert call_kwargs['project_id'] == 'test-project'
        assert call_kwargs['audit_table_name'] == 'data_pipeline_audit'
        assert call_kwargs['table_name'] == 'file'  # frame_name without .csv extension
        assert call_kwargs['current_rows_added'] == 2
        assert call_kwargs['bucket_name'] == 'test-bucket'
        assert call_kwargs['file_name'] == 'file.csv'

def test_send_to_gcs_audit_uses_frame_name_as_identifier():
    """Test that audit logging uses frame_name (without extension) as the table identifier"""
    df = pd.DataFrame({'a': [1, 2]})
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    
    with patch('gcp_utils_sds.buckets.storage.Client', return_value=mock_client), \
         patch('gcp_utils_sds.buckets.bigquery.Client'), \
         patch('gcp_utils_sds.buckets._log_gcs_upload_stats') as mock_log_stats:
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        send_to_gcs(
            'test-bucket',
            '',
            df,
            'assessment_results_group.csv',
            project_id='test-project',
            dataset_id='test-dataset'
        )
        
        # Verify frame_name without extension is used as table_name in audit record
        call_kwargs = mock_log_stats.call_args[1]
        assert call_kwargs['table_name'] == 'assessment_results_group'

def test_send_to_gcs_audit_for_empty_dataframe():
    """Test that audit logging works for empty dataframes"""
    df = pd.DataFrame()
    mock_client = MagicMock()
    
    with patch('gcp_utils_sds.buckets.storage.Client', return_value=mock_client), \
         patch('gcp_utils_sds.buckets.bigquery.Client'), \
         patch('gcp_utils_sds.buckets._log_gcs_upload_stats') as mock_log_stats:
        
        send_to_gcs(
            'test-bucket',
            '',
            df,
            'file.csv',
            project_id='test-project',
            dataset_id='test-dataset'
        )
        
        # Verify audit was called with 0 rows
        mock_log_stats.assert_called_once()
        call_kwargs = mock_log_stats.call_args[1]
        assert call_kwargs['current_rows_added'] == 0
