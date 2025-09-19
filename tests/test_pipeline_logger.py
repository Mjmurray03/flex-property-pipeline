"""
Tests for PipelineLogger class
"""

import pytest
import tempfile
import shutil
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from pipeline.pipeline_logger import PipelineLogger, ProcessingMetrics, FileProgressTracker


class TestProcessingMetrics:
    """Test cases for ProcessingMetrics"""
    
    def test_initialization(self):
        """Test ProcessingMetrics initialization"""
        metrics = ProcessingMetrics()
        
        assert metrics.files_processed == 0
        assert metrics.files_successful == 0
        assert metrics.files_failed == 0
        assert metrics.total_properties == 0
        assert metrics.total_flex_candidates == 0
        assert metrics.memory_usage_mb == 0.0
        assert metrics.peak_memory_mb == 0.0
        assert len(metrics.processing_errors) == 0
        assert metrics.start_time is not None
        assert metrics.end_time is None
    
    def test_duration_calculation(self):
        """Test duration calculation"""
        metrics = ProcessingMetrics()
        time.sleep(0.1)  # Small delay
        
        duration = metrics.duration_seconds
        assert duration > 0
        assert duration < 1  # Should be less than 1 second
        
        # Test with end time set
        metrics.end_time = datetime.now()
        duration_with_end = metrics.duration_seconds
        assert duration_with_end >= duration
    
    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        metrics = ProcessingMetrics()
        
        # No files processed
        assert metrics.success_rate == 0.0
        
        # Some files processed
        metrics.files_processed = 10
        metrics.files_successful = 8
        assert metrics.success_rate == 80.0
        
        # All files successful
        metrics.files_successful = 10
        assert metrics.success_rate == 100.0
    
    def test_processing_rate_calculation(self):
        """Test processing rate calculation"""
        metrics = ProcessingMetrics()
        
        # No time elapsed
        assert metrics.processing_rate == 0.0
        
        # Simulate some processing
        time.sleep(0.1)
        metrics.files_processed = 6
        
        rate = metrics.processing_rate
        assert rate > 0
        # Should be around 3600 files/minute (6 files in 0.1 seconds)


class TestPipelineLogger:
    """Test cases for PipelineLogger"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def pipeline_logger(self):
        """Create PipelineLogger instance"""
        # Use console-only logging for tests to avoid file permission issues
        with patch('pipeline.pipeline_logger.setup_logging') as mock_setup:
            mock_logger = MagicMock()
            mock_setup.return_value = mock_logger
            
            logger = PipelineLogger(
                log_level='DEBUG',
                log_file=None,
                enable_performance_monitoring=False  # Disable for testing
            )
            logger.logger = mock_logger
            return logger
    
    def test_initialization(self, pipeline_logger):
        """Test PipelineLogger initialization"""
        assert pipeline_logger.logger is not None
        assert pipeline_logger.enable_performance_monitoring is False
        assert isinstance(pipeline_logger.metrics, ProcessingMetrics)
        assert len(pipeline_logger.file_progress) == 0
        assert len(pipeline_logger.stage_timings) == 0
    
    def test_start_pipeline(self, pipeline_logger):
        """Test pipeline start logging"""
        pipeline_logger.start_pipeline(5, "/test/input")
        
        # Check that metrics were reset
        assert pipeline_logger.metrics.files_processed == 0
        assert len(pipeline_logger.file_progress) == 0
        assert len(pipeline_logger.stage_timings) == 0
    
    def test_stage_timing(self, pipeline_logger):
        """Test stage timing functionality"""
        stage_name = "test_stage"
        
        # Start stage
        pipeline_logger.start_stage(stage_name)
        assert stage_name in pipeline_logger.stage_timings
        
        # Small delay
        time.sleep(0.1)
        
        # End stage
        duration = pipeline_logger.end_stage(stage_name)
        assert duration > 0
        assert duration < 1  # Should be less than 1 second
    
    def test_end_nonexistent_stage(self, pipeline_logger):
        """Test ending a stage that wasn't started"""
        duration = pipeline_logger.end_stage("nonexistent_stage")
        assert duration == 0.0
    
    def test_file_processing_tracking(self, pipeline_logger):
        """Test file processing start and end"""
        file_path = "/test/file.xlsx"
        file_size = 1.5
        
        # Start file processing
        pipeline_logger.start_file_processing(file_path, file_size)
        
        assert file_path in pipeline_logger.file_progress
        file_info = pipeline_logger.file_progress[file_path]
        assert file_info['file_size_mb'] == file_size
        assert file_info['status'] == 'processing'
        
        # End file processing - success
        pipeline_logger.end_file_processing(
            file_path, 
            success=True, 
            property_count=100, 
            flex_candidate_count=25
        )
        
        # Check metrics updated
        assert pipeline_logger.metrics.files_processed == 1
        assert pipeline_logger.metrics.files_successful == 1
        assert pipeline_logger.metrics.files_failed == 0
        assert pipeline_logger.metrics.total_properties == 100
        assert pipeline_logger.metrics.total_flex_candidates == 25
        
        # Check file info updated
        file_info = pipeline_logger.file_progress[file_path]
        assert file_info['status'] == 'completed'
        assert file_info['property_count'] == 100
        assert file_info['flex_candidate_count'] == 25
    
    def test_file_processing_failure(self, pipeline_logger):
        """Test file processing failure tracking"""
        file_path = "/test/failed_file.xlsx"
        error_message = "File corrupted"
        
        pipeline_logger.start_file_processing(file_path, 1.0)
        pipeline_logger.end_file_processing(
            file_path, 
            success=False, 
            error_message=error_message
        )
        
        # Check metrics
        assert pipeline_logger.metrics.files_processed == 1
        assert pipeline_logger.metrics.files_successful == 0
        assert pipeline_logger.metrics.files_failed == 1
        assert len(pipeline_logger.metrics.processing_errors) == 1
        assert error_message in pipeline_logger.metrics.processing_errors[0]
        
        # Check file info
        file_info = pipeline_logger.file_progress[file_path]
        assert file_info['status'] == 'failed'
        assert file_info['error_message'] == error_message
    
    def test_end_file_processing_not_started(self, pipeline_logger):
        """Test ending file processing that wasn't started"""
        # Should not raise exception
        pipeline_logger.end_file_processing("/nonexistent/file.xlsx", True)
        
        # Metrics should not be updated
        assert pipeline_logger.metrics.files_processed == 0
    
    def test_aggregation_logging(self, pipeline_logger):
        """Test aggregation logging methods"""
        # Test aggregation start
        pipeline_logger.log_aggregation_start(5)
        
        # Test deduplication logging
        pipeline_logger.log_deduplication(1000, 50, 950)
    
    def test_export_logging(self, pipeline_logger):
        """Test export logging methods"""
        # Test export start
        pipeline_logger.log_export_start(500, ['Excel', 'CSV'])
        
        # Test export complete
        pipeline_logger.log_export_complete(
            "/output/results.xlsx", 
            500, 
            2.5, 
            10.0
        )
    
    def test_complete_pipeline(self, pipeline_logger):
        """Test pipeline completion"""
        # Add some test data
        pipeline_logger.metrics.files_processed = 5
        pipeline_logger.metrics.files_successful = 4
        pipeline_logger.metrics.files_failed = 1
        pipeline_logger.metrics.total_properties = 1000
        pipeline_logger.metrics.total_flex_candidates = 150
        
        final_metrics = pipeline_logger.complete_pipeline()
        
        assert final_metrics.end_time is not None
        assert final_metrics.files_processed == 5
        assert final_metrics.success_rate == 80.0
    
    def test_save_processing_log(self, pipeline_logger):
        """Test saving processing log to JSON"""
        # Add some test data
        pipeline_logger.metrics.files_processed = 3
        pipeline_logger.metrics.files_successful = 2
        pipeline_logger.metrics.files_failed = 1
        pipeline_logger.file_progress["/test/file1.xlsx"] = {
            'status': 'completed',
            'property_count': 100
        }
        
        # Use a temporary file for testing
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            log_path = f.name
        
        saved_path = pipeline_logger.save_processing_log(log_path)
        
        assert Path(saved_path).exists()
        
        # Verify log content
        with open(saved_path, 'r') as f:
            log_data = json.load(f)
        
        assert 'pipeline_summary' in log_data
        assert 'file_details' in log_data
        assert 'stage_timings' in log_data
        
        summary = log_data['pipeline_summary']
        assert summary['files_processed'] == 3
        assert summary['files_successful'] == 2
        assert summary['files_failed'] == 1
        
        # Cleanup
        Path(saved_path).unlink()
    
    def test_convenience_logging_methods(self, pipeline_logger):
        """Test convenience logging methods"""
        # These should not raise exceptions
        pipeline_logger.log_warning("Test warning")
        pipeline_logger.log_error("Test error")
        pipeline_logger.log_error("Test error with exception", Exception("Test exception"))
        pipeline_logger.log_success("Test success")
    
    @patch('psutil.Process')
    def test_memory_monitoring_enabled(self, mock_process):
        """Test memory monitoring when enabled"""
        # Mock memory info
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 1024 * 1024 * 512  # 512 MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        logger = PipelineLogger(enable_performance_monitoring=True)
        
        # Simulate file processing to trigger memory monitoring
        logger.start_file_processing("/test/file.xlsx", 1.0)
        logger.end_file_processing("/test/file.xlsx", True, 100, 25)
        
        # Check that memory was tracked
        assert logger.metrics.memory_usage_mb > 0
        assert logger.metrics.peak_memory_mb > 0


class TestFileProgressTracker:
    """Test cases for FileProgressTracker"""
    
    @pytest.fixture
    def pipeline_logger(self):
        """Create PipelineLogger for testing"""
        return PipelineLogger(enable_performance_monitoring=False)
    
    def test_initialization(self, pipeline_logger):
        """Test FileProgressTracker initialization"""
        file_path = "/test/file.xlsx"
        total_records = 1000
        
        tracker = FileProgressTracker(pipeline_logger, file_path, total_records)
        
        assert tracker.pipeline_logger == pipeline_logger
        assert tracker.file_path == file_path
        assert tracker.file_name == "file.xlsx"
        assert tracker.progress_logger is not None
    
    def test_progress_updates(self, pipeline_logger):
        """Test progress updates"""
        tracker = FileProgressTracker(pipeline_logger, "/test/file.xlsx", 100)
        
        # These should not raise exceptions
        tracker.update(10, "Processing records")
        tracker.update(20)
        tracker.complete("Processing complete")


if __name__ == "__main__":
    pytest.main([__file__])