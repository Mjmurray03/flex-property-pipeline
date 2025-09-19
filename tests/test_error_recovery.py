"""
Tests for error recovery and retry mechanisms
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch
import json

from pipeline.error_recovery import (
    ErrorRecoveryManager, RetryManager, ErrorClassifier,
    ProcessingError, ErrorCategory, ErrorSeverity, RetryStrategy,
    create_default_retry_strategy
)


class TestErrorClassifier:
    """Test error classification functionality"""
    
    def test_classify_file_access_errors(self):
        """Test classification of file access errors"""
        classifier = ErrorClassifier()
        
        # Test FileNotFoundError
        error = FileNotFoundError("File not found")
        category, severity = classifier.classify_error(error)
        assert category == ErrorCategory.FILE_ACCESS
        assert severity == ErrorSeverity.MEDIUM
        
        # Test PermissionError
        error = PermissionError("Permission denied")
        category, severity = classifier.classify_error(error)
        assert category == ErrorCategory.FILE_ACCESS
        assert severity == ErrorSeverity.MEDIUM
    
    def test_classify_data_format_errors(self):
        """Test classification of data format errors"""
        classifier = ErrorClassifier()
        
        # Test ValueError
        error = ValueError("Invalid data format")
        category, severity = classifier.classify_error(error)
        assert category == ErrorCategory.DATA_FORMAT
        assert severity == ErrorSeverity.LOW
        
        # Test KeyError
        error = KeyError("Missing column")
        category, severity = classifier.classify_error(error)
        assert category == ErrorCategory.DATA_FORMAT
        assert severity == ErrorSeverity.LOW
    
    def test_classify_memory_errors(self):
        """Test classification of memory errors"""
        classifier = ErrorClassifier()
        
        error = MemoryError("Out of memory")
        category, severity = classifier.classify_error(error)
        assert category == ErrorCategory.MEMORY
        assert severity == ErrorSeverity.HIGH
    
    def test_classify_unknown_errors(self):
        """Test classification of unknown errors"""
        classifier = ErrorClassifier()
        
        error = RuntimeError("Unknown error")
        category, severity = classifier.classify_error(error)
        assert category == ErrorCategory.UNKNOWN
        assert severity == ErrorSeverity.MEDIUM
    
    def test_create_processing_error(self):
        """Test creation of ProcessingError objects"""
        classifier = ErrorClassifier()
        
        exception = ValueError("Test error")
        error = classifier.create_processing_error(
            file_path="/test/file.xlsx",
            exception=exception,
            context={'test': 'context'}
        )
        
        assert error.file_path == "/test/file.xlsx"
        assert error.error_category == ErrorCategory.DATA_FORMAT
        assert error.error_severity == ErrorSeverity.LOW
        assert error.error_message == "Test error"
        assert error.exception_type == "ValueError"
        assert error.context == {'test': 'context'}


class TestRetryManager:
    """Test retry management functionality"""
    
    def test_calculate_delay(self):
        """Test delay calculation with exponential backoff"""
        strategy = RetryStrategy(base_delay=1.0, backoff_multiplier=2.0, jitter=False)
        manager = RetryManager(strategy)
        
        # Test exponential backoff
        delay1 = manager.calculate_delay(0, ErrorCategory.PROCESSING)
        delay2 = manager.calculate_delay(1, ErrorCategory.PROCESSING)
        delay3 = manager.calculate_delay(2, ErrorCategory.PROCESSING)
        
        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 4.0
    
    def test_should_retry_logic(self):
        """Test retry decision logic"""
        strategy = RetryStrategy(max_retries=3)
        manager = RetryManager(strategy)
        
        # Test normal retry
        error = ProcessingError(
            file_path="/test/file.xlsx",
            error_category=ErrorCategory.DATA_FORMAT,
            error_severity=ErrorSeverity.LOW,
            error_message="Test error",
            exception_type="ValueError",
            timestamp=None,
            retry_count=1
        )
        assert manager.should_retry(error) is True
        
        # Test max retries exceeded
        error.retry_count = 5
        assert manager.should_retry(error) is False
        
        # Test critical error
        error.retry_count = 1
        error.error_severity = ErrorSeverity.CRITICAL
        assert manager.should_retry(error) is False
    
    def test_retry_with_backoff_success(self):
        """Test successful retry with backoff"""
        strategy = RetryStrategy(max_retries=2, base_delay=0.01)  # Fast for testing
        manager = RetryManager(strategy)
        
        # Mock function that fails once then succeeds
        call_count = 0
        def mock_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First attempt fails")
            return "success"
        
        result, error = manager.retry_with_backoff(mock_function, "/test/file.xlsx")
        
        assert result == "success"
        assert error is None
        assert call_count == 2
    
    def test_retry_with_backoff_failure(self):
        """Test retry with backoff when all attempts fail"""
        strategy = RetryStrategy(max_retries=2, base_delay=0.01)
        manager = RetryManager(strategy)
        
        # Mock function that always fails
        def mock_function():
            raise ValueError("Always fails")
        
        result, error = manager.retry_with_backoff(mock_function, "/test/file.xlsx")
        
        assert result is None
        assert error is not None
        assert error.error_category == ErrorCategory.DATA_FORMAT
        assert error.retry_count == 2


class TestErrorRecoveryManager:
    """Test error recovery management"""
    
    @pytest.fixture
    def temp_error_log(self):
        """Create temporary error log file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)
    
    def test_process_with_recovery_success(self, temp_error_log):
        """Test successful processing with recovery"""
        manager = ErrorRecoveryManager(error_log_path=temp_error_log)
        
        def mock_function():
            return "success"
        
        result, success = manager.process_with_recovery(mock_function, "/test/file.xlsx")
        
        assert result == "success"
        assert success is True
        assert len(manager.failed_files) == 0
    
    def test_process_with_recovery_failure(self, temp_error_log):
        """Test processing failure with recovery"""
        strategy = RetryStrategy(max_retries=1, base_delay=0.01)
        manager = ErrorRecoveryManager(retry_strategy=strategy, error_log_path=temp_error_log)
        
        def mock_function():
            raise ValueError("Always fails")
        
        result, success = manager.process_with_recovery(mock_function, "/test/file.xlsx")
        
        assert result is None
        assert success is False
        assert len(manager.failed_files) == 1
        assert manager.failed_files[0].file_path == "/test/file.xlsx"
    
    def test_resume_failed_files(self, temp_error_log):
        """Test resuming failed files"""
        strategy = RetryStrategy(max_retries=1, base_delay=0.01)
        manager = ErrorRecoveryManager(retry_strategy=strategy, error_log_path=temp_error_log)
        
        # First, create some failed files
        def failing_function():
            raise ValueError("Initial failure")
        
        manager.process_with_recovery(failing_function, "/test/file1.xlsx")
        manager.process_with_recovery(failing_function, "/test/file2.xlsx")
        
        assert len(manager.failed_files) == 2
        
        # Now resume with a working function
        def working_function():
            return "success"
        
        results = manager.resume_failed_files(working_function)
        
        assert results['resumed_files'] == 2
        assert results['successful'] == 2
        assert results['failed'] == 0
        assert len(manager.recovered_files) == 2
    
    def test_generate_error_report(self, temp_error_log):
        """Test error report generation"""
        manager = ErrorRecoveryManager(error_log_path=temp_error_log)
        
        # Add some test errors
        error1 = ProcessingError(
            file_path="/test/file1.xlsx",
            error_category=ErrorCategory.FILE_ACCESS,
            error_severity=ErrorSeverity.MEDIUM,
            error_message="File not found",
            exception_type="FileNotFoundError",
            timestamp=None
        )
        
        error2 = ProcessingError(
            file_path="/test/file2.xlsx",
            error_category=ErrorCategory.DATA_FORMAT,
            error_severity=ErrorSeverity.LOW,
            error_message="Invalid format",
            exception_type="ValueError",
            timestamp=None
        )
        
        manager.failed_files = [error1, error2]
        manager.recovered_files = ["/test/file3.xlsx"]
        
        report = manager.generate_error_report()
        
        assert report['summary']['total_errors'] == 2
        assert report['summary']['recovered_files'] == 1
        assert report['error_breakdown']['by_category']['file_access'] == 1
        assert report['error_breakdown']['by_category']['data_format'] == 1
        assert len(report['troubleshooting_recommendations']) > 0
    
    def test_save_and_load_error_log(self, temp_error_log):
        """Test saving and loading error logs"""
        manager = ErrorRecoveryManager(error_log_path=temp_error_log)
        
        # Add test error
        error = ProcessingError(
            file_path="/test/file.xlsx",
            error_category=ErrorCategory.FILE_ACCESS,
            error_severity=ErrorSeverity.MEDIUM,
            error_message="Test error",
            exception_type="FileNotFoundError",
            timestamp=None
        )
        manager.failed_files = [error]
        
        # Save error log
        saved_path = manager.save_error_log()
        assert Path(saved_path).exists()
        
        # Load error log
        loaded_data = manager.load_error_log()
        assert loaded_data['summary']['total_errors'] == 1
        assert len(loaded_data['failed_files']) == 1
    
    def test_get_recovery_statistics(self, temp_error_log):
        """Test recovery statistics calculation"""
        manager = ErrorRecoveryManager(error_log_path=temp_error_log)
        
        # Add test data
        manager.failed_files = [Mock() for _ in range(3)]
        manager.recovered_files = ["file1.xlsx", "file2.xlsx"]
        manager.permanent_failures = [Mock()]
        
        stats = manager.get_recovery_statistics()
        
        assert stats['total_files_processed'] == 5
        assert stats['failed_files'] == 3
        assert stats['recovered_files'] == 2
        assert stats['permanent_failures'] == 1
        assert stats['recovery_rate'] == 0.4  # 2/5
        assert stats['failure_rate'] == 0.2   # 1/5


class TestRetryStrategy:
    """Test retry strategy configuration"""
    
    def test_default_retry_strategy(self):
        """Test default retry strategy creation"""
        strategy = create_default_retry_strategy()
        
        assert strategy.max_retries == 3
        assert strategy.base_delay == 1.0
        assert strategy.backoff_multiplier == 2.0
        assert strategy.jitter is True
        
        # Test category overrides
        file_config = strategy.get_retry_config(ErrorCategory.FILE_ACCESS)
        assert file_config['max_retries'] == 5
        
        memory_config = strategy.get_retry_config(ErrorCategory.MEMORY)
        assert memory_config['max_retries'] == 1
    
    def test_retry_strategy_overrides(self):
        """Test retry strategy category overrides"""
        strategy = RetryStrategy()
        strategy.category_overrides[ErrorCategory.TIMEOUT] = {
            'max_retries': 10,
            'base_delay': 5.0
        }
        
        config = strategy.get_retry_config(ErrorCategory.TIMEOUT)
        assert config['max_retries'] == 10
        assert config['base_delay'] == 5.0
        
        # Other categories should use defaults
        default_config = strategy.get_retry_config(ErrorCategory.DATA_FORMAT)
        assert default_config['max_retries'] == 3  # Default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])