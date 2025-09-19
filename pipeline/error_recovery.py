"""
Error Recovery and Retry Mechanisms for Scalable Multi-File Pipeline
Handles retry logic, error categorization, and recovery strategies
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import traceback


class ErrorCategory(Enum):
    """Categories of errors that can occur during pipeline processing"""
    FILE_ACCESS = "file_access"
    DATA_FORMAT = "data_format"
    PROCESSING = "processing"
    MEMORY = "memory"
    TIMEOUT = "timeout"
    NETWORK = "network"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Severity levels for errors"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ProcessingError:
    """Detailed information about a processing error"""
    file_path: str
    error_category: ErrorCategory
    error_severity: ErrorSeverity
    error_message: str
    exception_type: str
    timestamp: datetime
    retry_count: int = 0
    max_retries: int = 3
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization"""
        return {
            'file_path': self.file_path,
            'error_category': self.error_category.value,
            'error_severity': self.error_severity.value,
            'error_message': self.error_message,
            'exception_type': self.exception_type,
            'timestamp': self.timestamp.isoformat(),
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'stack_trace': self.stack_trace,
            'context': self.context
        }


@dataclass
class RetryStrategy:
    """Configuration for retry behavior"""
    max_retries: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    backoff_multiplier: float = 2.0  # Exponential backoff multiplier
    jitter: bool = True  # Add random jitter to delays
    
    # Category-specific retry settings
    category_overrides: Dict[ErrorCategory, Dict[str, Any]] = field(default_factory=dict)
    
    def get_retry_config(self, category: ErrorCategory) -> Dict[str, Any]:
        """Get retry configuration for specific error category"""
        base_config = {
            'max_retries': self.max_retries,
            'base_delay': self.base_delay,
            'max_delay': self.max_delay,
            'backoff_multiplier': self.backoff_multiplier,
            'jitter': self.jitter
        }
        
        if category in self.category_overrides:
            base_config.update(self.category_overrides[category])
        
        return base_config


class ErrorClassifier:
    """Classifies errors into categories and determines severity"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Error classification rules
        self.classification_rules = {
            # File access errors
            (FileNotFoundError, PermissionError, OSError): (ErrorCategory.FILE_ACCESS, ErrorSeverity.MEDIUM),
            
            # Data format errors
            (ValueError, KeyError, IndexError): (ErrorCategory.DATA_FORMAT, ErrorSeverity.LOW),
            
            # Memory errors
            (MemoryError,): (ErrorCategory.MEMORY, ErrorSeverity.HIGH),
            
            # Timeout errors
            (TimeoutError,): (ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM),
            
            # Processing errors (pandas, Excel, etc.)
            ('XLRDError', 'ParserError', 'EmptyDataError'): (ErrorCategory.DATA_FORMAT, ErrorSeverity.LOW),
        }
    
    def classify_error(self, exception: Exception, context: Dict[str, Any] = None) -> Tuple[ErrorCategory, ErrorSeverity]:
        """
        Classify an exception into category and severity
        
        Args:
            exception: The exception to classify
            context: Additional context about the error
            
        Returns:
            Tuple of (ErrorCategory, ErrorSeverity)
        """
        exception_type = type(exception)
        exception_name = exception_type.__name__
        
        # Check direct type matches
        for error_types, (category, severity) in self.classification_rules.items():
            if isinstance(error_types, tuple):
                if exception_type in error_types:
                    return category, severity
            elif exception_name in error_types:
                return category, severity
        
        # Check error message patterns
        error_message = str(exception).lower()
        
        if any(keyword in error_message for keyword in ['file not found', 'no such file', 'cannot open']):
            return ErrorCategory.FILE_ACCESS, ErrorSeverity.MEDIUM
        
        if any(keyword in error_message for keyword in ['memory', 'out of memory']):
            return ErrorCategory.MEMORY, ErrorSeverity.HIGH
        
        if any(keyword in error_message for keyword in ['timeout', 'timed out']):
            return ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM
        
        if any(keyword in error_message for keyword in ['network', 'connection', 'http']):
            return ErrorCategory.NETWORK, ErrorSeverity.MEDIUM
        
        if any(keyword in error_message for keyword in ['format', 'parse', 'decode', 'invalid']):
            return ErrorCategory.DATA_FORMAT, ErrorSeverity.LOW
        
        # Default classification
        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM
    
    def create_processing_error(self, 
                              file_path: str, 
                              exception: Exception, 
                              context: Dict[str, Any] = None) -> ProcessingError:
        """
        Create a ProcessingError from an exception
        
        Args:
            file_path: Path to the file being processed
            exception: The exception that occurred
            context: Additional context information
            
        Returns:
            ProcessingError instance
        """
        category, severity = self.classify_error(exception, context)
        
        return ProcessingError(
            file_path=file_path,
            error_category=category,
            error_severity=severity,
            error_message=str(exception),
            exception_type=type(exception).__name__,
            timestamp=datetime.now(),
            stack_trace=traceback.format_exc(),
            context=context or {}
        )


class RetryManager:
    """Manages retry logic and backoff strategies"""
    
    def __init__(self, strategy: RetryStrategy = None):
        self.strategy = strategy or RetryStrategy()
        self.logger = logging.getLogger(__name__)
    
    def calculate_delay(self, retry_count: int, category: ErrorCategory) -> float:
        """
        Calculate delay before next retry
        
        Args:
            retry_count: Current retry attempt number
            category: Error category
            
        Returns:
            Delay in seconds
        """
        config = self.strategy.get_retry_config(category)
        
        # Exponential backoff
        delay = config['base_delay'] * (config['backoff_multiplier'] ** retry_count)
        delay = min(delay, config['max_delay'])
        
        # Add jitter if enabled
        if config['jitter']:
            import random
            jitter = random.uniform(0.1, 0.3) * delay
            delay += jitter
        
        return delay
    
    def should_retry(self, error: ProcessingError) -> bool:
        """
        Determine if an error should be retried
        
        Args:
            error: ProcessingError to evaluate
            
        Returns:
            True if should retry, False otherwise
        """
        config = self.strategy.get_retry_config(error.error_category)
        
        # Check retry count
        if error.retry_count >= config['max_retries']:
            return False
        
        # Don't retry critical errors
        if error.error_severity == ErrorSeverity.CRITICAL:
            return False
        
        # Category-specific retry logic
        if error.error_category == ErrorCategory.MEMORY:
            # Only retry memory errors once
            return error.retry_count < 1
        
        if error.error_category == ErrorCategory.FILE_ACCESS:
            # Retry file access errors more aggressively
            return error.retry_count < config['max_retries']
        
        return True
    
    def retry_with_backoff(self, 
                          func: Callable, 
                          file_path: str, 
                          *args, 
                          **kwargs) -> Tuple[Any, Optional[ProcessingError]]:
        """
        Execute function with retry logic and exponential backoff
        
        Args:
            func: Function to execute
            file_path: Path to file being processed (for error reporting)
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function
            
        Returns:
            Tuple of (result, error) - result is None if all retries failed
        """
        classifier = ErrorClassifier()
        last_error = None
        
        for attempt in range(self.strategy.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    self.logger.info(f"Retry successful for {file_path} after {attempt} attempts")
                return result, None
                
            except Exception as e:
                error = classifier.create_processing_error(
                    file_path=file_path,
                    exception=e,
                    context={'attempt': attempt, 'function': func.__name__}
                )
                error.retry_count = attempt
                last_error = error
                
                if not self.should_retry(error):
                    self.logger.error(f"Not retrying {file_path} - {error.error_category.value} error: {error.error_message}")
                    break
                
                if attempt < self.strategy.max_retries:
                    delay = self.calculate_delay(attempt, error.error_category)
                    self.logger.warning(f"Retry {attempt + 1}/{self.strategy.max_retries} for {file_path} in {delay:.1f}s - {error.error_message}")
                    time.sleep(delay)
        
        return None, last_error


class ErrorRecoveryManager:
    """Main error recovery and retry management system"""
    
    def __init__(self, 
                 retry_strategy: RetryStrategy = None,
                 error_log_path: str = "logs/error_recovery.json"):
        self.retry_strategy = retry_strategy or RetryStrategy()
        self.retry_manager = RetryManager(self.retry_strategy)
        self.error_classifier = ErrorClassifier()
        self.error_log_path = Path(error_log_path)
        self.logger = logging.getLogger(__name__)
        
        # Error tracking
        self.failed_files: List[ProcessingError] = []
        self.recovered_files: List[str] = []
        self.permanent_failures: List[ProcessingError] = []
        
        # Ensure error log directory exists
        self.error_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def process_with_recovery(self, 
                            func: Callable, 
                            file_path: str, 
                            *args, 
                            **kwargs) -> Tuple[Any, bool]:
        """
        Process a file with error recovery and retry logic
        
        Args:
            func: Processing function to execute
            file_path: Path to file being processed
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function
            
        Returns:
            Tuple of (result, success) - result is None if processing failed
        """
        result, error = self.retry_manager.retry_with_backoff(
            func, file_path, *args, **kwargs
        )
        
        if error is None:
            if file_path in [e.file_path for e in self.failed_files]:
                self.recovered_files.append(file_path)
                self.logger.info(f"Successfully recovered file: {file_path}")
            return result, True
        else:
            self.failed_files.append(error)
            if not self.retry_manager.should_retry(error):
                self.permanent_failures.append(error)
            self.logger.error(f"Failed to process file: {file_path} - {error.error_message}")
            return None, False
    
    def resume_failed_files(self, 
                           processing_func: Callable, 
                           failed_files: List[str] = None) -> Dict[str, Any]:
        """
        Resume processing of previously failed files
        
        Args:
            processing_func: Function to use for processing files
            failed_files: List of file paths to retry (if None, uses stored failures)
            
        Returns:
            Dictionary with resume results
        """
        if failed_files is None:
            failed_files = [error.file_path for error in self.failed_files 
                          if error.file_path not in self.recovered_files]
        
        if not failed_files:
            self.logger.info("No failed files to resume")
            return {'resumed_files': 0, 'successful': 0, 'failed': 0}
        
        self.logger.info(f"Resuming processing of {len(failed_files)} failed files")
        
        successful = 0
        still_failed = 0
        
        for file_path in failed_files:
            self.logger.info(f"Resuming processing: {file_path}")
            result, success = self.process_with_recovery(processing_func, file_path)
            
            if success:
                successful += 1
            else:
                still_failed += 1
        
        results = {
            'resumed_files': len(failed_files),
            'successful': successful,
            'failed': still_failed,
            'recovery_rate': successful / len(failed_files) if failed_files else 0
        }
        
        self.logger.info(f"Resume completed: {successful}/{len(failed_files)} files recovered")
        return results
    
    def generate_error_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive error report with troubleshooting information
        
        Returns:
            Dictionary containing error analysis and troubleshooting info
        """
        # Categorize errors
        error_by_category = {}
        error_by_severity = {}
        
        for error in self.failed_files:
            category = error.error_category.value
            severity = error.error_severity.value
            
            if category not in error_by_category:
                error_by_category[category] = []
            error_by_category[category].append(error)
            
            if severity not in error_by_severity:
                error_by_severity[severity] = []
            error_by_severity[severity].append(error)
        
        # Generate troubleshooting recommendations
        recommendations = self._generate_troubleshooting_recommendations(error_by_category)
        
        report = {
            'summary': {
                'total_errors': len(self.failed_files),
                'recovered_files': len(self.recovered_files),
                'permanent_failures': len(self.permanent_failures),
                'recovery_rate': len(self.recovered_files) / len(self.failed_files) if self.failed_files else 0
            },
            'error_breakdown': {
                'by_category': {cat: len(errors) for cat, errors in error_by_category.items()},
                'by_severity': {sev: len(errors) for sev, errors in error_by_severity.items()}
            },
            'failed_files': [error.to_dict() for error in self.failed_files],
            'recovered_files': self.recovered_files,
            'troubleshooting_recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def _generate_troubleshooting_recommendations(self, 
                                                error_by_category: Dict[str, List[ProcessingError]]) -> List[Dict[str, str]]:
        """Generate troubleshooting recommendations based on error patterns"""
        recommendations = []
        
        if ErrorCategory.FILE_ACCESS.value in error_by_category:
            recommendations.append({
                'category': 'File Access Issues',
                'recommendation': 'Check file permissions, ensure files are not locked by other applications, and verify file paths are correct.',
                'affected_files': len(error_by_category[ErrorCategory.FILE_ACCESS.value])
            })
        
        if ErrorCategory.MEMORY.value in error_by_category:
            recommendations.append({
                'category': 'Memory Issues',
                'recommendation': 'Reduce batch size, increase system memory, or process files individually. Consider using chunked processing for large files.',
                'affected_files': len(error_by_category[ErrorCategory.MEMORY.value])
            })
        
        if ErrorCategory.DATA_FORMAT.value in error_by_category:
            recommendations.append({
                'category': 'Data Format Issues',
                'recommendation': 'Verify Excel file format, check for required columns, and ensure data types are consistent across files.',
                'affected_files': len(error_by_category[ErrorCategory.DATA_FORMAT.value])
            })
        
        if ErrorCategory.TIMEOUT.value in error_by_category:
            recommendations.append({
                'category': 'Timeout Issues',
                'recommendation': 'Increase timeout settings, reduce concurrent workers, or split large files into smaller chunks.',
                'affected_files': len(error_by_category[ErrorCategory.TIMEOUT.value])
            })
        
        return recommendations
    
    def save_error_log(self) -> str:
        """
        Save error log to file
        
        Returns:
            Path to saved error log file
        """
        try:
            report = self.generate_error_report()
            
            with open(self.error_log_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Error log saved to: {self.error_log_path}")
            return str(self.error_log_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save error log: {e}")
            raise
    
    def load_error_log(self, log_path: str = None) -> Dict[str, Any]:
        """
        Load error log from file
        
        Args:
            log_path: Path to error log file (uses default if None)
            
        Returns:
            Dictionary containing error log data
        """
        log_file = Path(log_path) if log_path else self.error_log_path
        
        try:
            with open(log_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Error log file not found: {log_file}")
            return {}
        except Exception as e:
            self.logger.error(f"Failed to load error log: {e}")
            raise
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """
        Get recovery statistics
        
        Returns:
            Dictionary with recovery statistics
        """
        total_files = len(self.failed_files) + len(self.recovered_files)
        
        return {
            'total_files_processed': total_files,
            'failed_files': len(self.failed_files),
            'recovered_files': len(self.recovered_files),
            'permanent_failures': len(self.permanent_failures),
            'recovery_rate': len(self.recovered_files) / total_files if total_files > 0 else 0,
            'failure_rate': len(self.permanent_failures) / total_files if total_files > 0 else 0
        }


# Convenience function for creating default retry strategies
def create_default_retry_strategy() -> RetryStrategy:
    """Create a default retry strategy with sensible defaults"""
    strategy = RetryStrategy(
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        backoff_multiplier=2.0,
        jitter=True
    )
    
    # Category-specific overrides
    strategy.category_overrides = {
        ErrorCategory.FILE_ACCESS: {'max_retries': 5, 'base_delay': 0.5},
        ErrorCategory.MEMORY: {'max_retries': 1, 'base_delay': 5.0},
        ErrorCategory.TIMEOUT: {'max_retries': 2, 'base_delay': 10.0},
        ErrorCategory.DATA_FORMAT: {'max_retries': 1, 'base_delay': 0.1}
    }
    
    return strategy