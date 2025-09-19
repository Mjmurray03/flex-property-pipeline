"""
Pipeline-specific logging and progress tracking for Scalable Multi-File Pipeline
Extends the existing logging infrastructure with pipeline-specific functionality
"""

import logging
import time
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from utils.logger import setup_logging, ProgressLogger
import json


@dataclass
class ProcessingMetrics:
    """Metrics for tracking processing performance"""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    files_processed: int = 0
    files_successful: int = 0
    files_failed: int = 0
    total_properties: int = 0
    total_flex_candidates: int = 0
    memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0
    processing_errors: List[str] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        """Calculate processing duration in seconds"""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.files_processed == 0:
            return 0.0
        return (self.files_successful / self.files_processed) * 100
    
    @property
    def processing_rate(self) -> float:
        """Calculate files processed per minute"""
        duration_minutes = self.duration_seconds / 60
        if duration_minutes == 0:
            return 0.0
        return self.files_processed / duration_minutes


class PipelineLogger:
    """
    Comprehensive logging system for the scalable multi-file pipeline.
    Provides progress tracking, performance monitoring, and detailed logging.
    """
    
    def __init__(self, 
                 log_level: str = 'INFO',
                 log_file: Optional[str] = None,
                 enable_performance_monitoring: bool = True):
        """
        Initialize the pipeline logger.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Custom log file path
            enable_performance_monitoring: Whether to track memory and performance metrics
        """
        self.logger = setup_logging(
            name='pipeline',
            level=log_level,
            log_file=log_file,
            console=True,
            file_logging=True
        )
        
        self.enable_performance_monitoring = enable_performance_monitoring
        self.metrics = ProcessingMetrics()
        self.file_progress: Dict[str, Dict[str, Any]] = {}
        self.stage_timings: Dict[str, float] = {}
        
        # Memory monitoring
        self.process = psutil.Process() if enable_performance_monitoring else None
        
        self.logger.info("Pipeline logger initialized")
    
    def start_pipeline(self, total_files: int, input_folder: str) -> None:
        """
        Log pipeline start and initialize tracking.
        
        Args:
            total_files: Total number of files to process
            input_folder: Input folder path
        """
        self.metrics = ProcessingMetrics()
        self.file_progress = {}
        self.stage_timings = {}
        
        self.logger.info("=" * 80)
        self.logger.info("SCALABLE MULTI-FILE PIPELINE STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"Input folder: {input_folder}")
        self.logger.info(f"Total files to process: {total_files}")
        self.logger.info(f"Start time: {self.metrics.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.enable_performance_monitoring:
            self._log_system_info()
    
    def _log_system_info(self) -> None:
        """Log system information for performance monitoring"""
        try:
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            
            self.logger.info(f"System Memory: {memory.total / (1024**3):.1f} GB total, "
                           f"{memory.available / (1024**3):.1f} GB available")
            self.logger.info(f"CPU Cores: {cpu_count}")
            
            if self.process:
                process_memory = self.process.memory_info().rss / (1024**2)
                self.logger.info(f"Initial process memory: {process_memory:.1f} MB")
                
        except Exception as e:
            self.logger.warning(f"Could not log system info: {e}")
    
    def start_stage(self, stage_name: str) -> None:
        """
        Start timing a processing stage.
        
        Args:
            stage_name: Name of the processing stage
        """
        self.stage_timings[stage_name] = time.time()
        self.logger.info(f"Starting stage: {stage_name}")
    
    def end_stage(self, stage_name: str) -> float:
        """
        End timing a processing stage and log duration.
        
        Args:
            stage_name: Name of the processing stage
            
        Returns:
            Duration in seconds
        """
        if stage_name not in self.stage_timings:
            self.logger.warning(f"Stage '{stage_name}' was not started")
            return 0.0
        
        duration = time.time() - self.stage_timings[stage_name]
        self.logger.info(f"Completed stage: {stage_name} ({duration:.2f}s)")
        return duration
    
    def start_file_processing(self, file_path: str, file_size_mb: float) -> None:
        """
        Log start of individual file processing.
        
        Args:
            file_path: Path to the file being processed
            file_size_mb: File size in megabytes
        """
        self.file_progress[file_path] = {
            'start_time': time.time(),
            'file_size_mb': file_size_mb,
            'status': 'processing'
        }
        
        self.logger.info(f"Processing file: {Path(file_path).name} ({file_size_mb:.2f} MB)")
    
    def end_file_processing(self, 
                          file_path: str, 
                          success: bool, 
                          property_count: int = 0,
                          flex_candidate_count: int = 0,
                          error_message: Optional[str] = None) -> None:
        """
        Log completion of individual file processing.
        
        Args:
            file_path: Path to the processed file
            success: Whether processing was successful
            property_count: Number of properties in the file
            flex_candidate_count: Number of flex candidates found
            error_message: Error message if processing failed
        """
        if file_path not in self.file_progress:
            self.logger.warning(f"File '{file_path}' processing was not started")
            return
        
        file_info = self.file_progress[file_path]
        duration = time.time() - file_info['start_time']
        file_name = Path(file_path).name
        
        # Update metrics
        self.metrics.files_processed += 1
        if success:
            self.metrics.files_successful += 1
            self.metrics.total_properties += property_count
            self.metrics.total_flex_candidates += flex_candidate_count
            
            self.logger.info(f"[SUCCESS] Completed: {file_name} ({duration:.2f}s) - "
                           f"{property_count} properties, {flex_candidate_count} flex candidates")
        else:
            self.metrics.files_failed += 1
            if error_message:
                self.metrics.processing_errors.append(f"{file_name}: {error_message}")
            
            self.logger.error(f"[FAILED] Failed: {file_name} ({duration:.2f}s) - {error_message or 'Unknown error'}")
        
        # Update file progress
        file_info.update({
            'end_time': time.time(),
            'duration': duration,
            'status': 'completed' if success else 'failed',
            'property_count': property_count,
            'flex_candidate_count': flex_candidate_count,
            'error_message': error_message
        })
        
        # Log progress summary
        self._log_progress_summary()
        
        # Monitor memory usage
        if self.enable_performance_monitoring:
            self._monitor_memory_usage()
    
    def _log_progress_summary(self) -> None:
        """Log current progress summary"""
        processed = self.metrics.files_processed
        successful = self.metrics.files_successful
        failed = self.metrics.files_failed
        
        if processed > 0:
            success_rate = (successful / processed) * 100
            self.logger.info(f"Progress: {processed} files processed "
                           f"({successful} successful, {failed} failed) - "
                           f"Success rate: {success_rate:.1f}%")
    
    def _monitor_memory_usage(self) -> None:
        """Monitor and log memory usage"""
        if not self.process:
            return
        
        try:
            memory_info = self.process.memory_info()
            current_memory_mb = memory_info.rss / (1024**2)
            
            self.metrics.memory_usage_mb = current_memory_mb
            if current_memory_mb > self.metrics.peak_memory_mb:
                self.metrics.peak_memory_mb = current_memory_mb
            
            # Log memory warnings
            if current_memory_mb > 2048:  # 2GB
                self.logger.warning(f"High memory usage: {current_memory_mb:.1f} MB")
            elif current_memory_mb > 1024:  # 1GB
                self.logger.info(f"Memory usage: {current_memory_mb:.1f} MB")
                
        except Exception as e:
            self.logger.debug(f"Could not monitor memory usage: {e}")
    
    def log_aggregation_start(self, total_results: int) -> None:
        """
        Log start of result aggregation.
        
        Args:
            total_results: Total number of result sets to aggregate
        """
        self.logger.info(f"Starting result aggregation for {total_results} result sets")
    
    def log_deduplication(self, 
                         original_count: int, 
                         duplicate_count: int, 
                         final_count: int) -> None:
        """
        Log deduplication results.
        
        Args:
            original_count: Original number of properties
            duplicate_count: Number of duplicates found
            final_count: Final number of unique properties
        """
        duplicate_rate = (duplicate_count / original_count) * 100 if original_count > 0 else 0
        
        self.logger.info(f"Deduplication complete: {original_count} -> {final_count} properties "
                        f"({duplicate_count} duplicates removed, {duplicate_rate:.1f}% duplicate rate)")
    
    def log_export_start(self, record_count: int, export_formats: List[str]) -> None:
        """
        Log start of export operations.
        
        Args:
            record_count: Number of records to export
            export_formats: List of export formats (e.g., ['Excel', 'CSV'])
        """
        formats_str = ", ".join(export_formats)
        self.logger.info(f"Starting export of {record_count} records to {formats_str}")
    
    def log_export_complete(self, 
                          export_path: str, 
                          record_count: int, 
                          file_size_mb: float,
                          export_time: float) -> None:
        """
        Log completion of export operation.
        
        Args:
            export_path: Path to exported file
            record_count: Number of records exported
            file_size_mb: Size of exported file in MB
            export_time: Time taken for export in seconds
        """
        rate = record_count / export_time if export_time > 0 else 0
        
        self.logger.info(f"[SUCCESS] Export complete: {Path(export_path).name} "
                        f"({record_count} records, {file_size_mb:.2f} MB, {export_time:.2f}s, "
                        f"{rate:.0f} records/sec)")
    
    def complete_pipeline(self) -> ProcessingMetrics:
        """
        Complete pipeline processing and log final summary.
        
        Returns:
            Final processing metrics
        """
        self.metrics.end_time = datetime.now()
        
        self.logger.info("=" * 80)
        self.logger.info("PIPELINE PROCESSING COMPLETE")
        self.logger.info("=" * 80)
        
        # Log summary statistics
        self._log_final_summary()
        
        # Log performance metrics
        if self.enable_performance_monitoring:
            self._log_performance_summary()
        
        # Log any errors
        if self.metrics.processing_errors:
            self._log_error_summary()
        
        self.logger.info("=" * 80)
        
        return self.metrics
    
    def _log_final_summary(self) -> None:
        """Log final processing summary"""
        m = self.metrics
        
        self.logger.info(f"Processing Summary:")
        self.logger.info(f"  Duration: {timedelta(seconds=int(m.duration_seconds))}")
        self.logger.info(f"  Files processed: {m.files_processed}")
        self.logger.info(f"  Successful: {m.files_successful}")
        self.logger.info(f"  Failed: {m.files_failed}")
        self.logger.info(f"  Success rate: {m.success_rate:.1f}%")
        self.logger.info(f"  Total properties: {m.total_properties:,}")
        self.logger.info(f"  Flex candidates: {m.total_flex_candidates:,}")
        
        if m.total_properties > 0:
            flex_rate = (m.total_flex_candidates / m.total_properties) * 100
            self.logger.info(f"  Flex candidate rate: {flex_rate:.2f}%")
        
        self.logger.info(f"  Processing rate: {m.processing_rate:.1f} files/minute")
    
    def _log_performance_summary(self) -> None:
        """Log performance metrics summary"""
        if not self.enable_performance_monitoring:
            return
        
        self.logger.info(f"Performance Metrics:")
        self.logger.info(f"  Peak memory usage: {self.metrics.peak_memory_mb:.1f} MB")
        self.logger.info(f"  Final memory usage: {self.metrics.memory_usage_mb:.1f} MB")
        
        # Log stage timings if available
        if self.stage_timings:
            self.logger.info(f"  Stage timings:")
            for stage, start_time in self.stage_timings.items():
                # Note: This shows start times, not durations
                self.logger.info(f"    {stage}: Started at {datetime.fromtimestamp(start_time).strftime('%H:%M:%S')}")
    
    def _log_error_summary(self) -> None:
        """Log summary of processing errors"""
        self.logger.warning(f"Processing Errors ({len(self.metrics.processing_errors)}):")
        for i, error in enumerate(self.metrics.processing_errors[:10], 1):  # Show first 10 errors
            self.logger.warning(f"  {i}. {error}")
        
        if len(self.metrics.processing_errors) > 10:
            remaining = len(self.metrics.processing_errors) - 10
            self.logger.warning(f"  ... and {remaining} more errors")
    
    def save_processing_log(self, output_path: str) -> str:
        """
        Save detailed processing log to JSON file.
        
        Args:
            output_path: Path for the log file
            
        Returns:
            Path to the saved log file
        """
        log_data = {
            'pipeline_summary': {
                'start_time': self.metrics.start_time.isoformat(),
                'end_time': self.metrics.end_time.isoformat() if self.metrics.end_time else None,
                'duration_seconds': self.metrics.duration_seconds,
                'files_processed': self.metrics.files_processed,
                'files_successful': self.metrics.files_successful,
                'files_failed': self.metrics.files_failed,
                'success_rate': self.metrics.success_rate,
                'total_properties': self.metrics.total_properties,
                'total_flex_candidates': self.metrics.total_flex_candidates,
                'processing_rate': self.metrics.processing_rate,
                'peak_memory_mb': self.metrics.peak_memory_mb,
                'processing_errors': self.metrics.processing_errors
            },
            'file_details': self.file_progress,
            'stage_timings': self.stage_timings
        }
        
        log_path = Path(output_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        self.logger.info(f"Processing log saved to: {log_path}")
        return str(log_path)
    
    def log_warning(self, message: str) -> None:
        """Log a warning message"""
        self.logger.warning(f"[WARNING] {message}")
    
    def log_error(self, message: str, exception: Optional[Exception] = None) -> None:
        """Log an error message"""
        if exception:
            self.logger.error(f"[ERROR] {message}: {str(exception)}", exc_info=True)
        else:
            self.logger.error(f"[ERROR] {message}")
    
    def log_success(self, message: str) -> None:
        """Log a success message"""
        self.logger.info(f"[SUCCESS] {message}")


class FileProgressTracker:
    """
    Specialized progress tracker for individual file processing.
    """
    
    def __init__(self, pipeline_logger: PipelineLogger, file_path: str, total_records: int):
        """
        Initialize file progress tracker.
        
        Args:
            pipeline_logger: Main pipeline logger
            file_path: Path to the file being processed
            total_records: Total number of records in the file
        """
        self.pipeline_logger = pipeline_logger
        self.file_path = file_path
        self.file_name = Path(file_path).name
        self.progress_logger = ProgressLogger(
            pipeline_logger.logger, 
            total_records, 
            f"Processing {self.file_name}"
        )
    
    def update(self, increment: int = 1, message: str = None) -> None:
        """Update progress"""
        self.progress_logger.update(increment, message)
    
    def complete(self, message: str = None) -> None:
        """Mark processing as complete"""
        self.progress_logger.complete(message)