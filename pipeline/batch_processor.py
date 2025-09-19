"""
Batch Processor for Scalable Multi-File Pipeline
Handles concurrent processing of multiple Excel files with progress tracking and error handling
"""

import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time
import threading
from queue import Queue

from pipeline.file_processor import FileProcessor, ProcessingResult
from pipeline.error_recovery import ErrorRecoveryManager, create_default_retry_strategy


@dataclass
class BatchProcessingStats:
    """Statistics for batch processing operations"""
    
    total_files: int = 0
    processed_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_properties: int = 0
    total_flex_candidates: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    processing_duration: float = 0.0
    average_processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary"""
        return {
            'total_files': self.total_files,
            'processed_files': self.processed_files,
            'successful_files': self.successful_files,
            'failed_files': self.failed_files,
            'total_properties': self.total_properties,
            'total_flex_candidates': self.total_flex_candidates,
            'processing_duration': self.processing_duration,
            'average_processing_time': self.average_processing_time,
            'success_rate': self.successful_files / max(1, self.processed_files),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None
        }


class ProgressTracker:
    """Thread-safe progress tracking for batch operations"""
    
    def __init__(self, total_files: int, progress_callback: Optional[Callable] = None):
        self.total_files = total_files
        self.processed_files = 0
        self.successful_files = 0
        self.failed_files = 0
        self.progress_callback = progress_callback
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def update_progress(self, result: ProcessingResult):
        """Update progress with a processing result"""
        with self._lock:
            self.processed_files += 1
            
            if result.success:
                self.successful_files += 1
            else:
                self.failed_files += 1
            
            # Calculate progress percentage
            progress_pct = (self.processed_files / self.total_files) * 100
            
            # Log progress
            self.logger.info(f"Progress: {self.processed_files}/{self.total_files} "
                           f"({progress_pct:.1f}%) - "
                           f"Success: {self.successful_files}, Failed: {self.failed_files}")
            
            # Call progress callback if provided
            if self.progress_callback:
                try:
                    self.progress_callback({
                        'processed': self.processed_files,
                        'total': self.total_files,
                        'successful': self.successful_files,
                        'failed': self.failed_files,
                        'progress_pct': progress_pct,
                        'current_file': result.file_path
                    })
                except Exception as e:
                    self.logger.warning(f"Progress callback failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current progress statistics"""
        with self._lock:
            return {
                'total_files': self.total_files,
                'processed_files': self.processed_files,
                'successful_files': self.successful_files,
                'failed_files': self.failed_files,
                'progress_pct': (self.processed_files / self.total_files) * 100 if self.total_files > 0 else 0
            }


class BatchProcessor:
    """
    Orchestrates concurrent processing of multiple Excel files
    
    Provides progress tracking, error handling, and performance metrics
    """
    
    def __init__(self, 
                 max_workers: int = 4,
                 min_flex_score: float = 4.0,
                 timeout_minutes: int = 30,
                 progress_callback: Optional[Callable] = None,
                 enable_error_recovery: bool = True,
                 error_log_path: str = "logs/batch_processing_errors.json",
                 enable_performance_optimization: bool = True,
                 memory_limit_gb: float = 4.0):
        """
        Initialize BatchProcessor
        
        Args:
            max_workers: Maximum number of concurrent file processors
            min_flex_score: Minimum flex score threshold
            timeout_minutes: Timeout for individual file processing
            progress_callback: Optional callback for progress updates
            enable_error_recovery: Whether to enable error recovery and retry mechanisms
            error_log_path: Path to error log file
            enable_performance_optimization: Whether to enable performance optimization
            memory_limit_gb: Memory limit for processing operations
        """
        self.max_workers = max_workers
        self.min_flex_score = min_flex_score
        self.timeout_seconds = timeout_minutes * 60
        self.progress_callback = progress_callback
        self.enable_error_recovery = enable_error_recovery
        self.enable_performance_optimization = enable_performance_optimization
        self.memory_limit_gb = memory_limit_gb
        
        self.logger = logging.getLogger(__name__)
        
        # Processing state
        self.stats = BatchProcessingStats()
        self.processing_results: List[ProcessingResult] = []
        self.error_queue = Queue()
        
        # Performance optimization
        if self.enable_performance_optimization:
            from pipeline.performance_optimizer import PerformanceOptimizer
            self.performance_optimizer = PerformanceOptimizer(
                memory_limit_gb=memory_limit_gb,
                chunk_size=10000,
                enable_monitoring=True,
                optimization_level="balanced"
            )
        else:
            self.performance_optimizer = None
        
        # Error recovery system
        if self.enable_error_recovery:
            retry_strategy = create_default_retry_strategy()
            self.error_recovery = ErrorRecoveryManager(
                retry_strategy=retry_strategy,
                error_log_path=error_log_path
            )
        else:
            self.error_recovery = None
        
        self.logger.info(f"BatchProcessor initialized with {max_workers} workers, "
                        f"min_flex_score={min_flex_score}, timeout={timeout_minutes}min, "
                        f"error_recovery={'enabled' if enable_error_recovery else 'disabled'}, "
                        f"performance_optimization={'enabled' if enable_performance_optimization else 'disabled'}")
    
    def process_files(self, file_paths: List[Path]) -> List[ProcessingResult]:
        """
        Process multiple files concurrently
        
        Args:
            file_paths: List of Excel file paths to process
            
        Returns:
            List of ProcessingResult objects
        """
        if not file_paths:
            self.logger.warning("No files provided for processing")
            return []
        
        self.logger.info(f"Starting batch processing of {len(file_paths)} files")
        
        # Initialize stats
        self.stats = BatchProcessingStats(
            total_files=len(file_paths),
            start_time=datetime.now()
        )
        
        # Initialize progress tracker
        progress_tracker = ProgressTracker(
            total_files=len(file_paths),
            progress_callback=self.progress_callback
        )
        
        # Process files concurrently
        results = self._process_files_concurrent(file_paths, progress_tracker)
        
        # Finalize stats
        self.stats.end_time = datetime.now()
        self.stats.processing_duration = (self.stats.end_time - self.stats.start_time).total_seconds()
        self.stats.processed_files = len(results)
        self.stats.successful_files = sum(1 for r in results if r.success)
        self.stats.failed_files = sum(1 for r in results if not r.success)
        
        # Calculate totals
        self.stats.total_properties = sum(r.property_count for r in results if r.success)
        self.stats.total_flex_candidates = sum(r.flex_candidate_count for r in results if r.success)
        
        # Calculate average processing time
        processing_times = [r.processing_time for r in results if r.processing_time > 0]
        self.stats.average_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        self.processing_results = results
        
        self.logger.info(f"Batch processing completed: {self.stats.successful_files}/{self.stats.processed_files} "
                        f"files successful in {self.stats.processing_duration:.2f}s")
        
        return results
    
    def _process_files_concurrent(self, file_paths: List[Path], progress_tracker: ProgressTracker) -> List[ProcessingResult]:
        """Process files using ThreadPoolExecutor for concurrency"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all file processing tasks
            future_to_path = {}
            
            for file_path in file_paths:
                future = executor.submit(self._process_single_file_with_timeout, file_path)
                future_to_path[future] = file_path
            
            # Collect results as they complete
            for future in as_completed(future_to_path, timeout=self.timeout_seconds * len(file_paths)):
                file_path = future_to_path[future]
                
                try:
                    result = future.result(timeout=self.timeout_seconds)
                    results.append(result)
                    progress_tracker.update_progress(result)
                    
                    # Log individual file completion
                    if result.success:
                        self.logger.debug(f"Completed {file_path.name}: "
                                        f"{result.flex_candidate_count} candidates found")
                    else:
                        self.logger.warning(f"Failed {file_path.name}: {result.error_message}")
                        
                except Exception as e:
                    # Handle timeout or other execution errors
                    error_result = ProcessingResult(
                        file_path=str(file_path),
                        success=False,
                        error_message=f"Processing timeout or error: {str(e)}",
                        processing_time=self.timeout_seconds
                    )
                    results.append(error_result)
                    progress_tracker.update_progress(error_result)
                    
                    self.logger.error(f"Processing failed for {file_path.name}: {e}")
        
        return results
    
    def _process_single_file_with_timeout(self, file_path: Path) -> ProcessingResult:
        """Process a single file with timeout handling, error recovery, and performance optimization"""
        def _process_file():
            # Create file processor for this file with performance optimization settings
            processor = FileProcessor(
                min_flex_score=self.min_flex_score,
                enable_performance_optimization=self.enable_performance_optimization,
                memory_limit_gb=self.memory_limit_gb / self.max_workers  # Divide memory among workers
            )
            # Process the file
            return processor.process_file(file_path)
        
        # Use error recovery if enabled
        if self.error_recovery:
            result, success = self.error_recovery.process_with_recovery(
                _process_file, str(file_path)
            )
            
            if success:
                return result
            else:
                # Create failed result from error recovery info
                return ProcessingResult(
                    file_path=str(file_path),
                    success=False,
                    error_message="Processing failed after retries - see error log for details",
                    processing_time=0.0
                )
        else:
            # Process without error recovery
            try:
                return _process_file()
            except Exception as e:
                return ProcessingResult(
                    file_path=str(file_path),
                    success=False,
                    error_message=f"Unexpected error: {str(e)}",
                    processing_time=0.0
                )
    
    async def process_files_async(self, file_paths: List[Path]) -> List[ProcessingResult]:
        """
        Process files asynchronously using asyncio
        
        Alternative implementation for async environments
        """
        if not file_paths:
            return []
        
        self.logger.info(f"Starting async batch processing of {len(file_paths)} files")
        
        # Initialize stats and progress tracker
        self.stats = BatchProcessingStats(
            total_files=len(file_paths),
            start_time=datetime.now()
        )
        
        progress_tracker = ProgressTracker(
            total_files=len(file_paths),
            progress_callback=self.progress_callback
        )
        
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(self.max_workers)
        
        # Process files concurrently
        tasks = [
            self._process_file_async(file_path, semaphore, progress_tracker)
            for file_path in file_paths
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = ProcessingResult(
                    file_path=str(file_paths[i]),
                    success=False,
                    error_message=f"Async processing error: {str(result)}"
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        # Finalize stats
        self._finalize_stats(processed_results)
        
        return processed_results
    
    async def _process_file_async(self, file_path: Path, semaphore: asyncio.Semaphore, 
                                 progress_tracker: ProgressTracker) -> ProcessingResult:
        """Process a single file asynchronously"""
        async with semaphore:
            # Run file processing in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            try:
                result = await loop.run_in_executor(
                    None, 
                    self._process_single_file_with_timeout, 
                    file_path
                )
                
                progress_tracker.update_progress(result)
                return result
                
            except Exception as e:
                error_result = ProcessingResult(
                    file_path=str(file_path),
                    success=False,
                    error_message=f"Async processing error: {str(e)}"
                )
                progress_tracker.update_progress(error_result)
                return error_result
    
    def _finalize_stats(self, results: List[ProcessingResult]):
        """Finalize processing statistics"""
        self.stats.end_time = datetime.now()
        self.stats.processing_duration = (self.stats.end_time - self.stats.start_time).total_seconds()
        self.stats.processed_files = len(results)
        self.stats.successful_files = sum(1 for r in results if r.success)
        self.stats.failed_files = sum(1 for r in results if not r.success)
        
        # Calculate totals
        self.stats.total_properties = sum(r.property_count for r in results if r.success)
        self.stats.total_flex_candidates = sum(r.flex_candidate_count for r in results if r.success)
        
        # Calculate average processing time
        processing_times = [r.processing_time for r in results if r.processing_time > 0]
        self.stats.average_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        self.processing_results = results
    
    def get_processing_stats(self) -> BatchProcessingStats:
        """Get current processing statistics"""
        return self.stats
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of processing errors"""
        failed_results = [r for r in self.processing_results if not r.success]
        
        error_types = {}
        for result in failed_results:
            error_msg = result.error_message or "Unknown error"
            # Categorize errors by type
            if "timeout" in error_msg.lower():
                error_type = "timeout"
            elif "file" in error_msg.lower() and ("not found" in error_msg.lower() or "missing" in error_msg.lower()):
                error_type = "file_not_found"
            elif "column" in error_msg.lower():
                error_type = "schema_error"
            elif "corrupted" in error_msg.lower() or "invalid" in error_msg.lower():
                error_type = "file_corruption"
            else:
                error_type = "processing_error"
            
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append({
                'file_path': result.file_path,
                'error_message': error_msg
            })
        
        return {
            'total_errors': len(failed_results),
            'error_types': error_types,
            'error_rate': len(failed_results) / max(1, len(self.processing_results))
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        successful_results = [r for r in self.processing_results if r.success]
        
        if not successful_results:
            return {
                'files_per_minute': 0,
                'properties_per_minute': 0,
                'candidates_per_minute': 0,
                'average_file_size': 0,
                'throughput_metrics': {}
            }
        
        duration_minutes = self.stats.processing_duration / 60 if self.stats.processing_duration > 0 else 1
        
        # Calculate throughput metrics
        files_per_minute = len(successful_results) / duration_minutes
        properties_per_minute = self.stats.total_properties / duration_minutes
        candidates_per_minute = self.stats.total_flex_candidates / duration_minutes
        
        # File size analysis
        file_sizes = []
        for result in successful_results:
            if result.source_file_info and 'file_size_mb' in result.source_file_info:
                file_sizes.append(result.source_file_info['file_size_mb'])
        
        avg_file_size = sum(file_sizes) / len(file_sizes) if file_sizes else 0
        
        return {
            'files_per_minute': round(files_per_minute, 2),
            'properties_per_minute': round(properties_per_minute, 2),
            'candidates_per_minute': round(candidates_per_minute, 2),
            'average_file_size_mb': round(avg_file_size, 2),
            'average_processing_time': round(self.stats.average_processing_time, 2),
            'total_duration': round(self.stats.processing_duration, 2),
            'worker_efficiency': round(self.stats.processing_duration / (self.max_workers * len(successful_results)), 2) if successful_results else 0
        }
    
    def resume_failed_files(self) -> Dict[str, Any]:
        """
        Resume processing of files that failed in the previous batch
        
        Returns:
            Dictionary with resume operation results
        """
        if not self.error_recovery:
            self.logger.warning("Error recovery not enabled - cannot resume failed files")
            return {'error': 'Error recovery not enabled'}
        
        def _process_file_for_resume(file_path: str):
            """Processing function for resume operation"""
            processor = FileProcessor(min_flex_score=self.min_flex_score)
            return processor.process_file(Path(file_path))
        
        self.logger.info("Resuming processing of failed files...")
        resume_results = self.error_recovery.resume_failed_files(_process_file_for_resume)
        
        # Update processing results with recovered files
        if resume_results['successful'] > 0:
            self.logger.info(f"Successfully recovered {resume_results['successful']} files")
            
            # Update stats
            self.stats.successful_files += resume_results['successful']
            self.stats.failed_files -= resume_results['successful']
        
        return resume_results
    
    def generate_error_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive error report with troubleshooting information
        
        Returns:
            Dictionary containing error analysis and recommendations
        """
        if not self.error_recovery:
            # Generate basic error report from processing results
            failed_results = [r for r in self.processing_results if not r.success]
            
            return {
                'summary': {
                    'total_errors': len(failed_results),
                    'error_rate': len(failed_results) / max(1, len(self.processing_results))
                },
                'failed_files': [
                    {
                        'file_path': r.file_path,
                        'error_message': r.error_message,
                        'processing_time': r.processing_time
                    }
                    for r in failed_results
                ],
                'message': 'Error recovery not enabled - limited error information available'
            }
        
        # Generate comprehensive error report using error recovery system
        return self.error_recovery.generate_error_report()
    
    def save_error_log(self) -> Optional[str]:
        """
        Save error log to file
        
        Returns:
            Path to saved error log file, or None if error recovery not enabled
        """
        if not self.error_recovery:
            self.logger.warning("Error recovery not enabled - cannot save error log")
            return None
        
        return self.error_recovery.save_error_log()
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """
        Get error recovery statistics
        
        Returns:
            Dictionary with recovery statistics
        """
        if not self.error_recovery:
            return {'error': 'Error recovery not enabled'}
        
        return self.error_recovery.get_recovery_statistics()
    
    def get_memory_usage_stats(self) -> Dict[str, Any]:
        """
        Get current memory usage statistics
        
        Returns:
            Dictionary with memory usage information
        """
        if not self.performance_optimizer:
            return {'error': 'Performance optimization not enabled'}
        
        from pipeline.performance_optimizer import get_memory_usage
        return get_memory_usage()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.performance_optimizer:
            return {'error': 'Performance optimization not enabled'}
        
        return self.performance_optimizer.get_performance_summary()
    
    def cleanup_performance_resources(self):
        """Clean up performance optimization resources"""
        if self.performance_optimizer:
            self.performance_optimizer.cleanup()
            self.logger.info("Performance optimization resources cleaned up")


# Convenience function for simple batch processing
def process_files_batch(file_paths: List[Path], 
                       max_workers: int = 4,
                       min_flex_score: float = 4.0,
                       progress_callback: Optional[Callable] = None,
                       enable_error_recovery: bool = True,
                       enable_performance_optimization: bool = True,
                       memory_limit_gb: float = 4.0) -> List[ProcessingResult]:
    """
    Simple function to process a batch of files
    
    Args:
        file_paths: List of Excel file paths
        max_workers: Number of concurrent workers
        min_flex_score: Minimum flex score threshold
        progress_callback: Optional progress callback function
        enable_error_recovery: Whether to enable error recovery and retry mechanisms
        enable_performance_optimization: Whether to enable performance optimization
        memory_limit_gb: Memory limit for processing operations
        
    Returns:
        List of ProcessingResult objects
    """
    processor = BatchProcessor(
        max_workers=max_workers,
        min_flex_score=min_flex_score,
        progress_callback=progress_callback,
        enable_error_recovery=enable_error_recovery,
        enable_performance_optimization=enable_performance_optimization,
        memory_limit_gb=memory_limit_gb
    )
    
    try:
        return processor.process_files(file_paths)
    finally:
        # Clean up performance resources
        processor.cleanup_performance_resources()