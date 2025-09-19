"""
Batch Processing Utility for Flex Property Classification
Handles multiple Excel files with progress tracking and error recovery
"""

import logging
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Union, Callable
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
import threading
from collections import defaultdict

from utils.logger import setup_logging
from utils.flex_config_manager import FlexConfigManager
from processors.advanced_flex_classifier import AdvancedFlexClassifier


@dataclass
class BatchResult:
    """Result from processing a single file in batch"""
    file_path: str
    success: bool
    candidates_found: int
    processing_time: float
    error_message: Optional[str] = None
    properties_processed: int = 0


@dataclass
class BatchSummary:
    """Summary of entire batch processing operation"""
    total_files: int
    successful_files: int
    failed_files: int
    total_properties: int
    total_candidates: int
    total_processing_time: float
    average_processing_time: float
    files_per_second: float
    properties_per_second: float
    error_summary: Dict[str, int]


class BatchProgressTracker:
    """Thread-safe progress tracker for batch operations"""
    
    def __init__(self, total_files: int):
        self.total_files = total_files
        self.completed_files = 0
        self.current_file = ""
        self.start_time = time.time()
        self._lock = threading.Lock()
        self.callbacks: List[Callable] = []
    
    def add_callback(self, callback: Callable[[int, int, str, float], None]):
        """Add progress callback function"""
        self.callbacks.append(callback)
    
    def update_progress(self, completed: int, current_file: str = ""):
        """Update progress and notify callbacks"""
        with self._lock:
            self.completed_files = completed
            self.current_file = current_file
            elapsed_time = time.time() - self.start_time
            
            for callback in self.callbacks:
                try:
                    callback(self.completed_files, self.total_files, self.current_file, elapsed_time)
                except Exception as e:
                    logging.getLogger(__name__).error(f"Error in progress callback: {e}")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information"""
        with self._lock:
            elapsed_time = time.time() - self.start_time
            return {
                'completed_files': self.completed_files,
                'total_files': self.total_files,
                'current_file': self.current_file,
                'elapsed_time': elapsed_time,
                'progress_percentage': (self.completed_files / self.total_files * 100) if self.total_files > 0 else 0
            }


class FlexBatchProcessor:
    """
    Batch processor for multiple Excel files containing property data
    
    Features:
    - Parallel processing with configurable worker count
    - Progress tracking and callbacks
    - Error recovery and detailed error reporting
    - Result aggregation and deduplication
    - Performance monitoring and optimization
    """
    
    def __init__(self, 
                 config_path: Optional[Path] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize batch processor
        
        Args:
            config_path: Optional path to configuration file
            logger: Optional logger instance
        """
        # Load configuration
        config_manager = FlexConfigManager(config_path)
        self.config = config_manager.load_config()
        
        # Set up logging
        if logger is None:
            self.logger = setup_logging(name='batch_processor', level='INFO')
        else:
            self.logger = logger
        
        # Initialize tracking
        self.progress_tracker: Optional[BatchProgressTracker] = None
        self.batch_results: List[BatchResult] = []
        self.aggregated_results: Optional[pd.DataFrame] = None
        
        self.logger.info("FlexBatchProcessor initialized")
    
    def discover_excel_files(self, 
                           input_paths: List[Union[str, Path]],
                           recursive: bool = True,
                           file_patterns: List[str] = None) -> List[Path]:
        """
        Discover Excel files in given paths
        
        Args:
            input_paths: List of file or directory paths
            recursive: Whether to search directories recursively
            file_patterns: Optional list of file patterns to match
            
        Returns:
            List of discovered Excel file paths
        """
        if file_patterns is None:
            file_patterns = ['*.xlsx', '*.xls', '*.xlsm']
        
        discovered_files = []
        
        for input_path in input_paths:
            path = Path(input_path)
            
            if path.is_file():
                # Single file
                if any(path.match(pattern) for pattern in file_patterns):
                    discovered_files.append(path)
                else:
                    self.logger.warning(f"File {path} does not match Excel patterns")
            
            elif path.is_dir():
                # Directory - search for Excel files
                if recursive:
                    for pattern in file_patterns:
                        discovered_files.extend(path.rglob(pattern))
                else:
                    for pattern in file_patterns:
                        discovered_files.extend(path.glob(pattern))
            
            else:
                self.logger.warning(f"Path {path} does not exist")
        
        # Remove duplicates and sort
        unique_files = sorted(list(set(discovered_files)))
        
        self.logger.info(f"Discovered {len(unique_files)} Excel files")
        return unique_files
    
    def process_files(self, 
                     file_paths: List[Union[str, Path]],
                     output_dir: Optional[Path] = None,
                     max_workers: Optional[int] = None,
                     progress_callback: Optional[Callable] = None) -> BatchSummary:
        """
        Process multiple Excel files in batch
        
        Args:
            file_paths: List of Excel file paths to process
            output_dir: Optional output directory for results
            max_workers: Optional number of worker threads
            progress_callback: Optional progress callback function
            
        Returns:
            BatchSummary with processing results
        """
        try:
            start_time = time.time()
            
            # Convert to Path objects
            file_paths = [Path(p) for p in file_paths]
            
            # Set up progress tracking
            self.progress_tracker = BatchProgressTracker(len(file_paths))
            if progress_callback:
                self.progress_tracker.add_callback(progress_callback)
            
            # Determine worker count
            if max_workers is None:
                max_workers = min(
                    self.config.advanced_settings.max_workers,
                    len(file_paths),
                    8  # Reasonable maximum
                )
            
            self.logger.info(f"Processing {len(file_paths)} files with {max_workers} workers")
            
            # Initialize results storage
            self.batch_results = []
            
            if self.config.advanced_settings.parallel_processing and max_workers > 1:
                # Parallel processing
                self._process_files_parallel(file_paths, max_workers)
            else:
                # Sequential processing
                self._process_files_sequential(file_paths)
            
            # Calculate summary
            end_time = time.time()
            total_time = end_time - start_time
            
            summary = self._create_batch_summary(total_time)
            
            # Aggregate results
            self._aggregate_results()
            
            # Export results if output directory specified
            if output_dir and self.aggregated_results is not None:
                self._export_batch_results(output_dir, summary)
            
            self.logger.info(f"Batch processing complete: {summary.successful_files}/{summary.total_files} files processed successfully")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")
            raise
    
    def _process_files_parallel(self, file_paths: List[Path], max_workers: int):
        """Process files using parallel execution"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_single_file, file_path): file_path
                for file_path in file_paths
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                
                try:
                    result = future.result()
                    self.batch_results.append(result)
                except Exception as e:
                    # Create error result
                    error_result = BatchResult(
                        file_path=str(file_path),
                        success=False,
                        candidates_found=0,
                        processing_time=0.0,
                        error_message=str(e)
                    )
                    self.batch_results.append(error_result)
                    self.logger.error(f"Error processing {file_path}: {e}")
                
                completed += 1
                self.progress_tracker.update_progress(completed, file_path.name)
    
    def _process_files_sequential(self, file_paths: List[Path]):
        """Process files sequentially"""
        for i, file_path in enumerate(file_paths):
            try:
                result = self._process_single_file(file_path)
                self.batch_results.append(result)
            except Exception as e:
                # Create error result
                error_result = BatchResult(
                    file_path=str(file_path),
                    success=False,
                    candidates_found=0,
                    processing_time=0.0,
                    error_message=str(e)
                )
                self.batch_results.append(error_result)
                self.logger.error(f"Error processing {file_path}: {e}")
            
            self.progress_tracker.update_progress(i + 1, file_path.name)
    
    def _process_single_file(self, file_path: Path) -> BatchResult:
        """Process a single Excel file"""
        start_time = time.time()
        
        try:
            self.logger.debug(f"Processing file: {file_path}")
            
            # Load data
            data = pd.read_excel(file_path)
            properties_processed = len(data)
            
            # Create classifier
            classifier = AdvancedFlexClassifier(data, self.config, self.logger)
            
            # Process
            candidates = classifier.classify_flex_properties_batch()
            
            # Add source file information
            if len(candidates) > 0:
                candidates['source_file'] = str(file_path)
                candidates['file_name'] = file_path.name
            
            processing_time = time.time() - start_time
            
            result = BatchResult(
                file_path=str(file_path),
                success=True,
                candidates_found=len(candidates),
                processing_time=processing_time,
                properties_processed=properties_processed
            )
            
            # Store candidates for aggregation
            if len(candidates) > 0:
                result.candidates_data = candidates
            
            self.logger.debug(f"Processed {file_path}: {len(candidates)} candidates in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error processing {file_path}: {e}")
            
            return BatchResult(
                file_path=str(file_path),
                success=False,
                candidates_found=0,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _aggregate_results(self):
        """Aggregate results from all processed files"""
        try:
            all_candidates = []
            
            for result in self.batch_results:
                if result.success and hasattr(result, 'candidates_data'):
                    all_candidates.append(result.candidates_data)
            
            if all_candidates:
                # Combine all DataFrames
                self.aggregated_results = pd.concat(all_candidates, ignore_index=True)
                
                # Perform deduplication if enabled
                if self.config.advanced_settings.enable_batch_processing:
                    self._deduplicate_results()
                
                self.logger.info(f"Aggregated {len(self.aggregated_results)} total candidates from {len(all_candidates)} files")
            else:
                self.aggregated_results = pd.DataFrame()
                self.logger.warning("No candidates found in any processed files")
                
        except Exception as e:
            self.logger.error(f"Error aggregating results: {e}")
            self.aggregated_results = pd.DataFrame()
    
    def _deduplicate_results(self):
        """Remove duplicate properties based on address, city, state"""
        try:
            if self.aggregated_results is None or len(self.aggregated_results) == 0:
                return
            
            initial_count = len(self.aggregated_results)
            
            # Find columns for deduplication
            address_col = None
            city_col = None
            state_col = None
            
            for col in self.aggregated_results.columns:
                col_lower = col.lower()
                if 'address' in col_lower and address_col is None:
                    address_col = col
                elif 'city' in col_lower and city_col is None:
                    city_col = col
                elif 'state' in col_lower and state_col is None:
                    state_col = col
            
            # Create deduplication key
            dedup_columns = []
            if address_col:
                dedup_columns.append(address_col)
            if city_col:
                dedup_columns.append(city_col)
            if state_col:
                dedup_columns.append(state_col)
            
            if dedup_columns:
                # Sort by flex_score (if available) to keep highest scoring duplicates
                if 'flex_score' in self.aggregated_results.columns:
                    self.aggregated_results = self.aggregated_results.sort_values('flex_score', ascending=False)
                
                # Remove duplicates
                self.aggregated_results = self.aggregated_results.drop_duplicates(
                    subset=dedup_columns, 
                    keep='first'
                ).reset_index(drop=True)
                
                final_count = len(self.aggregated_results)
                duplicates_removed = initial_count - final_count
                
                if duplicates_removed > 0:
                    self.logger.info(f"Removed {duplicates_removed} duplicate properties")
            else:
                self.logger.warning("No suitable columns found for deduplication")
                
        except Exception as e:
            self.logger.error(f"Error in deduplication: {e}")
    
    def _create_batch_summary(self, total_time: float) -> BatchSummary:
        """Create summary of batch processing results"""
        successful_files = sum(1 for r in self.batch_results if r.success)
        failed_files = len(self.batch_results) - successful_files
        total_properties = sum(r.properties_processed for r in self.batch_results)
        total_candidates = sum(r.candidates_found for r in self.batch_results if r.success)
        
        # Error summary
        error_summary = defaultdict(int)
        for result in self.batch_results:
            if not result.success and result.error_message:
                # Categorize errors
                error_msg = result.error_message.lower()
                if 'file not found' in error_msg or 'no such file' in error_msg:
                    error_summary['file_not_found'] += 1
                elif 'permission' in error_msg:
                    error_summary['permission_denied'] += 1
                elif 'excel' in error_msg or 'xlsx' in error_msg:
                    error_summary['excel_format_error'] += 1
                elif 'memory' in error_msg:
                    error_summary['memory_error'] += 1
                else:
                    error_summary['other_error'] += 1
        
        return BatchSummary(
            total_files=len(self.batch_results),
            successful_files=successful_files,
            failed_files=failed_files,
            total_properties=total_properties,
            total_candidates=total_candidates,
            total_processing_time=total_time,
            average_processing_time=total_time / len(self.batch_results) if self.batch_results else 0,
            files_per_second=len(self.batch_results) / total_time if total_time > 0 else 0,
            properties_per_second=total_properties / total_time if total_time > 0 else 0,
            error_summary=dict(error_summary)
        )
    
    def _export_batch_results(self, output_dir: Path, summary: BatchSummary):
        """Export batch processing results"""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            
            # Export aggregated candidates
            if self.aggregated_results is not None and len(self.aggregated_results) > 0:
                candidates_file = output_dir / f"batch_flex_candidates_{timestamp}.xlsx"
                self.aggregated_results.to_excel(candidates_file, index=False, engine='openpyxl')
                self.logger.info(f"Exported aggregated candidates to {candidates_file}")
            
            # Export batch summary
            summary_file = output_dir / f"batch_summary_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(asdict(summary), f, indent=2, default=str)
            
            # Export detailed results
            detailed_results = []
            for result in self.batch_results:
                result_dict = asdict(result)
                # Remove candidates_data to avoid serialization issues
                if 'candidates_data' in result_dict:
                    del result_dict['candidates_data']
                detailed_results.append(result_dict)
            
            detailed_file = output_dir / f"batch_detailed_results_{timestamp}.json"
            with open(detailed_file, 'w') as f:
                json.dump(detailed_results, f, indent=2, default=str)
            
            self.logger.info(f"Exported batch results to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error exporting batch results: {e}")
    
    def get_failed_files(self) -> List[BatchResult]:
        """Get list of files that failed processing"""
        return [result for result in self.batch_results if not result.success]
    
    def retry_failed_files(self, 
                          max_retries: int = 3,
                          progress_callback: Optional[Callable] = None) -> BatchSummary:
        """
        Retry processing failed files
        
        Args:
            max_retries: Maximum number of retry attempts
            progress_callback: Optional progress callback
            
        Returns:
            BatchSummary for retry operation
        """
        failed_files = self.get_failed_files()
        
        if not failed_files:
            self.logger.info("No failed files to retry")
            return BatchSummary(0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, {})
        
        self.logger.info(f"Retrying {len(failed_files)} failed files (max {max_retries} attempts)")
        
        retry_paths = [Path(result.file_path) for result in failed_files]
        
        # Remove failed results from batch_results
        self.batch_results = [result for result in self.batch_results if result.success]
        
        # Process retry files
        return self.process_files(
            retry_paths,
            max_workers=1,  # Use single worker for retries to avoid resource conflicts
            progress_callback=progress_callback
        )


def create_batch_processor_from_config(config_path: Optional[Path] = None) -> FlexBatchProcessor:
    """
    Create FlexBatchProcessor from configuration file
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configured FlexBatchProcessor instance
    """
    return FlexBatchProcessor(config_path)


if __name__ == '__main__':
    # Example usage
    def progress_callback(completed, total, current_file, elapsed_time):
        percentage = (completed / total * 100) if total > 0 else 0
        print(f"Progress: {completed}/{total} ({percentage:.1f}%) - {current_file} - {elapsed_time:.1f}s")
    
    # Create batch processor
    processor = create_batch_processor_from_config()
    
    # Discover files (example - would need actual Excel files)
    # files = processor.discover_excel_files(['data/input'], recursive=True)
    
    # Process files
    # summary = processor.process_files(files, Path('data/output'), progress_callback=progress_callback)
    
    print("Batch processor example completed")