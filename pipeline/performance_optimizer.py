"""
Performance Optimization and Memory Management for Scalable Multi-File Pipeline
Handles memory-efficient processing, chunked operations, and performance monitoring
"""

import logging
import psutil
import gc
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time
import threading
from contextlib import contextmanager
import warnings


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_memory_gb: float
    available_memory_gb: float
    used_memory_gb: float
    memory_percent: float
    process_memory_gb: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_memory_gb': round(self.total_memory_gb, 2),
            'available_memory_gb': round(self.available_memory_gb, 2),
            'used_memory_gb': round(self.used_memory_gb, 2),
            'memory_percent': round(self.memory_percent, 1),
            'process_memory_gb': round(self.process_memory_gb, 2),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations"""
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    memory_before: Optional[MemoryStats] = None
    memory_after: Optional[MemoryStats] = None
    memory_peak: Optional[MemoryStats] = None
    records_processed: int = 0
    throughput_records_per_second: float = 0.0
    
    def finalize(self, records_processed: int = 0):
        """Finalize metrics calculation"""
        if self.end_time is None:
            self.end_time = datetime.now()
        
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.records_processed = records_processed
        
        if self.duration_seconds > 0 and records_processed > 0:
            self.throughput_records_per_second = records_processed / self.duration_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'operation_name': self.operation_name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': round(self.duration_seconds, 3),
            'records_processed': self.records_processed,
            'throughput_records_per_second': round(self.throughput_records_per_second, 2),
            'memory_before': self.memory_before.to_dict() if self.memory_before else None,
            'memory_after': self.memory_after.to_dict() if self.memory_after else None,
            'memory_peak': self.memory_peak.to_dict() if self.memory_peak else None
        }


class MemoryMonitor:
    """Monitor memory usage and provide warnings"""
    
    def __init__(self, warning_threshold_percent: float = 80.0, critical_threshold_percent: float = 90.0):
        self.warning_threshold = warning_threshold_percent
        self.critical_threshold = critical_threshold_percent
        self.logger = logging.getLogger(__name__)
        self._monitoring = False
        self._monitor_thread = None
        self._peak_memory = None
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        # System memory
        memory = psutil.virtual_memory()
        
        # Process memory
        process = psutil.Process()
        process_memory_bytes = process.memory_info().rss
        
        return MemoryStats(
            total_memory_gb=memory.total / (1024**3),
            available_memory_gb=memory.available / (1024**3),
            used_memory_gb=memory.used / (1024**3),
            memory_percent=memory.percent,
            process_memory_gb=process_memory_bytes / (1024**3)
        )
    
    def check_memory_usage(self) -> Tuple[bool, str]:
        """
        Check current memory usage against thresholds
        
        Returns:
            Tuple of (is_safe, message)
        """
        stats = self.get_memory_stats()
        
        if stats.memory_percent >= self.critical_threshold:
            return False, f"CRITICAL: Memory usage at {stats.memory_percent:.1f}% (>{self.critical_threshold}%)"
        elif stats.memory_percent >= self.warning_threshold:
            return True, f"WARNING: Memory usage at {stats.memory_percent:.1f}% (>{self.warning_threshold}%)"
        else:
            return True, f"Memory usage normal: {stats.memory_percent:.1f}%"
    
    def start_monitoring(self, interval_seconds: float = 5.0):
        """Start continuous memory monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info(f"Memory monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        self.logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self, interval_seconds: float):
        """Memory monitoring loop"""
        while self._monitoring:
            try:
                stats = self.get_memory_stats()
                
                # Track peak memory
                if self._peak_memory is None or stats.memory_percent > self._peak_memory.memory_percent:
                    self._peak_memory = stats
                
                # Check thresholds
                is_safe, message = self.check_memory_usage()
                
                if not is_safe:
                    self.logger.error(message)
                elif "WARNING" in message:
                    self.logger.warning(message)
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
                time.sleep(interval_seconds)
    
    def get_peak_memory(self) -> Optional[MemoryStats]:
        """Get peak memory usage since monitoring started"""
        return self._peak_memory
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager to monitor memory usage during an operation"""
        start_stats = self.get_memory_stats()
        start_time = datetime.now()
        
        self.logger.debug(f"Starting operation '{operation_name}' - Memory: {start_stats.memory_percent:.1f}%")
        
        try:
            yield start_stats
        finally:
            end_stats = self.get_memory_stats()
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            memory_change = end_stats.process_memory_gb - start_stats.process_memory_gb
            
            self.logger.debug(f"Completed operation '{operation_name}' in {duration:.2f}s - "
                            f"Memory: {end_stats.memory_percent:.1f}% "
                            f"(Process: {memory_change:+.2f}GB)")


class ChunkedDataProcessor:
    """Process large DataFrames in chunks to manage memory usage"""
    
    def __init__(self, chunk_size: int = 10000, memory_monitor: Optional[MemoryMonitor] = None):
        self.chunk_size = chunk_size
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.logger = logging.getLogger(__name__)
    
    def process_dataframe_chunked(self, 
                                 df: pd.DataFrame, 
                                 processing_func: Callable[[pd.DataFrame], pd.DataFrame],
                                 operation_name: str = "chunked_processing") -> pd.DataFrame:
        """
        Process a large DataFrame in chunks
        
        Args:
            df: DataFrame to process
            processing_func: Function to apply to each chunk
            operation_name: Name for logging/monitoring
            
        Returns:
            Processed DataFrame
        """
        if len(df) <= self.chunk_size:
            # Process normally if small enough
            with self.memory_monitor.monitor_operation(f"{operation_name}_small"):
                return processing_func(df)
        
        self.logger.info(f"Processing {len(df)} records in chunks of {self.chunk_size}")
        
        processed_chunks = []
        total_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size
        
        for i, chunk_start in enumerate(range(0, len(df), self.chunk_size)):
            chunk_end = min(chunk_start + self.chunk_size, len(df))
            chunk = df.iloc[chunk_start:chunk_end].copy()
            
            chunk_name = f"{operation_name}_chunk_{i+1}_{total_chunks}"
            
            with self.memory_monitor.monitor_operation(chunk_name):
                processed_chunk = processing_func(chunk)
                processed_chunks.append(processed_chunk)
            
            # Force garbage collection after each chunk
            del chunk
            gc.collect()
            
            # Check memory usage
            is_safe, message = self.memory_monitor.check_memory_usage()
            if not is_safe:
                self.logger.error(f"Memory critical during chunk processing: {message}")
                # Could implement emergency measures here
            
            if i % 10 == 0:  # Log progress every 10 chunks
                self.logger.debug(f"Processed chunk {i+1}/{total_chunks}")
        
        # Combine results
        with self.memory_monitor.monitor_operation(f"{operation_name}_combine"):
            result = pd.concat(processed_chunks, ignore_index=True)
        
        # Cleanup
        del processed_chunks
        gc.collect()
        
        self.logger.info(f"Chunked processing completed: {len(result)} records")
        return result
    
    def read_excel_chunked(self, file_path: Path, chunk_size: Optional[int] = None) -> Iterator[pd.DataFrame]:
        """
        Read Excel file in chunks (if supported by the file format)
        
        Args:
            file_path: Path to Excel file
            chunk_size: Chunk size (uses instance default if None)
            
        Yields:
            DataFrame chunks
        """
        chunk_size = chunk_size or self.chunk_size
        
        try:
            # For Excel files, we need to read the entire file first
            # Then chunk it in memory (Excel doesn't support native chunked reading)
            with self.memory_monitor.monitor_operation(f"read_excel_{file_path.name}"):
                df = pd.read_excel(file_path)
            
            if len(df) <= chunk_size:
                yield df
                return
            
            # Yield chunks
            for start in range(0, len(df), chunk_size):
                end = min(start + chunk_size, len(df))
                chunk = df.iloc[start:end].copy()
                yield chunk
                
                # Force garbage collection
                gc.collect()
        
        except Exception as e:
            self.logger.error(f"Failed to read Excel file in chunks: {e}")
            raise


class PerformanceOptimizer:
    """Main performance optimization and memory management system"""
    
    def __init__(self, 
                 memory_limit_gb: float = 4.0,
                 chunk_size: int = 10000,
                 enable_monitoring: bool = True,
                 optimization_level: str = "balanced"):
        """
        Initialize performance optimizer
        
        Args:
            memory_limit_gb: Memory limit for operations
            chunk_size: Default chunk size for large operations
            enable_monitoring: Whether to enable memory monitoring
            optimization_level: Optimization level ('fast', 'balanced', 'memory_efficient')
        """
        self.memory_limit_gb = memory_limit_gb
        self.chunk_size = chunk_size
        self.optimization_level = optimization_level
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.memory_monitor = MemoryMonitor(
            warning_threshold_percent=70.0,
            critical_threshold_percent=85.0
        )
        
        self.chunked_processor = ChunkedDataProcessor(
            chunk_size=chunk_size,
            memory_monitor=self.memory_monitor
        )
        
        # Performance metrics tracking
        self.metrics: List[PerformanceMetrics] = []
        
        # Start monitoring if enabled
        if enable_monitoring:
            self.memory_monitor.start_monitoring()
        
        # Apply optimization settings
        self._apply_optimization_settings()
        
        self.logger.info(f"PerformanceOptimizer initialized - "
                        f"Memory limit: {memory_limit_gb}GB, "
                        f"Chunk size: {chunk_size}, "
                        f"Level: {optimization_level}")
    
    def _apply_optimization_settings(self):
        """Apply optimization settings based on level"""
        if self.optimization_level == "fast":
            # Prioritize speed over memory
            self.chunk_size = max(self.chunk_size, 50000)
            # Disable some pandas warnings for speed
            warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
            
        elif self.optimization_level == "memory_efficient":
            # Prioritize memory efficiency
            self.chunk_size = min(self.chunk_size, 5000)
            # Enable more aggressive garbage collection
            gc.set_threshold(700, 10, 10)  # More frequent GC
            
        else:  # balanced
            # Default balanced settings
            pass
    
    @contextmanager
    def track_performance(self, operation_name: str):
        """Context manager to track performance metrics"""
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=datetime.now(),
            memory_before=self.memory_monitor.get_memory_stats()
        )
        
        try:
            yield metrics
        finally:
            metrics.memory_after = self.memory_monitor.get_memory_stats()
            metrics.memory_peak = self.memory_monitor.get_peak_memory()
            metrics.finalize()
            
            self.metrics.append(metrics)
            
            # Log performance summary
            self.logger.info(f"Performance: {operation_name} completed in {metrics.duration_seconds:.2f}s "
                           f"({metrics.throughput_records_per_second:.0f} records/sec)")
    
    def optimize_dataframe_operations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply DataFrame optimizations to reduce memory usage
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Optimized DataFrame
        """
        with self.track_performance("dataframe_optimization") as metrics:
            original_memory = df.memory_usage(deep=True).sum() / (1024**2)  # MB
            
            # Optimize data types
            optimized_df = self._optimize_dtypes(df)
            
            # Remove unnecessary columns if any
            optimized_df = self._remove_empty_columns(optimized_df)
            
            # Optimize string columns
            optimized_df = self._optimize_string_columns(optimized_df)
            
            final_memory = optimized_df.memory_usage(deep=True).sum() / (1024**2)  # MB
            memory_saved = original_memory - final_memory
            
            metrics.records_processed = len(optimized_df)
            
            self.logger.info(f"DataFrame optimization: {memory_saved:.1f}MB saved "
                           f"({memory_saved/original_memory*100:.1f}% reduction)")
            
            return optimized_df
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types"""
        df_optimized = df.copy()
        
        for col in df_optimized.columns:
            col_type = df_optimized[col].dtype
            
            if col_type == 'object':
                # Try to convert to numeric if possible
                try:
                    df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
                except (ValueError, TypeError):
                    try:
                        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
                    except (ValueError, TypeError):
                        # Keep as object/string
                        pass
            
            elif col_type in ['int64', 'int32']:
                # Downcast integers
                df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
            
            elif col_type in ['float64', 'float32']:
                # Downcast floats
                df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        return df_optimized
    
    def _remove_empty_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns that are entirely empty or null"""
        empty_cols = df.columns[df.isnull().all()].tolist()
        
        if empty_cols:
            self.logger.debug(f"Removing {len(empty_cols)} empty columns: {empty_cols}")
            return df.drop(columns=empty_cols)
        
        return df
    
    def _optimize_string_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize string columns using categorical data type where beneficial"""
        df_optimized = df.copy()
        
        for col in df_optimized.select_dtypes(include=['object']).columns:
            # Convert to categorical if it saves memory
            unique_count = df_optimized[col].nunique()
            total_count = len(df_optimized[col])
            
            # Use categorical if less than 50% unique values
            if unique_count / total_count < 0.5 and unique_count > 1:
                df_optimized[col] = df_optimized[col].astype('category')
                self.logger.debug(f"Converted column '{col}' to categorical "
                                f"({unique_count} unique values)")
        
        return df_optimized
    
    def process_large_file(self, 
                          file_path: Path, 
                          processing_func: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
        """
        Process a large Excel file with memory optimization
        
        Args:
            file_path: Path to Excel file
            processing_func: Function to apply to the data
            
        Returns:
            Processed DataFrame
        """
        with self.track_performance(f"process_large_file_{file_path.name}") as metrics:
            # Check file size first
            file_size_mb = file_path.stat().st_size / (1024**2)
            
            if file_size_mb > 100:  # Large file threshold
                self.logger.info(f"Processing large file ({file_size_mb:.1f}MB) with chunking")
                
                # Process in chunks
                processed_chunks = []
                
                for chunk in self.chunked_processor.read_excel_chunked(file_path):
                    # Optimize chunk
                    optimized_chunk = self.optimize_dataframe_operations(chunk)
                    
                    # Process chunk
                    processed_chunk = processing_func(optimized_chunk)
                    processed_chunks.append(processed_chunk)
                    
                    # Force cleanup
                    del chunk, optimized_chunk
                    gc.collect()
                
                # Combine results
                result = pd.concat(processed_chunks, ignore_index=True)
                del processed_chunks
                gc.collect()
                
            else:
                # Process normally for smaller files
                df = pd.read_excel(file_path)
                optimized_df = self.optimize_dataframe_operations(df)
                result = processing_func(optimized_df)
                
                del df, optimized_df
                gc.collect()
            
            metrics.records_processed = len(result)
            return result
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics"""
        if not self.metrics:
            return {'message': 'No performance metrics available'}
        
        total_duration = sum(m.duration_seconds for m in self.metrics)
        total_records = sum(m.records_processed for m in self.metrics)
        
        # Calculate averages
        avg_throughput = sum(m.throughput_records_per_second for m in self.metrics if m.throughput_records_per_second > 0)
        avg_throughput = avg_throughput / len([m for m in self.metrics if m.throughput_records_per_second > 0])
        
        # Memory statistics
        memory_stats = []
        for m in self.metrics:
            if m.memory_before and m.memory_after:
                memory_change = m.memory_after.process_memory_gb - m.memory_before.process_memory_gb
                memory_stats.append(memory_change)
        
        return {
            'total_operations': len(self.metrics),
            'total_duration_seconds': round(total_duration, 2),
            'total_records_processed': total_records,
            'average_throughput_records_per_second': round(avg_throughput, 2),
            'memory_usage': {
                'average_change_gb': round(sum(memory_stats) / len(memory_stats), 3) if memory_stats else 0,
                'max_change_gb': round(max(memory_stats), 3) if memory_stats else 0,
                'min_change_gb': round(min(memory_stats), 3) if memory_stats else 0
            },
            'peak_memory': self.memory_monitor.get_peak_memory().to_dict() if self.memory_monitor.get_peak_memory() else None,
            'optimization_level': self.optimization_level,
            'chunk_size': self.chunk_size
        }
    
    def cleanup(self):
        """Cleanup resources and stop monitoring"""
        self.memory_monitor.stop_monitoring()
        
        # Force garbage collection
        gc.collect()
        
        self.logger.info("PerformanceOptimizer cleanup completed")


# Convenience functions
def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quick function to optimize DataFrame memory usage
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        Memory-optimized DataFrame
    """
    optimizer = PerformanceOptimizer(enable_monitoring=False)
    return optimizer.optimize_dataframe_operations(df)


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics
    
    Returns:
        Dictionary with memory statistics
    """
    monitor = MemoryMonitor()
    stats = monitor.get_memory_stats()
    return stats.to_dict()