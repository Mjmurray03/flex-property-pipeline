"""
Performance optimizations for data type operations.
Implements caching, lazy conversion, and batch processing to minimize performance impact.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Set
import hashlib
import time
from functools import lru_cache, wraps
from datetime import datetime, timedelta
import pickle
import os
from utils.logger import setup_logging


class DataTypeCache:
    """Cache for data type conversion results and validation outcomes"""
    
    def __init__(self, max_cache_size: int = 1000, cache_ttl_minutes: int = 60):
        self.max_cache_size = max_cache_size
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self.conversion_cache = {}
        self.validation_cache = {}
        self.column_analysis_cache = {}
        self.logger = setup_logging('data_type_cache')
    
    def _generate_cache_key(self, data: Union[pd.Series, pd.DataFrame], operation: str, **kwargs) -> str:
        """Generate a cache key for data and operation"""
        try:
            # Create a hash based on data characteristics and operation
            if isinstance(data, pd.Series):
                # For series, use dtype, length, and sample of values
                key_components = [
                    f"op:{operation}",
                    f"dtype:{str(data.dtype)}",
                    f"len:{str(len(data))}",
                    f"nulls:{str(data.isna().sum())}"
                ]
                
                # Add sample hash if data is not empty
                if len(data) > 0:
                    try:
                        sample_data = data.dropna().head(10).astype(str).tolist()
                        if sample_data:
                            key_components.append(f"sample:{str(hash(tuple(sample_data)))}")
                    except:
                        key_components.append("sample:unavailable")
            else:
                # For dataframes, use shape, dtypes, and column names
                key_components = [
                    f"op:{operation}",
                    f"shape:{str(data.shape)}",
                    f"dtypes:{str(list(data.dtypes.astype(str)))}",
                    f"cols:{str(list(data.columns))}"
                ]
            
            # Add kwargs to key (avoid parameter conflicts)
            for k, v in sorted(kwargs.items()):
                key_components.append(f"param_{k}:{str(v)}")
            
            # Create hash
            key_string = "|".join(key_components)
            return hashlib.md5(key_string.encode()).hexdigest()
            
        except Exception as e:
            self.logger.warning(f"Failed to generate cache key: {str(e)}")
            return f"fallback_{operation}_{int(time.time() * 1000000)}"
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        if 'timestamp' not in cache_entry:
            return False
        
        entry_time = cache_entry['timestamp']
        return datetime.now() - entry_time < self.cache_ttl
    
    def _cleanup_cache(self, cache_dict: Dict) -> None:
        """Remove expired entries and enforce size limits"""
        # Remove expired entries
        expired_keys = [
            key for key, entry in cache_dict.items()
            if not self._is_cache_valid(entry)
        ]
        for key in expired_keys:
            del cache_dict[key]
        
        # Enforce size limit (remove oldest entries)
        if len(cache_dict) > self.max_cache_size:
            # Sort by timestamp and remove oldest
            sorted_items = sorted(
                cache_dict.items(),
                key=lambda x: x[1].get('timestamp', datetime.min)
            )
            
            items_to_remove = len(cache_dict) - self.max_cache_size
            for key, _ in sorted_items[:items_to_remove]:
                del cache_dict[key]
    
    def get_conversion_result(self, series: pd.Series, column_name: str, **kwargs) -> Optional[Tuple[pd.Series, Dict]]:
        """Get cached conversion result"""
        # Merge column_name into kwargs to avoid parameter conflict
        cache_kwargs = {'column_name': column_name, **kwargs}
        cache_key = self._generate_cache_key(series, 'conversion', **cache_kwargs)
        
        if cache_key in self.conversion_cache:
            entry = self.conversion_cache[cache_key]
            if self._is_cache_valid(entry):
                self.logger.debug(f"Cache hit for conversion: {column_name}")
                return entry['result']
        
        return None
    
    def cache_conversion_result(self, series: pd.Series, column_name: str, result: Tuple[pd.Series, Dict], **kwargs) -> None:
        """Cache conversion result"""
        # Merge column_name into kwargs to avoid parameter conflict
        cache_kwargs = {'column_name': column_name, **kwargs}
        cache_key = self._generate_cache_key(series, 'conversion', **cache_kwargs)
        
        self.conversion_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now(),
            'column_name': column_name
        }
        
        # Cleanup if needed
        if len(self.conversion_cache) > self.max_cache_size * 1.2:
            self._cleanup_cache(self.conversion_cache)
        
        self.logger.debug(f"Cached conversion result for: {column_name}")
    
    def get_validation_result(self, series: pd.Series, column_name: str, **kwargs) -> Optional[bool]:
        """Get cached validation result"""
        # Merge column_name into kwargs to avoid parameter conflict
        cache_kwargs = {'column_name': column_name, **kwargs}
        cache_key = self._generate_cache_key(series, 'validation', **cache_kwargs)
        
        if cache_key in self.validation_cache:
            entry = self.validation_cache[cache_key]
            if self._is_cache_valid(entry):
                self.logger.debug(f"Cache hit for validation: {column_name}")
                return entry['result']
        
        return None
    
    def cache_validation_result(self, series: pd.Series, column_name: str, result: bool, **kwargs) -> None:
        """Cache validation result"""
        # Merge column_name into kwargs to avoid parameter conflict
        cache_kwargs = {'column_name': column_name, **kwargs}
        cache_key = self._generate_cache_key(series, 'validation', **cache_kwargs)
        
        self.validation_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now(),
            'column_name': column_name
        }
        
        # Cleanup if needed
        if len(self.validation_cache) > self.max_cache_size * 1.2:
            self._cleanup_cache(self.validation_cache)
        
        self.logger.debug(f"Cached validation result for: {column_name}")
    
    def get_column_analysis(self, series: pd.Series, column_name: str) -> Optional[Dict]:
        """Get cached column analysis"""
        cache_key = self._generate_cache_key(series, 'analysis', column_name=column_name)
        
        if cache_key in self.column_analysis_cache:
            entry = self.column_analysis_cache[cache_key]
            if self._is_cache_valid(entry):
                self.logger.debug(f"Cache hit for analysis: {column_name}")
                return entry['result']
        
        return None
    
    def cache_column_analysis(self, series: pd.Series, column_name: str, result: Dict) -> None:
        """Cache column analysis result"""
        cache_key = self._generate_cache_key(series, 'analysis', column_name=column_name)
        
        self.column_analysis_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now(),
            'column_name': column_name
        }
        
        # Cleanup if needed
        if len(self.column_analysis_cache) > self.max_cache_size * 1.2:
            self._cleanup_cache(self.column_analysis_cache)
        
        self.logger.debug(f"Cached analysis result for: {column_name}")
    
    def clear_cache(self) -> None:
        """Clear all caches"""
        cleared_count = (
            len(self.conversion_cache) + 
            len(self.validation_cache) + 
            len(self.column_analysis_cache)
        )
        
        self.conversion_cache.clear()
        self.validation_cache.clear()
        self.column_analysis_cache.clear()
        
        self.logger.info(f"Cleared {cleared_count} cache entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'conversion_cache_size': len(self.conversion_cache),
            'validation_cache_size': len(self.validation_cache),
            'analysis_cache_size': len(self.column_analysis_cache),
            'total_cache_entries': len(self.conversion_cache) + len(self.validation_cache) + len(self.column_analysis_cache),
            'max_cache_size': self.max_cache_size,
            'cache_ttl_minutes': self.cache_ttl.total_seconds() / 60
        }


class LazyDataTypeConverter:
    """Lazy data type converter that only converts when needed"""
    
    def __init__(self, cache: Optional[DataTypeCache] = None):
        self.cache = cache or DataTypeCache()
        self.logger = setup_logging('lazy_converter')
        self.conversion_queue = {}
        self.converted_columns = set()
    
    def mark_for_conversion(self, df: pd.DataFrame, column_name: str, target_type: str = 'numeric') -> None:
        """Mark a column for lazy conversion"""
        if column_name not in df.columns:
            self.logger.warning(f"Column '{column_name}' not found in DataFrame")
            return
        
        self.conversion_queue[column_name] = {
            'target_type': target_type,
            'marked_at': datetime.now(),
            'original_dtype': str(df[column_name].dtype)
        }
        
        self.logger.debug(f"Marked column '{column_name}' for lazy conversion to {target_type}")
    
    def convert_if_needed(self, df: pd.DataFrame, column_name: str, operation: str = 'unknown') -> pd.Series:
        """Convert column only if needed for the operation"""
        if column_name not in df.columns:
            raise KeyError(f"Column '{column_name}' not found in DataFrame")
        
        series = df[column_name]
        
        # Check if already converted
        if column_name in self.converted_columns:
            self.logger.debug(f"Column '{column_name}' already converted, using cached result")
            return series
        
        # Check if conversion is needed based on operation
        if self._is_conversion_needed(series, operation):
            self.logger.debug(f"Converting column '{column_name}' for operation: {operation}")
            
            # Check cache first - use separate kwargs to avoid conflicts
            cache_kwargs = {'operation': operation}
            cached_result = self.cache.get_conversion_result(series, column_name, **cache_kwargs)
            if cached_result is not None:
                converted_series, _ = cached_result
                return converted_series
            
            # Perform conversion
            from utils.data_type_utils import safe_numeric_conversion
            converted_series, report = safe_numeric_conversion(series, column_name, self.logger)
            
            # Cache the result
            self.cache.cache_conversion_result(series, column_name, (converted_series, report), **cache_kwargs)
            
            # Mark as converted
            self.converted_columns.add(column_name)
            
            return converted_series
        
        return series
    
    def _is_conversion_needed(self, series: pd.Series, operation: str) -> bool:
        """Determine if conversion is needed based on data type and operation"""
        # If already numeric, no conversion needed
        if pd.api.types.is_numeric_dtype(series):
            return False
        
        # Operations that require numeric data
        numeric_operations = {
            'mean', 'sum', 'min', 'max', 'std', 'median', 'count',
            'comparison', 'filter', 'calculation', 'aggregation'
        }
        
        # Check if operation requires numeric data
        for op in numeric_operations:
            if op in operation.lower():
                return True
        
        return False
    
    def batch_convert_queued(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert all queued columns in batch"""
        if not self.conversion_queue:
            return df
        
        self.logger.info(f"Batch converting {len(self.conversion_queue)} queued columns")
        
        converted_df = df.copy()
        conversion_reports = {}
        
        for column_name, conversion_info in self.conversion_queue.items():
            if column_name in df.columns:
                try:
                    # Check cache first
                    cached_result = self.cache.get_conversion_result(df[column_name], column_name)
                    if cached_result is not None:
                        converted_series, report = cached_result
                    else:
                        # Perform conversion
                        from utils.data_type_utils import safe_numeric_conversion
                        converted_series, report = safe_numeric_conversion(df[column_name], column_name, self.logger)
                        
                        # Cache the result
                        self.cache.cache_conversion_result(df[column_name], column_name, (converted_series, report))
                    
                    converted_df[column_name] = converted_series
                    conversion_reports[column_name] = report
                    self.converted_columns.add(column_name)
                    
                except Exception as e:
                    self.logger.error(f"Failed to convert column '{column_name}': {str(e)}")
        
        # Clear the queue
        self.conversion_queue.clear()
        
        self.logger.info(f"Batch conversion complete: {len(conversion_reports)} columns processed")
        return converted_df
    
    def get_conversion_stats(self) -> Dict[str, Any]:
        """Get conversion statistics"""
        return {
            'queued_conversions': len(self.conversion_queue),
            'completed_conversions': len(self.converted_columns),
            'queue_details': dict(self.conversion_queue),
            'converted_columns': list(self.converted_columns)
        }


class PerformanceOptimizedValidator:
    """Performance-optimized data type validator with smart caching"""
    
    def __init__(self, cache: Optional[DataTypeCache] = None):
        self.cache = cache or DataTypeCache()
        self.logger = setup_logging('optimized_validator')
        self.validation_history = {}
    
    def validate_column_fast(self, series: pd.Series, column_name: str, expected_type: str = 'numeric') -> bool:
        """Fast column validation with caching"""
        # Check cache first - use separate kwargs to avoid conflicts
        cache_kwargs = {'expected_type': expected_type}
        cached_result = self.cache.get_validation_result(series, column_name, **cache_kwargs)
        if cached_result is not None:
            return cached_result
        
        # Perform validation
        start_time = time.time()
        
        if expected_type == 'numeric':
            is_valid = pd.api.types.is_numeric_dtype(series)
        elif expected_type == 'categorical':
            is_valid = series.dtype.name == 'category'
        else:
            is_valid = str(series.dtype) == expected_type
        
        validation_time = time.time() - start_time
        
        # Cache the result
        self.cache.cache_validation_result(series, column_name, is_valid, **cache_kwargs)
        
        # Track validation history
        self.validation_history[column_name] = {
            'last_validated': datetime.now(),
            'validation_time': validation_time,
            'result': is_valid,
            'expected_type': expected_type
        }
        
        self.logger.debug(f"Validated column '{column_name}' in {validation_time:.4f}s: {is_valid}")
        return is_valid
    
    def batch_validate_columns(self, df: pd.DataFrame, column_types: Dict[str, str]) -> Dict[str, bool]:
        """Batch validate multiple columns efficiently"""
        self.logger.info(f"Batch validating {len(column_types)} columns")
        start_time = time.time()
        
        results = {}
        cache_hits = 0
        
        for column_name, expected_type in column_types.items():
            if column_name not in df.columns:
                results[column_name] = False
                continue
            
            # Check cache first - use separate kwargs to avoid conflicts
            cache_kwargs = {'expected_type': expected_type}
            cached_result = self.cache.get_validation_result(df[column_name], column_name, **cache_kwargs)
            if cached_result is not None:
                results[column_name] = cached_result
                cache_hits += 1
            else:
                # Perform validation
                results[column_name] = self.validate_column_fast(df[column_name], column_name, expected_type)
        
        total_time = time.time() - start_time
        cache_hit_rate = cache_hits / len(column_types) if column_types else 0
        
        self.logger.info(f"Batch validation complete in {total_time:.4f}s (cache hit rate: {cache_hit_rate:.2%})")
        
        return results
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        if not self.validation_history:
            return {'total_validations': 0}
        
        total_time = sum(v['validation_time'] for v in self.validation_history.values())
        avg_time = total_time / len(self.validation_history)
        
        return {
            'total_validations': len(self.validation_history),
            'total_validation_time': total_time,
            'average_validation_time': avg_time,
            'cache_stats': self.cache.get_cache_stats()
        }


class BatchProcessor:
    """Efficient batch processor for data type operations"""
    
    def __init__(self, batch_size: int = 10000, cache: Optional[DataTypeCache] = None):
        self.batch_size = batch_size
        self.cache = cache or DataTypeCache()
        self.logger = setup_logging('batch_processor')
    
    def process_large_dataframe(self, df: pd.DataFrame, operations: List[str]) -> pd.DataFrame:
        """Process large DataFrame in batches to optimize memory usage"""
        if len(df) <= self.batch_size:
            return self._process_dataframe(df, operations)
        
        self.logger.info(f"Processing large DataFrame ({len(df)} rows) in batches of {self.batch_size}")
        
        processed_chunks = []
        total_batches = (len(df) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(df), self.batch_size):
            batch_num = i // self.batch_size + 1
            self.logger.debug(f"Processing batch {batch_num}/{total_batches}")
            
            batch_df = df.iloc[i:i + self.batch_size].copy()
            processed_batch = self._process_dataframe(batch_df, operations)
            processed_chunks.append(processed_batch)
        
        # Combine all batches
        result_df = pd.concat(processed_chunks, ignore_index=True)
        self.logger.info(f"Batch processing complete: {len(result_df)} rows processed")
        
        return result_df
    
    def _process_dataframe(self, df: pd.DataFrame, operations: List[str]) -> pd.DataFrame:
        """Process a single DataFrame with specified operations"""
        processed_df = df.copy()
        
        for operation in operations:
            if operation == 'convert_categorical':
                processed_df = self._convert_categorical_columns(processed_df)
            elif operation == 'validate_types':
                self._validate_data_types(processed_df)
            # Add more operations as needed
        
        return processed_df
    
    def _convert_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical columns efficiently"""
        from utils.data_type_utils import convert_categorical_to_numeric
        
        converted_df, reports = convert_categorical_to_numeric(df, logger=self.logger)
        return converted_df
    
    def _validate_data_types(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Validate data types efficiently"""
        validator = PerformanceOptimizedValidator(self.cache)
        
        # Define expected types for common columns
        expected_types = {}
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['price', 'sqft', 'size', 'year', 'rate']):
                expected_types[col] = 'numeric'
        
        return validator.batch_validate_columns(df, expected_types)


# Global instances for reuse
_global_cache = None
_global_lazy_converter = None
_global_validator = None
_global_batch_processor = None


def get_performance_cache() -> DataTypeCache:
    """Get or create global performance cache"""
    global _global_cache
    if _global_cache is None:
        _global_cache = DataTypeCache()
    return _global_cache


def get_lazy_converter() -> LazyDataTypeConverter:
    """Get or create global lazy converter"""
    global _global_lazy_converter
    if _global_lazy_converter is None:
        _global_lazy_converter = LazyDataTypeConverter(get_performance_cache())
    return _global_lazy_converter


def get_optimized_validator() -> PerformanceOptimizedValidator:
    """Get or create global optimized validator"""
    global _global_validator
    if _global_validator is None:
        _global_validator = PerformanceOptimizedValidator(get_performance_cache())
    return _global_validator


def get_batch_processor() -> BatchProcessor:
    """Get or create global batch processor"""
    global _global_batch_processor
    if _global_batch_processor is None:
        _global_batch_processor = BatchProcessor(cache=get_performance_cache())
    return _global_batch_processor


# Performance monitoring decorator
def monitor_performance(operation_name: str):
    """Decorator to monitor performance of data type operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = setup_logging('performance_monitor')
            start_time = time.time()
            start_memory = _get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                
                end_time = time.time()
                end_memory = _get_memory_usage()
                
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                logger.info(f"{operation_name} completed in {execution_time:.4f}s, memory delta: {memory_delta:.2f}MB")
                
                return result
                
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                logger.error(f"{operation_name} failed after {execution_time:.4f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator


def _get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


# Convenience functions for common performance optimizations
def optimize_dataframe_processing(df: pd.DataFrame, operations: List[str] = None) -> pd.DataFrame:
    """Optimize DataFrame processing with caching and batch processing"""
    if operations is None:
        operations = ['convert_categorical', 'validate_types']
    
    batch_processor = get_batch_processor()
    return batch_processor.process_large_dataframe(df, operations)


def clear_all_caches() -> None:
    """Clear all performance caches"""
    cache = get_performance_cache()
    cache.clear_cache()


def get_performance_stats() -> Dict[str, Any]:
    """Get comprehensive performance statistics"""
    cache = get_performance_cache()
    lazy_converter = get_lazy_converter()
    validator = get_optimized_validator()
    
    return {
        'cache_stats': cache.get_cache_stats(),
        'conversion_stats': lazy_converter.get_conversion_stats(),
        'validation_stats': validator.get_validation_stats(),
        'timestamp': datetime.now().isoformat()
    }