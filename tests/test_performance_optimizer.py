"""
Tests for performance optimization and memory management
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import time

from pipeline.performance_optimizer import (
    PerformanceOptimizer, MemoryMonitor, ChunkedDataProcessor,
    MemoryStats, PerformanceMetrics, optimize_dataframe_memory, get_memory_usage
)


class TestMemoryMonitor:
    """Test memory monitoring functionality"""
    
    def test_get_memory_stats(self):
        """Test memory statistics collection"""
        monitor = MemoryMonitor()
        stats = monitor.get_memory_stats()
        
        assert isinstance(stats, MemoryStats)
        assert stats.total_memory_gb > 0
        assert stats.available_memory_gb > 0
        assert stats.memory_percent >= 0
        assert stats.process_memory_gb > 0
    
    def test_check_memory_usage(self):
        """Test memory usage checking"""
        monitor = MemoryMonitor(warning_threshold_percent=50.0, critical_threshold_percent=80.0)
        
        is_safe, message = monitor.check_memory_usage()
        
        assert isinstance(is_safe, bool)
        assert isinstance(message, str)
        assert "Memory usage" in message
    
    def test_monitor_operation_context(self):
        """Test memory monitoring context manager"""
        monitor = MemoryMonitor()
        
        with monitor.monitor_operation("test_operation") as start_stats:
            assert isinstance(start_stats, MemoryStats)
            # Simulate some work
            time.sleep(0.01)
    
    def test_memory_stats_to_dict(self):
        """Test MemoryStats serialization"""
        monitor = MemoryMonitor()
        stats = monitor.get_memory_stats()
        stats_dict = stats.to_dict()
        
        required_keys = ['total_memory_gb', 'available_memory_gb', 'used_memory_gb', 
                        'memory_percent', 'process_memory_gb', 'timestamp']
        
        for key in required_keys:
            assert key in stats_dict
            assert isinstance(stats_dict[key], (int, float, str))


class TestChunkedDataProcessor:
    """Test chunked data processing"""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing"""
        return pd.DataFrame({
            'A': range(1000),
            'B': np.random.randn(1000),
            'C': ['text_' + str(i) for i in range(1000)]
        })
    
    def test_process_dataframe_chunked_small(self, sample_dataframe):
        """Test chunked processing with small DataFrame"""
        processor = ChunkedDataProcessor(chunk_size=2000)  # Larger than DataFrame
        
        def processing_func(df):
            return df * 2  # Simple transformation
        
        result = processor.process_dataframe_chunked(
            sample_dataframe, 
            processing_func, 
            "test_small"
        )
        
        # Should process normally without chunking
        assert len(result) == len(sample_dataframe)
    
    def test_process_dataframe_chunked_large(self, sample_dataframe):
        """Test chunked processing with large DataFrame"""
        processor = ChunkedDataProcessor(chunk_size=100)  # Smaller than DataFrame
        
        def processing_func(df):
            # Add a new column
            df_copy = df.copy()
            df_copy['D'] = df_copy['A'] * 2
            return df_copy
        
        result = processor.process_dataframe_chunked(
            sample_dataframe, 
            processing_func, 
            "test_large"
        )
        
        # Should have processed all records
        assert len(result) == len(sample_dataframe)
        assert 'D' in result.columns
        assert (result['D'] == result['A'] * 2).all()
    
    def test_read_excel_chunked(self):
        """Test chunked Excel reading"""
        # Create temporary Excel file
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            # Create test data
            test_df = pd.DataFrame({
                'col1': range(500),
                'col2': np.random.randn(500)
            })
            test_df.to_excel(tmp_path, index=False)
            
            processor = ChunkedDataProcessor(chunk_size=100)
            
            chunks = list(processor.read_excel_chunked(tmp_path))
            
            # Should have multiple chunks
            assert len(chunks) > 1
            
            # Total records should match
            total_records = sum(len(chunk) for chunk in chunks)
            assert total_records == len(test_df)
            
        finally:
            tmp_path.unlink(missing_ok=True)


class TestPerformanceOptimizer:
    """Test performance optimization functionality"""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame with various data types"""
        return pd.DataFrame({
            'int_col': np.random.randint(0, 100, 1000),
            'float_col': np.random.randn(1000),
            'string_col': ['category_' + str(i % 10) for i in range(1000)],  # Categorical candidate
            'text_col': ['unique_text_' + str(i) for i in range(1000)],  # Not categorical
            'empty_col': [None] * 1000  # Empty column
        })
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        optimizer = PerformanceOptimizer(
            memory_limit_gb=2.0,
            chunk_size=5000,
            enable_monitoring=False,
            optimization_level="balanced"
        )
        
        assert optimizer.memory_limit_gb == 2.0
        assert optimizer.chunk_size == 5000
        assert optimizer.optimization_level == "balanced"
        assert optimizer.memory_monitor is not None
        assert optimizer.chunked_processor is not None
    
    def test_optimize_dataframe_operations(self, sample_dataframe):
        """Test DataFrame optimization"""
        optimizer = PerformanceOptimizer(enable_monitoring=False)
        
        original_memory = sample_dataframe.memory_usage(deep=True).sum()
        optimized_df = optimizer.optimize_dataframe_operations(sample_dataframe)
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        
        # Should have same number of records
        assert len(optimized_df) == len(sample_dataframe)
        
        # Should have removed empty column
        assert 'empty_col' not in optimized_df.columns
        
        # String column with repeated values should be categorical
        assert optimized_df['string_col'].dtype.name == 'category'
        
        # Memory should be reduced or same
        assert optimized_memory <= original_memory
    
    def test_track_performance_context(self):
        """Test performance tracking context manager"""
        optimizer = PerformanceOptimizer(enable_monitoring=False)
        
        with optimizer.track_performance("test_operation") as metrics:
            assert isinstance(metrics, PerformanceMetrics)
            assert metrics.operation_name == "test_operation"
            assert metrics.start_time is not None
            
            # Simulate work
            time.sleep(0.01)
        
        # Should have recorded metrics
        assert len(optimizer.metrics) == 1
        assert optimizer.metrics[0].duration_seconds > 0
    
    def test_process_large_file(self):
        """Test large file processing"""
        # Create temporary Excel file
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            # Create test data
            test_df = pd.DataFrame({
                'col1': range(100),
                'col2': np.random.randn(100)
            })
            test_df.to_excel(tmp_path, index=False)
            
            optimizer = PerformanceOptimizer(enable_monitoring=False)
            
            def processing_func(df):
                df_copy = df.copy()
                df_copy['processed'] = True
                return df_copy
            
            result = optimizer.process_large_file(tmp_path, processing_func)
            
            assert len(result) == len(test_df)
            assert 'processed' in result.columns
            assert result['processed'].all()
            
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_get_performance_summary(self):
        """Test performance summary generation"""
        optimizer = PerformanceOptimizer(enable_monitoring=False)
        
        # Add some mock metrics
        with optimizer.track_performance("test_op_1"):
            time.sleep(0.01)
        
        with optimizer.track_performance("test_op_2"):
            time.sleep(0.01)
        
        summary = optimizer.get_performance_summary()
        
        assert 'total_operations' in summary
        assert 'total_duration_seconds' in summary
        assert 'optimization_level' in summary
        assert summary['total_operations'] == 2
        assert summary['total_duration_seconds'] > 0
    
    def test_optimization_levels(self):
        """Test different optimization levels"""
        # Test fast optimization
        fast_optimizer = PerformanceOptimizer(
            optimization_level="fast",
            enable_monitoring=False
        )
        assert fast_optimizer.optimization_level == "fast"
        
        # Test memory efficient optimization
        memory_optimizer = PerformanceOptimizer(
            optimization_level="memory_efficient",
            enable_monitoring=False
        )
        assert memory_optimizer.optimization_level == "memory_efficient"
        
        # Test balanced optimization
        balanced_optimizer = PerformanceOptimizer(
            optimization_level="balanced",
            enable_monitoring=False
        )
        assert balanced_optimizer.optimization_level == "balanced"
    
    def test_cleanup(self):
        """Test optimizer cleanup"""
        optimizer = PerformanceOptimizer(enable_monitoring=False)
        
        # Should not raise any exceptions
        optimizer.cleanup()


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_optimize_dataframe_memory(self):
        """Test DataFrame memory optimization convenience function"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'empty_col': [None, None, None, None, None]
        })
        
        optimized_df = optimize_dataframe_memory(df)
        
        # Should have removed empty column
        assert 'empty_col' not in optimized_df.columns
        assert len(optimized_df) == len(df)
    
    def test_get_memory_usage(self):
        """Test memory usage convenience function"""
        memory_info = get_memory_usage()
        
        required_keys = ['total_memory_gb', 'available_memory_gb', 'memory_percent']
        for key in required_keys:
            assert key in memory_info
            assert isinstance(memory_info[key], (int, float))


class TestPerformanceMetrics:
    """Test performance metrics functionality"""
    
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics creation and finalization"""
        metrics = PerformanceMetrics(
            operation_name="test_operation",
            start_time=pd.Timestamp.now()
        )
        
        assert metrics.operation_name == "test_operation"
        assert metrics.duration_seconds == 0.0
        
        # Finalize with some data
        metrics.finalize(records_processed=1000)
        
        assert metrics.duration_seconds > 0
        assert metrics.records_processed == 1000
        assert metrics.throughput_records_per_second > 0
    
    def test_performance_metrics_to_dict(self):
        """Test PerformanceMetrics serialization"""
        metrics = PerformanceMetrics(
            operation_name="test_operation",
            start_time=pd.Timestamp.now()
        )
        metrics.finalize(records_processed=500)
        
        metrics_dict = metrics.to_dict()
        
        required_keys = ['operation_name', 'start_time', 'duration_seconds', 
                        'records_processed', 'throughput_records_per_second']
        
        for key in required_keys:
            assert key in metrics_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])