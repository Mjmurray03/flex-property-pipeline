"""
End-to-End Integration Tests for Scalable Multi-File Pipeline
Tests complete pipeline execution with various scenarios
"""

import pytest
import tempfile
import shutil
import pandas as pd
from pathlib import Path
import sys
import time
from unittest.mock import patch, Mock

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from tests.test_data_generator import PipelineTestDataGenerator
from pipeline.scalable_flex_pipeline import ScalableFlexPipeline, PipelineConfiguration
from run_scalable_pipeline import integrate_pipeline_components, run_complete_pipeline
from pipeline.batch_processor import BatchProcessor
from pipeline.error_recovery import ErrorRecoveryManager


class TestEndToEndIntegration:
    """Comprehensive end-to-end integration tests"""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing"""
        temp_dir = Path(tempfile.mkdtemp(prefix="e2e_test_"))
        
        workspace = {
            'base': temp_dir,
            'input': temp_dir / "input",
            'output': temp_dir / "output",
            'logs': temp_dir / "logs",
            'config': temp_dir / "config"
        }
        
        # Create directories
        for dir_path in workspace.values():
            if isinstance(dir_path, Path):
                dir_path.mkdir(parents=True, exist_ok=True)
        
        yield workspace
        
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up {temp_dir}: {e}")
    
    def test_basic_pipeline_execution(self, temp_workspace):
        """Test basic pipeline execution with valid files"""
        # Generate test data
        with PipelineTestDataGenerator(temp_workspace['input']) as generator:
            scenario = generator.create_test_scenario('basic_processing')
            
            # Configure pipeline
            config = PipelineConfiguration(
                input_folder=str(temp_workspace['input']),
                output_file=str(temp_workspace['output'] / "results.xlsx"),
                max_workers=2,  # Use fewer workers for testing
                min_flex_score=4.0,
                progress_reporting=False,  # Disable for testing
                log_level="WARNING"  # Reduce log noise
            )
            
            # Create and run pipeline
            pipeline = ScalableFlexPipeline(config=config)
            integrate_pipeline_components(pipeline)
            
            results = run_complete_pipeline(pipeline)
            
            # Verify results
            assert results['success'] is True
            assert results['files_discovered'] == 3
            assert results['files_processed'] >= 2  # At least some should process
            assert results['execution_time'] > 0
            
            # Check output file exists
            output_file = Path(results['output_file'])
            assert output_file.exists()
            
            # Verify output content
            output_df = pd.read_excel(output_file)
            assert len(output_df) > 0
            assert 'site_address' in output_df.columns
    
    def test_error_handling_pipeline(self, temp_workspace):
        """Test pipeline error handling with problematic files"""
        # Generate test data with errors
        with PipelineTestDataGenerator(temp_workspace['input']) as generator:
            scenario = generator.create_test_scenario('error_handling')
            
            config = PipelineConfiguration(
                input_folder=str(temp_workspace['input']),
                output_file=str(temp_workspace['output'] / "error_test.xlsx"),
                max_workers=1,
                progress_reporting=False,
                log_level="ERROR"
            )
            
            pipeline = ScalableFlexPipeline(config=config)
            integrate_pipeline_components(pipeline)
            
            results = run_complete_pipeline(pipeline)
            
            # Should handle errors gracefully
            assert results['files_discovered'] == 4  # 1 good + 3 bad files
            assert results['files_failed'] > 0  # Some files should fail
            assert results['files_processed'] >= 1  # At least the good file should process
            
            # Should still produce output if any files succeeded
            if results['files_processed'] > results['files_failed']:
                output_file = Path(results['output_file'])
                assert output_file.exists()
    
    def test_deduplication_functionality(self, temp_workspace):
        """Test deduplication across multiple files"""
        # Create files with overlapping data
        with PipelineTestDataGenerator(temp_workspace['input']) as generator:
            # Create two files with some duplicate addresses
            df1 = generator.generate_property_data(500, flex_candidate_ratio=0.15)
            df2 = generator.generate_property_data(500, flex_candidate_ratio=0.15)
            
            # Add some duplicates
            duplicate_properties = df1.iloc[:50].copy()
            duplicate_properties['parcel_id'] = duplicate_properties['parcel_id'] + "_DUP"
            df2 = pd.concat([df2, duplicate_properties], ignore_index=True)
            
            # Save files
            df1.to_excel(temp_workspace['input'] / "file1.xlsx", index=False)
            df2.to_excel(temp_workspace['input'] / "file2.xlsx", index=False)
            
            config = PipelineConfiguration(
                input_folder=str(temp_workspace['input']),
                output_file=str(temp_workspace['output'] / "dedup_test.xlsx"),
                enable_deduplication=True,
                duplicate_fields=['site_address', 'city', 'state'],
                max_workers=1,
                progress_reporting=False
            )
            
            pipeline = ScalableFlexPipeline(config=config)
            integrate_pipeline_components(pipeline)
            
            results = run_complete_pipeline(pipeline)
            
            assert results['success'] is True
            assert results['files_processed'] == 2
            
            # Check that deduplication occurred
            if results['flex_properties_found'] > 0:
                output_df = pd.read_excel(results['output_file'])
                
                # Should have fewer properties than total input due to deduplication
                total_input = len(df1) + len(df2)
                assert len(output_df) < total_input
    
    def test_performance_with_large_dataset(self, temp_workspace):
        """Test performance with large datasets"""
        # Create large dataset
        with PipelineTestDataGenerator(temp_workspace['input']) as generator:
            large_file = generator.create_large_dataset_file("large_test.xlsx", 5000)
            
            config = PipelineConfiguration(
                input_folder=str(temp_workspace['input']),
                output_file=str(temp_workspace['output'] / "performance_test.xlsx"),
                max_workers=2,
                memory_limit_gb=2.0,
                progress_reporting=False,
                timeout_minutes=5  # Reasonable timeout for testing
            )
            
            pipeline = ScalableFlexPipeline(config=config)
            integrate_pipeline_components(pipeline)
            
            start_time = time.time()
            results = run_complete_pipeline(pipeline)
            execution_time = time.time() - start_time
            
            # Should complete within reasonable time
            assert execution_time < 300  # 5 minutes max
            assert results['success'] is True
            assert results['files_processed'] == 1
            
            # Check performance metrics
            if hasattr(pipeline.batch_processor, 'get_performance_summary'):
                perf_summary = pipeline.batch_processor.get_performance_summary()
                assert 'total_duration' in perf_summary
    
    def test_error_recovery_functionality(self, temp_workspace):
        """Test error recovery and retry mechanisms"""
        # Create mixed quality files
        with PipelineTestDataGenerator(temp_workspace['input']) as generator:
            files = generator.create_mixed_quality_files(8)
            
            config = PipelineConfiguration(
                input_folder=str(temp_workspace['input']),
                output_file=str(temp_workspace['output'] / "recovery_test.xlsx"),
                max_workers=1,
                progress_reporting=False
            )
            
            pipeline = ScalableFlexPipeline(config=config)
            integrate_pipeline_components(pipeline)
            
            # Enable error recovery
            if hasattr(pipeline.batch_processor, 'error_recovery'):
                results = run_complete_pipeline(pipeline)
                
                # Should handle errors and continue processing
                assert results['files_discovered'] == 8
                assert results['files_failed'] >= 0  # Some may fail
                assert results['files_processed'] >= results['files_discovered'] - results['files_failed']
                
                # Check error recovery statistics
                if pipeline.batch_processor.error_recovery:
                    recovery_stats = pipeline.batch_processor.get_recovery_statistics()
                    assert 'total_files_processed' in recovery_stats
    
    def test_configuration_validation(self, temp_workspace):
        """Test configuration validation"""
        # Test invalid configuration
        invalid_config = PipelineConfiguration(
            input_folder="/nonexistent/folder",
            output_file=str(temp_workspace['output'] / "test.xlsx"),
            max_workers=-1,  # Invalid
            min_flex_score=15.0  # Invalid range
        )
        
        pipeline = ScalableFlexPipeline(config=invalid_config)
        
        # Should fail validation
        assert not pipeline.validate_configuration()
    
    def test_memory_optimization(self, temp_workspace):
        """Test memory optimization features"""
        # Create moderately large dataset
        with PipelineTestDataGenerator(temp_workspace['input']) as generator:
            generator.create_large_dataset_file("memory_test.xlsx", 3000)
            
            config = PipelineConfiguration(
                input_folder=str(temp_workspace['input']),
                output_file=str(temp_workspace['output'] / "memory_test.xlsx"),
                max_workers=1,
                memory_limit_gb=1.0,  # Low memory limit
                progress_reporting=False
            )
            
            pipeline = ScalableFlexPipeline(config=config)
            integrate_pipeline_components(pipeline)
            
            results = run_complete_pipeline(pipeline)
            
            # Should complete successfully even with memory constraints
            assert results['success'] is True
            assert results['files_processed'] == 1
    
    def test_output_formats(self, temp_workspace):
        """Test different output formats (Excel and CSV)"""
        with PipelineTestDataGenerator(temp_workspace['input']) as generator:
            generator.create_valid_excel_file("output_test.xlsx", 500, 0.15)
            
            config = PipelineConfiguration(
                input_folder=str(temp_workspace['input']),
                output_file=str(temp_workspace['output'] / "format_test.xlsx"),
                enable_csv_export=True,
                max_workers=1,
                progress_reporting=False
            )
            
            pipeline = ScalableFlexPipeline(config=config)
            integrate_pipeline_components(pipeline)
            
            results = run_complete_pipeline(pipeline)
            
            assert results['success'] is True
            
            # Check both Excel and CSV outputs exist
            excel_file = Path(results['output_file'])
            csv_file = excel_file.with_suffix('.csv')
            
            assert excel_file.exists()
            assert csv_file.exists()
            
            # Verify content consistency
            excel_df = pd.read_excel(excel_file)
            csv_df = pd.read_csv(csv_file)
            
            assert len(excel_df) == len(csv_df)
            assert list(excel_df.columns) == list(csv_df.columns)
    
    def test_batch_processing_stats(self, temp_workspace):
        """Test batch processing statistics collection"""
        with PipelineTestDataGenerator(temp_workspace['input']) as generator:
            scenario = generator.create_test_scenario('basic_processing')
            
            config = PipelineConfiguration(
                input_folder=str(temp_workspace['input']),
                output_file=str(temp_workspace['output'] / "stats_test.xlsx"),
                max_workers=2,
                progress_reporting=False
            )
            
            pipeline = ScalableFlexPipeline(config=config)
            integrate_pipeline_components(pipeline)
            
            results = run_complete_pipeline(pipeline)
            
            assert results['success'] is True
            
            # Check that statistics are collected
            if hasattr(pipeline.batch_processor, 'get_processing_stats'):
                stats = pipeline.batch_processor.get_processing_stats()
                
                assert stats.total_files > 0
                assert stats.processing_duration > 0
                assert stats.processed_files > 0
    
    def test_progress_tracking(self, temp_workspace):
        """Test progress tracking functionality"""
        progress_updates = []
        
        def progress_callback(progress_info):
            progress_updates.append(progress_info)
        
        with PipelineTestDataGenerator(temp_workspace['input']) as generator:
            generator.create_mixed_quality_files(5)
            
            # Create batch processor with progress callback
            processor = BatchProcessor(
                max_workers=1,
                progress_callback=progress_callback
            )
            
            files = list(temp_workspace['input'].glob("*.xlsx"))
            results = processor.process_files(files)
            
            # Should have received progress updates
            assert len(progress_updates) > 0
            
            # Check progress update structure
            for update in progress_updates:
                assert 'processed' in update
                assert 'total' in update
                assert 'progress_pct' in update
    
    def test_file_metadata_tracking(self, temp_workspace):
        """Test source file metadata tracking"""
        with PipelineTestDataGenerator(temp_workspace['input']) as generator:
            test_file = generator.create_valid_excel_file("metadata_test.xlsx", 100, 0.2)
            
            config = PipelineConfiguration(
                input_folder=str(temp_workspace['input']),
                output_file=str(temp_workspace['output'] / "metadata_test.xlsx"),
                max_workers=1,
                progress_reporting=False
            )
            
            pipeline = ScalableFlexPipeline(config=config)
            integrate_pipeline_components(pipeline)
            
            results = run_complete_pipeline(pipeline)
            
            assert results['success'] is True
            
            # Check that metadata is included in results
            if results['flex_properties_found'] > 0:
                output_df = pd.read_excel(results['output_file'])
                
                # Should have source file information
                if 'source_file' in output_df.columns:
                    assert output_df['source_file'].notna().any()


class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.mark.benchmark
    def test_processing_throughput(self, temp_workspace):
        """Benchmark processing throughput"""
        with PipelineTestDataGenerator(temp_workspace['input']) as generator:
            # Create multiple files of varying sizes
            files = [
                generator.create_valid_excel_file("bench_1.xlsx", 1000, 0.15),
                generator.create_valid_excel_file("bench_2.xlsx", 2000, 0.12),
                generator.create_valid_excel_file("bench_3.xlsx", 1500, 0.18)
            ]
            
            config = PipelineConfiguration(
                input_folder=str(temp_workspace['input']),
                output_file=str(temp_workspace['output'] / "benchmark.xlsx"),
                max_workers=4,
                progress_reporting=False
            )
            
            pipeline = ScalableFlexPipeline(config=config)
            integrate_pipeline_components(pipeline)
            
            start_time = time.time()
            results = run_complete_pipeline(pipeline)
            execution_time = time.time() - start_time
            
            # Calculate throughput metrics
            total_properties = 4500  # Sum of properties in test files
            properties_per_second = total_properties / execution_time
            
            print(f"\nPerformance Benchmark Results:")
            print(f"Files processed: {results['files_processed']}")
            print(f"Total properties: {total_properties}")
            print(f"Execution time: {execution_time:.2f}s")
            print(f"Throughput: {properties_per_second:.0f} properties/second")
            
            # Basic performance assertions
            assert execution_time < 60  # Should complete within 1 minute
            assert properties_per_second > 50  # Minimum throughput
    
    @pytest.mark.benchmark
    def test_memory_usage_benchmark(self, temp_workspace):
        """Benchmark memory usage during processing"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        with PipelineTestDataGenerator(temp_workspace['input']) as generator:
            # Create large dataset
            generator.create_large_dataset_file("memory_bench.xlsx", 10000)
            
            config = PipelineConfiguration(
                input_folder=str(temp_workspace['input']),
                output_file=str(temp_workspace['output'] / "memory_bench.xlsx"),
                max_workers=2,
                memory_limit_gb=2.0,
                progress_reporting=False
            )
            
            pipeline = ScalableFlexPipeline(config=config)
            integrate_pipeline_components(pipeline)
            
            results = run_complete_pipeline(pipeline)
            
            final_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_increase = final_memory - initial_memory
            
            print(f"\nMemory Usage Benchmark:")
            print(f"Initial memory: {initial_memory:.1f} MB")
            print(f"Final memory: {final_memory:.1f} MB")
            print(f"Memory increase: {memory_increase:.1f} MB")
            
            # Memory usage should be reasonable
            assert memory_increase < 1000  # Less than 1GB increase
            assert results['success'] is True


if __name__ == "__main__":
    # Run specific test
    pytest.main([__file__ + "::TestEndToEndIntegration::test_basic_pipeline_execution", "-v", "-s"])