"""
Integration tests for BatchProcessor class
Tests concurrent file processing, progress tracking, and error handling
"""

import unittest
import tempfile
import os
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time
from datetime import datetime

from pipeline.batch_processor import BatchProcessor, BatchProcessingStats, ProgressTracker, process_files_batch
from pipeline.file_processor import ProcessingResult


class TestBatchProcessingStats(unittest.TestCase):
    """Test BatchProcessingStats dataclass"""
    
    def test_stats_creation(self):
        """Test creating BatchProcessingStats"""
        stats = BatchProcessingStats(
            total_files=10,
            successful_files=8,
            failed_files=2,
            total_properties=1000,
            total_flex_candidates=150
        )
        
        self.assertEqual(stats.total_files, 10)
        self.assertEqual(stats.successful_files, 8)
        self.assertEqual(stats.failed_files, 2)
        self.assertEqual(stats.total_properties, 1000)
        self.assertEqual(stats.total_flex_candidates, 150)
    
    def test_stats_to_dict(self):
        """Test converting stats to dictionary"""
        start_time = datetime.now()
        stats = BatchProcessingStats(
            total_files=5,
            processed_files=5,
            successful_files=4,
            failed_files=1,
            start_time=start_time,
            processing_duration=10.5
        )
        
        stats_dict = stats.to_dict()
        
        self.assertEqual(stats_dict['total_files'], 5)
        self.assertEqual(stats_dict['successful_files'], 4)
        self.assertEqual(stats_dict['failed_files'], 1)
        self.assertEqual(stats_dict['processing_duration'], 10.5)
        self.assertEqual(stats_dict['success_rate'], 0.8)  # 4/5
        self.assertIsNotNone(stats_dict['start_time'])


class TestProgressTracker(unittest.TestCase):
    """Test ProgressTracker functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.progress_callback = Mock()
        self.tracker = ProgressTracker(
            total_files=5,
            progress_callback=self.progress_callback
        )
    
    def test_progress_tracker_initialization(self):
        """Test ProgressTracker initialization"""
        self.assertEqual(self.tracker.total_files, 5)
        self.assertEqual(self.tracker.processed_files, 0)
        self.assertEqual(self.tracker.successful_files, 0)
        self.assertEqual(self.tracker.failed_files, 0)
    
    def test_update_progress_success(self):
        """Test updating progress with successful result"""
        result = ProcessingResult(
            file_path="test1.xlsx",
            success=True,
            property_count=100,
            flex_candidate_count=25
        )
        
        self.tracker.update_progress(result)
        
        self.assertEqual(self.tracker.processed_files, 1)
        self.assertEqual(self.tracker.successful_files, 1)
        self.assertEqual(self.tracker.failed_files, 0)
        
        # Check callback was called
        self.progress_callback.assert_called_once()
        call_args = self.progress_callback.call_args[0][0]
        self.assertEqual(call_args['processed'], 1)
        self.assertEqual(call_args['total'], 5)
        self.assertEqual(call_args['successful'], 1)
        self.assertEqual(call_args['progress_pct'], 20.0)
    
    def test_update_progress_failure(self):
        """Test updating progress with failed result"""
        result = ProcessingResult(
            file_path="test1.xlsx",
            success=False,
            error_message="File not found"
        )
        
        self.tracker.update_progress(result)
        
        self.assertEqual(self.tracker.processed_files, 1)
        self.assertEqual(self.tracker.successful_files, 0)
        self.assertEqual(self.tracker.failed_files, 1)
    
    def test_get_stats(self):
        """Test getting progress statistics"""
        # Process some files
        for i in range(3):
            result = ProcessingResult(
                file_path=f"test{i}.xlsx",
                success=i < 2  # First 2 succeed, last fails
            )
            self.tracker.update_progress(result)
        
        stats = self.tracker.get_stats()
        
        self.assertEqual(stats['total_files'], 5)
        self.assertEqual(stats['processed_files'], 3)
        self.assertEqual(stats['successful_files'], 2)
        self.assertEqual(stats['failed_files'], 1)
        self.assertEqual(stats['progress_pct'], 60.0)  # 3/5 * 100


class TestBatchProcessor(unittest.TestCase):
    """Test BatchProcessor functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.progress_callback = Mock()
        self.processor = BatchProcessor(
            max_workers=2,
            min_flex_score=4.0,
            timeout_minutes=1,
            progress_callback=self.progress_callback
        )
        
        # Create temporary test files
        self.temp_dir = tempfile.mkdtemp()
        self.test_files = []
        
        for i in range(3):
            file_path = Path(self.temp_dir) / f"test_file_{i}.xlsx"
            
            # Create simple test data
            df = pd.DataFrame({
                'Property Address': [f'{100+i} Test St', f'{200+i} Main Ave'],
                'City': ['Boca Raton', 'Delray Beach'],
                'State': ['FL', 'FL'],
                'Lot Size Acres': [2.0, 1.5],
                'Zoning Code': ['IL', 'IP'],
                'Improvement Value': [500000, 750000]
            })
            
            df.to_excel(file_path, index=False, engine='openpyxl')
            self.test_files.append(file_path)
    
    def tearDown(self):
        """Clean up test files"""
        for file_path in self.test_files:
            if file_path.exists():
                os.unlink(file_path)
        os.rmdir(self.temp_dir)
    
    def test_processor_initialization(self):
        """Test BatchProcessor initialization"""
        processor = BatchProcessor(
            max_workers=4,
            min_flex_score=5.0,
            timeout_minutes=30
        )
        
        self.assertEqual(processor.max_workers, 4)
        self.assertEqual(processor.min_flex_score, 5.0)
        self.assertEqual(processor.timeout_seconds, 1800)  # 30 * 60
        self.assertIsNotNone(processor.stats)
    
    def test_process_empty_file_list(self):
        """Test processing empty file list"""
        results = self.processor.process_files([])
        
        self.assertEqual(len(results), 0)
        self.assertEqual(self.processor.stats.total_files, 0)
    
    @patch('pipeline.batch_processor.FileProcessor')
    def test_process_files_success(self, mock_file_processor_class):
        """Test successful file processing"""
        # Mock FileProcessor to return successful results
        mock_processor = Mock()
        mock_file_processor_class.return_value = mock_processor
        
        def mock_process_file(file_path):
            return ProcessingResult(
                file_path=str(file_path),
                success=True,
                property_count=100,
                flex_candidate_count=25,
                processing_time=1.0,
                source_file_info={'filename': file_path.name}
            )
        
        mock_processor.process_file.side_effect = mock_process_file
        
        # Process files
        results = self.processor.process_files(self.test_files)
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r.success for r in results))
        
        # Verify stats
        stats = self.processor.get_processing_stats()
        self.assertEqual(stats.total_files, 3)
        self.assertEqual(stats.successful_files, 3)
        self.assertEqual(stats.failed_files, 0)
        self.assertEqual(stats.total_properties, 300)  # 100 * 3
        self.assertEqual(stats.total_flex_candidates, 75)  # 25 * 3
    
    @patch('pipeline.batch_processor.FileProcessor')
    def test_process_files_with_failures(self, mock_file_processor_class):
        """Test file processing with some failures"""
        mock_processor = Mock()
        mock_file_processor_class.return_value = mock_processor
        
        def mock_process_file(file_path):
            # First file succeeds, others fail
            if "test_file_0" in str(file_path):
                return ProcessingResult(
                    file_path=str(file_path),
                    success=True,
                    property_count=100,
                    flex_candidate_count=25,
                    processing_time=1.0
                )
            else:
                return ProcessingResult(
                    file_path=str(file_path),
                    success=False,
                    error_message="Processing failed",
                    processing_time=0.5
                )
        
        mock_processor.process_file.side_effect = mock_process_file
        
        # Process files
        results = self.processor.process_files(self.test_files)
        
        # Verify results
        self.assertEqual(len(results), 3)
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        self.assertEqual(len(successful_results), 1)
        self.assertEqual(len(failed_results), 2)
        
        # Verify stats
        stats = self.processor.get_processing_stats()
        self.assertEqual(stats.successful_files, 1)
        self.assertEqual(stats.failed_files, 2)
    
    def test_get_error_summary(self):
        """Test error summary generation"""
        # Simulate some processing results with errors
        self.processor.processing_results = [
            ProcessingResult(
                file_path="file1.xlsx",
                success=True,
                property_count=100
            ),
            ProcessingResult(
                file_path="file2.xlsx",
                success=False,
                error_message="File timeout occurred"
            ),
            ProcessingResult(
                file_path="file3.xlsx",
                success=False,
                error_message="Required columns not found in file"
            ),
            ProcessingResult(
                file_path="file4.xlsx",
                success=False,
                error_message="File corrupted or invalid format"
            )
        ]
        
        error_summary = self.processor.get_error_summary()
        
        self.assertEqual(error_summary['total_errors'], 3)
        self.assertEqual(error_summary['error_rate'], 0.75)  # 3/4
        
        # Check error categorization
        error_types = error_summary['error_types']
        self.assertIn('timeout', error_types)
        self.assertIn('file_not_found', error_types)  # "columns not found" gets categorized as file_not_found
        self.assertIn('file_corruption', error_types)
    
    def test_get_performance_metrics(self):
        """Test performance metrics calculation"""
        # Set up mock processing results
        start_time = datetime.now()
        self.processor.stats = BatchProcessingStats(
            total_files=4,
            successful_files=3,
            failed_files=1,
            total_properties=300,
            total_flex_candidates=75,
            start_time=start_time,
            processing_duration=60.0,  # 1 minute
            average_processing_time=15.0
        )
        
        self.processor.processing_results = [
            ProcessingResult(
                file_path="file1.xlsx",
                success=True,
                property_count=100,
                flex_candidate_count=25,
                source_file_info={'file_size_mb': 2.5}
            ),
            ProcessingResult(
                file_path="file2.xlsx",
                success=True,
                property_count=100,
                flex_candidate_count=25,
                source_file_info={'file_size_mb': 3.0}
            ),
            ProcessingResult(
                file_path="file3.xlsx",
                success=True,
                property_count=100,
                flex_candidate_count=25,
                source_file_info={'file_size_mb': 2.0}
            ),
            ProcessingResult(
                file_path="file4.xlsx",
                success=False,
                error_message="Failed"
            )
        ]
        
        metrics = self.processor.get_performance_metrics()
        
        self.assertEqual(metrics['files_per_minute'], 3.0)  # 3 successful files / 1 minute
        self.assertEqual(metrics['properties_per_minute'], 300.0)  # 300 properties / 1 minute
        self.assertEqual(metrics['candidates_per_minute'], 75.0)  # 75 candidates / 1 minute
        self.assertEqual(metrics['average_file_size_mb'], 2.5)  # (2.5 + 3.0 + 2.0) / 3
        self.assertEqual(metrics['average_processing_time'], 15.0)
        self.assertEqual(metrics['total_duration'], 60.0)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""
    
    @patch('pipeline.batch_processor.BatchProcessor')
    def test_process_files_batch(self, mock_batch_processor_class):
        """Test process_files_batch convenience function"""
        mock_processor = Mock()
        mock_batch_processor_class.return_value = mock_processor
        
        # Mock return value
        expected_results = [
            ProcessingResult(file_path="test1.xlsx", success=True),
            ProcessingResult(file_path="test2.xlsx", success=True)
        ]
        mock_processor.process_files.return_value = expected_results
        
        # Test the convenience function
        file_paths = [Path("test1.xlsx"), Path("test2.xlsx")]
        progress_callback = Mock()
        
        results = process_files_batch(
            file_paths=file_paths,
            max_workers=4,
            min_flex_score=5.0,
            progress_callback=progress_callback
        )
        
        # Verify BatchProcessor was created with correct parameters
        mock_batch_processor_class.assert_called_once_with(
            max_workers=4,
            min_flex_score=5.0,
            progress_callback=progress_callback
        )
        
        # Verify process_files was called
        mock_processor.process_files.assert_called_once_with(file_paths)
        
        # Verify results
        self.assertEqual(results, expected_results)


if __name__ == '__main__':
    unittest.main()