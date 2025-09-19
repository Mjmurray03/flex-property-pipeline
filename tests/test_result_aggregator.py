"""
Unit tests for ResultAggregator class
Tests result combination, deduplication, and aggregation functionality
"""

import unittest
import pandas as pd
import tempfile
import os
from pathlib import Path
from datetime import datetime

from pipeline.result_aggregator import ResultAggregator, AggregationStats, aggregate_processing_results
from pipeline.file_processor import ProcessingResult


class TestAggregationStats(unittest.TestCase):
    """Test AggregationStats dataclass"""
    
    def test_stats_creation(self):
        """Test creating AggregationStats"""
        stats = AggregationStats(
            total_input_files=5,
            successful_files=4,
            total_properties_before=1000,
            total_properties_after=850,
            duplicates_removed=150
        )
        
        self.assertEqual(stats.total_input_files, 5)
        self.assertEqual(stats.successful_files, 4)
        self.assertEqual(stats.total_properties_before, 1000)
        self.assertEqual(stats.total_properties_after, 850)
        self.assertEqual(stats.duplicates_removed, 150)
    
    def test_stats_to_dict(self):
        """Test converting stats to dictionary"""
        stats = AggregationStats(
            total_input_files=5,
            successful_files=4,
            total_properties_before=1000,
            total_properties_after=850,
            duplicates_removed=150,
            unique_cities=25,
            unique_states=3
        )
        
        stats_dict = stats.to_dict()
        
        self.assertEqual(stats_dict['total_input_files'], 5)
        self.assertEqual(stats_dict['successful_files'], 4)
        self.assertEqual(stats_dict['deduplication_rate'], 0.15)  # 150/1000
        self.assertEqual(stats_dict['unique_cities'], 25)
        self.assertEqual(stats_dict['unique_states'], 3)


class TestResultAggregator(unittest.TestCase):
    """Test ResultAggregator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.aggregator = ResultAggregator(
            duplicate_fields=['site_address', 'city', 'state'],
            preserve_highest_score=True
        )
        
        # Create sample property data with duplicates
        self.sample_data_1 = pd.DataFrame({
            'parcel_id': ['P001', 'P002', 'P003'],
            'site_address': ['123 Main St', '456 Oak Ave', '789 Pine Rd'],
            'city': ['Boca Raton', 'Delray Beach', 'Boynton Beach'],
            'state': ['FL', 'FL', 'FL'],
            'flex_score': [8.5, 7.2, 6.8],
            'flex_classification': ['PRIME_FLEX', 'GOOD_FLEX', 'GOOD_FLEX'],
            'source_filename': ['file1.xlsx', 'file1.xlsx', 'file1.xlsx']
        })
        
        self.sample_data_2 = pd.DataFrame({
            'parcel_id': ['P004', 'P005', 'P002'],  # P002 is duplicate from file1
            'site_address': ['321 Elm St', '654 Maple Dr', '456 Oak Ave'],  # Last one is duplicate
            'city': ['West Palm Beach', 'Lake Worth', 'Delray Beach'],
            'state': ['FL', 'FL', 'FL'],
            'flex_score': [9.1, 5.5, 7.8],  # Duplicate has higher score
            'flex_classification': ['PRIME_FLEX', 'POTENTIAL_FLEX', 'GOOD_FLEX'],
            'source_filename': ['file2.xlsx', 'file2.xlsx', 'file2.xlsx']
        })
    
    def test_aggregator_initialization(self):
        """Test ResultAggregator initialization"""
        aggregator = ResultAggregator(
            duplicate_fields=['address', 'city'],
            preserve_highest_score=False,
            case_sensitive_matching=True
        )
        
        self.assertEqual(aggregator.duplicate_fields, ['address', 'city'])
        self.assertFalse(aggregator.preserve_highest_score)
        self.assertTrue(aggregator.case_sensitive_matching)
    
    def test_aggregate_empty_results(self):
        """Test aggregating empty results list"""
        result = self.aggregator.aggregate_results([])
        
        self.assertIsNone(result)
    
    def test_aggregate_no_successful_results(self):
        """Test aggregating results with no successful processing"""
        failed_results = [
            ProcessingResult(
                file_path="file1.xlsx",
                success=False,
                error_message="Processing failed"
            ),
            ProcessingResult(
                file_path="file2.xlsx",
                success=False,
                error_message="File corrupted"
            )
        ]
        
        result = self.aggregator.aggregate_results(failed_results)
        
        self.assertIsNotNone(result)
        self.assertTrue(result.empty)
    
    def test_aggregate_successful_results_no_duplicates(self):
        """Test aggregating successful results without duplicates"""
        # Create processing results without duplicates
        data1 = pd.DataFrame({
            'site_address': ['123 Main St', '456 Oak Ave'],
            'city': ['Boca Raton', 'Delray Beach'],
            'state': ['FL', 'FL'],
            'flex_score': [8.5, 7.2]
        })
        
        data2 = pd.DataFrame({
            'site_address': ['789 Pine Rd', '321 Elm St'],
            'city': ['Boynton Beach', 'West Palm Beach'],
            'state': ['FL', 'FL'],
            'flex_score': [6.8, 9.1]
        })
        
        processing_results = [
            ProcessingResult(
                file_path="file1.xlsx",
                success=True,
                flex_properties=data1,
                source_file_info={'filename': 'file1.xlsx'}
            ),
            ProcessingResult(
                file_path="file2.xlsx",
                success=True,
                flex_properties=data2,
                source_file_info={'filename': 'file2.xlsx'}
            )
        ]
        
        result = self.aggregator.aggregate_results(processing_results)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 4)  # All 4 properties should be included
        
        # Check sorting by flex score (descending)
        scores = result['flex_score'].tolist()
        self.assertEqual(scores, [9.1, 8.5, 7.2, 6.8])
        
        # Check stats
        stats = self.aggregator.get_aggregation_stats()
        self.assertEqual(stats.total_input_files, 2)
        self.assertEqual(stats.successful_files, 2)
        self.assertEqual(stats.total_properties_before, 4)
        self.assertEqual(stats.total_properties_after, 4)
        self.assertEqual(stats.duplicates_removed, 0)
    
    def test_aggregate_with_duplicates_preserve_highest(self):
        """Test aggregating results with duplicates, preserving highest score"""
        processing_results = [
            ProcessingResult(
                file_path="file1.xlsx",
                success=True,
                flex_properties=self.sample_data_1,
                source_file_info={'filename': 'file1.xlsx'}
            ),
            ProcessingResult(
                file_path="file2.xlsx",
                success=True,
                flex_properties=self.sample_data_2,
                source_file_info={'filename': 'file2.xlsx'}
            )
        ]
        
        result = self.aggregator.aggregate_results(processing_results)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 5)  # 6 total - 1 duplicate = 5
        
        # Check that duplicate was removed and higher score preserved
        oak_ave_properties = result[result['site_address'] == '456 Oak Ave']
        self.assertEqual(len(oak_ave_properties), 1)
        self.assertEqual(oak_ave_properties.iloc[0]['flex_score'], 7.8)  # Higher score from file2
        
        # Check stats
        stats = self.aggregator.get_aggregation_stats()
        self.assertEqual(stats.duplicates_removed, 1)
        self.assertEqual(stats.total_properties_before, 6)
        self.assertEqual(stats.total_properties_after, 5)
    
    def test_aggregate_with_duplicates_preserve_first(self):
        """Test aggregating results with duplicates, preserving first occurrence"""
        aggregator = ResultAggregator(
            duplicate_fields=['site_address', 'city', 'state'],
            preserve_highest_score=False  # Keep first occurrence
        )
        
        processing_results = [
            ProcessingResult(
                file_path="file1.xlsx",
                success=True,
                flex_properties=self.sample_data_1,
                source_file_info={'filename': 'file1.xlsx'}
            ),
            ProcessingResult(
                file_path="file2.xlsx",
                success=True,
                flex_properties=self.sample_data_2,
                source_file_info={'filename': 'file2.xlsx'}
            )
        ]
        
        result = aggregator.aggregate_results(processing_results)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 5)  # 6 total - 1 duplicate = 5
        
        # Check that first occurrence was preserved
        oak_ave_properties = result[result['site_address'] == '456 Oak Ave']
        self.assertEqual(len(oak_ave_properties), 1)
        self.assertEqual(oak_ave_properties.iloc[0]['flex_score'], 7.2)  # Lower score from file1 (first)
    
    def test_normalize_addresses(self):
        """Test address normalization for duplicate detection"""
        addresses = pd.Series([
            '123 Main Street',
            '123 Main St',
            '456 Oak Avenue',
            '456 Oak Ave',
            '789 North Pine Road',
            '789 N Pine Rd'
        ])
        
        normalized = self.aggregator._normalize_addresses(addresses)
        
        # Check that variations are normalized to same form (case insensitive comparison)
        self.assertEqual(normalized.iloc[0].lower(), normalized.iloc[1].lower())  # Main Street/St
        self.assertEqual(normalized.iloc[2].lower(), normalized.iloc[3].lower())  # Oak Avenue/Ave
        self.assertEqual(normalized.iloc[4].lower(), normalized.iloc[5].lower())  # North/N, Road/Rd
    
    def test_create_duplicate_key(self):
        """Test duplicate key creation"""
        df = pd.DataFrame({
            'site_address': ['123 Main St', '123 MAIN ST', '456 Oak Ave'],
            'city': ['Boca Raton', 'boca raton', 'Delray Beach'],
            'state': ['FL', 'FL', 'FL']
        })
        
        df_with_key = self.aggregator._create_duplicate_key(df)
        
        self.assertIn('duplicate_key', df_with_key.columns)
        
        # First two should have same key (case insensitive)
        self.assertEqual(df_with_key.iloc[0]['duplicate_key'], df_with_key.iloc[1]['duplicate_key'])
        
        # Third should have different key
        self.assertNotEqual(df_with_key.iloc[0]['duplicate_key'], df_with_key.iloc[2]['duplicate_key'])
    
    def test_calculate_score_distribution(self):
        """Test score distribution calculation"""
        scores = pd.Series([9.5, 8.2, 7.8, 6.5, 5.2, 4.1, 3.8, 2.5])
        
        distribution = self.aggregator._calculate_score_distribution(scores)
        
        self.assertEqual(distribution['score_8_to_10'], 2)  # 9.5, 8.2
        self.assertEqual(distribution['score_6_to_8'], 2)   # 7.8, 6.5
        self.assertEqual(distribution['score_4_to_6'], 2)   # 5.2, 4.1
        self.assertEqual(distribution['score_below_4'], 2)  # 3.8, 2.5
    
    def test_get_duplicate_analysis(self):
        """Test duplicate analysis functionality"""
        # Create DataFrame with known duplicates
        df = pd.DataFrame({
            'site_address': ['123 Main St', '123 Main St', '456 Oak Ave', '789 Pine Rd'],
            'city': ['Boca Raton', 'Boca Raton', 'Delray Beach', 'Boynton Beach'],
            'state': ['FL', 'FL', 'FL', 'FL'],
            'flex_score': [8.5, 7.2, 6.8, 9.1],
            'source_filename': ['file1.xlsx', 'file2.xlsx', 'file1.xlsx', 'file2.xlsx']
        })
        
        analysis = self.aggregator.get_duplicate_analysis(df)
        
        self.assertEqual(analysis['total_properties'], 4)
        self.assertEqual(analysis['potential_duplicates'], 1)  # One duplicate pair
        self.assertEqual(analysis['duplicate_groups_count'], 1)
        self.assertEqual(analysis['largest_duplicate_group'], 2)
        self.assertEqual(len(analysis['duplicate_groups']), 1)
        
        # Check duplicate group details
        group = analysis['duplicate_groups'][0]
        self.assertEqual(group['count'], 2)
        self.assertEqual(len(group['properties']), 2)
    
    def test_export_aggregated_results(self):
        """Test exporting aggregated results to Excel"""
        df = pd.DataFrame({
            'site_address': ['123 Main St', '456 Oak Ave'],
            'city': ['Boca Raton', 'Delray Beach'],
            'state': ['FL', 'FL'],
            'flex_score': [8.5, 7.2]
        })
        
        # Set up some stats
        self.aggregator.stats.successful_files = 2
        self.aggregator.stats.duplicates_removed = 1
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            success = self.aggregator.export_aggregated_results(df, tmp_path, include_metadata=True)
            
            self.assertTrue(success)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Verify file can be read back
            exported_df = pd.read_excel(tmp_path, engine='openpyxl')
            self.assertEqual(len(exported_df), 2)
            self.assertIn('aggregation_stats', exported_df.columns)
            self.assertIn('export_date', exported_df.columns)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_export_empty_dataframe(self):
        """Test exporting empty DataFrame"""
        df = pd.DataFrame()
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            success = self.aggregator.export_aggregated_results(df, tmp_path)
            
            self.assertFalse(success)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""
    
    def test_aggregate_processing_results(self):
        """Test aggregate_processing_results convenience function"""
        data = pd.DataFrame({
            'site_address': ['123 Main St', '456 Oak Ave'],
            'city': ['Boca Raton', 'Delray Beach'],
            'state': ['FL', 'FL'],
            'flex_score': [8.5, 7.2]
        })
        
        processing_results = [
            ProcessingResult(
                file_path="file1.xlsx",
                success=True,
                flex_properties=data,
                source_file_info={'filename': 'file1.xlsx'}
            )
        ]
        
        result = aggregate_processing_results(
            processing_results=processing_results,
            duplicate_fields=['site_address', 'city'],
            preserve_highest_score=True
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        
        # Check sorting
        scores = result['flex_score'].tolist()
        self.assertEqual(scores, [8.5, 7.2])


if __name__ == '__main__':
    unittest.main()