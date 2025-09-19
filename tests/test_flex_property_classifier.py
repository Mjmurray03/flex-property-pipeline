"""
Comprehensive test suite for Flex Property Classifier
Tests all classifier methods with various scenarios and edge cases
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys
import logging
from unittest.mock import Mock, patch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from processors.flex_property_classifier import FlexPropertyClassifier
from utils.logger import setup_logging


class TestFlexPropertyClassifier(unittest.TestCase):
    """Test cases for FlexPropertyClassifier with comprehensive coverage"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create test logger
        self.logger = setup_logging(name='test_classifier', level='DEBUG')
        
        # Create perfect test dataset
        self.perfect_data = pd.DataFrame({
            'Property Name': ['Flex Center A', 'Warehouse B', 'Industrial C'],
            'Property Type': ['Flex Industrial', 'Warehouse', 'Light Industrial'],
            'Building SqFt': [30000, 75000, 150000],
            'Lot Size Acres': [2.5, 7.0, 15.0],
            'Year Built': [1995, 1985, 1975],
            'Occupancy': [85, 95, 100],
            'Address': ['123 Flex St', '456 Warehouse Ave', '789 Industrial Blvd'],
            'City': ['Dallas', 'Houston', 'Austin'],
            'State': ['TX', 'TX', 'TX']
        })
        
        # Create minimal test dataset (only required columns)
        self.minimal_data = pd.DataFrame({
            'Property Type': ['Industrial', 'Warehouse', 'Office'],
            'Building SqFt': [25000, 50000, 10000],
            'Lot Size Acres': [1.0, 5.0, 0.3]
        })
        
        # Create dirty test dataset with missing values and inconsistent formats
        self.dirty_data = pd.DataFrame({
            'Property Type': ['flex industrial', None, 'WAREHOUSE', 'office space'],
            'Building SqFt': [30000, 'invalid', 75000, None],
            'Lot Size Acres': [2.5, None, 'bad_data', 15.0],
            'Year Built': [1995, None, 'old', 1985],
            'Occupancy': [85, None, 150, 'full']  # 150 is invalid percentage
        })
        
        # Create large test dataset for performance testing
        np.random.seed(42)  # For reproducible tests
        n_properties = 1000
        self.large_data = pd.DataFrame({
            'Property Type': np.random.choice(
                ['Industrial', 'Warehouse', 'Flex', 'Office', 'Retail'], 
                n_properties
            ),
            'Building SqFt': np.random.randint(10000, 200000, n_properties),
            'Lot Size Acres': np.random.uniform(0.1, 25.0, n_properties),
            'Year Built': np.random.randint(1970, 2020, n_properties),
            'Occupancy': np.random.uniform(50, 100, n_properties)
        })
        
        # Create no industrial dataset
        self.no_industrial_data = pd.DataFrame({
            'Property Type': ['Office', 'Retail', 'Residential'],
            'Building SqFt': [25000, 50000, 15000],
            'Lot Size Acres': [1.0, 5.0, 2.0]
        })
    
    def test_initialization_valid_dataframe(self):
        """Test successful initialization with valid DataFrame"""
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        
        self.assertIsInstance(classifier, FlexPropertyClassifier)
        self.assertEqual(len(classifier.data), 3)
        self.assertIsNotNone(classifier.logger)
        self.assertIsNone(classifier.flex_candidates)
        self.assertEqual(classifier.min_building_sqft, 20000)
        self.assertEqual(classifier.min_lot_acres, 0.5)
        self.assertEqual(classifier.max_lot_acres, 20.0)
    
    def test_initialization_empty_dataframe(self):
        """Test initialization fails with empty DataFrame"""
        empty_df = pd.DataFrame()
        
        with self.assertRaises(ValueError) as context:
            FlexPropertyClassifier(empty_df, self.logger)
        
        self.assertIn("DataFrame cannot be empty", str(context.exception))
    
    def test_initialization_invalid_input(self):
        """Test initialization fails with non-DataFrame input"""
        with self.assertRaises(TypeError) as context:
            FlexPropertyClassifier("not a dataframe", self.logger)
        
        self.assertIn("Input must be a pandas DataFrame", str(context.exception))
    
    def test_initialization_no_columns(self):
        """Test initialization fails with DataFrame having no columns"""
        no_cols_df = pd.DataFrame(index=[0, 1, 2])
        
        with self.assertRaises(ValueError) as context:
            FlexPropertyClassifier(no_cols_df, self.logger)
        
        # The empty DataFrame check happens first, so we get that error message
        self.assertIn("DataFrame cannot be empty", str(context.exception))
    
    def test_initialization_without_logger(self):
        """Test initialization creates logger when none provided"""
        classifier = FlexPropertyClassifier(self.perfect_data)
        
        self.assertIsNotNone(classifier.logger)
        self.assertIsInstance(classifier.logger, logging.Logger)
    
    def test_classify_flex_properties_perfect_data(self):
        """Test classification with perfect dataset"""
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        candidates = classifier.classify_flex_properties()
        
        # All properties should qualify (all are industrial, >20k sqft, 0.5-20 acres)
        self.assertEqual(len(candidates), 3)
        self.assertIsNotNone(classifier.flex_candidates)
        self.assertTrue(len(classifier.flex_candidates) > 0)
    
    def test_classify_flex_properties_minimal_data(self):
        """Test classification with minimal required columns"""
        classifier = FlexPropertyClassifier(self.minimal_data, self.logger)
        candidates = classifier.classify_flex_properties()
        
        # Should find 2 candidates (Industrial and Warehouse, both >20k sqft and valid lot size)
        self.assertEqual(len(candidates), 2)
    
    def test_classify_flex_properties_dirty_data(self):
        """Test classification handles dirty data gracefully"""
        classifier = FlexPropertyClassifier(self.dirty_data, self.logger)
        candidates = classifier.classify_flex_properties()
        
        # Should handle missing/invalid data and continue processing
        self.assertIsInstance(candidates, pd.DataFrame)
        # At least one property should qualify (flex industrial with valid data)
        self.assertGreaterEqual(len(candidates), 1)
    
    def test_classify_flex_properties_no_industrial(self):
        """Test classification with no industrial properties"""
        classifier = FlexPropertyClassifier(self.no_industrial_data, self.logger)
        candidates = classifier.classify_flex_properties()
        
        # Should return empty DataFrame
        self.assertEqual(len(candidates), 0)
        self.assertIsInstance(candidates, pd.DataFrame)
    
    def test_filter_by_industrial_type(self):
        """Test industrial property type filtering"""
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        
        # Test with mixed property types
        mixed_data = pd.DataFrame({
            'Property Type': ['Industrial', 'Office', 'Warehouse', 'Retail', 'Flex'],
            'Building SqFt': [30000] * 5,
            'Lot Size Acres': [2.0] * 5
        })
        
        filtered = classifier._filter_by_industrial_type(mixed_data)
        
        # Should keep Industrial, Warehouse, and Flex (3 properties)
        self.assertEqual(len(filtered), 3)
        
        # Check that correct types are kept
        kept_types = filtered['Property Type'].str.lower().tolist()
        self.assertIn('industrial', kept_types)
        self.assertIn('warehouse', kept_types)
        self.assertIn('flex', kept_types)
    
    def test_filter_by_industrial_type_missing_column(self):
        """Test industrial filtering when property type column is missing"""
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        
        # Data without property type column
        no_type_data = pd.DataFrame({
            'Building SqFt': [30000, 50000],
            'Lot Size Acres': [2.0, 5.0]
        })
        
        # Should return original data when column is missing
        filtered = classifier._filter_by_industrial_type(no_type_data)
        self.assertEqual(len(filtered), len(no_type_data))
    
    def test_filter_by_building_size(self):
        """Test building size filtering"""
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        
        # Test data with various building sizes
        size_data = pd.DataFrame({
            'Property Type': ['Industrial'] * 4,
            'Building SqFt': [15000, 25000, 50000, 250000],  # Below, above, good, too big
            'Lot Size Acres': [2.0] * 4
        })
        
        filtered = classifier._filter_by_building_size(size_data)
        
        # Should keep properties >= 20,000 sqft (3 properties)
        self.assertEqual(len(filtered), 3)
        
        # Check that sizes are all >= 20,000
        sizes = pd.to_numeric(filtered['Building SqFt'], errors='coerce')
        self.assertTrue(all(sizes >= 20000))
    
    def test_filter_by_building_size_missing_column(self):
        """Test building size filtering when column is missing"""
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        
        # Data without building size column
        no_size_data = pd.DataFrame({
            'Property Type': ['Industrial', 'Warehouse'],
            'Lot Size Acres': [2.0, 5.0]
        })
        
        # Should return original data when column is missing
        filtered = classifier._filter_by_building_size(no_size_data)
        self.assertEqual(len(filtered), len(no_size_data))
    
    def test_filter_by_lot_size(self):
        """Test lot size filtering"""
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        
        # Test data with various lot sizes
        lot_data = pd.DataFrame({
            'Property Type': ['Industrial'] * 5,
            'Building SqFt': [30000] * 5,
            'Lot Size Acres': [0.3, 1.0, 5.0, 15.0, 25.0]  # Too small, good, good, good, too big
        })
        
        filtered = classifier._filter_by_lot_size(lot_data)
        
        # Should keep properties between 0.5-20 acres (3 properties)
        self.assertEqual(len(filtered), 3)
        
        # Check that lot sizes are in valid range
        sizes = pd.to_numeric(filtered['Lot Size Acres'], errors='coerce')
        self.assertTrue(all((sizes >= 0.5) & (sizes <= 20.0)))
    
    def test_filter_by_lot_size_missing_column(self):
        """Test lot size filtering when column is missing"""
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        
        # Data without lot size column
        no_lot_data = pd.DataFrame({
            'Property Type': ['Industrial', 'Warehouse'],
            'Building SqFt': [30000, 50000]
        })
        
        # Should return original data when column is missing
        filtered = classifier._filter_by_lot_size(no_lot_data)
        self.assertEqual(len(filtered), len(no_lot_data))
    
    def test_find_column_case_insensitive(self):
        """Test column finding with case-insensitive matching"""
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        
        # Test DataFrame with various column name formats
        test_df = pd.DataFrame({
            'PROPERTY TYPE': ['Industrial'],
            'building_sqft': [30000],
            'Lot Size Acres': [2.0]
        })
        
        # Should find columns regardless of case
        prop_col = classifier._find_column(test_df, ['property type'])
        building_col = classifier._find_column(test_df, ['sqft'])  # Use 'sqft' which is in the search terms
        lot_col = classifier._find_column(test_df, ['lot size acres'])
        
        self.assertEqual(prop_col, 'PROPERTY TYPE')
        self.assertEqual(building_col, 'building_sqft')
        self.assertEqual(lot_col, 'Lot Size Acres')
    
    def test_find_column_not_found(self):
        """Test column finding when column doesn't exist"""
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        
        result = classifier._find_column(self.perfect_data, ['nonexistent column'])
        self.assertIsNone(result)
    
    def test_calculate_flex_score_perfect_property(self):
        """Test flex score calculation for perfect property"""
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        
        # Create perfect flex property
        perfect_property = pd.Series({
            'Property Type': 'Flex Industrial',
            'Building SqFt': 30000,  # 3 points (20k-50k range)
            'Lot Size Acres': 2.5,   # 2 points (1-5 acre range)
            'Year Built': 1995,      # 1 point (>=1990)
            'Occupancy': 85          # 1 point (<100%)
        })
        
        score = classifier.calculate_flex_score(perfect_property)
        
        # Should get 3 + 3 + 2 + 1 + 1 = 10 points (capped at max)
        self.assertEqual(score, 10.0)
    
    def test_calculate_flex_score_minimal_property(self):
        """Test flex score calculation with minimal data"""
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        
        # Property with only basic data
        minimal_property = pd.Series({
            'Property Type': 'Industrial',
            'Building SqFt': 75000,  # 2 points (50k-100k range)
            'Lot Size Acres': 7.0    # 1.5 points (5-10 acre range)
        })
        
        score = classifier.calculate_flex_score(minimal_property)
        
        # Should get 1.5 + 2 + 1.5 = 5.0 points
        self.assertEqual(score, 5.0)
    
    def test_calculate_flex_score_missing_data(self):
        """Test flex score calculation with missing data"""
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        
        # Property with missing/invalid data
        missing_property = pd.Series({
            'Property Type': None,
            'Building SqFt': 'invalid',
            'Lot Size Acres': None
        })
        
        score = classifier.calculate_flex_score(missing_property)
        
        # Should get 0 points for all factors
        self.assertEqual(score, 0.0)
    
    def test_score_building_size_ranges(self):
        """Test building size scoring for different ranges"""
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        
        # Test different size ranges
        test_cases = [
            (30000, 3.0),    # 20k-50k range
            (75000, 2.0),    # 50k-100k range
            (150000, 1.0),   # 100k-200k range
            (15000, 0.0),    # Below minimum
            (250000, 0.0)    # Above maximum
        ]
        
        for size, expected_score in test_cases:
            property_data = pd.Series({'Building SqFt': size})
            score = classifier._score_building_size(property_data)
            self.assertEqual(score, expected_score, 
                           f"Size {size} should score {expected_score}, got {score}")
    
    def test_score_property_type_categories(self):
        """Test property type scoring for different categories"""
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        
        # Test different property types
        test_cases = [
            ('Flex Industrial', 3.0),
            ('Warehouse', 2.5),
            ('Distribution Center', 2.5),
            ('Light Industrial', 2.0),
            ('Industrial', 1.5),
            ('Manufacturing', 1.0),
            ('Storage Facility', 1.0),
            ('Logistics Center', 1.0),
            ('Office Building', 0.0)
        ]
        
        for prop_type, expected_score in test_cases:
            property_data = pd.Series({'Property Type': prop_type})
            score = classifier._score_property_type(property_data)
            self.assertEqual(score, expected_score,
                           f"Type '{prop_type}' should score {expected_score}, got {score}")
    
    def test_score_lot_size_ranges(self):
        """Test lot size scoring for different ranges"""
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        
        # Test different lot size ranges
        test_cases = [
            (2.5, 2.0),    # 1-5 acre range (ideal)
            (7.0, 1.5),    # 5-10 acre range (good)
            (0.8, 1.0),    # 0.5-1 acre range (acceptable small)
            (15.0, 1.0),   # 10-20 acre range (acceptable large)
            (0.3, 0.0),    # Below minimum
            (25.0, 0.0)    # Above maximum
        ]
        
        for size, expected_score in test_cases:
            property_data = pd.Series({'Lot Size Acres': size})
            score = classifier._score_lot_size(property_data)
            self.assertEqual(score, expected_score,
                           f"Lot size {size} should score {expected_score}, got {score}")
    
    def test_score_age_condition_ranges(self):
        """Test age/condition scoring for different years"""
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        
        # Test different construction years
        test_cases = [
            (2000, 1.0),   # Modern (>=1990)
            (1990, 1.0),   # Exactly 1990
            (1985, 0.5),   # Decent (>=1980)
            (1980, 0.5),   # Exactly 1980
            (1975, 0.0),   # Too old
            (None, 0.0)    # Missing data
        ]
        
        for year, expected_score in test_cases:
            property_data = pd.Series({'Year Built': year})
            score = classifier._score_age_condition(property_data)
            self.assertEqual(score, expected_score,
                           f"Year {year} should score {expected_score}, got {score}")
    
    def test_score_occupancy_bonus(self):
        """Test occupancy bonus scoring"""
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        
        # Test different occupancy levels
        test_cases = [
            (85, 1.0),     # Less than 100% (as percentage)
            (0.85, 1.0),   # Less than 100% (as decimal)
            (100, 0.0),    # Fully occupied (as percentage)
            (1.0, 0.0),    # Fully occupied (as decimal)
            (None, 0.0)    # Missing data
        ]
        
        for occupancy, expected_score in test_cases:
            property_data = pd.Series({'Occupancy': occupancy})
            score = classifier._score_occupancy(property_data)
            self.assertEqual(score, expected_score,
                           f"Occupancy {occupancy} should score {expected_score}, got {score}")
    
    def test_get_top_candidates_before_classification(self):
        """Test get_top_candidates fails before classification"""
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        
        with self.assertRaises(Exception) as context:
            classifier.get_top_candidates()
        
        self.assertIn("Must run classify_flex_properties() first", str(context.exception))
    
    def test_get_top_candidates_with_results(self):
        """Test get_top_candidates with classification results"""
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        classifier.classify_flex_properties()
        
        top_candidates = classifier.get_top_candidates(n=2)
        
        self.assertIsInstance(top_candidates, pd.DataFrame)
        self.assertLessEqual(len(top_candidates), 2)
        self.assertIn('flex_score', top_candidates.columns)
        
        # Check that results are sorted by score (descending)
        if len(top_candidates) > 1:
            scores = top_candidates['flex_score'].tolist()
            self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_get_top_candidates_empty_results(self):
        """Test get_top_candidates with no candidates"""
        classifier = FlexPropertyClassifier(self.no_industrial_data, self.logger)
        classifier.classify_flex_properties()
        
        top_candidates = classifier.get_top_candidates()
        
        self.assertIsInstance(top_candidates, pd.DataFrame)
        self.assertEqual(len(top_candidates), 0)
    
    def test_export_results_before_classification(self):
        """Test export fails before classification"""
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        
        with self.assertRaises(Exception) as context:
            classifier.export_results()
        
        self.assertIn("No flex candidates available for export", str(context.exception))
    
    def test_export_results_with_custom_path(self):
        """Test export with custom output path"""
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        classifier.classify_flex_properties()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_path = Path(temp_dir) / 'custom_export.xlsx'
            result_path = classifier.export_results(str(custom_path))
            
            self.assertEqual(result_path, str(custom_path))
            self.assertTrue(custom_path.exists())
            
            # Verify file can be read back
            exported_df = pd.read_excel(custom_path)
            self.assertGreater(len(exported_df), 0)
            self.assertIn('Flex Score', exported_df.columns)
    
    def test_export_results_default_path(self):
        """Test export with default path"""
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        classifier.classify_flex_properties()
        
        # Clean up any existing export file
        default_path = Path('data/exports/private_flex_candidates.xlsx')
        if default_path.exists():
            default_path.unlink()
        
        try:
            result_path = classifier.export_results()
            
            self.assertTrue(Path(result_path).exists())
            self.assertEqual(Path(result_path).name, 'private_flex_candidates.xlsx')
            
            # Verify file content
            exported_df = pd.read_excel(result_path)
            self.assertGreater(len(exported_df), 0)
            
        finally:
            # Clean up
            if Path(result_path).exists():
                Path(result_path).unlink()
    
    def test_get_analysis_statistics_before_classification(self):
        """Test statistics fails before classification"""
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        
        with self.assertRaises(Exception) as context:
            classifier.get_analysis_statistics()
        
        self.assertIn("No analysis available", str(context.exception))
    
    def test_get_analysis_statistics_with_results(self):
        """Test analysis statistics with classification results"""
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        classifier.classify_flex_properties()
        
        stats = classifier.get_analysis_statistics()
        
        # Check required statistics fields
        required_fields = [
            'total_properties_analyzed',
            'total_flex_candidates', 
            'candidate_percentage',
            'score_statistics',
            'property_type_distribution',
            'size_distribution',
            'geographic_distribution'
        ]
        
        for field in required_fields:
            self.assertIn(field, stats)
        
        # Check specific values
        self.assertEqual(stats['total_properties_analyzed'], 3)
        self.assertGreater(stats['total_flex_candidates'], 0)
        self.assertGreater(stats['candidate_percentage'], 0)
        
        # Check score statistics structure
        if stats['total_flex_candidates'] > 0:
            score_stats = stats['score_statistics']
            score_fields = ['average_flex_score', 'min_score', 'max_score', 'median_score']
            for field in score_fields:
                self.assertIn(field, score_stats)
                self.assertIsInstance(score_stats[field], (int, float))
    
    def test_get_analysis_statistics_empty_results(self):
        """Test analysis statistics with no candidates"""
        classifier = FlexPropertyClassifier(self.no_industrial_data, self.logger)
        classifier.classify_flex_properties()
        
        stats = classifier.get_analysis_statistics()
        
        self.assertEqual(stats['total_flex_candidates'], 0)
        self.assertEqual(stats['candidate_percentage'], 0.0)
    
    def test_error_handling_with_invalid_data(self):
        """Test error handling with various invalid data scenarios"""
        # Test with completely invalid DataFrame
        invalid_data = pd.DataFrame({
            'Invalid Column': ['bad', 'data', 'here']
        })
        
        classifier = FlexPropertyClassifier(invalid_data, self.logger)
        
        # Should not crash, should handle gracefully
        candidates = classifier.classify_flex_properties()
        self.assertIsInstance(candidates, pd.DataFrame)
    
    def test_performance_with_large_dataset(self):
        """Test performance with large dataset"""
        import time
        
        classifier = FlexPropertyClassifier(self.large_data, self.logger)
        
        start_time = time.time()
        candidates = classifier.classify_flex_properties()
        classification_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(classification_time, 10.0, "Classification took too long")
        
        # Should handle large dataset without errors
        self.assertIsInstance(candidates, pd.DataFrame)
        
        # Test scoring performance
        if len(candidates) > 0:
            start_time = time.time()
            top_candidates = classifier.get_top_candidates(100)
            scoring_time = time.time() - start_time
            
            self.assertLess(scoring_time, 5.0, "Scoring took too long")
            self.assertIn('flex_score', top_candidates.columns)
    
    def test_memory_usage_with_large_dataset(self):
        """Test memory usage doesn't grow excessively"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        classifier = FlexPropertyClassifier(self.large_data, self.logger)
        classifier.classify_flex_properties()
        
        if len(classifier.flex_candidates) > 0:
            classifier.get_top_candidates(100)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (adjust threshold as needed)
        self.assertLess(memory_increase, 100, f"Memory usage increased by {memory_increase:.1f} MB")
    
    def test_data_integrity_after_processing(self):
        """Test that original data is not modified during processing"""
        original_data = self.perfect_data.copy()
        classifier = FlexPropertyClassifier(self.perfect_data, self.logger)
        
        classifier.classify_flex_properties()
        
        # Original data should be unchanged
        pd.testing.assert_frame_equal(self.perfect_data, original_data)
        
        # Classifier should have its own copy
        self.assertIsNot(classifier.data, self.perfect_data)
    
    def test_column_name_variations(self):
        """Test handling of various column name formats"""
        # Test data with different column name formats
        varied_columns_data = pd.DataFrame({
            'PROPERTY_TYPE': ['Industrial', 'Warehouse'],
            'building_square_feet': [30000, 50000],
            'Lot_Size_Acres': [2.0, 5.0],
            'year_built': [1995, 1985],
            'occupancy_rate': [85, 95]
        })
        
        classifier = FlexPropertyClassifier(varied_columns_data, self.logger)
        candidates = classifier.classify_flex_properties()
        
        # Should successfully process despite different column names
        self.assertIsInstance(candidates, pd.DataFrame)
        
        # Should be able to calculate scores
        if len(candidates) > 0:
            top_candidates = classifier.get_top_candidates()
            self.assertIn('flex_score', top_candidates.columns)
    
    def test_edge_case_single_property(self):
        """Test handling of single property dataset"""
        single_property = pd.DataFrame({
            'Property Type': ['Flex Industrial'],
            'Building SqFt': [30000],
            'Lot Size Acres': [2.5]
        })
        
        classifier = FlexPropertyClassifier(single_property, self.logger)
        candidates = classifier.classify_flex_properties()
        
        self.assertEqual(len(candidates), 1)
        
        top_candidates = classifier.get_top_candidates()
        self.assertEqual(len(top_candidates), 1)
        
        stats = classifier.get_analysis_statistics()
        self.assertEqual(stats['total_properties_analyzed'], 1)
        self.assertEqual(stats['total_flex_candidates'], 1)


class TestFlexPropertyClassifierIntegration(unittest.TestCase):
    """Integration tests with existing pipeline components"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.logger = setup_logging(name='test_integration', level='DEBUG')
        
        # Create realistic test data
        self.integration_data = pd.DataFrame({
            'Property Name': ['Flex Center Alpha', 'Distribution Hub Beta'],
            'Property Type': ['Flex Industrial', 'Distribution Center'],
            'Building SqFt': [45000, 85000],
            'Lot Size Acres': [3.2, 8.5],
            'Year Built': [1992, 1988],
            'Occupancy': [88, 92],
            'Address': ['100 Industrial Way', '200 Logistics Blvd'],
            'City': ['Dallas', 'Houston'],
            'State': ['TX', 'TX'],
            'County': ['Dallas County', 'Harris County'],
            'Zoning Code': ['I-2', 'I-3'],
            'Sale Date': ['2023-01-15', '2023-02-20'],
            'Sold Price': [2500000, 4200000],
            'Sold Price/SqFt': [55.56, 49.41],
            'Owner Name': ['Alpha Properties LLC', 'Beta Logistics Inc']
        })
    
    def test_integration_with_logger(self):
        """Test integration with existing logger infrastructure"""
        # Test that classifier works with existing logger setup
        classifier = FlexPropertyClassifier(self.integration_data, self.logger)
        
        # Should use provided logger
        self.assertEqual(classifier.logger, self.logger)
        
        # Should log classification steps
        with self.assertLogs(self.logger, level='INFO') as log_context:
            classifier.classify_flex_properties()
        
        # Check that appropriate log messages were generated
        log_messages = ' '.join(log_context.output)
        self.assertIn('Starting flex property classification', log_messages)
        self.assertIn('Classification complete', log_messages)
    
    def test_integration_complete_workflow(self):
        """Test complete workflow from classification to export"""
        classifier = FlexPropertyClassifier(self.integration_data, self.logger)
        
        # Step 1: Classify properties
        candidates = classifier.classify_flex_properties()
        self.assertGreater(len(candidates), 0)
        
        # Step 2: Get top candidates
        top_candidates = classifier.get_top_candidates(10)
        self.assertGreater(len(top_candidates), 0)
        self.assertIn('flex_score', top_candidates.columns)
        
        # Step 3: Get statistics
        stats = classifier.get_analysis_statistics()
        self.assertIn('total_flex_candidates', stats)
        self.assertGreater(stats['total_flex_candidates'], 0)
        
        # Step 4: Export results
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = Path(temp_dir) / 'integration_test_export.xlsx'
            result_path = classifier.export_results(str(export_path))
            
            self.assertTrue(Path(result_path).exists())
            
            # Verify exported data
            exported_df = pd.read_excel(result_path)
            self.assertGreater(len(exported_df), 0)
            self.assertIn('Flex Score', exported_df.columns)
    
    @patch('processors.flex_property_classifier.pd.read_excel')
    def test_integration_with_excel_loading_error(self, mock_read_excel):
        """Test integration when Excel loading fails"""
        # Mock Excel loading failure
        mock_read_excel.side_effect = Exception("File not found")
        
        # Should handle gracefully in real usage
        # This test ensures error handling patterns are consistent
        classifier = FlexPropertyClassifier(self.integration_data, self.logger)
        
        # Classification should still work with pre-loaded data
        candidates = classifier.classify_flex_properties()
        self.assertIsInstance(candidates, pd.DataFrame)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)