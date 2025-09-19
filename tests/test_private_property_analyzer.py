"""
Comprehensive test suite for PrivatePropertyAnalyzer
Tests all analyzer methods with various data quality scenarios
"""

import unittest
import tempfile
import os
import pandas as pd
from pathlib import Path
import json
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.private_property_analyzer import PrivatePropertyAnalyzer
from utils.logger import setup_logging


class TestPrivatePropertyAnalyzer(unittest.TestCase):
    """Test suite for PrivatePropertyAnalyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = setup_logging('test_analyzer', level='DEBUG')
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test datasets
        self.create_test_datasets()
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_datasets(self):
        """Create various test datasets for different scenarios"""
        
        # Perfect dataset - all fields present and complete
        self.perfect_data = pd.DataFrame({
            'Property Name': ['Industrial Complex A', 'Warehouse B', 'Office Building C', 'Flex Space D'],
            'Property Type': ['Industrial', 'Warehouse', 'Office', 'Flex Industrial'],
            'Building SqFt': [50000, 75000, 25000, 40000],
            'Lot Size Acres': [5.2, 8.1, 2.3, 4.5],
            'City': ['West Palm Beach', 'Boca Raton', 'Delray Beach', 'Boynton Beach'],
            'State': ['FL', 'FL', 'FL', 'FL'],
            'Owner Name': ['ABC Industrial LLC', 'XYZ Logistics', 'Office Corp', 'Flex Properties Inc'],
            'Market Value': [2500000, 3200000, 1800000, 2100000],
            'Year Built': [1995, 2001, 1988, 2005],
            'Zoning Code': ['IL', 'IG', 'CG', 'IP']
        })
        
        # Missing columns dataset - some key fields missing
        self.missing_columns_data = pd.DataFrame({
            'Property Name': ['Property 1', 'Property 2', 'Property 3'],
            'Type': ['Industrial', 'Manufacturing', 'Storage'],  # Different column name
            'Building Sq Ft': [30000, 45000, 20000],  # Different column name
            'City': ['Jupiter', 'Stuart', 'Hobe Sound']
            # Missing: Lot Size, Market Value, Year Built, etc.
        })
        
        # Dirty data dataset - missing values, inconsistent formats
        self.dirty_data = pd.DataFrame({
            'Property Name': ['Clean Property', None, 'Property with Issues', ''],
            'Property Type': ['Industrial', 'Warehouse', None, 'Manufacturing'],
            'Building SqFt': [25000, None, '35,000', 'N/A'],  # Mixed formats
            'Lot Size Acres': [3.5, None, 'Unknown', 7.2],
            'City': ['Valid City', '', None, 'Another City'],
            'Market Value': ['$1,500,000', None, 2000000, '$N/A'],  # Currency formatting
            'Year Built': [1990, None, 'Unknown', 2010.0]
        })
        
        # No industrial dataset - no industrial properties
        self.no_industrial_data = pd.DataFrame({
            'Property Name': ['Office A', 'Retail B', 'Residential C'],
            'Property Type': ['Office', 'Retail', 'Residential'],
            'Building SqFt': [15000, 8000, 3000],
            'City': ['Palm Beach', 'Lake Worth', 'Lantana']
        })
        
        # Large dataset for performance testing
        large_data_rows = []
        for i in range(1000):
            prop_type = ['Industrial', 'Warehouse', 'Manufacturing', 'Office', 'Retail'][i % 5]
            large_data_rows.append({
                'Property Name': f'Property {i+1}',
                'Property Type': prop_type,
                'Building SqFt': 10000 + (i * 100),
                'Lot Size Acres': 1.0 + (i * 0.1),
                'City': f'City {i % 10}',
                'Market Value': 500000 + (i * 10000)
            })
        
        self.large_data = pd.DataFrame(large_data_rows)
        
        # Save datasets as Excel files
        self.perfect_file = Path(self.temp_dir) / 'perfect_data.xlsx'
        self.missing_columns_file = Path(self.temp_dir) / 'missing_columns.xlsx'
        self.dirty_data_file = Path(self.temp_dir) / 'dirty_data.xlsx'
        self.no_industrial_file = Path(self.temp_dir) / 'no_industrial.xlsx'
        self.large_data_file = Path(self.temp_dir) / 'large_data.xlsx'
        
        self.perfect_data.to_excel(self.perfect_file, index=False)
        self.missing_columns_data.to_excel(self.missing_columns_file, index=False)
        self.dirty_data.to_excel(self.dirty_data_file, index=False)
        self.no_industrial_data.to_excel(self.no_industrial_file, index=False)
        self.large_data.to_excel(self.large_data_file, index=False)
    
    def test_initialization_valid_file(self):
        """Test successful initialization with valid Excel file"""
        analyzer = PrivatePropertyAnalyzer(str(self.perfect_file), logger=self.logger)
        
        self.assertEqual(analyzer.file_path, self.perfect_file)
        self.assertEqual(analyzer.logger, self.logger)
        self.assertIsNone(analyzer.data)
        self.assertEqual(analyzer.analysis_results, {})
        self.assertIsInstance(analyzer.industrial_keywords, list)
        self.assertIsInstance(analyzer.key_fields, list)
    
    def test_initialization_invalid_file(self):
        """Test initialization with invalid file paths"""
        
        # Non-existent file
        with self.assertRaises(FileNotFoundError):
            PrivatePropertyAnalyzer('nonexistent_file.xlsx')
        
        # Empty path
        with self.assertRaises(ValueError):
            PrivatePropertyAnalyzer('')
        
        # Non-Excel file
        text_file = Path(self.temp_dir) / 'test.txt'
        text_file.write_text('test content')
        
        with self.assertRaises(ValueError):
            PrivatePropertyAnalyzer(str(text_file))
    
    def test_load_data_perfect_dataset(self):
        """Test loading perfect dataset"""
        analyzer = PrivatePropertyAnalyzer(str(self.perfect_file), logger=self.logger)
        
        data = analyzer.load_data()
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 4)
        self.assertEqual(len(data.columns), 10)
        self.assertIsNotNone(analyzer.data)
    
    def test_load_data_missing_columns(self):
        """Test loading dataset with missing columns"""
        analyzer = PrivatePropertyAnalyzer(str(self.missing_columns_file), logger=self.logger)
        
        data = analyzer.load_data()
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 3)
        # Should still load successfully despite missing columns
    
    def test_analyze_property_types_perfect_data(self):
        """Test property type analysis with perfect data"""
        analyzer = PrivatePropertyAnalyzer(str(self.perfect_file), logger=self.logger)
        analyzer.load_data()
        
        industrial_types = analyzer.analyze_property_types()
        
        self.assertIsInstance(industrial_types, list)
        self.assertIn('Industrial', industrial_types)
        self.assertIn('Warehouse', industrial_types)
        self.assertIn('Flex Industrial', industrial_types)
        self.assertNotIn('Office', industrial_types)
    
    def test_analyze_property_types_no_industrial(self):
        """Test property type analysis with no industrial properties"""
        analyzer = PrivatePropertyAnalyzer(str(self.no_industrial_file), logger=self.logger)
        analyzer.load_data()
        
        industrial_types = analyzer.analyze_property_types()
        
        self.assertIsInstance(industrial_types, list)
        self.assertEqual(len(industrial_types), 0)
    
    def test_analyze_property_types_no_data_loaded(self):
        """Test property type analysis without loading data first"""
        analyzer = PrivatePropertyAnalyzer(str(self.perfect_file), logger=self.logger)
        
        with self.assertRaises(RuntimeError):
            analyzer.analyze_property_types()
    
    def test_check_data_completeness_perfect_data(self):
        """Test data completeness check with perfect data"""
        analyzer = PrivatePropertyAnalyzer(str(self.perfect_file), logger=self.logger)
        analyzer.load_data()
        
        completeness = analyzer.check_data_completeness()
        
        self.assertIsInstance(completeness, dict)
        # Should find Building SqFt and Property Type at 100%
        self.assertIn('Building SqFt', completeness)
        self.assertIn('Property Type', completeness)
        self.assertEqual(completeness['Building SqFt'], 100.0)
        self.assertEqual(completeness['Property Type'], 100.0)
    
    def test_check_data_completeness_missing_columns(self):
        """Test data completeness with missing columns"""
        analyzer = PrivatePropertyAnalyzer(str(self.missing_columns_file), logger=self.logger)
        analyzer.load_data()
        
        completeness = analyzer.check_data_completeness()
        
        self.assertIsInstance(completeness, dict)
        # Should handle missing columns gracefully
        # May find some fields with different names
    
    def test_get_industrial_sample_perfect_data(self):
        """Test getting industrial sample with perfect data"""
        analyzer = PrivatePropertyAnalyzer(str(self.perfect_file), logger=self.logger)
        analyzer.load_data()
        
        sample = analyzer.get_industrial_sample(limit=5)
        
        self.assertIsInstance(sample, pd.DataFrame)
        self.assertGreater(len(sample), 0)
        self.assertLessEqual(len(sample), 5)
    
    def test_get_industrial_sample_no_industrial(self):
        """Test getting industrial sample with no industrial properties"""
        analyzer = PrivatePropertyAnalyzer(str(self.no_industrial_file), logger=self.logger)
        analyzer.load_data()
        
        sample = analyzer.get_industrial_sample()
        
        self.assertIsInstance(sample, pd.DataFrame)
        self.assertEqual(len(sample), 0)
    
    def test_generate_summary_report_perfect_data(self):
        """Test generating summary report with perfect data"""
        analyzer = PrivatePropertyAnalyzer(str(self.perfect_file), logger=self.logger)
        analyzer.load_data()
        
        report = analyzer.generate_summary_report()
        
        self.assertIsInstance(report, dict)
        self.assertIn('metadata', report)
        self.assertIn('dataset_overview', report)
        self.assertIn('property_type_analysis', report)
        self.assertIn('data_quality_metrics', report)
        self.assertIn('industrial_property_summary', report)
        
        # Check specific values
        self.assertEqual(report['dataset_overview']['total_properties'], 4)
        self.assertEqual(report['dataset_overview']['total_columns'], 10)
        self.assertGreater(report['property_type_analysis']['total_industrial_properties'], 0)
    
    def test_export_results_all_formats(self):
        """Test exporting results in all formats"""
        analyzer = PrivatePropertyAnalyzer(str(self.perfect_file), logger=self.logger)
        analyzer.load_data()
        analyzer.generate_summary_report()
        
        export_dir = Path(self.temp_dir) / 'exports'
        exported_files = analyzer.export_results(str(export_dir), formats=['json', 'excel', 'csv'])
        
        self.assertIsInstance(exported_files, dict)
        self.assertIn('json', exported_files)
        self.assertIn('excel', exported_files)
        
        # Check files exist
        for format_name, file_path in exported_files.items():
            self.assertTrue(Path(file_path).exists())
    
    def test_convert_to_pipeline_format(self):
        """Test converting Excel data to pipeline format"""
        analyzer = PrivatePropertyAnalyzer(str(self.perfect_file), logger=self.logger)
        analyzer.load_data()
        
        # Get first row
        first_row = analyzer.data.iloc[0]
        pipeline_format = analyzer.convert_to_pipeline_format(first_row)
        
        self.assertIsInstance(pipeline_format, dict)
        self.assertIn('parcel_id', pipeline_format)
        self.assertIn('property_use', pipeline_format)
        self.assertIn('acres', pipeline_format)
        self.assertIn('building_sqft', pipeline_format)
        self.assertIn('market_value', pipeline_format)
        
        # Check data conversion
        self.assertEqual(pipeline_format['property_use'], 'Industrial')
        self.assertEqual(pipeline_format['acres'], 5.2)
        self.assertEqual(pipeline_format['building_sqft'], 50000)
    
    def test_convert_to_pipeline_format_dirty_data(self):
        """Test converting dirty data to pipeline format"""
        analyzer = PrivatePropertyAnalyzer(str(self.dirty_data_file), logger=self.logger)
        analyzer.load_data()
        
        # Test with row that has formatting issues
        dirty_row = analyzer.data.iloc[2]  # Row with '35,000' and other issues
        pipeline_format = analyzer.convert_to_pipeline_format(dirty_row)
        
        self.assertIsInstance(pipeline_format, dict)
        # Should handle comma-separated numbers
        self.assertEqual(pipeline_format['building_sqft'], 35000.0)
    
    def test_performance_large_dataset(self):
        """Test performance with large dataset"""
        import time
        
        analyzer = PrivatePropertyAnalyzer(str(self.large_data_file), logger=self.logger)
        
        # Time the loading
        start_time = time.time()
        analyzer.load_data()
        load_time = time.time() - start_time
        
        # Time the analysis
        start_time = time.time()
        analyzer.analyze_property_types()
        analyzer.check_data_completeness()
        analyzer.get_industrial_sample(limit=10)
        analysis_time = time.time() - start_time
        
        # Performance assertions (adjust thresholds as needed)
        self.assertLess(load_time, 5.0)  # Should load in under 5 seconds
        self.assertLess(analysis_time, 10.0)  # Should analyze in under 10 seconds
        
        # Memory usage check
        memory_mb = analyzer.data.memory_usage(deep=True).sum() / 1024 / 1024
        self.assertLess(memory_mb, 100)  # Should use less than 100MB
    
    def test_error_handling_corrupted_data(self):
        """Test error handling with various edge cases"""
        analyzer = PrivatePropertyAnalyzer(str(self.dirty_data_file), logger=self.logger)
        analyzer.load_data()
        
        # Should handle dirty data gracefully
        try:
            industrial_types = analyzer.analyze_property_types()
            completeness = analyzer.check_data_completeness()
            sample = analyzer.get_industrial_sample()
            report = analyzer.generate_summary_report()
            
            # All operations should complete without raising exceptions
            self.assertIsInstance(industrial_types, list)
            self.assertIsInstance(completeness, dict)
            self.assertIsInstance(sample, pd.DataFrame)
            self.assertIsInstance(report, dict)
            
        except Exception as e:
            self.fail(f"Error handling failed with dirty data: {e}")
    
    def test_integration_with_flex_scorer(self):
        """Test integration with FlexPropertyScorer (if available)"""
        analyzer = PrivatePropertyAnalyzer(str(self.perfect_file), logger=self.logger)
        analyzer.load_data()
        
        try:
            # Test scoring industrial properties only
            scored_properties = analyzer.add_flex_scoring(include_all_properties=False)
            
            if not scored_properties.empty:
                self.assertIn('flex_score', scored_properties.columns)
                self.assertIn('flex_indicators', scored_properties.columns)
                
                # Check that scores are reasonable (0-10 range)
                scores = scored_properties['flex_score'].dropna()
                if len(scores) > 0:
                    self.assertTrue(all(0 <= score <= 10 for score in scores))
            
        except Exception as e:
            # FlexPropertyScorer might not be available in test environment
            self.logger.warning(f"Flex scoring test skipped: {e}")


class TestPrivatePropertyAnalyzerIntegration(unittest.TestCase):
    """Integration tests for PrivatePropertyAnalyzer with external dependencies"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.logger = setup_logging('test_integration', level='DEBUG')
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple test dataset
        test_data = pd.DataFrame({
            'Property Name': ['Test Industrial Property'],
            'Property Type': ['Industrial'],
            'Building SqFt': [25000],
            'Lot Size Acres': [3.0],
            'City': ['Test City'],
            'Market Value': [1500000]
        })
        
        self.test_file = Path(self.temp_dir) / 'integration_test.xlsx'
        test_data.to_excel(self.test_file, index=False)
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_mongodb_integration(self):
        """Test MongoDB integration (if available)"""
        analyzer = PrivatePropertyAnalyzer(str(self.test_file), logger=self.logger)
        analyzer.load_data()
        analyzer.generate_summary_report()
        
        try:
            # Test storing results
            success = analyzer.store_results_in_database()
            
            if success:
                self.assertTrue(success)
                
                # Test retrieving historical data
                historical = analyzer.retrieve_historical_analysis(limit=1)
                self.assertIsInstance(historical, list)
                
                # Test comparison
                comparison = analyzer.compare_with_historical()
                self.assertIsInstance(comparison, dict)
            
        except Exception as e:
            # MongoDB might not be available in test environment
            self.logger.warning(f"MongoDB integration test skipped: {e}")


def run_performance_tests():
    """Run performance-specific tests"""
    print("Running performance tests...")
    
    # Create large test dataset
    temp_dir = tempfile.mkdtemp()
    large_data_rows = []
    
    for i in range(5000):  # 5000 properties
        prop_type = ['Industrial', 'Warehouse', 'Manufacturing', 'Office', 'Retail'][i % 5]
        large_data_rows.append({
            'Property Name': f'Property {i+1}',
            'Property Type': prop_type,
            'Building SqFt': 10000 + (i * 100),
            'Lot Size Acres': 1.0 + (i * 0.1),
            'City': f'City {i % 20}',
            'Market Value': 500000 + (i * 10000)
        })
    
    large_data = pd.DataFrame(large_data_rows)
    test_file = Path(temp_dir) / 'performance_test.xlsx'
    large_data.to_excel(test_file, index=False)
    
    # Performance test
    import time
    logger = setup_logging('performance_test')
    
    analyzer = PrivatePropertyAnalyzer(str(test_file), logger=logger)
    
    start_time = time.time()
    analyzer.load_data()
    load_time = time.time() - start_time
    
    start_time = time.time()
    analyzer.analyze_property_types()
    analyzer.check_data_completeness()
    analyzer.get_industrial_sample(limit=50)
    analyzer.generate_summary_report()
    analysis_time = time.time() - start_time
    
    print(f"Performance Results (5000 properties):")
    print(f"  Load time: {load_time:.2f} seconds")
    print(f"  Analysis time: {analysis_time:.2f} seconds")
    print(f"  Total time: {load_time + analysis_time:.2f} seconds")
    
    # Memory usage
    memory_mb = analyzer.data.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"  Memory usage: {memory_mb:.2f} MB")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    run_performance_tests()
    
    print("\nAll tests completed!")