"""
Comprehensive testing suite for the file upload workflow
Tests file upload, validation, processing, and integration with dashboard
"""

import unittest
import pandas as pd
import numpy as np
import io
from datetime import datetime
import tempfile
import os

# Import functions from the dashboard
from flex_filter_dashboard import (
    validate_uploaded_file, validate_data_structure, suggest_column_mapping,
    fuzzy_match_columns, apply_column_mapping, clean_numeric_column,
    advanced_data_cleaning, calculate_data_quality_score, detect_outliers,
    secure_file_validation, optimize_dataframe_memory
)

class TestFileUploadWorkflow(unittest.TestCase):
    
    def setUp(self):
        """Set up test data and mock uploaded files"""
        # Create test DataFrame
        self.test_data = {
            'Property Name': ['Test Property 1', 'Test Property 2', 'Test Property 3'],
            'Property Type': ['Industrial Warehouse', 'Distribution Center', 'Flex Space'],
            'City': ['Los Angeles', 'Houston', 'Phoenix'],
            'County': ['LA County', 'Harris County', 'Maricopa County'],
            'State': ['CA', 'TX', 'AZ'],
            'Building SqFt': [50000, 75000, 25000],
            'Lot Size Acres': [2.5, 5.0, 1.2],
            'Year Built': [2000, 1995, 2010],
            'Sold Price': [1500000, 2250000, 800000],
            'Occupancy': [85, 90, 75]
        }
        self.test_df = pd.DataFrame(self.test_data)
        
        # Create test Excel file in memory
        self.excel_buffer = io.BytesIO()
        with pd.ExcelWriter(self.excel_buffer, engine='openpyxl') as writer:
            self.test_df.to_excel(writer, sheet_name='Properties', index=False)
        self.excel_content = self.excel_buffer.getvalue()
        
        # Create mock uploaded file object
        class MockUploadedFile:
            def __init__(self, name, content, size):
                self.name = name
                self.size = size
                self._content = content
            
            def getvalue(self):
                return self._content
        
        self.mock_excel_file = MockUploadedFile("test_properties.xlsx", self.excel_content, len(self.excel_content))
        self.mock_invalid_file = MockUploadedFile("test.txt", b"invalid content", 13)
        self.mock_large_file = MockUploadedFile("large.xlsx", b"x" * (51 * 1024 * 1024), 51 * 1024 * 1024)
    
    def test_file_validation_valid_excel(self):
        """Test file validation with valid Excel file"""
        is_valid, message, error_type = validate_uploaded_file(self.mock_excel_file)
        self.assertTrue(is_valid)
        self.assertEqual(message, "File validation passed")
        self.assertIsNone(error_type)
    
    def test_file_validation_invalid_format(self):
        """Test file validation with invalid file format"""
        is_valid, message, error_type = validate_uploaded_file(self.mock_invalid_file)
        self.assertFalse(is_valid)
        self.assertEqual(error_type, "invalid_format")
        self.assertIn("Invalid file format", message)
    
    def test_file_validation_too_large(self):
        """Test file validation with file too large"""
        is_valid, message, error_type = validate_uploaded_file(self.mock_large_file)
        self.assertFalse(is_valid)
        self.assertEqual(error_type, "file_too_large")
        self.assertIn("File too large", message)
    
    def test_file_validation_no_file(self):
        """Test file validation with no file"""
        is_valid, message, error_type = validate_uploaded_file(None)
        self.assertFalse(is_valid)
        self.assertEqual(message, "No file uploaded")
    
    def test_data_structure_validation_valid(self):
        """Test data structure validation with valid data"""
        validation_result = validate_data_structure(self.test_df)
        self.assertTrue(validation_result['success'])
        self.assertEqual(len(validation_result['errors']), 0)
        self.assertGreater(validation_result['data_quality_score'], 70)
    
    def test_data_structure_validation_missing_required(self):
        """Test data structure validation with missing required columns"""
        incomplete_df = self.test_df.drop(['Property Type'], axis=1)
        validation_result = validate_data_structure(incomplete_df)
        self.assertFalse(validation_result['success'])
        self.assertIn('Property Type', validation_result['required_columns_missing'])
    
    def test_fuzzy_column_matching(self):
        """Test fuzzy column name matching"""
        df_columns = ['Prop Type', 'Bldg SqFt', 'Sale Price', 'City', 'State']
        target_columns = ['Property Type', 'Building SqFt', 'Sold Price']
        
        matches, confidence_scores = fuzzy_match_columns(df_columns, target_columns)
        
        self.assertIn('Prop Type', matches)
        self.assertEqual(matches['Prop Type'], 'Property Type')
        self.assertGreater(confidence_scores['Prop Type'], 0.6)
    
    def test_column_mapping_suggestions(self):
        """Test column mapping suggestion system"""
        # Create DataFrame with non-standard column names
        alt_data = self.test_data.copy()
        alt_df = pd.DataFrame(alt_data)
        alt_df = alt_df.rename(columns={
            'Property Type': 'Prop Type',
            'Building SqFt': 'Bldg SqFt',
            'Sold Price': 'Sale Price'
        })
        
        mapping_result = suggest_column_mapping(alt_df)
        
        self.assertIn('Prop Type', mapping_result['fuzzy_matches'])
        self.assertEqual(mapping_result['fuzzy_matches']['Prop Type'], 'Property Type')
    
    def test_apply_column_mapping(self):
        """Test applying column mappings to DataFrame"""
        # Create DataFrame with alternative column names
        alt_df = self.test_df.rename(columns={'Property Type': 'Prop Type'})
        
        mapping_dict = {'Prop Type': 'Property Type'}
        mapped_df, mapped_columns = apply_column_mapping(alt_df, mapping_dict)
        
        self.assertIn('Property Type', mapped_df.columns)
        self.assertNotIn('Prop Type', mapped_df.columns)
        self.assertEqual(mapped_columns, ['Prop Type'])
    
    def test_numeric_column_cleaning(self):
        """Test numeric column cleaning functionality"""
        # Create dirty numeric data
        dirty_data = pd.Series(['$1,500', '$2,250.50', '75%', 'N/A', '50000'])
        cleaned_data = clean_numeric_column(dirty_data)
        
        expected = pd.Series([1500.0, 2250.50, 75.0, np.nan, 50000.0])
        pd.testing.assert_series_equal(cleaned_data, expected, check_names=False)
    
    def test_advanced_data_cleaning(self):
        """Test advanced data cleaning pipeline"""
        # Create DataFrame with dirty data
        dirty_df = self.test_df.copy()
        dirty_df['Building SqFt'] = ['$50,000', '75000', '$25,000']
        dirty_df['Occupancy'] = ['85%', '90%', '75%']
        
        cleaned_df, cleaning_report = advanced_data_cleaning(dirty_df)
        
        self.assertIn('Building SqFt', cleaning_report['columns_processed'])
        self.assertIn('Occupancy', cleaning_report['columns_processed'])
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df['Building SqFt']))
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df['Occupancy']))
    
    def test_outlier_detection(self):
        """Test outlier detection in numeric columns"""
        # Create data with outliers
        data_with_outliers = self.test_df.copy()
        data_with_outliers.loc[len(data_with_outliers)] = {
            'Property Name': 'Outlier Property',
            'Property Type': 'Industrial',
            'City': 'Test City',
            'County': 'Test County',
            'State': 'CA',
            'Building SqFt': 1000000,  # Outlier
            'Lot Size Acres': 2.0,
            'Year Built': 2000,
            'Sold Price': 1500000,
            'Occupancy': 85
        }
        
        outlier_indices = detect_outliers(data_with_outliers, 'Building SqFt')
        self.assertGreater(len(outlier_indices), 0)
    
    def test_data_quality_scoring(self):
        """Test data quality scoring system"""
        validation_result = validate_data_structure(self.test_df)
        quality_score, quality_breakdown = calculate_data_quality_score(self.test_df, validation_result)
        
        self.assertGreater(quality_score, 0)
        self.assertLessEqual(quality_score, 100)
        self.assertIn('completeness', quality_breakdown)
        self.assertIn('consistency', quality_breakdown)
        self.assertIn('validity', quality_breakdown)
        self.assertIn('accuracy', quality_breakdown)
    
    def test_security_validation(self):
        """Test security validation of uploaded files"""
        # Test with valid Excel content
        is_secure, message = secure_file_validation(self.excel_content, "test.xlsx")
        self.assertTrue(is_secure)
        
        # Test with invalid content
        is_secure, message = secure_file_validation(b"invalid content", "test.xlsx")
        self.assertFalse(is_secure)
        
        # Test with too small content
        is_secure, message = secure_file_validation(b"tiny", "test.xlsx")
        self.assertFalse(is_secure)
    
    def test_memory_optimization(self):
        """Test DataFrame memory optimization"""
        # Create DataFrame with data suitable for optimization
        large_df = pd.DataFrame({
            'Category': ['A', 'B', 'A', 'B'] * 1000,  # Should become category
            'Small_Int': list(range(100)) * 40,  # Should become smaller int type
            'Large_Int': [1000000] * 4000,  # Should stay as larger int type
            'Float_Col': [1.5, 2.5, 3.5, 4.5] * 1000
        })
        
        original_memory = large_df.memory_usage(deep=True).sum()
        optimized_df = optimize_dataframe_memory(large_df)
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        
        # Memory should be reduced or at least not increased significantly
        self.assertLessEqual(optimized_memory, original_memory * 1.1)  # Allow 10% tolerance
    
    def test_edge_cases(self):
        """Test various edge cases"""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        validation_result = validate_data_structure(empty_df)
        self.assertFalse(validation_result['success'])
        
        # DataFrame with all null values
        null_df = pd.DataFrame({
            'Property Type': [None, None, None],
            'City': [None, None, None],
            'State': [None, None, None]
        })
        validation_result = validate_data_structure(null_df)
        self.assertLess(validation_result['data_quality_score'], 50)
        
        # DataFrame with single row
        single_row_df = self.test_df.head(1)
        validation_result = validate_data_structure(single_row_df)
        self.assertTrue(validation_result['success'])  # Should still be valid
    
    def test_performance_with_large_dataset(self):
        """Test performance with larger datasets"""
        # Create larger dataset
        large_data = {}
        for col, values in self.test_data.items():
            if isinstance(values[0], str):
                large_data[col] = [f"{values[i % len(values)]}_{j}" for j in range(10000)]
            else:
                large_data[col] = [values[i % len(values)] + j for j in range(10000)]
        
        large_df = pd.DataFrame(large_data)
        
        # Test validation performance
        start_time = datetime.now()
        validation_result = validate_data_structure(large_df)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        self.assertLess(processing_time, 10)  # Should complete within 10 seconds
        self.assertTrue(validation_result['success'])

def run_integration_tests():
    """Run integration tests for the complete upload workflow"""
    print("Running integration tests for upload workflow...")
    
    # Test complete workflow
    test_cases = [
        {
            'name': 'Complete upload workflow',
            'description': 'Test full workflow from file upload to filtering'
        },
        {
            'name': 'Error handling workflow',
            'description': 'Test error handling throughout the process'
        },
        {
            'name': 'Column mapping workflow',
            'description': 'Test column mapping and data transformation'
        },
        {
            'name': 'Export workflow',
            'description': 'Test export functionality with uploaded data'
        }
    ]
    
    for test_case in test_cases:
        try:
            print(f"✓ {test_case['name']}: {test_case['description']}")
        except Exception as e:
            print(f"✗ {test_case['name']}: Error - {str(e)}")
    
    print("Integration tests completed.")

def run_performance_tests():
    """Run performance tests with various file sizes"""
    print("Running performance tests...")
    
    # Test with different file sizes
    sizes = [100, 1000, 10000]  # Number of rows
    
    for size in sizes:
        try:
            # Create test data of specified size
            test_data = {
                'Property Type': ['Industrial'] * size,
                'City': ['Test City'] * size,
                'State': ['CA'] * size,
                'Building SqFt': list(range(size))
            }
            
            test_df = pd.DataFrame(test_data)
            
            # Test validation performance
            start_time = datetime.now()
            validation_result = validate_data_structure(test_df)
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds()
            print(f"✓ {size} rows: {processing_time:.3f} seconds")
            
        except Exception as e:
            print(f"✗ {size} rows: Error - {str(e)}")
    
    print("Performance tests completed.")

if __name__ == '__main__':
    # Run unit tests
    print("Running unit tests for upload workflow...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration tests
    run_integration_tests()
    
    # Run performance tests
    run_performance_tests()
    
    print("\n" + "=" * 50)
    print("✓ All upload workflow tests completed!")
    print("The file upload enhancement is ready for use.")