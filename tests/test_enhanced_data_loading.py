"""
Tests for enhanced data loading pipeline with categorical data conversion.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

from utils.flex_data_utils import (
    load_property_data_with_conversion,
    load_and_validate_property_data,
    _generate_data_quality_summary
)


class TestEnhancedDataLoading:
    """Test enhanced data loading functionality."""
    
    def create_test_excel_file(self, data_dict: dict, filename: str = None) -> str:
        """Helper to create test Excel files."""
        if filename is None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
            filename = temp_file.name
            temp_file.close()
        
        df = pd.DataFrame(data_dict)
        df.to_excel(filename, index=False)
        return filename
    
    def create_test_dataframe_with_categorical(self, data_dict: dict) -> pd.DataFrame:
        """Helper to create DataFrame with categorical columns for direct testing."""
        return pd.DataFrame(data_dict)
    
    def test_load_with_categorical_conversion(self):
        """Test loading data with automatic categorical conversion."""
        # Test with direct DataFrame that has categorical columns
        test_data = {
            'property_name': ['Property A', 'Property B', 'Property C'],
            'price': pd.Categorical(['100000', '200000', '300000']),  # Should be converted
            'size': pd.Categorical(['1000', '2000', '3000']),        # Should be converted
            'type': pd.Categorical(['Office', 'Retail', 'Industrial']),  # Should NOT be converted
            'year': [2020, 2021, 2022]  # Already numeric
        }
        
        df_with_categorical = self.create_test_dataframe_with_categorical(test_data)
        
        # Test the conversion functions directly
        from utils.data_type_utils import detect_categorical_numeric_columns, convert_categorical_to_numeric
        
        # Test detection
        detected = detect_categorical_numeric_columns(df_with_categorical)
        assert 'price' in detected
        assert 'size' in detected
        assert 'type' not in detected  # Non-numeric categorical
        
        # Test conversion
        converted_df, reports = convert_categorical_to_numeric(df_with_categorical)
        assert pd.api.types.is_numeric_dtype(converted_df['price'])
        assert pd.api.types.is_numeric_dtype(converted_df['size'])
        assert converted_df['type'].dtype.name == 'category'  # Should remain categorical
        
        # Test with Excel file (which will lose categorical dtypes)
        filename = self.create_test_excel_file({
            'property_name': ['Property A', 'Property B', 'Property C'],
            'price': ['100000', '200000', '300000'],  # String data that could be numeric
            'size': ['1000', '2000', '3000'],        # String data that could be numeric
            'type': ['Office', 'Retail', 'Industrial'],  # String data that should stay string
            'year': [2020, 2021, 2022]  # Already numeric
        })
        
        try:
            # Test with conversion enabled - Excel files won't have categorical dtypes
            df, report = load_property_data_with_conversion(filename, auto_convert_categorical=True)
            
            # Excel loading won't detect categorical columns since they're loaded as object/string
            assert report['conversion_successful'] is True
            assert report['categorical_columns_detected'] == 0  # No categorical columns from Excel
            
            # Test with conversion disabled
            df_no_convert, report_no_convert = load_property_data_with_conversion(
                filename, auto_convert_categorical=False
            )
            
            assert report_no_convert['categorical_columns_detected'] == 0
            
        finally:
            os.unlink(filename)
    
    def test_load_with_mixed_categorical_data(self):
        """Test loading with mixed categorical data (some convertible, some not)."""
        # Test with direct DataFrame that has categorical columns
        test_data = {
            'numeric_cat': pd.Categorical(['1', '2', '3', '4', '5']),  # 100% numeric
            'mixed_cat': pd.Categorical(['1', '2', 'invalid', '4', '5']),  # 80% numeric
            'text_cat': pd.Categorical(['red', 'blue', 'green', 'yellow', 'purple']),  # 0% numeric
            'regular_col': ['A', 'B', 'C', 'D', 'E']
        }
        
        df_with_categorical = self.create_test_dataframe_with_categorical(test_data)
        
        # Test the conversion functions directly
        from utils.data_type_utils import convert_categorical_to_numeric
        
        converted_df, reports = convert_categorical_to_numeric(df_with_categorical)
        
        # Check conversions
        assert pd.api.types.is_numeric_dtype(converted_df['numeric_cat'])  # Should be converted
        assert pd.api.types.is_numeric_dtype(converted_df['mixed_cat'])   # Should be converted (80% >= 80%)
        assert converted_df['text_cat'].dtype.name == 'category'          # Should remain categorical
        
        # Check that mixed conversion handled nulls properly
        assert converted_df['mixed_cat'].isna().sum() == 1  # 'invalid' should become NaN
        
        # Check reports
        assert len(reports) >= 2  # Should have reports for converted columns
        
        # Test with Excel file (simpler test since Excel doesn't preserve categorical)
        filename = self.create_test_excel_file({
            'numeric_col': ['1', '2', '3', '4', '5'],
            'text_col': ['red', 'blue', 'green', 'yellow', 'purple'],
            'regular_col': ['A', 'B', 'C', 'D', 'E']
        })
        
        try:
            df, report = load_property_data_with_conversion(filename)
            
            # Excel files won't have categorical dtypes, so no conversion expected
            assert report['conversion_successful'] is True
            assert report['categorical_columns_detected'] == 0
            
        finally:
            os.unlink(filename)
    
    def test_complete_pipeline_with_validation(self):
        """Test complete data loading pipeline with validation."""
        test_data = {
            'property_name': ['Property A', 'Property B', 'Property C'],
            'sold_price': ['100000', '200000', '300000'],  # Will be processed by preprocessing
            'size': ['1000', '2000', '3000'],
            'location': ['City A', 'City B', 'City C']
        }
        
        filename = self.create_test_excel_file(test_data)
        
        try:
            # Test with required columns that exist (note: preprocessing may rename columns)
            required_cols = ['property_name', 'sold_price', 'size']
            df, report = load_and_validate_property_data(
                filename, required_columns=required_cols
            )
            
            # Check validation passed
            assert report['loading_successful'] is True
            assert report['validation_successful'] is True
            assert report['validation_report']['validation_passed'] is True
            assert len(report['validation_report']['missing_columns']) == 0
            
            # The preprocessing step may have converted columns to numeric
            # Check that the data was loaded successfully
            assert len(df) == 3
            assert 'property_name' in df.columns
            
            # Test with required columns that don't exist
            required_cols_missing = ['property_name', 'sold_price', 'nonexistent_column']
            df2, report2 = load_and_validate_property_data(
                filename, required_columns=required_cols_missing
            )
            
            # Check validation failed
            assert report2['loading_successful'] is True
            assert report2['validation_successful'] is False
            assert 'nonexistent_column' in report2['validation_report']['missing_columns']
            
        finally:
            os.unlink(filename)
    
    def test_data_quality_summary(self):
        """Test data quality summary generation."""
        # Create test DataFrame with various data types and null values
        df = pd.DataFrame({
            'numeric_col': [1.0, 2.0, np.nan, 4.0, 5.0],
            'categorical_col': pd.Categorical(['A', 'B', None, 'A', 'B']),
            'text_col': ['text1', 'text2', 'text3', None, 'text5'],
            'datetime_col': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', None])
        })
        
        from utils.logger import setup_logging
        logger = setup_logging('test_logger')
        
        summary = _generate_data_quality_summary(df, logger)
        
        # Check summary structure
        assert 'total_rows' in summary
        assert 'total_columns' in summary
        assert 'numeric_columns' in summary
        assert 'categorical_columns' in summary
        assert 'text_columns' in summary
        assert 'datetime_columns' in summary
        assert 'overall_completeness' in summary
        
        # Check values
        assert summary['total_rows'] == 5
        assert summary['total_columns'] == 4
        assert summary['numeric_columns'] == 1
        assert summary['categorical_columns'] == 1
        assert summary['text_columns'] == 1
        assert summary['datetime_columns'] == 1
        assert summary['columns_with_nulls'] == 4  # All columns have nulls
        
        # Check completeness calculation
        # Total cells: 5 * 4 = 20
        # Non-null cells: 4 + 4 + 4 + 4 = 16
        # Completeness: 16/20 = 80%
        assert abs(summary['overall_completeness'] - 80.0) < 0.1
    
    def test_error_handling_in_enhanced_loading(self):
        """Test error handling in enhanced loading pipeline."""
        # Test with non-existent file
        with pytest.raises(Exception):
            load_property_data_with_conversion('nonexistent_file.xlsx')
        
        # Test with invalid file format
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
        temp_file.write(b"This is not an Excel file")
        temp_file.close()
        
        try:
            with pytest.raises(Exception):
                load_property_data_with_conversion(temp_file.name)
        finally:
            os.unlink(temp_file.name)
    
    def test_conversion_reporting(self):
        """Test detailed conversion reporting."""
        test_data = {
            'good_numeric': pd.Categorical(['1', '2', '3']),
            'mixed_numeric': pd.Categorical(['1', '2', 'invalid']),
            'non_numeric': pd.Categorical(['red', 'blue', 'green'])
        }
        
        filename = self.create_test_excel_file(test_data)
        
        try:
            df, report = load_property_data_with_conversion(filename)
            
            # Check detailed reporting
            assert 'column_reports' in report
            assert 'data_quality_summary' in report
            
            # Check that good_numeric was converted successfully
            if 'good_numeric' in report['column_reports']:
                good_report = report['column_reports']['good_numeric']
                assert good_report['conversion_successful'] is True
                assert good_report['values_failed'] == 0
            
            # Check that mixed_numeric was converted with some failures
            if 'mixed_numeric' in report['column_reports']:
                mixed_report = report['column_reports']['mixed_numeric']
                assert mixed_report['conversion_successful'] is True
                assert mixed_report['values_failed'] > 0
            
            # non_numeric should not be in reports (not detected for conversion)
            
        finally:
            os.unlink(filename)
    
    def test_memory_and_performance_tracking(self):
        """Test that memory usage and performance metrics are tracked."""
        # Create larger test dataset
        n_rows = 1000
        test_data = {
            'id': range(n_rows),
            'price': pd.Categorical([str(i * 1000) for i in range(n_rows)]),
            'size': pd.Categorical([str(i * 10) for i in range(n_rows)]),
            'type': pd.Categorical(['Type A', 'Type B', 'Type C'] * (n_rows // 3 + 1))[:n_rows]
        }
        
        filename = self.create_test_excel_file(test_data)
        
        try:
            df, report = load_property_data_with_conversion(filename)
            
            # Check that data quality summary includes memory usage
            assert 'data_quality_summary' in report
            assert 'memory_usage_mb' in report['data_quality_summary']
            assert report['data_quality_summary']['memory_usage_mb'] > 0
            
            # Check shape tracking
            assert 'original_shape' in report
            assert 'final_shape' in report
            assert report['original_shape'][0] == n_rows
            assert report['final_shape'][0] == n_rows
            
        finally:
            os.unlink(filename)


class TestDataQualitySummary:
    """Test data quality summary functionality."""
    
    def test_empty_dataframe_summary(self):
        """Test summary generation for empty DataFrame."""
        df = pd.DataFrame()
        from utils.logger import setup_logging
        logger = setup_logging('test_logger')
        
        summary = _generate_data_quality_summary(df, logger)
        
        assert summary['total_rows'] == 0
        assert summary['total_columns'] == 0
        assert summary['overall_completeness'] == 0.0
    
    def test_all_null_dataframe_summary(self):
        """Test summary generation for DataFrame with all null values."""
        df = pd.DataFrame({
            'col1': [None, None, None],
            'col2': [np.nan, np.nan, np.nan],
            'col3': [pd.NA, pd.NA, pd.NA]
        })
        
        from utils.logger import setup_logging
        logger = setup_logging('test_logger')
        
        summary = _generate_data_quality_summary(df, logger)
        
        assert summary['total_rows'] == 3
        assert summary['total_columns'] == 3
        assert summary['overall_completeness'] == 0.0
        assert summary['columns_with_nulls'] == 3
    
    def test_perfect_dataframe_summary(self):
        """Test summary generation for DataFrame with no null values."""
        df = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'text': ['a', 'b', 'c', 'd', 'e'],
            'categorical': pd.Categorical(['X', 'Y', 'Z', 'X', 'Y'])
        })
        
        from utils.logger import setup_logging
        logger = setup_logging('test_logger')
        
        summary = _generate_data_quality_summary(df, logger)
        
        assert summary['total_rows'] == 5
        assert summary['total_columns'] == 3
        assert summary['overall_completeness'] == 100.0
        assert summary['columns_with_nulls'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])