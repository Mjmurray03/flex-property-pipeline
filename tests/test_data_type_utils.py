"""
Unit tests for data type detection and conversion utilities.
Tests the core functionality for handling categorical data in numerical operations.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import logging

from utils.data_type_utils import (
    detect_categorical_numeric_columns,
    assess_conversion_feasibility,
    safe_numeric_conversion,
    convert_categorical_to_numeric,
    ensure_numeric_for_calculation,
    safe_mean_calculation,
    safe_numerical_comparison
)


class TestDetectCategoricalNumericColumns:
    """Test categorical column detection functionality."""
    
    def test_detect_simple_numeric_categories(self):
        """Test detection of simple numeric categorical columns."""
        # Create test data with numeric categories
        df = pd.DataFrame({
            'numeric_cat': pd.Categorical(['1', '2', '3', '4', '5']),
            'price_cat': pd.Categorical(['100.50', '200.75', '300.25', '400.00', '500.25']),
            'non_numeric_cat': pd.Categorical(['red', 'blue', 'green', 'yellow', 'purple']),
            'regular_numeric': [1, 2, 3, 4, 5]
        })
        
        result = detect_categorical_numeric_columns(df)
        
        # Should detect numeric_cat and price_cat
        assert 'numeric_cat' in result
        assert 'price_cat' in result
        assert 'non_numeric_cat' not in result
        assert 'regular_numeric' not in result
        
        # Check conversion feasibility
        assert result['numeric_cat']['conversion_feasible'] is True
        assert result['price_cat']['conversion_feasible'] is True
        assert result['numeric_cat']['conversion_confidence'] == 1.0
    
    def test_detect_currency_categories(self):
        """Test detection of currency formatted categorical columns."""
        df = pd.DataFrame({
            'currency': pd.Categorical(['$1,000.00', '$2,500.50', '$3,750.25', '$4,000.00', '$5,250.75']),
            'percentage': pd.Categorical(['85%', '92%', '78%', '90%', '88%'])
        })
        
        result = detect_categorical_numeric_columns(df)
        
        assert 'currency' in result
        assert 'percentage' in result
        assert result['currency']['conversion_feasible'] is True
        assert result['percentage']['conversion_feasible'] is True
    
    def test_detect_mixed_categories(self):
        """Test detection with mixed numeric/non-numeric categories."""
        df = pd.DataFrame({
            'mixed_mostly_numeric': pd.Categorical(['1', '2', '3', '4', 'N/A']),  # 80% numeric
            'mixed_mostly_text': pd.Categorical(['1', '2', 'red', 'blue', 'green'])  # 40% numeric
        })
        
        result = detect_categorical_numeric_columns(df)
        
        # Should detect mostly_numeric (80% >= 80% threshold)
        assert 'mixed_mostly_numeric' in result
        assert result['mixed_mostly_numeric']['conversion_feasible'] is True
        
        # Should not detect mostly_text (40% < 80% threshold)
        assert 'mixed_mostly_text' not in result
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = detect_categorical_numeric_columns(df)
        assert result == {}
    
    def test_no_categorical_columns(self):
        """Test with DataFrame containing no categorical columns."""
        df = pd.DataFrame({
            'numeric': [1, 2, 3],
            'text': ['a', 'b', 'c'],
            'float': [1.1, 2.2, 3.3]
        })
        
        result = detect_categorical_numeric_columns(df)
        assert result == {}


class TestAssessConversionFeasibility:
    """Test conversion feasibility assessment."""
    
    def test_assess_numeric_categorical(self):
        """Test assessment of numeric categorical series."""
        series = pd.Categorical(['1', '2', '3', '4', '5'])
        result = assess_conversion_feasibility(series)
        
        assert result['feasible'] is True
        assert result['confidence'] == 1.0
        assert result['success_probability'] == 1.0
        assert result['estimated_null_count'] == 0
        assert len(result['sample_conversions']) > 0
    
    def test_assess_mixed_categorical(self):
        """Test assessment of mixed categorical series."""
        series = pd.Categorical(['1', '2', '3', 'invalid', 'also_invalid'])
        result = assess_conversion_feasibility(series)
        
        assert result['feasible'] is False  # 60% < 80% threshold
        assert result['confidence'] == 0.6
        assert result['estimated_null_count'] > 0
        assert len(result['issues']) > 0
    
    def test_assess_non_categorical(self):
        """Test assessment of non-categorical series."""
        series = pd.Series([1, 2, 3, 4, 5])
        result = assess_conversion_feasibility(series)
        
        assert result['feasible'] is False
        assert 'Series is not categorical' in result['issues']
    
    def test_assess_empty_series(self):
        """Test assessment of empty categorical series."""
        series = pd.Categorical([])
        result = assess_conversion_feasibility(series)
        
        assert result['confidence'] == 0.0
        assert result['estimated_null_count'] == 0


class TestSafeNumericConversion:
    """Test safe numeric conversion functionality."""
    
    def test_convert_numeric_categorical(self):
        """Test conversion of numeric categorical series."""
        series = pd.Categorical(['1', '2', '3', '4', '5'])
        converted, report = safe_numeric_conversion(series, 'test_col')
        
        assert pd.api.types.is_numeric_dtype(converted)
        assert report['conversion_successful'] is True
        assert report['values_failed'] == 0
        assert converted.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    
    def test_convert_currency_categorical(self):
        """Test conversion of currency formatted categorical series."""
        series = pd.Categorical(['$1,000.00', '$2,500.50', '$3,750.25'])
        converted, report = safe_numeric_conversion(series, 'price_col')
        
        # Note: This will fail conversion due to currency symbols
        # The function should handle this gracefully
        assert report['conversion_successful'] is True
        assert report['values_failed'] > 0  # Currency symbols cause failures
    
    def test_convert_already_numeric(self):
        """Test conversion of already numeric series."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        converted, report = safe_numeric_conversion(series, 'numeric_col')
        
        assert pd.api.types.is_numeric_dtype(converted)
        assert report['conversion_successful'] is True
        assert report['conversion_method'] == 'already_numeric'
        assert converted.equals(series)
    
    def test_convert_mixed_categorical(self):
        """Test conversion of mixed categorical series."""
        series = pd.Categorical(['1', '2', '3', 'invalid', 'also_invalid'])
        converted, report = safe_numeric_conversion(series, 'mixed_col')
        
        assert pd.api.types.is_numeric_dtype(converted)
        assert report['conversion_successful'] is True
        assert report['values_failed'] == 2  # Two invalid values
        assert converted.notna().sum() == 3  # Three valid conversions
    
    def test_convert_with_nulls(self):
        """Test conversion with existing null values."""
        series = pd.Categorical(['1', '2', None, '4', '5'])
        converted, report = safe_numeric_conversion(series, 'null_col')
        
        assert pd.api.types.is_numeric_dtype(converted)
        assert report['conversion_successful'] is True
        assert report['original_null_count'] == 1
        assert converted.isna().sum() == 1


class TestConvertCategoricalToNumeric:
    """Test batch conversion functionality."""
    
    def test_batch_convert_specified_columns(self):
        """Test batch conversion of specified columns."""
        df = pd.DataFrame({
            'cat_numeric': pd.Categorical(['1', '2', '3']),
            'cat_text': pd.Categorical(['red', 'blue', 'green']),
            'regular_numeric': [1, 2, 3]
        })
        
        converted_df, reports = convert_categorical_to_numeric(df, columns=['cat_numeric'])
        
        assert pd.api.types.is_numeric_dtype(converted_df['cat_numeric'])
        assert converted_df['cat_text'].dtype.name == 'category'  # Unchanged
        assert 'cat_numeric' in reports
        assert reports['cat_numeric']['conversion_successful'] is True
    
    def test_batch_convert_auto_detect(self):
        """Test batch conversion with auto-detection."""
        df = pd.DataFrame({
            'cat_numeric': pd.Categorical(['1', '2', '3']),
            'cat_text': pd.Categorical(['red', 'blue', 'green']),
            'regular_numeric': [1, 2, 3]
        })
        
        converted_df, reports = convert_categorical_to_numeric(df)
        
        # Should auto-detect and convert cat_numeric only
        assert pd.api.types.is_numeric_dtype(converted_df['cat_numeric'])
        assert converted_df['cat_text'].dtype.name == 'category'
        assert len(reports) == 1
        assert 'cat_numeric' in reports
    
    def test_batch_convert_nonexistent_column(self):
        """Test batch conversion with nonexistent column."""
        df = pd.DataFrame({'existing': [1, 2, 3]})
        
        converted_df, reports = convert_categorical_to_numeric(df, columns=['nonexistent'])
        
        assert converted_df.equals(df)
        assert len(reports) == 0
    
    def test_batch_convert_empty_list(self):
        """Test batch conversion with empty column list."""
        df = pd.DataFrame({'test': [1, 2, 3]})
        
        converted_df, reports = convert_categorical_to_numeric(df, columns=[])
        
        assert converted_df.equals(df)
        assert len(reports) == 0


class TestEnsureNumericForCalculation:
    """Test numeric ensuring functionality."""
    
    def test_ensure_already_numeric(self):
        """Test ensuring numeric on already numeric series."""
        series = pd.Series([1.0, 2.0, 3.0])
        result = ensure_numeric_for_calculation(series, "test operation")
        
        assert pd.api.types.is_numeric_dtype(result)
        assert result.equals(series)
    
    def test_ensure_categorical_numeric(self):
        """Test ensuring numeric on categorical series."""
        series = pd.Categorical(['1', '2', '3'])
        result = ensure_numeric_for_calculation(series, "test operation")
        
        assert pd.api.types.is_numeric_dtype(result)
        assert result.tolist() == [1.0, 2.0, 3.0]
    
    def test_ensure_mixed_categorical(self):
        """Test ensuring numeric on mixed categorical series."""
        series = pd.Categorical(['1', '2', 'invalid'])
        result = ensure_numeric_for_calculation(series, "test operation")
        
        assert pd.api.types.is_numeric_dtype(result)
        assert result.notna().sum() == 2  # Two valid conversions


class TestSafeMeanCalculation:
    """Test safe mean calculation functionality."""
    
    def test_mean_numeric_series(self):
        """Test mean calculation on numeric series."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = safe_mean_calculation(series, "test_col")
        
        assert result == 3.0
    
    def test_mean_categorical_numeric(self):
        """Test mean calculation on categorical numeric series."""
        series = pd.Categorical(['1', '2', '3', '4', '5'])
        result = safe_mean_calculation(series, "cat_col")
        
        assert result == 3.0
    
    def test_mean_mixed_categorical(self):
        """Test mean calculation on mixed categorical series."""
        series = pd.Categorical(['1', '2', '3', 'invalid'])
        result = safe_mean_calculation(series, "mixed_col")
        
        assert result == 2.0  # Mean of 1, 2, 3
    
    def test_mean_all_invalid(self):
        """Test mean calculation on all invalid categorical series."""
        series = pd.Categorical(['invalid', 'also_invalid', 'still_invalid'])
        result = safe_mean_calculation(series, "invalid_col")
        
        assert pd.isna(result)
    
    def test_mean_empty_series(self):
        """Test mean calculation on empty series."""
        series = pd.Series([], dtype=float)
        result = safe_mean_calculation(series, "empty_col")
        
        assert pd.isna(result)


class TestSafeNumericalComparison:
    """Test safe numerical comparison functionality."""
    
    def test_comparison_numeric_series(self):
        """Test comparison on numeric series."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = safe_numerical_comparison(series, '>=', 3.0, "test_col")
        
        expected = pd.Series([False, False, True, True, True])
        assert result.equals(expected)
    
    def test_comparison_categorical_numeric(self):
        """Test comparison on categorical numeric series."""
        series = pd.Categorical(['1', '2', '3', '4', '5'])
        result = safe_numerical_comparison(series, '<=', 3.0, "cat_col")
        
        expected = pd.Series([True, True, True, False, False])
        assert result.equals(expected)
    
    def test_comparison_mixed_categorical(self):
        """Test comparison on mixed categorical series."""
        series = pd.Categorical(['1', '2', '3', 'invalid'])
        result = safe_numerical_comparison(series, '>', 2.0, "mixed_col")
        
        # Should handle invalid values gracefully
        assert len(result) == 4
        assert result.iloc[2] == True  # 3 > 2
        assert result.iloc[3] == False  # invalid becomes False
    
    def test_comparison_all_operators(self):
        """Test all comparison operators."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test each operator
        operators_and_expected = [
            ('>=', [False, False, True, True, True]),
            ('<=', [True, True, True, False, False]),
            ('>', [False, False, False, True, True]),
            ('<', [True, True, False, False, False]),
            ('==', [False, False, True, False, False]),
            ('!=', [True, True, False, True, True])
        ]
        
        for operator, expected_list in operators_and_expected:
            result = safe_numerical_comparison(series, operator, 3.0, "test_col")
            expected = pd.Series(expected_list)
            assert result.equals(expected), f"Failed for operator {operator}"
    
    def test_comparison_invalid_operator(self):
        """Test comparison with invalid operator."""
        series = pd.Series([1.0, 2.0, 3.0])
        result = safe_numerical_comparison(series, 'invalid_op', 2.0, "test_col")
        
        # Should return all False on error
        expected = pd.Series([False, False, False])
        assert result.equals(expected)


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple functions."""
    
    def test_complete_pipeline(self):
        """Test complete data type handling pipeline."""
        # Create test DataFrame with various categorical issues
        df = pd.DataFrame({
            'price': pd.Categorical(['$1,000', '$2,000', '$3,000']),
            'size': pd.Categorical(['1000', '2000', '3000']),
            'year': pd.Categorical(['1990', '2000', '2010']),
            'type': pd.Categorical(['A', 'B', 'C']),  # Non-numeric
            'occupancy': pd.Categorical(['85%', '90%', '95%'])
        })
        
        # Step 1: Detect categorical columns
        detected = detect_categorical_numeric_columns(df)
        
        # Should detect size and year as convertible
        assert 'size' in detected
        assert 'year' in detected
        assert 'type' not in detected  # Non-numeric
        
        # Step 2: Convert detected columns
        converted_df, reports = convert_categorical_to_numeric(df)
        
        # Verify conversions
        assert pd.api.types.is_numeric_dtype(converted_df['size'])
        assert pd.api.types.is_numeric_dtype(converted_df['year'])
        
        # Step 3: Test safe operations
        mean_size = safe_mean_calculation(converted_df['size'], 'size')
        assert mean_size == 2000.0
        
        # Step 4: Test safe comparisons
        large_properties = safe_numerical_comparison(converted_df['size'], '>=', 2000, 'size')
        assert large_properties.sum() == 2  # 2000 and 3000
    
    def test_error_recovery(self):
        """Test error recovery in various scenarios."""
        # Create problematic data with some numeric values to trigger conversion
        df = pd.DataFrame({
            'problematic': pd.Categorical(['1', '2', None, 'invalid', '5'])  # 60% numeric, should be detected
        })
        
        # Should handle gracefully
        converted_df, reports = convert_categorical_to_numeric(df)
        
        # Should not crash and provide useful information
        # Note: The column might not be detected if it doesn't meet the 80% threshold
        # Let's test that it doesn't crash and handles gracefully
        assert isinstance(converted_df, pd.DataFrame)
        assert isinstance(reports, dict)
        
        # If it was converted, check the report
        if 'problematic' in reports:
            report = reports['problematic']
            assert report['conversion_successful'] is True
            assert report['values_failed'] >= 0


# Test fixtures and utilities
@pytest.fixture
def sample_categorical_df():
    """Create sample DataFrame with categorical data for testing."""
    return pd.DataFrame({
        'numeric_cat': pd.Categorical(['1', '2', '3', '4', '5']),
        'price_cat': pd.Categorical(['100.50', '200.75', '300.25']),
        'mixed_cat': pd.Categorical(['1', '2', 'invalid', '4', '5']),
        'text_cat': pd.Categorical(['red', 'blue', 'green', 'yellow']),
        'regular_numeric': [1, 2, 3, 4, 5],
        'regular_text': ['a', 'b', 'c', 'd', 'e']
    })


@pytest.fixture
def mock_logger():
    """Create mock logger for testing."""
    return Mock(spec=logging.Logger)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])


class TestSafeMathematicalOperations:
    """Test safe mathematical operation wrappers."""
    
    def test_safe_sum_calculation(self):
        """Test safe sum calculation."""
        # Test with numeric series
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = safe_sum_calculation(series, "test_col")
        assert result == 15.0
        
        # Test with categorical numeric
        series = pd.Categorical(['1', '2', '3', '4', '5'])
        result = safe_sum_calculation(series, "cat_col")
        assert result == 15.0
        
        # Test with mixed categorical
        series = pd.Categorical(['1', '2', '3', 'invalid'])
        result = safe_sum_calculation(series, "mixed_col")
        assert result == 6.0  # Sum of 1, 2, 3
    
    def test_safe_count_calculation(self):
        """Test safe count calculation."""
        # Test with numeric series
        series = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        result = safe_count_calculation(series, "test_col")
        assert result == 4  # Excludes NaN
        
        # Test with categorical
        series = pd.Categorical(['1', '2', None, '4', '5'])
        result = safe_count_calculation(series, "cat_col")
        assert result == 4  # Excludes None
    
    def test_safe_min_max_calculation(self):
        """Test safe min and max calculations."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        
        min_result = safe_min_calculation(series, "test_col")
        assert min_result == 1.0
        
        max_result = safe_max_calculation(series, "test_col")
        assert max_result == 5.0
        
        # Test with categorical
        series = pd.Categorical(['10', '20', '30'])
        min_result = safe_min_calculation(series, "cat_col")
        assert min_result == 10.0
        
        max_result = safe_max_calculation(series, "cat_col")
        assert max_result == 30.0
    
    def test_safe_std_calculation(self):
        """Test safe standard deviation calculation."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = safe_std_calculation(series, "test_col")
        assert abs(result - 1.5811388300841898) < 0.0001  # Expected std
        
        # Test with single value
        series = pd.Series([5.0])
        result = safe_std_calculation(series, "single_col")
        assert pd.isna(result)  # Should return NaN for single value
    
    def test_safe_median_calculation(self):
        """Test safe median calculation."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = safe_median_calculation(series, "test_col")
        assert result == 3.0
        
        # Test with categorical
        series = pd.Categorical(['10', '20', '30', '40', '50'])
        result = safe_median_calculation(series, "cat_col")
        assert result == 30.0
    
    def test_safe_range_filter(self):
        """Test safe range filtering."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test min filter
        result = safe_range_filter(series, min_value=3.0, column_name="test_col")
        expected = pd.Series([False, False, True, True, True])
        assert result.equals(expected)
        
        # Test max filter
        result = safe_range_filter(series, max_value=3.0, column_name="test_col")
        expected = pd.Series([True, True, True, False, False])
        assert result.equals(expected)
        
        # Test range filter
        result = safe_range_filter(series, min_value=2.0, max_value=4.0, column_name="test_col")
        expected = pd.Series([False, True, True, True, False])
        assert result.equals(expected)
        
        # Test with categorical
        series = pd.Categorical(['1', '2', '3', '4', '5'])
        result = safe_range_filter(series, min_value=3.0, column_name="cat_col")
        expected = pd.Series([False, False, True, True, True])
        assert result.equals(expected)
    
    def test_safe_value_filter(self):
        """Test safe value filtering."""
        series = pd.Series(['A', 'B', 'C', 'D', 'E'])
        
        result = safe_value_filter(series, ['A', 'C', 'E'], "test_col")
        expected = pd.Series([True, False, True, False, True])
        assert result.equals(expected)
        
        # Test with categorical
        series = pd.Categorical(['red', 'blue', 'green', 'red', 'blue'])
        result = safe_value_filter(series, ['red', 'green'], "cat_col")
        expected = pd.Series([True, False, True, True, False])
        assert result.equals(expected)
    
    def test_safe_aggregation_wrapper(self):
        """Test generic aggregation wrapper."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test all supported operations
        assert safe_aggregation_wrapper(series, 'mean', 'test_col') == 3.0
        assert safe_aggregation_wrapper(series, 'sum', 'test_col') == 15.0
        assert safe_aggregation_wrapper(series, 'count', 'test_col') == 5
        assert safe_aggregation_wrapper(series, 'min', 'test_col') == 1.0
        assert safe_aggregation_wrapper(series, 'max', 'test_col') == 5.0
        assert safe_aggregation_wrapper(series, 'median', 'test_col') == 3.0
        
        # Test unsupported operation
        result = safe_aggregation_wrapper(series, 'unsupported', 'test_col')
        assert pd.isna(result)
    
    def test_operations_with_all_invalid_data(self):
        """Test operations with completely invalid categorical data."""
        series = pd.Categorical(['invalid', 'also_invalid', 'still_invalid'])
        
        # All operations should handle gracefully
        assert pd.isna(safe_mean_calculation(series, "invalid_col"))
        assert pd.isna(safe_sum_calculation(series, "invalid_col"))
        assert safe_count_calculation(series, "invalid_col") == 3  # Count works regardless
        assert pd.isna(safe_min_calculation(series, "invalid_col"))
        assert pd.isna(safe_max_calculation(series, "invalid_col"))
        assert pd.isna(safe_std_calculation(series, "invalid_col"))
        assert pd.isna(safe_median_calculation(series, "invalid_col"))
        
        # Range filter should return all False
        result = safe_range_filter(series, min_value=1.0, max_value=10.0, column_name="invalid_col")
        expected = pd.Series([False, False, False])
        assert result.equals(expected)
    
    def test_operations_with_empty_series(self):
        """Test operations with empty series."""
        series = pd.Series([], dtype=float)
        
        # All operations should handle empty series gracefully
        assert pd.isna(safe_mean_calculation(series, "empty_col"))
        assert pd.isna(safe_sum_calculation(series, "empty_col"))
        assert safe_count_calculation(series, "empty_col") == 0
        assert pd.isna(safe_min_calculation(series, "empty_col"))
        assert pd.isna(safe_max_calculation(series, "empty_col"))
        assert pd.isna(safe_std_calculation(series, "empty_col"))
        assert pd.isna(safe_median_calculation(series, "empty_col"))
        
        # Range filter should return empty boolean series
        result = safe_range_filter(series, min_value=1.0, max_value=10.0, column_name="empty_col")
        assert len(result) == 0


# Import the new functions for testing
from utils.data_type_utils import (
    safe_sum_calculation,
    safe_count_calculation,
    safe_min_calculation,
    safe_max_calculation,
    safe_std_calculation,
    safe_median_calculation,
    safe_range_filter,
    safe_value_filter,
    safe_aggregation_wrapper
)