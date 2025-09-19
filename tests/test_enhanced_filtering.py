"""
Tests for enhanced filtering utilities with categorical data handling.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import streamlit as st

from utils.enhanced_filtering import (
    safe_price_filter,
    safe_size_filter,
    safe_year_filter,
    safe_lot_size_filter,
    safe_occupancy_filter,
    apply_enhanced_filters,
    calculate_safe_metrics,
    preprocess_dataframe_for_filtering,
    validate_filter_data_types
)


class TestSafeFiltering:
    """Test safe filtering functions."""
    
    def test_safe_price_filter_numeric(self):
        """Test price filtering with numeric data."""
        df = pd.DataFrame({
            'Property Name': ['A', 'B', 'C', 'D', 'E'],
            'Sold Price': [100000, 200000, 300000, 400000, 500000]
        })
        
        # Test price range filter
        filtered = safe_price_filter(df, (200000, 400000))
        
        assert len(filtered) == 3  # Properties B, C, D
        assert filtered['Sold Price'].min() >= 200000
        assert filtered['Sold Price'].max() <= 400000
    
    def test_safe_price_filter_categorical(self):
        """Test price filtering with categorical data."""
        df = pd.DataFrame({
            'Property Name': ['A', 'B', 'C', 'D', 'E'],
            'Sold Price': pd.Categorical(['100000', '200000', '300000', '400000', '500000'])
        })
        
        # Test price range filter
        filtered = safe_price_filter(df, (200000, 400000))
        
        assert len(filtered) == 3  # Properties B, C, D
    
    def test_safe_price_filter_mixed_categorical(self):
        """Test price filtering with mixed categorical data."""
        df = pd.DataFrame({
            'Property Name': ['A', 'B', 'C', 'D', 'E'],
            'Sold Price': pd.Categorical(['100000', '200000', 'invalid', '400000', '500000'])
        })
        
        # Test price range filter - should handle invalid values gracefully
        filtered = safe_price_filter(df, (200000, 400000))
        
        # Should include B and D (C has invalid price)
        assert len(filtered) == 2
    
    def test_safe_size_filter(self):
        """Test building size filtering."""
        df = pd.DataFrame({
            'Property Name': ['A', 'B', 'C', 'D', 'E'],
            'Building SqFt': pd.Categorical(['10000', '20000', '30000', '40000', '50000'])
        })
        
        filtered = safe_size_filter(df, (20000, 40000))
        
        assert len(filtered) == 3  # Properties B, C, D
    
    def test_safe_year_filter(self):
        """Test year built filtering."""
        df = pd.DataFrame({
            'Property Name': ['A', 'B', 'C', 'D', 'E'],
            'Year Built': pd.Categorical(['1990', '2000', '2010', '2020', '2025'])
        })
        
        filtered = safe_year_filter(df, (2000, 2020))
        
        assert len(filtered) == 3  # Properties B, C, D
    
    def test_safe_lot_size_filter(self):
        """Test lot size filtering."""
        df = pd.DataFrame({
            'Property Name': ['A', 'B', 'C', 'D', 'E'],
            'Lot Size Acres': pd.Categorical(['1.0', '2.0', '3.0', '4.0', '5.0'])
        })
        
        filtered = safe_lot_size_filter(df, (2.0, 4.0))
        
        assert len(filtered) == 3  # Properties B, C, D
    
    def test_safe_occupancy_filter(self):
        """Test occupancy filtering."""
        df = pd.DataFrame({
            'Property Name': ['A', 'B', 'C', 'D', 'E'],
            'Occupancy': pd.Categorical(['50', '60', '70', '80', '90'])
        })
        
        filtered = safe_occupancy_filter(df, (60, 80))
        
        assert len(filtered) == 3  # Properties B, C, D
    
    def test_missing_column_handling(self):
        """Test handling of missing columns."""
        df = pd.DataFrame({
            'Property Name': ['A', 'B', 'C']
        })
        
        # Should return original DataFrame when column is missing
        filtered = safe_price_filter(df, (100000, 500000))
        assert len(filtered) == len(df)
        
        filtered = safe_size_filter(df, (10000, 50000))
        assert len(filtered) == len(df)


class TestEnhancedFiltering:
    """Test comprehensive enhanced filtering."""
    
    def create_test_dataframe(self):
        """Create test DataFrame with mixed data types."""
        return pd.DataFrame({
            'Property Name': ['Prop A', 'Prop B', 'Prop C', 'Prop D', 'Prop E'],
            'Sold Price': pd.Categorical(['100000', '200000', '300000', '400000', '500000']),
            'Building SqFt': pd.Categorical(['10000', '20000', '30000', '40000', '50000']),
            'Year Built': pd.Categorical(['1990', '2000', '2010', '2020', '2025']),
            'Lot Size Acres': pd.Categorical(['1.0', '2.0', '3.0', '4.0', '5.0']),
            'Occupancy': pd.Categorical(['50', '60', '70', '80', '90']),
            'County': ['County A', 'County B', 'County A', 'County C', 'County B'],
            'State': ['FL', 'FL', 'GA', 'FL', 'GA'],
            'Property Type': ['Industrial', 'Warehouse', 'Office', 'Flex', 'Industrial']
        })
    
    def test_apply_enhanced_filters_comprehensive(self):
        """Test applying multiple filters together."""
        df = self.create_test_dataframe()
        
        filter_params = {
            'size_range': (20000, 40000),
            'use_price_filter': True,
            'price_range': (200000, 400000),
            'use_year_filter': True,
            'year_range': (2000, 2020),
            'selected_counties': ['County A', 'County B'],
            'industrial_keywords': ['Industrial', 'Warehouse', 'Flex']
        }
        
        filtered = apply_enhanced_filters(df, filter_params)
        
        # Should apply all filters and return matching properties
        assert len(filtered) <= len(df)
        assert isinstance(filtered, pd.DataFrame)
    
    def test_apply_enhanced_filters_no_filters(self):
        """Test applying filters with no filter parameters."""
        df = self.create_test_dataframe()
        
        filter_params = {}
        
        filtered = apply_enhanced_filters(df, filter_params)
        
        # Should return original DataFrame
        assert len(filtered) == len(df)
    
    def test_apply_enhanced_filters_partial(self):
        """Test applying only some filters."""
        df = self.create_test_dataframe()
        
        filter_params = {
            'use_price_filter': True,
            'price_range': (200000, 400000)
        }
        
        filtered = apply_enhanced_filters(df, filter_params)
        
        # Should apply only price filter
        assert len(filtered) == 3  # Properties B, C, D


class TestSafeMetrics:
    """Test safe metrics calculation."""
    
    def test_calculate_safe_metrics_numeric(self):
        """Test metrics calculation with numeric data."""
        df = pd.DataFrame({
            'Sold Price': [100000, 200000, 300000, 400000, 500000],
            'Building SqFt': [10000, 20000, 30000, 40000, 50000],
            'Lot Size Acres': [1.0, 2.0, 3.0, 4.0, 5.0],
            'Occupancy': [50, 60, 70, 80, 90],
            'City': ['City A', 'City B', 'City A', 'City C', 'City B'],
            'County': ['County A', 'County B', 'County A', 'County C', 'County B'],
            'State': ['FL', 'FL', 'GA', 'FL', 'GA']
        })
        
        metrics = calculate_safe_metrics(df)
        
        assert metrics['total_properties'] == 5
        assert metrics['avg_price'] == 300000.0
        assert metrics['avg_size'] == 30000.0
        assert metrics['avg_lot_size'] == 3.0
        assert metrics['avg_occupancy'] == 70.0
        assert metrics['unique_cities'] == 3
        assert metrics['unique_counties'] == 3
        assert metrics['unique_states'] == 2
    
    def test_calculate_safe_metrics_categorical(self):
        """Test metrics calculation with categorical data."""
        df = pd.DataFrame({
            'Sold Price': pd.Categorical(['100000', '200000', '300000', '400000', '500000']),
            'Building SqFt': pd.Categorical(['10000', '20000', '30000', '40000', '50000']),
            'City': ['City A', 'City B', 'City A', 'City C', 'City B']
        })
        
        metrics = calculate_safe_metrics(df)
        
        assert metrics['total_properties'] == 5
        assert metrics['avg_price'] == 300000.0
        assert metrics['avg_size'] == 30000.0
        assert metrics['unique_cities'] == 3
    
    def test_calculate_safe_metrics_mixed_categorical(self):
        """Test metrics calculation with mixed categorical data."""
        df = pd.DataFrame({
            'Sold Price': pd.Categorical(['100000', '200000', 'invalid', '400000', '500000']),
            'Building SqFt': pd.Categorical(['10000', '20000', '30000', 'invalid', '50000']),
            'City': ['City A', 'City B', 'City A', 'City C', 'City B']
        })
        
        metrics = calculate_safe_metrics(df)
        
        assert metrics['total_properties'] == 5
        # Should calculate mean excluding invalid values
        assert metrics['avg_price'] == 300000.0  # (100000 + 200000 + 400000 + 500000) / 4
        assert metrics['avg_size'] == 27500.0    # (10000 + 20000 + 30000 + 50000) / 4
    
    def test_calculate_safe_metrics_missing_columns(self):
        """Test metrics calculation with missing columns."""
        df = pd.DataFrame({
            'Property Name': ['A', 'B', 'C'],
            'City': ['City A', 'City B', 'City A']
        })
        
        metrics = calculate_safe_metrics(df)
        
        assert metrics['total_properties'] == 3
        assert metrics['avg_price'] is None
        assert metrics['avg_size'] is None
        assert metrics['unique_cities'] == 2


class TestDataPreprocessing:
    """Test data preprocessing for filtering."""
    
    @patch('streamlit.info')
    @patch('streamlit.success')
    @patch('streamlit.warning')
    def test_preprocess_dataframe_for_filtering(self, mock_warning, mock_success, mock_info):
        """Test DataFrame preprocessing."""
        df = pd.DataFrame({
            'Property Name': ['A', 'B', 'C'],
            'Sold Price': pd.Categorical(['100000', '200000', '300000']),
            'Building SqFt': pd.Categorical(['10000', '20000', '30000']),
            'Property Type': pd.Categorical(['Industrial', 'Warehouse', 'Office'])
        })
        
        processed_df = preprocess_dataframe_for_filtering(df)
        
        # Should convert numeric categorical columns
        assert pd.api.types.is_numeric_dtype(processed_df['Sold Price'])
        assert pd.api.types.is_numeric_dtype(processed_df['Building SqFt'])
        # Should leave non-numeric categorical columns as-is
        assert processed_df['Property Type'].dtype.name == 'category'
    
    def test_validate_filter_data_types(self):
        """Test data type validation for filtering."""
        df = pd.DataFrame({
            'Sold Price': pd.Categorical(['100000', '200000', '300000']),  # Non-numeric
            'Building SqFt': [10000, 20000, 30000],  # Numeric
            'Year Built': pd.Categorical(['1990', '2000', '2010'])  # Non-numeric
        })
        
        filter_params = {
            'use_price_filter': True,
            'use_year_filter': True
        }
        
        warnings = validate_filter_data_types(df, filter_params)
        
        # Should warn about non-numeric columns used in filtering
        assert 'price' in warnings
        assert 'year' in warnings
        assert 'size' not in warnings  # Building SqFt is numeric
    
    def test_validate_filter_data_types_all_numeric(self):
        """Test validation with all numeric columns."""
        df = pd.DataFrame({
            'Sold Price': [100000, 200000, 300000],
            'Building SqFt': [10000, 20000, 30000],
            'Year Built': [1990, 2000, 2010]
        })
        
        filter_params = {
            'use_price_filter': True,
            'use_year_filter': True
        }
        
        warnings = validate_filter_data_types(df, filter_params)
        
        # Should have no warnings
        assert len(warnings) == 0


class TestErrorHandling:
    """Test error handling in enhanced filtering."""
    
    @patch('streamlit.warning')
    def test_filter_error_handling(self, mock_warning):
        """Test that filtering errors are handled gracefully."""
        # Create DataFrame with problematic data
        df = pd.DataFrame({
            'Sold Price': [None, None, None]  # All null values
        })
        
        # Should not crash - null values are correctly filtered out
        filtered = safe_price_filter(df, (100000, 500000))
        assert len(filtered) == 0  # All null values should be filtered out
        assert isinstance(filtered, pd.DataFrame)  # Should still return a DataFrame
    
    @patch('streamlit.error')
    def test_enhanced_filter_error_handling(self, mock_error):
        """Test error handling in comprehensive filtering."""
        # Create problematic DataFrame
        df = pd.DataFrame({
            'Property Name': ['A', 'B', 'C']
        })
        
        # Create filter params that might cause issues
        filter_params = {
            'use_price_filter': True,
            'price_range': (100000, 500000)  # But no price column exists
        }
        
        # Should handle gracefully
        filtered = apply_enhanced_filters(df, filter_params)
        assert isinstance(filtered, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])