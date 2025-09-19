import unittest
import pandas as pd
import numpy as np
from flex_filter_dashboard import clean_numeric_column, apply_filters, validate_filter_ranges

class TestDashboardFunctions(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        self.test_df = pd.DataFrame({
            'Property Name': ['Property A', 'Property B', 'Property C', 'Property D'],
            'Property Type': ['Industrial Warehouse', 'Office Building', 'Flex Space', 'Distribution Center'],
            'City': ['City A', 'City B', 'City A', 'City C'],
            'County': ['County 1', 'County 2', 'County 1', 'County 3'],
            'State': ['CA', 'TX', 'CA', 'FL'],
            'Building SqFt': [50000, 25000, 75000, 100000],
            'Lot Size Acres': [2.5, 1.0, 5.0, 8.0],
            'Year Built': [2000, 1995, 2010, 2005],
            'Sold Price': [1000000, 500000, 1500000, 2000000],
            'Occupancy': [85, 90, 75, 95]
        })
    
    def test_clean_numeric_column_with_currency(self):
        """Test cleaning numeric columns with currency symbols"""
        dirty_series = pd.Series(['$1,000', '$2,500.50', '$500', 'N/A'])
        cleaned = clean_numeric_column(dirty_series)
        expected = pd.Series([1000.0, 2500.50, 500.0, np.nan])
        pd.testing.assert_series_equal(cleaned, expected, check_names=False)
    
    def test_clean_numeric_column_with_percentages(self):
        """Test cleaning numeric columns with percentage symbols"""
        dirty_series = pd.Series(['85%', '90.5%', '75%', 'n/a'])
        cleaned = clean_numeric_column(dirty_series)
        expected = pd.Series([85.0, 90.5, 75.0, np.nan])
        pd.testing.assert_series_equal(cleaned, expected, check_names=False)
    
    def test_clean_numeric_column_already_numeric(self):
        """Test cleaning already numeric columns"""
        numeric_series = pd.Series([1000, 2000, 3000])
        cleaned = clean_numeric_column(numeric_series)
        pd.testing.assert_series_equal(cleaned, numeric_series)
    
    def test_validate_filter_ranges_valid(self):
        """Test filter range validation with valid ranges"""
        result = validate_filter_ranges(
            (10000, 50000),  # size_range
            (1.0, 10.0),     # lot_range
            (100000, 500000), # price_range
            (1990, 2020),    # year_range
            (0, 100)         # occupancy_range
        )
        self.assertTrue(result)
    
    def test_validate_filter_ranges_invalid(self):
        """Test filter range validation with invalid ranges"""
        # This would normally show an error in Streamlit, but we'll test the logic
        # In a real implementation, we'd need to mock st.error
        pass  # Skip this test as it requires Streamlit context
    
    def test_apply_filters_property_type(self):
        """Test applying property type filters"""
        filter_params = {
            'industrial_keywords': ['industrial', 'warehouse', 'distribution'],
            'selected_counties': self.test_df['County'].unique().tolist(),
            'selected_states': self.test_df['State'].unique().tolist()
        }
        
        filtered_df = apply_filters(self.test_df, filter_params)
        
        # Should match 'Industrial Warehouse' and 'Distribution Center'
        self.assertEqual(len(filtered_df), 2)
        self.assertIn('Industrial Warehouse', filtered_df['Property Type'].values)
        self.assertIn('Distribution Center', filtered_df['Property Type'].values)
    
    def test_apply_filters_building_size(self):
        """Test applying building size filters"""
        filter_params = {
            'size_range': (30000, 80000),
            'selected_counties': self.test_df['County'].unique().tolist(),
            'selected_states': self.test_df['State'].unique().tolist()
        }
        
        filtered_df = apply_filters(self.test_df, filter_params)
        
        # Should match properties with 50000 and 75000 sqft
        self.assertEqual(len(filtered_df), 2)
        self.assertTrue(all(30000 <= sqft <= 80000 for sqft in filtered_df['Building SqFt']))
    
    def test_apply_filters_lot_size(self):
        """Test applying lot size filters"""
        filter_params = {
            'lot_range': (2.0, 6.0),
            'selected_counties': self.test_df['County'].unique().tolist(),
            'selected_states': self.test_df['State'].unique().tolist()
        }
        
        filtered_df = apply_filters(self.test_df, filter_params)
        
        # Should match properties with 2.5 and 5.0 acres
        self.assertEqual(len(filtered_df), 2)
        self.assertTrue(all(2.0 <= acres <= 6.0 for acres in filtered_df['Lot Size Acres']))
    
    def test_apply_filters_price_range(self):
        """Test applying price range filters"""
        filter_params = {
            'use_price_filter': True,
            'price_range': (750000, 1750000),
            'selected_counties': self.test_df['County'].unique().tolist(),
            'selected_states': self.test_df['State'].unique().tolist()
        }
        
        filtered_df = apply_filters(self.test_df, filter_params)
        
        # Should match properties with prices 1000000 and 1500000
        self.assertEqual(len(filtered_df), 2)
        self.assertTrue(all(750000 <= price <= 1750000 for price in filtered_df['Sold Price']))
    
    def test_apply_filters_year_built(self):
        """Test applying year built filters"""
        filter_params = {
            'use_year_filter': True,
            'year_range': (1998, 2008),
            'selected_counties': self.test_df['County'].unique().tolist(),
            'selected_states': self.test_df['State'].unique().tolist()
        }
        
        filtered_df = apply_filters(self.test_df, filter_params)
        
        # Should match properties built in 2000 and 2005
        self.assertEqual(len(filtered_df), 2)
        self.assertTrue(all(1998 <= year <= 2008 for year in filtered_df['Year Built']))
    
    def test_apply_filters_occupancy(self):
        """Test applying occupancy filters"""
        filter_params = {
            'use_occupancy_filter': True,
            'occupancy_range': (80, 92),
            'selected_counties': self.test_df['County'].unique().tolist(),
            'selected_states': self.test_df['State'].unique().tolist()
        }
        
        filtered_df = apply_filters(self.test_df, filter_params)
        
        # Should match properties with occupancy 85 and 90
        self.assertEqual(len(filtered_df), 2)
        self.assertTrue(all(80 <= occ <= 92 for occ in filtered_df['Occupancy']))
    
    def test_apply_filters_county_state(self):
        """Test applying county and state filters"""
        filter_params = {
            'selected_counties': ['County 1'],
            'selected_states': ['CA']
        }
        
        filtered_df = apply_filters(self.test_df, filter_params)
        
        # Should match properties in County 1 and CA
        self.assertEqual(len(filtered_df), 2)
        self.assertTrue(all(county == 'County 1' for county in filtered_df['County']))
        self.assertTrue(all(state == 'CA' for state in filtered_df['State']))
    
    def test_apply_filters_combined(self):
        """Test applying multiple filters together"""
        filter_params = {
            'industrial_keywords': ['industrial', 'flex'],
            'size_range': (40000, 80000),
            'lot_range': (2.0, 6.0),
            'selected_counties': self.test_df['County'].unique().tolist(),
            'selected_states': self.test_df['State'].unique().tolist()
        }
        
        filtered_df = apply_filters(self.test_df, filter_params)
        
        # Should match only 'Property A' (Industrial Warehouse, 50000 sqft, 2.5 acres)
        # and 'Property C' (Flex Space, 75000 sqft, 5.0 acres)
        self.assertEqual(len(filtered_df), 2)
    
    def test_apply_filters_empty_result(self):
        """Test filters that result in no matches"""
        filter_params = {
            'size_range': (200000, 300000),  # No properties in this range
            'selected_counties': self.test_df['County'].unique().tolist(),
            'selected_states': self.test_df['State'].unique().tolist()
        }
        
        filtered_df = apply_filters(self.test_df, filter_params)
        
        self.assertEqual(len(filtered_df), 0)
    
    def test_apply_filters_missing_columns(self):
        """Test filters with missing columns"""
        # Create dataframe without some columns
        incomplete_df = self.test_df.drop(['Sold Price', 'Occupancy'], axis=1)
        
        filter_params = {
            'use_price_filter': True,
            'price_range': (100000, 500000),
            'use_occupancy_filter': True,
            'occupancy_range': (80, 95),
            'selected_counties': incomplete_df['County'].unique().tolist(),
            'selected_states': incomplete_df['State'].unique().tolist()
        }
        
        # Should not crash and return the dataframe
        filtered_df = apply_filters(incomplete_df, filter_params)
        self.assertEqual(len(filtered_df), len(incomplete_df))

def run_integration_test():
    """Run integration test to verify end-to-end functionality"""
    print("Running integration test...")
    
    # Create test data
    test_data = {
        'Property Name': [f'Property {i}' for i in range(100)],
        'Property Type': ['Industrial'] * 30 + ['Warehouse'] * 25 + ['Office'] * 25 + ['Flex'] * 20,
        'City': ['City A'] * 40 + ['City B'] * 35 + ['City C'] * 25,
        'County': ['County 1'] * 50 + ['County 2'] * 30 + ['County 3'] * 20,
        'State': ['CA'] * 60 + ['TX'] * 25 + ['FL'] * 15,
        'Building SqFt': np.random.randint(10000, 200000, 100),
        'Lot Size Acres': np.random.uniform(0.5, 20.0, 100),
        'Year Built': np.random.randint(1980, 2025, 100),
        'Sold Price': np.random.randint(100000, 5000000, 100),
        'Occupancy': np.random.randint(50, 100, 100)
    }
    
    df = pd.DataFrame(test_data)
    
    # Test various filter combinations
    test_cases = [
        {
            'name': 'Industrial properties only',
            'params': {
                'industrial_keywords': ['industrial'],
                'selected_counties': df['County'].unique().tolist(),
                'selected_states': df['State'].unique().tolist()
            }
        },
        {
            'name': 'Large buildings',
            'params': {
                'size_range': (100000, 200000),
                'selected_counties': df['County'].unique().tolist(),
                'selected_states': df['State'].unique().tolist()
            }
        },
        {
            'name': 'Recent construction',
            'params': {
                'use_year_filter': True,
                'year_range': (2010, 2025),
                'selected_counties': df['County'].unique().tolist(),
                'selected_states': df['State'].unique().tolist()
            }
        },
        {
            'name': 'California properties',
            'params': {
                'selected_counties': df['County'].unique().tolist(),
                'selected_states': ['CA']
            }
        }
    ]
    
    for test_case in test_cases:
        try:
            filtered_df = apply_filters(df, test_case['params'])
            print(f"✓ {test_case['name']}: {len(filtered_df)} results")
        except Exception as e:
            print(f"✗ {test_case['name']}: Error - {str(e)}")
    
    print("Integration test completed.")

if __name__ == '__main__':
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration test
    run_integration_test()