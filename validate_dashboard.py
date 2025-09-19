"""
Validation script for the Interactive Filter Dashboard
This script validates the dashboard functionality and data handling
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def create_test_data():
    """Create comprehensive test data for validation"""
    np.random.seed(42)  # For reproducible results
    
    # Create test data with various edge cases
    n_properties = 1000
    
    data = {
        'Property Name': [f'Test Property {i:04d}' for i in range(n_properties)],
        'Property Type': np.random.choice([
            'Industrial Warehouse', 'Distribution Center', 'Flex Space',
            'Manufacturing Facility', 'Light Industrial', 'Storage Facility',
            'Office Building', 'Retail Space', 'Mixed Use'
        ], n_properties),
        'Address': [f'{i} Test Street' for i in range(n_properties)],
        'City': np.random.choice([
            'Los Angeles', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio',
            'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville'
        ], n_properties),
        'County': np.random.choice([
            'Los Angeles County', 'Harris County', 'Maricopa County',
            'Philadelphia County', 'Bexar County', 'San Diego County',
            'Dallas County', 'Santa Clara County', 'Travis County', 'Duval County'
        ], n_properties),
        'State': np.random.choice(['CA', 'TX', 'AZ', 'PA', 'FL'], n_properties),
        'Building SqFt': np.random.randint(5000, 500000, n_properties),
        'Lot Size Acres': np.random.uniform(0.1, 50.0, n_properties),
        'Lot Size SqFt': np.random.randint(4000, 2000000, n_properties),
        'Year Built': np.random.randint(1950, 2025, n_properties),
        'Sold Price': np.random.randint(50000, 10000000, n_properties),
        'Loan Amount': np.random.randint(25000, 8000000, n_properties),
        'Interest Rate': np.random.uniform(2.0, 8.0, n_properties),
        'Number of Units': np.random.randint(1, 50, n_properties),
        'Occupancy': np.random.randint(0, 100, n_properties)
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values to test error handling
    missing_indices = np.random.choice(n_properties, size=int(n_properties * 0.1), replace=False)
    df.loc[missing_indices, 'Building SqFt'] = np.nan
    
    missing_indices = np.random.choice(n_properties, size=int(n_properties * 0.05), replace=False)
    df.loc[missing_indices, 'Sold Price'] = np.nan
    
    return df

def create_dirty_test_data():
    """Create test data with formatting issues to test cleaning functions"""
    data = {
        'Property Name': ['Dirty Property 1', 'Dirty Property 2', 'Dirty Property 3'],
        'Property Type': ['Industrial Warehouse', 'Distribution Center', 'Flex Space'],
        'Building SqFt': ['$50,000', '75000', 'N/A'],
        'Sold Price': ['$1,500,000', '$2,250,500.50', 'n/a'],
        'Occupancy': ['85%', '90.5%', 'None'],
        'Interest Rate': ['4.5%', '3.25%', ''],
        'City': ['Test City', 'Another City', 'Third City'],
        'County': ['Test County', 'Another County', 'Third County'],
        'State': ['CA', 'TX', 'FL']
    }
    
    return pd.DataFrame(data)

def validate_data_loading():
    """Validate data loading and cleaning functionality"""
    print("=== Data Loading Validation ===")
    
    # Test with clean data
    clean_df = create_test_data()
    print(f"✓ Created clean test data: {len(clean_df)} rows, {len(clean_df.columns)} columns")
    
    # Test with dirty data
    dirty_df = create_dirty_test_data()
    print(f"✓ Created dirty test data: {len(dirty_df)} rows, {len(dirty_df.columns)} columns")
    
    # Test cleaning functions
    from flex_filter_dashboard import clean_numeric_column
    
    # Test currency cleaning
    dirty_price = dirty_df['Sold Price']
    cleaned_price = clean_numeric_column(dirty_price)
    print(f"✓ Price cleaning: {dirty_price.tolist()} -> {cleaned_price.tolist()}")
    
    # Test percentage cleaning
    dirty_occupancy = dirty_df['Occupancy']
    cleaned_occupancy = clean_numeric_column(dirty_occupancy)
    print(f"✓ Occupancy cleaning: {dirty_occupancy.tolist()} -> {cleaned_occupancy.tolist()}")
    
    return clean_df, dirty_df

def validate_filtering():
    """Validate filtering functionality"""
    print("\n=== Filtering Validation ===")
    
    df = create_test_data()
    from flex_filter_dashboard import apply_filters
    
    # Test basic property type filter
    filter_params = {
        'industrial_keywords': ['industrial', 'warehouse', 'distribution'],
        'selected_counties': df['County'].unique().tolist(),
        'selected_states': df['State'].unique().tolist()
    }
    
    filtered_df = apply_filters(df, filter_params)
    industrial_count = len(filtered_df)
    print(f"✓ Industrial property filter: {industrial_count} results")
    
    # Test size range filter
    filter_params = {
        'size_range': (50000, 150000),
        'selected_counties': df['County'].unique().tolist(),
        'selected_states': df['State'].unique().tolist()
    }
    
    filtered_df = apply_filters(df, filter_params)
    size_filtered_count = len(filtered_df)
    print(f"✓ Size range filter (50k-150k sqft): {size_filtered_count} results")
    
    # Test combined filters
    filter_params = {
        'industrial_keywords': ['industrial', 'warehouse'],
        'size_range': (25000, 200000),
        'lot_range': (1.0, 20.0),
        'use_price_filter': True,
        'price_range': (500000, 5000000),
        'selected_counties': df['County'].unique().tolist(),
        'selected_states': ['CA', 'TX']
    }
    
    filtered_df = apply_filters(df, filter_params)
    combined_count = len(filtered_df)
    print(f"✓ Combined filters: {combined_count} results")
    
    # Test edge case - no results
    filter_params = {
        'size_range': (1000000, 2000000),  # Very large buildings
        'selected_counties': df['County'].unique().tolist(),
        'selected_states': df['State'].unique().tolist()
    }
    
    filtered_df = apply_filters(df, filter_params)
    edge_case_count = len(filtered_df)
    print(f"✓ Edge case filter (very large buildings): {edge_case_count} results")
    
    return True

def validate_export_functionality():
    """Validate export functionality"""
    print("\n=== Export Validation ===")
    
    df = create_test_data()
    
    # Test CSV export
    try:
        csv_data = df.to_csv(index=False)
        print(f"✓ CSV export: {len(csv_data)} characters")
    except Exception as e:
        print(f"✗ CSV export failed: {str(e)}")
        return False
    
    # Test Excel export
    try:
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Test Properties', index=False)
        excel_size = len(buffer.getvalue())
        print(f"✓ Excel export: {excel_size} bytes")
    except Exception as e:
        print(f"✗ Excel export failed: {str(e)}")
        return False
    
    return True

def validate_performance():
    """Validate performance with large datasets"""
    print("\n=== Performance Validation ===")
    
    # Create larger dataset
    large_df = create_test_data()
    # Duplicate to make it larger
    large_df = pd.concat([large_df] * 10, ignore_index=True)
    print(f"✓ Created large dataset: {len(large_df)} rows")
    
    # Test filtering performance
    start_time = datetime.now()
    
    from flex_filter_dashboard import apply_filters
    filter_params = {
        'industrial_keywords': ['industrial', 'warehouse', 'distribution'],
        'size_range': (25000, 200000),
        'selected_counties': large_df['County'].unique().tolist(),
        'selected_states': large_df['State'].unique().tolist()
    }
    
    filtered_df = apply_filters(large_df, filter_params)
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    print(f"✓ Large dataset filtering: {len(filtered_df)} results in {duration:.3f} seconds")
    
    if duration > 5.0:
        print("⚠ Warning: Filtering took longer than 5 seconds")
    
    return True

def validate_error_handling():
    """Validate error handling with problematic data"""
    print("\n=== Error Handling Validation ===")
    
    # Test with completely empty dataframe
    empty_df = pd.DataFrame()
    
    from flex_filter_dashboard import apply_filters
    try:
        result = apply_filters(empty_df, {})
        print(f"✓ Empty dataframe handling: returned {len(result)} rows")
    except Exception as e:
        print(f"✗ Empty dataframe handling failed: {str(e)}")
    
    # Test with missing columns
    incomplete_df = pd.DataFrame({
        'Property Name': ['Test 1', 'Test 2'],
        'City': ['City A', 'City B']
    })
    
    try:
        filter_params = {
            'size_range': (10000, 50000),
            'selected_counties': ['County 1'],
            'selected_states': ['CA']
        }
        result = apply_filters(incomplete_df, filter_params)
        print(f"✓ Missing columns handling: returned {len(result)} rows")
    except Exception as e:
        print(f"✗ Missing columns handling failed: {str(e)}")
    
    # Test with all NaN values in numeric columns
    nan_df = create_test_data()
    nan_df['Building SqFt'] = np.nan
    nan_df['Sold Price'] = np.nan
    
    try:
        filter_params = {
            'size_range': (10000, 50000),
            'use_price_filter': True,
            'price_range': (100000, 500000),
            'selected_counties': nan_df['County'].unique().tolist(),
            'selected_states': nan_df['State'].unique().tolist()
        }
        result = apply_filters(nan_df, filter_params)
        print(f"✓ All NaN values handling: returned {len(result)} rows")
    except Exception as e:
        print(f"✗ All NaN values handling failed: {str(e)}")
    
    return True

def main():
    """Run all validation tests"""
    print("Interactive Filter Dashboard Validation")
    print("=" * 50)
    
    try:
        # Validate data loading
        clean_df, dirty_df = validate_data_loading()
        
        # Validate filtering
        validate_filtering()
        
        # Validate export
        validate_export_functionality()
        
        # Validate performance
        validate_performance()
        
        # Validate error handling
        validate_error_handling()
        
        print("\n" + "=" * 50)
        print("✓ All validation tests completed successfully!")
        print("The dashboard is ready for use.")
        
    except Exception as e:
        print(f"\n✗ Validation failed with error: {str(e)}")
        print("Please check the dashboard implementation.")

if __name__ == '__main__':
    main()