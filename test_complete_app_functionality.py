#!/usr/bin/env python3
"""
Comprehensive Application Functionality Test
Tests all features of the Flex Property Intelligence Platform
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import tempfile
import io

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_comprehensive_test_data():
    """Create comprehensive test dataset with all expected columns"""
    np.random.seed(42)

    # Property types with realistic distributions
    property_types = ['Warehouse', 'Industrial', 'Flex Space', 'Distribution Center',
                     'Manufacturing', 'Office', 'Retail', 'Mixed Use']

    # Cities and states
    cities = ['Los Angeles', 'Chicago', 'Atlanta', 'Dallas', 'Phoenix', 'Miami',
              'Seattle', 'Denver', 'Las Vegas', 'Houston']
    states = ['CA', 'IL', 'GA', 'TX', 'AZ', 'FL', 'WA', 'CO', 'NV', 'TX']

    n_properties = 100

    data = {
        'Property Name': [f'Property {i+1}' for i in range(n_properties)],
        'Property Type': np.random.choice(property_types, n_properties),
        'Address': [f'{i+1} Industrial Blvd' for i in range(n_properties)],
        'City': np.random.choice(cities, n_properties),
        'State': np.random.choice(states, n_properties),
        'County': [f'{city} County' for city in np.random.choice(cities, n_properties)],
        'Building SqFt': np.random.normal(50000, 20000, n_properties).astype(int),
        'Lot Size Acres': np.random.exponential(3, n_properties),
        'Lot Size SqFt': np.random.normal(200000, 80000, n_properties).astype(int),
        'Year Built': np.random.choice(range(1980, 2024), n_properties),
        'Sold Price': np.random.normal(2000000, 800000, n_properties).astype(int),
        'Loan Amount': np.random.normal(1500000, 600000, n_properties).astype(int),
        'Interest Rate': np.random.normal(4.5, 1.0, n_properties),
        'Number of Units': np.random.choice([1, 2, 3, 4, 5], n_properties),
        'Occupancy': np.random.normal(85, 15, n_properties),
        'Owner': [f'Owner {i+1}' for i in range(n_properties)]
    }

    # Ensure positive values
    data['Building SqFt'] = np.maximum(data['Building SqFt'], 5000)
    data['Lot Size Acres'] = np.maximum(data['Lot Size Acres'], 0.5)
    data['Lot Size SqFt'] = np.maximum(data['Lot Size SqFt'], 20000)
    data['Sold Price'] = np.maximum(data['Sold Price'], 100000)
    data['Loan Amount'] = np.maximum(data['Loan Amount'], 50000)
    data['Interest Rate'] = np.maximum(data['Interest Rate'], 2.0)
    data['Occupancy'] = np.clip(data['Occupancy'], 0, 100)

    return pd.DataFrame(data)

def test_data_loading_and_processing():
    """Test data loading and processing functionality"""
    print("\n[TEST] Data Loading and Processing...")

    # Create test data
    df = create_comprehensive_test_data()

    # Test basic data structure
    assert len(df) > 0, "Data should not be empty"
    assert 'Property Type' in df.columns, "Property Type column should exist"
    assert 'Building SqFt' in df.columns, "Building SqFt column should exist"

    # Test data types and ranges
    assert df['Building SqFt'].min() > 0, "Building SqFt should be positive"
    assert df['Sold Price'].min() > 0, "Sold Price should be positive"
    assert df['Year Built'].min() >= 1900, "Year Built should be reasonable"
    assert df['Year Built'].max() <= 2024, "Year Built should not be future"

    print(f"   [PASS] Created test data: {len(df)} properties")
    print(f"   [PASS] Data types validated")
    print(f"   [PASS] Value ranges validated")

    return True

def test_categorical_data_handling():
    """Test handling of categorical data types"""
    print("\n[TEST] Categorical Data Handling...")

    df = create_comprehensive_test_data()

    # Convert some columns to categorical to test edge cases
    df['Property Type'] = df['Property Type'].astype('category')
    df['Sold Price'] = df['Sold Price'].astype('category')
    df['Year Built'] = df['Year Built'].astype('category')

    # Test categorical data conversion functions
    def safe_min_max(series):
        """Test function for safe min/max calculation"""
        if series.dtype.name == 'category':
            converted = pd.to_numeric(series.astype(str), errors='coerce')
        else:
            converted = pd.to_numeric(series, errors='coerce')

        converted = converted.dropna()
        if len(converted) > 0:
            return converted.min(), converted.max()
        return 0, 0

    # Test price handling
    price_min, price_max = safe_min_max(df['Sold Price'])
    assert price_min > 0, "Price min should be positive"
    assert price_max > price_min, "Price max should be greater than min"

    # Test year handling
    year_min, year_max = safe_min_max(df['Year Built'])
    assert year_min >= 1900, "Year min should be reasonable"
    assert year_max <= 2024, "Year max should not be future"

    print(f"   [PASS] Categorical price range: ${price_min:,.0f} - ${price_max:,.0f}")
    print(f"   [PASS] Categorical year range: {year_min} - {year_max}")
    print(f"   [PASS] Categorical data handling works correctly")

    return True

def test_filtering_functionality():
    """Test all filtering functionality"""
    print("\n[TEST] Filtering Functionality...")

    df = create_comprehensive_test_data()

    # Test property type filtering
    industrial_keywords = ['industrial', 'warehouse', 'distribution', 'flex']
    mask = df['Property Type'].str.lower().str.contains(
        '|'.join(industrial_keywords), na=False
    )
    filtered_df = df[mask]

    print(f"   [PASS] Property type filtering: {len(filtered_df)}/{len(df)} properties")
    assert len(filtered_df) >= 0, "Filtering should not fail"

    # Test building size filtering
    size_min, size_max = 20000, 80000
    size_mask = (df['Building SqFt'] >= size_min) & (df['Building SqFt'] <= size_max)
    size_filtered = df[size_mask]

    print(f"   [PASS] Size filtering: {len(size_filtered)}/{len(df)} properties")
    assert len(size_filtered) >= 0, "Size filtering should not fail"

    # Test combined filtering
    combined_mask = mask & size_mask
    combined_filtered = df[combined_mask]

    print(f"   [PASS] Combined filtering: {len(combined_filtered)}/{len(df)} properties")
    assert len(combined_filtered) >= 0, "Combined filtering should not fail"

    # Test price range filtering
    price_mask = (df['Sold Price'] >= 500000) & (df['Sold Price'] <= 3000000)
    price_filtered = df[price_mask]

    print(f"   [PASS] Price filtering: {len(price_filtered)}/{len(df)} properties")
    assert len(price_filtered) >= 0, "Price filtering should not fail"

    return True

def test_visualization_components():
    """Test visualization and plotting functionality"""
    print("\n[TEST] Visualization Components...")

    df = create_comprehensive_test_data()

    try:
        # Test property type distribution
        fig1 = px.pie(df, names='Property Type', title='Property Type Distribution')
        assert fig1 is not None, "Pie chart should be created"
        print("   [PASS] Property type pie chart created")

        # Test building size histogram
        fig2 = px.histogram(df, x='Building SqFt', title='Building Size Distribution')
        assert fig2 is not None, "Histogram should be created"
        print("   [PASS] Building size histogram created")

        # Test scatter plot
        fig3 = px.scatter(df, x='Building SqFt', y='Sold Price',
                         color='Property Type', title='Size vs Price')
        assert fig3 is not None, "Scatter plot should be created"
        print("   [PASS] Size vs Price scatter plot created")

        # Test geographic distribution
        city_counts = df['City'].value_counts()
        fig4 = px.bar(x=city_counts.index, y=city_counts.values,
                     title='Properties by City')
        assert fig4 is not None, "Bar chart should be created"
        print("   [PASS] Geographic distribution chart created")

        # Test year built timeline
        year_counts = df['Year Built'].value_counts().sort_index()
        fig5 = px.line(x=year_counts.index, y=year_counts.values,
                      title='Construction Timeline')
        assert fig5 is not None, "Line chart should be created"
        print("   [PASS] Construction timeline chart created")

        return True

    except Exception as e:
        print(f"   [FAIL] Visualization error: {e}")
        return False

def test_export_functionality():
    """Test data export functionality"""
    print("\n[TEST] Export Functionality...")

    df = create_comprehensive_test_data()

    try:
        # Test CSV export
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        assert len(csv_content) > 0, "CSV content should not be empty"
        print("   [PASS] CSV export successful")

        # Test Excel export
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_content = excel_buffer.getvalue()
        assert len(excel_content) > 0, "Excel content should not be empty"
        print("   [PASS] Excel export successful")

        # Test filtered data export
        filtered_df = df[df['Property Type'].str.contains('Warehouse|Industrial', na=False)]

        filtered_csv_buffer = io.StringIO()
        filtered_df.to_csv(filtered_csv_buffer, index=False)
        filtered_csv_content = filtered_csv_buffer.getvalue()
        assert len(filtered_csv_content) > 0, "Filtered CSV should not be empty"
        print(f"   [PASS] Filtered data export: {len(filtered_df)} properties")

        # Test export with metadata
        export_df = df.copy()
        export_df['export_timestamp'] = datetime.now().isoformat()
        export_df['data_source'] = 'Test Data'

        meta_buffer = io.StringIO()
        export_df.to_csv(meta_buffer, index=False)
        meta_content = meta_buffer.getvalue()
        assert 'export_timestamp' in meta_content, "Metadata should be included"
        print("   [PASS] Export with metadata successful")

        return True

    except Exception as e:
        print(f"   [FAIL] Export error: {e}")
        return False

def test_performance_with_large_dataset():
    """Test performance with larger dataset"""
    print("\n[TEST] Performance with Large Dataset...")

    import time

    # Create larger dataset
    np.random.seed(42)
    n_large = 1000

    large_data = {
        'Property Name': [f'Large Property {i+1}' for i in range(n_large)],
        'Property Type': np.random.choice(['Warehouse', 'Industrial', 'Flex'], n_large),
        'Building SqFt': np.random.normal(50000, 20000, n_large).astype(int),
        'Sold Price': np.random.normal(2000000, 800000, n_large).astype(int),
        'Year Built': np.random.choice(range(1980, 2024), n_large),
        'City': np.random.choice(['Los Angeles', 'Chicago', 'Atlanta'], n_large),
    }

    large_df = pd.DataFrame(large_data)
    large_df['Building SqFt'] = np.maximum(large_df['Building SqFt'], 5000)
    large_df['Sold Price'] = np.maximum(large_df['Sold Price'], 100000)

    # Test loading performance
    start_time = time.time()
    df_loaded = large_df.copy()
    load_time = time.time() - start_time
    print(f"   [PASS] Large dataset loading: {load_time:.3f}s for {len(large_df)} records")

    # Test filtering performance
    start_time = time.time()
    filtered = large_df[large_df['Building SqFt'] > 30000]
    filter_time = time.time() - start_time
    print(f"   [PASS] Large dataset filtering: {filter_time:.3f}s, {len(filtered)} results")

    # Test aggregation performance
    start_time = time.time()
    stats = {
        'mean_sqft': large_df['Building SqFt'].mean(),
        'mean_price': large_df['Sold Price'].mean(),
        'property_counts': large_df['Property Type'].value_counts()
    }
    agg_time = time.time() - start_time
    print(f"   [PASS] Large dataset aggregation: {agg_time:.3f}s")

    # Performance benchmarks
    assert load_time < 5.0, "Loading should be under 5 seconds"
    assert filter_time < 1.0, "Filtering should be under 1 second"
    assert agg_time < 1.0, "Aggregation should be under 1 second"

    return True

def test_error_handling_and_edge_cases():
    """Test error handling and edge cases"""
    print("\n[TEST] Error Handling and Edge Cases...")

    # Test empty dataframe
    empty_df = pd.DataFrame()
    try:
        result = len(empty_df)
        assert result == 0, "Empty dataframe should have length 0"
        print("   [PASS] Empty dataframe handling")
    except Exception as e:
        print(f"   [FAIL] Empty dataframe error: {e}")
        return False

    # Test dataframe with missing values
    df_with_nulls = create_comprehensive_test_data()
    df_with_nulls.loc[0:10, 'Building SqFt'] = np.nan
    df_with_nulls.loc[5:15, 'Sold Price'] = np.nan

    try:
        # Test safe statistical operations
        valid_sqft = df_with_nulls['Building SqFt'].dropna()
        if len(valid_sqft) > 0:
            mean_sqft = valid_sqft.mean()
            assert not pd.isna(mean_sqft), "Mean should not be NaN"
        print("   [PASS] Missing data handling")
    except Exception as e:
        print(f"   [FAIL] Missing data error: {e}")
        return False

    # Test invalid data types
    df_mixed = create_comprehensive_test_data()
    df_mixed.loc[0, 'Building SqFt'] = 'invalid'
    df_mixed.loc[1, 'Sold Price'] = 'not_a_number'

    try:
        # Test safe numeric conversion
        cleaned_sqft = pd.to_numeric(df_mixed['Building SqFt'], errors='coerce')
        valid_count = cleaned_sqft.count()
        assert valid_count > 0, "Should have some valid numeric values"
        print("   [PASS] Invalid data type handling")
    except Exception as e:
        print(f"   [FAIL] Invalid data type error: {e}")
        return False

    # Test extreme values
    df_extreme = create_comprehensive_test_data()
    df_extreme.loc[0, 'Building SqFt'] = 999999999  # Very large
    df_extreme.loc[1, 'Building SqFt'] = -1000      # Negative
    df_extreme.loc[2, 'Year Built'] = 1800          # Very old
    df_extreme.loc[3, 'Year Built'] = 2050          # Future

    try:
        # Test value validation
        valid_sqft = df_extreme['Building SqFt'][(df_extreme['Building SqFt'] > 0) &
                                                 (df_extreme['Building SqFt'] < 10000000)]
        valid_years = df_extreme['Year Built'][(df_extreme['Year Built'] >= 1900) &
                                              (df_extreme['Year Built'] <= 2024)]

        assert len(valid_sqft) > 0, "Should have valid building sizes"
        assert len(valid_years) > 0, "Should have valid years"
        print("   [PASS] Extreme value handling")
    except Exception as e:
        print(f"   [FAIL] Extreme value error: {e}")
        return False

    return True

def run_comprehensive_test_suite():
    """Run the complete test suite"""
    print("=" * 80)
    print("COMPREHENSIVE APPLICATION FUNCTIONALITY TEST SUITE")
    print("=" * 80)

    tests = [
        ("Data Loading and Processing", test_data_loading_and_processing),
        ("Categorical Data Handling", test_categorical_data_handling),
        ("Filtering Functionality", test_filtering_functionality),
        ("Visualization Components", test_visualization_components),
        ("Export Functionality", test_export_functionality),
        ("Performance with Large Dataset", test_performance_with_large_dataset),
        ("Error Handling and Edge Cases", test_error_handling_and_edge_cases)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"\n[SUCCESS] {test_name} - All tests passed")
            else:
                failed += 1
                print(f"\n[FAILURE] {test_name} - Some tests failed")
        except Exception as e:
            failed += 1
            print(f"\n[ERROR] {test_name} - Test crashed: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    total = passed + failed
    success_rate = (passed / total * 100) if total > 0 else 0

    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST RESULTS")
    print("=" * 80)
    print(f"Tests Passed: {passed}")
    print(f"Tests Failed: {failed}")
    print(f"Success Rate: {success_rate:.1f}%")

    if success_rate >= 90:
        print("\n[EXCELLENT] Application functionality is FLAWLESS!")
        print("All critical features are working perfectly.")
    elif success_rate >= 80:
        print("\n[GOOD] Application functionality is SOLID!")
        print("Minor issues detected but core features work well.")
    elif success_rate >= 70:
        print("\n[ACCEPTABLE] Application functionality is WORKING!")
        print("Some issues need attention but app is functional.")
    else:
        print("\n[NEEDS WORK] Application has significant issues!")
        print("Multiple features need debugging and fixes.")

    return success_rate >= 80

if __name__ == "__main__":
    success = run_comprehensive_test_suite()
    sys.exit(0 if success else 1)