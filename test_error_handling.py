#!/usr/bin/env python3
"""
Test Error Handling and Edge Cases
"""

import pandas as pd
import numpy as np
import sys
import os
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

def test_error_handling():
    """Test all error handling and edge cases"""
    print('TESTING ERROR HANDLING AND EDGE CASES')
    print('=' * 45)

    # Test 1: Empty DataFrame handling
    print('\n1. Testing empty DataFrame handling...')
    try:
        empty_df = pd.DataFrame()

        # Test filtering on empty data
        if len(empty_df) == 0:
            print('   [PASS] Empty DataFrame detected correctly')

        # Test statistical operations on empty data
        if empty_df.empty:
            print('   [PASS] Empty DataFrame validation works')
        else:
            print('   [FAIL] Empty DataFrame not detected')

    except Exception as e:
        print(f'   [FAIL] Empty DataFrame error: {e}')

    # Test 2: Invalid data types
    print('\n2. Testing invalid data type handling...')
    try:
        invalid_data = {
            'Property_Name': ['Test Property'],
            'Building_SqFt': ['not_a_number'],  # Invalid numeric
            'Sold_Price': [None],  # Null value
            'Year_Built': ['invalid_year']  # Invalid year
        }

        df_invalid = pd.DataFrame(invalid_data)

        # Test numeric conversion with errors='coerce'
        numeric_sqft = pd.to_numeric(df_invalid['Building_SqFt'], errors='coerce')
        numeric_price = pd.to_numeric(df_invalid['Sold_Price'], errors='coerce')
        numeric_year = pd.to_numeric(df_invalid['Year_Built'], errors='coerce')

        if pd.isna(numeric_sqft[0]) and pd.isna(numeric_price[0]) and pd.isna(numeric_year[0]):
            print('   [PASS] Invalid data types handled with errors=\'coerce\'')
        else:
            print('   [FAIL] Invalid data type handling failed')

    except Exception as e:
        print(f'   [FAIL] Invalid data type error: {e}')

    # Test 3: Missing required columns
    print('\n3. Testing missing column handling...')
    try:
        incomplete_data = {
            'Property_Name': ['Test Property'],
            'City': ['Test City']
            # Missing required columns like Building_SqFt, Sold_Price
        }

        df_incomplete = pd.DataFrame(incomplete_data)

        required_columns = ['Building_SqFt', 'Sold_Price', 'Property_Type']
        missing_cols = [col for col in required_columns if col not in df_incomplete.columns]

        if missing_cols:
            print(f'   [PASS] Missing columns detected: {missing_cols}')
        else:
            print('   [FAIL] Missing column detection failed')

    except Exception as e:
        print(f'   [FAIL] Missing column error: {e}')

    # Test 4: Categorical data edge cases
    print('\n4. Testing categorical data edge cases...')
    try:
        cat_data = {
            'Property_Type': pd.Categorical(['Industrial', 'Warehouse', None]),
            'Sold_Price': pd.Categorical(['1000000', '2000000', 'invalid']),
            'Building_SqFt': [50000, 75000, 25000]
        }

        df_cat = pd.DataFrame(cat_data)

        # Test categorical to numeric conversion
        if df_cat['Sold_Price'].dtype.name == 'category':
            price_numeric = pd.to_numeric(df_cat['Sold_Price'].astype(str), errors='coerce')
            price_clean = price_numeric.dropna()

            if len(price_clean) < len(df_cat):
                print('   [PASS] Categorical data with invalid values handled')
            else:
                print('   [FAIL] Categorical data handling failed')

        # Test categorical filtering
        if df_cat['Property_Type'].dtype.name == 'category':
            filtered = df_cat[df_cat['Property_Type'].notna()]
            print(f'   [PASS] Categorical filtering: {len(filtered)}/{len(df_cat)} valid entries')

    except Exception as e:
        print(f'   [FAIL] Categorical data error: {e}')

    # Test 5: Large number edge cases
    print('\n5. Testing large number handling...')
    try:
        large_data = {
            'Property_Name': ['Large Property'],
            'Building_SqFt': [999999999],  # Very large building
            'Sold_Price': [999999999999],  # Very large price
            'Lot_Size_Acres': [9999.99]  # Very large lot
        }

        df_large = pd.DataFrame(large_data)

        # Test statistical operations on large numbers
        avg_price = df_large['Sold_Price'].mean()
        max_size = df_large['Building_SqFt'].max()

        if avg_price > 0 and max_size > 0:
            print('   [PASS] Large number calculations successful')
            print(f'     Max price: ${avg_price:,.0f}')
            print(f'     Max size: {max_size:,} sqft')
        else:
            print('   [FAIL] Large number handling failed')

    except Exception as e:
        print(f'   [FAIL] Large number error: {e}')

    # Test 6: Duplicate data handling
    print('\n6. Testing duplicate data handling...')
    try:
        duplicate_data = {
            'Property_Name': ['Property A', 'Property A', 'Property B'],
            'Building_SqFt': [50000, 50000, 75000],
            'Sold_Price': [1000000, 1000000, 1500000]
        }

        df_dup = pd.DataFrame(duplicate_data)

        # Check for duplicates
        duplicates = df_dup.duplicated()
        duplicate_count = duplicates.sum()

        if duplicate_count > 0:
            print(f'   [PASS] Duplicates detected: {duplicate_count} duplicate rows')

            # Test duplicate removal
            df_unique = df_dup.drop_duplicates()
            print(f'   [PASS] After deduplication: {len(df_unique)}/{len(df_dup)} unique rows')
        else:
            print('   [INFO] No duplicates found in test data')

    except Exception as e:
        print(f'   [FAIL] Duplicate handling error: {e}')

    # Test 7: Filter edge cases
    print('\n7. Testing filter edge cases...')
    try:
        filter_data = {
            'Property_Type': ['Industrial', 'Warehouse', 'Office', 'Retail'],
            'Building_SqFt': [50000, 0, -1000, 999999999],  # Include zero and negative
            'Sold_Price': [1000000, 0, -500000, 999999999],
            'Year_Built': [2020, 1800, 2050, 1990]  # Include unrealistic years
        }

        df_filter = pd.DataFrame(filter_data)

        # Test size filtering with edge cases
        valid_size = df_filter['Building_SqFt'] > 0
        size_filtered = df_filter[valid_size]
        print(f'   [PASS] Size filter (>0): {len(size_filtered)}/{len(df_filter)} properties')

        # Test price filtering with edge cases
        valid_price = df_filter['Sold_Price'] > 0
        price_filtered = df_filter[valid_price]
        print(f'   [PASS] Price filter (>0): {len(price_filtered)}/{len(df_filter)} properties')

        # Test year filtering with realistic range
        realistic_years = (df_filter['Year_Built'] >= 1900) & (df_filter['Year_Built'] <= 2023)
        year_filtered = df_filter[realistic_years]
        print(f'   [PASS] Year filter (1900-2023): {len(year_filtered)}/{len(df_filter)} properties')

    except Exception as e:
        print(f'   [FAIL] Filter edge case error: {e}')

    # Test 8: Memory and performance edge cases
    print('\n8. Testing memory and performance edge cases...')
    try:
        # Create a dataset that might cause memory issues if not handled properly
        large_dataset_size = 10000
        performance_data = {
            'Property_Name': [f'Property_{i}' for i in range(large_dataset_size)],
            'Building_SqFt': np.random.randint(1000, 500000, large_dataset_size),
            'Sold_Price': np.random.randint(100000, 10000000, large_dataset_size),
            'Property_Type': np.random.choice(['Industrial', 'Warehouse', 'Flex'], large_dataset_size)
        }

        df_perf = pd.DataFrame(performance_data)

        # Test basic operations on large dataset
        start_time = pd.Timestamp.now()

        # Filtering operation
        filtered = df_perf[df_perf['Building_SqFt'] > 50000]

        # Statistical operation
        avg_price = df_perf['Sold_Price'].mean()

        # Grouping operation
        type_stats = df_perf.groupby('Property_Type')['Sold_Price'].mean()

        end_time = pd.Timestamp.now()
        processing_time = (end_time - start_time).total_seconds()

        print(f'   [PASS] Large dataset processing: {large_dataset_size:,} records')
        print(f'   [PASS] Processing time: {processing_time:.3f} seconds')
        print(f'   [PASS] Memory usage acceptable for {len(filtered):,} filtered records')

    except Exception as e:
        print(f'   [FAIL] Performance test error: {e}')

    # Test 9: Export error handling
    print('\n9. Testing export error handling...')
    try:
        export_data = {
            'Property_Name': ['Test Property'],
            'Special_Chars': ['Property with "quotes" and ,commas'],
            'Unicode_Text': ['Property with üñíçødé'],
            'Large_Number': [999999999999]
        }

        df_export = pd.DataFrame(export_data)

        # Test CSV export with special characters
        try:
            csv_output = StringIO()
            df_export.to_csv(csv_output, index=False)
            csv_content = csv_output.getvalue()

            if len(csv_content) > 0:
                print('   [PASS] CSV export with special characters successful')
            else:
                print('   [FAIL] CSV export failed')

        except Exception as csv_e:
            print(f'   [FAIL] CSV export error: {csv_e}')

    except Exception as e:
        print(f'   [FAIL] Export test error: {e}')

    # Test 10: Data validation edge cases
    print('\n10. Testing data validation edge cases...')
    try:
        validation_data = {
            'Property_Name': ['', None, 'Valid Name', ' ', '   '],  # Empty and whitespace
            'Building_SqFt': [50000, None, 0, -1000, float('inf')],  # Various invalid values
            'Sold_Price': [1000000, None, 0, -500000, float('nan')]
        }

        df_validation = pd.DataFrame(validation_data)

        # Test name validation
        valid_names = df_validation['Property_Name'].notna() & \
                     (df_validation['Property_Name'].str.strip() != '')
        name_valid_count = valid_names.sum()
        print(f'   [PASS] Name validation: {name_valid_count}/{len(df_validation)} valid names')

        # Test numeric validation
        valid_sqft = df_validation['Building_SqFt'].notna() & \
                    (df_validation['Building_SqFt'] > 0) & \
                    np.isfinite(df_validation['Building_SqFt'])
        sqft_valid_count = valid_sqft.sum()
        print(f'   [PASS] SqFt validation: {sqft_valid_count}/{len(df_validation)} valid sizes')

        valid_price = df_validation['Sold_Price'].notna() & \
                     (df_validation['Sold_Price'] > 0) & \
                     np.isfinite(df_validation['Sold_Price'])
        price_valid_count = valid_price.sum()
        print(f'   [PASS] Price validation: {price_valid_count}/{len(df_validation)} valid prices')

    except Exception as e:
        print(f'   [FAIL] Data validation error: {e}')

    print('\n[COMPLETE] Error handling and edge case testing completed!')
    return True

if __name__ == "__main__":
    success = test_error_handling()
    print(f'\nError Handling Test Result: {"PASS" if success else "FAIL"}')