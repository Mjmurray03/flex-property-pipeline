#!/usr/bin/env python3
"""
Test Performance with Large Datasets
"""

import pandas as pd
import numpy as np
import time
import psutil
import gc
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def test_large_dataset_performance():
    """Test performance with large datasets"""
    print('TESTING PERFORMANCE WITH LARGE DATASETS')
    print('=' * 45)

    initial_memory = get_memory_usage()
    print(f'Initial memory usage: {initial_memory:.2f} MB')

    # Test different dataset sizes
    dataset_sizes = [1000, 5000, 10000, 25000, 50000]

    for size in dataset_sizes:
        print(f'\n=== Testing with {size:,} records ===')

        # Test 1: Dataset Creation Performance
        print(f'1. Creating {size:,} record dataset...')
        start_time = time.time()

        # Create realistic property data
        np.random.seed(42)
        data = {
            'Property_Name': [f'Property_{i:06d}' for i in range(size)],
            'Property_Type': np.random.choice(
                ['Industrial', 'Warehouse', 'Flex Space', 'Manufacturing', 'Distribution'],
                size
            ),
            'City': np.random.choice(
                ['Los Angeles', 'Chicago', 'Miami', 'Dallas', 'Phoenix', 'Houston',
                 'Atlanta', 'Detroit', 'Cleveland', 'Memphis'],
                size
            ),
            'State': np.random.choice(
                ['CA', 'IL', 'FL', 'TX', 'AZ', 'GA', 'MI', 'OH', 'TN'],
                size
            ),
            'Building_SqFt': np.random.randint(5000, 500000, size),
            'Lot_Size_Acres': np.random.uniform(0.1, 50.0, size),
            'Year_Built': np.random.randint(1960, 2024, size),
            'Sold_Price': np.random.randint(100000, 25000000, size),
            'Price_per_SqFt': np.random.uniform(10, 200, size)
        }

        df = pd.DataFrame(data)
        creation_time = time.time() - start_time
        creation_memory = get_memory_usage()

        print(f'   [PASS] Dataset created in {creation_time:.3f} seconds')
        print(f'   [INFO] Memory usage: {creation_memory:.2f} MB')

        # Test 2: Basic Operations Performance
        print('2. Testing basic operations...')
        start_time = time.time()

        # Basic statistics
        avg_price = df['Sold_Price'].mean()
        median_price = df['Sold_Price'].median()
        std_price = df['Sold_Price'].std()
        total_sqft = df['Building_SqFt'].sum()

        basic_ops_time = time.time() - start_time
        print(f'   [PASS] Basic operations completed in {basic_ops_time:.3f} seconds')

        # Test 3: Filtering Performance
        print('3. Testing filtering performance...')
        start_time = time.time()

        # Multiple filter conditions
        industrial_filter = df['Property_Type'].str.contains('Industrial|Warehouse|Flex', na=False)
        size_filter = (df['Building_SqFt'] >= 25000) & (df['Building_SqFt'] <= 200000)
        price_filter = (df['Sold_Price'] >= 500000) & (df['Sold_Price'] <= 5000000)
        year_filter = df['Year_Built'] >= 1990

        # Apply combined filter
        combined_filter = industrial_filter & size_filter & price_filter & year_filter
        filtered_df = df[combined_filter]

        filter_time = time.time() - start_time
        filter_ratio = len(filtered_df) / len(df) * 100

        print(f'   [PASS] Filtering completed in {filter_time:.3f} seconds')
        print(f'   [INFO] Filtered to {len(filtered_df):,} records ({filter_ratio:.1f}%)')

        # Test 4: Grouping and Aggregation Performance
        print('4. Testing grouping and aggregation...')
        start_time = time.time()

        # Group by property type
        type_stats = df.groupby('Property_Type').agg({
            'Sold_Price': ['mean', 'median', 'count'],
            'Building_SqFt': ['mean', 'sum'],
            'Price_per_SqFt': 'mean'
        }).round(2)

        # Group by state
        state_stats = df.groupby('State').agg({
            'Sold_Price': ['mean', 'count'],
            'Building_SqFt': 'sum'
        }).round(2)

        grouping_time = time.time() - start_time
        print(f'   [PASS] Grouping operations completed in {grouping_time:.3f} seconds')

        # Test 5: Sorting Performance
        print('5. Testing sorting performance...')
        start_time = time.time()

        # Sort by multiple columns
        sorted_df = df.sort_values(['Sold_Price', 'Building_SqFt'], ascending=[False, False])

        sorting_time = time.time() - start_time
        print(f'   [PASS] Sorting completed in {sorting_time:.3f} seconds')

        # Test 6: Export Performance (CSV)
        print('6. Testing export performance...')
        start_time = time.time()

        # Export to CSV (in memory)
        csv_output = df.to_csv(index=False)

        export_time = time.time() - start_time
        export_size_mb = len(csv_output.encode('utf-8')) / 1024 / 1024

        print(f'   [PASS] CSV export completed in {export_time:.3f} seconds')
        print(f'   [INFO] Export size: {export_size_mb:.2f} MB')

        # Test 7: Categorical Data Performance
        print('7. Testing categorical data conversion...')
        start_time = time.time()

        # Convert to categorical and back
        df_cat = df.copy()
        df_cat['Property_Type'] = df_cat['Property_Type'].astype('category')
        df_cat['State'] = df_cat['State'].astype('category')

        # Convert categorical columns back to numeric where needed
        if df_cat['Sold_Price'].dtype.name == 'category':
            df_cat['Sold_Price'] = pd.to_numeric(df_cat['Sold_Price'].astype(str), errors='coerce')

        categorical_time = time.time() - start_time
        print(f'   [PASS] Categorical operations completed in {categorical_time:.3f} seconds')

        # Test 8: Memory Efficiency
        current_memory = get_memory_usage()
        memory_increase = current_memory - initial_memory

        print(f'8. Memory efficiency check...')
        print(f'   [INFO] Current memory: {current_memory:.2f} MB')
        print(f'   [INFO] Memory increase: {memory_increase:.2f} MB')

        # Calculate memory per record
        memory_per_record = memory_increase / size * 1024  # KB per record
        print(f'   [INFO] Memory per record: {memory_per_record:.3f} KB')

        # Performance summary for this dataset size
        total_time = creation_time + basic_ops_time + filter_time + grouping_time + sorting_time + export_time + categorical_time

        print(f'\n   PERFORMANCE SUMMARY for {size:,} records:')
        print(f'   Total processing time: {total_time:.3f} seconds')
        print(f'   Records per second: {size/total_time:,.0f}')
        print(f'   Memory efficiency: {memory_per_record:.3f} KB/record')

        # Performance benchmarks
        if total_time < size / 1000:  # Less than 1ms per record
            print(f'   [EXCELLENT] Performance is excellent')
        elif total_time < size / 500:  # Less than 2ms per record
            print(f'   [GOOD] Performance is good')
        elif total_time < size / 100:  # Less than 10ms per record
            print(f'   [ACCEPTABLE] Performance is acceptable')
        else:
            print(f'   [SLOW] Performance may need optimization')

        # Clean up for next iteration
        del df, filtered_df, sorted_df, type_stats, state_stats, df_cat, csv_output
        gc.collect()

        # Stop if memory usage gets too high (prevent system issues)
        if current_memory > 2000:  # 2GB limit
            print(f'\n   [WARNING] Memory usage too high, stopping at {size:,} records')
            break

    # Test 9: Stress Test with Real-World Operations
    print(f'\n=== STRESS TEST: Real-World Scenario ===')
    stress_test_size = 25000

    print(f'Creating stress test dataset with {stress_test_size:,} records...')
    start_time = time.time()

    # Create stress test data
    stress_data = {
        'Property_Name': [f'StressTest_Property_{i:06d}' for i in range(stress_test_size)],
        'Property_Type': np.random.choice(
            ['Industrial', 'Warehouse', 'Flex Space', 'Manufacturing', 'Distribution', 'Cold Storage'],
            stress_test_size
        ),
        'City': np.random.choice([
            'Los Angeles', 'Chicago', 'Miami', 'Dallas', 'Phoenix', 'Houston', 'Atlanta',
            'Detroit', 'Cleveland', 'Memphis', 'Indianapolis', 'Columbus', 'Kansas City'
        ], stress_test_size),
        'State': np.random.choice(
            ['CA', 'IL', 'FL', 'TX', 'AZ', 'GA', 'MI', 'OH', 'TN', 'IN', 'MO', 'KS'],
            stress_test_size
        ),
        'Building_SqFt': np.random.randint(1000, 1000000, stress_test_size),
        'Lot_Size_Acres': np.random.uniform(0.05, 100.0, stress_test_size),
        'Year_Built': np.random.randint(1950, 2024, stress_test_size),
        'Sold_Price': np.random.randint(50000, 50000000, stress_test_size),
        'Lease_Rate': np.random.uniform(2.0, 25.0, stress_test_size),
        'Occupancy_Rate': np.random.uniform(0.0, 100.0, stress_test_size)
    }

    stress_df = pd.DataFrame(stress_data)
    stress_creation_time = time.time() - start_time

    print(f'Stress test dataset created in {stress_creation_time:.3f} seconds')

    # Simulate real-world usage patterns
    operations = [
        ('Filter by property type', lambda: stress_df[stress_df['Property_Type'] == 'Industrial']),
        ('Price range filter', lambda: stress_df[(stress_df['Sold_Price'] >= 1000000) & (stress_df['Sold_Price'] <= 10000000)]),
        ('Size filter', lambda: stress_df[stress_df['Building_SqFt'] >= 50000]),
        ('State grouping', lambda: stress_df.groupby('State')['Sold_Price'].mean()),
        ('Type analysis', lambda: stress_df.groupby('Property_Type').agg({'Sold_Price': 'mean', 'Building_SqFt': 'sum'})),
        ('Sort by price', lambda: stress_df.sort_values('Sold_Price', ascending=False).head(1000)),
        ('Calculate metrics', lambda: stress_df.assign(Price_per_SqFt=stress_df['Sold_Price']/stress_df['Building_SqFt'])),
        ('Export subset', lambda: stress_df.sample(5000).to_csv(index=False))
    ]

    total_operations_time = 0
    for op_name, op_func in operations:
        start_time = time.time()
        result = op_func()
        op_time = time.time() - start_time
        total_operations_time += op_time
        print(f'   [PASS] {op_name}: {op_time:.3f} seconds')

    final_memory = get_memory_usage()
    print(f'\nSTRESS TEST SUMMARY:')
    print(f'Dataset size: {stress_test_size:,} records')
    print(f'Total operations time: {total_operations_time:.3f} seconds')
    print(f'Average operation time: {total_operations_time/len(operations):.3f} seconds')
    print(f'Final memory usage: {final_memory:.2f} MB')

    # Clean up
    del stress_df
    gc.collect()

    print('\n[COMPLETE] Performance testing with large datasets completed!')
    return True

if __name__ == "__main__":
    success = test_large_dataset_performance()
    print(f'\nPerformance Test Result: {"PASS" if success else "FAIL"}')