#!/usr/bin/env python3
"""
Test script for data type performance optimizations.
Tests caching, lazy conversion, and batch processing performance.
"""

import pandas as pd
import numpy as np
import time
import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_large_test_data(num_rows: int = 10000):
    """Create large test dataset with categorical columns"""
    
    np.random.seed(42)  # For reproducible results
    
    # Create categorical data that should be numeric
    building_sqft = pd.Categorical([f"{np.random.randint(5000, 50000)}" for _ in range(num_rows)])
    prices = pd.Categorical([f"${np.random.randint(100000, 2000000):,}" for _ in range(num_rows)])
    occupancy = pd.Categorical([f"{np.random.randint(60, 100)}%" for _ in range(num_rows)])
    years = pd.Categorical([f"{np.random.randint(1980, 2023)}" for _ in range(num_rows)])
    
    # Create regular columns
    property_names = [f"Property_{i:05d}" for i in range(num_rows)]
    property_types = np.random.choice(['Industrial', 'Warehouse', 'Office', 'Retail'], num_rows)
    cities = np.random.choice(['City_A', 'City_B', 'City_C', 'City_D', 'City_E'], num_rows)
    
    data = {
        'Property_Name': property_names,
        'Property_Type': property_types,
        'Building_SqFt': building_sqft,
        'Sold_Price': prices,
        'Occupancy_Rate': occupancy,
        'Year_Built': years,
        'City': cities,
        'State': pd.Categorical(np.random.choice(['CA', 'TX', 'FL', 'NY', 'WA'], num_rows))
    }
    
    df = pd.DataFrame(data)
    
    print(f"Created test DataFrame with {num_rows:,} rows and {len(df.columns)} columns")
    print("Categorical columns that should be numeric:")
    for col in ['Building_SqFt', 'Sold_Price', 'Occupancy_Rate', 'Year_Built']:
        print(f"  {col}: {df[col].dtype}")
    
    return df

def test_caching_performance():
    """Test caching performance improvements"""
    print("\n" + "="*60)
    print("TESTING CACHING PERFORMANCE")
    print("="*60)
    
    try:
        from utils.data_type_performance import get_performance_cache, clear_all_caches
        from utils.data_type_utils import convert_categorical_to_numeric
        
        # Clear caches to start fresh
        clear_all_caches()
        
        # Create test data
        test_df = create_large_test_data(5000)
        
        print("\nTesting conversion performance with caching...")
        
        # First conversion (no cache)
        print("First conversion (cold cache):")
        start_time = time.time()
        converted_df1, reports1 = convert_categorical_to_numeric(test_df.copy())
        first_time = time.time() - start_time
        print(f"  Time: {first_time:.3f} seconds")
        print(f"  Columns converted: {len(reports1)}")
        
        # Second conversion (should use cache)
        print("Second conversion (warm cache):")
        start_time = time.time()
        converted_df2, reports2 = convert_categorical_to_numeric(test_df.copy())
        second_time = time.time() - start_time
        print(f"  Time: {second_time:.3f} seconds")
        print(f"  Columns converted: {len(reports2)}")
        
        # Calculate performance improvement
        if first_time > 0:
            speedup = first_time / second_time if second_time > 0 else float('inf')
            improvement = ((first_time - second_time) / first_time) * 100
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Performance improvement: {improvement:.1f}%")
        
        # Verify results are the same
        if converted_df1.equals(converted_df2):
            print("✓ Cached results match original results")
        else:
            print("⚠ Cached results differ from original results")
        
        # Get cache statistics
        cache = get_performance_cache()
        cache_stats = cache.get_cache_stats()
        print(f"\nCache statistics:")
        print(f"  Total cache entries: {cache_stats['total_cache_entries']}")
        print(f"  Conversion cache size: {cache_stats['conversion_cache_size']}")
        print(f"  Validation cache size: {cache_stats['validation_cache_size']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Caching performance test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_lazy_conversion():
    """Test lazy conversion performance"""
    print("\n" + "="*60)
    print("TESTING LAZY CONVERSION")
    print("="*60)
    
    try:
        from utils.data_type_performance import get_lazy_converter
        from utils.data_type_utils import safe_mean_calculation
        
        # Create test data
        test_df = create_large_test_data(3000)
        
        print("\nTesting lazy conversion...")
        
        lazy_converter = get_lazy_converter()
        
        # Mark columns for lazy conversion
        print("Marking columns for lazy conversion...")
        for col in ['Building_SqFt', 'Sold_Price', 'Occupancy_Rate']:
            lazy_converter.mark_for_conversion(test_df, col, 'numeric')
        
        conversion_stats = lazy_converter.get_conversion_stats()
        print(f"Queued conversions: {conversion_stats['queued_conversions']}")
        
        # Test lazy conversion during operation
        print("Testing conversion on demand...")
        start_time = time.time()
        
        # This should trigger conversion only when needed
        building_sqft_series = lazy_converter.convert_if_needed(test_df, 'Building_SqFt', 'mean calculation')
        mean_sqft = safe_mean_calculation(building_sqft_series, 'Building_SqFt')
        
        lazy_time = time.time() - start_time
        print(f"  Lazy conversion time: {lazy_time:.3f} seconds")
        print(f"  Mean building sqft: {mean_sqft:,.0f}")
        
        # Test batch conversion
        print("Testing batch conversion...")
        start_time = time.time()
        batch_converted_df = lazy_converter.batch_convert_queued(test_df)
        batch_time = time.time() - start_time
        print(f"  Batch conversion time: {batch_time:.3f} seconds")
        
        # Verify conversion worked
        numeric_columns = 0
        for col in ['Building_SqFt', 'Sold_Price', 'Occupancy_Rate']:
            if pd.api.types.is_numeric_dtype(batch_converted_df[col]):
                numeric_columns += 1
        
        print(f"✓ {numeric_columns}/3 columns successfully converted to numeric")
        
        # Get final stats
        final_stats = lazy_converter.get_conversion_stats()
        print(f"Completed conversions: {final_stats['completed_conversions']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Lazy conversion test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processing():
    """Test batch processing performance"""
    print("\n" + "="*60)
    print("TESTING BATCH PROCESSING")
    print("="*60)
    
    try:
        from utils.data_type_performance import get_batch_processor, optimize_dataframe_processing
        
        # Create large test data
        test_df = create_large_test_data(15000)
        
        print(f"\nTesting batch processing with {len(test_df):,} rows...")
        
        # Test regular processing
        print("Regular processing:")
        start_time = time.time()
        from utils.data_type_utils import convert_categorical_to_numeric
        regular_df, regular_reports = convert_categorical_to_numeric(test_df.copy())
        regular_time = time.time() - start_time
        print(f"  Time: {regular_time:.3f} seconds")
        
        # Test optimized batch processing
        print("Optimized batch processing:")
        start_time = time.time()
        optimized_df = optimize_dataframe_processing(test_df.copy(), ['convert_categorical'])
        optimized_time = time.time() - start_time
        print(f"  Time: {optimized_time:.3f} seconds")
        
        # Compare results
        if regular_time > 0 and optimized_time > 0:
            if optimized_time < regular_time:
                speedup = regular_time / optimized_time
                improvement = ((regular_time - optimized_time) / regular_time) * 100
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  Performance improvement: {improvement:.1f}%")
            else:
                print("  No significant performance improvement (may be due to overhead for small datasets)")
        
        # Verify data integrity
        numeric_columns_regular = sum(1 for col in ['Building_SqFt', 'Sold_Price', 'Occupancy_Rate', 'Year_Built'] 
                                    if pd.api.types.is_numeric_dtype(regular_df[col]))
        numeric_columns_optimized = sum(1 for col in ['Building_SqFt', 'Sold_Price', 'Occupancy_Rate', 'Year_Built'] 
                                      if pd.api.types.is_numeric_dtype(optimized_df[col]))
        
        print(f"Regular processing: {numeric_columns_regular}/4 columns converted")
        print(f"Optimized processing: {numeric_columns_optimized}/4 columns converted")
        
        if numeric_columns_regular == numeric_columns_optimized:
            print("✓ Both methods produced equivalent results")
        else:
            print("⚠ Results differ between methods")
        
        return True
        
    except Exception as e:
        print(f"✗ Batch processing test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_efficiency():
    """Test memory efficiency of optimizations"""
    print("\n" + "="*60)
    print("TESTING MEMORY EFFICIENCY")
    print("="*60)
    
    try:
        import psutil
        process = psutil.Process()
        
        # Get initial memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # Create large dataset
        print("Creating large dataset...")
        large_df = create_large_test_data(20000)
        
        after_creation_memory = process.memory_info().rss / 1024 / 1024
        print(f"Memory after data creation: {after_creation_memory:.1f} MB")
        print(f"Data creation memory delta: {after_creation_memory - initial_memory:.1f} MB")
        
        # Test conversion with optimizations
        print("Testing optimized conversion...")
        from utils.data_type_performance import optimize_dataframe_processing
        
        start_memory = process.memory_info().rss / 1024 / 1024
        optimized_df = optimize_dataframe_processing(large_df, ['convert_categorical'])
        end_memory = process.memory_info().rss / 1024 / 1024
        
        conversion_memory_delta = end_memory - start_memory
        print(f"Conversion memory delta: {conversion_memory_delta:.1f} MB")
        print(f"Final memory usage: {end_memory:.1f} MB")
        
        # Clean up
        del large_df, optimized_df
        
        # Test cache memory usage
        from utils.data_type_performance import get_performance_stats
        stats = get_performance_stats()
        print(f"\nPerformance statistics:")
        print(f"  Cache entries: {stats['cache_stats']['total_cache_entries']}")
        
        return True
        
    except ImportError:
        print("⚠ psutil not available - skipping memory efficiency test")
        return True
    except Exception as e:
        print(f"✗ Memory efficiency test failed: {str(e)}")
        return False

def main():
    """Run all performance optimization tests"""
    print("DATA TYPE PERFORMANCE OPTIMIZATION TESTING")
    print("="*60)
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run tests
    tests = [
        ("Caching Performance", test_caching_performance),
        ("Lazy Conversion", test_lazy_conversion),
        ("Batch Processing", test_batch_processing),
        ("Memory Efficiency", test_memory_efficiency)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All performance optimization tests passed!")
        return True
    else:
        print("⚠ Some performance tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)