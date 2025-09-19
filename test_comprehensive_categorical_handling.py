#!/usr/bin/env python3
"""
Comprehensive test suite for categorical data handling.
Tests all mathematical operations, filtering, and edge cases with categorical inputs.
"""

import pandas as pd
import numpy as np
import sys
import os
import time
from datetime import datetime
from typing import Dict, List, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_comprehensive_test_data():
    """Create comprehensive test data with various categorical scenarios"""
    
    # Test data with different categorical patterns
    test_scenarios = {
        'simple_numeric': pd.Categorical(['100', '200', '300', '400', '500']),
        'currency_format': pd.Categorical(['$1,000', '$2,500', '$3,750', '$4,200', '$5,000']),
        'percentage_format': pd.Categorical(['85%', '92%', '78%', '88%', '95%']),
        'decimal_numbers': pd.Categorical(['10.5', '20.75', '30.25', '40.0', '50.5']),
        'mixed_valid_invalid': pd.Categorical(['100', '200', 'invalid', '400', '500']),
        'with_nulls': pd.Categorical(['100', '200', None, '400', '500']),
        'negative_numbers': pd.Categorical(['-100', '200', '-300', '400', '-500']),
        'scientific_notation': pd.Categorical(['1e2', '2e2', '3e2', '4e2', '5e2']),
        'year_format': pd.Categorical(['1995', '2000', '1988', '1992', '2005']),
        'non_numeric_categorical': pd.Categorical(['Red', 'Blue', 'Green', 'Yellow', 'Purple']),
        'empty_strings': pd.Categorical(['100', '', '300', '400', '500']),
        'whitespace_issues': pd.Categorical([' 100 ', '200', '300 ', ' 400', '500']),
        'comma_thousands': pd.Categorical(['1,000', '2,000', '3,000', '4,000', '5,000']),
        'parentheses_negative': pd.Categorical(['(100)', '200', '(300)', '400', '(500)']),
        'mixed_formats': pd.Categorical(['$1,000', '85%', '2.5', '(300)', '1e3'])
    }
    
    # Create DataFrame with all scenarios
    data = {}
    for scenario_name, categorical_data in test_scenarios.items():
        data[scenario_name] = categorical_data
    
    # Add some regular columns for context
    data['property_id'] = [f'PROP_{i:03d}' for i in range(5)]
    data['property_type'] = ['Industrial', 'Warehouse', 'Office', 'Retail', 'Mixed']
    
    df = pd.DataFrame(data)
    
    print(f"Created comprehensive test DataFrame with {len(df)} rows and {len(df.columns)} columns")
    print("Test scenarios included:")
    for scenario in test_scenarios.keys():
        print(f"  - {scenario}")
    
    return df

def test_mathematical_operations():
    """Test all mathematical operations with categorical data"""
    print("\n" + "="*60)
    print("TESTING MATHEMATICAL OPERATIONS")
    print("="*60)
    
    try:
        from utils.data_type_utils import (
            safe_mean_calculation, safe_sum_calculation, safe_min_calculation,
            safe_max_calculation, safe_std_calculation, safe_median_calculation,
            safe_count_calculation
        )
        
        test_df = create_comprehensive_test_data()
        
        # Test operations on each categorical column
        operations = [
            ('Mean', safe_mean_calculation),
            ('Sum', safe_sum_calculation),
            ('Min', safe_min_calculation),
            ('Max', safe_max_calculation),
            ('Standard Deviation', safe_std_calculation),
            ('Median', safe_median_calculation),
            ('Count', safe_count_calculation)
        ]
        
        results = {}
        
        for col in test_df.columns:
            if test_df[col].dtype.name == 'category':
                print(f"\nTesting operations on column: {col}")
                col_results = {}
                
                for op_name, op_func in operations:
                    try:
                        start_time = time.time()
                        result = op_func(test_df[col], col)
                        execution_time = time.time() - start_time
                        
                        col_results[op_name] = {
                            'result': result,
                            'execution_time': execution_time,
                            'success': True,
                            'error': None
                        }
                        
                        if pd.isna(result):
                            print(f"  {op_name}: N/A (no valid numeric data)")
                        else:
                            print(f"  {op_name}: {result} ({execution_time:.4f}s)")
                            
                    except Exception as e:
                        col_results[op_name] = {
                            'result': None,
                            'execution_time': 0,
                            'success': False,
                            'error': str(e)
                        }
                        print(f"  {op_name}: ERROR - {str(e)}")
                
                results[col] = col_results
        
        # Summary
        total_operations = sum(len(col_results) for col_results in results.values())
        successful_operations = sum(
            sum(1 for op_result in col_results.values() if op_result['success'])
            for col_results in results.values()
        )
        
        success_rate = (successful_operations / total_operations) * 100 if total_operations > 0 else 0
        
        print(f"\nMathematical Operations Summary:")
        print(f"  Total operations tested: {total_operations}")
        print(f"  Successful operations: {successful_operations}")
        print(f"  Success rate: {success_rate:.1f}%")
        
        return success_rate > 80  # Consider success if >80% of operations work
        
    except Exception as e:
        print(f"✗ Mathematical operations test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_comparison_operations():
    """Test comparison operations with categorical data"""
    print("\n" + "="*60)
    print("TESTING COMPARISON OPERATIONS")
    print("="*60)
    
    try:
        from utils.data_type_utils import safe_numerical_comparison, safe_range_filter
        
        test_df = create_comprehensive_test_data()
        
        # Test comparison operations
        comparison_tests = [
            ('Greater than 100', '>', 100),
            ('Less than 500', '<', 500),
            ('Greater than or equal to 200', '>=', 200),
            ('Less than or equal to 400', '<=', 400),
            ('Equal to 300', '==', 300),
            ('Not equal to 100', '!=', 100)
        ]
        
        results = {}
        
        for col in test_df.columns:
            if test_df[col].dtype.name == 'category':
                print(f"\nTesting comparisons on column: {col}")
                col_results = {}
                
                for test_name, operator, value in comparison_tests:
                    try:
                        start_time = time.time()
                        comparison_result = safe_numerical_comparison(
                            test_df[col], operator, value, col
                        )
                        execution_time = time.time() - start_time
                        
                        matches = comparison_result.sum()
                        total = len(comparison_result)
                        
                        col_results[test_name] = {
                            'matches': matches,
                            'total': total,
                            'execution_time': execution_time,
                            'success': True,
                            'error': None
                        }
                        
                        print(f"  {test_name}: {matches}/{total} matches ({execution_time:.4f}s)")
                        
                    except Exception as e:
                        col_results[test_name] = {
                            'matches': 0,
                            'total': 0,
                            'execution_time': 0,
                            'success': False,
                            'error': str(e)
                        }
                        print(f"  {test_name}: ERROR - {str(e)}")
                
                # Test range filtering
                try:
                    start_time = time.time()
                    range_result = safe_range_filter(test_df[col], 200, 400, col)
                    execution_time = time.time() - start_time
                    
                    matches = range_result.sum()
                    total = len(range_result)
                    
                    col_results['Range Filter (200-400)'] = {
                        'matches': matches,
                        'total': total,
                        'execution_time': execution_time,
                        'success': True,
                        'error': None
                    }
                    
                    print(f"  Range Filter (200-400): {matches}/{total} matches ({execution_time:.4f}s)")
                    
                except Exception as e:
                    col_results['Range Filter (200-400)'] = {
                        'matches': 0,
                        'total': 0,
                        'execution_time': 0,
                        'success': False,
                        'error': str(e)
                    }
                    print(f"  Range Filter (200-400): ERROR - {str(e)}")
                
                results[col] = col_results
        
        # Summary
        total_comparisons = sum(len(col_results) for col_results in results.values())
        successful_comparisons = sum(
            sum(1 for comp_result in col_results.values() if comp_result['success'])
            for col_results in results.values()
        )
        
        success_rate = (successful_comparisons / total_comparisons) * 100 if total_comparisons > 0 else 0
        
        print(f"\nComparison Operations Summary:")
        print(f"  Total comparisons tested: {total_comparisons}")
        print(f"  Successful comparisons: {successful_comparisons}")
        print(f"  Success rate: {success_rate:.1f}%")
        
        return success_rate > 80
        
    except Exception as e:
        print(f"✗ Comparison operations test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_data_type_conversion():
    """Test comprehensive data type conversion scenarios"""
    print("\n" + "="*60)
    print("TESTING DATA TYPE CONVERSION")
    print("="*60)
    
    try:
        from utils.data_type_utils import (
            convert_categorical_to_numeric, safe_numeric_conversion,
            detect_categorical_numeric_columns
        )
        
        test_df = create_comprehensive_test_data()
        
        print("Testing categorical column detection...")
        detection_results = detect_categorical_numeric_columns(test_df)
        
        print(f"Detected {len(detection_results)} columns for conversion:")
        for col, analysis in detection_results.items():
            confidence = analysis['conversion_confidence']
            print(f"  {col}: {confidence:.2f} confidence")
        
        print("\nTesting individual column conversions...")
        conversion_results = {}
        
        for col in test_df.columns:
            if test_df[col].dtype.name == 'category':
                print(f"\nTesting conversion of column: {col}")
                
                try:
                    start_time = time.time()
                    converted_series, report = safe_numeric_conversion(test_df[col], col)
                    execution_time = time.time() - start_time
                    
                    conversion_results[col] = {
                        'success': report['conversion_successful'],
                        'values_converted': report['values_converted'],
                        'values_failed': report['values_failed'],
                        'final_dtype': str(converted_series.dtype),
                        'execution_time': execution_time,
                        'error': None
                    }
                    
                    if report['conversion_successful']:
                        success_rate = (report['values_converted'] / len(test_df)) * 100
                        print(f"  ✓ Conversion successful: {success_rate:.1f}% success rate")
                        print(f"  Final data type: {converted_series.dtype}")
                        print(f"  Values converted: {report['values_converted']}")
                        print(f"  Values failed: {report['values_failed']}")
                    else:
                        print(f"  ✗ Conversion failed")
                        
                except Exception as e:
                    conversion_results[col] = {
                        'success': False,
                        'values_converted': 0,
                        'values_failed': 0,
                        'final_dtype': 'unknown',
                        'execution_time': 0,
                        'error': str(e)
                    }
                    print(f"  ✗ Conversion error: {str(e)}")
        
        print("\nTesting batch conversion...")
        try:
            start_time = time.time()
            batch_converted_df, batch_reports = convert_categorical_to_numeric(test_df)
            batch_execution_time = time.time() - start_time
            
            batch_successful = sum(1 for report in batch_reports.values() if report['conversion_successful'])
            batch_total = len(batch_reports)
            
            print(f"  Batch conversion completed in {batch_execution_time:.4f}s")
            print(f"  Successful conversions: {batch_successful}/{batch_total}")
            
            # Verify data integrity
            for col in batch_converted_df.columns:
                if col in batch_reports:
                    original_dtype = str(test_df[col].dtype)
                    final_dtype = str(batch_converted_df[col].dtype)
                    print(f"  {col}: {original_dtype} → {final_dtype}")
            
        except Exception as e:
            print(f"  ✗ Batch conversion failed: {str(e)}")
            return False
        
        # Summary
        total_conversions = len(conversion_results)
        successful_conversions = sum(1 for result in conversion_results.values() if result['success'])
        success_rate = (successful_conversions / total_conversions) * 100 if total_conversions > 0 else 0
        
        print(f"\nData Type Conversion Summary:")
        print(f"  Total columns tested: {total_conversions}")
        print(f"  Successful conversions: {successful_conversions}")
        print(f"  Success rate: {success_rate:.1f}%")
        
        return success_rate > 60  # Lower threshold due to intentionally difficult test cases
        
    except Exception as e:
        print(f"✗ Data type conversion test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling and recovery mechanisms"""
    print("\n" + "="*60)
    print("TESTING ERROR HANDLING")
    print("="*60)
    
    try:
        from utils.categorical_error_handler import (
            get_categorical_error_handler,
            handle_categorical_mean_error,
            handle_categorical_comparison_error
        )
        
        # Create problematic data that should trigger errors (all same length)
        problematic_data = {
            'completely_non_numeric': pd.Categorical(['Red', 'Blue', 'Green', 'Yellow', 'Purple']),
            'mixed_types': pd.Categorical(['100', 'invalid', '200', 'bad_data', '300']),
            'empty_categories': pd.Categorical([None, None, None, None, None]),  # Changed from empty to nulls
            'all_nulls': pd.Categorical([None, None, None, None, None])
        }
        
        problem_df = pd.DataFrame(problematic_data)
        
        print("Testing error handling with problematic data...")
        
        error_handler = get_categorical_error_handler()
        error_tests = []
        
        for col in problem_df.columns:
            print(f"\nTesting error handling for column: {col}")
            
            # Test mean calculation error handling
            try:
                # This should trigger an error
                result = problem_df[col].mean()
                print(f"  Mean calculation: Unexpected success - {result}")
                error_tests.append(False)
            except Exception as e:
                # Handle the error
                error_message = handle_categorical_mean_error(e, col)
                print(f"  Mean calculation error handled: ✓")
                print(f"  Error message length: {len(error_message)} characters")
                error_tests.append(True)
            
            # Test comparison error handling
            try:
                # This should trigger an error for non-numeric categorical data
                result = problem_df[col] > 100
                print(f"  Comparison operation: Unexpected success")
                error_tests.append(False)
            except Exception as e:
                # Handle the error
                error_message = handle_categorical_comparison_error(e, col, '>', 100)
                print(f"  Comparison error handled: ✓")
                print(f"  Error message length: {len(error_message)} characters")
                error_tests.append(True)
        
        # Test error statistics
        error_stats = error_handler.get_error_statistics()
        print(f"\nError Handler Statistics:")
        print(f"  Total errors logged: {error_stats['total_errors']}")
        print(f"  Error types: {list(error_stats['error_types'].keys())}")
        print(f"  Affected columns: {list(error_stats['affected_columns'].keys())}")
        
        # Test error log export
        try:
            log_file = error_handler.export_error_log('test_error_log.json')
            print(f"  Error log exported to: {log_file}")
            
            # Clean up
            if os.path.exists(log_file):
                os.remove(log_file)
                
        except Exception as e:
            print(f"  Error log export failed: {str(e)}")
        
        success_rate = (sum(error_tests) / len(error_tests)) * 100 if error_tests else 0
        
        print(f"\nError Handling Summary:")
        print(f"  Total error scenarios tested: {len(error_tests)}")
        print(f"  Successfully handled errors: {sum(error_tests)}")
        print(f"  Error handling success rate: {success_rate:.1f}%")
        
        return success_rate > 80
        
    except Exception as e:
        print(f"✗ Error handling test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_with_large_datasets():
    """Test performance with large datasets"""
    print("\n" + "="*60)
    print("TESTING PERFORMANCE WITH LARGE DATASETS")
    print("="*60)
    
    try:
        from utils.data_type_utils import convert_categorical_to_numeric
        from utils.data_type_performance import get_performance_stats, clear_all_caches
        
        # Create large dataset
        print("Creating large test dataset...")
        large_size = 50000
        
        large_data = {
            'prices': pd.Categorical([f"${np.random.randint(100000, 1000000):,}" for _ in range(large_size)]),
            'percentages': pd.Categorical([f"{np.random.randint(60, 100)}%" for _ in range(large_size)]),
            'square_feet': pd.Categorical([f"{np.random.randint(1000, 10000)}" for _ in range(large_size)]),
            'years': pd.Categorical([f"{np.random.randint(1980, 2024)}" for _ in range(large_size)])
        }
        
        large_df = pd.DataFrame(large_data)
        print(f"Created dataset with {len(large_df):,} rows and {len(large_df.columns)} columns")
        
        # Clear caches for fair testing
        clear_all_caches()
        
        # Test conversion performance
        print("Testing conversion performance...")
        start_time = time.time()
        converted_df, reports = convert_categorical_to_numeric(large_df)
        conversion_time = time.time() - start_time
        
        print(f"  Conversion completed in {conversion_time:.3f} seconds")
        print(f"  Processing rate: {len(large_df) / conversion_time:,.0f} rows/second")
        
        # Verify results
        successful_conversions = sum(1 for report in reports.values() if report['conversion_successful'])
        print(f"  Successful conversions: {successful_conversions}/{len(reports)}")
        
        # Test performance statistics
        perf_stats = get_performance_stats()
        print(f"  Cache entries created: {perf_stats['cache_stats']['total_cache_entries']}")
        
        # Performance benchmarks
        acceptable_time = 10.0  # seconds
        acceptable_rate = 1000   # rows per second
        
        time_ok = conversion_time < acceptable_time
        rate_ok = (len(large_df) / conversion_time) > acceptable_rate
        
        print(f"\nPerformance Benchmarks:")
        print(f"  Time under {acceptable_time}s: {'✓' if time_ok else '✗'} ({conversion_time:.3f}s)")
        print(f"  Rate over {acceptable_rate} rows/s: {'✓' if rate_ok else '✗'} ({len(large_df) / conversion_time:,.0f} rows/s)")
        
        return time_ok and rate_ok and successful_conversions > 0
        
    except Exception as e:
        print(f"✗ Performance test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_regression_scenarios():
    """Test specific regression scenarios to prevent future issues"""
    print("\n" + "="*60)
    print("TESTING REGRESSION SCENARIOS")
    print("="*60)
    
    try:
        from utils.data_type_utils import safe_mean_calculation, safe_numerical_comparison
        
        # Specific scenarios that have caused issues in the past
        regression_tests = [
            {
                'name': 'Percentage values with % symbol',
                'data': pd.Categorical(['85%', '92%', '78%', '88%', '95%']),
                'expected_mean': 87.6,
                'tolerance': 0.1
            },
            {
                'name': 'Currency values with $ and commas',
                'data': pd.Categorical(['$1,000', '$2,500', '$3,750', '$4,200', '$5,000']),
                'expected_mean': 3290.0,
                'tolerance': 1.0
            },
            {
                'name': 'Mixed valid and invalid values',
                'data': pd.Categorical(['100', '200', 'invalid', '400', '500']),
                'expected_mean': 300.0,  # Mean of valid values only
                'tolerance': 1.0
            },
            {
                'name': 'Negative numbers in parentheses',
                'data': pd.Categorical(['(100)', '200', '(300)', '400', '(500)']),
                'expected_mean': -60.0,  # -100 + 200 - 300 + 400 - 500 = -300 / 5 = -60
                'tolerance': 1.0
            },
            {
                'name': 'Scientific notation',
                'data': pd.Categorical(['1e2', '2e2', '3e2', '4e2', '5e2']),
                'expected_mean': 300.0,
                'tolerance': 1.0
            }
        ]
        
        regression_results = []
        
        for test_case in regression_tests:
            print(f"\nTesting: {test_case['name']}")
            
            try:
                # Test mean calculation
                calculated_mean = safe_mean_calculation(test_case['data'], test_case['name'])
                
                if pd.isna(calculated_mean):
                    print(f"  Mean calculation returned NaN")
                    regression_results.append(False)
                else:
                    expected = test_case['expected_mean']
                    tolerance = test_case['tolerance']
                    
                    if abs(calculated_mean - expected) <= tolerance:
                        print(f"  ✓ Mean calculation: {calculated_mean} (expected: {expected})")
                        regression_results.append(True)
                    else:
                        print(f"  ✗ Mean calculation: {calculated_mean} (expected: {expected}, tolerance: {tolerance})")
                        regression_results.append(False)
                
                # Test comparison operations
                comparison_result = safe_numerical_comparison(test_case['data'], '>', 0, test_case['name'])
                positive_count = comparison_result.sum()
                print(f"  Comparison (> 0): {positive_count} positive values found")
                
            except Exception as e:
                print(f"  ✗ Test failed with error: {str(e)}")
                regression_results.append(False)
        
        success_rate = (sum(regression_results) / len(regression_results)) * 100 if regression_results else 0
        
        print(f"\nRegression Test Summary:")
        print(f"  Total regression tests: {len(regression_results)}")
        print(f"  Passed tests: {sum(regression_results)}")
        print(f"  Success rate: {success_rate:.1f}%")
        
        return success_rate == 100.0  # All regression tests must pass
        
    except Exception as e:
        print(f"✗ Regression test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive categorical data handling tests"""
    print("COMPREHENSIVE CATEGORICAL DATA HANDLING TESTS")
    print("="*60)
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all test suites
    test_suites = [
        ("Mathematical Operations", test_mathematical_operations),
        ("Comparison Operations", test_comparison_operations),
        ("Data Type Conversion", test_data_type_conversion),
        ("Error Handling", test_error_handling),
        ("Performance with Large Datasets", test_performance_with_large_datasets),
        ("Regression Scenarios", test_regression_scenarios)
    ]
    
    results = {}
    total_start_time = time.time()
    
    for test_name, test_func in test_suites:
        print(f"\nRunning {test_name} tests...")
        suite_start_time = time.time()
        
        results[test_name] = test_func()
        
        suite_time = time.time() - suite_start_time
        print(f"{test_name} completed in {suite_time:.3f} seconds")
    
    total_time = time.time() - total_start_time
    
    # Final summary
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall Results:")
    print(f"  Test suites passed: {passed}/{total}")
    print(f"  Success rate: {(passed/total)*100:.1f}%")
    print(f"  Total execution time: {total_time:.3f} seconds")
    
    if passed == total:
        print("\n✓ All comprehensive tests passed!")
        print("  Categorical data handling is working correctly across all scenarios.")
        return True
    else:
        print(f"\n⚠ {total - passed} test suite(s) failed.")
        print("  Review the detailed output above for specific issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)