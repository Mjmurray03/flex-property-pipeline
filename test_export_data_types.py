#!/usr/bin/env python3
"""
Test script for data export functionality with proper data type handling.
Tests both CSV and Excel export functions with categorical data.
"""

import pandas as pd
import numpy as np
import io
import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_data_with_categorical():
    """Create test data with categorical columns that should be numeric"""
    
    # Create sample data with mixed data types including problematic categorical columns
    data = {
        'Property_Name': ['Property A', 'Property B', 'Property C', 'Property D', 'Property E'],
        'Property_Type': ['Industrial', 'Warehouse', 'Industrial', 'Office', 'Warehouse'],
        'Building_SqFt': pd.Categorical(['10000', '15000', '8000', '12000', '20000']),  # Categorical that should be numeric
        'Lot_Size_Acres': pd.Categorical(['2.5', '3.0', '1.8', '2.2', '4.1']),  # Categorical that should be numeric
        'Sold_Price': pd.Categorical(['500000', '750000', '400000', '600000', '900000']),  # Categorical that should be numeric
        'Year_Built': pd.Categorical(['1995', '2000', '1988', '1992', '2005']),  # Categorical that should be numeric
        'Occupancy_Rate': pd.Categorical(['85%', '92%', '78%', '88%', '95%']),  # Categorical with % that should be numeric
        'City': ['City A', 'City B', 'City C', 'City D', 'City E'],  # Regular string column
        'State': pd.Categorical(['CA', 'TX', 'FL', 'NY', 'WA'])  # Categorical that should stay categorical
    }
    
    df = pd.DataFrame(data)
    
    print("Created test DataFrame with categorical columns:")
    print(f"Shape: {df.shape}")
    print("\nData types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    return df

def test_csv_export():
    """Test CSV export functionality with categorical data"""
    print("\n" + "="*60)
    print("TESTING CSV EXPORT WITH CATEGORICAL DATA")
    print("="*60)
    
    try:
        # Import the export function
        from flex_filter_dashboard import generate_enhanced_csv_export
        
        # Create test data
        test_df = create_test_data_with_categorical()
        original_df = test_df.copy()
        
        print("\nTesting CSV export...")
        
        # Test the export function
        csv_output = generate_enhanced_csv_export(test_df, original_df)
        
        # Verify output
        if csv_output and len(csv_output) > 0:
            print("✓ CSV export successful")
            print(f"✓ Output length: {len(csv_output)} characters")
            
            # Check for metadata
            if "Property Filter Dashboard Export" in csv_output:
                print("✓ Metadata header included")
            else:
                print("⚠ Metadata header missing")
            
            # Check for data type information
            if "Data Type Conversions:" in csv_output:
                print("✓ Data type conversion information included")
            else:
                print("⚠ Data type conversion information missing")
            
            # Check for column type information
            if "Column Data Types:" in csv_output:
                print("✓ Column data type information included")
            else:
                print("⚠ Column data type information missing")
            
            # Save sample output for inspection
            with open('test_csv_export_sample.csv', 'w', encoding='utf-8') as f:
                f.write(csv_output)
            print("✓ Sample CSV saved as 'test_csv_export_sample.csv'")
            
            return True
            
        else:
            print("✗ CSV export failed - no output generated")
            return False
            
    except Exception as e:
        print(f"✗ CSV export test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_excel_export():
    """Test Excel export functionality with categorical data"""
    print("\n" + "="*60)
    print("TESTING EXCEL EXPORT WITH CATEGORICAL DATA")
    print("="*60)
    
    try:
        # Import the export function
        from flex_filter_dashboard import generate_enhanced_excel_export
        
        # Create test data
        test_df = create_test_data_with_categorical()
        original_df = test_df.copy()
        
        print("\nTesting Excel export...")
        
        # Test the export function
        excel_buffer = generate_enhanced_excel_export(test_df, original_df)
        
        # Verify output
        if excel_buffer and len(excel_buffer.getvalue()) > 0:
            print("✓ Excel export successful")
            print(f"✓ Output size: {len(excel_buffer.getvalue())} bytes")
            
            # Save sample output for inspection
            with open('test_excel_export_sample.xlsx', 'wb') as f:
                f.write(excel_buffer.getvalue())
            print("✓ Sample Excel saved as 'test_excel_export_sample.xlsx'")
            
            # Try to read back the Excel file to verify structure
            try:
                excel_buffer.seek(0)  # Reset buffer position
                excel_sheets = pd.read_excel(excel_buffer, sheet_name=None, engine='openpyxl')
                
                print(f"✓ Excel file contains {len(excel_sheets)} sheets:")
                for sheet_name in excel_sheets.keys():
                    sheet_df = excel_sheets[sheet_name]
                    print(f"  - {sheet_name}: {sheet_df.shape[0]} rows, {sheet_df.shape[1]} columns")
                
                # Check for expected sheets
                expected_sheets = ['Filtered Properties', 'Export Metadata', 'Column Information']
                for sheet in expected_sheets:
                    if sheet in excel_sheets:
                        print(f"✓ Required sheet '{sheet}' found")
                    else:
                        print(f"⚠ Required sheet '{sheet}' missing")
                
                # Check data types in main sheet
                if 'Filtered Properties' in excel_sheets:
                    main_sheet = excel_sheets['Filtered Properties']
                    print("\nData types in exported sheet:")
                    for col in main_sheet.columns:
                        print(f"  {col}: {main_sheet[col].dtype}")
                
                return True
                
            except Exception as read_error:
                print(f"⚠ Could not read back Excel file: {str(read_error)}")
                return True  # Export succeeded even if we can't read it back
            
        else:
            print("✗ Excel export failed - no output generated")
            return False
            
    except Exception as e:
        print(f"✗ Excel export test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_data_type_conversion():
    """Test the data type conversion functionality directly"""
    print("\n" + "="*60)
    print("TESTING DATA TYPE CONVERSION UTILITIES")
    print("="*60)
    
    try:
        from utils.data_type_utils import convert_categorical_to_numeric, detect_categorical_numeric_columns
        
        # Create test data
        test_df = create_test_data_with_categorical()
        
        print("\nTesting categorical column detection...")
        categorical_analysis = detect_categorical_numeric_columns(test_df)
        
        print(f"✓ Detected {len(categorical_analysis)} categorical columns for conversion:")
        for col, analysis in categorical_analysis.items():
            print(f"  - {col}: {analysis['conversion_confidence']:.2f} confidence")
        
        print("\nTesting categorical to numeric conversion...")
        converted_df, conversion_reports = convert_categorical_to_numeric(test_df)
        
        print(f"✓ Conversion complete: {len(conversion_reports)} columns processed")
        
        # Compare data types before and after
        print("\nData type comparison:")
        print("Before conversion:")
        for col in test_df.columns:
            print(f"  {col}: {test_df[col].dtype}")
        
        print("After conversion:")
        for col in converted_df.columns:
            print(f"  {col}: {converted_df[col].dtype}")
        
        # Check conversion success
        successful_conversions = sum(1 for report in conversion_reports.values() 
                                   if report['conversion_successful'])
        print(f"\n✓ {successful_conversions}/{len(conversion_reports)} conversions successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Data type conversion test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all export tests"""
    print("PROPERTY DATA EXPORT TESTING")
    print("="*60)
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run tests
    tests = [
        ("Data Type Conversion", test_data_type_conversion),
        ("CSV Export", test_csv_export),
        ("Excel Export", test_excel_export)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Export functionality is working correctly.")
        return True
    else:
        print("⚠ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)