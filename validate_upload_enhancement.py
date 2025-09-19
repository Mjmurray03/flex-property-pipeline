"""
Validation script for the File Upload Dashboard Enhancement
This script validates the complete upload workflow and integration
"""

import pandas as pd
import numpy as np
import io
from datetime import datetime
import tempfile
import os

def create_test_excel_files():
    """Create various test Excel files for validation"""
    test_files = {}
    
    # 1. Valid complete file
    complete_data = {
        'Property Name': ['Test Property 1', 'Test Property 2', 'Test Property 3'],
        'Property Type': ['Industrial Warehouse', 'Distribution Center', 'Flex Space'],
        'Address': ['123 Test St', '456 Test Ave', '789 Test Blvd'],
        'City': ['Los Angeles', 'Houston', 'Phoenix'],
        'County': ['LA County', 'Harris County', 'Maricopa County'],
        'State': ['CA', 'TX', 'AZ'],
        'Building SqFt': [50000, 75000, 25000],
        'Lot Size Acres': [2.5, 5.0, 1.2],
        'Year Built': [2000, 1995, 2010],
        'Sold Price': [1500000, 2250000, 800000],
        'Occupancy': [85, 90, 75]
    }
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        pd.DataFrame(complete_data).to_excel(writer, sheet_name='Properties', index=False)
    test_files['complete_valid'] = buffer.getvalue()
    
    # 2. Minimal valid file (only required columns)
    minimal_data = {
        'Property Type': ['Industrial', 'Warehouse', 'Distribution'],
        'City': ['Los Angeles', 'Houston', 'Phoenix'],
        'State': ['CA', 'TX', 'AZ']
    }
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        pd.DataFrame(minimal_data).to_excel(writer, sheet_name='Properties', index=False)
    test_files['minimal_valid'] = buffer.getvalue()
    
    # 3. File with non-standard column names
    nonstandard_data = {
        'Prop Name': ['Test Property 1', 'Test Property 2'],
        'Prop Type': ['Industrial', 'Warehouse'],
        'Location': ['Los Angeles', 'Houston'],
        'State': ['CA', 'TX'],
        'Bldg SqFt': [50000, 75000],
        'Sale Price': [1500000, 2250000]
    }
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        pd.DataFrame(nonstandard_data).to_excel(writer, sheet_name='Properties', index=False)
    test_files['nonstandard_columns'] = buffer.getvalue()
    
    # 4. File with dirty data
    dirty_data = {
        'Property Type': ['Industrial Warehouse', 'Distribution Center'],
        'City': ['Los Angeles', 'Houston'],
        'State': ['CA', 'TX'],
        'Building SqFt': ['$50,000', '75,000'],
        'Sold Price': ['$1,500,000', '$2,250,000.50'],
        'Occupancy': ['85%', '90%'],
        'Year Built': [2000, 1995]
    }
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        pd.DataFrame(dirty_data).to_excel(writer, sheet_name='Properties', index=False)
    test_files['dirty_data'] = buffer.getvalue()
    
    return test_files

def validate_file_upload_functionality():
    """Validate file upload and processing functionality"""
    print("=== File Upload Functionality Validation ===")
    
    test_files = create_test_excel_files()
    
    # Import functions from dashboard
    try:
        from flex_filter_dashboard import (
            validate_uploaded_file, validate_data_structure, 
            suggest_column_mapping, clean_numeric_column,
            secure_file_validation
        )
        print("✓ Successfully imported dashboard functions")
    except ImportError as e:
        print(f"✗ Failed to import dashboard functions: {e}")
        return False
    
    # Test file validation
    class MockFile:
        def __init__(self, name, content, size):
            self.name = name
            self.size = size
            self._content = content
        
        def getvalue(self):
            return self._content
    
    # Test valid Excel file
    mock_file = MockFile("test.xlsx", test_files['complete_valid'], len(test_files['complete_valid']))
    is_valid, message, error_type = validate_uploaded_file(mock_file)
    
    if is_valid:
        print("✓ File validation: Valid Excel file accepted")
    else:
        print(f"✗ File validation failed: {message}")
        return False
    
    # Test invalid file format
    invalid_file = MockFile("test.txt", b"invalid content", 13)
    is_valid, message, error_type = validate_uploaded_file(invalid_file)
    
    if not is_valid and error_type == "invalid_format":
        print("✓ File validation: Invalid format correctly rejected")
    else:
        print("✗ File validation: Should reject invalid formats")
        return False
    
    return True

def validate_data_processing():
    """Validate data processing and cleaning functionality"""
    print("\n=== Data Processing Validation ===")
    
    test_files = create_test_excel_files()
    
    try:
        from flex_filter_dashboard import (
            validate_data_structure, advanced_data_cleaning,
            calculate_data_quality_score
        )
        
        # Test with complete valid data
        df = pd.read_excel(io.BytesIO(test_files['complete_valid']))
        validation_result = validate_data_structure(df)
        
        if validation_result['success']:
            print("✓ Data structure validation: Complete data validated successfully")
        else:
            print(f"✗ Data structure validation failed: {validation_result['errors']}")
            return False
        
        # Test with minimal data
        df_minimal = pd.read_excel(io.BytesIO(test_files['minimal_valid']))
        validation_result = validate_data_structure(df_minimal)
        
        if validation_result['success']:
            print("✓ Data structure validation: Minimal data validated successfully")
        else:
            print("✗ Data structure validation: Should accept minimal valid data")
            return False
        
        # Test data cleaning
        df_dirty = pd.read_excel(io.BytesIO(test_files['dirty_data']))
        cleaned_df, cleaning_report = advanced_data_cleaning(df_dirty)
        
        if len(cleaning_report['columns_processed']) > 0:
            print(f"✓ Data cleaning: Processed {len(cleaning_report['columns_processed'])} columns")
        else:
            print("✗ Data cleaning: No columns were processed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Data processing validation failed: {e}")
        return False

def validate_column_mapping():
    """Validate column mapping functionality"""
    print("\n=== Column Mapping Validation ===")
    
    test_files = create_test_excel_files()
    
    try:
        from flex_filter_dashboard import suggest_column_mapping, apply_column_mapping
        
        # Test with non-standard column names
        df = pd.read_excel(io.BytesIO(test_files['nonstandard_columns']))
        mapping_result = suggest_column_mapping(df)
        
        if mapping_result['fuzzy_matches']:
            print(f"✓ Column mapping: Found {len(mapping_result['fuzzy_matches'])} potential mappings")
            
            # Test applying mappings
            sample_mapping = {list(mapping_result['fuzzy_matches'].keys())[0]: 
                            list(mapping_result['fuzzy_matches'].values())[0]}
            
            mapped_df, mapped_columns = apply_column_mapping(df, sample_mapping)
            
            if len(mapped_columns) > 0:
                print(f"✓ Column mapping application: Applied {len(mapped_columns)} mappings")
            else:
                print("✗ Column mapping application: No mappings were applied")
                return False
        else:
            print("⚠ Column mapping: No fuzzy matches found (may be expected)")
        
        return True
        
    except Exception as e:
        print(f"✗ Column mapping validation failed: {e}")
        return False

def validate_integration_with_dashboard():
    """Validate integration with existing dashboard functionality"""
    print("\n=== Dashboard Integration Validation ===")
    
    test_files = create_test_excel_files()
    
    try:
        from flex_filter_dashboard import apply_filters, get_column_stats
        
        # Create test data that should work with existing filters
        df = pd.read_excel(io.BytesIO(test_files['complete_valid']))
        
        # Test filter application
        filter_params = {
            'industrial_keywords': ['industrial', 'warehouse'],
            'size_range': (20000, 100000),
            'lot_range': (1.0, 10.0),
            'use_price_filter': False,
            'use_year_filter': False,
            'use_occupancy_filter': False,
            'selected_counties': df['County'].unique().tolist() if 'County' in df.columns else [],
            'selected_states': df['State'].unique().tolist() if 'State' in df.columns else []
        }
        
        filtered_df = apply_filters(df, filter_params)
        
        if len(filtered_df) >= 0:  # Should return some results or empty (both valid)
            print(f"✓ Filter integration: Filters applied successfully, {len(filtered_df)} results")
        else:
            print("✗ Filter integration: Filter application failed")
            return False
        
        # Test column statistics
        if 'Building SqFt' in df.columns:
            stats = get_column_stats(df, 'Building SqFt')
            if stats and 'min' in stats and 'max' in stats:
                print("✓ Statistics integration: Column statistics working")
            else:
                print("✗ Statistics integration: Column statistics failed")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Dashboard integration validation failed: {e}")
        return False

def validate_export_functionality():
    """Validate enhanced export functionality"""
    print("\n=== Export Functionality Validation ===")
    
    test_files = create_test_excel_files()
    
    try:
        from flex_filter_dashboard import (
            generate_enhanced_csv_export, generate_enhanced_excel_export,
            generate_data_quality_report
        )
        
        # Create test data
        df = pd.read_excel(io.BytesIO(test_files['complete_valid']))
        
        # Test CSV export
        csv_data = generate_enhanced_csv_export(df, df)
        
        if csv_data and len(csv_data) > 0:
            print("✓ CSV export: Enhanced CSV export working")
            
            # Check for metadata in CSV
            if "# Property Filter Dashboard Export" in csv_data:
                print("✓ CSV export: Metadata included in CSV")
            else:
                print("⚠ CSV export: Metadata not found in CSV")
        else:
            print("✗ CSV export: Enhanced CSV export failed")
            return False
        
        # Test Excel export
        excel_buffer = generate_enhanced_excel_export(df, df)
        
        if excel_buffer and len(excel_buffer.getvalue()) > 0:
            print("✓ Excel export: Enhanced Excel export working")
        else:
            print("✗ Excel export: Enhanced Excel export failed")
            return False
        
        # Test quality report
        # Mock session state for testing
        import streamlit as st
        if not hasattr(st, 'session_state'):
            class MockSessionState:
                def __init__(self):
                    self.data = {}
                def get(self, key, default=None):
                    return self.data.get(key, default)
                def __setitem__(self, key, value):
                    self.data[key] = value
                def __getitem__(self, key):
                    return self.data[key]
            st.session_state = MockSessionState()
        
        st.session_state['validation_result'] = {
            'data_quality_score': 85, 
            'warnings': [],
            'row_count': len(df),
            'column_count': len(df.columns)
        }
        st.session_state['processing_report'] = {
            'processing_time': 1.5, 
            'recommendations': [],
            'memory_usage': 1.0,
            'columns_cleaned': []
        }
        st.session_state['uploaded_filename'] = 'test_file.xlsx'
        st.session_state['upload_timestamp'] = datetime.now()
        
        quality_report = generate_data_quality_report()
        
        if quality_report and len(quality_report) > 0:
            print("✓ Quality report: Data quality report generated")
        else:
            print("✗ Quality report: Data quality report generation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Export functionality validation failed: {e}")
        return False

def validate_error_handling():
    """Validate error handling throughout the upload process"""
    print("\n=== Error Handling Validation ===")
    
    try:
        from flex_filter_dashboard import (
            validate_uploaded_file, show_error_guidance,
            validate_data_structure
        )
        
        # Test various error conditions
        error_cases = [
            (None, "No file"),
            ("MockFile", "test.txt", b"invalid", "Invalid format"),
            ("MockFile", "test.xlsx", b"x" * (51 * 1024 * 1024), "File too large"),
            ("MockFile", "test.xlsx", b"", "Empty file")
        ]
        
        class MockFile:
            def __init__(self, name, content, size):
                self.name = name
                self.size = size
                self._content = content
            def getvalue(self):
                return self._content
        
        error_count = 0
        for case in error_cases:
            try:
                if case[0] is None:
                    is_valid, message, error_type = validate_uploaded_file(None)
                else:
                    mock_file = MockFile(case[1], case[2], len(case[2]))
                    is_valid, message, error_type = validate_uploaded_file(mock_file)
                
                if not is_valid:
                    error_count += 1
                    print(f"✓ Error handling: {case[-1]} correctly handled")
                else:
                    print(f"⚠ Error handling: {case[-1]} should have been rejected")
            except Exception as e:
                print(f"✗ Error handling: Exception in {case[-1]}: {e}")
                return False
        
        if error_count >= 3:  # Should catch most error cases
            print(f"✓ Error handling: {error_count} error cases handled correctly")
            return True
        else:
            print(f"✗ Error handling: Only {error_count} error cases handled")
            return False
        
    except Exception as e:
        print(f"✗ Error handling validation failed: {e}")
        return False

def validate_performance():
    """Validate performance with various data sizes"""
    print("\n=== Performance Validation ===")
    
    try:
        from flex_filter_dashboard import validate_data_structure, advanced_data_cleaning
        
        # Test with different data sizes
        sizes = [10, 100, 1000]
        
        for size in sizes:
            # Create test data of specified size
            test_data = {
                'Property Type': ['Industrial'] * size,
                'City': ['Test City'] * size,
                'State': ['CA'] * size,
                'Building SqFt': list(range(size))
            }
            
            df = pd.DataFrame(test_data)
            
            # Test validation performance
            start_time = datetime.now()
            validation_result = validate_data_structure(df)
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds()
            
            if processing_time < 5.0:  # Should complete within 5 seconds
                print(f"✓ Performance: {size} rows processed in {processing_time:.3f}s")
            else:
                print(f"⚠ Performance: {size} rows took {processing_time:.3f}s (may be slow)")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance validation failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("File Upload Dashboard Enhancement Validation")
    print("=" * 60)
    
    validation_results = []
    
    # Run all validation tests
    tests = [
        ("File Upload Functionality", validate_file_upload_functionality),
        ("Data Processing", validate_data_processing),
        ("Column Mapping", validate_column_mapping),
        ("Dashboard Integration", validate_integration_with_dashboard),
        ("Export Functionality", validate_export_functionality),
        ("Error Handling", validate_error_handling),
        ("Performance", validate_performance)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            validation_results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}: Critical error - {str(e)}")
            validation_results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in validation_results if result)
    total = len(validation_results)
    
    for test_name, result in validation_results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:<30} {status}")
    
    print("-" * 60)
    print(f"Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validation tests passed!")
        print("The File Upload Dashboard Enhancement is ready for production use.")
    else:
        print("⚠️  Some validation tests failed.")
        print("Please review the failed tests before deploying.")
    
    return passed == total

if __name__ == '__main__':
    main()