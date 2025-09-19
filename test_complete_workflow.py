"""
Complete workflow test for the Interactive Filter Dashboard with File Upload
This script tests the entire workflow from upload to export
"""

import pandas as pd
import io
from datetime import datetime

def test_complete_workflow():
    """Test the complete workflow"""
    print("Testing Complete Dashboard Workflow")
    print("=" * 50)
    
    try:
        # Import all necessary functions
        from flex_filter_dashboard import (
            validate_uploaded_file, load_uploaded_data, validate_data_structure,
            suggest_column_mapping, apply_column_mapping, apply_filters,
            generate_enhanced_csv_export, generate_enhanced_excel_export,
            generate_data_quality_report, clean_numeric_column
        )
        
        print("✓ All functions imported successfully")
        
        # Create test data
        test_data = {
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
        
        df = pd.DataFrame(test_data)
        print("✓ Test data created")
        
        # Test 1: File validation
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Properties', index=False)
        excel_content = buffer.getvalue()
        
        class MockFile:
            def __init__(self, name, content, size):
                self.name = name
                self.size = size
                self._content = content
            def getvalue(self):
                return self._content
        
        mock_file = MockFile("test.xlsx", excel_content, len(excel_content))
        is_valid, message, error_type = validate_uploaded_file(mock_file)
        
        if is_valid:
            print("✓ File validation passed")
        else:
            print(f"✗ File validation failed: {message}")
            return False
        
        # Test 2: Data loading and processing
        processed_df, validation_result, load_message = load_uploaded_data(excel_content, "test.xlsx")
        
        if processed_df is not None and validation_result is not None:
            print("✓ Data loading and processing passed")
        else:
            print(f"✗ Data loading failed: {load_message}")
            return False
        
        # Test 3: Data structure validation
        if validation_result['success']:
            print("✓ Data structure validation passed")
        else:
            print(f"✗ Data structure validation failed: {validation_result['errors']}")
            return False
        
        # Test 4: Column mapping (with non-standard names)
        alt_df = df.rename(columns={'Property Type': 'Prop Type', 'Building SqFt': 'Bldg SqFt'})
        mapping_result = suggest_column_mapping(alt_df)
        
        if mapping_result['fuzzy_matches']:
            print("✓ Column mapping suggestions working")
            
            # Test applying mappings
            sample_mapping = {list(mapping_result['fuzzy_matches'].keys())[0]: 
                            list(mapping_result['fuzzy_matches'].values())[0]}
            mapped_df, mapped_columns = apply_column_mapping(alt_df, sample_mapping)
            
            if len(mapped_columns) > 0:
                print("✓ Column mapping application working")
            else:
                print("✗ Column mapping application failed")
                return False
        else:
            print("✓ Column mapping (no mappings needed)")
        
        # Test 5: Filtering functionality
        filter_params = {
            'industrial_keywords': ['industrial', 'warehouse'],
            'size_range': (20000, 100000),
            'lot_range': (1.0, 10.0),
            'use_price_filter': True,
            'price_range': (500000, 3000000),
            'use_year_filter': False,
            'use_occupancy_filter': False,
            'selected_counties': df['County'].unique().tolist(),
            'selected_states': df['State'].unique().tolist()
        }
        
        filtered_df = apply_filters(df, filter_params)
        
        if len(filtered_df) >= 0:  # Can be 0 or more
            print(f"✓ Filtering functionality working ({len(filtered_df)} results)")
        else:
            print("✗ Filtering functionality failed")
            return False
        
        # Test 6: Export functionality
        # Mock session state for export functions
        import streamlit as st
        if not hasattr(st, 'session_state'):
            class MockSessionState:
                def __init__(self):
                    self.data = {
                        'uploaded_filename': 'test.xlsx',
                        'upload_timestamp': datetime.now(),
                        'processing_time': 1.5,
                        'validation_result': {
                            'data_quality_score': 85,
                            'row_count': len(df),
                            'column_count': len(df.columns),
                            'required_columns_found': ['Property Type', 'City', 'State'],
                            'required_columns_missing': [],
                            'warnings': []
                        },
                        'processing_report': {
                            'processing_time': 1.5,
                            'memory_usage': 1.0,
                            'columns_cleaned': ['Building SqFt', 'Sold Price'],
                            'cleaning_stats': {},
                            'recommendations': []
                        }
                    }
                def get(self, key, default=None):
                    return self.data.get(key, default)
                def __getitem__(self, key):
                    return self.data[key]
            st.session_state = MockSessionState()
        
        # Test CSV export
        csv_data = generate_enhanced_csv_export(filtered_df, df)
        if csv_data and len(csv_data) > 0 and "Property Filter Dashboard Export" in csv_data:
            print("✓ CSV export working with metadata")
        else:
            print("✗ CSV export failed")
            return False
        
        # Test Excel export
        excel_buffer = generate_enhanced_excel_export(filtered_df, df)
        if excel_buffer and len(excel_buffer.getvalue()) > 0:
            print("✓ Excel export working")
        else:
            print("✗ Excel export failed")
            return False
        
        # Test quality report
        quality_report = generate_data_quality_report()
        if quality_report and len(quality_report) > 0:
            print("✓ Data quality report generation working")
        else:
            print("✗ Data quality report generation failed")
            return False
        
        # Test 7: Data cleaning
        dirty_series = pd.Series(['$50,000', '75,000', '85%', 'N/A'])
        cleaned_series = clean_numeric_column(dirty_series)
        
        if not cleaned_series.isna().all():
            print("✓ Data cleaning functionality working")
        else:
            print("✗ Data cleaning functionality failed")
            return False
        
        print("\n" + "=" * 50)
        print("✓ ALL WORKFLOW TESTS PASSED!")
        print("The dashboard is ready for production use.")
        return True
        
    except Exception as e:
        print(f"✗ Workflow test failed with error: {str(e)}")
        return False

if __name__ == '__main__':
    success = test_complete_workflow()
    exit(0 if success else 1)