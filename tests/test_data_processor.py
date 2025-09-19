"""
Unit tests for data processor
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import io

from app.components.data_processor import DataProcessor, ColumnMapping, DataQualityMetric


class TestDataProcessor:
    """Test data processor"""
    
    def setup_method(self):
        """Setup test method"""
        self.processor = DataProcessor()
        self.processor.initialize()
        
        # Create sample property data
        self.sample_data = pd.DataFrame({
            'Property ID': ['P001', 'P002', 'P003', 'P004', 'P005'],
            'Address': ['123 Main St', '456 Oak Ave', '789 Pine Rd', '321 Elm St', '654 Maple Dr'],
            'City': ['Seattle', 'Portland', 'San Francisco', 'Los Angeles', 'San Diego'],
            'State': ['WA', 'OR', 'CA', 'CA', 'CA'],
            'Property Type': ['Office', 'Retail', 'Industrial', 'Office', 'Retail'],
            'Building SqFt': [10000, 5000, 25000, 8000, 3000],
            'Lot Size (Acres)': [0.5, 0.3, 2.0, 0.4, 0.2],
            'Year Built': [1995, 2000, 1985, 2010, 2005],
            'Sale Price': [1500000, 800000, 3000000, 1200000, 600000]
        })
    
    def teardown_method(self):
        """Teardown test method"""
        self.processor.cleanup()
    
    def test_process_upload_dataframe(self):
        """Test processing uploaded DataFrame"""
        df_processed, report = self.processor.process_upload(self.sample_data)
        
        assert isinstance(df_processed, pd.DataFrame)
        assert len(df_processed) == 5
        assert report.rows_processed == 5
        assert report.quality_score > 0.5
        assert report.file_fingerprint is not None
    
    def test_process_upload_csv_string(self):
        """Test processing CSV string data"""
        csv_data = self.sample_data.to_csv(index=False)
        csv_io = io.StringIO(csv_data)
        
        df_processed, report = self.processor.process_upload(csv_io)
        
        assert isinstance(df_processed, pd.DataFrame)
        assert len(df_processed) == 5
        assert report.quality_score > 0.5
    
    def test_validate_structure_valid_data(self):
        """Test structure validation with valid data"""
        result = self.processor.validate_structure(self.sample_data)
        
        assert result.is_valid is True
        assert result.quality_score > 0.7
        assert len(result.errors) == 0
    
    def test_validate_structure_empty_data(self):
        """Test structure validation with empty data"""
        empty_df = pd.DataFrame()
        result = self.processor.validate_structure(empty_df)
        
        assert result.is_valid is False
        assert result.quality_score == 0.0
        assert "Dataset is empty" in result.errors
    
    def test_validate_structure_duplicate_columns(self):
        """Test structure validation with duplicate columns"""
        df_with_duplicates = self.sample_data.copy()
        df_with_duplicates.columns = list(df_with_duplicates.columns[:-1]) + [df_with_duplicates.columns[0]]
        
        result = self.processor.validate_structure(df_with_duplicates)
        
        assert result.is_valid is False
        assert any("Duplicate column names" in error for error in result.errors)
    
    def test_clean_and_transform(self):
        """Test data cleaning and transformation"""
        # Add some messy data
        messy_data = self.sample_data.copy()
        messy_data.loc[0, 'City'] = '  seattle  '  # Extra spaces
        messy_data.loc[1, 'Building SqFt'] = np.nan  # Missing value
        messy_data.loc[2, 'Sale Price'] = -100000  # Negative price (will be capped)
        
        df_cleaned = self.processor.clean_and_transform(messy_data)
        
        # Check column names are cleaned
        assert all('_' in col.lower() or col.islower() for col in df_cleaned.columns)
        
        # Check missing values are handled
        assert df_cleaned.isnull().sum().sum() == 0
        
        # Check calculated fields are added
        if 'price_per_sqft' in df_cleaned.columns:
            assert df_cleaned['price_per_sqft'].notna().any()
    
    def test_generate_quality_report(self):
        """Test quality report generation"""
        report = self.processor.generate_quality_report(self.sample_data)
        
        assert isinstance(report.overall_score, float)
        assert 0 <= report.overall_score <= 1
        assert len(report.metrics) > 0
        assert len(report.column_profiles) == len(self.sample_data.columns)
        assert report.data_fingerprint is not None
        
        # Check that all expected metrics are present
        metric_names = [metric.metric_name for metric in report.metrics]
        expected_metrics = ['Completeness', 'Consistency', 'Validity', 'Uniqueness']
        for expected in expected_metrics:
            assert expected in metric_names
    
    def test_suggest_column_mapping(self):
        """Test column mapping suggestions"""
        # Create data with non-standard column names
        non_standard_data = pd.DataFrame({
            'prop_id': ['P001', 'P002'],
            'street_address': ['123 Main St', '456 Oak Ave'],
            'municipality': ['Seattle', 'Portland'],
            'sqft': [10000, 5000],
            'construction_year': [1995, 2000]
        })
        
        mappings = self.processor.suggest_column_mapping(non_standard_data)
        
        assert len(mappings) > 0
        
        # Check that we get reasonable mappings
        mapping_dict = {m.original_name: m.suggested_name for m in mappings}
        
        # These should have high confidence mappings
        if 'prop_id' in mapping_dict:
            assert mapping_dict['prop_id'] == 'property_id'
        if 'street_address' in mapping_dict:
            assert mapping_dict['street_address'] == 'address'
    
    def test_data_fingerprint_consistency(self):
        """Test that identical data produces same fingerprint"""
        fingerprint1 = self.processor._generate_data_fingerprint(self.sample_data)
        fingerprint2 = self.processor._generate_data_fingerprint(self.sample_data.copy())
        
        assert fingerprint1 == fingerprint2
        
        # Different data should produce different fingerprint
        modified_data = self.sample_data.copy()
        modified_data.loc[0, 'City'] = 'Different City'
        fingerprint3 = self.processor._generate_data_fingerprint(modified_data)
        
        assert fingerprint1 != fingerprint3
    
    def test_assess_data_quality_completeness(self):
        """Test completeness assessment"""
        # Create data with missing values
        incomplete_data = self.sample_data.copy()
        incomplete_data.loc[0:2, 'Sale Price'] = np.nan
        incomplete_data.loc[1:3, 'Year Built'] = np.nan
        
        metrics = self.processor._assess_data_quality(incomplete_data)
        
        completeness_metric = next(m for m in metrics if m.metric_name == 'Completeness')
        assert completeness_metric.score < 1.0
        assert len(completeness_metric.issues) > 0
    
    def test_assess_data_quality_uniqueness(self):
        """Test uniqueness assessment"""
        # Create data with duplicates
        duplicate_data = self.sample_data.copy()
        duplicate_data = pd.concat([duplicate_data, duplicate_data.iloc[[0, 1]]], ignore_index=True)
        
        metrics = self.processor._assess_data_quality(duplicate_data)
        
        uniqueness_metric = next(m for m in metrics if m.metric_name == 'Uniqueness')
        assert uniqueness_metric.score < 1.0
        assert any('duplicate' in issue.lower() for issue in uniqueness_metric.issues)
    
    def test_clean_column_names(self):
        """Test column name cleaning"""
        messy_columns = ['Property ID!', 'Building Sq.Ft.', 'Sale Price ($)', 'Year-Built', '  City  ']
        cleaned = self.processor._clean_column_names(pd.Index(messy_columns))
        
        expected = ['property_id', 'building_sqft', 'sale_price', 'year_built', 'city']
        assert cleaned == expected
    
    def test_handle_missing_values_numeric(self):
        """Test missing value handling for numeric columns"""
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0:1, 'Building SqFt'] = np.nan
        data_with_missing.loc[2:3, 'Sale Price'] = np.nan
        
        cleaned = self.processor._handle_missing_values(data_with_missing)
        
        # Should have no missing values after cleaning
        assert cleaned['Building SqFt'].isnull().sum() == 0
        assert cleaned['Sale Price'].isnull().sum() == 0
    
    def test_handle_missing_values_text(self):
        """Test missing value handling for text columns"""
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0:1, 'City'] = np.nan
        data_with_missing.loc[2, 'Property Type'] = np.nan
        
        cleaned = self.processor._handle_missing_values(data_with_missing)
        
        # Should have no missing values after cleaning
        assert cleaned['City'].isnull().sum() == 0
        assert cleaned['Property Type'].isnull().sum() == 0
    
    def test_generate_calculated_fields(self):
        """Test calculated field generation"""
        # Ensure we have the right column names for calculations
        calc_data = self.sample_data.copy()
        calc_data.columns = ['property_id', 'address', 'city', 'state', 'property_type', 
                           'building_sqft', 'lot_size_acres', 'year_built', 'sold_price']
        
        enhanced = self.processor._generate_calculated_fields(calc_data)
        
        # Check that calculated fields are added
        assert 'price_per_sqft' in enhanced.columns
        assert 'building_age' in enhanced.columns
        assert 'building_efficiency' in enhanced.columns
        
        # Check calculations are correct
        current_year = datetime.now().year
        expected_age = current_year - calc_data['year_built']
        pd.testing.assert_series_equal(enhanced['building_age'], expected_age, check_names=False)
    
    def test_find_best_column_match(self):
        """Test column matching algorithm"""
        # Test exact match
        exact_match = self.processor._find_best_column_match('property_id')
        assert exact_match is not None
        assert exact_match.confidence == 1.0
        assert exact_match.mapping_type == 'exact'
        
        # Test fuzzy match
        fuzzy_match = self.processor._find_best_column_match('prop_id')
        assert fuzzy_match is not None
        assert fuzzy_match.confidence > 0.6
        assert fuzzy_match.mapping_type == 'fuzzy'
        assert fuzzy_match.suggested_name == 'property_id'
        
        # Test no match
        no_match = self.processor._find_best_column_match('completely_unrelated_column')
        assert no_match is None
    
    def test_column_profiles_numeric(self):
        """Test column profiling for numeric columns"""
        profiles = self.processor._generate_column_profiles(self.sample_data)
        
        sqft_profile = profiles['Building SqFt']
        assert sqft_profile['dtype'] == 'int64'
        assert sqft_profile['min'] == 3000
        assert sqft_profile['max'] == 25000
        assert sqft_profile['mean'] == 10200.0
        assert 'std' in sqft_profile
    
    def test_column_profiles_text(self):
        """Test column profiling for text columns"""
        profiles = self.processor._generate_column_profiles(self.sample_data)
        
        city_profile = profiles['City']
        assert city_profile['dtype'] == 'object'
        assert city_profile['unique_count'] == 5
        assert 'avg_length' in city_profile
        assert 'top_values' in city_profile
    
    def test_processing_statistics(self):
        """Test processing statistics tracking"""
        initial_stats = self.processor.get_processing_stats()
        
        # Process some data
        self.processor.process_upload(self.sample_data)
        
        updated_stats = self.processor.get_processing_stats()
        
        assert updated_stats['total_processed'] > initial_stats['total_processed']
        assert updated_stats['average_processing_time'] > 0