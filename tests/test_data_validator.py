"""
Tests for DataValidator class
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from pipeline.data_validator import (
    DataValidator, ValidationResult, SchemaDefinition, QualityMetrics
)


class TestSchemaDefinition:
    """Test cases for SchemaDefinition"""
    
    def test_initialization(self):
        """Test SchemaDefinition initialization"""
        schema = SchemaDefinition(
            required_columns={'col1', 'col2'},
            optional_columns={'col3'},
            column_types={'col1': 'string', 'col2': 'numeric'},
            nullable_columns={'col3'},
            min_records=5,
            max_null_percentage=25.0
        )
        
        assert schema.required_columns == {'col1', 'col2'}
        assert schema.optional_columns == {'col3'}
        assert schema.column_types['col1'] == 'string'
        assert schema.min_records == 5
        assert schema.max_null_percentage == 25.0


class TestValidationResult:
    """Test cases for ValidationResult"""
    
    def test_initialization(self):
        """Test ValidationResult initialization"""
        result = ValidationResult(
            is_valid=True,
            file_path="/test/file.xlsx",
            record_count=100
        )
        
        assert result.is_valid is True
        assert result.file_path == "/test/file.xlsx"
        assert result.record_count == 100
        assert len(result.validation_errors) == 0
        assert len(result.validation_warnings) == 0
        assert result.data_quality_score == 0.0


class TestQualityMetrics:
    """Test cases for QualityMetrics"""
    
    def test_initialization(self):
        """Test QualityMetrics initialization"""
        metrics = QualityMetrics()
        
        assert metrics.total_records == 0
        assert metrics.unique_records == 0
        assert metrics.duplicate_percentage == 0.0
        assert metrics.overall_quality_score == 0.0


class TestDataValidator:
    """Test cases for DataValidator"""
    
    @pytest.fixture
    def validator(self):
        """Create DataValidator instance"""
        mock_logger = MagicMock()
        return DataValidator(logger=mock_logger)
    
    @pytest.fixture
    def sample_valid_data(self):
        """Create sample valid DataFrame"""
        return pd.DataFrame({
            'Property Name': ['Property A', 'Property B', 'Property C'],
            'Address': ['123 Main St', '456 Oak Ave', '789 Pine Rd'],
            'City': ['Austin', 'Dallas', 'Houston'],
            'State': ['TX', 'TX', 'TX'],
            'Property Type': ['Warehouse', 'Flex', 'Industrial'],
            'Building SqFt': [50000, 75000, 60000],
            'Lot Size Acres': [2.5, 3.0, 2.8],
            'Year Built': [1995, 2000, 1988]
        })
    
    @pytest.fixture
    def sample_invalid_data(self):
        """Create sample invalid DataFrame"""
        return pd.DataFrame({
            'Property Name': ['Property A', 'Property B'],
            'Address': ['123 Main St', '456 Oak Ave'],
            'City': ['Austin', 'Dallas'],
            'State': ['TX', 'TEXAS'],  # Invalid state format
            'Property Type': ['Warehouse', 'Flex'],
            'Building SqFt': [50000, 'invalid'],  # Invalid numeric value
            'Lot Size Acres': [2.5, None],
            'Year Built': [1995, 2050]  # Future year
        })
    
    def test_initialization(self, validator):
        """Test DataValidator initialization"""
        assert validator.logger is not None
        assert validator.default_schema is not None
        assert 'Property Name' in validator.default_schema.required_columns
        assert 'Building SqFt' in validator.default_schema.required_columns
    
    def test_validate_file_schema_valid_data(self, validator, sample_valid_data):
        """Test schema validation with valid data"""
        result = validator.validate_file_schema(sample_valid_data, "/test/valid.xlsx")
        
        assert result.is_valid is True
        assert result.record_count == 3
        assert len(result.validation_errors) == 0
        assert len(result.missing_columns) == 0
        assert result.data_quality_score > 80  # Should have high quality score
    
    def test_validate_file_schema_missing_columns(self, validator):
        """Test schema validation with missing required columns"""
        df = pd.DataFrame({
            'Property Name': ['Property A'],
            'Address': ['123 Main St']
            # Missing required columns: City, State, Property Type, Building SqFt
        })
        
        result = validator.validate_file_schema(df, "/test/missing.xlsx")
        
        assert result.is_valid is False
        assert len(result.missing_columns) > 0
        assert 'City' in result.missing_columns
        assert 'State' in result.missing_columns
        assert len(result.schema_issues) > 0
    
    def test_validate_file_schema_extra_columns(self, validator, sample_valid_data):
        """Test schema validation with extra columns"""
        df = sample_valid_data.copy()
        df['Extra Column'] = ['A', 'B', 'C']
        
        result = validator.validate_file_schema(df, "/test/extra.xlsx")
        
        assert result.is_valid is True  # Extra columns are warnings, not errors
        assert 'Extra Column' in result.extra_columns
        assert len(result.validation_warnings) > 0
    
    def test_validate_file_schema_insufficient_records(self, validator):
        """Test schema validation with insufficient records"""
        df = pd.DataFrame()  # Empty DataFrame
        
        result = validator.validate_file_schema(df, "/test/empty.xlsx")
        
        assert result.is_valid is False
        assert result.record_count == 0
        assert any("Insufficient records" in error for error in result.validation_errors)
    
    def test_validate_data_types_numeric_issues(self, validator, sample_invalid_data):
        """Test data type validation with numeric issues"""
        result = validator.validate_file_schema(sample_invalid_data, "/test/invalid.xlsx")
        
        assert 'Building SqFt' in result.data_type_issues
        assert len(result.validation_warnings) > 0
    
    def test_validate_schema_consistency_multiple_files(self, validator, sample_valid_data):
        """Test schema consistency across multiple files"""
        # Create validation results for multiple files
        result1 = validator.validate_file_schema(sample_valid_data, "/test/file1.xlsx")
        
        # Create second file with different schema
        df2 = sample_valid_data.copy()
        df2['New Column'] = ['X', 'Y', 'Z']
        result2 = validator.validate_file_schema(df2, "/test/file2.xlsx")
        
        consistency_report = validator.validate_schema_consistency([result1, result2])
        
        assert consistency_report['total_files'] == 2
        assert consistency_report['valid_files'] >= 0
        assert 'average_quality_score' in consistency_report
    
    def test_assess_data_quality_empty_dataframe(self, validator):
        """Test data quality assessment with empty DataFrame"""
        df = pd.DataFrame()
        
        metrics = validator.assess_data_quality(df)
        
        assert metrics.total_records == 0
        assert metrics.unique_records == 0
        assert metrics.duplicate_percentage == 0.0
    
    def test_assess_data_quality_with_duplicates(self, validator):
        """Test data quality assessment with duplicate records"""
        df = pd.DataFrame({
            'Property Name': ['Property A', 'Property B', 'Property A'],
            'Address': ['123 Main St', '456 Oak Ave', '123 Main St'],
            'City': ['Austin', 'Dallas', 'Austin'],
            'State': ['TX', 'TX', 'TX'],
            'Property Type': ['Warehouse', 'Flex', 'Warehouse'],
            'Building SqFt': [50000, 75000, 50000]
        })
        
        metrics = validator.assess_data_quality(df)
        
        assert metrics.total_records == 3
        assert metrics.duplicate_records > 0
        assert metrics.duplicate_percentage > 0
    
    def test_assess_data_quality_completeness(self, validator):
        """Test data quality assessment for completeness"""
        df = pd.DataFrame({
            'Property Name': ['Property A', 'Property B', None],
            'Address': ['123 Main St', '456 Oak Ave', '789 Pine Rd'],
            'City': ['Austin', None, 'Houston'],
            'State': ['TX', 'TX', 'TX'],
            'Property Type': ['Warehouse', 'Flex', 'Industrial'],
            'Building SqFt': [50000, 75000, 60000]
        })
        
        metrics = validator.assess_data_quality(df)
        
        assert metrics.total_records == 3
        assert 'Property Name' in metrics.column_completeness
        assert metrics.column_completeness['Property Name'] < 100  # Has null values
        assert metrics.column_completeness['State'] == 100  # No null values
    
    def test_generate_validation_report_empty_results(self, validator):
        """Test validation report generation with empty results"""
        report = validator.generate_validation_report([])
        
        assert 'error' in report
    
    def test_generate_validation_report_with_results(self, validator, sample_valid_data):
        """Test validation report generation with results"""
        result1 = validator.validate_file_schema(sample_valid_data, "/test/file1.xlsx")
        result2 = validator.validate_file_schema(sample_valid_data, "/test/file2.xlsx")
        
        quality_metrics = validator.assess_data_quality(sample_valid_data)
        
        report = validator.generate_validation_report([result1, result2], quality_metrics)
        
        assert 'validation_summary' in report
        assert 'quality_summary' in report
        assert 'issues_summary' in report
        assert 'file_details' in report
        assert 'aggregated_quality' in report
        
        assert report['validation_summary']['total_files'] == 2
        assert len(report['file_details']) == 2
    
    def test_validate_data_consistency_valid_data(self, validator, sample_valid_data):
        """Test data consistency validation with valid data"""
        issues = validator.validate_data_consistency(sample_valid_data)
        
        # Should have minimal issues with valid data
        assert isinstance(issues, list)
    
    def test_validate_data_consistency_invalid_ranges(self, validator):
        """Test data consistency validation with invalid ranges"""
        df = pd.DataFrame({
            'Property Name': ['Property A', 'Property B'],
            'Address': ['123 Main St', '456 Oak Ave'],
            'City': ['Austin', 'Dallas'],
            'State': ['TX', 'TEXAS'],  # Invalid state format
            'Property Type': ['Warehouse', 'Flex'],
            'Building SqFt': [50000, 50000000000],  # Extremely large value
            'Lot Size Acres': [2.5, -1.0],  # Negative value
            'Year Built': [1995, 2050]  # Future year
        })
        
        issues = validator.validate_data_consistency(df)
        
        assert len(issues) > 0
        assert any("State format" in issue for issue in issues)
        assert any("Building SqFt" in issue for issue in issues)
        assert any("Year Built" in issue for issue in issues)
    
    def test_validate_data_consistency_duplicates(self, validator):
        """Test data consistency validation with duplicates"""
        df = pd.DataFrame({
            'Property Name': ['Property A', 'Property B', 'Property A'],
            'Address': ['123 Main St', '456 Oak Ave', '123 Main St'],
            'City': ['Austin', 'Dallas', 'Austin'],
            'State': ['TX', 'TX', 'TX'],
            'Property Type': ['Warehouse', 'Flex', 'Warehouse'],
            'Building SqFt': [50000, 75000, 50000]
        })
        
        issues = validator.validate_data_consistency(df)
        
        assert any("duplicates" in issue for issue in issues)
    
    def test_validate_data_consistency_empty_dataframe(self, validator):
        """Test data consistency validation with empty DataFrame"""
        df = pd.DataFrame()
        
        issues = validator.validate_data_consistency(df)
        
        assert issues == []
    
    def test_custom_schema_validation(self, validator):
        """Test validation with custom schema"""
        custom_schema = SchemaDefinition(
            required_columns={'Name', 'Value'},
            optional_columns={'Description'},
            column_types={'Name': 'string', 'Value': 'numeric'},
            min_records=2
        )
        
        df = pd.DataFrame({
            'Name': ['Item A', 'Item B'],
            'Value': [100, 200],
            'Description': ['Desc A', 'Desc B']
        })
        
        result = validator.validate_file_schema(df, "/test/custom.xlsx", custom_schema)
        
        assert result.is_valid is True
        assert result.record_count == 2
    
    def test_null_percentage_validation(self, validator):
        """Test null percentage validation"""
        df = pd.DataFrame({
            'Property Name': ['Property A', None, None],
            'Address': ['123 Main St', '456 Oak Ave', '789 Pine Rd'],
            'City': ['Austin', 'Dallas', 'Houston'],
            'State': ['TX', 'TX', 'TX'],
            'Property Type': ['Warehouse', 'Flex', 'Industrial'],
            'Building SqFt': [50000, 75000, 60000]
        })
        
        result = validator.validate_file_schema(df, "/test/nulls.xlsx")
        
        assert 'Property Name' in result.null_percentages
        assert result.null_percentages['Property Name'] > 50  # 2/3 are null
        assert len(result.validation_warnings) > 0


if __name__ == "__main__":
    pytest.main([__file__])