"""
Data Validation and Quality Checks for Scalable Multi-File Pipeline
Provides schema validation, data type validation, and quality metrics
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class ValidationResult:
    """Result of data validation operation"""
    is_valid: bool
    file_path: str
    record_count: int
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    schema_issues: List[str] = field(default_factory=list)
    data_quality_score: float = 0.0
    missing_columns: List[str] = field(default_factory=list)
    extra_columns: List[str] = field(default_factory=list)
    data_type_issues: Dict[str, str] = field(default_factory=dict)
    null_percentages: Dict[str, float] = field(default_factory=dict)


@dataclass
class SchemaDefinition:
    """Definition of expected data schema"""
    required_columns: Set[str]
    optional_columns: Set[str] = field(default_factory=set)
    column_types: Dict[str, str] = field(default_factory=dict)
    nullable_columns: Set[str] = field(default_factory=set)
    min_records: int = 1
    max_null_percentage: float = 50.0  # Maximum allowed null percentage per column


@dataclass
class QualityMetrics:
    """Data quality metrics for aggregated results"""
    total_records: int = 0
    unique_records: int = 0
    duplicate_records: int = 0
    duplicate_percentage: float = 0.0
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    overall_quality_score: float = 0.0
    column_completeness: Dict[str, float] = field(default_factory=dict)
    data_type_consistency: Dict[str, float] = field(default_factory=dict)
    value_distribution: Dict[str, Dict[str, int]] = field(default_factory=dict)


class DataValidator:
    """
    Comprehensive data validation and quality assessment system.
    Validates schema compatibility, data types, and provides quality metrics.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the DataValidator.
        
        Args:
            logger: Logger instance for validation messages
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Default schema for flex property data
        self.default_schema = SchemaDefinition(
            required_columns={
                'Property Name', 'Address', 'City', 'State', 
                'Property Type', 'Building SqFt'
            },
            optional_columns={
                'Lot Size Acres', 'Year Built', 'Occupancy Rate',
                'County', 'Zip Code', 'Owner Name', 'Flex Score'
            },
            column_types={
                'Property Name': 'string',
                'Address': 'string',
                'City': 'string',
                'State': 'string',
                'Property Type': 'string',
                'Building SqFt': 'numeric',
                'Lot Size Acres': 'numeric',
                'Year Built': 'numeric',
                'Occupancy Rate': 'numeric',
                'Flex Score': 'numeric'
            },
            nullable_columns={
                'Lot Size Acres', 'Year Built', 'Occupancy Rate',
                'County', 'Zip Code', 'Owner Name', 'Flex Score'
            },
            min_records=1,
            max_null_percentage=80.0
        )
    
    def validate_file_schema(self, 
                           df: pd.DataFrame, 
                           file_path: str,
                           schema: Optional[SchemaDefinition] = None) -> ValidationResult:
        """
        Validate DataFrame against expected schema.
        
        Args:
            df: DataFrame to validate
            file_path: Path to the source file
            schema: Schema definition (uses default if None)
            
        Returns:
            ValidationResult with validation details
        """
        if schema is None:
            schema = self.default_schema
        
        result = ValidationResult(
            is_valid=True,
            file_path=file_path,
            record_count=len(df)
        )
        
        # Check minimum record count
        if len(df) < schema.min_records:
            result.is_valid = False
            result.validation_errors.append(
                f"Insufficient records: {len(df)} < {schema.min_records} required"
            )
        
        # Check for required columns
        df_columns = set(df.columns)
        missing_required = schema.required_columns - df_columns
        if missing_required:
            result.is_valid = False
            result.missing_columns = list(missing_required)
            result.schema_issues.append(
                f"Missing required columns: {', '.join(missing_required)}"
            )
        
        # Check for extra columns (not necessarily an error)
        all_expected = schema.required_columns | schema.optional_columns
        extra_columns = df_columns - all_expected
        if extra_columns:
            result.extra_columns = list(extra_columns)
            result.validation_warnings.append(
                f"Extra columns found: {', '.join(extra_columns)}"
            )
        
        # Validate data types
        self._validate_data_types(df, schema, result)
        
        # Check null percentages
        self._check_null_percentages(df, schema, result)
        
        # Calculate data quality score
        result.data_quality_score = self._calculate_quality_score(df, schema, result)
        
        self.logger.info(f"Schema validation for {Path(file_path).name}: "
                        f"{'PASSED' if result.is_valid else 'FAILED'} "
                        f"(Quality: {result.data_quality_score:.1f}%)")
        
        return result
    
    def _validate_data_types(self, 
                           df: pd.DataFrame, 
                           schema: SchemaDefinition, 
                           result: ValidationResult) -> None:
        """Validate data types for each column"""
        for column, expected_type in schema.column_types.items():
            if column not in df.columns:
                continue
            
            series = df[column]
            
            # Skip validation for columns that are entirely null
            if series.isna().all():
                continue
            
            type_issues = []
            
            if expected_type == 'numeric':
                # Check if column can be converted to numeric
                try:
                    pd.to_numeric(series, errors='coerce')
                    # Count how many values couldn't be converted
                    numeric_series = pd.to_numeric(series, errors='coerce')
                    non_numeric_count = numeric_series.isna().sum() - series.isna().sum()
                    
                    if non_numeric_count > 0:
                        percentage = (non_numeric_count / len(series)) * 100
                        type_issues.append(f"{non_numeric_count} non-numeric values ({percentage:.1f}%)")
                        
                except Exception as e:
                    type_issues.append(f"Cannot convert to numeric: {str(e)}")
            
            elif expected_type == 'string':
                # Check for non-string types (excluding NaN)
                non_string_mask = ~series.isna() & ~series.astype(str).eq(series.astype(str))
                if non_string_mask.any():
                    non_string_count = non_string_mask.sum()
                    percentage = (non_string_count / len(series)) * 100
                    type_issues.append(f"{non_string_count} non-string values ({percentage:.1f}%)")
            
            if type_issues:
                result.data_type_issues[column] = '; '.join(type_issues)
                result.validation_warnings.append(f"Data type issues in {column}: {'; '.join(type_issues)}")
    
    def _check_null_percentages(self, 
                              df: pd.DataFrame, 
                              schema: SchemaDefinition, 
                              result: ValidationResult) -> None:
        """Check null percentages for each column"""
        for column in df.columns:
            null_count = df[column].isna().sum()
            null_percentage = (null_count / len(df)) * 100
            result.null_percentages[column] = null_percentage
            
            # Check if null percentage exceeds threshold for non-nullable columns
            if column not in schema.nullable_columns and null_percentage > 0:
                result.validation_warnings.append(
                    f"Column '{column}' has {null_percentage:.1f}% null values but is not marked as nullable"
                )
            
            # Check if null percentage exceeds maximum allowed
            if null_percentage > schema.max_null_percentage:
                result.validation_warnings.append(
                    f"Column '{column}' has {null_percentage:.1f}% null values "
                    f"(exceeds {schema.max_null_percentage}% threshold)"
                )
    
    def _calculate_quality_score(self, 
                               df: pd.DataFrame, 
                               schema: SchemaDefinition, 
                               result: ValidationResult) -> float:
        """Calculate overall data quality score (0-100)"""
        scores = []
        
        # Schema completeness score (required columns present)
        required_present = len(schema.required_columns - set(result.missing_columns))
        schema_score = (required_present / len(schema.required_columns)) * 100
        scores.append(schema_score)
        
        # Data completeness score (low null percentages)
        if result.null_percentages:
            avg_null_percentage = sum(result.null_percentages.values()) / len(result.null_percentages)
            completeness_score = max(0, 100 - avg_null_percentage)
            scores.append(completeness_score)
        
        # Data type consistency score
        total_columns_with_types = len([col for col in schema.column_types.keys() if col in df.columns])
        columns_with_issues = len(result.data_type_issues)
        if total_columns_with_types > 0:
            type_score = ((total_columns_with_types - columns_with_issues) / total_columns_with_types) * 100
            scores.append(type_score)
        
        # Record count score (penalize very small datasets)
        if len(df) >= 100:
            record_score = 100
        elif len(df) >= 10:
            record_score = 80
        elif len(df) >= 1:
            record_score = 60
        else:
            record_score = 0
        scores.append(record_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def validate_schema_consistency(self, 
                                  validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Check schema consistency across multiple files.
        
        Args:
            validation_results: List of validation results from multiple files
            
        Returns:
            Dictionary with consistency analysis
        """
        if not validation_results:
            return {"error": "No validation results provided"}
        
        # Collect all columns from all files
        all_columns = set()
        file_columns = {}
        
        for result in validation_results:
            # We need to get the actual columns from the files
            # For now, we'll use the schema information
            file_columns[result.file_path] = set()
            all_columns.update(file_columns[result.file_path])
        
        # Find common columns
        if file_columns:
            common_columns = set.intersection(*file_columns.values()) if file_columns else set()
        else:
            common_columns = set()
        
        # Find files with schema differences
        schema_differences = {}
        for file_path, columns in file_columns.items():
            missing_common = common_columns - columns
            extra_columns = columns - common_columns
            
            if missing_common or extra_columns:
                schema_differences[file_path] = {
                    'missing_common': list(missing_common),
                    'extra_columns': list(extra_columns)
                }
        
        consistency_report = {
            'total_files': len(validation_results),
            'valid_files': sum(1 for r in validation_results if r.is_valid),
            'invalid_files': sum(1 for r in validation_results if not r.is_valid),
            'all_columns_found': sorted(all_columns),
            'common_columns': sorted(common_columns),
            'schema_differences': schema_differences,
            'average_quality_score': sum(r.data_quality_score for r in validation_results) / len(validation_results),
            'quality_score_range': {
                'min': min(r.data_quality_score for r in validation_results),
                'max': max(r.data_quality_score for r in validation_results)
            }
        }
        
        self.logger.info(f"Schema consistency check: {consistency_report['valid_files']}/{consistency_report['total_files']} files valid")
        
        return consistency_report
    
    def assess_data_quality(self, df: pd.DataFrame) -> QualityMetrics:
        """
        Assess overall data quality for aggregated dataset.
        
        Args:
            df: Aggregated DataFrame to assess
            
        Returns:
            QualityMetrics with detailed quality assessment
        """
        metrics = QualityMetrics()
        
        if df.empty:
            return metrics
        
        metrics.total_records = len(df)
        
        # Calculate duplicates based on address matching
        if all(col in df.columns for col in ['Address', 'City', 'State']):
            duplicate_mask = df.duplicated(subset=['Address', 'City', 'State'], keep=False)
            metrics.duplicate_records = duplicate_mask.sum()
            metrics.unique_records = metrics.total_records - metrics.duplicate_records
            metrics.duplicate_percentage = (metrics.duplicate_records / metrics.total_records) * 100
        else:
            metrics.unique_records = metrics.total_records
        
        # Calculate column completeness
        for column in df.columns:
            non_null_count = df[column].notna().sum()
            metrics.column_completeness[column] = (non_null_count / metrics.total_records) * 100
        
        # Overall completeness score
        if metrics.column_completeness:
            metrics.completeness_score = sum(metrics.column_completeness.values()) / len(metrics.column_completeness)
        
        # Data type consistency (simplified)
        numeric_columns = ['Building SqFt', 'Lot Size Acres', 'Year Built', 'Occupancy Rate', 'Flex Score']
        for column in numeric_columns:
            if column in df.columns:
                try:
                    numeric_series = pd.to_numeric(df[column], errors='coerce')
                    valid_numeric = numeric_series.notna().sum()
                    total_non_null = df[column].notna().sum()
                    if total_non_null > 0:
                        consistency = (valid_numeric / total_non_null) * 100
                        metrics.data_type_consistency[column] = consistency
                except Exception:
                    metrics.data_type_consistency[column] = 0.0
        
        # Consistency score
        if metrics.data_type_consistency:
            metrics.consistency_score = sum(metrics.data_type_consistency.values()) / len(metrics.data_type_consistency)
        
        # Value distribution for key categorical columns
        categorical_columns = ['Property Type', 'State', 'City']
        for column in categorical_columns:
            if column in df.columns:
                value_counts = df[column].value_counts().head(10).to_dict()
                metrics.value_distribution[column] = value_counts
        
        # Overall quality score
        scores = [metrics.completeness_score, metrics.consistency_score]
        
        # Penalize high duplicate percentage
        duplicate_penalty = min(metrics.duplicate_percentage, 50)  # Cap penalty at 50%
        duplicate_score = 100 - duplicate_penalty
        scores.append(duplicate_score)
        
        metrics.overall_quality_score = sum(s for s in scores if s > 0) / len([s for s in scores if s > 0])
        
        self.logger.info(f"Data quality assessment: {metrics.total_records:,} records, "
                        f"{metrics.duplicate_percentage:.1f}% duplicates, "
                        f"{metrics.overall_quality_score:.1f}% quality score")
        
        return metrics
    
    def generate_validation_report(self, 
                                 validation_results: List[ValidationResult],
                                 quality_metrics: Optional[QualityMetrics] = None) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Args:
            validation_results: List of validation results
            quality_metrics: Optional quality metrics for aggregated data
            
        Returns:
            Dictionary with comprehensive validation report
        """
        if not validation_results:
            return {"error": "No validation results provided"}
        
        # Summary statistics
        total_files = len(validation_results)
        valid_files = sum(1 for r in validation_results if r.is_valid)
        total_records = sum(r.record_count for r in validation_results)
        
        # Collect all errors and warnings
        all_errors = []
        all_warnings = []
        all_schema_issues = []
        
        for result in validation_results:
            all_errors.extend(result.validation_errors)
            all_warnings.extend(result.validation_warnings)
            all_schema_issues.extend(result.schema_issues)
        
        # Quality score statistics
        quality_scores = [r.data_quality_score for r in validation_results]
        
        report = {
            'validation_summary': {
                'total_files': total_files,
                'valid_files': valid_files,
                'invalid_files': total_files - valid_files,
                'success_rate': (valid_files / total_files) * 100,
                'total_records': total_records
            },
            'quality_summary': {
                'average_quality_score': sum(quality_scores) / len(quality_scores),
                'min_quality_score': min(quality_scores),
                'max_quality_score': max(quality_scores),
                'files_above_80_percent': sum(1 for score in quality_scores if score >= 80),
                'files_below_50_percent': sum(1 for score in quality_scores if score < 50)
            },
            'issues_summary': {
                'total_errors': len(all_errors),
                'total_warnings': len(all_warnings),
                'total_schema_issues': len(all_schema_issues),
                'common_errors': self._get_common_issues(all_errors),
                'common_warnings': self._get_common_issues(all_warnings)
            },
            'file_details': [
                {
                    'file_path': result.file_path,
                    'is_valid': result.is_valid,
                    'record_count': result.record_count,
                    'quality_score': result.data_quality_score,
                    'error_count': len(result.validation_errors),
                    'warning_count': len(result.validation_warnings)
                }
                for result in validation_results
            ]
        }
        
        # Add aggregated quality metrics if provided
        if quality_metrics:
            report['aggregated_quality'] = {
                'total_records': quality_metrics.total_records,
                'unique_records': quality_metrics.unique_records,
                'duplicate_percentage': quality_metrics.duplicate_percentage,
                'completeness_score': quality_metrics.completeness_score,
                'consistency_score': quality_metrics.consistency_score,
                'overall_quality_score': quality_metrics.overall_quality_score
            }
        
        return report
    
    def _get_common_issues(self, issues: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
        """Get most common issues from a list"""
        if not issues:
            return []
        
        # Count occurrences of each issue
        issue_counts = {}
        for issue in issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Sort by count and return top N
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'issue': issue, 'count': count}
            for issue, count in sorted_issues[:top_n]
        ]
    
    def validate_data_consistency(self, 
                                df: pd.DataFrame, 
                                consistency_rules: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Validate data consistency using business rules.
        
        Args:
            df: DataFrame to validate
            consistency_rules: Optional custom consistency rules
            
        Returns:
            List of consistency issues found
        """
        issues = []
        
        if df.empty:
            return issues
        
        # Default consistency rules for flex property data
        if consistency_rules is None:
            consistency_rules = {
                'building_sqft_range': (1000, 10000000),  # 1K to 10M sqft
                'lot_size_range': (0.1, 1000),  # 0.1 to 1000 acres
                'year_built_range': (1800, datetime.now().year + 2),
                'occupancy_rate_range': (0, 100),
                'flex_score_range': (0, 10)
            }
        
        # Check Building SqFt range
        if 'Building SqFt' in df.columns:
            min_sqft, max_sqft = consistency_rules.get('building_sqft_range', (0, float('inf')))
            numeric_sqft = pd.to_numeric(df['Building SqFt'], errors='coerce')
            out_of_range = ((numeric_sqft < min_sqft) | (numeric_sqft > max_sqft)) & numeric_sqft.notna()
            if out_of_range.any():
                count = out_of_range.sum()
                issues.append(f"{count} records have Building SqFt outside valid range ({min_sqft:,}-{max_sqft:,})")
        
        # Check Lot Size Acres range
        if 'Lot Size Acres' in df.columns:
            min_acres, max_acres = consistency_rules.get('lot_size_range', (0, float('inf')))
            numeric_acres = pd.to_numeric(df['Lot Size Acres'], errors='coerce')
            out_of_range = ((numeric_acres < min_acres) | (numeric_acres > max_acres)) & numeric_acres.notna()
            if out_of_range.any():
                count = out_of_range.sum()
                issues.append(f"{count} records have Lot Size Acres outside valid range ({min_acres}-{max_acres})")
        
        # Check Year Built range
        if 'Year Built' in df.columns:
            min_year, max_year = consistency_rules.get('year_built_range', (1800, 2030))
            numeric_year = pd.to_numeric(df['Year Built'], errors='coerce')
            out_of_range = ((numeric_year < min_year) | (numeric_year > max_year)) & numeric_year.notna()
            if out_of_range.any():
                count = out_of_range.sum()
                issues.append(f"{count} records have Year Built outside valid range ({min_year}-{max_year})")
        
        # Check State format (should be 2-letter codes)
        if 'State' in df.columns:
            invalid_states = df['State'].notna() & (df['State'].str.len() != 2)
            if invalid_states.any():
                count = invalid_states.sum()
                issues.append(f"{count} records have invalid State format (should be 2-letter code)")
        
        # Check for duplicate addresses
        if all(col in df.columns for col in ['Address', 'City', 'State']):
            duplicates = df.duplicated(subset=['Address', 'City', 'State'], keep=False)
            if duplicates.any():
                count = duplicates.sum()
                unique_duplicates = len(df[duplicates].drop_duplicates(subset=['Address', 'City', 'State']))
                issues.append(f"{count} records are duplicates ({unique_duplicates} unique addresses)")
        
        return issues