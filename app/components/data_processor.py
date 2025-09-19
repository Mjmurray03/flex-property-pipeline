"""
Intelligent data validation and quality assessment processor
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import re
from difflib import SequenceMatcher
import warnings

from ..core.interfaces import IDataProcessor, ValidationResult, ProcessingReport
from ..core.base_classes import ProcessorBase
from config.settings import get_config


@dataclass
class ColumnMapping:
    """Column mapping suggestion"""
    original_name: str
    suggested_name: str
    confidence: float
    mapping_type: str  # exact, fuzzy, pattern


@dataclass
class DataQualityMetric:
    """Data quality metric"""
    metric_name: str
    score: float
    description: str
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Comprehensive data quality report"""
    overall_score: float
    metrics: List[DataQualityMetric]
    column_profiles: Dict[str, Dict[str, Any]]
    data_fingerprint: str
    processing_time: float
    recommendations: List[str]
    warnings: List[str]
    generated_at: datetime


class DataProcessor(ProcessorBase, IDataProcessor):
    """Intelligent data processor with validation and quality assessment"""
    
    def __init__(self):
        super().__init__("DataProcessor")
        self.config = get_config()
        
        # Standard column mappings for property data
        self.standard_columns = {
            # Property identification
            'property_id': ['id', 'prop_id', 'property_id', 'parcel_id', 'pin'],
            'name': ['name', 'property_name', 'building_name', 'site_name'],
            
            # Location
            'address': ['address', 'street_address', 'full_address', 'location'],
            'city': ['city', 'municipality', 'town'],
            'state': ['state', 'province', 'region'],
            'county': ['county', 'district', 'parish'],
            'zip_code': ['zip', 'zipcode', 'postal_code', 'zip_code'],
            
            # Property details
            'property_type': ['type', 'property_type', 'building_type', 'use_type'],
            'building_sqft': ['sqft', 'square_feet', 'building_sqft', 'floor_area', 'gfa'],
            'lot_size_acres': ['lot_size', 'acres', 'lot_acres', 'land_area'],
            'year_built': ['year_built', 'construction_year', 'built_year', 'year_constructed'],
            
            # Financial
            'sold_price': ['price', 'sale_price', 'sold_price', 'market_value'],
            'assessed_value': ['assessed_value', 'tax_value', 'appraised_value'],
            'rent': ['rent', 'rental_rate', 'monthly_rent'],
            
            # Additional metrics
            'occupancy_rate': ['occupancy', 'occupancy_rate', 'occupied_percent'],
            'cap_rate': ['cap_rate', 'capitalization_rate', 'yield'],
            'noi': ['noi', 'net_operating_income', 'operating_income']
        }
        
        # Data type expectations
        self.column_types = {
            'property_id': 'string',
            'name': 'string',
            'address': 'string',
            'city': 'string',
            'state': 'string',
            'county': 'string',
            'zip_code': 'string',
            'property_type': 'string',
            'building_sqft': 'numeric',
            'lot_size_acres': 'numeric',
            'year_built': 'integer',
            'sold_price': 'numeric',
            'assessed_value': 'numeric',
            'rent': 'numeric',
            'occupancy_rate': 'numeric',
            'cap_rate': 'numeric',
            'noi': 'numeric'
        }
    
    def _do_initialize(self) -> None:
        """Initialize data processor"""
        self.logger.info("Initializing intelligent data processor")
    
    def _do_cleanup(self) -> None:
        """Cleanup data processor"""
        pass

    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data and return validation results"""
        validation_result = self.validate_structure(df)

        return {
            'is_valid': validation_result.is_valid,
            'quality_score': validation_result.quality_score,
            'errors': validation_result.errors,
            'warnings': validation_result.warnings,
            'recommendations': getattr(validation_result, 'recommendations', [])
        }

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data using existing clean_and_transform method"""
        return self.clean_and_transform(df)

    def add_calculated_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated fields to the dataframe"""
        df_enhanced = df.copy()

        # Add price per sqft if both price and sqft are available
        if 'sold_price' in df_enhanced.columns and 'building_sqft' in df_enhanced.columns:
            df_enhanced['price_per_sqft'] = df_enhanced['sold_price'] / df_enhanced['building_sqft']

        # Add building age if year built is available
        if 'year_built' in df_enhanced.columns:
            current_year = datetime.now().year
            df_enhanced['building_age'] = current_year - df_enhanced['year_built']

        # Add lot efficiency ratio if both building sqft and lot size are available
        if 'building_sqft' in df_enhanced.columns and 'lot_size_sqft' in df_enhanced.columns:
            df_enhanced['lot_efficiency'] = df_enhanced['building_sqft'] / df_enhanced['lot_size_sqft']

        return df_enhanced

    def prepare_export_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for export with metadata"""
        df_export = df.copy()

        # Add export metadata
        df_export['export_timestamp'] = datetime.now().isoformat()
        df_export['data_source'] = 'Flex Property Intelligence Platform'

        return df_export
    
    def process_upload(self, file_data: Any) -> Tuple[pd.DataFrame, ProcessingReport]:
        """Process uploaded file data with comprehensive validation"""
        start_time = datetime.now()
        
        try:
            # Load data based on file type
            if hasattr(file_data, 'name'):
                if file_data.name.endswith('.csv'):
                    df = pd.read_csv(file_data)
                elif file_data.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_data)
                else:
                    raise ValueError(f"Unsupported file type: {file_data.name}")
            else:
                # Assume it's already a DataFrame or CSV data
                if isinstance(file_data, pd.DataFrame):
                    df = file_data.copy()
                else:
                    df = pd.read_csv(file_data)
            
            # Generate data fingerprint
            fingerprint = self._generate_data_fingerprint(df)
            
            # Validate structure
            validation_result = self.validate_structure(df)
            
            # Clean and transform if validation passes
            if validation_result.is_valid or validation_result.quality_score > 0.5:
                df_cleaned = self.clean_and_transform(df)
            else:
                df_cleaned = df
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Generate processing report
            report = ProcessingReport(
                file_fingerprint=fingerprint,
                processing_time=processing_time,
                memory_usage=df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
                rows_processed=len(df),
                columns_cleaned=list(df_cleaned.columns),
                quality_score=validation_result.quality_score,
                warnings=validation_result.warnings,
                recommendations=self._generate_recommendations(df, validation_result),
                created_at=datetime.now()
            )
            
            self._record_processing(processing_time, True)
            return df_cleaned, report
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._record_processing(processing_time, False)
            self.logger.error(f"Error processing upload: {str(e)}")
            raise
    
    def validate_structure(self, df: pd.DataFrame) -> ValidationResult:
        """Validate data structure and quality"""
        errors = []
        warnings = []
        quality_metrics = []
        
        # Basic structure validation
        if df.empty:
            errors.append("Dataset is empty")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                quality_score=0.0
            )
        
        # Check minimum columns
        if len(df.columns) < 3:
            warnings.append("Dataset has very few columns (less than 3)")
        
        # Check for duplicate columns
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            errors.append(f"Duplicate column names found: {duplicate_cols}")
        
        # Validate column names
        column_issues = self._validate_column_names(df.columns)
        warnings.extend(column_issues)
        
        # Data quality assessment
        quality_metrics = self._assess_data_quality(df)
        
        # Calculate overall quality score
        if quality_metrics:
            quality_score = np.mean([metric.score for metric in quality_metrics])
        else:
            quality_score = 0.5  # Neutral score if no metrics
        
        # Determine if valid (no critical errors and reasonable quality)
        is_valid = len(errors) == 0 and quality_score > 0.3
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            quality_score=quality_score
        )
    
    def clean_and_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and transform data with detailed logging"""
        df_cleaned = df.copy()
        
        # Clean column names
        df_cleaned.columns = self._clean_column_names(df_cleaned.columns)
        
        # Handle missing values
        df_cleaned = self._handle_missing_values(df_cleaned)
        
        # Clean numeric columns
        df_cleaned = self._clean_numeric_columns(df_cleaned)
        
        # Clean text columns
        df_cleaned = self._clean_text_columns(df_cleaned)
        
        # Remove duplicates
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        if len(df_cleaned) < initial_rows:
            self.logger.info(f"Removed {initial_rows - len(df_cleaned)} duplicate rows")
        
        # Generate calculated fields
        df_cleaned = self._generate_calculated_fields(df_cleaned)
        
        return df_cleaned
    
    def generate_quality_report(self, df: pd.DataFrame) -> QualityReport:
        """Generate comprehensive data quality report"""
        start_time = datetime.now()
        
        # Assess data quality
        quality_metrics = self._assess_data_quality(df)
        
        # Generate column profiles
        column_profiles = self._generate_column_profiles(df)
        
        # Calculate overall score
        overall_score = np.mean([metric.score for metric in quality_metrics]) if quality_metrics else 0.5
        
        # Generate recommendations
        recommendations = self._generate_quality_recommendations(quality_metrics, column_profiles)
        
        # Generate warnings
        warnings = self._generate_quality_warnings(df, quality_metrics)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QualityReport(
            overall_score=overall_score,
            metrics=quality_metrics,
            column_profiles=column_profiles,
            data_fingerprint=self._generate_data_fingerprint(df),
            processing_time=processing_time,
            recommendations=recommendations,
            warnings=warnings,
            generated_at=datetime.now()
        )
    
    def suggest_column_mapping(self, df: pd.DataFrame) -> List[ColumnMapping]:
        """Suggest column mappings using fuzzy matching"""
        mappings = []
        
        for col in df.columns:
            best_match = self._find_best_column_match(col)
            if best_match:
                mappings.append(best_match)
        
        return mappings
    
    def _generate_data_fingerprint(self, df: pd.DataFrame) -> str:
        """Generate unique fingerprint for dataset"""
        # Create fingerprint based on structure and sample data
        fingerprint_data = {
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'shape': df.shape,
            'sample_hash': hashlib.md5(str(df.head().values).encode()).hexdigest()
        }
        
        fingerprint_str = str(fingerprint_data)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()
    
    def _validate_column_names(self, columns: pd.Index) -> List[str]:
        """Validate column names and suggest improvements"""
        issues = []
        
        for col in columns:
            # Check for problematic characters
            if re.search(r'[^\w\s-]', col):
                issues.append(f"Column '{col}' contains special characters")
            
            # Check for very long names
            if len(col) > 50:
                issues.append(f"Column '{col}' has very long name (>50 chars)")
            
            # Check for numeric-only names
            if col.isdigit():
                issues.append(f"Column '{col}' is numeric-only")
        
        return issues
    
    def _assess_data_quality(self, df: pd.DataFrame) -> List[DataQualityMetric]:
        """Assess comprehensive data quality metrics"""
        metrics = []
        
        # Completeness metric
        completeness_score = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        completeness_issues = []
        completeness_recommendations = []
        
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            completeness_issues.append(f"Missing values in columns: {missing_cols}")
            completeness_recommendations.append("Consider imputation strategies for missing values")
        
        metrics.append(DataQualityMetric(
            metric_name="Completeness",
            score=completeness_score,
            description="Percentage of non-missing values",
            issues=completeness_issues,
            recommendations=completeness_recommendations
        ))
        
        # Consistency metric
        consistency_score, consistency_issues, consistency_recommendations = self._assess_consistency(df)
        metrics.append(DataQualityMetric(
            metric_name="Consistency",
            score=consistency_score,
            description="Data format and type consistency",
            issues=consistency_issues,
            recommendations=consistency_recommendations
        ))
        
        # Validity metric
        validity_score, validity_issues, validity_recommendations = self._assess_validity(df)
        metrics.append(DataQualityMetric(
            metric_name="Validity",
            score=validity_score,
            description="Data conforms to expected formats and ranges",
            issues=validity_issues,
            recommendations=validity_recommendations
        ))
        
        # Uniqueness metric
        uniqueness_score = 1 - (len(df) - len(df.drop_duplicates())) / len(df)
        uniqueness_issues = []
        uniqueness_recommendations = []
        
        if uniqueness_score < 1.0:
            duplicate_count = len(df) - len(df.drop_duplicates())
            uniqueness_issues.append(f"Found {duplicate_count} duplicate rows")
            uniqueness_recommendations.append("Remove duplicate records")
        
        metrics.append(DataQualityMetric(
            metric_name="Uniqueness",
            score=uniqueness_score,
            description="Percentage of unique records",
            issues=uniqueness_issues,
            recommendations=uniqueness_recommendations
        ))
        
        return metrics
    
    def _assess_consistency(self, df: pd.DataFrame) -> Tuple[float, List[str], List[str]]:
        """Assess data consistency"""
        issues = []
        recommendations = []
        scores = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed case in text columns
                if df[col].notna().any():
                    unique_values = df[col].dropna().unique()
                    if len(unique_values) > 1:
                        # Check for case inconsistencies
                        lower_values = set(str(v).lower() for v in unique_values)
                        if len(lower_values) < len(unique_values):
                            issues.append(f"Column '{col}' has case inconsistencies")
                            recommendations.append(f"Standardize case in column '{col}'")
                            scores.append(0.7)
                        else:
                            scores.append(1.0)
                    else:
                        scores.append(1.0)
            else:
                scores.append(1.0)
        
        overall_score = np.mean(scores) if scores else 1.0
        return overall_score, issues, recommendations
    
    def _assess_validity(self, df: pd.DataFrame) -> Tuple[float, List[str], List[str]]:
        """Assess data validity"""
        issues = []
        recommendations = []
        scores = []
        
        for col in df.columns:
            col_score = 1.0
            
            # Check numeric columns for outliers
            if pd.api.types.is_numeric_dtype(df[col]):
                if not df[col].empty:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    if len(outliers) > 0:
                        outlier_percentage = len(outliers) / len(df) * 100
                        if outlier_percentage > 5:  # More than 5% outliers
                            issues.append(f"Column '{col}' has {outlier_percentage:.1f}% outliers")
                            recommendations.append(f"Review outliers in column '{col}'")
                            col_score = max(0.5, 1 - outlier_percentage / 100)
            
            # Check for negative values where they shouldn't be
            if col.lower() in ['price', 'sqft', 'acres', 'year_built'] and pd.api.types.is_numeric_dtype(df[col]):
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    issues.append(f"Column '{col}' has {negative_count} negative values")
                    recommendations.append(f"Review negative values in column '{col}'")
                    col_score = min(col_score, 0.8)
            
            scores.append(col_score)
        
        overall_score = np.mean(scores) if scores else 1.0
        return overall_score, issues, recommendations
    
    def _generate_column_profiles(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Generate detailed profiles for each column"""
        profiles = {}
        
        for col in df.columns:
            profile = {
                'name': col,
                'dtype': str(df[col].dtype),
                'non_null_count': df[col].notna().sum(),
                'null_count': df[col].isnull().sum(),
                'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_count': df[col].nunique(),
                'unique_percentage': (df[col].nunique() / len(df)) * 100 if len(df) > 0 else 0
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                profile.update({
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'q25': df[col].quantile(0.25),
                    'q75': df[col].quantile(0.75)
                })
            elif df[col].dtype == 'object':
                profile.update({
                    'avg_length': df[col].astype(str).str.len().mean(),
                    'max_length': df[col].astype(str).str.len().max(),
                    'min_length': df[col].astype(str).str.len().min()
                })
                
                # Most common values
                value_counts = df[col].value_counts().head(5)
                profile['top_values'] = value_counts.to_dict()
            
            profiles[col] = profile
        
        return profiles
    
    def _clean_column_names(self, columns: pd.Index) -> List[str]:
        """Clean and standardize column names"""
        cleaned = []
        
        for col in columns:
            # Convert to lowercase and replace spaces/special chars with underscores
            clean_name = re.sub(r'[^\w\s]', '', str(col).lower())
            clean_name = re.sub(r'\s+', '_', clean_name.strip())
            
            # Remove leading/trailing underscores
            clean_name = clean_name.strip('_')
            
            # Ensure not empty
            if not clean_name:
                clean_name = f"column_{len(cleaned)}"
            
            cleaned.append(clean_name)
        
        return cleaned
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with intelligent strategies"""
        df_cleaned = df.copy()
        
        for col in df_cleaned.columns:
            missing_count = df_cleaned[col].isnull().sum()
            if missing_count > 0:
                missing_percentage = (missing_count / len(df_cleaned)) * 100
                
                if missing_percentage > 50:
                    self.logger.warning(f"Column '{col}' has {missing_percentage:.1f}% missing values")
                
                # Strategy based on data type and missing percentage
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    if missing_percentage < 20:
                        # Use median for numeric columns with low missing percentage
                        df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                    else:
                        # Use forward fill for high missing percentage
                        df_cleaned[col].fillna(method='ffill', inplace=True)
                        df_cleaned[col].fillna(0, inplace=True)  # Fill remaining with 0
                
                elif df_cleaned[col].dtype == 'object':
                    if missing_percentage < 20:
                        # Use mode for categorical columns
                        mode_value = df_cleaned[col].mode()
                        if not mode_value.empty:
                            df_cleaned[col].fillna(mode_value[0], inplace=True)
                        else:
                            df_cleaned[col].fillna('Unknown', inplace=True)
                    else:
                        df_cleaned[col].fillna('Unknown', inplace=True)
        
        return df_cleaned
    
    def _clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric columns"""
        df_cleaned = df.copy()
        
        for col in df_cleaned.columns:
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                # Remove extreme outliers (beyond 3 standard deviations)
                if df_cleaned[col].std() > 0:
                    mean = df_cleaned[col].mean()
                    std = df_cleaned[col].std()
                    outlier_mask = np.abs(df_cleaned[col] - mean) > 3 * std
                    
                    if outlier_mask.sum() > 0:
                        self.logger.info(f"Capping {outlier_mask.sum()} extreme outliers in column '{col}'")
                        # Cap outliers instead of removing
                        upper_cap = mean + 3 * std
                        lower_cap = mean - 3 * std
                        df_cleaned[col] = df_cleaned[col].clip(lower=lower_cap, upper=upper_cap)
        
        return df_cleaned
    
    def _clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text columns"""
        df_cleaned = df.copy()
        
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                # Strip whitespace
                df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
                
                # Standardize case for certain columns
                if any(keyword in col.lower() for keyword in ['city', 'state', 'type']):
                    df_cleaned[col] = df_cleaned[col].str.title()
                
                # Clean up common issues
                df_cleaned[col] = df_cleaned[col].replace(['nan', 'NaN', 'null', 'NULL', ''], np.nan)
        
        return df_cleaned
    
    def _generate_calculated_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate calculated fields based on available data"""
        df_enhanced = df.copy()
        
        # Price per square foot
        if 'sold_price' in df_enhanced.columns and 'building_sqft' in df_enhanced.columns:
            mask = (df_enhanced['building_sqft'] > 0) & (df_enhanced['sold_price'] > 0)
            df_enhanced.loc[mask, 'price_per_sqft'] = (
                df_enhanced.loc[mask, 'sold_price'] / df_enhanced.loc[mask, 'building_sqft']
            )
        
        # Building age
        if 'year_built' in df_enhanced.columns:
            current_year = datetime.now().year
            df_enhanced['building_age'] = current_year - df_enhanced['year_built']
        
        # Building efficiency (sqft per acre)
        if 'building_sqft' in df_enhanced.columns and 'lot_size_acres' in df_enhanced.columns:
            mask = df_enhanced['lot_size_acres'] > 0
            df_enhanced.loc[mask, 'building_efficiency'] = (
                df_enhanced.loc[mask, 'building_sqft'] / (df_enhanced.loc[mask, 'lot_size_acres'] * 43560)
            )
        
        return df_enhanced
    
    def _find_best_column_match(self, column_name: str) -> Optional[ColumnMapping]:
        """Find best matching standard column name"""
        column_lower = column_name.lower().strip()
        best_match = None
        best_confidence = 0.0
        
        for standard_col, variations in self.standard_columns.items():
            for variation in variations:
                # Exact match
                if column_lower == variation.lower():
                    return ColumnMapping(
                        original_name=column_name,
                        suggested_name=standard_col,
                        confidence=1.0,
                        mapping_type="exact"
                    )
                
                # Fuzzy match
                similarity = SequenceMatcher(None, column_lower, variation.lower()).ratio()
                if similarity > best_confidence and similarity > 0.6:
                    best_confidence = similarity
                    best_match = ColumnMapping(
                        original_name=column_name,
                        suggested_name=standard_col,
                        confidence=similarity,
                        mapping_type="fuzzy"
                    )
        
        return best_match
    
    def _generate_recommendations(self, df: pd.DataFrame, validation_result: ValidationResult) -> List[str]:
        """Generate processing recommendations"""
        recommendations = []
        
        if validation_result.quality_score < 0.7:
            recommendations.append("Consider data cleaning to improve quality score")
        
        # Check for missing standard columns
        current_cols = [col.lower() for col in df.columns]
        missing_important = []
        
        important_cols = ['address', 'city', 'property_type', 'building_sqft']
        for col in important_cols:
            if not any(col in current_col for current_col in current_cols):
                missing_important.append(col)
        
        if missing_important:
            recommendations.append(f"Consider adding missing important columns: {missing_important}")
        
        return recommendations
    
    def _generate_quality_recommendations(self, metrics: List[DataQualityMetric], profiles: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        for metric in metrics:
            recommendations.extend(metric.recommendations)
        
        # Additional recommendations based on profiles
        for col, profile in profiles.items():
            if profile['null_percentage'] > 30:
                recommendations.append(f"Column '{col}' has high missing values ({profile['null_percentage']:.1f}%)")
            
            if profile['unique_percentage'] < 1 and profile['unique_count'] > 1:
                recommendations.append(f"Column '{col}' has many duplicate values")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _generate_quality_warnings(self, df: pd.DataFrame, metrics: List[DataQualityMetric]) -> List[str]:
        """Generate quality warnings"""
        warnings = []
        
        for metric in metrics:
            if metric.score < 0.5:
                warnings.append(f"Low {metric.metric_name.lower()} score: {metric.score:.2f}")
        
        return warnings