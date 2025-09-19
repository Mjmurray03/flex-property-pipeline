"""
Data type detection and conversion utilities for handling categorical data in numerical operations.
Addresses pandas categorical data type issues that cause runtime errors in filtering and calculations.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import re
from utils.logger import setup_logging
from utils.categorical_error_handler import (
    get_categorical_error_handler, 
    handle_categorical_mean_error,
    handle_categorical_comparison_error,
    handle_categorical_conversion_error,
    log_conversion_success,
    log_data_quality_issue
)
from utils.data_type_performance import (
    get_performance_cache,
    get_lazy_converter,
    get_optimized_validator,
    monitor_performance
)


def detect_categorical_numeric_columns(df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> Dict[str, Dict]:
    """
    Detect categorical columns that should be numeric and assess their conversion potential.
    
    Args:
        df: DataFrame to analyze
        logger: Optional logger instance
        
    Returns:
        Dictionary mapping column names to conversion metadata:
        {
            'column_name': {
                'original_dtype': str,
                'categories': list,
                'numeric_categories': int,
                'conversion_feasible': bool,
                'conversion_confidence': float,
                'sample_values': list,
                'issues': list
            }
        }
    """
    if logger is None:
        logger = setup_logging('categorical_detector')
    
    logger.info("Detecting categorical columns that should be numeric...")
    
    categorical_columns = {}
    
    for col in df.columns:
        if df[col].dtype.name == 'category':
            logger.debug(f"Analyzing categorical column: {col}")
            
            # Get categories
            categories = df[col].cat.categories.tolist()
            
            # Analyze categories for numeric patterns
            analysis = _analyze_categories_for_numeric(categories, col, logger)
            
            if analysis['conversion_feasible']:
                categorical_columns[col] = {
                    'original_dtype': str(df[col].dtype),
                    'categories': categories,
                    'numeric_categories': analysis['numeric_count'],
                    'conversion_feasible': analysis['conversion_feasible'],
                    'conversion_confidence': analysis['confidence'],
                    'sample_values': categories[:5],  # First 5 categories as sample
                    'issues': analysis['issues'],
                    'recommended_target_type': analysis['target_type']
                }
                
                logger.info(f"Column '{col}': Conversion feasible (confidence: {analysis['confidence']:.2f})")
            else:
                logger.debug(f"Column '{col}': Not suitable for numeric conversion")
    
    logger.info(f"Found {len(categorical_columns)} categorical columns suitable for numeric conversion")
    return categorical_columns


def _analyze_categories_for_numeric(categories: List, column_name: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Analyze categorical categories to determine if they can be converted to numeric.
    
    Args:
        categories: List of category values
        column_name: Name of the column being analyzed
        logger: Logger instance
        
    Returns:
        Analysis results dictionary
    """
    numeric_count = 0
    issues = []
    confidence = 0.0
    target_type = 'float64'
    
    # Patterns for numeric detection
    numeric_patterns = [
        r'^\d+$',                    # Pure integers
        r'^\d+\.\d+$',              # Pure decimals
        r'^\$\d+(?:,\d{3})*(?:\.\d{2})?$',  # Currency format
        r'^\d+(?:,\d{3})*$',        # Comma-separated integers
        r'^\d+(?:,\d{3})*\.\d+$',   # Comma-separated decimals
        r'^\d+%$',                  # Percentages
        r'^-?\d+(?:\.\d+)?$',       # Negative numbers
    ]
    
    for category in categories:
        category_str = str(category).strip()
        
        # Skip NaN/null values
        if pd.isna(category) or category_str.lower() in ['nan', 'null', 'none', '']:
            continue
            
        # Check if matches any numeric pattern
        is_numeric = False
        for pattern in numeric_patterns:
            if re.match(pattern, category_str):
                is_numeric = True
                break
        
        # Also try direct pandas conversion
        if not is_numeric:
            try:
                pd.to_numeric(category_str)
                is_numeric = True
            except (ValueError, TypeError):
                pass
        
        if is_numeric:
            numeric_count += 1
        else:
            issues.append(f"Non-numeric category: '{category_str}'")
    
    total_categories = len([c for c in categories if not pd.isna(c)])
    
    if total_categories > 0:
        confidence = numeric_count / total_categories
    
    # Determine if conversion is feasible
    conversion_feasible = confidence >= 0.8  # 80% of categories must be numeric
    
    # Determine target type
    if conversion_feasible:
        # Check if all numeric categories are integers
        all_integers = True
        for category in categories:
            if pd.isna(category):
                continue
            try:
                val = pd.to_numeric(str(category).strip().replace('$', '').replace(',', '').replace('%', ''))
                if not pd.isna(val) and val != int(val):
                    all_integers = False
                    break
            except:
                continue
        
        target_type = 'int64' if all_integers else 'float64'
    
    return {
        'numeric_count': numeric_count,
        'total_count': total_categories,
        'confidence': confidence,
        'conversion_feasible': conversion_feasible,
        'issues': issues,
        'target_type': target_type
    }


def assess_conversion_feasibility(series: Union[pd.Series, pd.Categorical], logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Assess if a categorical series can be converted to numeric and provide detailed analysis.
    
    Args:
        series: Pandas Series or Categorical to assess
        logger: Optional logger instance
        
    Returns:
        Feasibility assessment dictionary:
        {
            'feasible': bool,
            'confidence': float,
            'success_probability': float,
            'estimated_null_count': int,
            'sample_conversions': dict,
            'issues': list,
            'recommendations': list
        }
    """
    if logger is None:
        logger = setup_logging('conversion_assessor')
    
    # Convert Categorical to Series if needed
    if isinstance(series, pd.Categorical):
        series = pd.Series(series)
    
    if series.dtype.name != 'category':
        return {
            'feasible': False,
            'confidence': 0.0,
            'success_probability': 0.0,
            'estimated_null_count': 0,
            'sample_conversions': {},
            'issues': ['Series is not categorical'],
            'recommendations': ['No conversion needed - series is already numeric or non-categorical']
        }
    
    logger.debug(f"Assessing conversion feasibility for categorical series with {len(series)} values")
    
    # Get sample of actual values (not just categories)
    sample_size = min(100, len(series))
    non_null_series = series.dropna()
    if len(non_null_series) == 0:
        sample_values = []
    else:
        sample_values = non_null_series.sample(n=min(sample_size, len(non_null_series)), random_state=42).tolist()
    
    successful_conversions = 0
    failed_conversions = 0
    sample_conversions = {}
    issues = []
    
    for value in sample_values:
        try:
            converted = pd.to_numeric(str(value).strip(), errors='raise')
            successful_conversions += 1
            sample_conversions[str(value)] = float(converted)
        except (ValueError, TypeError) as e:
            failed_conversions += 1
            issues.append(f"Failed to convert '{value}': {str(e)}")
    
    total_sample = successful_conversions + failed_conversions
    confidence = successful_conversions / total_sample if total_sample > 0 else 0.0
    
    # Estimate null count for full series
    estimated_null_count = int((failed_conversions / total_sample) * len(series)) if total_sample > 0 else 0
    
    # Determine feasibility
    feasible = confidence >= 0.8
    success_probability = confidence
    
    # Generate recommendations
    recommendations = []
    if feasible:
        recommendations.append("Conversion recommended - high success probability")
        if estimated_null_count > 0:
            recommendations.append(f"Expect approximately {estimated_null_count} null values after conversion")
    else:
        recommendations.append("Conversion not recommended - too many non-numeric values")
        recommendations.append("Consider data cleaning or alternative handling strategies")
    
    return {
        'feasible': feasible,
        'confidence': confidence,
        'success_probability': success_probability,
        'estimated_null_count': estimated_null_count,
        'sample_conversions': sample_conversions,
        'issues': issues[:10],  # Limit to first 10 issues
        'recommendations': recommendations
    }


@monitor_performance("safe_numeric_conversion")
def safe_numeric_conversion(series: Union[pd.Series, pd.Categorical], column_name: str = "unknown", 
                          logger: Optional[logging.Logger] = None) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Safely convert a series to numeric with comprehensive error handling and reporting.
    
    Args:
        series: Series or Categorical to convert
        column_name: Name of the column for logging
        logger: Optional logger instance
        
    Returns:
        Tuple of (converted_series, conversion_report)
    """
    if logger is None:
        logger = setup_logging('safe_converter')
    
    logger.debug(f"Starting safe numeric conversion for column '{column_name}'")
    
    # Convert Categorical to Series if needed
    if isinstance(series, pd.Categorical):
        series = pd.Series(series)
    
    original_dtype = str(series.dtype)
    original_length = len(series)
    original_null_count = series.isna().sum()
    
    # Initialize conversion report
    conversion_report = {
        'column_name': column_name,
        'original_dtype': original_dtype,
        'original_length': original_length,
        'original_null_count': original_null_count,
        'conversion_successful': False,
        'target_dtype': 'float64',
        'final_null_count': original_null_count,
        'values_converted': 0,
        'values_failed': 0,
        'conversion_method': 'none',
        'sample_original': [],
        'sample_converted': [],
        'warnings': [],
        'errors': []
    }
    
    try:
        # If already numeric, return as-is
        if pd.api.types.is_numeric_dtype(series):
            conversion_report.update({
                'conversion_successful': True,
                'target_dtype': str(series.dtype),
                'conversion_method': 'already_numeric'
            })
            logger.debug(f"Column '{column_name}' is already numeric")
            return series, conversion_report
        
        # Store sample of original values
        sample_mask = series.notna()
        if sample_mask.any():
            sample_original = series[sample_mask].head(5).tolist()
            conversion_report['sample_original'] = [str(x) for x in sample_original]
        
        # Enhanced conversion with preprocessing for common formats
        if series.dtype.name == 'category':
            # For categorical data, convert to string first and preprocess
            logger.debug(f"Converting categorical column '{column_name}' to numeric")
            string_series = series.astype(str)
            
            # Preprocess common numeric formats
            cleaned_series = _preprocess_numeric_strings(string_series, column_name, logger)
            
            converted_series = pd.to_numeric(cleaned_series, errors='coerce')
            conversion_report['conversion_method'] = 'categorical_to_numeric_with_preprocessing'
        else:
            # For other data types, also apply preprocessing
            logger.debug(f"Converting {original_dtype} column '{column_name}' to numeric")
            string_series = series.astype(str)
            cleaned_series = _preprocess_numeric_strings(string_series, column_name, logger)
            converted_series = pd.to_numeric(cleaned_series, errors='coerce')
            conversion_report['conversion_method'] = 'direct_conversion_with_preprocessing'
        
        # Calculate conversion statistics
        final_null_count = converted_series.isna().sum()
        # Values converted = non-null values in result that weren't null in original
        values_converted = (converted_series.notna() & series.notna()).sum()
        # Values failed = originally non-null values that became null
        values_failed = (converted_series.isna() & series.notna()).sum()
        
        # Store sample of converted values
        if converted_series.notna().any():
            sample_converted = converted_series.dropna().head(5).tolist()
            conversion_report['sample_converted'] = sample_converted
        
        # Update conversion report
        conversion_report.update({
            'conversion_successful': True,
            'target_dtype': str(converted_series.dtype),
            'final_null_count': final_null_count,
            'values_converted': values_converted,
            'values_failed': values_failed
        })
        
        # Add warnings for failed conversions
        if values_failed > 0:
            warning_msg = f"{values_failed} values could not be converted and became null"
            conversion_report['warnings'].append(warning_msg)
            logger.warning(f"Column '{column_name}': {warning_msg}")
        
        # Success message
        success_rate = (values_converted / original_length) * 100 if original_length > 0 else 0
        logger.info(f"Column '{column_name}': Conversion successful ({success_rate:.1f}% success rate)")
        
        return converted_series, conversion_report
        
    except Exception as e:
        error_msg = f"Conversion failed: {str(e)}"
        conversion_report['errors'].append(error_msg)
        logger.error(f"Column '{column_name}': {error_msg}")
        
        # Return original series on failure
        return series, conversion_report


def _preprocess_numeric_strings(series: pd.Series, column_name: str, logger: logging.Logger) -> pd.Series:
    """
    Preprocess string series to handle common numeric formats before conversion.
    
    Args:
        series: String series to preprocess
        column_name: Name of the column for logging
        logger: Logger instance
        
    Returns:
        Preprocessed series ready for numeric conversion
    """
    logger.debug(f"Preprocessing numeric strings for column '{column_name}'")
    
    # Create a copy to avoid modifying original
    cleaned = series.copy()
    
    # Handle null/empty values
    cleaned = cleaned.fillna('')
    
    # Remove common non-numeric characters and patterns
    preprocessing_steps = []
    
    # Step 1: Handle percentage values (e.g., "85%" -> "85")
    if cleaned.str.contains('%', na=False).any():
        cleaned = cleaned.str.replace('%', '', regex=False)
        preprocessing_steps.append("Removed percentage symbols")
    
    # Step 2: Handle currency symbols (e.g., "$1,000" -> "1000")
    if cleaned.str.contains(r'[\$£€¥]', na=False, regex=True).any():
        cleaned = cleaned.str.replace(r'[\$£€¥]', '', regex=True)
        preprocessing_steps.append("Removed currency symbols")
    
    # Step 3: Handle comma separators (e.g., "1,000" -> "1000")
    if cleaned.str.contains(',', na=False).any():
        cleaned = cleaned.str.replace(',', '', regex=False)
        preprocessing_steps.append("Removed comma separators")
    
    # Step 4: Handle parentheses for negative numbers (e.g., "(100)" -> "-100")
    if cleaned.str.contains(r'\([0-9.,]+\)', na=False, regex=True).any():
        cleaned = cleaned.str.replace(r'\(([0-9.,]+)\)', r'-\1', regex=True)
        preprocessing_steps.append("Converted parentheses to negative signs")
    
    # Step 5: Handle extra whitespace
    cleaned = cleaned.str.strip()
    
    # Step 6: Handle common text representations of zero
    zero_patterns = ['none', 'null', 'n/a', 'na', '-', '']
    for pattern in zero_patterns:
        if cleaned.str.lower().eq(pattern).any():
            cleaned = cleaned.replace(pattern, '0', regex=False)
            cleaned = cleaned.str.replace(pattern.upper(), '0', regex=False)
    
    # Step 7: Handle decimal points and ensure proper format
    # Remove multiple decimal points, keep only the first one
    cleaned = cleaned.str.replace(r'\.(?=.*\.)', '', regex=True)
    
    # Log preprocessing steps
    if preprocessing_steps:
        logger.debug(f"Column '{column_name}' preprocessing: {', '.join(preprocessing_steps)}")
    
    return cleaned


@monitor_performance("batch_categorical_conversion")
def convert_categorical_to_numeric(df: pd.DataFrame, columns: Optional[List[str]] = None,
                                 logger: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Convert multiple categorical columns to numeric with comprehensive reporting and caching.
    
    Args:
        df: DataFrame to process
        columns: Specific columns to convert (if None, auto-detect)
        logger: Optional logger instance
        
    Returns:
        Tuple of (converted_dataframe, conversion_reports)
    """
    if logger is None:
        logger = setup_logging('batch_converter')
    
    logger.info("Starting batch categorical to numeric conversion...")
    
    # Get performance cache
    cache = get_performance_cache()
    
    # Auto-detect columns if not specified
    if columns is None:
        # Check cache for column analysis
        cached_analysis = None
        try:
            # Create a simple key for the dataframe structure
            df_key = f"{df.shape}_{list(df.dtypes.astype(str))}"
            cached_analysis = cache.get_column_analysis(pd.Series([df_key]), 'categorical_detection')
        except:
            pass
        
        if cached_analysis is not None:
            columns = cached_analysis.get('convertible_columns', [])
            logger.info(f"Using cached column analysis: {len(columns)} columns for conversion")
        else:
            categorical_analysis = detect_categorical_numeric_columns(df, logger)
            columns = list(categorical_analysis.keys())
            
            # Cache the analysis
            try:
                cache.cache_column_analysis(
                    pd.Series([df_key]), 
                    'categorical_detection',
                    {'convertible_columns': columns, 'analysis': categorical_analysis}
                )
            except:
                pass
            
            logger.info(f"Auto-detected {len(columns)} columns for conversion: {columns}")
    
    if not columns:
        logger.info("No categorical columns found for conversion")
        return df.copy(), {}
    
    # Process each column with caching
    converted_df = df.copy()
    conversion_reports = {}
    cache_hits = 0
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame - skipping")
            continue
        
        # Check cache first
        cached_result = cache.get_conversion_result(df[col], col)
        if cached_result is not None:
            converted_series, report = cached_result
            cache_hits += 1
            logger.debug(f"Using cached conversion for column: {col}")
        else:
            logger.debug(f"Converting column: {col}")
            converted_series, report = safe_numeric_conversion(df[col], col, logger)
            
            # Cache the result
            cache.cache_conversion_result(df[col], col, (converted_series, report))
        
        converted_df[col] = converted_series
        conversion_reports[col] = report
    
    # Summary logging
    successful_conversions = sum(1 for report in conversion_reports.values() 
                               if report['conversion_successful'])
    
    cache_hit_rate = cache_hits / len(columns) if columns else 0
    logger.info(f"Batch conversion complete: {successful_conversions}/{len(columns)} columns converted successfully (cache hit rate: {cache_hit_rate:.2%})")
    
    return converted_df, conversion_reports


# Utility functions for safe operations
def ensure_numeric_for_calculation(series: Union[pd.Series, pd.Categorical], operation: str = "calculation",
                                 logger: Optional[logging.Logger] = None) -> pd.Series:
    """
    Ensure a series is numeric before mathematical operations with caching.
    
    Args:
        series: Series to ensure is numeric
        operation: Description of the operation for logging
        logger: Optional logger instance
        
    Returns:
        Numeric series ready for calculations
    """
    if logger is None:
        logger = setup_logging('numeric_ensurer')
    
    # Convert Categorical to Series if needed
    if isinstance(series, pd.Categorical):
        series = pd.Series(series)
    
    # Fast path: already numeric
    if pd.api.types.is_numeric_dtype(series):
        return series
    
    # Direct conversion without using lazy converter to avoid cache conflicts
    logger.debug(f"Converting series to numeric for {operation}")
    converted_series, _ = safe_numeric_conversion(series, f"temp_for_{operation}", logger)
    return converted_series


def safe_mean_calculation(series: Union[pd.Series, pd.Categorical], column_name: str = "unknown",
                         logger: Optional[logging.Logger] = None) -> float:
    """
    Safely calculate mean with categorical data handling.
    
    Args:
        series: Series to calculate mean for
        column_name: Name of the column for logging
        logger: Optional logger instance
        
    Returns:
        Mean value or NaN if calculation fails
    """
    if logger is None:
        logger = setup_logging('safe_calculator')
    
    try:
        # Convert Categorical to Series if needed
        if isinstance(series, pd.Categorical):
            series = pd.Series(series)
        
        # Ensure series is numeric
        numeric_series = ensure_numeric_for_calculation(series, "mean calculation", logger)
        
        if numeric_series.notna().any():
            result = numeric_series.mean()
            logger.debug(f"Mean calculated for '{column_name}': {result}")
            return result
        else:
            logger.warning(f"No valid numeric values found in '{column_name}' for mean calculation")
            return np.nan
            
    except Exception as e:
        # Use specialized error handler for categorical data errors
        error_message = handle_categorical_mean_error(e, column_name)
        logger.error(f"Error calculating mean for '{column_name}': {str(e)}")
        return np.nan


def safe_numerical_comparison(series: Union[pd.Series, pd.Categorical], operator: str, value: Union[int, float],
                            column_name: str = "unknown", logger: Optional[logging.Logger] = None) -> pd.Series:
    """
    Safely perform numerical comparisons with categorical data handling.
    
    Args:
        series: Series to compare
        operator: Comparison operator ('>=', '<=', '>', '<', '==', '!=')
        value: Value to compare against
        column_name: Name of the column for logging
        logger: Optional logger instance
        
    Returns:
        Boolean series with comparison results
    """
    if logger is None:
        logger = setup_logging('safe_comparator')
    
    try:
        # Convert Categorical to Series if needed
        if isinstance(series, pd.Categorical):
            original_index = range(len(series))
            series = pd.Series(series)
        else:
            original_index = series.index
        
        # Ensure series is numeric
        numeric_series = ensure_numeric_for_calculation(series, f"{operator} comparison", logger)
        
        # Perform comparison
        if operator == '>=':
            result = numeric_series >= value
        elif operator == '<=':
            result = numeric_series <= value
        elif operator == '>':
            result = numeric_series > value
        elif operator == '<':
            result = numeric_series < value
        elif operator == '==':
            result = numeric_series == value
        elif operator == '!=':
            result = numeric_series != value
        else:
            raise ValueError(f"Unsupported operator: {operator}")
        
        logger.debug(f"Comparison '{column_name}' {operator} {value} completed")
        return result
        
    except Exception as e:
        logger.error(f"Error in comparison '{column_name}' {operator} {value}: {str(e)}")
        # Return all False on error with proper index
        return pd.Series([False] * len(series), index=original_index)


# Additional safe mathematical operation wrappers
def safe_sum_calculation(series: Union[pd.Series, pd.Categorical], column_name: str = "unknown",
                        logger: Optional[logging.Logger] = None) -> float:
    """
    Safely calculate sum with categorical data handling.
    
    Args:
        series: Series to calculate sum for
        column_name: Name of the column for logging
        logger: Optional logger instance
        
    Returns:
        Sum value or NaN if calculation fails
    """
    if logger is None:
        logger = setup_logging('safe_calculator')
    
    try:
        # Convert Categorical to Series if needed
        if isinstance(series, pd.Categorical):
            series = pd.Series(series)
        
        # Ensure series is numeric
        numeric_series = ensure_numeric_for_calculation(series, "sum calculation", logger)
        
        if numeric_series.notna().any():
            result = numeric_series.sum()
            logger.debug(f"Sum calculated for '{column_name}': {result}")
            return result
        else:
            logger.warning(f"No valid numeric values found in '{column_name}' for sum calculation")
            return np.nan
            
    except Exception as e:
        logger.error(f"Error calculating sum for '{column_name}': {str(e)}")
        return np.nan


def safe_count_calculation(series: Union[pd.Series, pd.Categorical], column_name: str = "unknown",
                          logger: Optional[logging.Logger] = None) -> int:
    """
    Safely calculate count with categorical data handling.
    
    Args:
        series: Series to calculate count for
        column_name: Name of the column for logging
        logger: Optional logger instance
        
    Returns:
        Count of non-null values
    """
    if logger is None:
        logger = setup_logging('safe_calculator')
    
    try:
        # Convert Categorical to Series if needed
        if isinstance(series, pd.Categorical):
            series = pd.Series(series)
        
        result = series.count()
        logger.debug(f"Count calculated for '{column_name}': {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error calculating count for '{column_name}': {str(e)}")
        return 0


def safe_min_calculation(series: Union[pd.Series, pd.Categorical], column_name: str = "unknown",
                        logger: Optional[logging.Logger] = None) -> float:
    """
    Safely calculate minimum with categorical data handling.
    
    Args:
        series: Series to calculate minimum for
        column_name: Name of the column for logging
        logger: Optional logger instance
        
    Returns:
        Minimum value or NaN if calculation fails
    """
    if logger is None:
        logger = setup_logging('safe_calculator')
    
    try:
        # Convert Categorical to Series if needed
        if isinstance(series, pd.Categorical):
            series = pd.Series(series)
        
        # Ensure series is numeric
        numeric_series = ensure_numeric_for_calculation(series, "min calculation", logger)
        
        if numeric_series.notna().any():
            result = numeric_series.min()
            logger.debug(f"Minimum calculated for '{column_name}': {result}")
            return result
        else:
            logger.warning(f"No valid numeric values found in '{column_name}' for min calculation")
            return np.nan
            
    except Exception as e:
        logger.error(f"Error calculating minimum for '{column_name}': {str(e)}")
        return np.nan


def safe_max_calculation(series: Union[pd.Series, pd.Categorical], column_name: str = "unknown",
                        logger: Optional[logging.Logger] = None) -> float:
    """
    Safely calculate maximum with categorical data handling.
    
    Args:
        series: Series to calculate maximum for
        column_name: Name of the column for logging
        logger: Optional logger instance
        
    Returns:
        Maximum value or NaN if calculation fails
    """
    if logger is None:
        logger = setup_logging('safe_calculator')
    
    try:
        # Convert Categorical to Series if needed
        if isinstance(series, pd.Categorical):
            series = pd.Series(series)
        
        # Ensure series is numeric
        numeric_series = ensure_numeric_for_calculation(series, "max calculation", logger)
        
        if numeric_series.notna().any():
            result = numeric_series.max()
            logger.debug(f"Maximum calculated for '{column_name}': {result}")
            return result
        else:
            logger.warning(f"No valid numeric values found in '{column_name}' for max calculation")
            return np.nan
            
    except Exception as e:
        logger.error(f"Error calculating maximum for '{column_name}': {str(e)}")
        return np.nan


def safe_std_calculation(series: Union[pd.Series, pd.Categorical], column_name: str = "unknown",
                        logger: Optional[logging.Logger] = None) -> float:
    """
    Safely calculate standard deviation with categorical data handling.
    
    Args:
        series: Series to calculate standard deviation for
        column_name: Name of the column for logging
        logger: Optional logger instance
        
    Returns:
        Standard deviation or NaN if calculation fails
    """
    if logger is None:
        logger = setup_logging('safe_calculator')
    
    try:
        # Convert Categorical to Series if needed
        if isinstance(series, pd.Categorical):
            series = pd.Series(series)
        
        # Ensure series is numeric
        numeric_series = ensure_numeric_for_calculation(series, "std calculation", logger)
        
        if numeric_series.notna().any() and len(numeric_series.dropna()) > 1:
            result = numeric_series.std()
            logger.debug(f"Standard deviation calculated for '{column_name}': {result}")
            return result
        else:
            logger.warning(f"Insufficient valid numeric values in '{column_name}' for std calculation")
            return np.nan
            
    except Exception as e:
        logger.error(f"Error calculating standard deviation for '{column_name}': {str(e)}")
        return np.nan


def safe_median_calculation(series: Union[pd.Series, pd.Categorical], column_name: str = "unknown",
                           logger: Optional[logging.Logger] = None) -> float:
    """
    Safely calculate median with categorical data handling.
    
    Args:
        series: Series to calculate median for
        column_name: Name of the column for logging
        logger: Optional logger instance
        
    Returns:
        Median value or NaN if calculation fails
    """
    if logger is None:
        logger = setup_logging('safe_calculator')
    
    try:
        # Convert Categorical to Series if needed
        if isinstance(series, pd.Categorical):
            series = pd.Series(series)
        
        # Ensure series is numeric
        numeric_series = ensure_numeric_for_calculation(series, "median calculation", logger)
        
        if numeric_series.notna().any():
            result = numeric_series.median()
            logger.debug(f"Median calculated for '{column_name}': {result}")
            return result
        else:
            logger.warning(f"No valid numeric values found in '{column_name}' for median calculation")
            return np.nan
            
    except Exception as e:
        logger.error(f"Error calculating median for '{column_name}': {str(e)}")
        return np.nan


def safe_range_filter(series: Union[pd.Series, pd.Categorical], min_value: Optional[float] = None, 
                     max_value: Optional[float] = None, column_name: str = "unknown",
                     logger: Optional[logging.Logger] = None) -> pd.Series:
    """
    Safely apply range filter with categorical data handling.
    
    Args:
        series: Series to filter
        min_value: Minimum value (inclusive)
        max_value: Maximum value (inclusive)
        column_name: Name of the column for logging
        logger: Optional logger instance
        
    Returns:
        Boolean series indicating which values are within range
    """
    if logger is None:
        logger = setup_logging('safe_filter')
    
    try:
        # Convert Categorical to Series if needed
        if isinstance(series, pd.Categorical):
            original_index = range(len(series))
            series = pd.Series(series)
        else:
            original_index = series.index
        
        # Ensure series is numeric
        numeric_series = ensure_numeric_for_calculation(series, "range filter", logger)
        
        # Apply range filter
        result = pd.Series([True] * len(numeric_series), index=numeric_series.index)
        
        if min_value is not None:
            result = result & (numeric_series >= min_value)
        
        if max_value is not None:
            result = result & (numeric_series <= max_value)
        
        # Handle NaN values - they should be excluded from range
        result = result & numeric_series.notna()
        
        logger.debug(f"Range filter applied to '{column_name}': {result.sum()} values in range")
        return result
        
    except Exception as e:
        logger.error(f"Error applying range filter to '{column_name}': {str(e)}")
        # Return all False on error with proper index
        return pd.Series([False] * len(series), index=original_index)


def safe_range_filter(series: Union[pd.Series, pd.Categorical], min_value: Optional[float] = None, 
                     max_value: Optional[float] = None, column_name: str = "unknown",
                     logger: Optional[logging.Logger] = None) -> pd.Series:
    """
    Safely apply range filter with categorical data handling.
    
    Args:
        series: Series to filter
        min_value: Minimum value (inclusive)
        max_value: Maximum value (inclusive)
        column_name: Name of the column for logging
        logger: Optional logger instance
        
    Returns:
        Boolean series indicating which values are within range
    """
    if logger is None:
        logger = setup_logging('safe_filter')
    
    try:
        # Convert Categorical to Series if needed
        if isinstance(series, pd.Categorical):
            original_index = range(len(series))
            series = pd.Series(series)
        else:
            original_index = series.index
        
        # Ensure series is numeric
        numeric_series = ensure_numeric_for_calculation(series, "range filter", logger)
        
        # Apply range filter
        result = pd.Series([True] * len(numeric_series), index=numeric_series.index)
        
        if min_value is not None:
            result = result & (numeric_series >= min_value)
        
        if max_value is not None:
            result = result & (numeric_series <= max_value)
        
        # Handle NaN values - they should be excluded from range
        result = result & numeric_series.notna()
        
        logger.debug(f"Range filter applied to '{column_name}': {result.sum()} values in range")
        return result
        
    except Exception as e:
        logger.error(f"Error applying range filter to '{column_name}': {str(e)}")
        # Return all False on error
        return pd.Series([False] * len(series), index=original_index)


def safe_value_filter(series: Union[pd.Series, pd.Categorical], values: List[Any], 
                     column_name: str = "unknown", logger: Optional[logging.Logger] = None) -> pd.Series:
    """
    Safely filter series for specific values with categorical data handling.
    
    Args:
        series: Series to filter
        values: List of values to match
        column_name: Name of the column for logging
        logger: Optional logger instance
        
    Returns:
        Boolean series indicating which values match the filter
    """
    if logger is None:
        logger = setup_logging('safe_filter')
    
    try:
        # Convert Categorical to Series if needed
        if isinstance(series, pd.Categorical):
            original_index = range(len(series))
            series = pd.Series(series)
        else:
            original_index = series.index
        
        # Apply value filter
        result = series.isin(values)
        
        logger.debug(f"Value filter applied to '{column_name}': {result.sum()} values match")
        return result
        
    except Exception as e:
        logger.error(f"Error applying value filter to '{column_name}': {str(e)}")
        # Return all False on error
        return pd.Series([False] * len(series), index=original_index)


def safe_aggregation_wrapper(series: Union[pd.Series, pd.Categorical], operation: str, 
                            column_name: str = "unknown", logger: Optional[logging.Logger] = None) -> Any:
    """
    Generic wrapper for safe aggregation operations.
    
    Args:
        series: Series to aggregate
        operation: Operation name ('mean', 'sum', 'count', 'min', 'max', 'std', 'median')
        column_name: Name of the column for logging
        logger: Optional logger instance
        
    Returns:
        Aggregated value or appropriate default on error
    """
    if logger is None:
        logger = setup_logging('safe_aggregator')
    
    operation_map = {
        'mean': safe_mean_calculation,
        'sum': safe_sum_calculation,
        'count': safe_count_calculation,
        'min': safe_min_calculation,
        'max': safe_max_calculation,
        'std': safe_std_calculation,
        'median': safe_median_calculation
    }
    
    if operation not in operation_map:
        logger.error(f"Unsupported aggregation operation: {operation}")
        return np.nan if operation != 'count' else 0
    
    try:
        return operation_map[operation](series, column_name, logger)
    except Exception as e:
        logger.error(f"Error in aggregation wrapper for '{column_name}' operation '{operation}': {str(e)}")
        return np.nan if operation != 'count' else 0