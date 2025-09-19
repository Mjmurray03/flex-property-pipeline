"""
Enhanced filtering utilities with categorical data handling for the Interactive Filter Dashboard.
Addresses pandas categorical data type issues in filtering and metrics calculations.
"""

import logging
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Any, Optional, Union

from utils.logger import setup_logging
from utils.data_type_utils import (
    safe_mean_calculation,
    safe_numerical_comparison,
    safe_range_filter,
    ensure_numeric_for_calculation,
    convert_categorical_to_numeric
)


def safe_price_filter(df: pd.DataFrame, price_range: Tuple[float, float], 
                     price_column: str = 'Sold Price', logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Safely apply price range filter with categorical data handling.
    
    Args:
        df: DataFrame to filter
        price_range: Tuple of (min_price, max_price)
        price_column: Name of the price column
        logger: Optional logger instance
        
    Returns:
        Filtered DataFrame
    """
    if logger is None:
        logger = setup_logging('price_filter')
    
    if price_column not in df.columns:
        logger.warning(f"Price column '{price_column}' not found in DataFrame")
        return df
    
    try:
        logger.debug(f"Applying price filter: ${price_range[0]:,} - ${price_range[1]:,}")
        
        # Use safe range filter
        price_mask = safe_range_filter(
            df[price_column], 
            min_value=price_range[0], 
            max_value=price_range[1],
            column_name=price_column,
            logger=logger
        )
        
        filtered_df = df[price_mask]
        
        logger.info(f"Price filter applied: {len(filtered_df)}/{len(df)} properties match price range")
        return filtered_df
        
    except Exception as e:
        logger.error(f"Error in safe price filter: {str(e)}")
        st.warning(f"Error applying price filter: {str(e)}")
        return df


def safe_size_filter(df: pd.DataFrame, size_range: Tuple[float, float], 
                    size_column: str = 'Building SqFt', logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Safely apply building size range filter with categorical data handling.
    
    Args:
        df: DataFrame to filter
        size_range: Tuple of (min_size, max_size)
        size_column: Name of the size column
        logger: Optional logger instance
        
    Returns:
        Filtered DataFrame
    """
    if logger is None:
        logger = setup_logging('size_filter')
    
    if size_column not in df.columns:
        logger.warning(f"Size column '{size_column}' not found in DataFrame")
        return df
    
    try:
        logger.debug(f"Applying size filter: {size_range[0]:,} - {size_range[1]:,} sqft")
        
        # Use safe range filter
        size_mask = safe_range_filter(
            df[size_column], 
            min_value=size_range[0], 
            max_value=size_range[1],
            column_name=size_column,
            logger=logger
        )
        
        filtered_df = df[size_mask]
        
        logger.info(f"Size filter applied: {len(filtered_df)}/{len(df)} properties match size range")
        return filtered_df
        
    except Exception as e:
        logger.error(f"Error in safe size filter: {str(e)}")
        st.warning(f"Error applying size filter: {str(e)}")
        return df


def safe_year_filter(df: pd.DataFrame, year_range: Tuple[int, int], 
                    year_column: str = 'Year Built', logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Safely apply year built range filter with categorical data handling.
    
    Args:
        df: DataFrame to filter
        year_range: Tuple of (min_year, max_year)
        year_column: Name of the year column
        logger: Optional logger instance
        
    Returns:
        Filtered DataFrame
    """
    if logger is None:
        logger = setup_logging('year_filter')
    
    if year_column not in df.columns:
        logger.warning(f"Year column '{year_column}' not found in DataFrame")
        return df
    
    try:
        logger.debug(f"Applying year filter: {year_range[0]} - {year_range[1]}")
        
        # Use safe range filter
        year_mask = safe_range_filter(
            df[year_column], 
            min_value=year_range[0], 
            max_value=year_range[1],
            column_name=year_column,
            logger=logger
        )
        
        filtered_df = df[year_mask]
        
        logger.info(f"Year filter applied: {len(filtered_df)}/{len(df)} properties match year range")
        return filtered_df
        
    except Exception as e:
        logger.error(f"Error in safe year filter: {str(e)}")
        st.warning(f"Error applying year filter: {str(e)}")
        return df


def safe_lot_size_filter(df: pd.DataFrame, lot_range: Tuple[float, float], 
                        lot_column: str = 'Lot Size Acres', logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Safely apply lot size range filter with categorical data handling.
    
    Args:
        df: DataFrame to filter
        lot_range: Tuple of (min_lot_size, max_lot_size)
        lot_column: Name of the lot size column
        logger: Optional logger instance
        
    Returns:
        Filtered DataFrame
    """
    if logger is None:
        logger = setup_logging('lot_filter')
    
    if lot_column not in df.columns:
        logger.warning(f"Lot size column '{lot_column}' not found in DataFrame")
        return df
    
    try:
        logger.debug(f"Applying lot size filter: {lot_range[0]} - {lot_range[1]} acres")
        
        # Use safe range filter
        lot_mask = safe_range_filter(
            df[lot_column], 
            min_value=lot_range[0], 
            max_value=lot_range[1],
            column_name=lot_column,
            logger=logger
        )
        
        filtered_df = df[lot_mask]
        
        logger.info(f"Lot size filter applied: {len(filtered_df)}/{len(df)} properties match lot size range")
        return filtered_df
        
    except Exception as e:
        logger.error(f"Error in safe lot size filter: {str(e)}")
        st.warning(f"Error applying lot size filter: {str(e)}")
        return df


def safe_occupancy_filter(df: pd.DataFrame, occupancy_range: Tuple[float, float], 
                         occupancy_column: str = 'Occupancy', logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Safely apply occupancy range filter with categorical data handling.
    
    Args:
        df: DataFrame to filter
        occupancy_range: Tuple of (min_occupancy, max_occupancy)
        occupancy_column: Name of the occupancy column
        logger: Optional logger instance
        
    Returns:
        Filtered DataFrame
    """
    if logger is None:
        logger = setup_logging('occupancy_filter')
    
    if occupancy_column not in df.columns:
        logger.warning(f"Occupancy column '{occupancy_column}' not found in DataFrame")
        return df
    
    try:
        logger.debug(f"Applying occupancy filter: {occupancy_range[0]}% - {occupancy_range[1]}%")
        
        # Use safe range filter
        occupancy_mask = safe_range_filter(
            df[occupancy_column], 
            min_value=occupancy_range[0], 
            max_value=occupancy_range[1],
            column_name=occupancy_column,
            logger=logger
        )
        
        filtered_df = df[occupancy_mask]
        
        logger.info(f"Occupancy filter applied: {len(filtered_df)}/{len(df)} properties match occupancy range")
        return filtered_df
        
    except Exception as e:
        logger.error(f"Error in safe occupancy filter: {str(e)}")
        st.warning(f"Error applying occupancy filter: {str(e)}")
        return df


def apply_enhanced_filters(df: pd.DataFrame, filter_params: Dict[str, Any], 
                          logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Apply all filters using enhanced categorical-safe filtering functions.
    
    Args:
        df: DataFrame to filter
        filter_params: Dictionary of filter parameters
        logger: Optional logger instance
        
    Returns:
        Filtered DataFrame
    """
    if logger is None:
        logger = setup_logging('enhanced_filter')
    
    logger.info("Starting enhanced filtering with categorical data handling...")
    
    filtered_df = df.copy()
    original_count = len(filtered_df)
    
    try:
        # Apply size filter
        if 'size_range' in filter_params:
            filtered_df = safe_size_filter(filtered_df, filter_params['size_range'], logger=logger)
        
        # Apply lot size filter
        if 'lot_range' in filter_params:
            filtered_df = safe_lot_size_filter(filtered_df, filter_params['lot_range'], logger=logger)
        
        # Apply price filter
        if filter_params.get('use_price_filter') and 'price_range' in filter_params:
            filtered_df = safe_price_filter(filtered_df, filter_params['price_range'], logger=logger)
        
        # Apply year filter
        if filter_params.get('use_year_filter') and 'year_range' in filter_params:
            filtered_df = safe_year_filter(filtered_df, filter_params['year_range'], logger=logger)
        
        # Apply occupancy filter
        if filter_params.get('use_occupancy_filter') and 'occupancy_range' in filter_params:
            filtered_df = safe_occupancy_filter(filtered_df, filter_params['occupancy_range'], logger=logger)
        
        # Apply county filter (this doesn't need categorical handling)
        if 'selected_counties' in filter_params and filter_params['selected_counties']:
            if 'County' in filtered_df.columns:
                county_mask = filtered_df['County'].isin(filter_params['selected_counties'])
                filtered_df = filtered_df[county_mask]
                logger.debug(f"County filter applied: {len(filtered_df)} properties match selected counties")
        
        # Apply state filter (this doesn't need categorical handling)
        if 'selected_states' in filter_params and filter_params['selected_states']:
            if 'State' in filtered_df.columns:
                state_mask = filtered_df['State'].isin(filter_params['selected_states'])
                filtered_df = filtered_df[state_mask]
                logger.debug(f"State filter applied: {len(filtered_df)} properties match selected states")
        
        # Apply property type filter (this doesn't need categorical handling)
        if 'industrial_keywords' in filter_params and filter_params['industrial_keywords']:
            if 'Property Type' in filtered_df.columns:
                # Create pattern for industrial keywords
                pattern = '|'.join(filter_params['industrial_keywords'])
                type_mask = filtered_df['Property Type'].str.contains(pattern, case=False, na=False)
                filtered_df = filtered_df[type_mask]
                logger.debug(f"Property type filter applied: {len(filtered_df)} properties match industrial keywords")
        
        final_count = len(filtered_df)
        logger.info(f"Enhanced filtering complete: {final_count}/{original_count} properties remain")
        
        return filtered_df
        
    except Exception as e:
        logger.error(f"Error in enhanced filtering: {str(e)}")
        st.error(f"Error applying filters: {str(e)}")
        return df


def calculate_safe_metrics(df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Calculate metrics safely with categorical data handling.
    
    Args:
        df: DataFrame to calculate metrics for
        logger: Optional logger instance
        
    Returns:
        Dictionary of calculated metrics
    """
    if logger is None:
        logger = setup_logging('safe_metrics')
    
    metrics = {}
    
    try:
        # Basic counts
        metrics['total_properties'] = len(df)
        
        # Safe price metrics
        if 'Sold Price' in df.columns:
            avg_price = safe_mean_calculation(df['Sold Price'], 'Sold Price', logger)
            metrics['avg_price'] = avg_price if not pd.isna(avg_price) else None
        else:
            metrics['avg_price'] = None
        
        # Safe size metrics
        if 'Building SqFt' in df.columns:
            avg_size = safe_mean_calculation(df['Building SqFt'], 'Building SqFt', logger)
            metrics['avg_size'] = avg_size if not pd.isna(avg_size) else None
        else:
            metrics['avg_size'] = None
        
        # Safe lot size metrics
        if 'Lot Size Acres' in df.columns:
            avg_lot_size = safe_mean_calculation(df['Lot Size Acres'], 'Lot Size Acres', logger)
            metrics['avg_lot_size'] = avg_lot_size if not pd.isna(avg_lot_size) else None
        else:
            metrics['avg_lot_size'] = None
        
        # Safe occupancy metrics
        if 'Occupancy' in df.columns:
            avg_occupancy = safe_mean_calculation(df['Occupancy'], 'Occupancy', logger)
            metrics['avg_occupancy'] = avg_occupancy if not pd.isna(avg_occupancy) else None
        else:
            metrics['avg_occupancy'] = None
        
        # Geographic metrics (these don't need categorical handling)
        if 'City' in df.columns:
            metrics['unique_cities'] = df['City'].nunique()
        else:
            metrics['unique_cities'] = 0
        
        if 'County' in df.columns:
            metrics['unique_counties'] = df['County'].nunique()
        else:
            metrics['unique_counties'] = 0
        
        if 'State' in df.columns:
            metrics['unique_states'] = df['State'].nunique()
        else:
            metrics['unique_states'] = 0
        
        logger.debug(f"Safe metrics calculated for {metrics['total_properties']} properties")
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating safe metrics: {str(e)}")
        return {'total_properties': len(df), 'error': str(e)}


def preprocess_dataframe_for_filtering(df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Preprocess DataFrame to ensure all numeric columns are properly typed for filtering.
    
    Args:
        df: DataFrame to preprocess
        logger: Optional logger instance
        
    Returns:
        Preprocessed DataFrame with converted categorical columns
    """
    if logger is None:
        logger = setup_logging('filter_preprocessor')
    
    logger.info("Preprocessing DataFrame for enhanced filtering...")
    
    try:
        # Convert categorical columns that should be numeric
        processed_df, conversion_reports = convert_categorical_to_numeric(df, logger=logger)
        
        if conversion_reports:
            logger.info(f"Converted {len(conversion_reports)} categorical columns to numeric")
            
            # Show user feedback about conversions
            for col, report in conversion_reports.items():
                if report['conversion_successful']:
                    if report['values_failed'] > 0:
                        st.info(f"Column '{col}': Converted to numeric ({report['values_failed']} invalid values became null)")
                    else:
                        st.success(f"Column '{col}': Successfully converted to numeric")
                else:
                    st.warning(f"Column '{col}': Could not convert to numeric")
        
        return processed_df
        
    except Exception as e:
        logger.error(f"Error preprocessing DataFrame for filtering: {str(e)}")
        st.warning(f"Error preprocessing data: {str(e)}")
        return df


def validate_filter_data_types(df: pd.DataFrame, filter_params: Dict[str, Any], 
                              logger: Optional[logging.Logger] = None) -> Dict[str, str]:
    """
    Validate that columns used in filtering have appropriate data types.
    
    Args:
        df: DataFrame to validate
        filter_params: Filter parameters to check
        logger: Optional logger instance
        
    Returns:
        Dictionary of validation warnings
    """
    if logger is None:
        logger = setup_logging('filter_validator')
    
    warnings = {}
    
    try:
        # Check price column
        if filter_params.get('use_price_filter') and 'Sold Price' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['Sold Price']):
                warnings['price'] = f"Price column has non-numeric type: {df['Sold Price'].dtype}"
        
        # Check size column
        if 'Building SqFt' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['Building SqFt']):
                warnings['size'] = f"Size column has non-numeric type: {df['Building SqFt'].dtype}"
        
        # Check year column
        if filter_params.get('use_year_filter') and 'Year Built' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['Year Built']):
                warnings['year'] = f"Year column has non-numeric type: {df['Year Built'].dtype}"
        
        # Check lot size column
        if 'Lot Size Acres' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['Lot Size Acres']):
                warnings['lot_size'] = f"Lot size column has non-numeric type: {df['Lot Size Acres'].dtype}"
        
        # Check occupancy column
        if filter_params.get('use_occupancy_filter') and 'Occupancy' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['Occupancy']):
                warnings['occupancy'] = f"Occupancy column has non-numeric type: {df['Occupancy'].dtype}"
        
        if warnings:
            logger.warning(f"Data type validation found {len(warnings)} issues")
            for col, warning in warnings.items():
                logger.warning(f"  {col}: {warning}")
        else:
            logger.debug("All filter columns have appropriate data types")
        
        return warnings
        
    except Exception as e:
        logger.error(f"Error validating filter data types: {str(e)}")
        return {'validation_error': str(e)}