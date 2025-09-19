"""
Data loading and preprocessing utilities for Flex Property Classifier
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd

from utils.logger import setup_logging
from utils.data_type_utils import (
    detect_categorical_numeric_columns,
    convert_categorical_to_numeric,
    safe_numeric_conversion
)


def load_property_data(file_path: str, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Load property data from Excel file with comprehensive error handling
    
    Args:
        file_path: Path to Excel file
        logger: Optional logger instance
        
    Returns:
        DataFrame containing property data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
        Exception: For other loading errors
    """
    if logger is None:
        logger = setup_logging('flex_data_loader')
    
    try:
        logger.info(f"Loading property data from: {file_path}")
        
        # Validate file path
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path_obj.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        # Check file extension
        if file_path_obj.suffix.lower() not in ['.xlsx', '.xls']:
            raise ValueError(f"File must be Excel format (.xlsx or .xls): {file_path}")
        
        # Load Excel file
        df = pd.read_excel(file_path)
        
        # Basic validation
        if df.empty:
            raise ValueError("Excel file contains no data")
        
        if len(df.columns) == 0:
            raise ValueError("Excel file contains no columns")
        
        logger.info(f"Successfully loaded {len(df):,} properties with {len(df.columns)} columns")
        
        # Log column names for reference
        logger.debug(f"Columns: {', '.join(df.columns)}")
        
        return df
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        raise
        
    except PermissionError as e:
        logger.error(f"Permission denied accessing file: {file_path}. "
                    f"Check file permissions or close if open in another application.")
        raise Exception(f"Permission denied: {file_path}") from e
        
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse Excel file: {file_path}. "
                    f"File may be corrupted or in unsupported format.")
        raise Exception(f"Excel parsing error: {str(e)}") from e
        
    except Exception as e:
        logger.error(f"Unexpected error loading file: {file_path} - {str(e)}")
        raise Exception(f"Failed to load Excel file: {str(e)}") from e


def load_property_data_with_conversion(file_path: str, auto_convert_categorical: bool = True,
                                     logger: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Enhanced property data loading with automatic categorical data type conversion.
    
    Args:
        file_path: Path to Excel file
        auto_convert_categorical: Whether to automatically convert categorical columns to numeric
        logger: Optional logger instance
        
    Returns:
        Tuple of (DataFrame, conversion_report)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
        Exception: For other loading errors
    """
    if logger is None:
        logger = setup_logging('enhanced_data_loader')
    
    # Load the basic data first
    logger.info(f"Loading property data with enhanced categorical conversion from: {file_path}")
    df = load_property_data(file_path, logger)
    
    # Initialize conversion report
    conversion_report = {
        'file_path': file_path,
        'original_shape': df.shape,
        'categorical_columns_detected': 0,
        'categorical_columns_converted': 0,
        'conversion_successful': True,
        'column_reports': {},
        'data_quality_summary': {},
        'warnings': [],
        'errors': []
    }
    
    try:
        if auto_convert_categorical:
            logger.info("Starting automatic categorical data type conversion...")
            
            # Detect categorical columns that should be numeric
            categorical_analysis = detect_categorical_numeric_columns(df, logger)
            conversion_report['categorical_columns_detected'] = len(categorical_analysis)
            
            if categorical_analysis:
                logger.info(f"Found {len(categorical_analysis)} categorical columns suitable for conversion")
                
                # Convert categorical columns to numeric
                converted_df, column_reports = convert_categorical_to_numeric(df, logger=logger)
                
                # Update conversion report
                conversion_report['categorical_columns_converted'] = len(column_reports)
                conversion_report['column_reports'] = column_reports
                
                # Check for any conversion failures
                failed_conversions = [col for col, report in column_reports.items() 
                                    if not report['conversion_successful']]
                
                if failed_conversions:
                    warning_msg = f"Some categorical conversions failed: {failed_conversions}"
                    conversion_report['warnings'].append(warning_msg)
                    logger.warning(warning_msg)
                
                # Update DataFrame
                df = converted_df
                
                logger.info(f"Categorical conversion complete: {len(column_reports)} columns processed")
            else:
                logger.info("No categorical columns requiring conversion found")
        
        # Generate data quality summary
        conversion_report['data_quality_summary'] = _generate_data_quality_summary(df, logger)
        conversion_report['final_shape'] = df.shape
        
        logger.info(f"Enhanced data loading complete: {df.shape[0]:,} rows, {df.shape[1]} columns")
        
        return df, conversion_report
        
    except Exception as e:
        error_msg = f"Error during enhanced data loading: {str(e)}"
        conversion_report['errors'].append(error_msg)
        conversion_report['conversion_successful'] = False
        logger.error(error_msg)
        
        # Return original DataFrame with error report
        return df, conversion_report


def _generate_data_quality_summary(df: pd.DataFrame, logger: logging.Logger) -> Dict[str, Any]:
    """
    Generate a comprehensive data quality summary for the loaded DataFrame.
    
    Args:
        df: DataFrame to analyze
        logger: Logger instance
        
    Returns:
        Data quality summary dictionary
    """
    try:
        logger.debug("Generating data quality summary...")
        
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': 0,
            'categorical_columns': 0,
            'text_columns': 0,
            'datetime_columns': 0,
            'columns_with_nulls': 0,
            'overall_completeness': 0.0,
            'column_completeness': {},
            'data_types': {},
            'memory_usage_mb': 0.0
        }
        
        # Analyze each column
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isna().sum()
            completeness = ((len(df) - null_count) / len(df)) * 100 if len(df) > 0 else 0
            
            # Categorize data types
            if pd.api.types.is_numeric_dtype(dtype):
                summary['numeric_columns'] += 1
            elif dtype.name == 'category':
                summary['categorical_columns'] += 1
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                summary['datetime_columns'] += 1
            else:
                summary['text_columns'] += 1
            
            # Track columns with nulls
            if null_count > 0:
                summary['columns_with_nulls'] += 1
            
            # Store column-specific info
            summary['column_completeness'][col] = completeness
            summary['data_types'][col] = str(dtype)
        
        # Calculate overall completeness
        total_cells = len(df) * len(df.columns)
        if total_cells > 0:
            non_null_cells = df.count().sum()
            summary['overall_completeness'] = (non_null_cells / total_cells) * 100
        
        # Calculate memory usage
        summary['memory_usage_mb'] = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        logger.debug(f"Data quality summary: {summary['overall_completeness']:.1f}% complete, "
                    f"{summary['numeric_columns']} numeric columns, "
                    f"{summary['categorical_columns']} categorical columns")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating data quality summary: {str(e)}")
        return {'error': str(e)}


def load_and_validate_property_data(file_path: str, required_columns: Optional[List[str]] = None,
                                   auto_convert_categorical: bool = True,
                                   logger: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Complete data loading pipeline with validation and categorical conversion.
    
    Args:
        file_path: Path to Excel file
        required_columns: List of required column names (optional)
        auto_convert_categorical: Whether to automatically convert categorical columns
        logger: Optional logger instance
        
    Returns:
        Tuple of (validated_dataframe, pipeline_report)
    """
    if logger is None:
        logger = setup_logging('complete_data_pipeline')
    
    pipeline_report = {
        'loading_successful': False,
        'validation_successful': False,
        'conversion_report': {},
        'validation_report': {},
        'pipeline_warnings': [],
        'pipeline_errors': []
    }
    
    try:
        logger.info("Starting complete data loading and validation pipeline...")
        
        # Step 1: Load data with categorical conversion
        df, conversion_report = load_property_data_with_conversion(
            file_path, auto_convert_categorical, logger
        )
        pipeline_report['loading_successful'] = True
        pipeline_report['conversion_report'] = conversion_report
        
        # Step 2: Validate required columns if specified
        if required_columns:
            logger.info("Validating required columns...")
            missing_columns = []
            for col in required_columns:
                if col not in df.columns:
                    missing_columns.append(col)
            
            if missing_columns:
                warning_msg = f"Missing required columns: {missing_columns}"
                pipeline_report['pipeline_warnings'].append(warning_msg)
                logger.warning(warning_msg)
            else:
                logger.info("All required columns found")
            
            pipeline_report['validation_report'] = {
                'required_columns': required_columns,
                'missing_columns': missing_columns,
                'validation_passed': len(missing_columns) == 0
            }
            pipeline_report['validation_successful'] = len(missing_columns) == 0
        else:
            pipeline_report['validation_successful'] = True
        
        # Step 3: Final preprocessing
        logger.info("Applying final preprocessing...")
        df = preprocess_property_data(df, logger)
        
        logger.info("Complete data loading pipeline finished successfully")
        return df, pipeline_report
        
    except Exception as e:
        error_msg = f"Pipeline error: {str(e)}"
        pipeline_report['pipeline_errors'].append(error_msg)
        logger.error(error_msg)
        raise Exception(error_msg) from e


def validate_required_columns(df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> List[str]:
    """
    Validate presence of required columns for flex analysis
    
    Args:
        df: DataFrame to validate
        logger: Optional logger instance
        
    Returns:
        List of missing required column categories
    """
    if logger is None:
        logger = setup_logging('column_validator')
    
    try:
        logger.info("Validating required columns for flex analysis...")
        
        # Define required column categories and their possible names
        required_columns = {
            'property_type': ['property type', 'type', 'property_type', 'prop_type'],
            'building_size': ['building sqft', 'building_sqft', 'sqft', 'square_feet', 'building_sf'],
            'lot_size': ['lot size acres', 'lot_size_acres', 'acres', 'lot_acres', 'land_acres']
        }
        
        missing_categories = []
        found_columns = {}
        
        for category, search_terms in required_columns.items():
            found = False
            for term in search_terms:
                for col in df.columns:
                    if term.lower() in col.lower():
                        found_columns[category] = col
                        found = True
                        break
                if found:
                    break
            
            if not found:
                missing_categories.append(category)
        
        # Log results
        if missing_categories:
            logger.warning(f"Missing required columns: {', '.join(missing_categories)}")
            logger.info("Analysis will proceed with available data, but results may be limited")
        else:
            logger.info("✅ All required columns found")
        
        for category, col_name in found_columns.items():
            logger.info(f"  {category}: '{col_name}'")
        
        return missing_categories
        
    except Exception as e:
        logger.error(f"Error validating columns: {str(e)}")
        return []


def normalize_property_types(df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Normalize property type values for consistent analysis
    
    Args:
        df: DataFrame to normalize
        logger: Optional logger instance
        
    Returns:
        DataFrame with normalized property types
    """
    if logger is None:
        logger = setup_logging('property_normalizer')
    
    try:
        logger.info("Normalizing property types...")
        
        # Find property type column
        type_col = None
        for col in df.columns:
            if any(term in col.lower() for term in ['property type', 'type', 'property_type']):
                type_col = col
                break
        
        if type_col is None:
            logger.warning("No property type column found - skipping normalization")
            return df
        
        # Create normalized copy
        df_normalized = df.copy()
        
        # Normalize property types
        original_types = df_normalized[type_col].value_counts()
        logger.info(f"Original property types: {len(original_types)} unique values")
        
        # Convert to string and lowercase for processing
        normalized_values = df_normalized[type_col].astype(str).str.lower().str.strip()
        
        # Define normalization mappings
        type_mappings = {
            'industrial': ['industrial', 'ind', 'manufacturing', 'mfg'],
            'warehouse': ['warehouse', 'wh', 'storage', 'distribution'],
            'flex': ['flex', 'flex space', 'flexible'],
            'office': ['office', 'off', 'commercial office'],
            'retail': ['retail', 'store', 'shop'],
            'mixed_use': ['mixed', 'mixed use', 'multi-use']
        }
        
        # Apply normalization
        for standard_type, variants in type_mappings.items():
            for variant in variants:
                mask = normalized_values.str.contains(variant, na=False, regex=False)
                df_normalized.loc[mask, type_col] = standard_type
        
        # Log normalization results
        normalized_types = df_normalized[type_col].value_counts()
        logger.info(f"Normalized property types: {len(normalized_types)} unique values")
        
        # Show mapping results
        for norm_type, count in normalized_types.head(10).items():
            logger.debug(f"  {norm_type}: {count} properties")
        
        return df_normalized
        
    except Exception as e:
        logger.error(f"Error normalizing property types: {str(e)}")
        return df


def preprocess_property_data(df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for property data
    
    Args:
        df: Raw property DataFrame
        logger: Optional logger instance
        
    Returns:
        Preprocessed DataFrame ready for flex analysis
    """
    if logger is None:
        logger = setup_logging('data_preprocessor')
    
    try:
        logger.info("Starting comprehensive data preprocessing...")
        
        # Start with copy of original data
        processed_df = df.copy()
        
        # Step 1: Basic data cleaning
        logger.info("Step 1: Basic data cleaning...")
        
        # Remove completely empty rows
        initial_rows = len(processed_df)
        processed_df = processed_df.dropna(how='all')
        empty_rows_removed = initial_rows - len(processed_df)
        
        if empty_rows_removed > 0:
            logger.info(f"Removed {empty_rows_removed} completely empty rows")
        
        # Step 2: Column name standardization
        logger.info("Step 2: Standardizing column names...")
        
        # Create mapping for common column variations
        column_mappings = {
            'property_type': ['property type', 'type', 'prop type', 'property_type'],
            'building_sqft': ['building sqft', 'building_sqft', 'sqft', 'square feet', 'building_sf'],
            'lot_size_acres': ['lot size acres', 'lot_size_acres', 'acres', 'lot acres', 'land_acres'],
            'year_built': ['year built', 'year_built', 'built year', 'construction_year'],
            'property_name': ['property name', 'property_name', 'name', 'building_name'],
            'address': ['address', 'street_address', 'location'],
            'city': ['city', 'municipality'],
            'state': ['state', 'st'],
            'county': ['county'],
            'owner_name': ['owner name', 'owner_name', 'owner'],
            'sale_date': ['sale date', 'sale_date', 'sold_date'],
            'sold_price': ['sold price', 'sold_price', 'sale_price', 'price'],
            'occupancy': ['occupancy', 'occupancy_rate', 'occupied_percent']
        }
        
        # Apply column mappings
        for standard_name, variations in column_mappings.items():
            for variation in variations:
                matching_cols = [col for col in processed_df.columns 
                               if variation.lower() in col.lower()]
                if matching_cols:
                    # Use first match and rename it
                    old_name = matching_cols[0]
                    if old_name != standard_name:
                        processed_df = processed_df.rename(columns={old_name: standard_name})
                        logger.debug(f"Renamed '{old_name}' to '{standard_name}'")
                    break
        
        # Step 3: Data type conversion and validation
        logger.info("Step 3: Converting data types...")
        
        # Numeric columns that should be converted
        numeric_columns = ['building_sqft', 'lot_size_acres', 'year_built', 'sold_price', 'occupancy']
        
        for col in numeric_columns:
            if col in processed_df.columns:
                original_dtype = processed_df[col].dtype
                
                # Convert to numeric, coercing errors to NaN
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                
                # Log conversion results
                null_count = processed_df[col].isna().sum()
                total_count = len(processed_df)
                
                if null_count > 0:
                    logger.warning(f"Column '{col}': {null_count}/{total_count} "
                                 f"({(null_count/total_count)*100:.1f}%) values could not be converted to numeric")
                else:
                    logger.debug(f"Column '{col}': Successfully converted from {original_dtype} to numeric")
        
        # Step 4: Property type normalization
        logger.info("Step 4: Normalizing property types...")
        processed_df = normalize_property_types(processed_df, logger)
        
        # Step 5: Data validation and quality checks
        logger.info("Step 5: Final data validation...")
        
        # Check for reasonable value ranges
        validation_checks = {
            'building_sqft': (100, 10000000),  # 100 sqft to 10M sqft
            'lot_size_acres': (0.01, 1000),   # 0.01 to 1000 acres
            'year_built': (1800, 2030),       # 1800 to 2030
            'occupancy': (0, 100)             # 0% to 100%
        }
        
        for col, (min_val, max_val) in validation_checks.items():
            if col in processed_df.columns:
                col_data = processed_df[col].dropna()
                if len(col_data) > 0:
                    outliers = ((col_data < min_val) | (col_data > max_val)).sum()
                    if outliers > 0:
                        logger.warning(f"Column '{col}': {outliers} values outside reasonable range "
                                     f"({min_val}-{max_val})")
        
        # Final summary
        final_rows = len(processed_df)
        final_cols = len(processed_df.columns)
        
        logger.info(f"✅ Preprocessing complete: {final_rows:,} properties, {final_cols} columns")
        logger.info(f"Data quality: {((processed_df.count().sum() / (final_rows * final_cols)) * 100):.1f}% complete")
        
        return processed_df
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise Exception(f"Data preprocessing failed: {str(e)}") from e


def create_sample_dataset(output_path: str = "data/sample_flex_properties.xlsx", 
                         logger: Optional[logging.Logger] = None) -> str:
    """
    Create a sample dataset for testing the flex classifier
    
    Args:
        output_path: Path where to save the sample file
        logger: Optional logger instance
        
    Returns:
        Path to created sample file
    """
    if logger is None:
        logger = setup_logging('sample_creator')
    
    try:
        logger.info("Creating sample dataset for testing...")
        
        # Create sample data
        sample_data = {
            'Property Name': [
                'Industrial Park A', 'Flex Center B', 'Warehouse Complex C',
                'Manufacturing Hub D', 'Distribution Center E', 'Office Building F',
                'Retail Plaza G', 'Light Industrial H', 'Storage Facility I', 'Logistics Center J'
            ],
            'Property Type': [
                'Industrial', 'Flex', 'Warehouse', 'Manufacturing', 'Distribution',
                'Office', 'Retail', 'Light Industrial', 'Storage', 'Logistics'
            ],
            'Building SqFt': [
                45000, 35000, 85000, 120000, 75000, 25000, 15000, 40000, 60000, 95000
            ],
            'Lot Size Acres': [
                3.2, 2.8, 8.5, 12.0, 6.5, 1.2, 0.8, 3.5, 4.2, 7.8
            ],
            'Year Built': [
                1995, 2005, 1988, 1975, 1992, 2010, 2015, 1998, 1985, 2000
            ],
            'Address': [
                '123 Industrial Way', '456 Flex Blvd', '789 Warehouse St',
                '321 Manufacturing Dr', '654 Distribution Ave', '987 Office Pkwy',
                '147 Retail Rd', '258 Light Industrial Ln', '369 Storage Ct', '741 Logistics Loop'
            ],
            'City': [
                'Industrial City', 'Flex Town', 'Warehouse Village', 'Manufacturing Heights',
                'Distribution Center', 'Office Park', 'Retail District', 'Light Industrial Zone',
                'Storage City', 'Logistics Hub'
            ],
            'State': ['FL'] * 10,
            'County': ['Palm Beach'] * 10,
            'Sold Price': [
                2500000, 1800000, 4200000, 6000000, 3500000, 1200000, 800000, 2200000, 2800000, 4500000
            ],
            'Owner Name': [
                'Industrial Holdings LLC', 'Flex Properties Inc', 'Warehouse Investments',
                'Manufacturing Corp', 'Distribution Partners', 'Office Realty',
                'Retail Ventures', 'Light Industrial Fund', 'Storage Solutions', 'Logistics Group'
            ],
            'Occupancy': [
                85, 92, 78, 100, 88, 95, 100, 82, 75, 90
            ]
        }
        
        # Create DataFrame
        sample_df = pd.DataFrame(sample_data)
        
        # Ensure output directory exists
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to Excel
        sample_df.to_excel(output_path, index=False)
        
        logger.info(f"✅ Sample dataset created: {output_path}")
        logger.info(f"Sample contains {len(sample_df)} properties with various types and sizes")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error creating sample dataset: {str(e)}")
        raise Exception(f"Failed to create sample dataset: {str(e)}") from e


# Test functions
if __name__ == "__main__":
    # Test the utilities
    logger = setup_logging('flex_data_utils_test', level='DEBUG')
    
    try:
        # Create sample dataset
        sample_file = create_sample_dataset(logger=logger)
        
        # Test loading
        df = load_property_data(sample_file, logger=logger)
        
        # Test validation
        missing_cols = validate_required_columns(df, logger=logger)
        
        # Test preprocessing
        processed_df = preprocess_property_data(df, logger=logger)
        
        logger.info("✅ All utility functions tested successfully")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")