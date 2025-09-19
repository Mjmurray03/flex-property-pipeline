"""
Private Property Data Analyzer
Loads and analyzes private property data from Excel files to identify flex industrial properties
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
import pandas as pd
from datetime import datetime

from utils.logger import setup_logging


class PrivatePropertyAnalyzer:
    """
    Analyzer for private property data from Excel files
    
    Provides comprehensive analysis of property datasets including:
    - Data structure analysis and quality assessment
    - Industrial property identification and categorization
    - Data completeness metrics for key fields
    - Sample property display and reporting
    """
    
    def __init__(self, file_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the Private Property Analyzer
        
        Args:
            file_path: Path to the Excel file containing property data
            logger: Optional logger instance (will create one if not provided)
        
        Raises:
            ValueError: If file_path is invalid or file doesn't exist
            FileNotFoundError: If the specified file cannot be found
        """
        # Validate and store file path
        self.file_path = self._validate_file_path(file_path)
        
        # Set up logging
        if logger is None:
            self.logger = setup_logging(
                name='private_property_analyzer',
                level='INFO'
            )
        else:
            self.logger = logger
        
        # Initialize data storage
        self.data: Optional[pd.DataFrame] = None
        self.analysis_results: Dict[str, Any] = {}
        
        # Industrial property keywords for identification
        self.industrial_keywords = [
            'industrial', 'warehouse', 'distribution', 'flex', 
            'manufacturing', 'storage', 'logistics'
        ]
        
        # Key fields for data completeness analysis
        self.key_fields = [
            'Building SqFt', 'Property Type', 'Sale Date', 'Sold Price',
            'Year Built', 'Lot Size Acres', 'Zoning Code', 'County'
        ]
        
        self.logger.info(f"PrivatePropertyAnalyzer initialized for file: {self.file_path}")
    
    def _validate_file_path(self, file_path: str) -> Path:
        """
        Validate the provided file path
        
        Args:
            file_path: Path to validate
            
        Returns:
            Path object for the validated file
            
        Raises:
            ValueError: If file_path is empty or invalid
            FileNotFoundError: If file doesn't exist
        """
        if not file_path or not isinstance(file_path, str):
            raise ValueError("File path must be a non-empty string")
        
        path_obj = Path(file_path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path_obj.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        # Check if it's an Excel file
        if path_obj.suffix.lower() not in ['.xlsx', '.xls']:
            raise ValueError(f"File must be an Excel file (.xlsx or .xls): {file_path}")
        
        return path_obj
    
    def load_data(self) -> pd.DataFrame:
        """
        Load property data from Excel file
        
        Returns:
            DataFrame containing the loaded property data
            
        Raises:
            Exception: If file loading fails with detailed error information
        """
        try:
            self.logger.info(f"Loading data from: {self.file_path}")
            
            # Load Excel file with error handling
            self.data = pd.read_excel(self.file_path)
            
            # Log basic data information
            total_properties = len(self.data)
            total_columns = len(self.data.columns)
            
            self.logger.info(f"Successfully loaded {total_properties:,} properties with {total_columns} columns")
            
            # Log data types and non-null counts
            self.logger.info("Data structure overview:")
            for col in self.data.columns:
                dtype = str(self.data[col].dtype)
                non_null_count = self.data[col].count()
                null_percentage = ((len(self.data) - non_null_count) / len(self.data)) * 100
                
                self.logger.info(f"  {col}: {dtype} - {non_null_count:,} non-null ({null_percentage:.1f}% missing)")
            
            return self.data
            
        except FileNotFoundError as e:
            error_msg = f"Excel file not found: {self.file_path}. Please check the file path and ensure the file exists."
            self.logger.error(error_msg)
            raise Exception(error_msg) from e
            
        except PermissionError as e:
            error_msg = f"Permission denied accessing file: {self.file_path}. Please check file permissions or close the file if it's open in another application."
            self.logger.error(error_msg)
            raise Exception(error_msg) from e
            
        except pd.errors.ParserError as e:
            error_msg = f"Failed to parse Excel file: {self.file_path}. The file may be corrupted or in an unsupported format. Error: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg) from e
            
        except Exception as e:
            error_msg = f"Unexpected error loading Excel file: {self.file_path}. Error: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg) from e
    
    def analyze_property_types(self) -> List[str]:
        """
        Analyze property types and identify industrial properties
        
        Returns:
            List of industrial property types found in the dataset
            
        Raises:
            RuntimeError: If data hasn't been loaded yet
        """
        if self.data is None:
            raise RuntimeError("Data must be loaded before analysis. Call load_data() first.")
        
        try:
            self.logger.info("Analyzing property types...")
            
            # Find property type column (case-insensitive)
            property_type_col = None
            for col in self.data.columns:
                if 'property type' in col.lower() or 'type' in col.lower():
                    property_type_col = col
                    break
            
            if property_type_col is None:
                self.logger.warning("No 'Property Type' column found. Skipping property type analysis.")
                return []
            
            # Get unique property types and their counts
            property_types = self.data[property_type_col].value_counts()
            
            self.logger.info(f"Found {len(property_types)} unique property types:")
            for prop_type, count in property_types.items():
                self.logger.info(f"  {prop_type}: {count:,} properties")
            
            # Identify industrial properties
            industrial_types = []
            for prop_type in property_types.index:
                if pd.isna(prop_type):
                    continue
                    
                prop_type_lower = str(prop_type).lower()
                for keyword in self.industrial_keywords:
                    if keyword in prop_type_lower:
                        industrial_types.append(prop_type)
                        break
            
            if industrial_types:
                self.logger.info(f"Industrial property types identified: {len(industrial_types)}")
                total_industrial = sum(property_types[itype] for itype in industrial_types)
                self.logger.info(f"Total industrial properties: {total_industrial:,}")
                
                for itype in industrial_types:
                    count = property_types[itype]
                    self.logger.info(f"  {itype}: {count:,} properties")
            else:
                self.logger.info("No industrial properties detected based on keywords")
            
            return industrial_types
            
        except Exception as e:
            error_msg = f"Error analyzing property types: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg) from e
    
    def check_data_completeness(self) -> Dict[str, float]:
        """
        Assess data completeness for key fields
        
        Returns:
            Dictionary mapping field names to completeness percentages
            
        Raises:
            RuntimeError: If data hasn't been loaded yet
        """
        if self.data is None:
            raise RuntimeError("Data must be loaded before analysis. Call load_data() first.")
        
        try:
            self.logger.info("Checking data completeness for key fields...")
            
            completeness = {}
            total_properties = len(self.data)
            
            for field in self.key_fields:
                # Find matching column (case-insensitive, flexible matching)
                matching_col = None
                for col in self.data.columns:
                    if field.lower() in col.lower() or col.lower() in field.lower():
                        matching_col = col
                        break
                
                if matching_col is None:
                    self.logger.warning(f"Field '{field}' not found in dataset. Skipping.")
                    continue
                
                # Calculate completeness
                non_null_count = self.data[matching_col].count()
                completeness_pct = (non_null_count / total_properties) * 100
                completeness[field] = completeness_pct
                
                self.logger.info(f"  {field} ({matching_col}): {non_null_count:,}/{total_properties:,} ({completeness_pct:.1f}% complete)")
            
            if completeness:
                avg_completeness = sum(completeness.values()) / len(completeness)
                self.logger.info(f"Average data completeness: {avg_completeness:.1f}%")
            else:
                self.logger.warning("No key fields found for completeness analysis")
            
            return completeness
            
        except Exception as e:
            error_msg = f"Error checking data completeness: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg) from e
    
    def get_industrial_sample(self, limit: int = 10) -> pd.DataFrame:
        """
        Get sample of industrial properties with key details
        
        Args:
            limit: Maximum number of sample properties to return
            
        Returns:
            DataFrame containing sample industrial properties
            
        Raises:
            RuntimeError: If data hasn't been loaded yet
        """
        if self.data is None:
            raise RuntimeError("Data must be loaded before analysis. Call load_data() first.")
        
        try:
            self.logger.info(f"Retrieving sample of industrial properties (limit: {limit})...")
            
            # First identify industrial properties
            industrial_types = self.analyze_property_types()
            
            if not industrial_types:
                self.logger.info("No industrial properties found for sample display")
                return pd.DataFrame()
            
            # Find property type column
            property_type_col = None
            for col in self.data.columns:
                if 'property type' in col.lower() or 'type' in col.lower():
                    property_type_col = col
                    break
            
            if property_type_col is None:
                self.logger.warning("Cannot filter industrial properties - no property type column found")
                return pd.DataFrame()
            
            # Filter for industrial properties
            industrial_mask = self.data[property_type_col].isin(industrial_types)
            industrial_properties = self.data[industrial_mask]
            
            if len(industrial_properties) == 0:
                self.logger.info("No industrial properties found after filtering")
                return pd.DataFrame()
            
            # Select relevant columns for display
            display_columns = []
            column_mapping = {
                'property name': 'Property Name',
                'property type': 'Property Type', 
                'building sqft': 'Building SqFt',
                'city': 'City',
                'state': 'State'
            }
            
            for search_term, display_name in column_mapping.items():
                matching_col = None
                for col in self.data.columns:
                    if search_term in col.lower():
                        matching_col = col
                        break
                
                if matching_col:
                    display_columns.append(matching_col)
            
            # Get sample (first N properties)
            sample = industrial_properties.head(limit)
            
            if display_columns:
                sample_display = sample[display_columns]
            else:
                # Fallback to all columns if specific ones not found
                sample_display = sample
            
            self.logger.info(f"Retrieved {len(sample_display)} industrial property samples")
            
            # Log sample details
            for idx, (_, row) in enumerate(sample_display.iterrows(), 1):
                self.logger.info(f"  Sample {idx}: {dict(row)}")
            
            return sample_display
            
        except Exception as e:
            error_msg = f"Error retrieving industrial property sample: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg) from e
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary report combining all analysis results
        
        Returns:
            Dictionary containing structured analysis results with metadata
            
        Raises:
            RuntimeError: If data hasn't been loaded yet
        """
        if self.data is None:
            raise RuntimeError("Data must be loaded before generating report. Call load_data() first.")
        
        try:
            self.logger.info("Generating comprehensive summary report...")
            
            # Initialize report structure
            report = {
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'file_path': str(self.file_path),
                    'analyzer_version': '1.0.0'
                },
                'dataset_overview': {},
                'property_type_analysis': {},
                'data_quality_metrics': {},
                'industrial_property_summary': {},
                'analysis_status': 'success',
                'errors': []
            }
            
            # Dataset overview
            total_properties = len(self.data)
            total_columns = len(self.data.columns)
            
            report['dataset_overview'] = {
                'total_properties': total_properties,
                'total_columns': total_columns,
                'column_names': list(self.data.columns),
                'memory_usage_mb': round(self.data.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            }
            
            self.logger.info(f"Dataset overview: {total_properties:,} properties, {total_columns} columns")
            
            # Property type analysis
            try:
                industrial_types = self.analyze_property_types()
                
                # Get property type column for detailed analysis
                property_type_col = None
                for col in self.data.columns:
                    if 'property type' in col.lower() or 'type' in col.lower():
                        property_type_col = col
                        break
                
                if property_type_col:
                    property_type_counts = self.data[property_type_col].value_counts().to_dict()
                    total_industrial = sum(property_type_counts.get(itype, 0) for itype in industrial_types)
                    
                    report['property_type_analysis'] = {
                        'total_unique_types': len(property_type_counts),
                        'all_property_types': property_type_counts,
                        'industrial_types_found': industrial_types,
                        'total_industrial_properties': total_industrial,
                        'industrial_percentage': round((total_industrial / total_properties) * 100, 2) if total_properties > 0 else 0
                    }
                else:
                    report['property_type_analysis'] = {
                        'error': 'No property type column found',
                        'industrial_types_found': [],
                        'total_industrial_properties': 0
                    }
                    
            except Exception as e:
                self.logger.error(f"Property type analysis failed: {e}")
                report['property_type_analysis'] = {'error': str(e)}
                report['errors'].append(f"Property type analysis: {str(e)}")
            
            # Data quality metrics
            try:
                completeness = self.check_data_completeness()
                
                if completeness:
                    avg_completeness = sum(completeness.values()) / len(completeness)
                    min_completeness = min(completeness.values())
                    max_completeness = max(completeness.values())
                    
                    report['data_quality_metrics'] = {
                        'field_completeness': completeness,
                        'average_completeness': round(avg_completeness, 2),
                        'min_completeness': round(min_completeness, 2),
                        'max_completeness': round(max_completeness, 2),
                        'fields_analyzed': len(completeness),
                        'complete_fields': len([f for f, pct in completeness.items() if pct == 100.0]),
                        'incomplete_fields': len([f for f, pct in completeness.items() if pct < 100.0])
                    }
                else:
                    report['data_quality_metrics'] = {
                        'error': 'No key fields found for analysis',
                        'fields_analyzed': 0
                    }
                    
            except Exception as e:
                self.logger.error(f"Data quality analysis failed: {e}")
                report['data_quality_metrics'] = {'error': str(e)}
                report['errors'].append(f"Data quality analysis: {str(e)}")
            
            # Industrial property summary
            try:
                industrial_sample = self.get_industrial_sample(limit=5)
                
                report['industrial_property_summary'] = {
                    'sample_count': len(industrial_sample),
                    'sample_properties': industrial_sample.to_dict('records') if not industrial_sample.empty else [],
                    'has_industrial_properties': not industrial_sample.empty
                }
                
            except Exception as e:
                self.logger.error(f"Industrial property summary failed: {e}")
                report['industrial_property_summary'] = {'error': str(e)}
                report['errors'].append(f"Industrial property summary: {str(e)}")
            
            # Update analysis status
            if report['errors']:
                report['analysis_status'] = 'partial_success'
                self.logger.warning(f"Report generated with {len(report['errors'])} errors")
            else:
                self.logger.info("Report generated successfully")
            
            # Store results for future reference
            self.analysis_results = report
            
            # Log summary
            self.logger.info("=== ANALYSIS SUMMARY ===")
            self.logger.info(f"Total Properties: {report['dataset_overview']['total_properties']:,}")
            
            if 'total_industrial_properties' in report.get('property_type_analysis', {}):
                industrial_count = report['property_type_analysis']['total_industrial_properties']
                industrial_pct = report['property_type_analysis']['industrial_percentage']
                self.logger.info(f"Industrial Properties: {industrial_count:,} ({industrial_pct}%)")
            
            if 'average_completeness' in report.get('data_quality_metrics', {}):
                avg_quality = report['data_quality_metrics']['average_completeness']
                self.logger.info(f"Average Data Quality: {avg_quality}%")
            
            self.logger.info("========================")
            
            return report
            
        except Exception as e:
            error_msg = f"Error generating summary report: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg) from e
    
    def export_results(self, output_dir: str = "data/analysis_results", formats: List[str] = None) -> Dict[str, str]:
        """
        Export analysis results to multiple formats
        
        Args:
            output_dir: Directory to save export files
            formats: List of formats to export ('excel', 'json', 'csv'). Defaults to all formats.
            
        Returns:
            Dictionary mapping format names to file paths
            
        Raises:
            RuntimeError: If no analysis results are available
        """
        if not self.analysis_results:
            raise RuntimeError("No analysis results available. Run generate_summary_report() first.")
        
        if formats is None:
            formats = ['excel', 'json', 'csv']
        
        try:
            self.logger.info(f"Exporting analysis results to formats: {formats}")
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp for file naming
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f"property_analysis_{timestamp}"
            
            exported_files = {}
            
            # Export to JSON
            if 'json' in formats:
                json_file = output_path / f"{base_filename}.json"
                
                with open(json_file, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(self.analysis_results, f, indent=2, default=str, ensure_ascii=False)
                
                exported_files['json'] = str(json_file)
                self.logger.info(f"JSON export saved: {json_file}")
            
            # Export to Excel
            if 'excel' in formats:
                excel_file = output_path / f"{base_filename}.xlsx"
                
                with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                    # Summary sheet
                    summary_data = {
                        'Metric': [],
                        'Value': []
                    }
                    
                    # Dataset overview
                    overview = self.analysis_results.get('dataset_overview', {})
                    summary_data['Metric'].extend(['Total Properties', 'Total Columns', 'Memory Usage (MB)'])
                    summary_data['Value'].extend([
                        overview.get('total_properties', 'N/A'),
                        overview.get('total_columns', 'N/A'),
                        overview.get('memory_usage_mb', 'N/A')
                    ])
                    
                    # Property type analysis
                    prop_analysis = self.analysis_results.get('property_type_analysis', {})
                    if 'total_industrial_properties' in prop_analysis:
                        summary_data['Metric'].extend(['Industrial Properties', 'Industrial Percentage'])
                        summary_data['Value'].extend([
                            prop_analysis.get('total_industrial_properties', 'N/A'),
                            f"{prop_analysis.get('industrial_percentage', 'N/A')}%"
                        ])
                    
                    # Data quality
                    quality = self.analysis_results.get('data_quality_metrics', {})
                    if 'average_completeness' in quality:
                        summary_data['Metric'].extend(['Average Data Completeness', 'Complete Fields', 'Incomplete Fields'])
                        summary_data['Value'].extend([
                            f"{quality.get('average_completeness', 'N/A')}%",
                            quality.get('complete_fields', 'N/A'),
                            quality.get('incomplete_fields', 'N/A')
                        ])
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Property types sheet
                    if 'all_property_types' in prop_analysis:
                        prop_types_data = []
                        for prop_type, count in prop_analysis['all_property_types'].items():
                            is_industrial = prop_type in prop_analysis.get('industrial_types_found', [])
                            prop_types_data.append({
                                'Property Type': prop_type,
                                'Count': count,
                                'Is Industrial': 'Yes' if is_industrial else 'No'
                            })
                        
                        prop_types_df = pd.DataFrame(prop_types_data)
                        prop_types_df.to_excel(writer, sheet_name='Property Types', index=False)
                    
                    # Data completeness sheet
                    if 'field_completeness' in quality:
                        completeness_data = []
                        for field, percentage in quality['field_completeness'].items():
                            total_props = overview.get('total_properties', 0)
                            non_null_count = int((percentage / 100) * total_props) if total_props > 0 else 0
                            
                            completeness_data.append({
                                'Field': field,
                                'Completeness %': round(percentage, 2),
                                'Non-null Count': non_null_count,
                                'Missing Count': total_props - non_null_count
                            })
                        
                        completeness_df = pd.DataFrame(completeness_data)
                        completeness_df.to_excel(writer, sheet_name='Data Completeness', index=False)
                    
                    # Industrial sample sheet
                    industrial_summary = self.analysis_results.get('industrial_property_summary', {})
                    if industrial_summary.get('sample_properties'):
                        sample_df = pd.DataFrame(industrial_summary['sample_properties'])
                        sample_df.to_excel(writer, sheet_name='Industrial Sample', index=False)
                
                exported_files['excel'] = str(excel_file)
                self.logger.info(f"Excel export saved: {excel_file}")
            
            # Export to CSV (industrial properties only)
            if 'csv' in formats and self.data is not None:
                csv_file = output_path / f"{base_filename}_industrial_properties.csv"
                
                try:
                    # Get industrial properties for CSV export
                    industrial_sample = self.get_industrial_sample(limit=1000)  # Export up to 1000
                    
                    if not industrial_sample.empty:
                        industrial_sample.to_csv(csv_file, index=False, encoding='utf-8')
                        exported_files['csv'] = str(csv_file)
                        self.logger.info(f"CSV export saved: {csv_file} ({len(industrial_sample)} properties)")
                    else:
                        self.logger.warning("No industrial properties found for CSV export")
                        
                except Exception as e:
                    self.logger.warning(f"CSV export failed: {e}")
            
            self.logger.info(f"Export complete. Files saved to: {output_path}")
            return exported_files
            
        except Exception as e:
            error_msg = f"Error exporting results: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg) from e
    
    def convert_to_pipeline_format(self, property_row: pd.Series) -> Dict[str, Any]:
        """
        Convert Excel property data to pipeline property format for scoring
        
        Args:
            property_row: Pandas Series containing property data from Excel
            
        Returns:
            Dictionary in format expected by FlexPropertyScorer
        """
        try:
            # Create base property dictionary
            property_dict = {
                'parcel_id': property_row.get('Parcel ID', f"excel_{property_row.name}"),
                'property_use': property_row.get('Property Type', ''),
                'owner_name': property_row.get('Owner Name', ''),
                'address': property_row.get('Property Name', '') or property_row.get('Address', ''),
                'municipality': property_row.get('City', ''),
                'market_value': 0,
                'assessed_value': 0,
                'acres': 0,
                'zoning': '',
                'building_sqft': 0,
                'year_built': None
            }
            
            # Map Excel columns to pipeline format with flexible column matching
            column_mappings = {
                # Size fields
                'acres': ['Lot Size Acres', 'Acres', 'Land Size', 'Lot Acres'],
                'building_sqft': ['Building SqFt', 'Building Sq Ft', 'Building Square Feet', 'Sq Ft'],
                
                # Value fields  
                'market_value': ['Market Value', 'Assessed Value', 'Sale Price', 'Sold Price', 'Value'],
                'assessed_value': ['Assessed Value', 'Tax Value', 'Appraised Value'],
                
                # Other fields
                'zoning': ['Zoning', 'Zoning Code', 'Zone'],
                'year_built': ['Year Built', 'Built Year', 'Construction Year'],
                'municipality': ['City', 'Municipality', 'Location'],
                'address': ['Address', 'Street Address', 'Property Address'],
                'owner_name': ['Owner', 'Owner Name', 'Property Owner']
            }
            
            # Apply mappings with case-insensitive matching
            for target_field, possible_columns in column_mappings.items():
                for col_name in possible_columns:
                    # Find matching column in Excel data (case-insensitive)
                    matching_col = None
                    for excel_col in property_row.index:
                        if col_name.lower() in excel_col.lower() or excel_col.lower() in col_name.lower():
                            matching_col = excel_col
                            break
                    
                    if matching_col and pd.notna(property_row.get(matching_col)):
                        value = property_row[matching_col]
                        
                        # Convert numeric fields
                        if target_field in ['acres', 'building_sqft', 'market_value', 'assessed_value']:
                            try:
                                # Handle string values that might contain commas or currency symbols
                                if isinstance(value, str):
                                    # Remove common formatting
                                    clean_value = value.replace(',', '').replace('$', '').replace(' ', '')
                                    property_dict[target_field] = float(clean_value) if clean_value else 0
                                else:
                                    property_dict[target_field] = float(value) if pd.notna(value) else 0
                            except (ValueError, TypeError):
                                property_dict[target_field] = 0
                        
                        # Convert year built
                        elif target_field == 'year_built':
                            try:
                                year = int(float(value)) if pd.notna(value) else None
                                property_dict[target_field] = year if year and year > 1800 else None
                            except (ValueError, TypeError):
                                property_dict[target_field] = None
                        
                        # String fields
                        else:
                            property_dict[target_field] = str(value) if pd.notna(value) else ''
                        
                        break  # Use first matching column
            
            return property_dict
            
        except Exception as e:
            self.logger.error(f"Error converting property to pipeline format: {e}")
            # Return basic format to avoid breaking the analysis
            return {
                'parcel_id': f"excel_{property_row.name}",
                'property_use': str(property_row.get('Property Type', '')),
                'acres': 0,
                'building_sqft': 0,
                'market_value': 0,
                'assessed_value': 0,
                'zoning': '',
                'municipality': str(property_row.get('City', '')),
                'address': str(property_row.get('Property Name', '')),
                'owner_name': str(property_row.get('Owner Name', ''))
            }
    
    def add_flex_scoring(self, include_all_properties: bool = False) -> pd.DataFrame:
        """
        Add flex scores to industrial properties using FlexPropertyScorer
        
        Args:
            include_all_properties: If True, score all properties. If False, only industrial properties.
            
        Returns:
            DataFrame with flex scores added
            
        Raises:
            RuntimeError: If data hasn't been loaded yet
        """
        if self.data is None:
            raise RuntimeError("Data must be loaded before scoring. Call load_data() first.")
        
        try:
            from processors.flex_scorer import FlexPropertyScorer
            
            self.logger.info("Adding flex scores to properties...")
            
            # Initialize scorer
            scorer = FlexPropertyScorer()
            
            # Determine which properties to score
            if include_all_properties:
                properties_to_score = self.data.copy()
                self.logger.info(f"Scoring all {len(properties_to_score)} properties")
            else:
                # Get industrial properties only
                industrial_types = self.analyze_property_types()
                
                if not industrial_types:
                    self.logger.warning("No industrial properties found for scoring")
                    return pd.DataFrame()
                
                # Find property type column
                property_type_col = None
                for col in self.data.columns:
                    if 'property type' in col.lower() or 'type' in col.lower():
                        property_type_col = col
                        break
                
                if property_type_col is None:
                    self.logger.warning("Cannot filter properties - no property type column found")
                    return pd.DataFrame()
                
                # Filter for industrial properties
                industrial_mask = self.data[property_type_col].isin(industrial_types)
                properties_to_score = self.data[industrial_mask].copy()
                self.logger.info(f"Scoring {len(properties_to_score)} industrial properties")
            
            if len(properties_to_score) == 0:
                self.logger.warning("No properties to score")
                return pd.DataFrame()
            
            # Add scoring columns
            properties_to_score['flex_score'] = 0.0
            properties_to_score['flex_indicators'] = None
            properties_to_score['scoring_errors'] = None
            
            successful_scores = 0
            failed_scores = 0
            
            # Score each property
            for idx, (_, row) in enumerate(properties_to_score.iterrows()):
                try:
                    # Convert to pipeline format
                    pipeline_property = self.convert_to_pipeline_format(row)
                    
                    # Calculate flex score
                    score, indicators = scorer.calculate_flex_score(pipeline_property)
                    
                    # Store results
                    properties_to_score.at[row.name, 'flex_score'] = score
                    properties_to_score.at[row.name, 'flex_indicators'] = str(indicators)
                    
                    successful_scores += 1
                    
                    if (idx + 1) % 100 == 0:  # Log progress every 100 properties
                        self.logger.info(f"Scored {idx + 1}/{len(properties_to_score)} properties")
                
                except Exception as e:
                    self.logger.warning(f"Failed to score property {row.name}: {e}")
                    properties_to_score.at[row.name, 'scoring_errors'] = str(e)
                    failed_scores += 1
            
            self.logger.info(f"Scoring complete: {successful_scores} successful, {failed_scores} failed")
            
            # Sort by flex score (highest first)
            scored_properties = properties_to_score.sort_values('flex_score', ascending=False)
            
            # Log top scoring properties
            top_properties = scored_properties.head(5)
            self.logger.info("Top 5 flex scoring properties:")
            for idx, (_, prop) in enumerate(top_properties.iterrows(), 1):
                name = prop.get('Property Name', 'Unknown')
                score = prop.get('flex_score', 0)
                prop_type = prop.get('Property Type', 'Unknown')
                self.logger.info(f"  {idx}. {name} - Score: {score:.2f} ({prop_type})")
            
            return scored_properties
            
        except ImportError as e:
            error_msg = f"FlexPropertyScorer not available: {e}"
            self.logger.error(error_msg)
            raise Exception(error_msg) from e
            
        except Exception as e:
            error_msg = f"Error adding flex scores: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg) from e
    
    def store_results_in_database(self, collection_name: str = "private_property_analysis") -> bool:
        """
        Store analysis results in MongoDB using existing database infrastructure
        
        Args:
            collection_name: Name of MongoDB collection to store results
            
        Returns:
            True if storage successful, False otherwise
        """
        try:
            from database.mongodb_client import get_db_manager
            
            if not self.analysis_results:
                self.logger.warning("No analysis results to store. Run generate_summary_report() first.")
                return False
            
            self.logger.info(f"Storing analysis results in MongoDB collection: {collection_name}")
            
            # Get database manager
            db_manager = get_db_manager()
            
            # Prepare document for storage
            storage_document = {
                'analysis_id': f"private_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'file_path': str(self.file_path),
                'analysis_timestamp': datetime.now(),
                'results': self.analysis_results,
                'analyzer_version': '1.0.0'
            }
            
            # Store in MongoDB
            result = db_manager.db[collection_name].insert_one(storage_document)
            
            if result.inserted_id:
                self.logger.info(f"Analysis results stored with ID: {result.inserted_id}")
                return True
            else:
                self.logger.error("Failed to store analysis results")
                return False
                
        except ImportError as e:
            self.logger.warning(f"MongoDB client not available: {e}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error storing results in database: {e}")
            return False
    
    def store_scored_properties(self, scored_properties: pd.DataFrame, collection_name: str = "private_scored_properties") -> int:
        """
        Store scored properties in MongoDB for future analysis
        
        Args:
            scored_properties: DataFrame containing properties with flex scores
            collection_name: Name of MongoDB collection to store properties
            
        Returns:
            Number of properties successfully stored
        """
        try:
            from database.mongodb_client import get_db_manager
            
            if scored_properties.empty:
                self.logger.warning("No scored properties to store")
                return 0
            
            self.logger.info(f"Storing {len(scored_properties)} scored properties in MongoDB")
            
            # Get database manager
            db_manager = get_db_manager()
            
            # Convert DataFrame to documents
            documents = []
            for _, row in scored_properties.iterrows():
                doc = {
                    'analysis_id': f"private_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'source_file': str(self.file_path),
                    'stored_timestamp': datetime.now(),
                    'property_data': row.to_dict()
                }
                
                # Convert numpy types to native Python types for MongoDB
                for key, value in doc['property_data'].items():
                    if pd.isna(value):
                        doc['property_data'][key] = None
                    elif hasattr(value, 'item'):  # numpy types
                        doc['property_data'][key] = value.item()
                
                documents.append(doc)
            
            # Batch insert using existing infrastructure
            inserted_count = db_manager.batch_insert(collection_name, documents, batch_size=100)
            
            self.logger.info(f"Successfully stored {inserted_count} scored properties")
            return inserted_count
            
        except ImportError as e:
            self.logger.warning(f"MongoDB client not available: {e}")
            return 0
            
        except Exception as e:
            self.logger.error(f"Error storing scored properties: {e}")
            return 0
    
    def retrieve_historical_analysis(self, limit: int = 10) -> List[Dict]:
        """
        Retrieve historical analysis results from MongoDB
        
        Args:
            limit: Maximum number of historical analyses to retrieve
            
        Returns:
            List of historical analysis documents
        """
        try:
            from database.mongodb_client import get_db_manager
            
            self.logger.info(f"Retrieving up to {limit} historical analyses")
            
            # Get database manager
            db_manager = get_db_manager()
            
            # Query historical analyses
            pipeline = [
                {'$sort': {'analysis_timestamp': -1}},  # Most recent first
                {'$limit': limit}
            ]
            
            historical_analyses = list(db_manager.db.private_property_analysis.aggregate(pipeline))
            
            self.logger.info(f"Retrieved {len(historical_analyses)} historical analyses")
            
            # Log summary of historical analyses
            for i, analysis in enumerate(historical_analyses, 1):
                timestamp = analysis.get('analysis_timestamp', 'Unknown')
                file_path = analysis.get('file_path', 'Unknown')
                total_props = analysis.get('results', {}).get('dataset_overview', {}).get('total_properties', 'Unknown')
                
                self.logger.info(f"  {i}. {timestamp} - {file_path} ({total_props} properties)")
            
            return historical_analyses
            
        except ImportError as e:
            self.logger.warning(f"MongoDB client not available: {e}")
            return []
            
        except Exception as e:
            self.logger.error(f"Error retrieving historical analysis: {e}")
            return []
    
    def compare_with_historical(self, historical_limit: int = 5) -> Dict[str, Any]:
        """
        Compare current analysis with historical analyses
        
        Args:
            historical_limit: Number of historical analyses to compare against
            
        Returns:
            Dictionary containing comparison results
        """
        try:
            if not self.analysis_results:
                self.logger.warning("No current analysis results to compare. Run generate_summary_report() first.")
                return {}
            
            self.logger.info("Comparing current analysis with historical data")
            
            # Get historical analyses
            historical_analyses = self.retrieve_historical_analysis(limit=historical_limit)
            
            if not historical_analyses:
                self.logger.info("No historical analyses found for comparison")
                return {'comparison_available': False}
            
            # Extract metrics for comparison
            current_metrics = {
                'total_properties': self.analysis_results.get('dataset_overview', {}).get('total_properties', 0),
                'industrial_properties': self.analysis_results.get('property_type_analysis', {}).get('total_industrial_properties', 0),
                'industrial_percentage': self.analysis_results.get('property_type_analysis', {}).get('industrial_percentage', 0),
                'avg_completeness': self.analysis_results.get('data_quality_metrics', {}).get('average_completeness', 0)
            }
            
            # Compare with historical
            comparison_results = {
                'comparison_available': True,
                'current_analysis': current_metrics,
                'historical_analyses': [],
                'trends': {}
            }
            
            historical_metrics = []
            for analysis in historical_analyses:
                results = analysis.get('results', {})
                metrics = {
                    'timestamp': analysis.get('analysis_timestamp'),
                    'file_path': analysis.get('file_path'),
                    'total_properties': results.get('dataset_overview', {}).get('total_properties', 0),
                    'industrial_properties': results.get('property_type_analysis', {}).get('total_industrial_properties', 0),
                    'industrial_percentage': results.get('property_type_analysis', {}).get('industrial_percentage', 0),
                    'avg_completeness': results.get('data_quality_metrics', {}).get('average_completeness', 0)
                }
                historical_metrics.append(metrics)
                comparison_results['historical_analyses'].append(metrics)
            
            # Calculate trends
            if historical_metrics:
                avg_historical_props = sum(h['total_properties'] for h in historical_metrics) / len(historical_metrics)
                avg_historical_industrial = sum(h['industrial_properties'] for h in historical_metrics) / len(historical_metrics)
                avg_historical_quality = sum(h['avg_completeness'] for h in historical_metrics) / len(historical_metrics)
                
                comparison_results['trends'] = {
                    'properties_vs_avg': current_metrics['total_properties'] - avg_historical_props,
                    'industrial_vs_avg': current_metrics['industrial_properties'] - avg_historical_industrial,
                    'quality_vs_avg': current_metrics['avg_completeness'] - avg_historical_quality
                }
                
                self.logger.info(f"Comparison complete: Current vs Historical Average")
                self.logger.info(f"  Properties: {current_metrics['total_properties']:,} vs {avg_historical_props:.0f} (diff: {comparison_results['trends']['properties_vs_avg']:+.0f})")
                self.logger.info(f"  Industrial: {current_metrics['industrial_properties']:,} vs {avg_historical_industrial:.0f} (diff: {comparison_results['trends']['industrial_vs_avg']:+.0f})")
                self.logger.info(f"  Quality: {current_metrics['avg_completeness']:.1f}% vs {avg_historical_quality:.1f}% (diff: {comparison_results['trends']['quality_vs_avg']:+.1f}%)")
            
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"Error comparing with historical data: {e}")
            return {'comparison_available': False, 'error': str(e)}