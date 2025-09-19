"""
File Discovery and Validation System
Handles scanning input folders for Excel files and validating file formats
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
import pandas as pd
from datetime import datetime
import os

from utils.logger import setup_logging


class FileDiscovery:
    """
    Handles file discovery and validation for the scalable flex pipeline
    
    Scans input folders for Excel files, validates formats, and extracts metadata
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize file discovery system
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or setup_logging('file_discovery')
        
        # Supported file extensions
        self.supported_extensions = ['.xlsx', '.xls']
        
        # Required columns for flex property analysis (flexible matching)
        self.required_column_patterns = [
            ['address', 'property address', 'site address'],
            ['city'],
            ['state'],
            ['property type', 'type', 'use type'],
            ['building', 'sqft', 'square feet', 'building size']
        ]
        
        # Optional but recommended columns
        self.recommended_columns = [
            ['county'],
            ['acres', 'lot size', 'lot acres'],
            ['year built', 'built year'],
            ['zoning', 'zone'],
            ['owner', 'owner name']
        ]
    
    def scan_input_folder(self, folder_path: str, file_pattern: str = "*.xlsx", 
                         recursive: bool = False) -> List[Path]:
        """
        Scan input folder for Excel files
        
        Args:
            folder_path: Path to input folder
            file_pattern: File pattern to match (e.g., "*.xlsx")
            recursive: Whether to scan subdirectories recursively
            
        Returns:
            List of Path objects for discovered Excel files
        """
        try:
            input_path = Path(folder_path)
            
            if not input_path.exists():
                self.logger.error(f"Input folder does not exist: {folder_path}")
                return []
            
            if not input_path.is_dir():
                self.logger.error(f"Input path is not a directory: {folder_path}")
                return []
            
            self.logger.info(f"Scanning folder: {input_path}")
            self.logger.info(f"Pattern: {file_pattern}, Recursive: {recursive}")
            
            discovered_files = []
            
            if recursive:
                # Recursive scan
                pattern_parts = file_pattern.split('.')
                if len(pattern_parts) >= 2:
                    extension = '.' + pattern_parts[-1]
                    for file_path in input_path.rglob(f"*{extension}"):
                        if file_path.is_file() and self._is_supported_file(file_path):
                            discovered_files.append(file_path)
                else:
                    # Fallback to all Excel files
                    for ext in self.supported_extensions:
                        discovered_files.extend(input_path.rglob(f"*{ext}"))
            else:
                # Non-recursive scan
                pattern_parts = file_pattern.split('.')
                if len(pattern_parts) >= 2:
                    extension = '.' + pattern_parts[-1]
                    for file_path in input_path.glob(f"*{extension}"):
                        if file_path.is_file() and self._is_supported_file(file_path):
                            discovered_files.append(file_path)
                else:
                    # Fallback to all Excel files
                    for ext in self.supported_extensions:
                        discovered_files.extend(input_path.glob(f"*{ext}"))
            
            # Remove duplicates and sort
            discovered_files = sorted(list(set(discovered_files)))
            
            self.logger.info(f"Discovered {len(discovered_files)} Excel files")
            
            # Log file details
            for file_path in discovered_files:
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                self.logger.debug(f"  {file_path.name} ({file_size:.1f} MB)")
            
            return discovered_files
            
        except Exception as e:
            self.logger.error(f"Error scanning input folder: {e}")
            return []
    
    def _is_supported_file(self, file_path: Path) -> bool:
        """
        Check if file has supported extension
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file is supported, False otherwise
        """
        return file_path.suffix.lower() in self.supported_extensions
    
    def validate_file_format(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate Excel file format and structure
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            Dictionary containing validation results
        """
        validation_result = {
            'file_path': str(file_path),
            'is_valid': False,
            'can_read': False,
            'has_data': False,
            'column_count': 0,
            'row_count': 0,
            'required_columns_found': [],
            'missing_columns': [],
            'recommended_columns_found': [],
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check if file exists and is readable
            if not file_path.exists():
                validation_result['errors'].append("File does not exist")
                return validation_result
            
            if not file_path.is_file():
                validation_result['errors'].append("Path is not a file")
                return validation_result
            
            # Try to read the Excel file
            try:
                # Read just the header first to check structure
                df_header = pd.read_excel(file_path, nrows=0)
                validation_result['can_read'] = True
                validation_result['column_count'] = len(df_header.columns)
                
                # Read a small sample to check for data
                df_sample = pd.read_excel(file_path, nrows=5)
                validation_result['row_count'] = len(df_sample)
                validation_result['has_data'] = len(df_sample) > 0
                
                # Check column structure
                columns = [col.lower().strip() for col in df_header.columns]
                
                # Check for required columns
                required_found = []
                missing_required = []
                
                for required_group in self.required_column_patterns:
                    found = False
                    for pattern in required_group:
                        if any(pattern.lower() in col for col in columns):
                            required_found.append(pattern)
                            found = True
                            break
                    
                    if not found:
                        missing_required.append(required_group[0])  # Use first pattern as representative
                
                validation_result['required_columns_found'] = required_found
                validation_result['missing_columns'] = missing_required
                
                # Check for recommended columns
                recommended_found = []
                for recommended_group in self.recommended_columns:
                    for pattern in recommended_group:
                        if any(pattern.lower() in col for col in columns):
                            recommended_found.append(pattern)
                            break
                
                validation_result['recommended_columns_found'] = recommended_found
                
                # Determine if file is valid
                validation_result['is_valid'] = (
                    validation_result['can_read'] and 
                    validation_result['has_data'] and 
                    len(validation_result['missing_columns']) == 0
                )
                
                # Add warnings for missing recommended columns
                missing_recommended = []
                for recommended_group in self.recommended_columns:
                    found = False
                    for pattern in recommended_group:
                        if any(pattern.lower() in col for col in columns):
                            found = True
                            break
                    if not found:
                        missing_recommended.append(recommended_group[0])
                
                if missing_recommended:
                    validation_result['warnings'].append(
                        f"Missing recommended columns: {', '.join(missing_recommended)}"
                    )
                
                # Check for empty data
                if validation_result['row_count'] == 0:
                    validation_result['warnings'].append("File appears to be empty")
                
                # Check for very small datasets
                if validation_result['row_count'] < 10:
                    validation_result['warnings'].append(
                        f"File has very few rows ({validation_result['row_count']})"
                    )
                
            except Exception as read_error:
                validation_result['errors'].append(f"Cannot read Excel file: {str(read_error)}")
                validation_result['can_read'] = False
            
        except Exception as e:
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from Excel file
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            Dictionary containing file metadata
        """
        metadata = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_size_mb': 0.0,
            'modified_date': None,
            'created_date': None,
            'sheet_count': 0,
            'sheet_names': [],
            'total_rows': 0,
            'total_columns': 0,
            'estimated_properties': 0
        }
        
        try:
            # Basic file information
            stat = file_path.stat()
            metadata['file_size_mb'] = stat.st_size / (1024 * 1024)
            metadata['modified_date'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
            metadata['created_date'] = datetime.fromtimestamp(stat.st_ctime).isoformat()
            
            # Excel-specific information
            try:
                # Get sheet information
                excel_file = pd.ExcelFile(file_path)
                metadata['sheet_count'] = len(excel_file.sheet_names)
                metadata['sheet_names'] = excel_file.sheet_names
                
                # Read the first sheet to get dimensions
                df = pd.read_excel(file_path, sheet_name=0)
                metadata['total_rows'] = len(df)
                metadata['total_columns'] = len(df.columns)
                
                # Estimate number of properties (assuming each row is a property)
                metadata['estimated_properties'] = max(0, len(df) - 1)  # Subtract header row
                
            except Exception as excel_error:
                self.logger.warning(f"Could not read Excel metadata for {file_path}: {excel_error}")
        
        except Exception as e:
            self.logger.error(f"Error extracting metadata for {file_path}: {e}")
        
        return metadata
    
    def validate_batch_files(self, file_paths: List[Path]) -> Dict[str, Any]:
        """
        Validate a batch of files and provide summary
        
        Args:
            file_paths: List of file paths to validate
            
        Returns:
            Dictionary containing batch validation summary
        """
        batch_results = {
            'total_files': len(file_paths),
            'valid_files': 0,
            'invalid_files': 0,
            'files_with_warnings': 0,
            'total_estimated_properties': 0,
            'total_size_mb': 0.0,
            'validation_details': [],
            'schema_compatibility': True,
            'common_columns': [],
            'schema_differences': []
        }
        
        try:
            self.logger.info(f"Validating batch of {len(file_paths)} files...")
            
            all_columns = []
            valid_files_columns = []
            
            for file_path in file_paths:
                self.logger.debug(f"Validating: {file_path.name}")
                
                # Validate individual file
                validation = self.validate_file_format(file_path)
                metadata = self.get_file_metadata(file_path)
                
                # Combine validation and metadata
                file_result = {**validation, **metadata}
                batch_results['validation_details'].append(file_result)
                
                # Update counters
                if validation['is_valid']:
                    batch_results['valid_files'] += 1
                    
                    # Collect columns for schema analysis
                    try:
                        df_header = pd.read_excel(file_path, nrows=0)
                        file_columns = [col.lower().strip() for col in df_header.columns]
                        valid_files_columns.append(file_columns)
                        all_columns.extend(file_columns)
                    except:
                        pass
                else:
                    batch_results['invalid_files'] += 1
                
                if validation['warnings']:
                    batch_results['files_with_warnings'] += 1
                
                batch_results['total_estimated_properties'] += metadata['estimated_properties']
                batch_results['total_size_mb'] += metadata['file_size_mb']
            
            # Analyze schema compatibility
            if valid_files_columns:
                # Find common columns across all valid files
                common_columns = set(valid_files_columns[0])
                for file_columns in valid_files_columns[1:]:
                    common_columns = common_columns.intersection(set(file_columns))
                
                batch_results['common_columns'] = sorted(list(common_columns))
                
                # Check for schema differences
                all_unique_columns = set(all_columns)
                if len(common_columns) < len(all_unique_columns):
                    batch_results['schema_compatibility'] = False
                    
                    # Identify differences
                    for i, file_columns in enumerate(valid_files_columns):
                        missing_from_file = all_unique_columns - set(file_columns)
                        if missing_from_file:
                            batch_results['schema_differences'].append({
                                'file_index': i,
                                'file_name': batch_results['validation_details'][i]['file_name'],
                                'missing_columns': sorted(list(missing_from_file))
                            })
            
            # Log summary
            self.logger.info(f"Batch validation complete:")
            self.logger.info(f"  Valid files: {batch_results['valid_files']}/{batch_results['total_files']}")
            self.logger.info(f"  Files with warnings: {batch_results['files_with_warnings']}")
            self.logger.info(f"  Total estimated properties: {batch_results['total_estimated_properties']:,}")
            self.logger.info(f"  Total size: {batch_results['total_size_mb']:.1f} MB")
            self.logger.info(f"  Schema compatibility: {batch_results['schema_compatibility']}")
            
            if not batch_results['schema_compatibility']:
                self.logger.warning("Schema differences detected between files - may affect aggregation")
            
        except Exception as e:
            self.logger.error(f"Error during batch validation: {e}")
        
        return batch_results
    
    def filter_valid_files(self, file_paths: List[Path]) -> List[Path]:
        """
        Filter list to only include valid files
        
        Args:
            file_paths: List of file paths to filter
            
        Returns:
            List of valid file paths
        """
        valid_files = []
        
        for file_path in file_paths:
            validation = self.validate_file_format(file_path)
            if validation['is_valid']:
                valid_files.append(file_path)
            else:
                self.logger.warning(f"Excluding invalid file: {file_path.name}")
                for error in validation['errors']:
                    self.logger.warning(f"  Error: {error}")
        
        self.logger.info(f"Filtered to {len(valid_files)} valid files out of {len(file_paths)} total")
        return valid_files


if __name__ == "__main__":
    # Test the file discovery system
    discovery = FileDiscovery()
    
    # Test folder scanning
    files = discovery.scan_input_folder("data/raw", recursive=False)
    print(f"Discovered {len(files)} files")
    
    # Test validation
    if files:
        validation = discovery.validate_batch_files(files[:3])  # Test first 3 files
        print(f"Validation results: {validation['valid_files']} valid files")