"""
File Processor for Scalable Multi-File Pipeline
Handles individual Excel file processing with flex property classification
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import traceback

from processors.flex_scorer import FlexPropertyScorer
from pipeline.performance_optimizer import PerformanceOptimizer


@dataclass
class ProcessingResult:
    """Standardized result structure for file processing operations"""
    
    file_path: str
    success: bool
    flex_properties: Optional[pd.DataFrame] = None
    property_count: int = 0
    flex_candidate_count: int = 0
    processing_time: float = 0.0
    error_message: Optional[str] = None
    source_file_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ProcessingResult to dictionary for serialization"""
        return {
            'file_path': self.file_path,
            'success': self.success,
            'property_count': self.property_count,
            'flex_candidate_count': self.flex_candidate_count,
            'processing_time': self.processing_time,
            'error_message': self.error_message,
            'source_file_info': self.source_file_info,
            'has_flex_properties': self.flex_properties is not None and len(self.flex_properties) > 0
        }


class FileProcessor:
    """
    Processes individual Excel files with flex property classification
    
    Integrates with existing FlexPropertyScorer to analyze properties
    and add source file metadata tracking
    """
    
    def __init__(self, 
                 min_flex_score: float = 4.0,
                 enable_performance_optimization: bool = True,
                 memory_limit_gb: float = 2.0):
        """
        Initialize FileProcessor
        
        Args:
            min_flex_score: Minimum flex score threshold for candidates
            enable_performance_optimization: Whether to enable performance optimization
            memory_limit_gb: Memory limit for processing operations
        """
        self.logger = logging.getLogger(__name__)
        self.min_flex_score = min_flex_score
        self.flex_scorer = FlexPropertyScorer()
        
        # Performance optimization
        if enable_performance_optimization:
            self.performance_optimizer = PerformanceOptimizer(
                memory_limit_gb=memory_limit_gb,
                chunk_size=10000,
                enable_monitoring=False,  # Disable monitoring for individual files
                optimization_level="balanced"
            )
        else:
            self.performance_optimizer = None
        
        # Expected column mappings for property data
        self.column_mappings = {
            'parcel_id': ['parcel_id', 'parcel', 'id', 'property_id', 'parcel id'],
            'site_address': ['site_address', 'address', 'property_address', 'street_address', 'property address'],
            'city': ['city', 'municipality'],
            'state': ['state', 'st'],
            'zip_code': ['zip_code', 'zip', 'postal_code'],
            'acres': ['acres', 'lot_size_acres', 'land_acres', 'lot size acres'],
            'zoning': ['zoning', 'zone', 'zoning_code', 'zoning code'],
            'improvement_value': ['improvement_value', 'building_value', 'improvements'],
            'land_market_value': ['land_market_value', 'land_value', 'assessed_land_value'],
            'total_market_value': ['total_market_value', 'market_value', 'assessed_value']
        }
    
    def process_file(self, file_path: Path) -> ProcessingResult:
        """
        Process a single Excel file for flex property candidates
        
        Args:
            file_path: Path to Excel file to process
            
        Returns:
            ProcessingResult with processing outcome and data
        """
        start_time = datetime.now()
        file_str = str(file_path)
        
        self.logger.info(f"Processing file: {file_path.name}")
        
        try:
            # Extract source file metadata
            source_info = self._extract_file_metadata(file_path)
            
            # Load and optimize Excel file
            if self.performance_optimizer:
                # Use performance-optimized loading for large files
                def load_and_normalize(file_path):
                    df = pd.read_excel(file_path)
                    if df is None or df.empty:
                        return None
                    return self._normalize_columns(df)
                
                normalized_df = self.performance_optimizer.process_large_file(
                    file_path, load_and_normalize
                )
            else:
                # Standard loading
                df = self._load_excel_file(file_path)
                
                if df is None or df.empty:
                    return ProcessingResult(
                        file_path=file_str,
                        success=False,
                        error_message="File is empty or could not be loaded",
                        source_file_info=source_info,
                        processing_time=(datetime.now() - start_time).total_seconds()
                    )
                
                normalized_df = self._normalize_columns(df)
            
            if normalized_df is None or normalized_df.empty:
                return ProcessingResult(
                    file_path=file_str,
                    success=False,
                    error_message="Required columns not found in file or file is empty",
                    source_file_info=source_info,
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Apply memory optimization if enabled
            if self.performance_optimizer:
                normalized_df = self.performance_optimizer.optimize_dataframe_operations(normalized_df)
            
            # Process properties for flex classification
            flex_properties = self._classify_properties(normalized_df, source_info)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = ProcessingResult(
                file_path=file_str,
                success=True,
                flex_properties=flex_properties,
                property_count=len(normalized_df),
                flex_candidate_count=len(flex_properties) if flex_properties is not None else 0,
                processing_time=processing_time,
                source_file_info=source_info
            )
            
            self.logger.info(f"Successfully processed {file_path.name}: "
                           f"{result.property_count} properties, "
                           f"{result.flex_candidate_count} flex candidates "
                           f"in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Error processing {file_path.name}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            
            return ProcessingResult(
                file_path=file_str,
                success=False,
                error_message=error_msg,
                source_file_info=self._extract_file_metadata(file_path),
                processing_time=processing_time
            )
    
    def _extract_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from the source file"""
        try:
            stat = file_path.stat()
            return {
                'filename': file_path.name,
                'file_size_bytes': stat.st_size,
                'file_size_mb': round(stat.st_size / (1024 * 1024), 2),
                'modified_date': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'processed_date': datetime.now().isoformat(),
                'file_extension': file_path.suffix.lower()
            }
        except Exception as e:
            self.logger.warning(f"Could not extract metadata from {file_path}: {e}")
            return {
                'filename': file_path.name,
                'processed_date': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _load_excel_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load Excel file with error handling"""
        try:
            # Try to read the Excel file
            df = pd.read_excel(file_path, engine='openpyxl')
            
            if df.empty:
                self.logger.warning(f"File {file_path.name} is empty")
                return None
            
            self.logger.debug(f"Loaded {len(df)} rows from {file_path.name}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load Excel file {file_path}: {e}")
            return None
    
    def _normalize_columns(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Normalize column names to match expected schema"""
        try:
            # Convert column names to lowercase for matching
            df_columns_lower = [col.lower().strip() for col in df.columns]
            column_mapping = {}
            
            # Map columns based on our expected mappings
            for standard_col, possible_names in self.column_mappings.items():
                for possible_name in possible_names:
                    if possible_name.lower() in df_columns_lower:
                        original_col = df.columns[df_columns_lower.index(possible_name.lower())]
                        column_mapping[original_col] = standard_col
                        break
            
            # Check if we have minimum required columns
            required_columns = ['site_address', 'city']  # Minimum for property identification
            mapped_standard_cols = set(column_mapping.values())
            
            missing_required = [col for col in required_columns if col not in mapped_standard_cols]
            if missing_required:
                self.logger.error(f"Missing required columns: {missing_required}")
                self.logger.debug(f"Available columns: {list(df.columns)}")
                return None
            
            # Rename columns
            df_normalized = df.rename(columns=column_mapping)
            
            # Add missing optional columns with default values
            for standard_col in self.column_mappings.keys():
                if standard_col not in df_normalized.columns:
                    if standard_col in ['acres', 'improvement_value', 'land_market_value', 'total_market_value']:
                        df_normalized[standard_col] = 0
                    else:
                        df_normalized[standard_col] = ''
            
            return df_normalized
            
        except Exception as e:
            self.logger.error(f"Failed to normalize columns: {e}")
            return None
    
    def _classify_properties(self, df: pd.DataFrame, source_info: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Classify properties for flex potential and filter candidates"""
        try:
            flex_candidates = []
            
            for idx, row in df.iterrows():
                try:
                    # Convert row to property data dictionary
                    property_data = self._row_to_property_data(row)
                    
                    # Calculate flex score
                    flex_score, indicators = self.flex_scorer.calculate_flex_score(property_data)
                    
                    # Only include properties that meet minimum score threshold
                    if flex_score >= self.min_flex_score:
                        # Add flex scoring results to the row data
                        flex_row = row.copy()
                        flex_row['flex_score'] = flex_score
                        flex_row['flex_classification'] = self.flex_scorer.get_flex_classification(flex_score)
                        
                        # Add indicator scores
                        flex_row['zoning_score'] = indicators.get('zoning_score', 0)
                        flex_row['size_score'] = indicators.get('size_score', 0)
                        flex_row['building_score'] = indicators.get('building_score', 0)
                        flex_row['location_score'] = indicators.get('location_score', 0)
                        flex_row['activity_score'] = indicators.get('activity_score', 0)
                        flex_row['value_score'] = indicators.get('value_score', 0)
                        
                        # Add source file information
                        flex_row['source_filename'] = source_info.get('filename', '')
                        flex_row['source_processed_date'] = source_info.get('processed_date', '')
                        flex_row['source_file_size_mb'] = source_info.get('file_size_mb', 0)
                        
                        flex_candidates.append(flex_row)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process row {idx}: {e}")
                    continue
            
            if not flex_candidates:
                self.logger.info("No flex candidates found meeting minimum score threshold")
                return pd.DataFrame()  # Return empty DataFrame instead of None
            
            flex_df = pd.DataFrame(flex_candidates)
            
            # Sort by flex score descending
            flex_df = flex_df.sort_values('flex_score', ascending=False)
            
            return flex_df
            
        except Exception as e:
            self.logger.error(f"Failed to classify properties: {e}")
            return None
    
    def _row_to_property_data(self, row: pd.Series) -> Dict[str, Any]:
        """Convert DataFrame row to property data dictionary for flex scorer"""
        
        # Handle NaN values by converting to appropriate defaults
        def safe_get(key, default=None):
            value = row.get(key, default)
            if pd.isna(value):
                return default
            return value
        
        property_data = {
            'parcel_id': safe_get('parcel_id', ''),
            'site_address': safe_get('site_address', ''),
            'city': safe_get('city', ''),
            'state': safe_get('state', ''),
            'zip_code': safe_get('zip_code', ''),
            'acres': safe_get('acres', 0),
            'zoning': safe_get('zoning', ''),
            'improvement_value': safe_get('improvement_value', 0),
            'land_market_value': safe_get('land_market_value', 0),
            'total_market_value': safe_get('total_market_value', 0),
            'municipality': safe_get('city', ''),  # Use city as municipality fallback
        }
        
        # Convert numeric fields to proper types
        numeric_fields = ['acres', 'improvement_value', 'land_market_value', 'total_market_value']
        for field in numeric_fields:
            try:
                value = property_data[field]
                if isinstance(value, str):
                    # Remove common formatting characters
                    value = value.replace('$', '').replace(',', '').strip()
                    if value == '' or value == 'N/A':
                        value = 0
                property_data[field] = float(value) if value else 0
            except (ValueError, TypeError):
                property_data[field] = 0
        
        return property_data
    
    def validate_file_format(self, file_path: Path) -> bool:
        """
        Validate that the file can be processed
        
        Args:
            file_path: Path to file to validate
            
        Returns:
            True if file appears to be processable, False otherwise
        """
        try:
            # Check file extension
            if file_path.suffix.lower() not in ['.xlsx', '.xls']:
                return False
            
            # Check if file exists and is readable
            if not file_path.exists() or not file_path.is_file():
                return False
            
            # Try to peek at the file structure
            try:
                df = pd.read_excel(file_path, nrows=1, engine='openpyxl')
                return not df.empty
            except:
                return False
                
        except Exception:
            return False
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about the processor configuration"""
        return {
            'min_flex_score': self.min_flex_score,
            'expected_columns': list(self.column_mappings.keys()),
            'scorer_weights': self.flex_scorer.weights,
            'ideal_flex_criteria': self.flex_scorer.ideal_flex
        }