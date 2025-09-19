"""
Result Aggregator for Scalable Multi-File Pipeline
Combines and deduplicates flex property results from multiple files
"""

import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

from pipeline.file_processor import ProcessingResult


@dataclass
class AggregationStats:
    """Statistics for result aggregation operations"""
    
    total_input_files: int = 0
    successful_files: int = 0
    total_properties_before: int = 0
    total_properties_after: int = 0
    duplicates_removed: int = 0
    unique_addresses: int = 0
    unique_cities: int = 0
    unique_states: int = 0
    score_distribution: Dict[str, int] = field(default_factory=dict)
    aggregation_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary"""
        return {
            'total_input_files': self.total_input_files,
            'successful_files': self.successful_files,
            'total_properties_before': self.total_properties_before,
            'total_properties_after': self.total_properties_after,
            'duplicates_removed': self.duplicates_removed,
            'deduplication_rate': self.duplicates_removed / max(1, self.total_properties_before),
            'unique_addresses': self.unique_addresses,
            'unique_cities': self.unique_cities,
            'unique_states': self.unique_states,
            'score_distribution': self.score_distribution,
            'aggregation_time': self.aggregation_time
        }


class ResultAggregator:
    """
    Aggregates and deduplicates flex property results from multiple files
    
    Combines DataFrames, removes duplicates based on address matching,
    and sorts results by flex score
    """
    
    def __init__(self, 
                 duplicate_fields: List[str] = None,
                 preserve_highest_score: bool = True,
                 case_sensitive_matching: bool = False):
        """
        Initialize ResultAggregator
        
        Args:
            duplicate_fields: Fields to use for duplicate detection
            preserve_highest_score: Whether to keep highest scoring duplicate
            case_sensitive_matching: Whether duplicate matching is case sensitive
        """
        self.duplicate_fields = duplicate_fields or ['site_address', 'city', 'state']
        self.preserve_highest_score = preserve_highest_score
        self.case_sensitive_matching = case_sensitive_matching
        
        self.logger = logging.getLogger(__name__)
        self.stats = AggregationStats()
        
        self.logger.info(f"ResultAggregator initialized with duplicate fields: {self.duplicate_fields}")
    
    def aggregate_results(self, processing_results: List[ProcessingResult]) -> Optional[pd.DataFrame]:
        """
        Aggregate results from multiple file processing operations
        
        Args:
            processing_results: List of ProcessingResult objects
            
        Returns:
            Combined and deduplicated DataFrame, or None if no results
        """
        start_time = datetime.now()
        
        if not processing_results:
            self.logger.warning("No processing results provided for aggregation")
            return None
        
        self.logger.info(f"Starting aggregation of {len(processing_results)} processing results")
        
        # Initialize stats
        self.stats = AggregationStats(
            total_input_files=len(processing_results)
        )
        
        # Extract successful results with flex properties
        successful_results = [r for r in processing_results if r.success and r.flex_properties is not None]
        self.stats.successful_files = len(successful_results)
        
        if not successful_results:
            self.logger.warning("No successful results with flex properties found")
            return pd.DataFrame()  # Return empty DataFrame
        
        # Combine all DataFrames
        combined_df = self._combine_dataframes(successful_results)
        
        if combined_df is None or combined_df.empty:
            self.logger.warning("No data to aggregate after combining DataFrames")
            return pd.DataFrame()
        
        self.stats.total_properties_before = len(combined_df)
        
        # Remove duplicates
        deduplicated_df = self._deduplicate_properties(combined_df)
        
        if deduplicated_df is None or deduplicated_df.empty:
            self.logger.warning("No data remaining after deduplication")
            return pd.DataFrame()
        
        self.stats.total_properties_after = len(deduplicated_df)
        self.stats.duplicates_removed = self.stats.total_properties_before - self.stats.total_properties_after
        
        # Sort by flex score
        sorted_df = self._sort_by_score(deduplicated_df)
        
        # Calculate final statistics
        self._calculate_final_stats(sorted_df)
        
        # Record aggregation time
        self.stats.aggregation_time = (datetime.now() - start_time).total_seconds()
        
        self.logger.info(f"Aggregation completed: {self.stats.total_properties_before} -> "
                        f"{self.stats.total_properties_after} properties "
                        f"({self.stats.duplicates_removed} duplicates removed) "
                        f"in {self.stats.aggregation_time:.2f}s")
        
        return sorted_df
    
    def _combine_dataframes(self, successful_results: List[ProcessingResult]) -> Optional[pd.DataFrame]:
        """Combine DataFrames from multiple processing results"""
        try:
            dataframes = []
            
            for result in successful_results:
                if result.flex_properties is not None and not result.flex_properties.empty:
                    df = result.flex_properties.copy()
                    
                    # Ensure required columns exist
                    self._ensure_required_columns(df)
                    
                    # Add aggregation metadata
                    df['aggregation_source_file'] = result.source_file_info.get('filename', 'unknown')
                    df['aggregation_processed_date'] = datetime.now().isoformat()
                    
                    dataframes.append(df)
            
            if not dataframes:
                return None
            
            # Combine all DataFrames
            combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
            
            self.logger.debug(f"Combined {len(dataframes)} DataFrames into {len(combined_df)} total properties")
            
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Failed to combine DataFrames: {e}")
            return None
    
    def _ensure_required_columns(self, df: pd.DataFrame):
        """Ensure DataFrame has required columns for processing"""
        required_columns = ['flex_score'] + self.duplicate_fields
        
        for col in required_columns:
            if col not in df.columns:
                if col == 'flex_score':
                    df[col] = 0.0
                else:
                    df[col] = ''
                self.logger.warning(f"Missing column '{col}' added with default values")
    
    def _deduplicate_properties(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Remove duplicate properties based on configured fields"""
        try:
            if df.empty:
                return df
            
            # Create duplicate detection key
            df_with_key = self._create_duplicate_key(df)
            
            if 'duplicate_key' not in df_with_key.columns:
                self.logger.error("Failed to create duplicate detection key")
                return df
            
            # Group by duplicate key and handle duplicates
            if self.preserve_highest_score:
                # Keep the record with the highest flex score for each duplicate group
                deduplicated_df = df_with_key.loc[df_with_key.groupby('duplicate_key')['flex_score'].idxmax()]
            else:
                # Keep the first occurrence of each duplicate group
                deduplicated_df = df_with_key.drop_duplicates(subset=['duplicate_key'], keep='first')
            
            # Remove the temporary duplicate key column
            deduplicated_df = deduplicated_df.drop(columns=['duplicate_key'])
            
            duplicates_found = len(df) - len(deduplicated_df)
            
            if duplicates_found > 0:
                self.logger.info(f"Removed {duplicates_found} duplicate properties")
            
            return deduplicated_df
            
        except Exception as e:
            self.logger.error(f"Failed to deduplicate properties: {e}")
            return df  # Return original DataFrame if deduplication fails
    
    def _create_duplicate_key(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a key for duplicate detection based on configured fields"""
        try:
            df_copy = df.copy()
            
            # Create normalized values for duplicate detection
            key_parts = []
            
            for field in self.duplicate_fields:
                if field in df_copy.columns:
                    # Get field values and handle NaN
                    field_values = df_copy[field].fillna('').astype(str)
                    
                    if not self.case_sensitive_matching:
                        field_values = field_values.str.lower()
                    
                    # Normalize whitespace and common variations
                    field_values = field_values.str.strip()
                    field_values = field_values.str.replace(r'\s+', ' ', regex=True)  # Multiple spaces to single
                    
                    # For addresses, normalize common abbreviations
                    if 'address' in field.lower():
                        field_values = self._normalize_addresses(field_values)
                    
                    key_parts.append(field_values)
                else:
                    # Field not found, use empty string
                    key_parts.append(pd.Series([''] * len(df_copy)))
                    self.logger.warning(f"Duplicate field '{field}' not found in DataFrame")
            
            # Combine key parts
            if key_parts:
                df_copy['duplicate_key'] = key_parts[0]
                for part in key_parts[1:]:
                    df_copy['duplicate_key'] += '|' + part
            else:
                # Fallback: use row index as key (no deduplication)
                df_copy['duplicate_key'] = df_copy.index.astype(str)
            
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Failed to create duplicate key: {e}")
            df_copy = df.copy()
            df_copy['duplicate_key'] = df_copy.index.astype(str)  # Fallback
            return df_copy
    
    def _normalize_addresses(self, addresses: pd.Series) -> pd.Series:
        """Normalize address strings for better duplicate detection"""
        try:
            # Common address normalizations
            normalized = addresses.copy()
            
            # Normalize common abbreviations
            abbreviations = {
                r'\bstreet\b': 'st',
                r'\bavenue\b': 'ave',
                r'\bboulevard\b': 'blvd',
                r'\bdrive\b': 'dr',
                r'\broad\b': 'rd',
                r'\blane\b': 'ln',
                r'\bcourt\b': 'ct',
                r'\bplace\b': 'pl',
                r'\bcircle\b': 'cir',
                r'\bparkway\b': 'pkwy',
                r'\bnorth\b': 'n',
                r'\bsouth\b': 's',
                r'\beast\b': 'e',
                r'\bwest\b': 'w'
            }
            
            for full_form, abbrev in abbreviations.items():
                normalized = normalized.str.replace(full_form, abbrev, regex=True, case=False)
            
            # Remove common punctuation
            normalized = normalized.str.replace(r'[.,#]', '', regex=True)
            
            return normalized
            
        except Exception as e:
            self.logger.warning(f"Address normalization failed: {e}")
            return addresses
    
    def _sort_by_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort DataFrame by flex score in descending order"""
        try:
            if 'flex_score' in df.columns:
                sorted_df = df.sort_values('flex_score', ascending=False).reset_index(drop=True)
                self.logger.debug(f"Sorted {len(sorted_df)} properties by flex score")
                return sorted_df
            else:
                self.logger.warning("flex_score column not found, returning unsorted DataFrame")
                return df.reset_index(drop=True)
                
        except Exception as e:
            self.logger.error(f"Failed to sort by score: {e}")
            return df.reset_index(drop=True)
    
    def _calculate_final_stats(self, df: pd.DataFrame):
        """Calculate final aggregation statistics"""
        try:
            if df.empty:
                return
            
            # Geographic coverage
            if 'site_address' in df.columns:
                self.stats.unique_addresses = df['site_address'].nunique()
            
            if 'city' in df.columns:
                self.stats.unique_cities = df['city'].nunique()
            
            if 'state' in df.columns:
                self.stats.unique_states = df['state'].nunique()
            
            # Score distribution
            if 'flex_score' in df.columns:
                self.stats.score_distribution = self._calculate_score_distribution(df['flex_score'])
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate final stats: {e}")
    
    def _calculate_score_distribution(self, scores: pd.Series) -> Dict[str, int]:
        """Calculate distribution of flex scores"""
        try:
            distribution = {
                'score_8_to_10': len(scores[(scores >= 8) & (scores <= 10)]),
                'score_6_to_8': len(scores[(scores >= 6) & (scores < 8)]),
                'score_4_to_6': len(scores[(scores >= 4) & (scores < 6)]),
                'score_below_4': len(scores[scores < 4])
            }
            
            return distribution
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate score distribution: {e}")
            return {}
    
    def get_aggregation_stats(self) -> AggregationStats:
        """Get current aggregation statistics"""
        return self.stats
    
    def get_duplicate_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze potential duplicates in the dataset"""
        try:
            if df.empty:
                return {'total_properties': 0, 'potential_duplicates': 0, 'duplicate_groups': []}
            
            # Create duplicate detection key for analysis
            df_with_key = self._create_duplicate_key(df)
            
            # Find duplicate groups
            duplicate_groups = df_with_key.groupby('duplicate_key').size()
            duplicate_groups = duplicate_groups[duplicate_groups > 1]
            
            analysis = {
                'total_properties': len(df),
                'potential_duplicates': duplicate_groups.sum() - len(duplicate_groups),  # Total duplicates minus group count
                'duplicate_groups_count': len(duplicate_groups),
                'largest_duplicate_group': duplicate_groups.max() if not duplicate_groups.empty else 0,
                'duplicate_groups': []
            }
            
            # Get details of duplicate groups (limit to top 10 for performance)
            for key, count in duplicate_groups.head(10).items():
                group_properties = df_with_key[df_with_key['duplicate_key'] == key]
                
                group_info = {
                    'duplicate_key': key,
                    'count': count,
                    'properties': []
                }
                
                for _, prop in group_properties.iterrows():
                    prop_info = {
                        'address': prop.get('site_address', ''),
                        'city': prop.get('city', ''),
                        'state': prop.get('state', ''),
                        'flex_score': prop.get('flex_score', 0),
                        'source_file': prop.get('source_filename', '')
                    }
                    group_info['properties'].append(prop_info)
                
                analysis['duplicate_groups'].append(group_info)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze duplicates: {e}")
            return {'error': str(e)}
    
    def export_aggregated_results(self, df: pd.DataFrame, output_path: str, 
                                 include_metadata: bool = True) -> bool:
        """
        Export aggregated results to Excel file
        
        Args:
            df: Aggregated DataFrame to export
            output_path: Path for output Excel file
            include_metadata: Whether to include aggregation metadata
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            if df.empty:
                self.logger.warning("No data to export")
                return False
            
            # Prepare export DataFrame
            export_df = df.copy()
            
            if include_metadata:
                # Add aggregation metadata columns
                export_df['aggregation_stats'] = f"Total files: {self.stats.successful_files}, " \
                                               f"Duplicates removed: {self.stats.duplicates_removed}"
                export_df['export_date'] = datetime.now().isoformat()
            
            # Export to Excel
            export_df.to_excel(output_path, index=False, engine='openpyxl')
            
            self.logger.info(f"Exported {len(export_df)} aggregated properties to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export aggregated results: {e}")
            return False


# Convenience function for simple aggregation
def aggregate_processing_results(processing_results: List[ProcessingResult],
                               duplicate_fields: List[str] = None,
                               preserve_highest_score: bool = True) -> Optional[pd.DataFrame]:
    """
    Simple function to aggregate processing results
    
    Args:
        processing_results: List of ProcessingResult objects
        duplicate_fields: Fields to use for duplicate detection
        preserve_highest_score: Whether to keep highest scoring duplicate
        
    Returns:
        Aggregated DataFrame or None
    """
    aggregator = ResultAggregator(
        duplicate_fields=duplicate_fields,
        preserve_highest_score=preserve_highest_score
    )
    
    return aggregator.aggregate_results(processing_results)