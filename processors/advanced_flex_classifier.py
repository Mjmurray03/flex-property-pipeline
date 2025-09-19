"""
Advanced Flex Property Classifier
Enhanced version with configurable scoring, batch processing, and advanced analytics
"""

import logging
import time
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Optional, Dict, List, Any, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import threading
from collections import defaultdict

from utils.logger import setup_logging
from utils.flex_config_manager import FlexConfigManager, FlexClassifierConfig
from processors.flex_property_classifier import FlexPropertyClassifier


@dataclass
class ProcessingMetrics:
    """Metrics for tracking processing performance"""
    start_time: float
    end_time: Optional[float] = None
    properties_processed: int = 0
    candidates_found: int = 0
    processing_rate: float = 0.0
    memory_usage_mb: float = 0.0
    errors_encountered: int = 0
    
    def calculate_metrics(self):
        """Calculate derived metrics"""
        if self.end_time and self.start_time:
            duration = self.end_time - self.start_time
            if duration > 0:
                self.processing_rate = self.properties_processed / duration


@dataclass
class GeographicAnalysis:
    """Results of geographic distribution analysis"""
    state_distribution: Dict[str, int]
    city_distribution: Dict[str, int]
    county_distribution: Dict[str, int]
    top_markets: List[Tuple[str, int]]
    geographic_concentration: float


@dataclass
class SizeDistributionAnalysis:
    """Results of size distribution analysis"""
    building_size_distribution: Dict[str, int]
    lot_size_distribution: Dict[str, int]
    size_correlations: Dict[str, float]
    optimal_size_ranges: Dict[str, Tuple[float, float]]


class AdvancedFlexClassifier(FlexPropertyClassifier):
    """
    Advanced Flex Property Classifier with enhanced features:
    - Configurable scoring criteria and weights
    - Batch processing capabilities
    - Progress tracking and performance monitoring
    - Advanced analytics and reporting
    """
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 config: Optional[FlexClassifierConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize Advanced Flex Property Classifier
        
        Args:
            df: DataFrame containing property data
            config: Optional configuration object
            logger: Optional logger instance
        """
        # Load configuration
        if config is None:
            config_manager = FlexConfigManager()
            config = config_manager.load_config()
        
        self.config = config
        
        # Initialize base classifier with basic settings
        super().__init__(df, logger)
        
        # Override with configurable settings
        self._apply_configuration()
        
        # Initialize advanced features
        self.processing_metrics = ProcessingMetrics(start_time=time.time())
        self.batch_results: List[pd.DataFrame] = []
        self.progress_callback: Optional[callable] = None
        self._lock = threading.Lock()
        
        # Analytics storage
        self.geographic_analysis: Optional[GeographicAnalysis] = None
        self.size_distribution_analysis: Optional[SizeDistributionAnalysis] = None
        
        self.logger.info(f"AdvancedFlexClassifier initialized with configuration version {config.version}")
    
    def _apply_configuration(self):
        """Apply configuration settings to classifier"""
        # Update filtering criteria
        criteria = self.config.filtering_criteria
        self.min_building_sqft = criteria.min_building_sqft
        self.min_lot_acres = criteria.min_lot_acres
        self.max_lot_acres = criteria.max_lot_acres
        self.industrial_keywords = criteria.industrial_keywords.copy()
        
        # Update scoring configuration
        self.max_flex_score = self.config.max_flex_score
        
        self.logger.info("Configuration applied to classifier")
    
    def set_progress_callback(self, callback: callable):
        """
        Set callback function for progress updates
        
        Args:
            callback: Function that accepts (current, total, message) parameters
        """
        self.progress_callback = callback
    
    def _update_progress(self, current: int, total: int, message: str = ""):
        """Update progress if callback is set"""
        if self.progress_callback:
            self.progress_callback(current, total, message)
    
    def calculate_flex_score_advanced(self, row: pd.Series) -> Tuple[float, Dict[str, float]]:
        """
        Calculate flex score using configurable weights and return breakdown
        
        Args:
            row: Property data as pandas Series
            
        Returns:
            Tuple of (total_score, score_breakdown)
        """
        try:
            weights = self.config.scoring_weights
            breakdown = {}
            
            # Building size scoring with configurable weight
            building_score = self._score_building_size_advanced(row)
            weighted_building = building_score * weights.building_size_weight
            breakdown['building_size'] = weighted_building
            
            # Property type scoring with configurable weight
            type_score = self._score_property_type_advanced(row)
            weighted_type = type_score * weights.property_type_weight
            breakdown['property_type'] = weighted_type
            
            # Lot size scoring with configurable weight
            lot_score = self._score_lot_size_advanced(row)
            weighted_lot = lot_score * weights.lot_size_weight
            breakdown['lot_size'] = weighted_lot
            
            # Age/condition scoring with configurable weight
            age_score = self._score_age_condition_advanced(row)
            weighted_age = age_score * weights.age_condition_weight
            breakdown['age_condition'] = weighted_age
            
            # Occupancy bonus with configurable weight
            occupancy_score = self._score_occupancy(row)
            weighted_occupancy = occupancy_score * weights.occupancy_weight
            breakdown['occupancy'] = weighted_occupancy
            
            # Calculate total score
            total_score = sum(breakdown.values())
            
            # Cap at maximum score
            final_score = min(total_score, self.max_flex_score)
            breakdown['final_score'] = final_score
            
            return final_score, breakdown
            
        except Exception as e:
            self._handle_error("calculate_flex_score_advanced", e)
            return 0.0, {}
    
    def _score_building_size_advanced(self, row: pd.Series) -> float:
        """Score building size using configurable ranges"""
        try:
            building_col = self._find_column(pd.DataFrame([row]), 
                                           ['building sqft', 'building_sqft', 'sqft', 'square_feet'])
            
            if building_col is None:
                return 0.0
            
            building_size = pd.to_numeric(row.get(building_col), errors='coerce')
            
            if pd.isna(building_size):
                return 0.0
            
            # Use configurable ranges
            ranges = self.config.scoring_weights.building_size_ranges
            
            for range_name, range_config in ranges.items():
                min_size = range_config['min']
                max_size = range_config['max']
                score = range_config['score']
                
                if min_size <= building_size < max_size:
                    return score
            
            return 0.0
                
        except Exception as e:
            self._handle_error("_score_building_size_advanced", e)
            return 0.0
    
    def _score_property_type_advanced(self, row: pd.Series) -> float:
        """Score property type using configurable scores"""
        try:
            type_col = self._find_column(pd.DataFrame([row]), 
                                       ['property type', 'type', 'property_type'])
            
            if type_col is None:
                return 0.0
            
            prop_type = str(row.get(type_col, '')).lower()
            
            # Use configurable property type scores
            type_scores = self.config.scoring_weights.property_type_scores
            
            # Check for exact matches first
            for type_key, score in type_scores.items():
                if type_key.lower() in prop_type:
                    return score
            
            return 0.0
                
        except Exception as e:
            self._handle_error("_score_property_type_advanced", e)
            return 0.0
    
    def _score_lot_size_advanced(self, row: pd.Series) -> float:
        """Score lot size using configurable ranges"""
        try:
            lot_col = self._find_column(pd.DataFrame([row]), 
                                      ['lot size acres', 'lot_size_acres', 'acres', 'lot_acres'])
            
            if lot_col is None:
                return 0.0
            
            lot_size = pd.to_numeric(row.get(lot_col), errors='coerce')
            
            if pd.isna(lot_size):
                return 0.0
            
            # Use configurable ranges
            ranges = self.config.scoring_weights.lot_size_ranges
            
            for range_name, range_config in ranges.items():
                min_size = range_config['min']
                max_size = range_config['max']
                score = range_config['score']
                
                if min_size <= lot_size <= max_size:
                    return score
            
            return 0.0
                
        except Exception as e:
            self._handle_error("_score_lot_size_advanced", e)
            return 0.0
    
    def _score_age_condition_advanced(self, row: pd.Series) -> float:
        """Score age/condition using configurable thresholds"""
        try:
            year_col = self._find_column(pd.DataFrame([row]), 
                                       ['year built', 'year_built', 'built_year'])
            
            if year_col is None:
                return 0.0
            
            year_built = pd.to_numeric(row.get(year_col), errors='coerce')
            
            if pd.isna(year_built):
                return 0.0
            
            # Use configurable thresholds
            thresholds = self.config.scoring_weights.age_thresholds
            
            if year_built >= thresholds['modern']:
                return 1.0  # Modern construction
            elif year_built >= thresholds['decent']:
                return 0.5  # Decent condition
            else:
                return 0.0
                
        except Exception as e:
            self._handle_error("_score_age_condition_advanced", e)
            return 0.0
    
    def classify_flex_properties_batch(self, batch_size: Optional[int] = None) -> pd.DataFrame:
        """
        Classify properties using batch processing for large datasets
        
        Args:
            batch_size: Optional batch size (uses config default if not provided)
            
        Returns:
            DataFrame containing all flex candidates
        """
        try:
            self.processing_metrics.start_time = time.time()
            
            if not self.config.advanced_settings.enable_batch_processing:
                # Fall back to standard processing
                return self.classify_flex_properties()
            
            batch_size = batch_size or self.config.advanced_settings.batch_size
            total_properties = len(self.data)
            
            self.logger.info(f"Starting batch processing: {total_properties} properties in batches of {batch_size}")
            
            # Split data into batches
            batches = []
            for i in range(0, total_properties, batch_size):
                batch_data = self.data.iloc[i:i + batch_size].copy()
                batches.append(batch_data)
            
            all_candidates = []
            
            # Process batches
            for i, batch_data in enumerate(batches):
                self._update_progress(i, len(batches), f"Processing batch {i+1}/{len(batches)}")
                
                # Create temporary classifier for this batch
                batch_classifier = FlexPropertyClassifier(batch_data, self.logger)
                
                # Apply same configuration
                batch_classifier.min_building_sqft = self.min_building_sqft
                batch_classifier.min_lot_acres = self.min_lot_acres
                batch_classifier.max_lot_acres = self.max_lot_acres
                batch_classifier.industrial_keywords = self.industrial_keywords
                
                # Process batch
                batch_candidates = batch_classifier.classify_flex_properties()
                
                if len(batch_candidates) > 0:
                    # Add batch identifier
                    batch_candidates['batch_id'] = i
                    all_candidates.append(batch_candidates)
                
                self.processing_metrics.properties_processed += len(batch_data)
            
            # Combine all results
            if all_candidates:
                self.flex_candidates = pd.concat(all_candidates, ignore_index=True)
                # Remove batch_id column
                self.flex_candidates = self.flex_candidates.drop('batch_id', axis=1)
            else:
                self.flex_candidates = pd.DataFrame()
            
            self.processing_metrics.end_time = time.time()
            self.processing_metrics.candidates_found = len(self.flex_candidates)
            self.processing_metrics.calculate_metrics()
            
            self.logger.info(f"Batch processing complete: {len(self.flex_candidates)} candidates found")
            self.logger.info(f"Processing rate: {self.processing_metrics.processing_rate:.1f} properties/second")
            
            return self.flex_candidates
            
        except Exception as e:
            self._handle_error("classify_flex_properties_batch", e, continue_processing=False)
    
    def process_multiple_files(self, file_paths: List[Union[str, Path]]) -> Dict[str, pd.DataFrame]:
        """
        Process multiple Excel files and return combined results
        
        Args:
            file_paths: List of paths to Excel files
            
        Returns:
            Dictionary mapping file paths to their results
        """
        try:
            results = {}
            total_files = len(file_paths)
            
            self.logger.info(f"Processing {total_files} files")
            
            if self.config.advanced_settings.parallel_processing:
                # Parallel processing
                max_workers = min(self.config.advanced_settings.max_workers, total_files)
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_file = {
                        executor.submit(self._process_single_file, file_path): file_path 
                        for file_path in file_paths
                    }
                    
                    for i, future in enumerate(future_to_file):
                        self._update_progress(i, total_files, f"Processing file {i+1}/{total_files}")
                        
                        file_path = future_to_file[future]
                        try:
                            result = future.result()
                            results[str(file_path)] = result
                        except Exception as e:
                            self.logger.error(f"Error processing {file_path}: {e}")
                            results[str(file_path)] = pd.DataFrame()
            else:
                # Sequential processing
                for i, file_path in enumerate(file_paths):
                    self._update_progress(i, total_files, f"Processing {Path(file_path).name}")
                    
                    try:
                        result = self._process_single_file(file_path)
                        results[str(file_path)] = result
                    except Exception as e:
                        self.logger.error(f"Error processing {file_path}: {e}")
                        results[str(file_path)] = pd.DataFrame()
            
            self.logger.info(f"Completed processing {total_files} files")
            return results
            
        except Exception as e:
            self._handle_error("process_multiple_files", e, continue_processing=False)
    
    def _process_single_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Process a single Excel file"""
        try:
            # Load data
            data = pd.read_excel(file_path)
            
            # Create classifier for this file
            classifier = AdvancedFlexClassifier(data, self.config, self.logger)
            
            # Process
            candidates = classifier.classify_flex_properties_batch()
            
            # Add source file information
            if len(candidates) > 0:
                candidates['source_file'] = str(file_path)
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return pd.DataFrame()
    
    def perform_geographic_analysis(self) -> GeographicAnalysis:
        """
        Perform geographic distribution analysis of flex candidates
        
        Returns:
            GeographicAnalysis results
        """
        try:
            if self.flex_candidates is None or len(self.flex_candidates) == 0:
                raise ValueError("No flex candidates available. Run classification first.")
            
            self.logger.info("Performing geographic analysis")
            
            # Initialize distributions
            state_dist = defaultdict(int)
            city_dist = defaultdict(int)
            county_dist = defaultdict(int)
            
            # Find geographic columns
            state_col = self._find_column(self.flex_candidates, ['state', 'st'])
            city_col = self._find_column(self.flex_candidates, ['city'])
            county_col = self._find_column(self.flex_candidates, ['county'])
            
            # Count distributions
            if state_col:
                state_counts = self.flex_candidates[state_col].value_counts()
                state_dist.update(state_counts.to_dict())
            
            if city_col:
                city_counts = self.flex_candidates[city_col].value_counts()
                city_dist.update(city_counts.to_dict())
            
            if county_col:
                county_counts = self.flex_candidates[county_col].value_counts()
                county_dist.update(county_counts.to_dict())
            
            # Calculate top markets (by state)
            top_markets = sorted(state_dist.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Calculate geographic concentration (Herfindahl index for states)
            total_properties = sum(state_dist.values())
            if total_properties > 0:
                concentration = sum((count / total_properties) ** 2 for count in state_dist.values())
            else:
                concentration = 0.0
            
            analysis = GeographicAnalysis(
                state_distribution=dict(state_dist),
                city_distribution=dict(city_dist),
                county_distribution=dict(county_dist),
                top_markets=top_markets,
                geographic_concentration=concentration
            )
            
            self.geographic_analysis = analysis
            self.logger.info(f"Geographic analysis complete: {len(state_dist)} states, {len(city_dist)} cities")
            
            return analysis
            
        except Exception as e:
            self._handle_error("perform_geographic_analysis", e, continue_processing=False)
    
    def perform_size_distribution_analysis(self) -> SizeDistributionAnalysis:
        """
        Perform size distribution analysis of flex candidates
        
        Returns:
            SizeDistributionAnalysis results
        """
        try:
            if self.flex_candidates is None or len(self.flex_candidates) == 0:
                raise ValueError("No flex candidates available. Run classification first.")
            
            self.logger.info("Performing size distribution analysis")
            
            # Find size columns
            building_col = self._find_column(self.flex_candidates, ['building sqft', 'building_sqft', 'sqft'])
            lot_col = self._find_column(self.flex_candidates, ['lot size acres', 'lot_size_acres', 'acres'])
            
            building_dist = {}
            lot_dist = {}
            correlations = {}
            optimal_ranges = {}
            
            # Building size distribution
            if building_col:
                building_sizes = pd.to_numeric(self.flex_candidates[building_col], errors='coerce').dropna()
                
                # Create size buckets
                building_dist = {
                    '20k-50k': len(building_sizes[(building_sizes >= 20000) & (building_sizes < 50000)]),
                    '50k-100k': len(building_sizes[(building_sizes >= 50000) & (building_sizes < 100000)]),
                    '100k-200k': len(building_sizes[(building_sizes >= 100000) & (building_sizes < 200000)]),
                    '200k+': len(building_sizes[building_sizes >= 200000])
                }
                
                # Calculate optimal range (highest scoring properties)
                if 'flex_score' in self.flex_candidates.columns:
                    high_scoring = self.flex_candidates[self.flex_candidates['flex_score'] >= 7]
                    if len(high_scoring) > 0:
                        high_scoring_sizes = pd.to_numeric(high_scoring[building_col], errors='coerce').dropna()
                        if len(high_scoring_sizes) > 0:
                            optimal_ranges['building_size'] = (
                                float(high_scoring_sizes.quantile(0.25)),
                                float(high_scoring_sizes.quantile(0.75))
                            )
            
            # Lot size distribution
            if lot_col:
                lot_sizes = pd.to_numeric(self.flex_candidates[lot_col], errors='coerce').dropna()
                
                # Create size buckets
                lot_dist = {
                    '0.5-1 acres': len(lot_sizes[(lot_sizes >= 0.5) & (lot_sizes < 1.0)]),
                    '1-5 acres': len(lot_sizes[(lot_sizes >= 1.0) & (lot_sizes < 5.0)]),
                    '5-10 acres': len(lot_sizes[(lot_sizes >= 5.0) & (lot_sizes < 10.0)]),
                    '10-20 acres': len(lot_sizes[(lot_sizes >= 10.0) & (lot_sizes <= 20.0)]),
                    '20+ acres': len(lot_sizes[lot_sizes > 20.0])
                }
                
                # Calculate optimal range
                if 'flex_score' in self.flex_candidates.columns:
                    high_scoring = self.flex_candidates[self.flex_candidates['flex_score'] >= 7]
                    if len(high_scoring) > 0:
                        high_scoring_lots = pd.to_numeric(high_scoring[lot_col], errors='coerce').dropna()
                        if len(high_scoring_lots) > 0:
                            optimal_ranges['lot_size'] = (
                                float(high_scoring_lots.quantile(0.25)),
                                float(high_scoring_lots.quantile(0.75))
                            )
            
            # Calculate correlations
            if building_col and lot_col and 'flex_score' in self.flex_candidates.columns:
                numeric_data = self.flex_candidates[[building_col, lot_col, 'flex_score']].apply(
                    pd.to_numeric, errors='coerce'
                ).dropna()
                
                if len(numeric_data) > 1:
                    corr_matrix = numeric_data.corr()
                    correlations = {
                        'building_size_vs_score': float(corr_matrix.loc[building_col, 'flex_score']),
                        'lot_size_vs_score': float(corr_matrix.loc[lot_col, 'flex_score']),
                        'building_vs_lot_size': float(corr_matrix.loc[building_col, lot_col])
                    }
            
            analysis = SizeDistributionAnalysis(
                building_size_distribution=building_dist,
                lot_size_distribution=lot_dist,
                size_correlations=correlations,
                optimal_size_ranges=optimal_ranges
            )
            
            self.size_distribution_analysis = analysis
            self.logger.info("Size distribution analysis complete")
            
            return analysis
            
        except Exception as e:
            self._handle_error("perform_size_distribution_analysis", e, continue_processing=False)
    
    def export_advanced_results(self, 
                              output_dir: Optional[Path] = None,
                              include_analytics: bool = True) -> Dict[str, str]:
        """
        Export results in multiple formats with advanced analytics
        
        Args:
            output_dir: Optional output directory
            include_analytics: Whether to include analytics reports
            
        Returns:
            Dictionary mapping format to file path
        """
        try:
            if self.flex_candidates is None or len(self.flex_candidates) == 0:
                raise ValueError("No flex candidates available for export")
            
            if output_dir is None:
                output_dir = Path('data/exports/advanced')
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            exported_files = {}
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            
            # Export in configured formats
            for format_type in self.config.advanced_settings.export_formats:
                filename = f"flex_candidates_advanced_{timestamp}.{format_type}"
                file_path = output_dir / filename
                
                if format_type == 'xlsx':
                    self.flex_candidates.to_excel(file_path, index=False, engine='openpyxl')
                elif format_type == 'csv':
                    self.flex_candidates.to_csv(file_path, index=False)
                elif format_type == 'json':
                    self.flex_candidates.to_json(file_path, orient='records', indent=2)
                elif format_type == 'parquet':
                    self.flex_candidates.to_parquet(file_path, index=False)
                
                exported_files[format_type] = str(file_path)
                self.logger.info(f"Exported {format_type.upper()} to {file_path}")
            
            # Export analytics if requested
            if include_analytics:
                analytics_dir = output_dir / 'analytics'
                analytics_dir.mkdir(exist_ok=True)
                
                # Export geographic analysis
                if self.geographic_analysis:
                    geo_file = analytics_dir / f"geographic_analysis_{timestamp}.json"
                    with open(geo_file, 'w') as f:
                        json.dump(self.geographic_analysis.__dict__, f, indent=2)
                    exported_files['geographic_analysis'] = str(geo_file)
                
                # Export size distribution analysis
                if self.size_distribution_analysis:
                    size_file = analytics_dir / f"size_analysis_{timestamp}.json"
                    with open(size_file, 'w') as f:
                        json.dump(self.size_distribution_analysis.__dict__, f, indent=2)
                    exported_files['size_analysis'] = str(size_file)
                
                # Export processing metrics
                metrics_file = analytics_dir / f"processing_metrics_{timestamp}.json"
                with open(metrics_file, 'w') as f:
                    json.dump(self.processing_metrics.__dict__, f, indent=2, default=str)
                exported_files['processing_metrics'] = str(metrics_file)
            
            self.logger.info(f"Advanced export complete: {len(exported_files)} files created")
            return exported_files
            
        except Exception as e:
            self._handle_error("export_advanced_results", e, continue_processing=False)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Returns:
            Dictionary containing performance metrics and recommendations
        """
        try:
            report = {
                'processing_metrics': self.processing_metrics.__dict__,
                'configuration_summary': {
                    'batch_processing_enabled': self.config.advanced_settings.enable_batch_processing,
                    'batch_size': self.config.advanced_settings.batch_size,
                    'parallel_processing': self.config.advanced_settings.parallel_processing,
                    'max_workers': self.config.advanced_settings.max_workers
                },
                'data_summary': {
                    'total_properties': len(self.data),
                    'flex_candidates': len(self.flex_candidates) if self.flex_candidates is not None else 0,
                    'conversion_rate': 0.0
                },
                'recommendations': []
            }
            
            # Calculate conversion rate
            if len(self.data) > 0 and self.flex_candidates is not None:
                report['data_summary']['conversion_rate'] = len(self.flex_candidates) / len(self.data) * 100
            
            # Generate recommendations
            if self.processing_metrics.processing_rate > 0:
                if self.processing_metrics.processing_rate < 100:  # Less than 100 properties/second
                    report['recommendations'].append(
                        "Consider enabling batch processing or increasing batch size for better performance"
                    )
                
                if len(self.data) > 10000 and not self.config.advanced_settings.parallel_processing:
                    report['recommendations'].append(
                        "Enable parallel processing for large datasets to improve performance"
                    )
            
            if report['data_summary']['conversion_rate'] < 1.0:
                report['recommendations'].append(
                    "Low conversion rate detected. Consider adjusting filtering criteria to be less restrictive"
                )
            elif report['data_summary']['conversion_rate'] > 20.0:
                report['recommendations'].append(
                    "High conversion rate detected. Consider tightening filtering criteria for better selectivity"
                )
            
            return report
            
        except Exception as e:
            self._handle_error("get_performance_report", e)
            return {'error': str(e)}


def create_advanced_classifier_from_config(df: pd.DataFrame, 
                                         config_path: Optional[Path] = None) -> AdvancedFlexClassifier:
    """
    Create AdvancedFlexClassifier from configuration file
    
    Args:
        df: DataFrame containing property data
        config_path: Optional path to configuration file
        
    Returns:
        Configured AdvancedFlexClassifier instance
    """
    config_manager = FlexConfigManager(config_path)
    config = config_manager.load_config()
    
    return AdvancedFlexClassifier(df, config)


if __name__ == '__main__':
    # Example usage
    import numpy as np
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Property Type': np.random.choice(['Industrial', 'Warehouse', 'Flex', 'Office'], 100),
        'Building SqFt': np.random.randint(15000, 200000, 100),
        'Lot Size Acres': np.random.uniform(0.3, 25.0, 100),
        'Year Built': np.random.randint(1970, 2020, 100),
        'Occupancy': np.random.uniform(60, 100, 100),
        'City': np.random.choice(['Dallas', 'Houston', 'Austin'], 100),
        'State': ['TX'] * 100
    })
    
    # Create advanced classifier
    classifier = create_advanced_classifier_from_config(sample_data)
    
    # Set progress callback
    def progress_callback(current, total, message):
        print(f"Progress: {current}/{total} - {message}")
    
    classifier.set_progress_callback(progress_callback)
    
    # Process with batch processing
    candidates = classifier.classify_flex_properties_batch()
    print(f"Found {len(candidates)} flex candidates")
    
    # Perform analytics
    if len(candidates) > 0:
        geo_analysis = classifier.perform_geographic_analysis()
        size_analysis = classifier.perform_size_distribution_analysis()
        
        # Export results
        exported_files = classifier.export_advanced_results()
        print(f"Exported files: {list(exported_files.keys())}")
        
        # Get performance report
        performance = classifier.get_performance_report()
        print(f"Processing rate: {performance['processing_metrics']['processing_rate']:.1f} properties/second")