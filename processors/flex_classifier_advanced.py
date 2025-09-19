"""
Advanced Features and Optimization for Flex Property Classifier
Extends the base classifier with configurable scoring, batch processing, and advanced analytics
"""

import logging
import time
import json
from typing import Optional, Dict, List, Any, Union, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from tqdm import tqdm

from processors.flex_property_classifier import FlexPropertyClassifier
from utils.logger import setup_logging


@dataclass
class ScoringConfiguration:
    """Configuration for flexible scoring criteria and weights"""
    
    # Building size scoring ranges and points
    building_size_ranges: Dict[str, Dict[str, Union[int, float]]] = None
    
    # Property type scoring weights
    property_type_weights: Dict[str, float] = None
    
    # Lot size scoring ranges and points
    lot_size_ranges: Dict[str, Dict[str, Union[float, int]]] = None
    
    # Age/condition scoring thresholds
    age_thresholds: Dict[str, Union[int, float]] = None
    
    # Occupancy bonus configuration
    occupancy_config: Dict[str, float] = None
    
    # Overall scoring weights for each factor
    factor_weights: Dict[str, float] = None
    
    # Maximum possible score
    max_score: float = 10.0
    
    def __post_init__(self):
        """Initialize default values if not provided"""
        if self.building_size_ranges is None:
            self.building_size_ranges = {
                "ideal": {"min": 20000, "max": 50000, "points": 3.0},
                "good": {"min": 50000, "max": 100000, "points": 2.0},
                "acceptable": {"min": 100000, "max": 200000, "points": 1.0}
            }
        
        if self.property_type_weights is None:
            self.property_type_weights = {
                "flex": 3.0,
                "warehouse": 2.5,
                "distribution": 2.5,
                "light industrial": 2.0,
                "industrial": 1.5,
                "manufacturing": 1.0,
                "storage": 1.0,
                "logistics": 1.0
            }
        
        if self.lot_size_ranges is None:
            self.lot_size_ranges = {
                "ideal": {"min": 1.0, "max": 5.0, "points": 2.0},
                "good": {"min": 5.0, "max": 10.0, "points": 1.5},
                "acceptable_small": {"min": 0.5, "max": 1.0, "points": 1.0},
                "acceptable_large": {"min": 10.0, "max": 20.0, "points": 1.0}
            }
        
        if self.age_thresholds is None:
            self.age_thresholds = {
                "modern": {"threshold": 1990, "points": 1.0},
                "decent": {"threshold": 1980, "points": 0.5}
            }
        
        if self.occupancy_config is None:
            self.occupancy_config = {
                "bonus_threshold": 100.0,
                "bonus_points": 1.0
            }
        
        if self.factor_weights is None:
            self.factor_weights = {
                "building_size": 1.0,
                "property_type": 1.0,
                "lot_size": 1.0,
                "age_condition": 1.0,
                "occupancy": 1.0
            }
    
    def save_to_file(self, file_path: str) -> None:
        """Save configuration to JSON file"""
        config_dict = asdict(self)
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'ScoringConfiguration':
        """Load configuration from JSON file"""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


@dataclass
class ProcessingMetrics:
    """Metrics for tracking processing performance"""
    
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_properties: int = 0
    total_candidates: int = 0
    processing_time: float = 0.0
    average_time_per_file: float = 0.0
    properties_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    
    def update_timing(self, start_time: float) -> None:
        """Update timing metrics"""
        self.processing_time = time.time() - start_time
        if self.processed_files > 0:
            self.average_time_per_file = self.processing_time / self.processed_files
        if self.processing_time > 0:
            self.properties_per_second = self.total_properties / self.processing_time


class AdvancedFlexClassifier(FlexPropertyClassifier):
    """
    Advanced Flex Property Classifier with configurable scoring and batch processing
    """
    
    def __init__(self, df: pd.DataFrame = None, 
                 scoring_config: Optional[ScoringConfiguration] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize Advanced Flex Classifier
        
        Args:
            df: DataFrame containing property data (optional for batch processing)
            scoring_config: Custom scoring configuration
            logger: Optional logger instance
        """
        # Initialize base classifier if DataFrame provided
        if df is not None:
            super().__init__(df, logger)
        else:
            # Initialize minimal attributes for batch processing
            self.data = None
            self.flex_candidates = None
            if logger is None:
                self.logger = setup_logging(name='advanced_flex_classifier', level='INFO')
            else:
                self.logger = logger
        
        # Set up advanced features
        self.scoring_config = scoring_config or ScoringConfiguration()
        self.processing_metrics = ProcessingMetrics()
        self.progress_callback: Optional[Callable] = None
        
        # Batch processing attributes
        self.batch_results: List[pd.DataFrame] = []
        self.batch_files: List[str] = []
        self.failed_files: List[Dict[str, str]] = []
        
        self.logger.info("Advanced Flex Property Classifier initialized")
    
    def set_scoring_configuration(self, config: ScoringConfiguration) -> None:
        """Update scoring configuration"""
        self.scoring_config = config
        self.logger.info("Scoring configuration updated")
    
    def set_progress_callback(self, callback: Callable[[str, float], None]) -> None:
        """Set callback function for progress updates"""
        self.progress_callback = callback
    
    def calculate_flex_score_advanced(self, row: pd.Series) -> Dict[str, Any]:
        """
        Calculate flex score using configurable criteria with detailed breakdown
        
        Args:
            row: Property data as pandas Series
            
        Returns:
            Dictionary with score and breakdown
        """
        try:
            scores = {}
            
            # Building size scoring with configurable ranges
            building_score = self._score_building_size_advanced(row)
            scores['building_size'] = building_score * self.scoring_config.factor_weights['building_size']
            
            # Property type scoring with configurable weights
            type_score = self._score_property_type_advanced(row)
            scores['property_type'] = type_score * self.scoring_config.factor_weights['property_type']
            
            # Lot size scoring with configurable ranges
            lot_score = self._score_lot_size_advanced(row)
            scores['lot_size'] = lot_score * self.scoring_config.factor_weights['lot_size']
            
            # Age/condition scoring with configurable thresholds
            age_score = self._score_age_condition_advanced(row)
            scores['age_condition'] = age_score * self.scoring_config.factor_weights['age_condition']
            
            # Occupancy bonus with configurable threshold
            occupancy_score = self._score_occupancy_advanced(row)
            scores['occupancy'] = occupancy_score * self.scoring_config.factor_weights['occupancy']
            
            # Calculate total score
            total_score = sum(scores.values())
            final_score = min(total_score, self.scoring_config.max_score)
            
            return {
                'total_score': final_score,
                'score_breakdown': scores,
                'max_possible': self.scoring_config.max_score
            }
            
        except Exception as e:
            self._handle_error("calculate_flex_score_advanced", e)
            return {
                'total_score': 0.0,
                'score_breakdown': {},
                'max_possible': self.scoring_config.max_score
            }
    
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
            
            # Check against configurable ranges
            for range_name, range_config in self.scoring_config.building_size_ranges.items():
                min_size = range_config['min']
                max_size = range_config.get('max', float('inf'))
                
                if min_size <= building_size < max_size:
                    return range_config['points']
            
            return 0.0
            
        except Exception as e:
            self._handle_error("_score_building_size_advanced", e)
            return 0.0
    
    def _score_property_type_advanced(self, row: pd.Series) -> float:
        """Score property type using configurable weights"""
        try:
            type_col = self._find_column(pd.DataFrame([row]), 
                                       ['property type', 'type', 'property_type'])
            
            if type_col is None:
                return 0.0
            
            prop_type = str(row.get(type_col, '')).lower()
            
            # Check against configurable weights
            for type_keyword, weight in self.scoring_config.property_type_weights.items():
                if type_keyword.lower() in prop_type:
                    return weight
            
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
            
            # Check against configurable ranges
            for range_name, range_config in self.scoring_config.lot_size_ranges.items():
                min_size = range_config['min']
                max_size = range_config.get('max', float('inf'))
                
                if min_size <= lot_size <= max_size:
                    return range_config['points']
            
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
            
            # Check against configurable thresholds (sorted by threshold descending)
            sorted_thresholds = sorted(
                self.scoring_config.age_thresholds.items(),
                key=lambda x: x[1]['threshold'],
                reverse=True
            )
            
            for threshold_name, threshold_config in sorted_thresholds:
                if year_built >= threshold_config['threshold']:
                    return threshold_config['points']
            
            return 0.0
            
        except Exception as e:
            self._handle_error("_score_age_condition_advanced", e)
            return 0.0
    
    def _score_occupancy_advanced(self, row: pd.Series) -> float:
        """Score occupancy using configurable bonus"""
        try:
            occupancy_col = self._find_column(pd.DataFrame([row]), 
                                            ['occupancy', 'occupancy_rate', 'occupied'])
            
            if occupancy_col is None:
                return 0.0
            
            occupancy = pd.to_numeric(row.get(occupancy_col), errors='coerce')
            
            if pd.isna(occupancy):
                return 0.0
            
            # Convert percentage if needed (assume values > 1 are percentages)
            if occupancy > 1:
                occupancy = occupancy / 100
            
            # Check against configurable threshold
            threshold = self.scoring_config.occupancy_config['bonus_threshold'] / 100
            if occupancy < threshold:
                return self.scoring_config.occupancy_config['bonus_points']
            
            return 0.0
            
        except Exception as e:
            self._handle_error("_score_occupancy_advanced", e)
            return 0.0
    
    def process_multiple_files(self, file_paths: List[str], 
                             max_workers: int = 4,
                             show_progress: bool = True) -> Dict[str, Any]:
        """
        Process multiple Excel files in batch with progress tracking
        
        Args:
            file_paths: List of Excel file paths to process
            max_workers: Maximum number of concurrent workers
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with aggregated results and metrics
        """
        start_time = time.time()
        self.processing_metrics = ProcessingMetrics()
        self.processing_metrics.total_files = len(file_paths)
        
        self.batch_results = []
        self.failed_files = []
        
        self.logger.info(f"Starting batch processing of {len(file_paths)} files with {max_workers} workers")
        
        # Progress tracking
        if show_progress:
            progress_bar = tqdm(total=len(file_paths), desc="Processing files")
        
        # Process files concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_single_file, file_path): file_path 
                for file_path in file_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                
                try:
                    result = future.result()
                    if result is not None:
                        self.batch_results.append(result)
                        self.processing_metrics.processed_files += 1
                        self.processing_metrics.total_properties += len(result)
                        
                        # Count candidates (properties with flex_score > 0)
                        if 'flex_score' in result.columns:
                            candidates = len(result[result['flex_score'] > 0])
                            self.processing_metrics.total_candidates += candidates
                    else:
                        self.processing_metrics.failed_files += 1
                        
                except Exception as e:
                    self.failed_files.append({
                        'file_path': file_path,
                        'error': str(e)
                    })
                    self.processing_metrics.failed_files += 1
                    self.logger.error(f"Failed to process {file_path}: {e}")
                
                # Update progress
                if show_progress:
                    progress_bar.update(1)
                
                if self.progress_callback:
                    progress = (self.processing_metrics.processed_files + self.processing_metrics.failed_files) / len(file_paths)
                    self.progress_callback(f"Processed {file_path}", progress)
        
        if show_progress:
            progress_bar.close()
        
        # Update final metrics
        self.processing_metrics.update_timing(start_time)
        
        # Aggregate results
        aggregated_results = self._aggregate_batch_results()
        
        self.logger.info(f"Batch processing complete: {self.processing_metrics.processed_files} successful, "
                        f"{self.processing_metrics.failed_files} failed")
        
        return {
            'aggregated_results': aggregated_results,
            'processing_metrics': asdict(self.processing_metrics),
            'failed_files': self.failed_files
        }
    
    def _process_single_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Process a single Excel file"""
        try:
            self.logger.debug(f"Processing file: {file_path}")
            
            # Load data
            df = pd.read_excel(file_path)
            
            if df.empty:
                self.logger.warning(f"Empty file: {file_path}")
                return None
            
            # Create classifier instance for this file
            classifier = FlexPropertyClassifier(df, self.logger)
            
            # Override scoring method with advanced version
            classifier.calculate_flex_score = lambda row: self.calculate_flex_score_advanced(row)['total_score']
            
            # Classify properties
            candidates = classifier.classify_flex_properties()
            
            if len(candidates) == 0:
                self.logger.info(f"No candidates found in {file_path}")
                return pd.DataFrame()
            
            # Add source file information
            candidates['source_file'] = file_path
            candidates['processed_timestamp'] = pd.Timestamp.now()
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def _aggregate_batch_results(self) -> pd.DataFrame:
        """Aggregate results from multiple files"""
        if not self.batch_results:
            return pd.DataFrame()
        
        # Combine all results
        combined_df = pd.concat(self.batch_results, ignore_index=True)
        
        # Remove duplicates based on address, city, state
        # Keep the one with highest flex score
        address_cols = []
        for col in ['address', 'city', 'state']:
            found_col = self._find_column_in_df(combined_df, [col])
            if found_col:
                address_cols.append(found_col)
        
        if address_cols and 'flex_score' in combined_df.columns:
            # Sort by flex score descending, then drop duplicates
            combined_df = combined_df.sort_values('flex_score', ascending=False)
            combined_df = combined_df.drop_duplicates(subset=address_cols, keep='first')
            
            self.logger.info(f"Removed duplicates, {len(combined_df)} unique properties remain")
        
        # Sort final results by flex score
        if 'flex_score' in combined_df.columns:
            combined_df = combined_df.sort_values('flex_score', ascending=False)
        
        return combined_df
    
    def _find_column_in_df(self, df: pd.DataFrame, search_terms: List[str]) -> Optional[str]:
        """Find column in DataFrame (helper method)"""
        for term in search_terms:
            for col in df.columns:
                if term.lower() in col.lower():
                    return col
        return None
    
    def generate_advanced_analytics(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate advanced analytics including geographic and size distribution analysis
        
        Args:
            results_df: DataFrame with classification results
            
        Returns:
            Dictionary with advanced analytics
        """
        try:
            analytics = {
                'overview': {},
                'geographic_analysis': {},
                'size_distribution': {},
                'score_analysis': {},
                'property_type_analysis': {},
                'temporal_analysis': {},
                'market_insights': {}
            }
            
            if results_df.empty:
                return analytics
            
            # Overview statistics
            analytics['overview'] = {
                'total_properties': len(results_df),
                'total_candidates': len(results_df[results_df.get('flex_score', 0) > 0]) if 'flex_score' in results_df.columns else 0,
                'average_score': float(results_df['flex_score'].mean()) if 'flex_score' in results_df.columns else 0,
                'score_std': float(results_df['flex_score'].std()) if 'flex_score' in results_df.columns else 0
            }
            
            # Geographic analysis
            analytics['geographic_analysis'] = self._analyze_geographic_distribution(results_df)
            
            # Size distribution analysis
            analytics['size_distribution'] = self._analyze_size_distribution(results_df)
            
            # Score analysis
            if 'flex_score' in results_df.columns:
                analytics['score_analysis'] = self._analyze_score_distribution(results_df)
            
            # Property type analysis
            analytics['property_type_analysis'] = self._analyze_property_types(results_df)
            
            # Temporal analysis (if date columns available)
            analytics['temporal_analysis'] = self._analyze_temporal_patterns(results_df)
            
            # Market insights
            analytics['market_insights'] = self._generate_market_insights(results_df)
            
            return analytics
            
        except Exception as e:
            self._handle_error("generate_advanced_analytics", e)
            return {}
    
    def _analyze_geographic_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze geographic distribution of properties"""
        geo_analysis = {}
        
        # State distribution
        state_col = self._find_column_in_df(df, ['state'])
        if state_col:
            state_dist = df[state_col].value_counts().to_dict()
            geo_analysis['by_state'] = state_dist
            
            # Top states by candidate count
            if 'flex_score' in df.columns:
                candidates_by_state = df[df['flex_score'] > 0][state_col].value_counts().to_dict()
                geo_analysis['candidates_by_state'] = candidates_by_state
        
        # City distribution
        city_col = self._find_column_in_df(df, ['city'])
        if city_col:
            city_dist = df[city_col].value_counts().head(20).to_dict()
            geo_analysis['top_cities'] = city_dist
        
        # County distribution
        county_col = self._find_column_in_df(df, ['county'])
        if county_col:
            county_dist = df[county_col].value_counts().head(15).to_dict()
            geo_analysis['top_counties'] = county_dist
        
        return geo_analysis
    
    def _analyze_size_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze building and lot size distributions"""
        size_analysis = {}
        
        # Building size analysis
        building_col = self._find_column_in_df(df, ['building sqft', 'building_sqft', 'sqft'])
        if building_col:
            building_sizes = pd.to_numeric(df[building_col], errors='coerce').dropna()
            
            if len(building_sizes) > 0:
                size_analysis['building_size'] = {
                    'mean': float(building_sizes.mean()),
                    'median': float(building_sizes.median()),
                    'std': float(building_sizes.std()),
                    'min': float(building_sizes.min()),
                    'max': float(building_sizes.max()),
                    'quartiles': {
                        'q25': float(building_sizes.quantile(0.25)),
                        'q75': float(building_sizes.quantile(0.75))
                    }
                }
                
                # Size categories
                size_categories = {
                    'small_20k_50k': len(building_sizes[(building_sizes >= 20000) & (building_sizes < 50000)]),
                    'medium_50k_100k': len(building_sizes[(building_sizes >= 50000) & (building_sizes < 100000)]),
                    'large_100k_200k': len(building_sizes[(building_sizes >= 100000) & (building_sizes < 200000)]),
                    'extra_large_200k_plus': len(building_sizes[building_sizes >= 200000])
                }
                size_analysis['building_size']['categories'] = size_categories
        
        # Lot size analysis
        lot_col = self._find_column_in_df(df, ['lot size acres', 'lot_size_acres', 'acres'])
        if lot_col:
            lot_sizes = pd.to_numeric(df[lot_col], errors='coerce').dropna()
            
            if len(lot_sizes) > 0:
                size_analysis['lot_size'] = {
                    'mean': float(lot_sizes.mean()),
                    'median': float(lot_sizes.median()),
                    'std': float(lot_sizes.std()),
                    'min': float(lot_sizes.min()),
                    'max': float(lot_sizes.max())
                }
                
                # Lot size categories
                lot_categories = {
                    'small_0.5_1': len(lot_sizes[(lot_sizes >= 0.5) & (lot_sizes < 1.0)]),
                    'ideal_1_5': len(lot_sizes[(lot_sizes >= 1.0) & (lot_sizes <= 5.0)]),
                    'good_5_10': len(lot_sizes[(lot_sizes > 5.0) & (lot_sizes <= 10.0)]),
                    'large_10_20': len(lot_sizes[(lot_sizes > 10.0) & (lot_sizes <= 20.0)]),
                    'extra_large_20_plus': len(lot_sizes[lot_sizes > 20.0])
                }
                size_analysis['lot_size']['categories'] = lot_categories
        
        return size_analysis
    
    def _analyze_score_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze flex score distribution"""
        scores = df['flex_score'].dropna()
        
        if len(scores) == 0:
            return {}
        
        score_analysis = {
            'distribution': {
                'mean': float(scores.mean()),
                'median': float(scores.median()),
                'std': float(scores.std()),
                'min': float(scores.min()),
                'max': float(scores.max())
            },
            'score_ranges': {
                'excellent_8_10': len(scores[scores >= 8.0]),
                'good_6_8': len(scores[(scores >= 6.0) & (scores < 8.0)]),
                'fair_4_6': len(scores[(scores >= 4.0) & (scores < 6.0)]),
                'poor_2_4': len(scores[(scores >= 2.0) & (scores < 4.0)]),
                'very_poor_0_2': len(scores[scores < 2.0])
            },
            'percentiles': {
                'p10': float(scores.quantile(0.1)),
                'p25': float(scores.quantile(0.25)),
                'p50': float(scores.quantile(0.5)),
                'p75': float(scores.quantile(0.75)),
                'p90': float(scores.quantile(0.9))
            }
        }
        
        return score_analysis
    
    def _analyze_property_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze property type distribution"""
        type_col = self._find_column_in_df(df, ['property type', 'type', 'property_type'])
        
        if not type_col:
            return {}
        
        type_analysis = {
            'distribution': df[type_col].value_counts().to_dict(),
            'industrial_types': {}
        }
        
        # Focus on industrial types
        industrial_keywords = ['industrial', 'warehouse', 'distribution', 'flex', 'manufacturing', 'storage', 'logistics']
        
        for keyword in industrial_keywords:
            matching_types = df[df[type_col].str.contains(keyword, case=False, na=False)]
            if len(matching_types) > 0:
                type_analysis['industrial_types'][keyword] = {
                    'count': len(matching_types),
                    'avg_score': float(matching_types['flex_score'].mean()) if 'flex_score' in matching_types.columns else 0
                }
        
        return type_analysis
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns in the data"""
        temporal_analysis = {}
        
        # Year built analysis
        year_col = self._find_column_in_df(df, ['year built', 'year_built', 'built_year'])
        if year_col:
            years = pd.to_numeric(df[year_col], errors='coerce').dropna()
            
            if len(years) > 0:
                temporal_analysis['construction_years'] = {
                    'mean_year': float(years.mean()),
                    'median_year': float(years.median()),
                    'oldest': int(years.min()),
                    'newest': int(years.max())
                }
                
                # Decade distribution
                decades = {}
                for year in years:
                    decade = int(year // 10 * 10)
                    decades[f"{decade}s"] = decades.get(f"{decade}s", 0) + 1
                
                temporal_analysis['by_decade'] = decades
        
        # Sale date analysis (if available)
        sale_date_col = self._find_column_in_df(df, ['sale date', 'sale_date', 'sold_date'])
        if sale_date_col:
            try:
                sale_dates = pd.to_datetime(df[sale_date_col], errors='coerce').dropna()
                
                if len(sale_dates) > 0:
                    temporal_analysis['sales'] = {
                        'earliest_sale': sale_dates.min().isoformat(),
                        'latest_sale': sale_dates.max().isoformat(),
                        'by_year': sale_dates.dt.year.value_counts().to_dict()
                    }
            except:
                pass  # Skip if date parsing fails
        
        return temporal_analysis
    
    def _generate_market_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate market insights and recommendations"""
        insights = {
            'recommendations': [],
            'market_trends': {},
            'investment_opportunities': {}
        }
        
        if df.empty or 'flex_score' not in df.columns:
            return insights
        
        # High-scoring properties analysis
        high_score_properties = df[df['flex_score'] >= 7.0]
        
        if len(high_score_properties) > 0:
            insights['recommendations'].append(
                f"Found {len(high_score_properties)} high-scoring properties (score >= 7.0) "
                f"representing {len(high_score_properties)/len(df)*100:.1f}% of the dataset"
            )
            
            # Geographic concentration of high-scoring properties
            state_col = self._find_column_in_df(high_score_properties, ['state'])
            if state_col:
                top_states = high_score_properties[state_col].value_counts().head(3)
                insights['recommendations'].append(
                    f"Top states for high-scoring properties: {', '.join(top_states.index.tolist())}"
                )
        
        # Size optimization insights
        building_col = self._find_column_in_df(df, ['building sqft', 'building_sqft', 'sqft'])
        if building_col:
            building_sizes = pd.to_numeric(df[building_col], errors='coerce').dropna()
            optimal_size_properties = building_sizes[(building_sizes >= 20000) & (building_sizes <= 50000)]
            
            if len(optimal_size_properties) > 0:
                insights['recommendations'].append(
                    f"Found {len(optimal_size_properties)} properties in optimal size range (20k-50k sqft)"
                )
        
        # Market trends
        if len(df) > 100:  # Only for larger datasets
            insights['market_trends'] = {
                'total_market_size': len(df),
                'flex_candidate_rate': len(df[df['flex_score'] > 0]) / len(df) * 100,
                'high_quality_rate': len(df[df['flex_score'] >= 7.0]) / len(df) * 100
            }
        
        return insights
    
    def export_advanced_results(self, results_df: pd.DataFrame, 
                              analytics: Dict[str, Any],
                              output_dir: str = "data/exports/advanced") -> Dict[str, str]:
        """
        Export advanced results with multiple formats and detailed analytics
        
        Args:
            results_df: DataFrame with results
            analytics: Advanced analytics dictionary
            output_dir: Output directory for exports
            
        Returns:
            Dictionary with paths to exported files
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            exported_files = {}
            
            # Export main results to Excel with multiple sheets
            excel_path = output_path / f"flex_analysis_advanced_{timestamp}.xlsx"
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Main results
                if not results_df.empty:
                    results_df.to_excel(writer, sheet_name='Flex_Candidates', index=False)
                
                # Analytics summary
                analytics_df = self._convert_analytics_to_dataframe(analytics)
                analytics_df.to_excel(writer, sheet_name='Analytics_Summary', index=False)
                
                # Top candidates (if available)
                if not results_df.empty and 'flex_score' in results_df.columns:
                    top_candidates = results_df.head(50)
                    top_candidates.to_excel(writer, sheet_name='Top_50_Candidates', index=False)
            
            exported_files['excel'] = str(excel_path)
            
            # Export analytics to JSON
            json_path = output_path / f"analytics_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(analytics, f, indent=2, default=str)
            
            exported_files['analytics_json'] = str(json_path)
            
            # Export configuration
            config_path = output_path / f"scoring_config_{timestamp}.json"
            self.scoring_config.save_to_file(str(config_path))
            exported_files['config'] = str(config_path)
            
            # Export processing metrics
            metrics_path = output_path / f"processing_metrics_{timestamp}.json"
            with open(metrics_path, 'w') as f:
                json.dump(asdict(self.processing_metrics), f, indent=2)
            
            exported_files['metrics'] = str(metrics_path)
            
            self.logger.info(f"Advanced results exported to {output_dir}")
            return exported_files
            
        except Exception as e:
            self._handle_error("export_advanced_results", e)
            return {}
    
    def _convert_analytics_to_dataframe(self, analytics: Dict[str, Any]) -> pd.DataFrame:
        """Convert analytics dictionary to DataFrame for Excel export"""
        rows = []
        
        def flatten_dict(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        flattened = flatten_dict(analytics)
        
        for key, value in flattened.items():
            rows.append({'Metric': key, 'Value': str(value)})
        
        return pd.DataFrame(rows)


def create_default_scoring_config() -> ScoringConfiguration:
    """Create a default scoring configuration"""
    return ScoringConfiguration()


def load_scoring_config(file_path: str) -> ScoringConfiguration:
    """Load scoring configuration from file"""
    return ScoringConfiguration.load_from_file(file_path)


if __name__ == '__main__':
    # Example usage
    print("Advanced Flex Property Classifier - Example Usage")
    
    # Create default configuration
    config = create_default_scoring_config()
    print(f"Default configuration created with max score: {config.max_score}")
    
    # Save configuration example
    config.save_to_file("config/default_scoring_config.json")
    print("Configuration saved to config/default_scoring_config.json")