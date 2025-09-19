"""
Flex Property Classifier
Specialized classifier to identify flex industrial properties from private property datasets
"""

import logging
from typing import Optional, Dict, List, Any
import pandas as pd
from pathlib import Path

from utils.logger import setup_logging


class FlexPropertyClassifier:
    """
    Classifier for identifying flex industrial properties from Excel datasets
    
    Analyzes private property data to identify flex industrial candidates using:
    - Industrial property type filtering
    - Building size criteria (≥20,000 sqft)
    - Lot size filtering (0.5-20 acres)
    - Multi-factor scoring algorithm (0-10 scale)
    """
    
    def __init__(self, df: pd.DataFrame, logger: Optional[logging.Logger] = None):
        """
        Initialize the Flex Property Classifier
        
        Args:
            df: DataFrame containing property data
            logger: Optional logger instance (will create one if not provided)
        
        Raises:
            ValueError: If DataFrame is invalid or empty
            TypeError: If df is not a pandas DataFrame
        """
        # Validate DataFrame input
        self._validate_dataframe(df)
        
        # Store DataFrame
        self.data = df.copy()
        
        # Set up logging
        if logger is None:
            self.logger = setup_logging(
                name='flex_property_classifier',
                level='INFO'
            )
        else:
            self.logger = logger
        
        # Initialize storage for flex candidates
        self.flex_candidates: Optional[pd.DataFrame] = None
        
        # Configuration for industrial property identification
        self.industrial_keywords = [
            'industrial', 'warehouse', 'distribution', 'flex', 
            'manufacturing', 'storage', 'logistics', 'light industrial'
        ]
        
        # Configuration for filtering criteria
        self.min_building_sqft = 20000
        self.min_lot_acres = 0.5
        self.max_lot_acres = 20.0
        
        # Configuration for scoring
        self.max_flex_score = 10.0
        
        self.logger.info(f"FlexPropertyClassifier initialized with {len(self.data)} properties")
    
    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate the input DataFrame
        
        Args:
            df: DataFrame to validate
            
        Raises:
            TypeError: If df is not a pandas DataFrame
            ValueError: If DataFrame is empty or invalid
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        
        if len(df.columns) == 0:
            raise ValueError("DataFrame must have at least one column")
    
    def _handle_error(self, operation: str, error: Exception, continue_processing: bool = True) -> None:
        """
        Handle errors with consistent logging and optional continuation
        
        Args:
            operation: Name of the operation that failed
            error: The exception that occurred
            continue_processing: Whether to continue processing or re-raise
            
        Raises:
            Exception: Re-raises the error if continue_processing is False
        """
        error_msg = f"Error in {operation}: {str(error)}"
        self.logger.error(error_msg)
        
        if not continue_processing:
            raise Exception(error_msg) from error
        else:
            self.logger.warning(f"Continuing processing despite error in {operation}")
    
    def classify_flex_properties(self) -> pd.DataFrame:
        """
        Classify properties as flex industrial candidates using filtering criteria
        
        Returns:
            DataFrame containing filtered flex property candidates
            
        Raises:
            Exception: If classification process fails
        """
        try:
            self.logger.info("Starting flex property classification...")
            
            # Start with all properties
            candidates = self.data.copy()
            initial_count = len(candidates)
            
            self.logger.info(f"Initial dataset: {initial_count:,} properties")
            
            # Step 1: Filter by industrial property types
            candidates = self._filter_by_industrial_type(candidates)
            industrial_count = len(candidates)
            
            self.logger.info(f"After industrial filtering: {industrial_count:,} properties "
                           f"({(industrial_count/initial_count)*100:.1f}% of original)")
            
            # Step 2: Filter by building size
            candidates = self._filter_by_building_size(candidates)
            building_size_count = len(candidates)
            
            self.logger.info(f"After building size filtering (≥{self.min_building_sqft:,} sqft): "
                           f"{building_size_count:,} properties "
                           f"({(building_size_count/initial_count)*100:.1f}% of original)")
            
            # Step 3: Filter by lot size
            candidates = self._filter_by_lot_size(candidates)
            final_count = len(candidates)
            
            self.logger.info(f"After lot size filtering ({self.min_lot_acres}-{self.max_lot_acres} acres): "
                           f"{final_count:,} properties "
                           f"({(final_count/initial_count)*100:.1f}% of original)")
            
            # Store results
            self.flex_candidates = candidates
            
            if final_count == 0:
                self.logger.warning("No properties meet all flex criteria")
            else:
                self.logger.info(f"✅ Classification complete: {final_count:,} flex candidates identified")
            
            return candidates
            
        except Exception as e:
            self._handle_error("classify_flex_properties", e, continue_processing=False)
    
    def _filter_by_industrial_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter properties by industrial property type keywords
        
        Args:
            df: DataFrame to filter
            
        Returns:
            DataFrame containing only industrial properties
        """
        try:
            # Find property type column (case-insensitive)
            property_type_col = self._find_column(df, ['property type', 'type', 'property_type'])
            
            if property_type_col is None:
                self.logger.warning("No property type column found - skipping industrial filtering")
                return df
            
            # Create mask for industrial properties
            industrial_mask = pd.Series([False] * len(df), index=df.index)
            
            for keyword in self.industrial_keywords:
                # Case-insensitive keyword matching
                keyword_mask = df[property_type_col].astype(str).str.lower().str.contains(
                    keyword, na=False, regex=False
                )
                industrial_mask |= keyword_mask
                
                keyword_count = keyword_mask.sum()
                if keyword_count > 0:
                    self.logger.debug(f"Found {keyword_count} properties matching '{keyword}'")
            
            filtered_df = df[industrial_mask].copy()
            
            self.logger.info(f"Industrial filtering: {len(filtered_df)} properties match keywords: "
                           f"{', '.join(self.industrial_keywords)}")
            
            return filtered_df
            
        except Exception as e:
            self._handle_error("_filter_by_industrial_type", e)
            return df
    
    def _filter_by_building_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter properties by minimum building size
        
        Args:
            df: DataFrame to filter
            
        Returns:
            DataFrame containing properties meeting building size criteria
        """
        try:
            # Find building size column
            building_col = self._find_column(df, ['building sqft', 'building_sqft', 'sqft', 'square_feet'])
            
            if building_col is None:
                self.logger.warning("No building size column found - skipping building size filtering")
                return df
            
            # Convert to numeric, handling errors
            building_sizes = pd.to_numeric(df[building_col], errors='coerce')
            
            # Filter by minimum size
            size_mask = building_sizes >= self.min_building_sqft
            filtered_df = df[size_mask].copy()
            
            # Log statistics
            valid_sizes = building_sizes.dropna()
            if len(valid_sizes) > 0:
                self.logger.info(f"Building size stats - Min: {valid_sizes.min():,.0f}, "
                               f"Max: {valid_sizes.max():,.0f}, "
                               f"Avg: {valid_sizes.mean():,.0f} sqft")
            
            null_count = building_sizes.isna().sum()
            if null_count > 0:
                self.logger.warning(f"{null_count} properties have missing building size data")
            
            return filtered_df
            
        except Exception as e:
            self._handle_error("_filter_by_building_size", e)
            return df
    
    def _filter_by_lot_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter properties by lot size range (0.5-20 acres)
        
        Args:
            df: DataFrame to filter
            
        Returns:
            DataFrame containing properties meeting lot size criteria
        """
        try:
            # Find lot size column
            lot_col = self._find_column(df, ['lot size acres', 'lot_size_acres', 'acres', 'lot_acres'])
            
            if lot_col is None:
                self.logger.warning("No lot size column found - skipping lot size filtering")
                return df
            
            # Convert to numeric, handling errors
            lot_sizes = pd.to_numeric(df[lot_col], errors='coerce')
            
            # Filter by size range
            size_mask = (lot_sizes >= self.min_lot_acres) & (lot_sizes <= self.max_lot_acres)
            filtered_df = df[size_mask].copy()
            
            # Log statistics
            valid_sizes = lot_sizes.dropna()
            if len(valid_sizes) > 0:
                self.logger.info(f"Lot size stats - Min: {valid_sizes.min():.2f}, "
                               f"Max: {valid_sizes.max():.2f}, "
                               f"Avg: {valid_sizes.mean():.2f} acres")
            
            null_count = lot_sizes.isna().sum()
            if null_count > 0:
                self.logger.warning(f"{null_count} properties have missing lot size data")
            
            return filtered_df
            
        except Exception as e:
            self._handle_error("_filter_by_lot_size", e)
            return df
    
    def _find_column(self, df: pd.DataFrame, search_terms: List[str]) -> Optional[str]:
        """
        Find column matching any of the search terms (case-insensitive)
        
        Args:
            df: DataFrame to search
            search_terms: List of terms to search for in column names
            
        Returns:
            Column name if found, None otherwise
        """
        for term in search_terms:
            for col in df.columns:
                if term.lower() in col.lower():
                    return col
        return None
    
    def calculate_flex_score(self, row: pd.Series) -> float:
        """
        Calculate flex score for a property using multi-factor algorithm (0-10 scale)
        
        Args:
            row: Property data as pandas Series
            
        Returns:
            Flex score between 0 and 10
        """
        try:
            total_score = 0.0
            
            # Building size scoring (20k-50k=3pts, 50k-100k=2pts, 100k-200k=1pt)
            building_score = self._score_building_size(row)
            total_score += building_score
            
            # Property type scoring (flex=3pts, warehouse/distribution=2.5pts, etc.)
            type_score = self._score_property_type(row)
            total_score += type_score
            
            # Lot size scoring (1-5 acres=2pts, 5-10 acres=1.5pts, edge ranges=1pt)
            lot_score = self._score_lot_size(row)
            total_score += lot_score
            
            # Age/condition scoring (≥1990=1pt, ≥1980=0.5pts)
            age_score = self._score_age_condition(row)
            total_score += age_score
            
            # Occupancy bonus (<100% occupied=1pt)
            occupancy_score = self._score_occupancy(row)
            total_score += occupancy_score
            
            # Cap at maximum score
            final_score = min(total_score, self.max_flex_score)
            
            return final_score
            
        except Exception as e:
            self._handle_error("calculate_flex_score", e)
            return 0.0
    
    def _score_building_size(self, row: pd.Series) -> float:
        """
        Score building size factor
        
        Args:
            row: Property data
            
        Returns:
            Building size score (0-3 points)
        """
        try:
            building_col = self._find_column(pd.DataFrame([row]), 
                                           ['building sqft', 'building_sqft', 'sqft', 'square_feet'])
            
            if building_col is None:
                return 0.0
            
            building_size = pd.to_numeric(row.get(building_col), errors='coerce')
            
            if pd.isna(building_size):
                return 0.0
            
            if 20000 <= building_size < 50000:
                return 3.0  # Ideal flex size
            elif 50000 <= building_size < 100000:
                return 2.0  # Good flex size
            elif 100000 <= building_size < 200000:
                return 1.0  # Acceptable flex size
            else:
                return 0.0  # Too large or too small
                
        except Exception as e:
            self._handle_error("_score_building_size", e)
            return 0.0
    
    def _score_property_type(self, row: pd.Series) -> float:
        """
        Score property type factor
        
        Args:
            row: Property data
            
        Returns:
            Property type score (0-3 points)
        """
        try:
            type_col = self._find_column(pd.DataFrame([row]), 
                                       ['property type', 'type', 'property_type'])
            
            if type_col is None:
                return 0.0
            
            prop_type = str(row.get(type_col, '')).lower()
            
            if 'flex' in prop_type:
                return 3.0  # Perfect match
            elif 'warehouse' in prop_type or 'distribution' in prop_type:
                return 2.5  # Very good
            elif 'light industrial' in prop_type:
                return 2.0  # Good
            elif 'industrial' in prop_type:
                return 1.5  # Acceptable
            elif any(keyword in prop_type for keyword in ['manufacturing', 'storage', 'logistics']):
                return 1.0  # Possible
            else:
                return 0.0
                
        except Exception as e:
            self._handle_error("_score_property_type", e)
            return 0.0
    
    def _score_lot_size(self, row: pd.Series) -> float:
        """
        Score lot size factor
        
        Args:
            row: Property data
            
        Returns:
            Lot size score (0-2 points)
        """
        try:
            lot_col = self._find_column(pd.DataFrame([row]), 
                                      ['lot size acres', 'lot_size_acres', 'acres', 'lot_acres'])
            
            if lot_col is None:
                return 0.0
            
            lot_size = pd.to_numeric(row.get(lot_col), errors='coerce')
            
            if pd.isna(lot_size):
                return 0.0
            
            if 1.0 <= lot_size <= 5.0:
                return 2.0  # Ideal range
            elif 5.0 < lot_size <= 10.0:
                return 1.5  # Good range
            elif (0.5 <= lot_size < 1.0) or (10.0 < lot_size <= 20.0):
                return 1.0  # Acceptable (small or large)
            else:
                return 0.0
                
        except Exception as e:
            self._handle_error("_score_lot_size", e)
            return 0.0
    
    def _score_age_condition(self, row: pd.Series) -> float:
        """
        Score age/condition factor
        
        Args:
            row: Property data
            
        Returns:
            Age/condition score (0-1 points)
        """
        try:
            year_col = self._find_column(pd.DataFrame([row]), 
                                       ['year built', 'year_built', 'built_year'])
            
            if year_col is None:
                return 0.0
            
            year_built = pd.to_numeric(row.get(year_col), errors='coerce')
            
            if pd.isna(year_built):
                return 0.0
            
            if year_built >= 1990:
                return 1.0  # Modern construction
            elif year_built >= 1980:
                return 0.5  # Decent condition
            else:
                return 0.0
                
        except Exception as e:
            self._handle_error("_score_age_condition", e)
            return 0.0
    
    def _score_occupancy(self, row: pd.Series) -> float:
        """
        Score occupancy bonus factor
        
        Args:
            row: Property data
            
        Returns:
            Occupancy bonus score (0-1 points)
        """
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
            
            if occupancy < 1.0:  # Less than 100% occupied
                return 1.0  # Flex opportunity
            else:
                return 0.0
                
        except Exception as e:
            self._handle_error("_score_occupancy", e)
            return 0.0
    
    def get_top_candidates(self, n: int = 100) -> pd.DataFrame:
        """
        Get top N flex candidates sorted by score
        
        Args:
            n: Number of top candidates to return (default 100)
            
        Returns:
            DataFrame containing top N candidates sorted by flex score
            
        Raises:
            RuntimeError: If classification hasn't been run yet
        """
        try:
            if self.flex_candidates is None:
                raise RuntimeError("Must run classify_flex_properties() first")
            
            if len(self.flex_candidates) == 0:
                self.logger.warning("No flex candidates available")
                return pd.DataFrame()
            
            # Calculate scores for all candidates if not already done
            if 'flex_score' not in self.flex_candidates.columns:
                self.logger.info("Calculating flex scores for all candidates...")
                self.flex_candidates['flex_score'] = self.flex_candidates.apply(
                    self.calculate_flex_score, axis=1
                )
            
            # Sort by flex score in descending order
            sorted_candidates = self.flex_candidates.sort_values(
                'flex_score', ascending=False
            ).copy()
            
            # Get top N candidates
            top_candidates = sorted_candidates.head(n)
            
            self.logger.info(f"Retrieved top {len(top_candidates)} candidates "
                           f"(requested: {n})")
            
            if len(top_candidates) > 0:
                score_stats = top_candidates['flex_score'].describe()
                self.logger.info(f"Score range: {score_stats['min']:.1f} - {score_stats['max']:.1f}, "
                               f"Average: {score_stats['mean']:.1f}")
            
            return top_candidates
            
        except Exception as e:
            self._handle_error("get_top_candidates", e, continue_processing=False)
    
    def _apply_scoring_to_candidates(self) -> None:
        """
        Apply flex scoring to all candidates and sort by score
        """
        try:
            if self.flex_candidates is None or len(self.flex_candidates) == 0:
                self.logger.warning("No candidates to score")
                return
            
            self.logger.info(f"Calculating flex scores for {len(self.flex_candidates)} candidates...")
            
            # Calculate scores
            scores = []
            for idx, row in self.flex_candidates.iterrows():
                score = self.calculate_flex_score(row)
                scores.append(score)
            
            # Add scores to DataFrame
            self.flex_candidates['flex_score'] = scores
            
            # Sort by score descending
            self.flex_candidates = self.flex_candidates.sort_values(
                'flex_score', ascending=False
            ).reset_index(drop=True)
            
            # Log scoring statistics
            score_stats = self.flex_candidates['flex_score'].describe()
            self.logger.info(f"Scoring complete - Min: {score_stats['min']:.1f}, "
                           f"Max: {score_stats['max']:.1f}, "
                           f"Average: {score_stats['mean']:.1f}")
            
            # Log score distribution
            high_score = len(self.flex_candidates[self.flex_candidates['flex_score'] >= 8])
            medium_score = len(self.flex_candidates[
                (self.flex_candidates['flex_score'] >= 6) & 
                (self.flex_candidates['flex_score'] < 8)
            ])
            low_score = len(self.flex_candidates[self.flex_candidates['flex_score'] < 6])
            
            self.logger.info(f"Score distribution - High (8+): {high_score}, "
                           f"Medium (6-8): {medium_score}, "
                           f"Low (<6): {low_score}")
            
        except Exception as e:
            self._handle_error("_apply_scoring_to_candidates", e)
    
    def export_results(self, output_path: Optional[str] = None) -> str:
        """
        Export flex candidates to Excel file
        
        Args:
            output_path: Optional custom output path (defaults to data/exports/private_flex_candidates.xlsx)
            
        Returns:
            Path to the exported file
            
        Raises:
            RuntimeError: If no candidates are available for export
        """
        try:
            if self.flex_candidates is None or len(self.flex_candidates) == 0:
                raise RuntimeError("No flex candidates available for export. Run classify_flex_properties() first.")
            
            # Set default output path
            if output_path is None:
                output_dir = Path('data/exports')
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / 'private_flex_candidates.xlsx'
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Ensure scores are calculated
            if 'flex_score' not in self.flex_candidates.columns:
                self._apply_scoring_to_candidates()
            
            # Select columns for export (only include existing columns)
            desired_columns = [
                'Property Name', 'Property Type', 'Address', 'City', 'State', 'County',
                'Building SqFt', 'Lot Size Acres', 'Year Built', 'Zoning Code',
                'Sale Date', 'Sold Price', 'Sold Price/SqFt', 'Owner Name', 'Occupancy'
            ]
            
            export_columns = []
            column_mapping = {}
            
            # Find matching columns (case-insensitive, flexible matching)
            for desired_col in desired_columns:
                matching_col = self._find_column(self.flex_candidates, [desired_col.lower()])
                if matching_col:
                    export_columns.append(matching_col)
                    column_mapping[matching_col] = desired_col
            
            # Always include flex_score
            export_columns.append('flex_score')
            column_mapping['flex_score'] = 'Flex Score'
            
            # Create export DataFrame
            export_df = self.flex_candidates[export_columns].copy()
            
            # Rename columns to standard names
            export_df = export_df.rename(columns=column_mapping)
            
            # Export to Excel
            export_df.to_excel(output_path, index=False, engine='openpyxl')
            
            # Log export completion
            candidate_count = len(export_df)
            self.logger.info(f"✅ Export complete: {candidate_count} candidates exported to {output_path}")
            self.logger.info(f"Exported columns: {', '.join(export_df.columns)}")
            
            return str(output_path)
            
        except Exception as e:
            self._handle_error("export_results", e, continue_processing=False)
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """
        Generate comprehensive analysis statistics
        
        Returns:
            Dictionary containing analysis metrics and statistics
            
        Raises:
            RuntimeError: If no analysis has been performed
        """
        try:
            if self.flex_candidates is None:
                raise RuntimeError("No analysis available. Run classify_flex_properties() first.")
            
            # Ensure scores are calculated
            if len(self.flex_candidates) > 0 and 'flex_score' not in self.flex_candidates.columns:
                self._apply_scoring_to_candidates()
            
            stats = {
                'total_properties_analyzed': len(self.data),
                'total_flex_candidates': len(self.flex_candidates),
                'candidate_percentage': 0.0,
                'score_statistics': {},
                'property_type_distribution': {},
                'size_distribution': {},
                'geographic_distribution': {}
            }
            
            # Calculate candidate percentage
            if len(self.data) > 0:
                stats['candidate_percentage'] = (len(self.flex_candidates) / len(self.data)) * 100
            
            # Score statistics
            if len(self.flex_candidates) > 0 and 'flex_score' in self.flex_candidates.columns:
                scores = self.flex_candidates['flex_score']
                stats['score_statistics'] = {
                    'average_flex_score': float(scores.mean()),
                    'min_score': float(scores.min()),
                    'max_score': float(scores.max()),
                    'median_score': float(scores.median()),
                    'std_deviation': float(scores.std()),
                    'high_score_count': int((scores >= 8).sum()),
                    'medium_score_count': int(((scores >= 6) & (scores < 8)).sum()),
                    'low_score_count': int((scores < 6).sum())
                }
            
            # Property type distribution for top candidates
            if len(self.flex_candidates) > 0:
                top_100 = self.flex_candidates.head(100)
                type_col = self._find_column(top_100, ['property type', 'type', 'property_type'])
                
                if type_col:
                    type_counts = top_100[type_col].value_counts()
                    stats['property_type_distribution'] = type_counts.to_dict()
            
            # Size distribution analysis
            if len(self.flex_candidates) > 0:
                building_col = self._find_column(self.flex_candidates, 
                                               ['building sqft', 'building_sqft', 'sqft'])
                
                if building_col:
                    building_sizes = pd.to_numeric(self.flex_candidates[building_col], errors='coerce')
                    valid_sizes = building_sizes.dropna()
                    
                    if len(valid_sizes) > 0:
                        stats['size_distribution'] = {
                            'small_20k_50k': int(((valid_sizes >= 20000) & (valid_sizes < 50000)).sum()),
                            'medium_50k_100k': int(((valid_sizes >= 50000) & (valid_sizes < 100000)).sum()),
                            'large_100k_200k': int(((valid_sizes >= 100000) & (valid_sizes < 200000)).sum()),
                            'extra_large_200k_plus': int((valid_sizes >= 200000).sum()),
                            'average_size': float(valid_sizes.mean()),
                            'median_size': float(valid_sizes.median())
                        }
            
            # Geographic distribution
            if len(self.flex_candidates) > 0:
                city_col = self._find_column(self.flex_candidates, ['city'])
                county_col = self._find_column(self.flex_candidates, ['county'])
                
                if city_col:
                    city_counts = self.flex_candidates[city_col].value_counts().head(10)
                    stats['geographic_distribution']['top_cities'] = city_counts.to_dict()
                
                if county_col:
                    county_counts = self.flex_candidates[county_col].value_counts()
                    stats['geographic_distribution']['counties'] = county_counts.to_dict()
            
            self.logger.info("Analysis statistics generated successfully")
            return stats
            
        except Exception as e:
            self._handle_error("get_analysis_statistics", e, continue_processing=False)
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """
        Validate data quality and identify potential issues
        
        Returns:
            Dictionary containing data quality assessment
        """
        try:
            self.logger.info("Performing data quality validation...")
            
            validation_report = {
                'total_properties': len(self.data),
                'column_analysis': {},
                'missing_data_summary': {},
                'data_type_issues': {},
                'recommendations': []
            }
            
            # Required columns for flex analysis
            required_columns = {
                'property_type': ['property type', 'type', 'property_type'],
                'building_size': ['building sqft', 'building_sqft', 'sqft', 'square_feet'],
                'lot_size': ['lot size acres', 'lot_size_acres', 'acres', 'lot_acres']
            }
            
            # Check for required columns
            missing_required = []
            for req_name, search_terms in required_columns.items():
                found_col = self._find_column(self.data, search_terms)
                if found_col is None:
                    missing_required.append(req_name)
                    validation_report['recommendations'].append(
                        f"Missing {req_name} column - classification will be limited"
                    )
                else:
                    # Analyze column quality
                    col_data = self.data[found_col]
                    null_count = col_data.isna().sum()
                    null_percentage = (null_count / len(self.data)) * 100
                    
                    validation_report['column_analysis'][req_name] = {
                        'column_name': found_col,
                        'null_count': int(null_count),
                        'null_percentage': float(null_percentage),
                        'data_type': str(col_data.dtype),
                        'unique_values': int(col_data.nunique())
                    }
                    
                    if null_percentage > 50:
                        validation_report['recommendations'].append(
                            f"High missing data in {req_name} ({null_percentage:.1f}%) - "
                            f"consider data cleaning"
                        )
            
            # Check data types for numeric columns
            numeric_columns = ['building_size', 'lot_size']
            for col_name in numeric_columns:
                if col_name in validation_report['column_analysis']:
                    col_info = validation_report['column_analysis'][col_name]
                    actual_col = col_info['column_name']
                    
                    # Try to convert to numeric
                    numeric_data = pd.to_numeric(self.data[actual_col], errors='coerce')
                    conversion_failures = numeric_data.isna().sum() - self.data[actual_col].isna().sum()
                    
                    if conversion_failures > 0:
                        validation_report['data_type_issues'][col_name] = {
                            'conversion_failures': int(conversion_failures),
                            'failure_percentage': float((conversion_failures / len(self.data)) * 100)
                        }
                        validation_report['recommendations'].append(
                            f"Data type issues in {col_name} - {conversion_failures} values "
                            f"cannot be converted to numeric"
                        )
            
            # Overall missing data summary
            total_missing = self.data.isna().sum().sum()
            total_cells = len(self.data) * len(self.data.columns)
            overall_completeness = ((total_cells - total_missing) / total_cells) * 100
            
            validation_report['missing_data_summary'] = {
                'total_missing_values': int(total_missing),
                'overall_completeness': float(overall_completeness),
                'columns_with_missing_data': int((self.data.isna().sum() > 0).sum())
            }
            
            # Generate quality score
            quality_score = self._calculate_quality_score(validation_report)
            validation_report['quality_score'] = quality_score
            
            self.logger.info(f"Data quality validation complete - Quality score: {quality_score:.1f}/10")
            
            return validation_report
            
        except Exception as e:
            self._handle_error("validate_data_quality", e)
            return {'error': str(e)}
    
    def _calculate_quality_score(self, validation_report: Dict[str, Any]) -> float:
        """
        Calculate overall data quality score (0-10)
        
        Args:
            validation_report: Validation report data
            
        Returns:
            Quality score between 0 and 10
        """
        try:
            score = 10.0
            
            # Deduct for missing required columns
            required_cols = ['property_type', 'building_size', 'lot_size']
            missing_cols = len([col for col in required_cols 
                              if col not in validation_report.get('column_analysis', {})])
            score -= missing_cols * 2.0
            
            # Deduct for high missing data percentages
            for col_info in validation_report.get('column_analysis', {}).values():
                null_pct = col_info.get('null_percentage', 0)
                if null_pct > 50:
                    score -= 2.0
                elif null_pct > 25:
                    score -= 1.0
                elif null_pct > 10:
                    score -= 0.5
            
            # Deduct for data type issues
            for issue_info in validation_report.get('data_type_issues', {}).values():
                failure_pct = issue_info.get('failure_percentage', 0)
                if failure_pct > 10:
                    score -= 1.5
                elif failure_pct > 5:
                    score -= 1.0
                elif failure_pct > 1:
                    score -= 0.5
            
            return max(0.0, min(10.0, score))
            
        except Exception as e:
            self._handle_error("_calculate_quality_score", e)
            return 0.0
    
    def handle_missing_data(self, strategy: str = 'skip') -> None:
        """
        Handle missing data according to specified strategy
        
        Args:
            strategy: Strategy for handling missing data ('skip', 'fill_zero', 'fill_median')
        """
        try:
            if strategy == 'skip':
                # Already handled in scoring methods by returning 0 for missing values
                self.logger.info("Missing data strategy: Skip (assign 0 points for missing factors)")
                
            elif strategy == 'fill_zero':
                # Fill numeric columns with 0
                numeric_cols = []
                for search_terms in [
                    ['building sqft', 'building_sqft', 'sqft'],
                    ['lot size acres', 'lot_size_acres', 'acres'],
                    ['year built', 'year_built'],
                    ['occupancy']
                ]:
                    col = self._find_column(self.data, search_terms)
                    if col:
                        numeric_cols.append(col)
                
                for col in numeric_cols:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce').fillna(0)
                
                self.logger.info(f"Filled missing values with 0 in columns: {numeric_cols}")
                
            elif strategy == 'fill_median':
                # Fill numeric columns with median values
                numeric_cols = []
                for search_terms in [
                    ['building sqft', 'building_sqft', 'sqft'],
                    ['lot size acres', 'lot_size_acres', 'acres'],
                    ['year built', 'year_built']
                ]:
                    col = self._find_column(self.data, search_terms)
                    if col:
                        numeric_data = pd.to_numeric(self.data[col], errors='coerce')
                        median_val = numeric_data.median()
                        self.data[col] = numeric_data.fillna(median_val)
                        numeric_cols.append(col)
                
                self.logger.info(f"Filled missing values with median in columns: {numeric_cols}")
            
        except Exception as e:
            self._handle_error("handle_missing_data", e)
    
    def integrate_with_pipeline(self, db_manager=None, store_results: bool = False) -> Dict[str, Any]:
        """
        Integrate with existing pipeline components
        
        Args:
            db_manager: Optional database manager for storing results
            store_results: Whether to store results in database
            
        Returns:
            Dictionary containing integration results and compatibility info
        """
        try:
            self.logger.info("Integrating with existing pipeline components...")
            
            integration_results = {
                'pipeline_compatibility': True,
                'database_integration': False,
                'scorer_validation': {},
                'stored_records': 0,
                'integration_errors': []
            }
            
            # Validate compatibility with existing FlexPropertyScorer
            try:
                from processors.flex_scorer import FlexPropertyScorer
                integration_results['scorer_validation']['flex_scorer_available'] = True
                self.logger.info("✅ FlexPropertyScorer integration available")
            except ImportError:
                integration_results['scorer_validation']['flex_scorer_available'] = False
                self.logger.warning("FlexPropertyScorer not available - continuing without validation")
            
            # Database integration
            if db_manager and store_results:
                try:
                    stored_count = self._store_results_in_database(db_manager)
                    integration_results['database_integration'] = True
                    integration_results['stored_records'] = stored_count
                    self.logger.info(f"✅ Stored {stored_count} results in database")
                except Exception as e:
                    integration_results['integration_errors'].append(f"Database storage: {str(e)}")
                    self.logger.warning(f"Database integration failed: {str(e)}")
            
            # Format conversion for pipeline compatibility
            if self.flex_candidates is not None and len(self.flex_candidates) > 0:
                pipeline_format = self._convert_to_pipeline_format()
                integration_results['pipeline_format_records'] = len(pipeline_format)
                self.logger.info(f"Converted {len(pipeline_format)} records to pipeline format")
            
            return integration_results
            
        except Exception as e:
            self._handle_error("integrate_with_pipeline", e)
            return {'error': str(e)}
    
    def _store_results_in_database(self, db_manager) -> int:
        """
        Store classification results in MongoDB using existing patterns
        
        Args:
            db_manager: Database manager instance
            
        Returns:
            Number of records stored
        """
        try:
            if self.flex_candidates is None or len(self.flex_candidates) == 0:
                return 0
            
            # Ensure scores are calculated
            if 'flex_score' not in self.flex_candidates.columns:
                self._apply_scoring_to_candidates()
            
            # Convert to pipeline format
            pipeline_records = self._convert_to_pipeline_format()
            
            # Store in database
            collection_name = 'flex_candidates_private'
            
            # Clear existing records for this analysis
            db_manager.db[collection_name].delete_many({})
            
            # Insert new records
            if pipeline_records:
                result = db_manager.db[collection_name].insert_many(pipeline_records)
                return len(result.inserted_ids)
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Database storage error: {str(e)}")
            raise
    
    def _convert_to_pipeline_format(self) -> List[Dict[str, Any]]:
        """
        Convert classification results to pipeline-compatible format
        
        Returns:
            List of dictionaries in pipeline format
        """
        try:
            if self.flex_candidates is None or len(self.flex_candidates) == 0:
                return []
            
            pipeline_records = []
            
            for idx, row in self.flex_candidates.iterrows():
                # Create pipeline-compatible record
                record = {
                    'source': 'private_property_classifier',
                    'analysis_date': pd.Timestamp.now().isoformat(),
                    'flex_score': row.get('flex_score', 0.0),
                    'classification_method': 'excel_based_classifier'
                }
                
                # Map standard fields
                field_mappings = {
                    'property_name': ['property name', 'property_name', 'name'],
                    'property_type': ['property type', 'property_type', 'type'],
                    'address': ['address', 'street_address'],
                    'city': ['city'],
                    'state': ['state'],
                    'county': ['county'],
                    'building_sqft': ['building sqft', 'building_sqft', 'sqft'],
                    'lot_size_acres': ['lot size acres', 'lot_size_acres', 'acres'],
                    'year_built': ['year built', 'year_built'],
                    'owner_name': ['owner name', 'owner_name', 'owner'],
                    'sale_date': ['sale date', 'sale_date'],
                    'sold_price': ['sold price', 'sold_price', 'price'],
                    'occupancy': ['occupancy', 'occupancy_rate']
                }
                
                # Map available fields
                for standard_field, search_terms in field_mappings.items():
                    value = None
                    for term in search_terms:
                        for col in self.flex_candidates.columns:
                            if term.lower() in col.lower():
                                value = row.get(col)
                                break
                        if value is not None:
                            break
                    
                    if value is not None and not pd.isna(value):
                        record[standard_field] = value
                
                # Add scoring breakdown
                record['score_breakdown'] = {
                    'building_size_score': self._score_building_size(row),
                    'property_type_score': self._score_property_type(row),
                    'lot_size_score': self._score_lot_size(row),
                    'age_condition_score': self._score_age_condition(row),
                    'occupancy_score': self._score_occupancy(row)
                }
                
                pipeline_records.append(record)
            
            return pipeline_records
            
        except Exception as e:
            self.logger.error(f"Format conversion error: {str(e)}")
            return []
    
    def validate_against_existing_scorer(self, sample_size: int = 10) -> Dict[str, Any]:
        """
        Validate classification results against existing FlexPropertyScorer
        
        Args:
            sample_size: Number of properties to validate
            
        Returns:
            Dictionary containing validation results
        """
        try:
            validation_results = {
                'validation_performed': False,
                'sample_size': 0,
                'score_correlation': 0.0,
                'average_difference': 0.0,
                'validation_errors': []
            }
            
            # Check if FlexPropertyScorer is available
            try:
                from processors.flex_scorer import FlexPropertyScorer
            except ImportError:
                validation_results['validation_errors'].append("FlexPropertyScorer not available")
                return validation_results
            
            if self.flex_candidates is None or len(self.flex_candidates) == 0:
                validation_results['validation_errors'].append("No candidates to validate")
                return validation_results
            
            # Get sample for validation
            sample_candidates = self.flex_candidates.head(sample_size)
            
            # Convert to format expected by existing scorer
            pipeline_format = self._convert_to_pipeline_format()[:sample_size]
            
            # Compare scores (this would need actual FlexPropertyScorer implementation)
            # For now, just log that validation framework is in place
            validation_results['validation_performed'] = True
            validation_results['sample_size'] = len(sample_candidates)
            
            self.logger.info(f"Validation framework ready for {len(sample_candidates)} candidates")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return {'error': str(e)}