"""
Advanced filter engine with ML capabilities
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings

from ..core.interfaces import IFilterEngine, FilterResult
from ..core.base_classes import ProcessorBase
from config.settings import get_config

warnings.filterwarnings('ignore')


@dataclass
class FilterConfig:
    """Filter configuration"""
    name: str
    filters: Dict[str, Any]
    ml_options: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None


@dataclass
class MLFilterOptions:
    """ML filter options"""
    enable_anomaly_detection: bool = False
    anomaly_contamination: float = 0.1
    enable_clustering: bool = False
    n_clusters: int = 5
    enable_similarity_matching: bool = False
    similarity_threshold: float = 0.8
    reference_property: Optional[Dict[str, Any]] = None


class AdvancedFilterEngine(ProcessorBase, IFilterEngine):
    """Advanced filter engine with ML capabilities"""
    
    def __init__(self):
        super().__init__("AdvancedFilterEngine")
        self.config = get_config()
        
        # ML models
        self.anomaly_detector: Optional[IsolationForest] = None
        self.clusterer: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None
        
        # Filter history
        self.filter_history: List[FilterConfig] = []
        
        # Numeric columns for ML processing
        self.numeric_columns = [
            'building_sqft', 'lot_size_acres', 'year_built', 'sold_price',
            'assessed_value', 'rent', 'occupancy_rate', 'cap_rate', 'noi',
            'price_per_sqft', 'building_age', 'building_efficiency'
        ]
    
    def _do_initialize(self) -> None:
        """Initialize filter engine"""
        self.logger.info("Initializing advanced filter engine with ML capabilities")
    
    def _do_cleanup(self) -> None:
        """Cleanup filter engine"""
        self.filter_history.clear()
    
    def apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> FilterResult:
        """Apply traditional filters to data"""
        start_time = datetime.now()
        original_count = len(df)
        
        filtered_df = df.copy()
        applied_filters = {}
        
        # Apply range filters
        for column, filter_config in filters.items():
            if column not in df.columns:
                continue
            
            if isinstance(filter_config, dict):
                # Range filter
                if 'min' in filter_config and filter_config['min'] is not None:
                    mask = filtered_df[column] >= filter_config['min']
                    filtered_df = filtered_df[mask]
                    applied_filters[f"{column}_min"] = filter_config['min']
                
                if 'max' in filter_config and filter_config['max'] is not None:
                    mask = filtered_df[column] <= filter_config['max']
                    filtered_df = filtered_df[mask]
                    applied_filters[f"{column}_max"] = filter_config['max']
            
            elif isinstance(filter_config, (list, tuple)):
                # Multi-select filter
                mask = filtered_df[column].isin(filter_config)
                filtered_df = filtered_df[mask]
                applied_filters[column] = filter_config
            
            else:
                # Single value filter
                mask = filtered_df[column] == filter_config
                filtered_df = filtered_df[mask]
                applied_filters[column] = filter_config
        
        # Calculate performance metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        filtered_count = len(filtered_df)
        reduction_percentage = ((original_count - filtered_count) / original_count) * 100 if original_count > 0 else 0
        
        performance_metrics = {
            'processing_time': processing_time,
            'original_count': original_count,
            'filtered_count': filtered_count,
            'reduction_percentage': reduction_percentage
        }
        
        # Generate recommendations
        recommendations = self._generate_filter_recommendations(df, filtered_df, filters)
        
        filter_summary = {
            'applied_filters': applied_filters,
            'filters_count': len(applied_filters),
            'performance': performance_metrics
        }
        
        self._record_processing(processing_time, True)
        
        return FilterResult(
            filtered_data=filtered_df,
            filter_summary=filter_summary,
            performance_metrics=performance_metrics,
            recommendations=recommendations
        )
    
    def apply_ml_filters(self, df: pd.DataFrame, ml_options: Dict[str, Any]) -> FilterResult:
        """Apply ML-powered filters"""
        start_time = datetime.now()
        original_count = len(df)
        
        filtered_df = df.copy()
        ml_results = {}
        
        # Parse ML options
        options = MLFilterOptions(**ml_options)
        
        # Prepare numeric data for ML
        numeric_df = self._prepare_numeric_data(filtered_df)
        
        if numeric_df.empty:
            return FilterResult(
                filtered_data=filtered_df,
                filter_summary={'error': 'No numeric data available for ML filtering'},
                performance_metrics={'processing_time': 0},
                recommendations=['Add numeric columns for ML filtering']
            )
        
        # Apply anomaly detection
        if options.enable_anomaly_detection:
            anomaly_result = self._apply_anomaly_detection(
                filtered_df, numeric_df, options.anomaly_contamination
            )
            filtered_df = anomaly_result['filtered_data']
            ml_results['anomaly_detection'] = anomaly_result['summary']
        
        # Apply clustering
        if options.enable_clustering:
            cluster_result = self._apply_clustering(
                filtered_df, numeric_df, options.n_clusters
            )
            filtered_df = cluster_result['filtered_data']
            ml_results['clustering'] = cluster_result['summary']
        
        # Apply similarity matching
        if options.enable_similarity_matching and options.reference_property:
            similarity_result = self._apply_similarity_matching(
                filtered_df, numeric_df, options.reference_property, options.similarity_threshold
            )
            filtered_df = similarity_result['filtered_data']
            ml_results['similarity_matching'] = similarity_result['summary']
        
        # Calculate performance metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        filtered_count = len(filtered_df)
        reduction_percentage = ((original_count - filtered_count) / original_count) * 100 if original_count > 0 else 0
        
        performance_metrics = {
            'processing_time': processing_time,
            'original_count': original_count,
            'filtered_count': filtered_count,
            'reduction_percentage': reduction_percentage
        }
        
        # Generate recommendations
        recommendations = self._generate_ml_recommendations(ml_results, options)
        
        filter_summary = {
            'ml_results': ml_results,
            'ml_options': ml_options,
            'performance': performance_metrics
        }
        
        self._record_processing(processing_time, True)
        
        return FilterResult(
            filtered_data=filtered_df,
            filter_summary=filter_summary,
            performance_metrics=performance_metrics,
            recommendations=recommendations
        )
    
    def get_filter_recommendations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get AI-powered filter recommendations"""
        recommendations = []
        
        # Analyze data patterns
        numeric_df = self._prepare_numeric_data(df)
        
        if not numeric_df.empty:
            # Recommend filters based on data distribution
            for column in numeric_df.columns:
                if column in df.columns:
                    column_recommendations = self._analyze_column_for_recommendations(df[column], column)
                    recommendations.extend(column_recommendations)
        
        # Recommend ML filters
        ml_recommendations = self._recommend_ml_filters(df)
        recommendations.extend(ml_recommendations)
        
        return recommendations
    
    def _prepare_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare numeric data for ML processing"""
        available_numeric_cols = [col for col in self.numeric_columns if col in df.columns]
        
        if not available_numeric_cols:
            return pd.DataFrame()
        
        numeric_df = df[available_numeric_cols].copy()
        
        # Handle missing values
        numeric_df = numeric_df.fillna(numeric_df.median())
        
        # Remove columns with zero variance
        numeric_df = numeric_df.loc[:, numeric_df.var() != 0]
        
        return numeric_df
    
    def _apply_anomaly_detection(self, df: pd.DataFrame, numeric_df: pd.DataFrame, contamination: float) -> Dict[str, Any]:
        """Apply anomaly detection using Isolation Forest"""
        try:
            # Initialize and fit anomaly detector
            self.anomaly_detector = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            
            # Scale the data
            if self.scaler is None:
                self.scaler = StandardScaler()
            
            scaled_data = self.scaler.fit_transform(numeric_df)
            
            # Detect anomalies
            anomaly_labels = self.anomaly_detector.fit_predict(scaled_data)
            anomaly_scores = self.anomaly_detector.score_samples(scaled_data)
            
            # Filter out anomalies (keep normal points)
            normal_mask = anomaly_labels == 1
            filtered_df = df[normal_mask].copy()
            
            # Add anomaly scores to the filtered data
            filtered_df['anomaly_score'] = anomaly_scores[normal_mask]
            
            anomalies_removed = len(df) - len(filtered_df)
            
            summary = {
                'anomalies_detected': anomalies_removed,
                'anomaly_percentage': (anomalies_removed / len(df)) * 100,
                'contamination_used': contamination,
                'features_used': list(numeric_df.columns)
            }
            
            return {
                'filtered_data': filtered_df,
                'summary': summary
            }
        
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {str(e)}")
            return {
                'filtered_data': df,
                'summary': {'error': str(e)}
            }
    
    def _apply_clustering(self, df: pd.DataFrame, numeric_df: pd.DataFrame, n_clusters: int) -> Dict[str, Any]:
        """Apply K-means clustering for market segmentation"""
        try:
            # Initialize and fit clusterer
            self.clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            
            # Scale the data
            if self.scaler is None:
                self.scaler = StandardScaler()
            
            scaled_data = self.scaler.fit_transform(numeric_df)
            
            # Perform clustering
            cluster_labels = self.clusterer.fit_predict(scaled_data)
            
            # Add cluster labels to the data
            filtered_df = df.copy()
            filtered_df['market_segment'] = cluster_labels
            
            # Calculate cluster statistics
            cluster_stats = {}
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_size = cluster_mask.sum()
                cluster_stats[f'cluster_{i}'] = {
                    'size': int(cluster_size),
                    'percentage': (cluster_size / len(df)) * 100
                }
            
            summary = {
                'n_clusters': n_clusters,
                'cluster_stats': cluster_stats,
                'features_used': list(numeric_df.columns),
                'inertia': float(self.clusterer.inertia_)
            }
            
            return {
                'filtered_data': filtered_df,
                'summary': summary
            }
        
        except Exception as e:
            self.logger.error(f"Error in clustering: {str(e)}")
            return {
                'filtered_data': df,
                'summary': {'error': str(e)}
            }
    
    def _apply_similarity_matching(self, df: pd.DataFrame, numeric_df: pd.DataFrame, 
                                 reference_property: Dict[str, Any], threshold: float) -> Dict[str, Any]:
        """Apply similarity matching using cosine similarity"""
        try:
            # Prepare reference property vector
            reference_vector = []
            reference_features = []
            
            for col in numeric_df.columns:
                if col in reference_property:
                    reference_vector.append(reference_property[col])
                    reference_features.append(col)
                else:
                    # Use median value for missing features
                    reference_vector.append(numeric_df[col].median())
                    reference_features.append(col)
            
            if not reference_vector:
                return {
                    'filtered_data': df,
                    'summary': {'error': 'No matching features found in reference property'}
                }
            
            # Scale the data
            if self.scaler is None:
                self.scaler = StandardScaler()
            
            scaled_data = self.scaler.fit_transform(numeric_df)
            scaled_reference = self.scaler.transform([reference_vector])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(scaled_data, scaled_reference).flatten()
            
            # Filter based on similarity threshold
            similar_mask = similarities >= threshold
            filtered_df = df[similar_mask].copy()
            
            # Add similarity scores
            filtered_df['similarity_score'] = similarities[similar_mask]
            
            # Sort by similarity (highest first)
            filtered_df = filtered_df.sort_values('similarity_score', ascending=False)
            
            summary = {
                'similar_properties': len(filtered_df),
                'similarity_threshold': threshold,
                'features_used': reference_features,
                'avg_similarity': float(similarities[similar_mask].mean()) if len(filtered_df) > 0 else 0,
                'max_similarity': float(similarities.max()),
                'min_similarity': float(similarities.min())
            }
            
            return {
                'filtered_data': filtered_df,
                'summary': summary
            }
        
        except Exception as e:
            self.logger.error(f"Error in similarity matching: {str(e)}")
            return {
                'filtered_data': df,
                'summary': {'error': str(e)}
            }
    
    def _analyze_column_for_recommendations(self, series: pd.Series, column_name: str) -> List[Dict[str, Any]]:
        """Analyze column and generate filter recommendations"""
        recommendations = []
        
        if pd.api.types.is_numeric_dtype(series):
            # Numeric column recommendations
            q25, q75 = series.quantile([0.25, 0.75])
            iqr = q75 - q25
            
            # Recommend filtering outliers
            if iqr > 0:
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                if len(outliers) > 0:
                    recommendations.append({
                        'type': 'outlier_filter',
                        'column': column_name,
                        'description': f'Remove {len(outliers)} outliers in {column_name}',
                        'suggested_filter': {
                            'min': float(lower_bound),
                            'max': float(upper_bound)
                        },
                        'confidence': 0.8
                    })
            
            # Recommend high-value filter
            if 'price' in column_name.lower() or 'value' in column_name.lower():
                high_value_threshold = series.quantile(0.8)
                recommendations.append({
                    'type': 'high_value_filter',
                    'column': column_name,
                    'description': f'Focus on high-value properties (top 20%)',
                    'suggested_filter': {
                        'min': float(high_value_threshold)
                    },
                    'confidence': 0.7
                })
        
        elif series.dtype == 'object':
            # Categorical column recommendations
            value_counts = series.value_counts()
            
            # Recommend filtering by most common values
            if len(value_counts) > 1:
                top_values = value_counts.head(3).index.tolist()
                recommendations.append({
                    'type': 'common_values_filter',
                    'column': column_name,
                    'description': f'Filter by most common {column_name} values',
                    'suggested_filter': top_values,
                    'confidence': 0.6
                })
        
        return recommendations
    
    def _recommend_ml_filters(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Recommend ML filters based on data characteristics"""
        recommendations = []
        
        numeric_df = self._prepare_numeric_data(df)
        
        if not numeric_df.empty:
            # Recommend anomaly detection for large datasets
            if len(df) > 1000:
                recommendations.append({
                    'type': 'ml_anomaly_detection',
                    'description': 'Use anomaly detection to identify unusual properties',
                    'suggested_options': {
                        'enable_anomaly_detection': True,
                        'anomaly_contamination': 0.1
                    },
                    'confidence': 0.8
                })
            
            # Recommend clustering for market segmentation
            if len(df) > 100:
                optimal_clusters = min(5, max(2, len(df) // 50))
                recommendations.append({
                    'type': 'ml_clustering',
                    'description': 'Use clustering for market segmentation analysis',
                    'suggested_options': {
                        'enable_clustering': True,
                        'n_clusters': optimal_clusters
                    },
                    'confidence': 0.7
                })
        
        return recommendations
    
    def _generate_filter_recommendations(self, original_df: pd.DataFrame, filtered_df: pd.DataFrame, 
                                       filters: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on filter results"""
        recommendations = []
        
        reduction_percentage = ((len(original_df) - len(filtered_df)) / len(original_df)) * 100
        
        if reduction_percentage > 90:
            recommendations.append("Filters are very restrictive - consider relaxing some criteria")
        elif reduction_percentage < 10:
            recommendations.append("Filters have minimal impact - consider adding more specific criteria")
        
        if len(filtered_df) < 10:
            recommendations.append("Very few results remaining - consider broadening filter ranges")
        
        return recommendations
    
    def _generate_ml_recommendations(self, ml_results: Dict[str, Any], options: MLFilterOptions) -> List[str]:
        """Generate recommendations based on ML filter results"""
        recommendations = []
        
        if 'anomaly_detection' in ml_results:
            anomaly_result = ml_results['anomaly_detection']
            if 'anomaly_percentage' in anomaly_result:
                if anomaly_result['anomaly_percentage'] > 20:
                    recommendations.append("High percentage of anomalies detected - review data quality")
                elif anomaly_result['anomaly_percentage'] < 5:
                    recommendations.append("Few anomalies detected - consider lowering contamination parameter")
        
        if 'clustering' in ml_results:
            cluster_result = ml_results['clustering']
            if 'inertia' in cluster_result:
                recommendations.append("Consider analyzing cluster characteristics for market insights")
        
        if 'similarity_matching' in ml_results:
            similarity_result = ml_results['similarity_matching']
            if 'similar_properties' in similarity_result:
                if similarity_result['similar_properties'] == 0:
                    recommendations.append("No similar properties found - consider lowering similarity threshold")
                elif similarity_result['similar_properties'] > 100:
                    recommendations.append("Many similar properties found - consider raising similarity threshold")
        
        return recommendations
    
    def save_filter_config(self, name: str, filters: Dict[str, Any], 
                          ml_options: Dict[str, Any], user_id: Optional[str] = None) -> str:
        """Save filter configuration"""
        config = FilterConfig(
            name=name,
            filters=filters,
            ml_options=ml_options,
            user_id=user_id
        )
        
        self.filter_history.append(config)
        
        # Keep only recent configurations (last 100)
        if len(self.filter_history) > 100:
            self.filter_history = self.filter_history[-100:]
        
        return f"filter_config_{len(self.filter_history)}"
    
    def load_filter_config(self, config_name: str) -> Optional[FilterConfig]:
        """Load saved filter configuration"""
        for config in self.filter_history:
            if config.name == config_name:
                return config
        return None
    
    def get_filter_history(self, user_id: Optional[str] = None) -> List[FilterConfig]:
        """Get filter history"""
        if user_id:
            return [config for config in self.filter_history if config.user_id == user_id]
        return self.filter_history.copy()