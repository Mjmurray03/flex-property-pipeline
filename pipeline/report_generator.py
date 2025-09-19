"""
Report Generator for Scalable Multi-File Pipeline
Creates comprehensive summary statistics and reports for flex property analysis
"""

import pandas as pd
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pipeline.batch_processor import BatchProcessingStats
from pipeline.result_aggregator import AggregationStats


@dataclass
class ReportSummary:
    """Summary statistics for pipeline execution report"""
    
    # File processing stats
    total_files_processed: int = 0
    successful_files: int = 0
    failed_files: int = 0
    
    # Property stats
    total_properties_analyzed: int = 0
    unique_flex_properties: int = 0
    duplicates_removed: int = 0
    
    # Score analysis
    average_flex_score: float = 0.0
    median_flex_score: float = 0.0
    highest_flex_score: float = 0.0
    lowest_flex_score: float = 0.0
    
    # Score distribution
    prime_flex_count: int = 0      # 8-10
    good_flex_count: int = 0       # 6-8
    potential_flex_count: int = 0  # 4-6
    unlikely_flex_count: int = 0   # <4
    
    # Geographic coverage
    unique_states: int = 0
    unique_cities: int = 0
    unique_counties: int = 0
    
    # Property characteristics
    average_building_size: float = 0.0
    average_lot_size: float = 0.0
    average_improvement_value: float = 0.0
    
    # Performance metrics
    processing_duration: float = 0.0
    files_per_minute: float = 0.0
    properties_per_minute: float = 0.0
    
    # Report metadata
    report_generated_date: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report summary to dictionary"""
        return {
            'file_processing': {
                'total_files_processed': self.total_files_processed,
                'successful_files': self.successful_files,
                'failed_files': self.failed_files,
                'success_rate': self.successful_files / max(1, self.total_files_processed)
            },
            'property_analysis': {
                'total_properties_analyzed': self.total_properties_analyzed,
                'unique_flex_properties': self.unique_flex_properties,
                'duplicates_removed': self.duplicates_removed,
                'deduplication_rate': self.duplicates_removed / max(1, self.total_properties_analyzed) if self.total_properties_analyzed > 0 else 0
            },
            'score_statistics': {
                'average_flex_score': round(self.average_flex_score, 2),
                'median_flex_score': round(self.median_flex_score, 2),
                'highest_flex_score': round(self.highest_flex_score, 2),
                'lowest_flex_score': round(self.lowest_flex_score, 2)
            },
            'score_distribution': {
                'prime_flex_8_to_10': self.prime_flex_count,
                'good_flex_6_to_8': self.good_flex_count,
                'potential_flex_4_to_6': self.potential_flex_count,
                'unlikely_flex_below_4': self.unlikely_flex_count
            },
            'geographic_coverage': {
                'unique_states': self.unique_states,
                'unique_cities': self.unique_cities,
                'unique_counties': self.unique_counties
            },
            'property_characteristics': {
                'average_building_size_sqft': round(self.average_building_size, 0),
                'average_lot_size_acres': round(self.average_lot_size, 2),
                'average_improvement_value': round(self.average_improvement_value, 0)
            },
            'performance_metrics': {
                'processing_duration_seconds': round(self.processing_duration, 2),
                'files_per_minute': round(self.files_per_minute, 2),
                'properties_per_minute': round(self.properties_per_minute, 2)
            },
            'report_metadata': {
                'generated_date': self.report_generated_date,
                'report_version': '1.0'
            }
        }


class ReportGenerator:
    """
    Generates comprehensive reports and statistics for pipeline execution
    
    Creates summary statistics, score distribution analysis, geographic coverage,
    and top candidates lists
    """
    
    def __init__(self):
        """Initialize ReportGenerator"""
        self.logger = logging.getLogger(__name__)
        self.report_summary = ReportSummary()
    
    def generate_comprehensive_report(self, 
                                    aggregated_df: pd.DataFrame,
                                    batch_stats: BatchProcessingStats,
                                    aggregation_stats: AggregationStats,
                                    top_candidates_count: int = 10) -> Dict[str, Any]:
        """
        Generate comprehensive pipeline execution report
        
        Args:
            aggregated_df: Final aggregated DataFrame with flex properties
            batch_stats: Statistics from batch processing
            aggregation_stats: Statistics from result aggregation
            top_candidates_count: Number of top candidates to include
            
        Returns:
            Comprehensive report dictionary
        """
        self.logger.info("Generating comprehensive pipeline report")
        
        try:
            # Generate report summary
            summary = self._generate_report_summary(aggregated_df, batch_stats, aggregation_stats)
            
            # Generate detailed analyses
            score_analysis = self._analyze_score_distribution(aggregated_df)
            geographic_analysis = self._analyze_geographic_coverage(aggregated_df)
            property_analysis = self._analyze_property_characteristics(aggregated_df)
            top_candidates = self._get_top_candidates(aggregated_df, top_candidates_count)
            
            # Compile comprehensive report
            report = {
                'executive_summary': summary.to_dict(),
                'detailed_analysis': {
                    'score_distribution': score_analysis,
                    'geographic_coverage': geographic_analysis,
                    'property_characteristics': property_analysis
                },
                'top_candidates': top_candidates,
                'processing_details': {
                    'batch_processing': batch_stats.to_dict() if batch_stats else {},
                    'result_aggregation': aggregation_stats.to_dict() if aggregation_stats else {}
                },
                'data_quality': self._assess_data_quality(aggregated_df),
                'recommendations': self._generate_recommendations(aggregated_df, summary)
            }
            
            self.logger.info(f"Report generated successfully with {len(aggregated_df)} properties analyzed")
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive report: {e}")
            return {
                'error': str(e),
                'generated_date': datetime.now().isoformat()
            }
    
    def _generate_report_summary(self, 
                               df: pd.DataFrame, 
                               batch_stats: BatchProcessingStats,
                               aggregation_stats: AggregationStats) -> ReportSummary:
        """Generate executive summary statistics"""
        
        summary = ReportSummary()
        
        # File processing stats
        if batch_stats:
            summary.total_files_processed = batch_stats.processed_files
            summary.successful_files = batch_stats.successful_files
            summary.failed_files = batch_stats.failed_files
            summary.processing_duration = batch_stats.processing_duration
            
            # Performance metrics
            if batch_stats.processing_duration > 0:
                summary.files_per_minute = (batch_stats.successful_files * 60) / batch_stats.processing_duration
                summary.properties_per_minute = (batch_stats.total_properties * 60) / batch_stats.processing_duration
        
        # Aggregation stats
        if aggregation_stats:
            summary.total_properties_analyzed = aggregation_stats.total_properties_before
            summary.duplicates_removed = aggregation_stats.duplicates_removed
            summary.unique_states = aggregation_stats.unique_states
            summary.unique_cities = aggregation_stats.unique_cities
        
        # Property analysis from DataFrame
        if not df.empty:
            summary.unique_flex_properties = len(df)
            
            # Score statistics
            if 'flex_score' in df.columns:
                scores = df['flex_score'].dropna()
                if not scores.empty:
                    summary.average_flex_score = scores.mean()
                    summary.median_flex_score = scores.median()
                    summary.highest_flex_score = scores.max()
                    summary.lowest_flex_score = scores.min()
                    
                    # Score distribution
                    summary.prime_flex_count = len(scores[(scores >= 8) & (scores <= 10)])
                    summary.good_flex_count = len(scores[(scores >= 6) & (scores < 8)])
                    summary.potential_flex_count = len(scores[(scores >= 4) & (scores < 6)])
                    summary.unlikely_flex_count = len(scores[scores < 4])
            
            # Property characteristics
            if 'acres' in df.columns:
                lot_sizes = df['acres'].dropna()
                summary.average_lot_size = lot_sizes.mean() if not lot_sizes.empty else 0
            
            if 'improvement_value' in df.columns:
                improvements = df['improvement_value'].dropna()
                summary.average_improvement_value = improvements.mean() if not improvements.empty else 0
        
        self.report_summary = summary
        return summary
    
    def _analyze_score_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze flex score distribution in detail"""
        
        if df.empty or 'flex_score' not in df.columns:
            return {'error': 'No flex score data available'}
        
        scores = df['flex_score'].dropna()
        
        if scores.empty:
            return {'error': 'No valid flex scores found'}
        
        # Score ranges analysis
        score_ranges = {
            'excellent_9_to_10': len(scores[(scores >= 9) & (scores <= 10)]),
            'very_good_8_to_9': len(scores[(scores >= 8) & (scores < 9)]),
            'good_7_to_8': len(scores[(scores >= 7) & (scores < 8)]),
            'fair_6_to_7': len(scores[(scores >= 6) & (scores < 7)]),
            'marginal_5_to_6': len(scores[(scores >= 5) & (scores < 6)]),
            'poor_4_to_5': len(scores[(scores >= 4) & (scores < 5)]),
            'very_poor_below_4': len(scores[scores < 4])
        }
        
        # Statistical analysis
        percentiles = scores.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
        
        # Score component analysis (if available)
        component_analysis = {}
        score_components = ['zoning_score', 'size_score', 'building_score', 'location_score', 'activity_score', 'value_score']
        
        for component in score_components:
            if component in df.columns:
                component_scores = df[component].dropna()
                if not component_scores.empty:
                    component_analysis[component] = {
                        'average': round(component_scores.mean(), 2),
                        'median': round(component_scores.median(), 2),
                        'std_dev': round(component_scores.std(), 2)
                    }
        
        return {
            'score_ranges': score_ranges,
            'statistical_summary': {
                'count': len(scores),
                'mean': round(scores.mean(), 2),
                'median': round(scores.median(), 2),
                'std_dev': round(scores.std(), 2),
                'min': round(scores.min(), 2),
                'max': round(scores.max(), 2)
            },
            'percentiles': {f'p{int(k*100)}': round(v, 2) for k, v in percentiles.items()},
            'component_analysis': component_analysis
        }
    
    def _analyze_geographic_coverage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze geographic distribution of properties"""
        
        if df.empty:
            return {'error': 'No data available for geographic analysis'}
        
        analysis = {}
        
        # State analysis
        if 'state' in df.columns:
            state_counts = df['state'].value_counts().to_dict()
            state_scores = df.groupby('state')['flex_score'].agg(['count', 'mean', 'max']).round(2)
            
            analysis['states'] = {
                'total_states': len(state_counts),
                'property_counts_by_state': state_counts,
                'score_analysis_by_state': state_scores.to_dict('index') if not state_scores.empty else {}
            }
        
        # City analysis
        if 'city' in df.columns:
            city_counts = df['city'].value_counts().head(20).to_dict()  # Top 20 cities
            city_scores = df.groupby('city')['flex_score'].agg(['count', 'mean', 'max']).round(2)
            top_cities_by_score = city_scores.sort_values('mean', ascending=False).head(10)
            
            analysis['cities'] = {
                'total_cities': df['city'].nunique(),
                'top_cities_by_count': city_counts,
                'top_cities_by_avg_score': top_cities_by_score.to_dict('index') if not top_cities_by_score.empty else {}
            }
        
        # County analysis (if available)
        if 'county' in df.columns:
            county_counts = df['county'].value_counts().to_dict()
            analysis['counties'] = {
                'total_counties': len(county_counts),
                'property_counts_by_county': county_counts
            }
        
        # Zoning analysis
        if 'zoning' in df.columns:
            zoning_counts = df['zoning'].value_counts().to_dict()
            zoning_scores = df.groupby('zoning')['flex_score'].agg(['count', 'mean']).round(2)
            
            analysis['zoning'] = {
                'total_zoning_types': len(zoning_counts),
                'property_counts_by_zoning': zoning_counts,
                'avg_scores_by_zoning': zoning_scores.to_dict('index') if not zoning_scores.empty else {}
            }
        
        return analysis
    
    def _analyze_property_characteristics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze property size and value characteristics"""
        
        if df.empty:
            return {'error': 'No data available for property analysis'}
        
        analysis = {}
        
        # Lot size analysis
        if 'acres' in df.columns:
            acres = df['acres'].dropna()
            if not acres.empty:
                analysis['lot_size'] = {
                    'average_acres': round(acres.mean(), 2),
                    'median_acres': round(acres.median(), 2),
                    'min_acres': round(acres.min(), 2),
                    'max_acres': round(acres.max(), 2),
                    'size_distribution': {
                        'under_1_acre': len(acres[acres < 1]),
                        '1_to_3_acres': len(acres[(acres >= 1) & (acres < 3)]),
                        '3_to_5_acres': len(acres[(acres >= 3) & (acres < 5)]),
                        '5_to_10_acres': len(acres[(acres >= 5) & (acres < 10)]),
                        'over_10_acres': len(acres[acres >= 10])
                    }
                }
        
        # Building value analysis
        if 'improvement_value' in df.columns:
            improvements = df['improvement_value'].dropna()
            if not improvements.empty:
                analysis['improvement_value'] = {
                    'average_value': round(improvements.mean(), 0),
                    'median_value': round(improvements.median(), 0),
                    'min_value': round(improvements.min(), 0),
                    'max_value': round(improvements.max(), 0),
                    'value_distribution': {
                        'under_250k': len(improvements[improvements < 250000]),
                        '250k_to_500k': len(improvements[(improvements >= 250000) & (improvements < 500000)]),
                        '500k_to_1m': len(improvements[(improvements >= 500000) & (improvements < 1000000)]),
                        '1m_to_2m': len(improvements[(improvements >= 1000000) & (improvements < 2000000)]),
                        'over_2m': len(improvements[improvements >= 2000000])
                    }
                }
        
        # Land value analysis
        if 'land_market_value' in df.columns:
            land_values = df['land_market_value'].dropna()
            if not land_values.empty:
                analysis['land_value'] = {
                    'average_value': round(land_values.mean(), 0),
                    'median_value': round(land_values.median(), 0),
                    'min_value': round(land_values.min(), 0),
                    'max_value': round(land_values.max(), 0)
                }
        
        # Value ratios
        if 'improvement_value' in df.columns and 'land_market_value' in df.columns:
            df_clean = df[['improvement_value', 'land_market_value']].dropna()
            df_clean = df_clean[(df_clean['land_market_value'] > 0)]  # Avoid division by zero
            
            if not df_clean.empty:
                ratios = df_clean['improvement_value'] / df_clean['land_market_value']
                analysis['value_ratios'] = {
                    'avg_improvement_to_land_ratio': round(ratios.mean(), 2),
                    'median_improvement_to_land_ratio': round(ratios.median(), 2)
                }
        
        return analysis
    
    def _get_top_candidates(self, df: pd.DataFrame, count: int = 10) -> List[Dict[str, Any]]:
        """Get top flex property candidates with key details"""
        
        if df.empty or 'flex_score' not in df.columns:
            return []
        
        # Sort by flex score and get top candidates
        top_df = df.nlargest(count, 'flex_score')
        
        candidates = []
        
        for _, row in top_df.iterrows():
            candidate = {
                'rank': len(candidates) + 1,
                'flex_score': round(row.get('flex_score', 0), 2),
                'classification': row.get('flex_classification', 'Unknown'),
                'address': row.get('site_address', 'Unknown'),
                'city': row.get('city', 'Unknown'),
                'state': row.get('state', 'Unknown'),
                'lot_size_acres': round(row.get('acres', 0), 2),
                'zoning': row.get('zoning', 'Unknown'),
                'improvement_value': int(row.get('improvement_value', 0)),
                'land_value': int(row.get('land_market_value', 0)),
                'source_file': row.get('source_filename', 'Unknown')
            }
            
            # Add score components if available
            score_components = {}
            for component in ['zoning_score', 'size_score', 'building_score', 'location_score', 'activity_score', 'value_score']:
                if component in row and pd.notna(row[component]):
                    score_components[component] = round(row[component], 1)
            
            if score_components:
                candidate['score_breakdown'] = score_components
            
            candidates.append(candidate)
        
        return candidates
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality and completeness"""
        
        if df.empty:
            return {'error': 'No data available for quality assessment'}
        
        total_records = len(df)
        
        # Check completeness of key fields
        key_fields = ['site_address', 'city', 'state', 'flex_score', 'acres', 'zoning']
        completeness = {}
        
        for field in key_fields:
            if field in df.columns:
                non_null_count = df[field].notna().sum()
                completeness[field] = {
                    'complete_records': non_null_count,
                    'missing_records': total_records - non_null_count,
                    'completeness_rate': round(non_null_count / total_records, 3)
                }
        
        # Check for data anomalies
        anomalies = []
        
        # Check for unrealistic flex scores
        if 'flex_score' in df.columns:
            invalid_scores = df[(df['flex_score'] < 0) | (df['flex_score'] > 10)]
            if not invalid_scores.empty:
                anomalies.append(f"{len(invalid_scores)} records with invalid flex scores (outside 0-10 range)")
        
        # Check for negative values in numeric fields
        numeric_fields = ['acres', 'improvement_value', 'land_market_value']
        for field in numeric_fields:
            if field in df.columns:
                negative_values = df[df[field] < 0]
                if not negative_values.empty:
                    anomalies.append(f"{len(negative_values)} records with negative {field}")
        
        return {
            'total_records': total_records,
            'field_completeness': completeness,
            'data_anomalies': anomalies,
            'overall_quality_score': self._calculate_quality_score(completeness, anomalies)
        }
    
    def _calculate_quality_score(self, completeness: Dict, anomalies: List) -> float:
        """Calculate overall data quality score (0-100)"""
        
        if not completeness:
            return 0.0
        
        # Base score from completeness rates
        completeness_rates = [field_info['completeness_rate'] for field_info in completeness.values()]
        avg_completeness = sum(completeness_rates) / len(completeness_rates) if completeness_rates else 0
        
        # Penalty for anomalies
        anomaly_penalty = min(len(anomalies) * 5, 30)  # Max 30 point penalty
        
        quality_score = (avg_completeness * 100) - anomaly_penalty
        return max(0, min(100, quality_score))
    
    def _generate_recommendations(self, df: pd.DataFrame, summary: ReportSummary) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        
        if df.empty:
            recommendations.append("No flex properties found. Consider adjusting scoring criteria or expanding data sources.")
            return recommendations
        
        # Score-based recommendations
        if summary.prime_flex_count > 0:
            recommendations.append(f"Prioritize {summary.prime_flex_count} prime flex properties (score 8-10) for immediate acquisition consideration.")
        
        if summary.good_flex_count > summary.prime_flex_count * 2:
            recommendations.append(f"Consider detailed site visits for {summary.good_flex_count} good flex properties (score 6-8) to identify upgrade potential.")
        
        # Geographic recommendations
        if 'state' in df.columns and df['state'].nunique() > 1:
            state_analysis = df.groupby('state')['flex_score'].agg(['count', 'mean']).sort_values('mean', ascending=False)
            top_state = state_analysis.index[0]
            recommendations.append(f"Focus expansion efforts in {top_state} which shows highest average flex scores.")
        
        # Size-based recommendations
        if 'acres' in df.columns:
            ideal_size_properties = df[(df['acres'] >= 1) & (df['acres'] <= 5)]
            if len(ideal_size_properties) > 0:
                avg_score_ideal = ideal_size_properties['flex_score'].mean()
                recommendations.append(f"Properties between 1-5 acres show strong flex potential (avg score: {avg_score_ideal:.1f}).")
        
        # Data quality recommendations
        if summary.total_files_processed > summary.successful_files:
            failure_rate = (summary.failed_files / summary.total_files_processed) * 100
            if failure_rate > 10:
                recommendations.append(f"Address data quality issues - {failure_rate:.1f}% of files failed processing.")
        
        # Performance recommendations
        if summary.files_per_minute < 1:
            recommendations.append("Consider increasing processing workers or optimizing file formats to improve throughput.")
        
        return recommendations
    
    def export_report(self, report: Dict[str, Any], output_path: str, format_type: str = 'json') -> bool:
        """
        Export report to file
        
        Args:
            report: Report dictionary to export
            output_path: Path for output file
            format_type: Export format ('json', 'excel', 'html')
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if format_type.lower() == 'json':
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
            
            elif format_type.lower() == 'excel':
                # Create Excel report with multiple sheets
                with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                    # Executive summary sheet
                    summary_df = pd.DataFrame([report.get('executive_summary', {})])
                    summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
                    
                    # Top candidates sheet
                    if 'top_candidates' in report and report['top_candidates']:
                        candidates_df = pd.DataFrame(report['top_candidates'])
                        candidates_df.to_excel(writer, sheet_name='Top Candidates', index=False)
                    
                    # Score distribution sheet
                    if 'detailed_analysis' in report and 'score_distribution' in report['detailed_analysis']:
                        score_data = report['detailed_analysis']['score_distribution']
                        if 'score_ranges' in score_data:
                            score_df = pd.DataFrame([score_data['score_ranges']])
                            score_df.to_excel(writer, sheet_name='Score Distribution', index=False)
            
            elif format_type.lower() == 'html':
                # Generate HTML report
                html_content = self._generate_html_report(report)
                with open(output_file, 'w') as f:
                    f.write(html_content)
            
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
            
            self.logger.info(f"Report exported successfully to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export report: {e}")
            return False
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report content"""
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Flex Property Pipeline Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Flex Property Pipeline Report</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>"""
        
        # Add executive summary
        if 'executive_summary' in report:
            html += "<div class='section'><h2>Executive Summary</h2>"
            summary = report['executive_summary']
            
            if 'file_processing' in summary:
                fp = summary['file_processing']
                html += f"<div class='metric'>Files Processed: {fp.get('total_files_processed', 0)}</div>"
                html += f"<div class='metric'>Success Rate: {fp.get('success_rate', 0):.1%}</div>"
            
            if 'property_analysis' in summary:
                pa = summary['property_analysis']
                html += f"<div class='metric'>Flex Properties: {pa.get('unique_flex_properties', 0)}</div>"
                html += f"<div class='metric'>Duplicates Removed: {pa.get('duplicates_removed', 0)}</div>"
            
            html += "</div>"
        
        # Add top candidates table
        if 'top_candidates' in report and report['top_candidates']:
            html += "<div class='section'><h2>Top Flex Property Candidates</h2>"
            html += "<table><tr><th>Rank</th><th>Score</th><th>Address</th><th>City</th><th>State</th><th>Acres</th><th>Zoning</th></tr>"
            
            for candidate in report['top_candidates'][:10]:
                html += f"""
                <tr>
                    <td>{candidate.get('rank', '')}</td>
                    <td>{candidate.get('flex_score', '')}</td>
                    <td>{candidate.get('address', '')}</td>
                    <td>{candidate.get('city', '')}</td>
                    <td>{candidate.get('state', '')}</td>
                    <td>{candidate.get('lot_size_acres', '')}</td>
                    <td>{candidate.get('zoning', '')}</td>
                </tr>
                """
            
            html += "</table></div>"
        
        html += "</body></html>"
        return html


# Convenience function for simple report generation
def generate_pipeline_report(aggregated_df: pd.DataFrame,
                           batch_stats: BatchProcessingStats = None,
                           aggregation_stats: AggregationStats = None,
                           output_path: str = None) -> Dict[str, Any]:
    """
    Simple function to generate pipeline report
    
    Args:
        aggregated_df: Final aggregated DataFrame
        batch_stats: Batch processing statistics
        aggregation_stats: Aggregation statistics
        output_path: Optional path to export report
        
    Returns:
        Report dictionary
    """
    generator = ReportGenerator()
    
    report = generator.generate_comprehensive_report(
        aggregated_df=aggregated_df,
        batch_stats=batch_stats,
        aggregation_stats=aggregation_stats
    )
    
    if output_path:
        generator.export_report(report, output_path, 'json')
    
    return report