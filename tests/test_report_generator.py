"""
Unit tests for ReportGenerator class
Tests comprehensive reporting and statistics functionality
"""

import unittest
import pandas as pd
import tempfile
import os
import json
from pathlib import Path
from datetime import datetime

from pipeline.report_generator import ReportGenerator, ReportSummary, generate_pipeline_report
from pipeline.batch_processor import BatchProcessingStats
from pipeline.result_aggregator import AggregationStats


class TestReportSummary(unittest.TestCase):
    """Test ReportSummary dataclass"""
    
    def test_summary_creation(self):
        """Test creating ReportSummary"""
        summary = ReportSummary(
            total_files_processed=10,
            successful_files=8,
            failed_files=2,
            unique_flex_properties=150,
            average_flex_score=7.2
        )
        
        self.assertEqual(summary.total_files_processed, 10)
        self.assertEqual(summary.successful_files, 8)
        self.assertEqual(summary.failed_files, 2)
        self.assertEqual(summary.unique_flex_properties, 150)
        self.assertEqual(summary.average_flex_score, 7.2)
    
    def test_summary_to_dict(self):
        """Test converting summary to dictionary"""
        summary = ReportSummary(
            total_files_processed=5,
            successful_files=4,
            failed_files=1,
            unique_flex_properties=100,
            duplicates_removed=25,
            average_flex_score=6.8,
            prime_flex_count=20,
            good_flex_count=35
        )
        
        summary_dict = summary.to_dict()
        
        # Check structure
        self.assertIn('file_processing', summary_dict)
        self.assertIn('property_analysis', summary_dict)
        self.assertIn('score_statistics', summary_dict)
        self.assertIn('score_distribution', summary_dict)
        
        # Check calculated values
        self.assertEqual(summary_dict['file_processing']['success_rate'], 0.8)  # 4/5
        # Note: deduplication_rate calculation needs total_properties_analyzed to be set
        summary.total_properties_analyzed = 100  # Set this for proper calculation
        summary_dict = summary.to_dict()  # Recalculate
        self.assertEqual(summary_dict['property_analysis']['deduplication_rate'], 0.25)  # 25/100
        self.assertEqual(summary_dict['score_distribution']['prime_flex_8_to_10'], 20)


class TestReportGenerator(unittest.TestCase):
    """Test ReportGenerator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = ReportGenerator()
        
        # Create sample aggregated data
        self.sample_df = pd.DataFrame({
            'parcel_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
            'site_address': ['123 Main St', '456 Oak Ave', '789 Pine Rd', '321 Elm St', '654 Maple Dr'],
            'city': ['Boca Raton', 'Delray Beach', 'Boynton Beach', 'West Palm Beach', 'Lake Worth'],
            'state': ['FL', 'FL', 'FL', 'FL', 'FL'],
            'county': ['Palm Beach', 'Palm Beach', 'Palm Beach', 'Palm Beach', 'Palm Beach'],
            'zoning': ['IL', 'IP', 'CG', 'IL', 'IG'],
            'acres': [2.5, 1.8, 5.0, 3.2, 1.2],
            'flex_score': [9.2, 8.1, 6.8, 7.5, 5.2],
            'flex_classification': ['PRIME_FLEX', 'PRIME_FLEX', 'GOOD_FLEX', 'GOOD_FLEX', 'POTENTIAL_FLEX'],
            'zoning_score': [10, 9, 6, 10, 8],
            'size_score': [8, 7, 9, 8, 6],
            'building_score': [9, 8, 6, 7, 5],
            'location_score': [9, 8, 7, 7, 6],
            'activity_score': [8, 7, 6, 6, 4],
            'value_score': [9, 8, 7, 8, 5],
            'improvement_value': [750000, 500000, 300000, 600000, 250000],
            'land_market_value': [300000, 200000, 150000, 250000, 100000],
            'source_filename': ['file1.xlsx', 'file1.xlsx', 'file2.xlsx', 'file2.xlsx', 'file3.xlsx']
        })
        
        # Create sample batch stats
        self.batch_stats = BatchProcessingStats(
            total_files=3,
            processed_files=3,
            successful_files=3,
            failed_files=0,
            total_properties=500,
            total_flex_candidates=5,
            processing_duration=120.0,
            average_processing_time=40.0
        )
        
        # Create sample aggregation stats
        self.aggregation_stats = AggregationStats(
            total_input_files=3,
            successful_files=3,
            total_properties_before=10,
            total_properties_after=5,
            duplicates_removed=5,
            unique_addresses=5,
            unique_cities=5,
            unique_states=1
        )
    
    def test_generator_initialization(self):
        """Test ReportGenerator initialization"""
        generator = ReportGenerator()
        
        self.assertIsNotNone(generator.logger)
        self.assertIsNotNone(generator.report_summary)
    
    def test_generate_comprehensive_report(self):
        """Test generating comprehensive report"""
        report = self.generator.generate_comprehensive_report(
            aggregated_df=self.sample_df,
            batch_stats=self.batch_stats,
            aggregation_stats=self.aggregation_stats,
            top_candidates_count=3
        )
        
        # Check report structure
        self.assertIn('executive_summary', report)
        self.assertIn('detailed_analysis', report)
        self.assertIn('top_candidates', report)
        self.assertIn('processing_details', report)
        self.assertIn('data_quality', report)
        self.assertIn('recommendations', report)
        
        # Check executive summary
        exec_summary = report['executive_summary']
        self.assertIn('file_processing', exec_summary)
        self.assertIn('property_analysis', exec_summary)
        self.assertIn('score_statistics', exec_summary)
        
        # Check top candidates
        top_candidates = report['top_candidates']
        self.assertEqual(len(top_candidates), 3)
        self.assertEqual(top_candidates[0]['rank'], 1)
        self.assertEqual(top_candidates[0]['flex_score'], 9.2)  # Highest score
        
        # Check detailed analysis
        detailed = report['detailed_analysis']
        self.assertIn('score_distribution', detailed)
        self.assertIn('geographic_coverage', detailed)
        self.assertIn('property_characteristics', detailed)
    
    def test_generate_report_empty_dataframe(self):
        """Test generating report with empty DataFrame"""
        empty_df = pd.DataFrame()
        
        report = self.generator.generate_comprehensive_report(
            aggregated_df=empty_df,
            batch_stats=self.batch_stats,
            aggregation_stats=self.aggregation_stats
        )
        
        # Should still generate report structure
        self.assertIn('executive_summary', report)
        self.assertIn('top_candidates', report)
        
        # Top candidates should be empty
        self.assertEqual(len(report['top_candidates']), 0)
    
    def test_analyze_score_distribution(self):
        """Test score distribution analysis"""
        analysis = self.generator._analyze_score_distribution(self.sample_df)
        
        self.assertIn('score_ranges', analysis)
        self.assertIn('statistical_summary', analysis)
        self.assertIn('percentiles', analysis)
        self.assertIn('component_analysis', analysis)
        
        # Check score ranges
        score_ranges = analysis['score_ranges']
        self.assertEqual(score_ranges['excellent_9_to_10'], 1)  # Score 9.2
        self.assertEqual(score_ranges['very_good_8_to_9'], 1)   # Score 8.1
        
        # Check statistical summary
        stats = analysis['statistical_summary']
        self.assertEqual(stats['count'], 5)
        self.assertAlmostEqual(stats['mean'], 7.36, places=1)  # Average of scores
        
        # Check component analysis
        components = analysis['component_analysis']
        self.assertIn('zoning_score', components)
        self.assertIn('size_score', components)
    
    def test_analyze_geographic_coverage(self):
        """Test geographic coverage analysis"""
        analysis = self.generator._analyze_geographic_coverage(self.sample_df)
        
        self.assertIn('states', analysis)
        self.assertIn('cities', analysis)
        self.assertIn('counties', analysis)
        self.assertIn('zoning', analysis)
        
        # Check state analysis
        states = analysis['states']
        self.assertEqual(states['total_states'], 1)  # Only FL
        self.assertIn('FL', states['property_counts_by_state'])
        self.assertEqual(states['property_counts_by_state']['FL'], 5)
        
        # Check city analysis
        cities = analysis['cities']
        self.assertEqual(cities['total_cities'], 5)  # 5 unique cities
        
        # Check zoning analysis
        zoning = analysis['zoning']
        self.assertIn('IL', zoning['property_counts_by_zoning'])
    
    def test_analyze_property_characteristics(self):
        """Test property characteristics analysis"""
        analysis = self.generator._analyze_property_characteristics(self.sample_df)
        
        self.assertIn('lot_size', analysis)
        self.assertIn('improvement_value', analysis)
        self.assertIn('land_value', analysis)
        self.assertIn('value_ratios', analysis)
        
        # Check lot size analysis
        lot_size = analysis['lot_size']
        self.assertAlmostEqual(lot_size['average_acres'], 2.74, places=1)  # Average of acres
        self.assertIn('size_distribution', lot_size)
        
        # Check improvement value analysis
        improvement = analysis['improvement_value']
        self.assertEqual(improvement['average_value'], 480000)  # Average of improvement values
        self.assertIn('value_distribution', improvement)
        
        # Check value ratios
        ratios = analysis['value_ratios']
        self.assertIn('avg_improvement_to_land_ratio', ratios)
    
    def test_get_top_candidates(self):
        """Test getting top candidates"""
        candidates = self.generator._get_top_candidates(self.sample_df, count=3)
        
        self.assertEqual(len(candidates), 3)
        
        # Check first candidate (highest score)
        top_candidate = candidates[0]
        self.assertEqual(top_candidate['rank'], 1)
        self.assertEqual(top_candidate['flex_score'], 9.2)
        self.assertEqual(top_candidate['address'], '123 Main St')
        self.assertEqual(top_candidate['city'], 'Boca Raton')
        self.assertIn('score_breakdown', top_candidate)
        
        # Check score breakdown
        breakdown = top_candidate['score_breakdown']
        self.assertIn('zoning_score', breakdown)
        self.assertIn('size_score', breakdown)
        
        # Check sorting (scores should be descending)
        scores = [c['flex_score'] for c in candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_assess_data_quality(self):
        """Test data quality assessment"""
        assessment = self.generator._assess_data_quality(self.sample_df)
        
        self.assertIn('total_records', assessment)
        self.assertIn('field_completeness', assessment)
        self.assertIn('data_anomalies', assessment)
        self.assertIn('overall_quality_score', assessment)
        
        # Check total records
        self.assertEqual(assessment['total_records'], 5)
        
        # Check field completeness
        completeness = assessment['field_completeness']
        self.assertIn('site_address', completeness)
        self.assertIn('flex_score', completeness)
        
        # All fields should be complete in our test data
        for field_info in completeness.values():
            self.assertEqual(field_info['completeness_rate'], 1.0)
        
        # Check quality score (should be high for complete data)
        self.assertGreaterEqual(assessment['overall_quality_score'], 90)
    
    def test_assess_data_quality_with_missing_data(self):
        """Test data quality assessment with missing data"""
        # Create DataFrame with missing values
        df_with_missing = self.sample_df.copy()
        df_with_missing.loc[0, 'site_address'] = None
        df_with_missing.loc[1, 'flex_score'] = None
        df_with_missing.loc[2, 'acres'] = -1  # Invalid value
        
        assessment = self.generator._assess_data_quality(df_with_missing)
        
        # Check completeness rates
        completeness = assessment['field_completeness']
        self.assertEqual(completeness['site_address']['completeness_rate'], 0.8)  # 4/5
        self.assertEqual(completeness['flex_score']['completeness_rate'], 0.8)    # 4/5
        
        # Check anomalies
        anomalies = assessment['data_anomalies']
        self.assertGreater(len(anomalies), 0)  # Should detect negative acres
        
        # Quality score should be lower
        self.assertLess(assessment['overall_quality_score'], 90)
    
    def test_generate_recommendations(self):
        """Test recommendation generation"""
        summary = ReportSummary(
            prime_flex_count=2,
            good_flex_count=2,
            total_files_processed=3,
            successful_files=3,
            files_per_minute=1.5
        )
        
        recommendations = self.generator._generate_recommendations(self.sample_df, summary)
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Should include recommendation about prime flex properties
        prime_rec = any('prime flex' in rec.lower() for rec in recommendations)
        self.assertTrue(prime_rec)
    
    def test_export_report_json(self):
        """Test exporting report to JSON"""
        report = {
            'executive_summary': {'test': 'data'},
            'top_candidates': [{'rank': 1, 'score': 9.2}]
        }
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            success = self.generator.export_report(report, tmp_path, 'json')
            
            self.assertTrue(success)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Verify file content
            with open(tmp_path, 'r') as f:
                exported_data = json.load(f)
            
            self.assertEqual(exported_data['executive_summary']['test'], 'data')
            self.assertEqual(len(exported_data['top_candidates']), 1)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_export_report_excel(self):
        """Test exporting report to Excel"""
        report = {
            'executive_summary': {
                'file_processing': {'total_files_processed': 3},
                'property_analysis': {'unique_flex_properties': 5}
            },
            'top_candidates': [
                {'rank': 1, 'flex_score': 9.2, 'address': '123 Main St'},
                {'rank': 2, 'flex_score': 8.1, 'address': '456 Oak Ave'}
            ],
            'detailed_analysis': {
                'score_distribution': {
                    'score_ranges': {'excellent_9_to_10': 1, 'very_good_8_to_9': 1}
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            success = self.generator.export_report(report, tmp_path, 'excel')
            
            self.assertTrue(success)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Verify Excel file can be read
            excel_data = pd.read_excel(tmp_path, sheet_name=None, engine='openpyxl')
            self.assertIn('Executive Summary', excel_data)
            self.assertIn('Top Candidates', excel_data)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_export_report_html(self):
        """Test exporting report to HTML"""
        report = {
            'executive_summary': {
                'file_processing': {'total_files_processed': 3, 'success_rate': 1.0},
                'property_analysis': {'unique_flex_properties': 5}
            },
            'top_candidates': [
                {'rank': 1, 'flex_score': 9.2, 'address': '123 Main St', 'city': 'Boca Raton', 'state': 'FL', 'lot_size_acres': 2.5, 'zoning': 'IL'}
            ]
        }
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            success = self.generator.export_report(report, tmp_path, 'html')
            
            self.assertTrue(success)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Verify HTML content
            with open(tmp_path, 'r') as f:
                html_content = f.read()
            
            self.assertIn('<html>', html_content)
            self.assertIn('Flex Property Pipeline Report', html_content)
            self.assertIn('123 Main St', html_content)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""
    
    def test_generate_pipeline_report(self):
        """Test generate_pipeline_report convenience function"""
        df = pd.DataFrame({
            'flex_score': [9.2, 8.1],
            'site_address': ['123 Main St', '456 Oak Ave'],
            'city': ['Boca Raton', 'Delray Beach'],
            'state': ['FL', 'FL']
        })
        
        batch_stats = BatchProcessingStats(
            total_files=2,
            successful_files=2,
            processing_duration=60.0
        )
        
        report = generate_pipeline_report(
            aggregated_df=df,
            batch_stats=batch_stats
        )
        
        self.assertIn('executive_summary', report)
        self.assertIn('top_candidates', report)
        self.assertEqual(len(report['top_candidates']), 2)
    
    def test_generate_pipeline_report_with_export(self):
        """Test generate_pipeline_report with file export"""
        df = pd.DataFrame({
            'flex_score': [9.2],
            'site_address': ['123 Main St'],
            'city': ['Boca Raton'],
            'state': ['FL']
        })
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            report = generate_pipeline_report(
                aggregated_df=df,
                output_path=tmp_path
            )
            
            self.assertIn('executive_summary', report)
            self.assertTrue(os.path.exists(tmp_path))
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


if __name__ == '__main__':
    unittest.main()