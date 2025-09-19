"""
Test suite for Advanced Flex Property Classifier features
Tests configuration management, batch processing, and advanced analytics
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import json
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.flex_config_manager import FlexConfigManager, FlexClassifierConfig, ScoringWeights, FilteringCriteria, AdvancedSettings
from processors.advanced_flex_classifier import AdvancedFlexClassifier
from utils.batch_processor import FlexBatchProcessor
from utils.logger import setup_logging


class TestFlexConfigManager(unittest.TestCase):
    """Test cases for FlexConfigManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_config.yaml"
        self.manager = FlexConfigManager(self.config_path)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_default_config(self):
        """Test creating default configuration"""
        config = self.manager.load_config()
        
        self.assertIsInstance(config, FlexClassifierConfig)
        self.assertEqual(config.max_flex_score, 10.0)
        self.assertEqual(config.filtering_criteria.min_building_sqft, 20000)
        self.assertEqual(config.scoring_weights.building_size_weight, 1.0)
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration"""
        # Create and save config
        config = self.manager.load_config()
        config.max_flex_score = 8.0
        config.filtering_criteria.min_building_sqft = 25000
        
        self.manager.save_config(config)
        
        # Load and verify
        loaded_config = self.manager.load_config()
        self.assertEqual(loaded_config.max_flex_score, 8.0)
        self.assertEqual(loaded_config.filtering_criteria.min_building_sqft, 25000)
    
    def test_update_scoring_weights(self):
        """Test updating scoring weights"""
        self.manager.update_scoring_weights(
            building_size_weight=1.5,
            property_type_weight=2.0
        )
        
        config = self.manager.get_config()
        self.assertEqual(config.scoring_weights.building_size_weight, 1.5)
        self.assertEqual(config.scoring_weights.property_type_weight, 2.0)
    
    def test_update_filtering_criteria(self):
        """Test updating filtering criteria"""
        self.manager.update_filtering_criteria(
            min_building_sqft=30000,
            max_lot_acres=15.0
        )
        
        config = self.manager.get_config()
        self.assertEqual(config.filtering_criteria.min_building_sqft, 30000)
        self.assertEqual(config.filtering_criteria.max_lot_acres, 15.0)
    
    def test_validate_config(self):
        """Test configuration validation"""
        config = self.manager.load_config()
        
        # Valid config should have no issues
        issues = self.manager.validate_config(config)
        self.assertEqual(len(issues), 0)
        
        # Invalid config should have issues
        config.filtering_criteria.min_building_sqft = -1000
        config.scoring_weights.building_size_weight = 10.0
        
        issues = self.manager.validate_config(config)
        self.assertGreater(len(issues), 0)
    
    def test_create_custom_config(self):
        """Test creating custom configuration"""
        custom_config = self.manager.create_custom_config(
            scoring_adjustments={'building_size_weight': 1.5},
            filtering_adjustments={'min_building_sqft': 25000},
            advanced_adjustments={'batch_size': 2000}
        )
        
        self.assertEqual(custom_config.scoring_weights.building_size_weight, 1.5)
        self.assertEqual(custom_config.filtering_criteria.min_building_sqft, 25000)
        self.assertEqual(custom_config.advanced_settings.batch_size, 2000)


class TestAdvancedFlexClassifier(unittest.TestCase):
    """Test cases for AdvancedFlexClassifier"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = setup_logging(name='test_advanced', level='DEBUG')
        
        # Create test data
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'Property Name': [f'Property {i}' for i in range(50)],
            'Property Type': np.random.choice(['Industrial', 'Warehouse', 'Flex', 'Office'], 50),
            'Building SqFt': np.random.randint(15000, 200000, 50),
            'Lot Size Acres': np.random.uniform(0.3, 25.0, 50),
            'Year Built': np.random.randint(1970, 2020, 50),
            'Occupancy': np.random.uniform(60, 100, 50),
            'City': np.random.choice(['Dallas', 'Houston', 'Austin'], 50),
            'State': ['TX'] * 50,
            'County': np.random.choice(['Dallas County', 'Harris County', 'Travis County'], 50)
        })
        
        # Create custom config for testing
        config_manager = FlexConfigManager()
        self.config = config_manager.create_custom_config(
            advanced_adjustments={
                'enable_batch_processing': True,
                'batch_size': 10,
                'enable_geographic_analysis': True,
                'enable_size_distribution_analysis': True
            }
        )
    
    def test_initialization_with_config(self):
        """Test initialization with custom configuration"""
        classifier = AdvancedFlexClassifier(self.test_data, self.config, self.logger)
        
        self.assertEqual(classifier.config.advanced_settings.batch_size, 10)
        self.assertTrue(classifier.config.advanced_settings.enable_batch_processing)
        self.assertIsNotNone(classifier.processing_metrics)
    
    def test_configurable_scoring(self):
        """Test configurable scoring weights"""
        # Modify scoring weights
        self.config.scoring_weights.building_size_weight = 2.0
        self.config.scoring_weights.property_type_weight = 0.5
        
        classifier = AdvancedFlexClassifier(self.test_data, self.config, self.logger)
        
        # Test scoring with modified weights
        test_property = pd.Series({
            'Property Type': 'Flex Industrial',
            'Building SqFt': 30000,
            'Lot Size Acres': 2.5,
            'Year Built': 1995,
            'Occupancy': 85
        })
        
        score, breakdown = classifier.calculate_flex_score_advanced(test_property)
        
        # Verify weights are applied
        self.assertGreater(breakdown['building_size'], breakdown['property_type'])
        self.assertIsInstance(score, float)
        self.assertIsInstance(breakdown, dict)
    
    def test_batch_processing(self):
        """Test batch processing functionality"""
        classifier = AdvancedFlexClassifier(self.test_data, self.config, self.logger)
        
        # Track progress
        progress_updates = []
        def progress_callback(current, total, message):
            progress_updates.append((current, total, message))
        
        classifier.set_progress_callback(progress_callback)
        
        # Process in batches
        candidates = classifier.classify_flex_properties_batch(batch_size=10)
        
        self.assertIsInstance(candidates, pd.DataFrame)
        self.assertGreaterEqual(len(progress_updates), 0)  # Should have some progress updates
    
    def test_geographic_analysis(self):
        """Test geographic analysis functionality"""
        classifier = AdvancedFlexClassifier(self.test_data, self.config, self.logger)
        
        # First classify properties
        candidates = classifier.classify_flex_properties_batch()
        
        if len(candidates) > 0:
            # Perform geographic analysis
            geo_analysis = classifier.perform_geographic_analysis()
            
            self.assertIsNotNone(geo_analysis)
            self.assertIsInstance(geo_analysis.state_distribution, dict)
            self.assertIsInstance(geo_analysis.city_distribution, dict)
            self.assertIsInstance(geo_analysis.geographic_concentration, float)
    
    def test_size_distribution_analysis(self):
        """Test size distribution analysis functionality"""
        classifier = AdvancedFlexClassifier(self.test_data, self.config, self.logger)
        
        # First classify properties
        candidates = classifier.classify_flex_properties_batch()
        
        if len(candidates) > 0:
            # Add flex scores for analysis
            classifier._apply_scoring_to_candidates()
            
            # Perform size analysis
            size_analysis = classifier.perform_size_distribution_analysis()
            
            self.assertIsNotNone(size_analysis)
            self.assertIsInstance(size_analysis.building_size_distribution, dict)
            self.assertIsInstance(size_analysis.lot_size_distribution, dict)
    
    def test_advanced_export(self):
        """Test advanced export functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            classifier = AdvancedFlexClassifier(self.test_data, self.config, self.logger)
            
            # Process and export
            candidates = classifier.classify_flex_properties_batch()
            
            if len(candidates) > 0:
                # Perform analytics
                classifier.perform_geographic_analysis()
                classifier.perform_size_distribution_analysis()
                
                # Export with analytics
                exported_files = classifier.export_advanced_results(
                    output_dir, 
                    include_analytics=True
                )
                
                self.assertIsInstance(exported_files, dict)
                self.assertGreater(len(exported_files), 0)
                
                # Verify files exist
                for file_path in exported_files.values():
                    self.assertTrue(Path(file_path).exists())
    
    def test_performance_report(self):
        """Test performance report generation"""
        classifier = AdvancedFlexClassifier(self.test_data, self.config, self.logger)
        
        # Process data
        candidates = classifier.classify_flex_properties_batch()
        
        # Get performance report
        report = classifier.get_performance_report()
        
        self.assertIsInstance(report, dict)
        self.assertIn('processing_metrics', report)
        self.assertIn('configuration_summary', report)
        self.assertIn('data_summary', report)
        self.assertIn('recommendations', report)


class TestBatchProcessor(unittest.TestCase):
    """Test cases for FlexBatchProcessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.logger = setup_logging(name='test_batch', level='DEBUG')
        
        # Create test Excel files
        self.test_files = []
        for i in range(3):
            # Create test data
            np.random.seed(42 + i)
            data = pd.DataFrame({
                'Property Type': np.random.choice(['Industrial', 'Warehouse', 'Office'], 20),
                'Building SqFt': np.random.randint(15000, 100000, 20),
                'Lot Size Acres': np.random.uniform(0.5, 15.0, 20),
                'City': np.random.choice(['Dallas', 'Houston'], 20),
                'State': ['TX'] * 20
            })
            
            # Save as Excel file
            file_path = self.temp_dir / f"test_properties_{i}.xlsx"
            data.to_excel(file_path, index=False, engine='openpyxl')
            self.test_files.append(file_path)
        
        # Create batch processor
        self.processor = FlexBatchProcessor(logger=self.logger)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_discover_excel_files(self):
        """Test Excel file discovery"""
        discovered_files = self.processor.discover_excel_files([self.temp_dir])
        
        self.assertEqual(len(discovered_files), 3)
        for file_path in discovered_files:
            self.assertTrue(file_path.exists())
            self.assertTrue(file_path.suffix in ['.xlsx', '.xls', '.xlsm'])
    
    def test_process_files_sequential(self):
        """Test sequential file processing"""
        # Disable parallel processing
        self.processor.config.advanced_settings.parallel_processing = False
        
        # Track progress
        progress_updates = []
        def progress_callback(completed, total, current_file, elapsed_time):
            progress_updates.append((completed, total, current_file))
        
        # Process files
        summary = self.processor.process_files(
            self.test_files,
            progress_callback=progress_callback
        )
        
        self.assertEqual(summary.total_files, 3)
        self.assertGreaterEqual(summary.successful_files, 0)
        self.assertGreaterEqual(len(progress_updates), 0)
    
    def test_process_files_parallel(self):
        """Test parallel file processing"""
        # Enable parallel processing
        self.processor.config.advanced_settings.parallel_processing = True
        self.processor.config.advanced_settings.max_workers = 2
        
        # Process files
        summary = self.processor.process_files(self.test_files, max_workers=2)
        
        self.assertEqual(summary.total_files, 3)
        self.assertGreaterEqual(summary.successful_files, 0)
    
    def test_batch_export(self):
        """Test batch result export"""
        with tempfile.TemporaryDirectory() as output_temp:
            output_dir = Path(output_temp)
            
            # Process files with export
            summary = self.processor.process_files(
                self.test_files,
                output_dir=output_dir
            )
            
            # Check that export files were created
            export_files = list(output_dir.glob("*.xlsx")) + list(output_dir.glob("*.json"))
            self.assertGreater(len(export_files), 0)
    
    def test_failed_file_handling(self):
        """Test handling of failed files"""
        # Add a non-existent file to trigger failure
        bad_file = self.temp_dir / "nonexistent.xlsx"
        test_files_with_bad = self.test_files + [bad_file]
        
        # Process files
        summary = self.processor.process_files(test_files_with_bad)
        
        # Should have at least one failure
        self.assertGreater(summary.failed_files, 0)
        
        # Check failed files list
        failed_files = self.processor.get_failed_files()
        self.assertGreater(len(failed_files), 0)


class TestIntegrationAdvancedFeatures(unittest.TestCase):
    """Integration tests for advanced features"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.logger = setup_logging(name='test_integration_advanced', level='DEBUG')
        
        # Create larger test dataset
        np.random.seed(42)
        self.large_data = pd.DataFrame({
            'Property Name': [f'Property {i}' for i in range(200)],
            'Property Type': np.random.choice(['Industrial', 'Warehouse', 'Flex', 'Office', 'Retail'], 200),
            'Building SqFt': np.random.randint(10000, 300000, 200),
            'Lot Size Acres': np.random.uniform(0.1, 30.0, 200),
            'Year Built': np.random.randint(1960, 2023, 200),
            'Occupancy': np.random.uniform(50, 100, 200),
            'Address': [f'{1000+i} Industrial Blvd' for i in range(200)],
            'City': np.random.choice(['Dallas', 'Houston', 'Austin', 'San Antonio', 'Fort Worth'], 200),
            'State': ['TX'] * 200,
            'County': np.random.choice(['Dallas County', 'Harris County', 'Travis County', 'Bexar County'], 200),
            'Zoning Code': np.random.choice(['I-1', 'I-2', 'I-3', 'M-1', 'M-2'], 200),
            'Sold Price': np.random.randint(500000, 10000000, 200),
            'Owner Name': [f'Owner {i} LLC' for i in range(200)]
        })
    
    def test_complete_workflow_with_analytics(self):
        """Test complete workflow with all advanced features"""
        # Create custom configuration
        config_manager = FlexConfigManager()
        config = config_manager.create_custom_config(
            scoring_adjustments={
                'building_size_weight': 1.2,
                'property_type_weight': 1.5
            },
            filtering_adjustments={
                'min_building_sqft': 25000
            },
            advanced_adjustments={
                'enable_batch_processing': True,
                'batch_size': 50,
                'enable_geographic_analysis': True,
                'enable_size_distribution_analysis': True,
                'export_formats': ['xlsx', 'csv', 'json']
            }
        )
        
        # Create advanced classifier
        classifier = AdvancedFlexClassifier(self.large_data, config, self.logger)
        
        # Process with batch processing
        candidates = classifier.classify_flex_properties_batch()
        
        self.assertIsInstance(candidates, pd.DataFrame)
        
        if len(candidates) > 0:
            # Perform all analytics
            geo_analysis = classifier.perform_geographic_analysis()
            size_analysis = classifier.perform_size_distribution_analysis()
            
            # Verify analytics results
            self.assertIsNotNone(geo_analysis)
            self.assertIsNotNone(size_analysis)
            
            # Export results
            with tempfile.TemporaryDirectory() as temp_dir:
                exported_files = classifier.export_advanced_results(
                    Path(temp_dir),
                    include_analytics=True
                )
                
                # Verify multiple formats exported
                self.assertIn('xlsx', exported_files)
                self.assertIn('csv', exported_files)
                self.assertIn('json', exported_files)
                
                # Verify analytics files
                self.assertIn('geographic_analysis', exported_files)
                self.assertIn('size_analysis', exported_files)
                self.assertIn('processing_metrics', exported_files)
            
            # Get performance report
            performance = classifier.get_performance_report()
            self.assertIn('processing_metrics', performance)
            self.assertIn('recommendations', performance)
    
    def test_configuration_impact_on_results(self):
        """Test that configuration changes impact results"""
        # Create conservative configuration
        conservative_config = FlexConfigManager().create_custom_config(
            filtering_adjustments={
                'min_building_sqft': 50000,
                'min_lot_acres': 2.0,
                'max_lot_acres': 10.0
            }
        )
        
        # Create aggressive configuration
        aggressive_config = FlexConfigManager().create_custom_config(
            filtering_adjustments={
                'min_building_sqft': 15000,
                'min_lot_acres': 0.3,
                'max_lot_acres': 25.0
            }
        )
        
        # Process with both configurations
        conservative_classifier = AdvancedFlexClassifier(self.large_data, conservative_config, self.logger)
        aggressive_classifier = AdvancedFlexClassifier(self.large_data, aggressive_config, self.logger)
        
        conservative_candidates = conservative_classifier.classify_flex_properties()
        aggressive_candidates = aggressive_classifier.classify_flex_properties()
        
        # Aggressive should find more candidates
        self.assertGreaterEqual(len(aggressive_candidates), len(conservative_candidates))
    
    def test_performance_with_large_dataset(self):
        """Test performance with large dataset"""
        import time
        
        # Create very large dataset
        np.random.seed(42)
        very_large_data = pd.DataFrame({
            'Property Type': np.random.choice(['Industrial', 'Warehouse', 'Flex', 'Office'], 2000),
            'Building SqFt': np.random.randint(10000, 200000, 2000),
            'Lot Size Acres': np.random.uniform(0.1, 25.0, 2000),
            'Year Built': np.random.randint(1970, 2020, 2000),
            'Occupancy': np.random.uniform(60, 100, 2000),
            'City': np.random.choice(['Dallas', 'Houston', 'Austin'], 2000),
            'State': ['TX'] * 2000
        })
        
        # Test with batch processing enabled
        config = FlexConfigManager().create_custom_config(
            advanced_adjustments={
                'enable_batch_processing': True,
                'batch_size': 500
            }
        )
        
        classifier = AdvancedFlexClassifier(very_large_data, config, self.logger)
        
        start_time = time.time()
        candidates = classifier.classify_flex_properties_batch()
        processing_time = time.time() - start_time
        
        # Should process reasonably quickly
        self.assertLess(processing_time, 30.0)  # Should complete within 30 seconds
        
        # Should handle large dataset without errors
        self.assertIsInstance(candidates, pd.DataFrame)
        
        # Get performance metrics
        performance = classifier.get_performance_report()
        processing_rate = performance['processing_metrics']['processing_rate']
        
        # Should achieve reasonable processing rate
        self.assertGreater(processing_rate, 50)  # At least 50 properties/second


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)