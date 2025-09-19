"""
Tests for Integration Manager
Tests integration with existing pipeline components
"""

import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.integration_manager import (
    IntegrationManager, ComponentCompatibilityChecker, LegacyDataMigrator,
    ExistingSystemIntegrator, OutputFormatConverter, IntegrationResult
)


class TestComponentCompatibilityChecker:
    """Test component compatibility checking"""
    
    @pytest.fixture
    def checker(self):
        """Create ComponentCompatibilityChecker instance"""
        return ComponentCompatibilityChecker()
    
    def test_initialization(self, checker):
        """Test checker initialization"""
        assert checker.logger is not None
        assert isinstance(checker.compatibility_results, dict)
    
    @patch('pipeline.integration_manager.FlexPropertyClassifier')
    def test_flex_classifier_compatibility_success(self, mock_classifier_class, checker):
        """Test successful flex classifier compatibility check"""
        # Mock classifier instance
        mock_classifier = Mock()
        mock_classifier.classify_properties.return_value = pd.DataFrame({'test': [1]})
        mock_classifier.get_flex_candidates.return_value = pd.DataFrame({'test': [1]})
        mock_classifier.version = '1.0'
        mock_classifier_class.return_value = mock_classifier
        
        result = checker.check_flex_classifier_compatibility()
        
        assert result is True
        assert 'flex_classifier' in checker.compatibility_results
        assert checker.compatibility_results['flex_classifier']['compatible'] is True
    
    @patch('pipeline.integration_manager.FlexPropertyClassifier')
    def test_flex_classifier_compatibility_failure(self, mock_classifier_class, checker):
        """Test flex classifier compatibility check failure"""
        # Mock classifier to raise exception
        mock_classifier_class.side_effect = Exception("Import failed")
        
        result = checker.check_flex_classifier_compatibility()
        
        assert result is False
        assert 'flex_classifier' in checker.compatibility_results
        assert checker.compatibility_results['flex_classifier']['compatible'] is False
    
    @patch('pipeline.integration_manager.MongoDBClient', create=True)
    def test_database_compatibility_success(self, mock_db_class, checker):
        """Test successful database compatibility check"""
        # Mock database client
        mock_db = Mock()
        mock_db.insert_one.return_value = True
        mock_db.delete_one.return_value = True
        mock_db_class.return_value = mock_db
        
        result = checker.check_database_compatibility()
        
        assert result is True
        assert 'database' in checker.compatibility_results
        assert checker.compatibility_results['database']['compatible'] is True
    
    @patch('pipeline.integration_manager.FlexDataUtils', create=True)
    def test_data_utils_compatibility(self, mock_utils_class, checker):
        """Test data utilities compatibility check"""
        # Mock data utils
        mock_utils = Mock()
        mock_utils.normalize_column_names = Mock()
        mock_utils.validate_data_types = Mock()
        mock_utils_class.return_value = mock_utils
        
        result = checker.check_data_utils_compatibility()
        
        assert result is True
        assert 'data_utils' in checker.compatibility_results
        assert checker.compatibility_results['data_utils']['compatible'] is True
    
    def test_full_compatibility_check(self, checker):
        """Test full compatibility check"""
        # Mock individual check methods
        checker.check_flex_classifier_compatibility = Mock(return_value=True)
        checker.check_database_compatibility = Mock(return_value=False)
        checker.check_data_utils_compatibility = Mock(return_value=True)
        
        results = checker.run_full_compatibility_check()
        
        assert 'flex_classifier' in results
        assert 'database' in results
        assert 'data_utils' in results
        assert results['flex_classifier'] is True
        assert results['database'] is False
        assert results['data_utils'] is True


class TestLegacyDataMigrator:
    """Test legacy data migration"""
    
    @pytest.fixture
    def migrator(self):
        """Create LegacyDataMigrator instance"""
        return LegacyDataMigrator()
    
    @pytest.fixture
    def temp_files(self):
        """Create temporary files for testing"""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create sample legacy data
        legacy_data = pd.DataFrame({
            'site_address': ['123 Main St', '456 Oak Ave'],
            'property_city': ['Springfield', 'Riverside'],
            'property_state': ['IL', 'CA'],
            'building_size': [2000, 1500],
            'lot_acres': [0.25, 0.20],
            'property_type': ['Industrial', 'Warehouse'],
            'score': [8.5, 7.2]
        })
        
        legacy_file = temp_dir / "legacy_results.xlsx"
        legacy_data.to_excel(legacy_file, index=False)
        
        output_file = temp_dir / "migrated_results.xlsx"
        
        yield {
            'legacy_file': legacy_file,
            'output_file': output_file,
            'temp_dir': temp_dir
        }
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_migrate_single_file_results(self, migrator, temp_files):
        """Test migration of single file results"""
        result = migrator.migrate_single_file_results(
            str(temp_files['legacy_file']),
            str(temp_files['output_file'])
        )
        
        assert result.success is True
        assert result.records_processed == 2
        assert result.records_integrated == 2
        assert temp_files['output_file'].exists()
        
        # Verify migrated data
        migrated_df = pd.read_excel(temp_files['output_file'])
        assert 'Address' in migrated_df.columns
        assert 'City' in migrated_df.columns
        assert 'State' in migrated_df.columns
        assert 'Flex Score' in migrated_df.columns
        assert 'Source_File' in migrated_df.columns
    
    def test_standardize_legacy_columns(self, migrator):
        """Test legacy column standardization"""
        legacy_df = pd.DataFrame({
            'site_address': ['123 Main St'],
            'property_city': ['Springfield'],
            'building_size': [2000],
            'score': [8.5]
        })
        
        standardized_df = migrator._standardize_legacy_columns(legacy_df)
        
        assert 'Address' in standardized_df.columns
        assert 'City' in standardized_df.columns
        assert 'Building SqFt' in standardized_df.columns
        assert 'Flex Score' in standardized_df.columns
    
    def test_add_pipeline_metadata(self, migrator):
        """Test addition of pipeline metadata"""
        df = pd.DataFrame({'Address': ['123 Main St']})
        
        df_with_metadata = migrator._add_pipeline_metadata(df, '/path/to/source.xlsx')
        
        assert 'Source_File' in df_with_metadata.columns
        assert 'Migration_Date' in df_with_metadata.columns
        assert 'Pipeline_Version' in df_with_metadata.columns
        assert df_with_metadata['Source_File'].iloc[0] == 'source.xlsx'


class TestExistingSystemIntegrator:
    """Test integration with existing systems"""
    
    @pytest.fixture
    def integrator(self):
        """Create ExistingSystemIntegrator instance"""
        return ExistingSystemIntegrator()
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing"""
        return pd.DataFrame({
            'Address': ['123 Main St', '456 Oak Ave'],
            'City': ['Springfield', 'Riverside'],
            'State': ['IL', 'CA'],
            'Property Type': ['Industrial', 'Warehouse'],
            'Building SqFt': [2000, 1500],
            'Lot Size Acres': [0.25, 0.20]
        })
    
    @patch('pipeline.integration_manager.FlexPropertyClassifier')
    def test_integrate_with_flex_classifier(self, mock_classifier_class, integrator, sample_dataframe):
        """Test integration with FlexPropertyClassifier"""
        # Mock classifier
        mock_classifier = Mock()
        mock_classifier.classify_properties.return_value = sample_dataframe
        mock_classifier.get_flex_candidates.return_value = sample_dataframe
        mock_classifier.version = '1.0'
        mock_classifier_class.return_value = mock_classifier
        
        # Mock compatibility check
        integrator.compatibility_checker.check_flex_classifier_compatibility = Mock(return_value=True)
        
        result_df, integration_result = integrator.integrate_with_flex_classifier(sample_dataframe)
        
        assert integration_result.success is True
        assert integration_result.records_processed == 2
        assert len(result_df) == 2
    
    @patch('pipeline.integration_manager.MongoDBClient', create=True)
    def test_integrate_with_database(self, mock_db_class, integrator, sample_dataframe):
        """Test database integration"""
        # Mock database client
        mock_db = Mock()
        mock_db.insert_many.return_value = True
        mock_db_class.return_value = mock_db
        
        # Mock compatibility check
        integrator.compatibility_checker.check_database_compatibility = Mock(return_value=True)
        
        result = integrator.integrate_with_database(sample_dataframe)
        
        assert result.success is True
        assert result.records_processed == 2
        assert result.records_integrated == 2
    
    @patch('pipeline.integration_manager.PrivatePropertyAnalyzer')
    def test_integrate_with_private_analyzer(self, mock_analyzer_class, integrator, sample_dataframe):
        """Test integration with PrivatePropertyAnalyzer"""
        # Mock analyzer
        mock_analyzer = Mock()
        mock_analyzer.analyze_properties.return_value = sample_dataframe
        mock_analyzer.version = '1.0'
        mock_analyzer_class.return_value = mock_analyzer
        
        result_df, integration_result = integrator.integrate_with_private_analyzer(sample_dataframe)
        
        assert integration_result.success is True
        assert integration_result.records_processed == 2
        assert len(result_df) == 2


class TestOutputFormatConverter:
    """Test output format conversion"""
    
    @pytest.fixture
    def converter(self):
        """Create OutputFormatConverter instance"""
        return OutputFormatConverter()
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing"""
        return pd.DataFrame({
            'Address': ['123 Main St', '456 Oak Ave'],
            'City': ['Springfield', 'Riverside'],
            'State': ['IL', 'CA'],
            'Building SqFt': [2000, 1500],
            'Lot Size Acres': [0.25, 0.20],
            'Property Type': ['Industrial', 'Warehouse'],
            'Flex Score': [8.5, 7.2]
        })
    
    @pytest.fixture
    def temp_output_file(self):
        """Create temporary output file"""
        temp_dir = Path(tempfile.mkdtemp())
        output_file = temp_dir / "converted_output.xlsx"
        
        yield output_file
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_convert_to_classic_format(self, converter, sample_dataframe):
        """Test conversion to classic format"""
        classic_df = converter._convert_to_classic_format(sample_dataframe)
        
        assert 'site_address' in classic_df.columns
        assert 'property_city' in classic_df.columns
        assert 'property_state' in classic_df.columns
        assert 'building_size' in classic_df.columns
        assert 'flex_score' in classic_df.columns
    
    def test_convert_to_enhanced_format(self, converter, sample_dataframe):
        """Test conversion to enhanced format"""
        enhanced_df = converter._convert_to_enhanced_format(sample_dataframe)
        
        assert 'processing_date' in enhanced_df.columns
        assert 'pipeline_version' in enhanced_df.columns
        assert 'data_quality_score' in enhanced_df.columns
        assert len(enhanced_df) == len(sample_dataframe)
    
    def test_convert_to_minimal_format(self, converter, sample_dataframe):
        """Test conversion to minimal format"""
        minimal_df = converter._convert_to_minimal_format(sample_dataframe)
        
        expected_columns = ['Address', 'City', 'State', 'Flex Score']
        for col in expected_columns:
            assert col in minimal_df.columns
        
        # Should only have essential columns
        assert len(minimal_df.columns) == len(expected_columns)
    
    def test_convert_to_legacy_format_success(self, converter, sample_dataframe, temp_output_file):
        """Test successful legacy format conversion"""
        result = converter.convert_to_legacy_format(
            sample_dataframe,
            str(temp_output_file),
            'classic'
        )
        
        assert result.success is True
        assert result.records_processed == 2
        assert temp_output_file.exists()
    
    def test_convert_to_legacy_format_invalid_type(self, converter, sample_dataframe, temp_output_file):
        """Test legacy format conversion with invalid type"""
        result = converter.convert_to_legacy_format(
            sample_dataframe,
            str(temp_output_file),
            'invalid_type'
        )
        
        assert result.success is False
        assert 'Unknown format type' in result.error_message


class TestIntegrationManager:
    """Test main integration manager"""
    
    @pytest.fixture
    def manager(self):
        """Create IntegrationManager instance"""
        return IntegrationManager()
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing"""
        return pd.DataFrame({
            'Address': ['123 Main St', '456 Oak Ave'],
            'City': ['Springfield', 'Riverside'],
            'State': ['IL', 'CA'],
            'Flex Score': [8.5, 7.2]
        })
    
    def test_initialization(self, manager):
        """Test manager initialization"""
        assert manager.logger is not None
        assert manager.compatibility_checker is not None
        assert manager.system_integrator is not None
        assert manager.data_migrator is not None
        assert manager.format_converter is not None
        assert isinstance(manager.integration_results, list)
    
    def test_run_integration_workflow_minimal(self, manager, sample_dataframe):
        """Test minimal integration workflow"""
        integration_config = {
            'check_compatibility': True,
            'integrate_flex_classifier': False,
            'integrate_database': False,
            'integrate_private_analyzer': False,
            'convert_legacy_format': False
        }
        
        # Mock compatibility check
        manager.compatibility_checker.run_full_compatibility_check = Mock(
            return_value={'flex_classifier': True, 'database': True, 'data_utils': True}
        )
        
        results = manager.run_integration_workflow(sample_dataframe, integration_config)
        
        assert results['success'] is True
        assert 'compatibility_check' in results['steps_completed']
        assert results['total_records'] == 2
    
    def test_run_integration_workflow_full(self, manager, sample_dataframe):
        """Test full integration workflow"""
        integration_config = {
            'check_compatibility': True,
            'integrate_flex_classifier': True,
            'integrate_database': True,
            'integrate_private_analyzer': True,
            'convert_legacy_format': True,
            'legacy_output_path': '/tmp/legacy_output.xlsx',
            'legacy_format_type': 'classic',
            'collection_name': 'test_collection'
        }
        
        # Mock all integration methods to succeed
        manager.compatibility_checker.run_full_compatibility_check = Mock(
            return_value={'flex_classifier': True, 'database': True, 'data_utils': True}
        )
        
        manager.system_integrator.integrate_with_flex_classifier = Mock(
            return_value=(sample_dataframe, IntegrationResult(
                success=True, component_name='flex_classifier', records_processed=2, records_integrated=2
            ))
        )
        
        manager.system_integrator.integrate_with_database = Mock(
            return_value=IntegrationResult(
                success=True, component_name='database', records_processed=2, records_integrated=2
            )
        )
        
        manager.system_integrator.integrate_with_private_analyzer = Mock(
            return_value=(sample_dataframe, IntegrationResult(
                success=True, component_name='private_analyzer', records_processed=2, records_integrated=2
            ))
        )
        
        manager.format_converter.convert_to_legacy_format = Mock(
            return_value=IntegrationResult(
                success=True, component_name='format_converter', records_processed=2, records_integrated=2
            )
        )
        
        results = manager.run_integration_workflow(sample_dataframe, integration_config)
        
        assert results['success'] is True
        assert len(results['steps_completed']) == 5
        assert len(results['steps_failed']) == 0
        assert len(results['integration_results']) == 4  # All except compatibility check
    
    def test_get_integration_summary_empty(self, manager):
        """Test integration summary with no operations"""
        summary = manager.get_integration_summary()
        
        assert 'message' in summary
        assert 'No integration operations performed' in summary['message']
    
    def test_get_integration_summary_with_results(self, manager):
        """Test integration summary with results"""
        # Add mock integration results
        manager.integration_results = [
            IntegrationResult(
                success=True, component_name='test1', records_processed=10, 
                records_integrated=8, integration_time=1.5
            ),
            IntegrationResult(
                success=False, component_name='test2', records_processed=5, 
                records_integrated=0, integration_time=0.5, error_message='Test error'
            )
        ]
        
        summary = manager.get_integration_summary()
        
        assert summary['total_operations'] == 2
        assert summary['successful_operations'] == 1
        assert summary['failed_operations'] == 1
        assert summary['success_rate'] == 50.0
        assert summary['total_records_processed'] == 15
        assert summary['total_records_integrated'] == 8
        assert summary['total_integration_time'] == 2.0


class TestIntegrationResult:
    """Test IntegrationResult dataclass"""
    
    def test_integration_result_creation(self):
        """Test IntegrationResult creation"""
        result = IntegrationResult(
            success=True,
            component_name='test_component',
            records_processed=100,
            records_integrated=95,
            integration_time=2.5,
            metadata={'test': 'data'}
        )
        
        assert result.success is True
        assert result.component_name == 'test_component'
        assert result.records_processed == 100
        assert result.records_integrated == 95
        assert result.integration_time == 2.5
        assert result.metadata == {'test': 'data'}
        assert result.error_message is None
    
    def test_integration_result_failure(self):
        """Test IntegrationResult for failure case"""
        result = IntegrationResult(
            success=False,
            component_name='failed_component',
            error_message='Integration failed'
        )
        
        assert result.success is False
        assert result.component_name == 'failed_component'
        assert result.error_message == 'Integration failed'
        assert result.records_processed == 0
        assert result.records_integrated == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])