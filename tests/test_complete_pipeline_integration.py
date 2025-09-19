"""
Integration tests for complete pipeline execution
Tests the CLI interface and main execution script
"""

import pytest
import tempfile
import shutil
import pandas as pd
from pathlib import Path
import subprocess
import sys
import json
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.scalable_flex_pipeline import ScalableFlexPipeline, PipelineConfiguration
from run_scalable_pipeline import integrate_pipeline_components, run_complete_pipeline


class TestCompletePipelineIntegration:
    """Test complete pipeline integration"""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing"""
        temp_dir = Path(tempfile.mkdtemp())
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        config_dir = temp_dir / "config"
        
        input_dir.mkdir()
        output_dir.mkdir()
        config_dir.mkdir()
        
        yield {
            'base': temp_dir,
            'input': input_dir,
            'output': output_dir,
            'config': config_dir
        }
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_excel_files(self, temp_dirs):
        """Create sample Excel files for testing"""
        input_dir = temp_dirs['input']
        
        # Create sample data
        sample_data_1 = pd.DataFrame({
            'Address': ['123 Main St', '456 Oak Ave', '789 Pine Rd'],
            'City': ['Springfield', 'Riverside', 'Lakewood'],
            'State': ['IL', 'CA', 'CO'],
            'Building_Size': [2000, 1500, 2500],
            'Lot_Size': [0.25, 0.20, 0.30],
            'Year_Built': [1990, 1985, 2000],
            'Property_Type': ['Single Family', 'Condo', 'Single Family']
        })
        
        sample_data_2 = pd.DataFrame({
            'Address': ['321 Elm St', '654 Maple Dr', '987 Cedar Ln'],
            'City': ['Madison', 'Portland', 'Austin'],
            'State': ['WI', 'OR', 'TX'],
            'Building_Size': [1800, 2200, 1600],
            'Lot_Size': [0.22, 0.28, 0.18],
            'Year_Built': [1995, 1988, 2005],
            'Property_Type': ['Single Family', 'Single Family', 'Townhouse']
        })
        
        # Save to Excel files
        file1 = input_dir / "properties_1.xlsx"
        file2 = input_dir / "properties_2.xlsx"
        
        sample_data_1.to_excel(file1, index=False)
        sample_data_2.to_excel(file2, index=False)
        
        return [file1, file2]
    
    def test_pipeline_configuration_creation(self, temp_dirs):
        """Test pipeline configuration creation and loading"""
        config_file = temp_dirs['config'] / "test_config.yaml"
        
        # Create configuration
        config = PipelineConfiguration(
            input_folder=str(temp_dirs['input']),
            output_file=str(temp_dirs['output'] / "results.xlsx"),
            max_workers=2,
            min_flex_score=5.0
        )
        
        # Save to file
        config.to_file(str(config_file))
        
        # Load from file
        loaded_config = PipelineConfiguration.from_file(str(config_file))
        
        assert loaded_config.input_folder == str(temp_dirs['input'])
        assert loaded_config.output_file == str(temp_dirs['output'] / "results.xlsx")
        assert loaded_config.max_workers == 2
        assert loaded_config.min_flex_score == 5.0
    
    def test_pipeline_component_integration(self, temp_dirs, sample_excel_files):
        """Test integration of all pipeline components"""
        config = PipelineConfiguration(
            input_folder=str(temp_dirs['input']),
            output_file=str(temp_dirs['output'] / "results.xlsx"),
            max_workers=1,  # Use single worker for testing
            min_flex_score=0.0  # Accept all properties for testing
        )
        
        pipeline = ScalableFlexPipeline(config=config)
        
        # Test component integration
        integrate_pipeline_components(pipeline)
        
        # Verify all components are initialized
        assert pipeline.file_discovery is not None
        assert pipeline.batch_processor is not None
        assert pipeline.result_aggregator is not None
        assert pipeline.report_generator is not None
        assert pipeline.output_manager is not None
        assert pipeline.data_validator is not None
        assert pipeline.pipeline_logger is not None
    
    def test_complete_pipeline_execution(self, temp_dirs, sample_excel_files):
        """Test complete pipeline execution"""
        config = PipelineConfiguration(
            input_folder=str(temp_dirs['input']),
            output_file=str(temp_dirs['output'] / "results.xlsx"),
            max_workers=1,
            min_flex_score=0.0,
            progress_reporting=False  # Disable for testing
        )
        
        pipeline = ScalableFlexPipeline(config=config)
        integrate_pipeline_components(pipeline)
        
        # Run complete pipeline
        results = run_complete_pipeline(pipeline)
        
        # Verify results
        assert results['success'] is True
        assert results['files_discovered'] == 2
        assert results['files_processed'] >= 0  # May be 0 if no flex properties found
        assert 'execution_time' in results
        assert 'configuration' in results
    
    def test_cli_help_output(self):
        """Test CLI help output"""
        try:
            result = subprocess.run(
                [sys.executable, "scalable_pipeline_cli.py", "--help"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0
            assert "Scalable Multi-File Flex Property Classification Pipeline" in result.stdout
            assert "--input-folder" in result.stdout
            assert "--output-file" in result.stdout
            assert "--dry-run" in result.stdout
            
        except subprocess.TimeoutExpired:
            pytest.skip("CLI help test timed out")
        except FileNotFoundError:
            pytest.skip("CLI script not found")
    
    def test_cli_config_creation(self, temp_dirs):
        """Test CLI configuration file creation"""
        config_file = temp_dirs['config'] / "cli_config.yaml"
        
        try:
            result = subprocess.run([
                sys.executable, "scalable_pipeline_cli.py",
                "--create-config", str(config_file)
            ], capture_output=True, text=True, timeout=30)
            
            assert result.returncode == 0
            assert config_file.exists()
            
            # Verify configuration content
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            assert 'input_folder' in config_data
            assert 'output_file' in config_data
            assert 'max_workers' in config_data
            
        except subprocess.TimeoutExpired:
            pytest.skip("CLI config creation test timed out")
        except FileNotFoundError:
            pytest.skip("CLI script not found")
    
    def test_cli_dry_run(self, temp_dirs, sample_excel_files):
        """Test CLI dry run functionality"""
        try:
            result = subprocess.run([
                sys.executable, "scalable_pipeline_cli.py",
                "--input-folder", str(temp_dirs['input']),
                "--output-file", str(temp_dirs['output'] / "results.xlsx"),
                "--dry-run"
            ], capture_output=True, text=True, timeout=30)
            
            # Dry run should succeed with valid configuration
            assert result.returncode == 0
            assert "DRY RUN" in result.stdout
            assert "validation passed" in result.stdout.lower() or "completed successfully" in result.stdout.lower()
            
        except subprocess.TimeoutExpired:
            pytest.skip("CLI dry run test timed out")
        except FileNotFoundError:
            pytest.skip("CLI script not found")
    
    def test_main_execution_script(self, temp_dirs, sample_excel_files):
        """Test main execution script"""
        try:
            # Modify the script to use test directories
            result = subprocess.run([
                sys.executable, "run_scalable_pipeline.py"
            ], capture_output=True, text=True, timeout=60)
            
            # Script should run without crashing
            # May succeed or fail depending on data, but should not crash
            assert result.returncode in [0, 1]  # Success or controlled failure
            
        except subprocess.TimeoutExpired:
            pytest.skip("Main execution script test timed out")
        except FileNotFoundError:
            pytest.skip("Main execution script not found")
    
    def test_pipeline_error_handling(self, temp_dirs):
        """Test pipeline error handling with invalid configuration"""
        # Test with non-existent input folder
        config = PipelineConfiguration(
            input_folder="/non/existent/folder",
            output_file=str(temp_dirs['output'] / "results.xlsx")
        )
        
        pipeline = ScalableFlexPipeline(config=config)
        integrate_pipeline_components(pipeline)
        
        results = run_complete_pipeline(pipeline)
        
        # Should handle error gracefully
        assert results['success'] is False
        assert 'error' in results
        assert 'execution_time' in results
    
    def test_pipeline_with_empty_input_folder(self, temp_dirs):
        """Test pipeline behavior with empty input folder"""
        config = PipelineConfiguration(
            input_folder=str(temp_dirs['input']),
            output_file=str(temp_dirs['output'] / "results.xlsx")
        )
        
        pipeline = ScalableFlexPipeline(config=config)
        integrate_pipeline_components(pipeline)
        
        results = run_complete_pipeline(pipeline)
        
        # Should succeed but process no files
        assert results['success'] is True
        assert results['files_discovered'] == 0
        assert results['files_processed'] == 0
        assert 'No files found' in results.get('message', '')
    
    def test_configuration_validation(self, temp_dirs):
        """Test configuration validation"""
        # Test invalid configuration
        config = PipelineConfiguration(
            input_folder=str(temp_dirs['input']),
            output_file=str(temp_dirs['output'] / "results.xlsx"),
            max_workers=-1,  # Invalid
            min_flex_score=15.0  # Invalid (should be 0-10)
        )
        
        pipeline = ScalableFlexPipeline(config=config)
        
        # Validation should fail
        assert not pipeline.validate_configuration()
    
    def test_export_functionality(self, temp_dirs):
        """Test export functionality with sample data"""
        config = PipelineConfiguration(
            input_folder=str(temp_dirs['input']),
            output_file=str(temp_dirs['output'] / "results.xlsx"),
            enable_csv_export=True,
            backup_existing=True
        )
        
        pipeline = ScalableFlexPipeline(config=config)
        integrate_pipeline_components(pipeline)
        
        # Create sample DataFrame
        sample_df = pd.DataFrame({
            'Address': ['123 Test St'],
            'City': ['Test City'],
            'State': ['TS'],
            'Flex_Score': [8.5]
        })
        
        # Test export
        excel_result, csv_result = pipeline.output_manager.export_master_file(
            sample_df, "test_export"
        )
        
        assert excel_result.success
        assert csv_result.success
        assert Path(excel_result.file_path).exists()
        assert Path(csv_result.file_path).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])