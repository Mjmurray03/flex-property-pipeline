"""
Tests for the test data generator
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from tests.test_data_generator import PipelineTestDataGenerator, create_test_scenario, create_sample_files


class TestPipelineTestDataGenerator:
    """Test the test data generator functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    def test_generate_property_data(self, temp_dir):
        """Test property data generation"""
        generator = PipelineTestDataGenerator(temp_dir)
        
        df = generator.generate_property_data(100, flex_candidate_ratio=0.2)
        
        assert len(df) >= 100  # May have duplicates added
        assert 'site_address' in df.columns
        assert 'city' in df.columns
        assert 'state' in df.columns
        assert 'acres' in df.columns
        
        # Check that some properties are marked as flex candidates
        flex_candidates = df[df['is_flex_candidate'] == True]
        assert len(flex_candidates) > 0
    
    def test_create_valid_excel_file(self, temp_dir):
        """Test valid Excel file creation"""
        generator = PipelineTestDataGenerator(temp_dir)
        
        file_path = generator.create_valid_excel_file("test.xlsx", 50, 0.15)
        
        assert file_path.exists()
        assert file_path.suffix == '.xlsx'
        
        # Verify file content
        df = pd.read_excel(file_path)
        assert len(df) == 50
        assert 'site_address' in df.columns
    
    def test_create_empty_excel_file(self, temp_dir):
        """Test empty Excel file creation"""
        generator = PipelineTestDataGenerator(temp_dir)
        
        file_path = generator.create_empty_excel_file("empty.xlsx")
        
        assert file_path.exists()
        
        # Verify file is empty
        df = pd.read_excel(file_path)
        assert len(df) == 0
    
    def test_create_invalid_format_file(self, temp_dir):
        """Test invalid format file creation"""
        generator = PipelineTestDataGenerator(temp_dir)
        
        file_path = generator.create_invalid_format_file("invalid.xlsx")
        
        assert file_path.exists()
        
        # Verify file has wrong columns
        df = pd.read_excel(file_path)
        assert 'site_address' not in df.columns
        assert 'wrong_column_1' in df.columns
    
    def test_create_test_scenario(self, temp_dir):
        """Test test scenario creation"""
        generator = PipelineTestDataGenerator(temp_dir)
        
        scenario = generator.create_test_scenario('basic_processing')
        
        assert 'description' in scenario
        assert 'files' in scenario
        assert len(scenario['files']) == 3
        
        # Verify all files exist
        for file_path in scenario['files']:
            assert file_path.exists()
    
    def test_get_scenario_summary(self, temp_dir):
        """Test scenario summary generation"""
        generator = PipelineTestDataGenerator(temp_dir)
        
        scenario = generator.create_test_scenario('error_handling')
        summary = generator.get_scenario_summary(scenario)
        
        assert 'total_files' in summary
        assert 'valid_files' in summary
        assert 'invalid_files' in summary
        assert summary['total_files'] == 4
        assert summary['valid_files'] >= 1
        assert summary['invalid_files'] >= 1
    
    def test_context_manager(self, temp_dir):
        """Test context manager functionality"""
        with PipelineTestDataGenerator(temp_dir) as generator:
            file_path = generator.create_valid_excel_file("context_test.xlsx", 10)
            assert file_path.exists()
        
        # Directory should be cleaned up after context exit
        # Note: temp_dir fixture handles cleanup, so we can't test this directly
    
    def test_convenience_functions(self, temp_dir):
        """Test convenience functions"""
        # Test create_test_scenario function
        scenario = create_test_scenario('basic_processing', temp_dir)
        assert 'files' in scenario
        assert len(scenario['files']) > 0
        
        # Test create_sample_files function
        files = create_sample_files(3, temp_dir)
        assert len(files) == 3
        for file_path in files:
            assert file_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])