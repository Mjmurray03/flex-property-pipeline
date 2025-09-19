"""
Tests for OutputManager class
"""

import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import json
import os

from pipeline.output_manager import OutputManager, ExportResult


class TestOutputManager:
    """Test cases for OutputManager"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing"""
        return pd.DataFrame({
            'Property Name': ['Property A', 'Property B', 'Property C'],
            'Address': ['123 Main St', '456 Oak Ave', '789 Pine Rd'],
            'City': ['Austin', 'Dallas', 'Houston'],
            'State': ['TX', 'TX', 'TX'],
            'Flex Score': [8.5, 7.2, 9.1],
            'Building SqFt': [50000, 75000, 60000]
        })
    
    @pytest.fixture
    def output_manager(self, temp_dir):
        """Create OutputManager instance with temp directory"""
        return OutputManager(base_output_dir=temp_dir, enable_backup=True)
    
    def test_initialization(self, temp_dir):
        """Test OutputManager initialization"""
        manager = OutputManager(base_output_dir=temp_dir, enable_backup=False)
        
        assert manager.base_output_dir == Path(temp_dir)
        assert manager.enable_backup is False
        assert Path(temp_dir).exists()
    
    def test_ensure_directory_exists(self, temp_dir):
        """Test directory creation"""
        manager = OutputManager(base_output_dir=temp_dir)
        
        new_dir = Path(temp_dir) / "new_folder" / "subfolder"
        manager._ensure_directory_exists(new_dir)
        
        assert new_dir.exists()
    
    def test_export_to_excel_success(self, output_manager, sample_dataframe):
        """Test successful Excel export"""
        output_path = output_manager.base_output_dir / "test_export.xlsx"
        
        result = output_manager.export_to_excel(sample_dataframe, str(output_path))
        
        assert result.success is True
        assert result.record_count == 3
        assert result.file_size_mb >= 0  # Small files might round to 0
        assert result.export_time >= 0
        assert result.error_message is None
        assert Path(output_path).exists()
        
        # Verify file content
        loaded_df = pd.read_excel(output_path)
        assert len(loaded_df) == 3
        assert 'Property Name' in loaded_df.columns
    
    def test_export_to_csv_success(self, output_manager, sample_dataframe):
        """Test successful CSV export"""
        output_path = output_manager.base_output_dir / "test_export.csv"
        
        result = output_manager.export_to_csv(sample_dataframe, str(output_path))
        
        assert result.success is True
        assert result.record_count == 3
        assert result.file_size_mb >= 0  # Small files might round to 0
        assert result.export_time >= 0
        assert result.error_message is None
        assert Path(output_path).exists()
        
        # Verify file content
        loaded_df = pd.read_csv(output_path)
        assert len(loaded_df) == 3
        assert 'Property Name' in loaded_df.columns
    
    def test_backup_creation(self, output_manager, sample_dataframe):
        """Test backup file creation"""
        output_path = output_manager.base_output_dir / "test_backup.xlsx"
        
        # Create initial file
        output_manager.export_to_excel(sample_dataframe, str(output_path))
        
        # Export again to trigger backup
        result = output_manager.export_to_excel(sample_dataframe, str(output_path))
        
        assert result.success is True
        
        # Check for backup file
        backup_files = list(output_manager.base_output_dir.glob("test_backup_backup_*.xlsx"))
        assert len(backup_files) == 1
    
    def test_backup_disabled(self, temp_dir, sample_dataframe):
        """Test export with backup disabled"""
        manager = OutputManager(base_output_dir=temp_dir, enable_backup=False)
        output_path = manager.base_output_dir / "test_no_backup.xlsx"
        
        # Create initial file
        manager.export_to_excel(sample_dataframe, str(output_path))
        
        # Export again (should not create backup)
        result = manager.export_to_excel(sample_dataframe, str(output_path))
        
        assert result.success is True
        
        # Check no backup files created
        backup_files = list(manager.base_output_dir.glob("*_backup_*"))
        assert len(backup_files) == 0
    
    def test_export_master_file(self, output_manager, sample_dataframe):
        """Test master file export in both formats"""
        excel_result, csv_result = output_manager.export_master_file(sample_dataframe)
        
        assert excel_result.success is True
        assert csv_result.success is True
        
        excel_path = Path(excel_result.file_path)
        csv_path = Path(csv_result.file_path)
        
        assert excel_path.exists()
        assert csv_path.exists()
        assert excel_path.suffix == '.xlsx'
        assert csv_path.suffix == '.csv'
    
    def test_versioned_export(self, output_manager, sample_dataframe):
        """Test versioned export with timestamp"""
        excel_result, csv_result = output_manager.create_versioned_export(sample_dataframe)
        
        assert excel_result.success is True
        assert csv_result.success is True
        
        # Check that filenames contain timestamp
        excel_path = Path(excel_result.file_path)
        csv_path = Path(csv_result.file_path)
        
        assert "_20" in excel_path.stem  # Should contain year
        assert "_20" in csv_path.stem
    
    def test_save_export_metadata(self, output_manager):
        """Test export metadata saving"""
        export_results = [
            ExportResult(
                success=True,
                file_path="test.xlsx",
                record_count=100,
                file_size_mb=1.5,
                export_time=2.3
            )
        ]
        
        processing_stats = {
            "total_files": 5,
            "successful_files": 4,
            "failed_files": 1
        }
        
        metadata_path = output_manager.save_export_metadata(export_results, processing_stats)
        
        assert Path(metadata_path).exists()
        
        # Verify metadata content
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        assert "export_timestamp" in metadata
        assert len(metadata["export_results"]) == 1
        assert metadata["processing_statistics"]["total_files"] == 5
    
    def test_cleanup_old_backups(self, output_manager, sample_dataframe):
        """Test cleanup of old backup files"""
        output_path = output_manager.base_output_dir / "test_cleanup.xlsx"
        
        # Create multiple backups by exporting multiple times
        import time
        for i in range(7):
            output_manager.export_to_excel(sample_dataframe, str(output_path))
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        # Should have 6 backup files (first export creates no backup)
        backup_files = list(output_manager.base_output_dir.glob("*_backup_*"))
        assert len(backup_files) == 6
        
        # Cleanup keeping only 3 backups
        removed_count = output_manager.cleanup_old_backups(max_backups=3)
        
        assert removed_count == 3
        
        # Should now have only 3 backup files
        remaining_backups = list(output_manager.base_output_dir.glob("*_backup_*"))
        assert len(remaining_backups) == 3
    
    def test_get_output_summary(self, output_manager, sample_dataframe):
        """Test output directory summary"""
        # Create some files
        output_manager.export_to_excel(sample_dataframe, 
                                     str(output_manager.base_output_dir / "test1.xlsx"))
        output_manager.export_to_csv(sample_dataframe, 
                                   str(output_manager.base_output_dir / "test2.csv"))
        
        summary = output_manager.get_output_summary()
        
        assert summary["total_files"] >= 2
        assert summary["excel_files"] >= 1
        assert summary["csv_files"] >= 1
        assert summary["total_size_mb"] >= 0  # Small files might round to 0
        assert "output_directory" in summary
    
    def test_export_error_handling(self, output_manager):
        """Test error handling in export operations"""        
        with patch('pandas.DataFrame.to_excel', side_effect=Exception("Export failed")):
            result = output_manager.export_to_excel(pd.DataFrame(), "test.xlsx")
            
            assert result.success is False
            assert "Failed to export to Excel" in result.error_message
    
    def test_file_size_calculation(self, output_manager, sample_dataframe):
        """Test file size calculation"""
        output_path = output_manager.base_output_dir / "size_test.xlsx"
        
        result = output_manager.export_to_excel(sample_dataframe, str(output_path))
        
        assert result.file_size_mb >= 0  # Small files might round to 0
        
        # Verify size calculation
        actual_size = output_manager._get_file_size_mb(Path(output_path))
        assert actual_size == result.file_size_mb
    
    def test_nonexistent_file_backup(self, output_manager):
        """Test backup creation for nonexistent file"""
        nonexistent_path = Path(output_manager.base_output_dir) / "nonexistent.xlsx"
        
        backup_path = output_manager._create_backup(nonexistent_path)
        
        assert backup_path is None


if __name__ == "__main__":
    pytest.main([__file__])