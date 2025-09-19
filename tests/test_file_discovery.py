"""
Unit tests for File Discovery and Validation System
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import os

from pipeline.file_discovery import FileDiscovery


class TestFileDiscovery(unittest.TestCase):
    """Test cases for FileDiscovery class"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.discovery = FileDiscovery()
        
        # Create test Excel files
        self.create_test_files()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def create_test_files(self):
        """Create test Excel files with different structures"""
        
        # Valid file with all required columns
        valid_data = pd.DataFrame({
            'Property Address': ['123 Main St', '456 Oak Ave'],
            'City': ['Miami', 'Tampa'],
            'State': ['FL', 'FL'],
            'Property Type': ['Industrial', 'Warehouse'],
            'Building SqFt': [25000, 35000],
            'County': ['Miami-Dade', 'Hillsborough'],
            'Lot Size Acres': [2.5, 3.2],
            'Year Built': [1995, 2000]
        })
        valid_file = self.temp_path / 'valid_properties.xlsx'
        valid_data.to_excel(valid_file, index=False)
        
        # File missing required columns
        invalid_data = pd.DataFrame({
            'Name': ['Property A', 'Property B'],
            'Value': [100000, 200000]
        })
        invalid_file = self.temp_path / 'invalid_properties.xlsx'
        invalid_data.to_excel(invalid_file, index=False)
        
        # Empty file
        empty_data = pd.DataFrame()
        empty_file = self.temp_path / 'empty_properties.xlsx'
        empty_data.to_excel(empty_file, index=False)
        
        # File with minimal required columns
        minimal_data = pd.DataFrame({
            'Address': ['789 Pine St'],
            'City': ['Orlando'],
            'State': ['FL'],
            'Type': ['Industrial'],
            'Building Size': [20000]
        })
        minimal_file = self.temp_path / 'minimal_properties.xlsx'
        minimal_data.to_excel(minimal_file, index=False)
        
        # Create subdirectory with file
        sub_dir = self.temp_path / 'subdir'
        sub_dir.mkdir()
        sub_file = sub_dir / 'sub_properties.xlsx'
        valid_data.to_excel(sub_file, index=False)
        
        # Create non-Excel file
        text_file = self.temp_path / 'not_excel.txt'
        text_file.write_text('This is not an Excel file')
    
    def test_scan_input_folder_non_recursive(self):
        """Test non-recursive folder scanning"""
        files = self.discovery.scan_input_folder(str(self.temp_path), recursive=False)
        
        # Should find Excel files in root directory only
        file_names = [f.name for f in files]
        self.assertIn('valid_properties.xlsx', file_names)
        self.assertIn('invalid_properties.xlsx', file_names)
        self.assertIn('empty_properties.xlsx', file_names)
        self.assertIn('minimal_properties.xlsx', file_names)
        self.assertNotIn('sub_properties.xlsx', file_names)  # In subdirectory
        self.assertNotIn('not_excel.txt', file_names)  # Not Excel file
    
    def test_scan_input_folder_recursive(self):
        """Test recursive folder scanning"""
        files = self.discovery.scan_input_folder(str(self.temp_path), recursive=True)
        
        # Should find Excel files in all directories
        file_names = [f.name for f in files]
        self.assertIn('valid_properties.xlsx', file_names)
        self.assertIn('sub_properties.xlsx', file_names)  # In subdirectory
        self.assertNotIn('not_excel.txt', file_names)  # Not Excel file
    
    def test_scan_nonexistent_folder(self):
        """Test scanning non-existent folder"""
        files = self.discovery.scan_input_folder('/nonexistent/folder')
        self.assertEqual(len(files), 0)
    
    def test_validate_valid_file(self):
        """Test validation of valid Excel file"""
        valid_file = self.temp_path / 'valid_properties.xlsx'
        result = self.discovery.validate_file_format(valid_file)
        
        self.assertTrue(result['is_valid'])
        self.assertTrue(result['can_read'])
        self.assertTrue(result['has_data'])
        self.assertEqual(result['row_count'], 2)
        self.assertEqual(result['column_count'], 8)
        self.assertEqual(len(result['errors']), 0)
        self.assertGreater(len(result['required_columns_found']), 0)
    
    def test_validate_invalid_file(self):
        """Test validation of file missing required columns"""
        invalid_file = self.temp_path / 'invalid_properties.xlsx'
        result = self.discovery.validate_file_format(invalid_file)
        
        self.assertFalse(result['is_valid'])
        self.assertTrue(result['can_read'])  # Can read, but missing columns
        self.assertTrue(result['has_data'])
        self.assertGreater(len(result['missing_columns']), 0)
    
    def test_validate_empty_file(self):
        """Test validation of empty Excel file"""
        empty_file = self.temp_path / 'empty_properties.xlsx'
        result = self.discovery.validate_file_format(empty_file)
        
        self.assertFalse(result['is_valid'])
        self.assertTrue(result['can_read'])
        self.assertFalse(result['has_data'])
        self.assertEqual(result['row_count'], 0)
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file"""
        nonexistent_file = self.temp_path / 'nonexistent.xlsx'
        result = self.discovery.validate_file_format(nonexistent_file)
        
        self.assertFalse(result['is_valid'])
        self.assertFalse(result['can_read'])
        self.assertIn('File does not exist', result['errors'])
    
    def test_get_file_metadata(self):
        """Test file metadata extraction"""
        valid_file = self.temp_path / 'valid_properties.xlsx'
        metadata = self.discovery.get_file_metadata(valid_file)
        
        self.assertEqual(metadata['file_name'], 'valid_properties.xlsx')
        self.assertGreater(metadata['file_size_mb'], 0)
        self.assertEqual(metadata['sheet_count'], 1)
        self.assertEqual(metadata['total_rows'], 2)
        self.assertEqual(metadata['total_columns'], 8)
        self.assertEqual(metadata['estimated_properties'], 2)
        self.assertIsNotNone(metadata['modified_date'])
    
    def test_validate_batch_files(self):
        """Test batch file validation"""
        files = self.discovery.scan_input_folder(str(self.temp_path), recursive=False)
        batch_result = self.discovery.validate_batch_files(files)
        
        self.assertEqual(batch_result['total_files'], len(files))
        self.assertGreater(batch_result['valid_files'], 0)
        self.assertGreater(batch_result['invalid_files'], 0)
        self.assertGreater(batch_result['total_estimated_properties'], 0)
        self.assertGreater(batch_result['total_size_mb'], 0)
        self.assertEqual(len(batch_result['validation_details']), len(files))
    
    def test_filter_valid_files(self):
        """Test filtering to only valid files"""
        all_files = self.discovery.scan_input_folder(str(self.temp_path), recursive=False)
        valid_files = self.discovery.filter_valid_files(all_files)
        
        # Should have fewer valid files than total files
        self.assertLessEqual(len(valid_files), len(all_files))
        
        # All returned files should be valid
        for file_path in valid_files:
            result = self.discovery.validate_file_format(file_path)
            self.assertTrue(result['is_valid'])
    
    def test_supported_file_extensions(self):
        """Test that only supported file extensions are processed"""
        # Create files with different extensions
        xlsx_file = self.temp_path / 'test.xlsx'
        xls_file = self.temp_path / 'test.xls'
        csv_file = self.temp_path / 'test.csv'
        
        # Create minimal Excel files
        test_data = pd.DataFrame({'A': [1], 'B': [2]})
        test_data.to_excel(xlsx_file, index=False)
        test_data.to_excel(xls_file, index=False)
        test_data.to_csv(csv_file, index=False)
        
        files = self.discovery.scan_input_folder(str(self.temp_path))
        file_names = [f.name for f in files]
        
        self.assertIn('test.xlsx', file_names)
        self.assertIn('test.xls', file_names)
        self.assertNotIn('test.csv', file_names)  # CSV not supported


if __name__ == '__main__':
    unittest.main()