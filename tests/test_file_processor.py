"""
Unit tests for FileProcessor class
Tests individual file processing components and integration with FlexPropertyScorer
"""

import unittest
import pandas as pd
import tempfile
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from pipeline.file_processor import FileProcessor, ProcessingResult


class TestProcessingResult(unittest.TestCase):
    """Test ProcessingResult dataclass functionality"""
    
    def test_processing_result_creation(self):
        """Test creating ProcessingResult with various parameters"""
        result = ProcessingResult(
            file_path="test.xlsx",
            success=True,
            property_count=100,
            flex_candidate_count=25,
            processing_time=1.5
        )
        
        self.assertEqual(result.file_path, "test.xlsx")
        self.assertTrue(result.success)
        self.assertEqual(result.property_count, 100)
        self.assertEqual(result.flex_candidate_count, 25)
        self.assertEqual(result.processing_time, 1.5)
        self.assertIsNone(result.error_message)
        self.assertIsNone(result.flex_properties)
    
    def test_processing_result_to_dict(self):
        """Test converting ProcessingResult to dictionary"""
        df = pd.DataFrame({'test': [1, 2, 3]})
        result = ProcessingResult(
            file_path="test.xlsx",
            success=True,
            flex_properties=df,
            property_count=100,
            flex_candidate_count=25,
            processing_time=1.5,
            source_file_info={'filename': 'test.xlsx'}
        )
        
        result_dict = result.to_dict()
        
        self.assertEqual(result_dict['file_path'], "test.xlsx")
        self.assertTrue(result_dict['success'])
        self.assertEqual(result_dict['property_count'], 100)
        self.assertEqual(result_dict['flex_candidate_count'], 25)
        self.assertTrue(result_dict['has_flex_properties'])
        self.assertEqual(result_dict['source_file_info']['filename'], 'test.xlsx')


class TestFileProcessor(unittest.TestCase):
    """Test FileProcessor class functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = FileProcessor(min_flex_score=4.0)
        
        # Create sample property data
        self.sample_data = {
            'parcel_id': ['P001', 'P002', 'P003'],
            'site_address': ['123 Main St', '456 Oak Ave', '789 Pine Rd'],
            'city': ['Boca Raton', 'Delray Beach', 'Boynton Beach'],
            'state': ['FL', 'FL', 'FL'],
            'zip_code': ['33431', '33444', '33435'],
            'acres': [2.5, 1.8, 5.0],
            'zoning': ['IL', 'IP', 'CG'],
            'improvement_value': [500000, 750000, 300000],
            'land_market_value': [200000, 300000, 150000],
            'total_market_value': [700000, 1050000, 450000]
        }
    
    def test_processor_initialization(self):
        """Test FileProcessor initialization"""
        processor = FileProcessor(min_flex_score=5.0)
        
        self.assertEqual(processor.min_flex_score, 5.0)
        self.assertIsNotNone(processor.flex_scorer)
        self.assertIsNotNone(processor.column_mappings)
    
    def test_extract_file_metadata(self):
        """Test file metadata extraction"""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(b'test content')
        
        try:
            metadata = self.processor._extract_file_metadata(tmp_path)
            
            self.assertIn('filename', metadata)
            self.assertIn('file_size_bytes', metadata)
            self.assertIn('file_size_mb', metadata)
            self.assertIn('modified_date', metadata)
            self.assertIn('processed_date', metadata)
            self.assertIn('file_extension', metadata)
            
            self.assertEqual(metadata['filename'], tmp_path.name)
            self.assertEqual(metadata['file_extension'], '.xlsx')
            self.assertGreater(metadata['file_size_bytes'], 0)
            
        finally:
            # Clean up
            os.unlink(tmp_path)
    
    def test_normalize_columns_success(self):
        """Test successful column normalization"""
        # Create DataFrame with various column name formats
        df = pd.DataFrame({
            'Parcel ID': ['P001', 'P002'],
            'Property Address': ['123 Main St', '456 Oak Ave'],
            'City': ['Boca Raton', 'Delray Beach'],
            'State': ['FL', 'FL'],
            'Lot Size Acres': [2.5, 1.8],
            'Zoning Code': ['IL', 'IP']
        })
        
        normalized_df = self.processor._normalize_columns(df)
        
        self.assertIsNotNone(normalized_df)
        self.assertIn('parcel_id', normalized_df.columns)
        self.assertIn('site_address', normalized_df.columns)
        self.assertIn('city', normalized_df.columns)
        self.assertIn('state', normalized_df.columns)
        self.assertIn('acres', normalized_df.columns)
        self.assertIn('zoning', normalized_df.columns)
    
    def test_normalize_columns_missing_required(self):
        """Test column normalization with missing required columns"""
        # Create DataFrame missing required columns
        df = pd.DataFrame({
            'Parcel ID': ['P001', 'P002'],
            'State': ['FL', 'FL']
            # Missing address and city
        })
        
        normalized_df = self.processor._normalize_columns(df)
        
        self.assertIsNone(normalized_df)
    
    def test_row_to_property_data(self):
        """Test converting DataFrame row to property data dictionary"""
        df = pd.DataFrame(self.sample_data)
        row = df.iloc[0]
        
        property_data = self.processor._row_to_property_data(row)
        
        self.assertEqual(property_data['parcel_id'], 'P001')
        self.assertEqual(property_data['site_address'], '123 Main St')
        self.assertEqual(property_data['city'], 'Boca Raton')
        self.assertEqual(property_data['acres'], 2.5)
        self.assertEqual(property_data['zoning'], 'IL')
        self.assertEqual(property_data['improvement_value'], 500000)
    
    def test_row_to_property_data_with_nan(self):
        """Test converting row with NaN values"""
        data = self.sample_data.copy()
        data['acres'][0] = None  # Set to None to simulate NaN
        
        df = pd.DataFrame(data)
        row = df.iloc[0]
        
        property_data = self.processor._row_to_property_data(row)
        
        self.assertEqual(property_data['acres'], 0)  # Should default to 0
        self.assertEqual(property_data['parcel_id'], 'P001')  # Other values should remain
    
    def test_row_to_property_data_string_numbers(self):
        """Test converting row with string formatted numbers"""
        data = {
            'parcel_id': ['P001'],
            'site_address': ['123 Main St'],
            'city': ['Boca Raton'],
            'state': ['FL'],
            'acres': ['2.5'],  # String number
            'improvement_value': ['$500,000'],  # Formatted currency
            'land_market_value': ['200000']  # String number
        }
        
        df = pd.DataFrame(data)
        row = df.iloc[0]
        
        property_data = self.processor._row_to_property_data(row)
        
        self.assertEqual(property_data['acres'], 2.5)
        self.assertEqual(property_data['improvement_value'], 500000)
        self.assertEqual(property_data['land_market_value'], 200000)
    
    @patch('pipeline.file_processor.pd.read_excel')
    def test_load_excel_file_success(self, mock_read_excel):
        """Test successful Excel file loading"""
        # Mock pandas read_excel to return sample data
        mock_read_excel.return_value = pd.DataFrame(self.sample_data)
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx') as tmp_file:
            tmp_path = Path(tmp_file.name)
            
            df = self.processor._load_excel_file(tmp_path)
            
            self.assertIsNotNone(df)
            self.assertEqual(len(df), 3)
            mock_read_excel.assert_called_once_with(tmp_path, engine='openpyxl')
    
    @patch('pipeline.file_processor.pd.read_excel')
    def test_load_excel_file_empty(self, mock_read_excel):
        """Test loading empty Excel file"""
        # Mock pandas read_excel to return empty DataFrame
        mock_read_excel.return_value = pd.DataFrame()
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx') as tmp_file:
            tmp_path = Path(tmp_file.name)
            
            df = self.processor._load_excel_file(tmp_path)
            
            self.assertIsNone(df)
    
    @patch('pipeline.file_processor.pd.read_excel')
    def test_load_excel_file_error(self, mock_read_excel):
        """Test Excel file loading with error"""
        # Mock pandas read_excel to raise exception
        mock_read_excel.side_effect = Exception("File corrupted")
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx') as tmp_file:
            tmp_path = Path(tmp_file.name)
            
            df = self.processor._load_excel_file(tmp_path)
            
            self.assertIsNone(df)
    
    def test_validate_file_format_valid_xlsx(self):
        """Test file format validation for valid xlsx file"""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
            # Create a simple Excel file
            df = pd.DataFrame({'test': [1, 2, 3]})
            df.to_excel(tmp_path, index=False, engine='openpyxl')
        
        try:
            is_valid = self.processor.validate_file_format(tmp_path)
            self.assertTrue(is_valid)
        finally:
            os.unlink(tmp_path)
    
    def test_validate_file_format_invalid_extension(self):
        """Test file format validation for invalid extension"""
        with tempfile.NamedTemporaryFile(suffix='.txt') as tmp_file:
            tmp_path = Path(tmp_file.name)
            
            is_valid = self.processor.validate_file_format(tmp_path)
            self.assertFalse(is_valid)
    
    def test_validate_file_format_nonexistent_file(self):
        """Test file format validation for nonexistent file"""
        fake_path = Path("nonexistent_file.xlsx")
        
        is_valid = self.processor.validate_file_format(fake_path)
        self.assertFalse(is_valid)
    
    def test_get_processing_stats(self):
        """Test getting processor statistics"""
        stats = self.processor.get_processing_stats()
        
        self.assertIn('min_flex_score', stats)
        self.assertIn('expected_columns', stats)
        self.assertIn('scorer_weights', stats)
        self.assertIn('ideal_flex_criteria', stats)
        
        self.assertEqual(stats['min_flex_score'], 4.0)
        self.assertIsInstance(stats['expected_columns'], list)
    
    @patch('pipeline.file_processor.pd.read_excel')
    def test_classify_properties_with_candidates(self, mock_read_excel):
        """Test property classification with flex candidates"""
        # Create mock data that should score well for flex
        high_score_data = {
            'parcel_id': ['P001'],
            'site_address': ['123 Industrial Blvd'],
            'city': ['Boca Raton'],
            'state': ['FL'],
            'acres': [2.5],  # Good size
            'zoning': ['IL'],  # Perfect zoning
            'improvement_value': [500000],
            'land_market_value': [200000]
        }
        
        df = pd.DataFrame(high_score_data)
        source_info = {'filename': 'test.xlsx', 'processed_date': datetime.now().isoformat()}
        
        # Mock the flex scorer to return a high score
        with patch.object(self.processor.flex_scorer, 'calculate_flex_score') as mock_score:
            mock_score.return_value = (8.5, {
                'zoning_score': 10,
                'size_score': 8,
                'building_score': 6,
                'location_score': 7,
                'activity_score': 5,
                'value_score': 6
            })
            
            with patch.object(self.processor.flex_scorer, 'get_flex_classification') as mock_class:
                mock_class.return_value = 'PRIME_FLEX'
                
                flex_df = self.processor._classify_properties(df, source_info)
                
                self.assertIsNotNone(flex_df)
                self.assertEqual(len(flex_df), 1)
                self.assertEqual(flex_df.iloc[0]['flex_score'], 8.5)
                self.assertEqual(flex_df.iloc[0]['flex_classification'], 'PRIME_FLEX')
                self.assertEqual(flex_df.iloc[0]['source_filename'], 'test.xlsx')
    
    def test_classify_properties_no_candidates(self):
        """Test property classification with no flex candidates"""
        df = pd.DataFrame(self.sample_data)
        source_info = {'filename': 'test.xlsx'}
        
        # Mock the flex scorer to return low scores
        with patch.object(self.processor.flex_scorer, 'calculate_flex_score') as mock_score:
            mock_score.return_value = (2.0, {})  # Below threshold
            
            flex_df = self.processor._classify_properties(df, source_info)
            
            self.assertIsNotNone(flex_df)
            self.assertEqual(len(flex_df), 0)  # No candidates should meet threshold


if __name__ == '__main__':
    unittest.main()