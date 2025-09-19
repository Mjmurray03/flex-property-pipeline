"""
Tests for Code Pattern Extractor
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from codex_integration.pattern_extractor import CodePatternExtractor, CodePattern, ModuleAnalysis


class TestCodePatternExtractor(unittest.TestCase):
    """Test cases for CodePatternExtractor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir)
        
        # Create test directory structure
        (self.project_root / "processors").mkdir()
        (self.project_root / "pipeline").mkdir()
        (self.project_root / "utils").mkdir()
        
        # Create test Python files
        self._create_test_files()
        
        self.extractor = CodePatternExtractor(str(self.project_root))
    
    def _create_test_files(self):
        """Create test Python files with various patterns"""
        
        # Test classifier file
        classifier_content = '''"""
Test Flex Property Classifier
"""

import pandas as pd
from typing import Optional, Dict, List

class TestFlexPropertyClassifier:
    """Test classifier for flex properties"""
    
    def __init__(self, df: pd.DataFrame):
        self.data = df
        self.MIN_BUILDING_SQFT = 20000
    
    def classify_properties(self) -> pd.DataFrame:
        """Classify properties as flex candidates"""
        return self.data
    
    def calculate_flex_score(self, row: pd.Series) -> float:
        """Calculate flex score for property"""
        return 5.0
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data"""
        return True

def get_top_candidates(n: int = 100) -> List[Dict]:
    """Get top N candidates"""
    return []

SCORING_WEIGHTS = {
    'size': 0.3,
    'location': 0.4,
    'condition': 0.3
}
'''
        
        with open(self.project_root / "processors" / "test_classifier.py", 'w') as f:
            f.write(classifier_content)
        
        # Test pipeline file
        pipeline_content = '''"""
Test Pipeline Component
"""

from typing import Dict, Any
import asyncio

class TestPipeline:
    """Test pipeline for processing"""
    
    def __init__(self):
        self.config = {}
    
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data"""
        return data
    
    def export_results(self, results: Dict) -> str:
        """Export processing results"""
        return "exported"

def transform_property_data(data: Dict) -> Dict:
    """Transform property data"""
    return data
'''
        
        with open(self.project_root / "pipeline" / "test_pipeline.py", 'w') as f:
            f.write(pipeline_content)
    
    def test_initialization(self):
        """Test extractor initialization"""
        self.assertEqual(self.extractor.project_root, self.project_root)
        self.assertIsNotNone(self.extractor.logger)
        self.assertIn('classes', self.extractor.patterns)
        self.assertIn('functions', self.extractor.patterns)
    
    def test_extract_patterns_from_project(self):
        """Test pattern extraction from project"""
        patterns = self.extractor.extract_patterns_from_project()
        
        # Check that patterns were extracted
        self.assertIsInstance(patterns, dict)
        self.assertIn('pattern_counts', patterns)
        self.assertIn('property_analysis_patterns', patterns)
        
        # Check that some patterns were found
        self.assertGreater(patterns['pattern_counts']['classes'], 0)
        self.assertGreater(patterns['pattern_counts']['functions'], 0)
    
    def test_analyze_file(self):
        """Test single file analysis"""
        test_file = self.project_root / "processors" / "test_classifier.py"
        analysis = self.extractor._analyze_file(test_file)
        
        self.assertIsInstance(analysis, ModuleAnalysis)
        self.assertEqual(analysis.module_name, "test_classifier")
        self.assertGreater(len(analysis.classes), 0)
        self.assertGreater(len(analysis.functions), 0)
        self.assertGreater(len(analysis.imports), 0)
    
    def test_extract_class_pattern(self):
        """Test class pattern extraction"""
        patterns = self.extractor.extract_patterns_from_project()
        
        # Find the test classifier class
        classifier_patterns = [
            p for p in self.extractor.patterns['classes']
            if p.name == 'TestFlexPropertyClassifier'
        ]
        
        self.assertEqual(len(classifier_patterns), 1)
        pattern = classifier_patterns[0]
        self.assertEqual(pattern.pattern_type, 'class')
        self.assertIn('TestFlexPropertyClassifier', pattern.signature)
    
    def test_extract_function_pattern(self):
        """Test function pattern extraction"""
        patterns = self.extractor.extract_patterns_from_project()
        
        # Find scoring function
        scoring_functions = [
            p for p in self.extractor.patterns['functions']
            if 'score' in p.name.lower()
        ]
        
        self.assertGreater(len(scoring_functions), 0)
        
        # Check function signature format
        for func in scoring_functions:
            self.assertEqual(func.pattern_type, 'function')
            self.assertTrue(func.signature.startswith('def '))
    
    def test_extract_constant_pattern(self):
        """Test constant pattern extraction"""
        patterns = self.extractor.extract_patterns_from_project()
        
        # Find constants
        constants = [
            p for p in self.extractor.patterns['constants']
            if p.name == 'SCORING_WEIGHTS'
        ]
        
        self.assertEqual(len(constants), 1)
        constant = constants[0]
        self.assertEqual(constant.pattern_type, 'constant')
        self.assertIn('SCORING_WEIGHTS', constant.signature)
    
    def test_property_analysis_patterns(self):
        """Test property analysis specific pattern extraction"""
        patterns = self.extractor.extract_patterns_from_project()
        
        # Check that property-specific patterns were identified
        prop_patterns = patterns['property_analysis_patterns']
        
        self.assertIn('scoring_algorithms', prop_patterns)
        self.assertIn('data_validation', prop_patterns)
        self.assertIn('pipeline_components', prop_patterns)
        
        # Should find at least some patterns
        self.assertGreater(prop_patterns['scoring_algorithms'], 0)
    
    def test_naming_conventions(self):
        """Test naming convention analysis"""
        patterns = self.extractor.extract_patterns_from_project()
        
        # Check that naming conventions were tracked
        self.assertGreater(len(self.extractor.naming_conventions['class_names']), 0)
        self.assertGreater(len(self.extractor.naming_conventions['function_names']), 0)
        
        # Check specific naming patterns
        self.assertIn('TestFlexPropertyClassifier', self.extractor.naming_conventions['class_names'])
    
    def test_get_patterns_for_context(self):
        """Test getting patterns for specific context"""
        # First extract patterns
        self.extractor.extract_patterns_from_project()
        
        # Get classifier context
        classifier_context = self.extractor.get_patterns_for_context('classifier')
        
        self.assertIn('relevant_classes', classifier_context)
        self.assertIn('relevant_functions', classifier_context)
        self.assertIn('common_imports', classifier_context)
        self.assertIn('naming_conventions', classifier_context)
        
        # Should find classifier-related patterns
        classifier_classes = [
            p for p in classifier_context['relevant_classes']
            if 'classifier' in p.name.lower()
        ]
        self.assertGreater(len(classifier_classes), 0)
    
    def test_save_patterns_to_file(self):
        """Test saving patterns to JSON file"""
        # Extract patterns first
        self.extractor.extract_patterns_from_project()
        
        # Save to temporary file
        output_path = self.project_root / "test_patterns.json"
        saved_path = self.extractor.save_patterns_to_file(str(output_path))
        
        self.assertEqual(saved_path, str(output_path))
        self.assertTrue(output_path.exists())
        
        # Verify JSON content
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        self.assertIn('summary', data)
        self.assertIn('modules', data)
        self.assertIn('property_patterns', data)
    
    def test_complexity_calculation(self):
        """Test complexity score calculation"""
        patterns = self.extractor.extract_patterns_from_project()
        
        # Check that complexity scores were calculated
        complexity_scores = patterns['module_complexity']
        self.assertGreater(len(complexity_scores), 0)
        
        # All scores should be non-negative integers
        for score in complexity_scores.values():
            self.assertIsInstance(score, int)
            self.assertGreaterEqual(score, 0)
    
    def test_common_imports_extraction(self):
        """Test common imports identification"""
        patterns = self.extractor.extract_patterns_from_project()
        
        common_imports = patterns['common_imports']
        self.assertIsInstance(common_imports, list)
        
        # Should find some imports (pandas might not be in test data)
        if common_imports:
            # Check structure of import tuples
            for import_item in common_imports:
                self.assertIsInstance(import_item, tuple)
                self.assertEqual(len(import_item), 2)  # (import_name, count)
                self.assertIsInstance(import_item[1], int)  # count should be integer
    
    def test_pattern_examples(self):
        """Test pattern examples generation"""
        patterns = self.extractor.extract_patterns_from_project()
        
        examples = patterns['pattern_examples']
        self.assertIn('classes', examples)
        self.assertIn('functions', examples)
        self.assertIn('property_analysis', examples)
        
        # Should have some class examples
        self.assertGreater(len(examples['classes']), 0)
        
        # Class examples should be signatures
        for example in examples['classes']:
            self.assertTrue(example.startswith('class '))


class TestCodePattern(unittest.TestCase):
    """Test cases for CodePattern dataclass"""
    
    def test_code_pattern_creation(self):
        """Test CodePattern creation"""
        pattern = CodePattern(
            pattern_type='function',
            name='test_function',
            signature='def test_function(arg1, arg2)',
            docstring='Test function docstring',
            file_path='test.py',
            line_number=10
        )
        
        self.assertEqual(pattern.pattern_type, 'function')
        self.assertEqual(pattern.name, 'test_function')
        self.assertEqual(pattern.signature, 'def test_function(arg1, arg2)')
        self.assertEqual(pattern.docstring, 'Test function docstring')
        self.assertEqual(pattern.file_path, 'test.py')
        self.assertEqual(pattern.line_number, 10)
        self.assertEqual(pattern.usage_count, 1)
        self.assertEqual(pattern.examples, [])
        self.assertEqual(pattern.related_patterns, [])


class TestModuleAnalysis(unittest.TestCase):
    """Test cases for ModuleAnalysis dataclass"""
    
    def test_module_analysis_creation(self):
        """Test ModuleAnalysis creation"""
        analysis = ModuleAnalysis(
            module_name='test_module',
            file_path='test/test_module.py'
        )
        
        self.assertEqual(analysis.module_name, 'test_module')
        self.assertEqual(analysis.file_path, 'test/test_module.py')
        self.assertEqual(analysis.classes, [])
        self.assertEqual(analysis.functions, [])
        self.assertEqual(analysis.imports, [])
        self.assertEqual(analysis.constants, [])
        self.assertEqual(analysis.decorators, [])
        self.assertIsNone(analysis.docstring)
        self.assertEqual(analysis.complexity_score, 0)


if __name__ == '__main__':
    unittest.main()