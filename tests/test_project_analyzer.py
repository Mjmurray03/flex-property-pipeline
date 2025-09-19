"""
Unit tests for project structure analyzer
"""

import unittest
from unittest.mock import patch, mock_open, MagicMock
import ast
import json
from pathlib import Path

from codex_integration.project_analyzer import (
    ProjectStructureAnalyzer, ModuleInfo, ProjectStructure
)


class TestProjectStructureAnalyzer(unittest.TestCase):
    """Test cases for ProjectStructureAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = ProjectStructureAnalyzer(project_root='/test/project')
    
    def test_init(self):
        """Test analyzer initialization"""
        self.assertEqual(str(self.analyzer.project_root), '/test/project')
        self.assertIn('property', self.analyzer.property_keywords)
        self.assertIn('classifier', self.analyzer.data_processing_patterns)
    
    @patch('os.walk')
    def test_analyze_directories(self, mock_walk):
        """Test directory analysis"""
        mock_walk.return_value = [
            ('/test/project', ['processors', 'utils', '__pycache__'], ['main.py']),
            ('/test/project/processors', [], ['classifier.py']),
            ('/test/project/utils', [], ['logger.py']),
            ('/test/project/__pycache__', [], ['main.pyc'])
        ]
        
        directories = self.analyzer._analyze_directories()
        
        self.assertIn('processors', directories)
        self.assertIn('utils', directories)
        self.assertNotIn('__pycache__', directories)
    
    def test_analyze_python_file_success(self):
        """Test successful Python file analysis"""
        python_code = '''
"""Test module docstring"""
import os
from typing import Dict

class TestClass:
    def __init__(self):
        pass
    
    def public_method(self):
        pass
    
    def _private_method(self):
        pass

def test_function():
    if True:
        for i in range(10):
            pass
'''
        
        with patch('builtins.open', mock_open(read_data=python_code)):
            result = self.analyzer._analyze_python_file(Path('/test/file.py'))
        
        self.assertIsInstance(result, ModuleInfo)
        self.assertEqual(result.name, 'file')
        self.assertIn('TestClass', result.classes)
        self.assertIn('test_function', result.functions)
        self.assertNotIn('_private_method', result.functions)
        self.assertIn('os', result.imports)
        self.assertEqual(result.docstring, 'Test module docstring')
        self.assertGreater(result.complexity_score, 0)
    
    def test_analyze_python_file_syntax_error(self):
        """Test Python file analysis with syntax error"""
        invalid_code = 'def invalid_syntax(\n    pass'
        
        with patch('builtins.open', mock_open(read_data=invalid_code)):
            result = self.analyzer._analyze_python_file(Path('/test/invalid.py'))
        
        self.assertIsNone(result)
    
    def test_calculate_complexity_score(self):
        """Test complexity score calculation"""
        # Simple code with control structures
        code = '''
def test_function():
    if True:
        for i in range(10):
            try:
                pass
            except:
                pass
    while False:
        pass

class TestClass:
    def method1(self):
        pass
    
    def method2(self):
        pass
'''
        tree = ast.parse(code)
        score = self.analyzer._calculate_complexity_score(tree)
        
        # Should have complexity from if, for, try, while, class, and methods
        self.assertGreater(score, 0)
    
    def test_extract_data_models(self):
        """Test data model extraction"""
        modules = {
            'models/property.py': ModuleInfo(
                name='property',
                path='models/property.py',
                classes=['PropertyData', 'FlexProperty', 'Helper']
            ),
            'utils/logger.py': ModuleInfo(
                name='logger',
                path='utils/logger.py',
                classes=['Logger']
            )
        }
        
        data_models = self.analyzer._extract_data_models(modules)
        
        self.assertIn('models/property.py:PropertyData', data_models)
        self.assertIn('models/property.py:FlexProperty', data_models)
        self.assertNotIn('models/property.py:Helper', data_models)
        self.assertNotIn('utils/logger.py:Logger', data_models)
    
    @patch('os.walk')
    def test_find_config_files(self, mock_walk):
        """Test configuration file discovery"""
        mock_walk.return_value = [
            ('/test/project', ['config'], ['main.py', 'settings.json']),
            ('/test/project/config', [], ['app.yaml', 'database.ini', '.env'])
        ]
        
        config_files = self.analyzer._find_config_files()
        
        self.assertIn('settings.json', config_files)
        self.assertIn('config/app.yaml', config_files)
        self.assertIn('config/database.ini', config_files)
        self.assertIn('config/.env', config_files)
        self.assertNotIn('main.py', config_files)
    
    @patch('os.walk')
    def test_find_test_files(self, mock_walk):
        """Test test file discovery"""
        mock_walk.return_value = [
            ('/test/project', ['tests'], ['main.py']),
            ('/test/project/tests', [], ['test_main.py', 'helper_test.py', 'utils.py'])
        ]
        
        test_files = self.analyzer._find_test_files()
        
        self.assertIn('tests/test_main.py', test_files)
        self.assertIn('tests/helper_test.py', test_files)
        self.assertIn('tests/utils.py', test_files)  # In tests directory
        self.assertNotIn('main.py', test_files)
    
    @patch('os.walk')
    def test_find_documentation(self, mock_walk):
        """Test documentation file discovery"""
        mock_walk.return_value = [
            ('/test/project', ['docs'], ['README.md', 'main.py']),
            ('/test/project/docs', [], ['api.rst', 'guide.txt', 'LICENSE'])
        ]
        
        doc_files = self.analyzer._find_documentation()
        
        self.assertIn('README.md', doc_files)
        self.assertIn('docs/api.rst', doc_files)
        self.assertIn('docs/guide.txt', doc_files)
        self.assertIn('docs/LICENSE', doc_files)
        self.assertNotIn('main.py', doc_files)
    
    def test_analyze_dependencies(self):
        """Test dependency analysis"""
        modules = {
            'main.py': ModuleInfo(
                name='main',
                path='main.py',
                imports=['processors.classifier', 'utils.logger', 'os', 'sys']
            ),
            'processors/classifier.py': ModuleInfo(
                name='classifier',
                path='processors/classifier.py',
                imports=['utils.logger', 'pandas', 'numpy']
            )
        }
        
        dependencies = self.analyzer._analyze_dependencies(modules)
        
        self.assertIn('main.py', dependencies)
        self.assertIn('processors.classifier', dependencies['main.py'])
        self.assertIn('utils.logger', dependencies['main.py'])
        self.assertNotIn('os', dependencies['main.py'])  # External dependency
        
        self.assertIn('processors/classifier.py', dependencies)
        self.assertIn('utils.logger', dependencies['processors/classifier.py'])
    
    def test_extract_coding_patterns(self):
        """Test coding pattern extraction"""
        modules = {
            'processors/flex_property_classifier.py': ModuleInfo(
                name='flex_property_classifier',
                path='processors/flex_property_classifier.py',
                docstring='Property classification module'
            ),
            'pipeline/data_processor.py': ModuleInfo(
                name='data_processor',
                path='pipeline/data_processor.py',
                imports=['logging'],
                functions=['setup_logger']
            ),
            'tests/test_classifier.py': ModuleInfo(
                name='test_classifier',
                path='tests/test_classifier.py'
            )
        }
        
        patterns = self.analyzer._extract_coding_patterns(modules)
        
        self.assertIn('processors/flex_property_classifier.py', patterns['property_analysis'])
        self.assertIn('pipeline/data_processor.py', patterns['data_processing'])
        self.assertIn('pipeline/data_processor.py', patterns['logging'])
        self.assertIn('tests/test_classifier.py', patterns['testing'])
    
    def test_get_relevant_modules(self):
        """Test relevant module retrieval"""
        # Mock the analyze_project_structure method
        mock_structure = ProjectStructure(root_path='/test/project')
        mock_structure.modules = {
            'processors/flex_classifier.py': ModuleInfo(
                name='flex_classifier',
                path='processors/flex_classifier.py',
                classes=['FlexClassifier'],
                functions=['classify_property'],
                docstring='Flex property classification'
            ),
            'utils/logger.py': ModuleInfo(
                name='logger',
                path='utils/logger.py',
                classes=['Logger'],
                functions=['setup_logging']
            ),
            'processors/property_scorer.py': ModuleInfo(
                name='property_scorer',
                path='processors/property_scorer.py',
                classes=['PropertyScorer'],
                functions=['score_property']
            )
        }
        
        with patch.object(self.analyzer, 'analyze_project_structure', return_value=mock_structure):
            relevant = self.analyzer.get_relevant_modules('flex property classification')
        
        # Should prioritize modules with matching terms
        self.assertIn('processors/flex_classifier.py', relevant)
        self.assertTrue(relevant.index('processors/flex_classifier.py') < 
                       relevant.index('processors/property_scorer.py'))
    
    def test_structure_to_dict_and_back(self):
        """Test structure serialization and deserialization"""
        # Create test structure
        structure = ProjectStructure(root_path='/test/project')
        structure.modules['test.py'] = ModuleInfo(
            name='test',
            path='test.py',
            classes=['TestClass'],
            functions=['test_func'],
            imports=['os'],
            docstring='Test module',
            size_lines=100,
            complexity_score=5.0
        )
        structure.directories = ['processors', 'utils']
        structure.data_models = ['test.py:TestClass']
        
        # Convert to dict and back
        dict_data = self.analyzer._structure_to_dict(structure)
        restored_structure = self.analyzer._dict_to_structure(dict_data)
        
        # Verify restoration
        self.assertEqual(restored_structure.root_path, structure.root_path)
        self.assertEqual(len(restored_structure.modules), len(structure.modules))
        self.assertEqual(restored_structure.modules['test.py'].name, 'test')
        self.assertEqual(restored_structure.directories, structure.directories)
        self.assertEqual(restored_structure.data_models, structure.data_models)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('os.path.getmtime')
    def test_cache_structure(self, mock_getmtime, mock_json_dump, mock_file):
        """Test structure caching"""
        mock_getmtime.return_value = 1234567890
        
        structure = ProjectStructure(root_path='/test/project')
        self.analyzer._cache_structure(structure)
        
        mock_file.assert_called_once()
        mock_json_dump.assert_called_once()
    
    @patch('builtins.open', new_callable=mock_open, 
           read_data='{"timestamp": 1234567890, "structure": {"root_path": "/test"}}')
    @patch('json.load')
    def test_load_cached_structure(self, mock_json_load, mock_file):
        """Test cached structure loading"""
        mock_json_load.return_value = {
            'timestamp': 1234567890,
            'structure': {
                'root_path': '/test/project',
                'modules': {},
                'directories': [],
                'data_models': [],
                'api_endpoints': [],
                'config_files': [],
                'test_files': [],
                'documentation': [],
                'dependencies': {},
                'patterns': {}
            }
        }
        
        structure = self.analyzer._load_cached_structure()
        
        self.assertIsInstance(structure, ProjectStructure)
        self.assertEqual(structure.root_path, '/test/project')


if __name__ == '__main__':
    unittest.main()