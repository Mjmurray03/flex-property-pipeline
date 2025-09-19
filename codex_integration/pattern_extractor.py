"""
Code Pattern Extractor
Analyzes existing codebase to extract patterns for Codex context generation
"""

import ast
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
import logging

from utils.logger import setup_logging


@dataclass
class CodePattern:
    """Represents a code pattern found in the codebase"""
    pattern_type: str  # 'class', 'function', 'import', 'decorator', etc.
    name: str
    signature: str
    docstring: Optional[str] = None
    file_path: str = ""
    line_number: int = 0
    usage_count: int = 1
    examples: List[str] = field(default_factory=list)
    related_patterns: List[str] = field(default_factory=list)


@dataclass
class ModuleAnalysis:
    """Analysis results for a single module"""
    module_name: str
    file_path: str
    classes: List[CodePattern] = field(default_factory=list)
    functions: List[CodePattern] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    constants: List[CodePattern] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    complexity_score: int = 0


class CodePatternExtractor:
    """
    Extracts code patterns from the property analysis codebase
    
    Analyzes existing code to identify:
    - Class and function patterns
    - Import conventions
    - Data processing patterns
    - Pipeline component interfaces
    - Property analysis domain patterns
    """
    
    def __init__(self, project_root: str = ".", logger: Optional[logging.Logger] = None):
        """
        Initialize the pattern extractor
        
        Args:
            project_root: Root directory of the project
            logger: Optional logger instance
        """
        self.project_root = Path(project_root)
        self.logger = logger or setup_logging(name='pattern_extractor')
        
        # Pattern storage
        self.patterns: Dict[str, List[CodePattern]] = {
            'classes': [],
            'functions': [],
            'imports': [],
            'decorators': [],
            'constants': []
        }
        
        # Module analysis results
        self.module_analyses: Dict[str, ModuleAnalysis] = {}
        
        # Property analysis specific patterns
        self.property_patterns = {
            'data_validation': [],
            'scoring_algorithms': [],
            'pipeline_components': [],
            'data_processing': [],
            'export_formats': []
        }
        
        # Common naming conventions
        self.naming_conventions = {
            'class_names': set(),
            'function_names': set(),
            'variable_names': set(),
            'constant_names': set()
        }
        
        self.logger.info(f"CodePatternExtractor initialized for project: {self.project_root}")
    
    def extract_patterns_from_project(self) -> Dict[str, Any]:
        """
        Extract all patterns from the project
        
        Returns:
            Dictionary containing extracted patterns and analysis
        """
        try:
            self.logger.info("Starting pattern extraction from project...")
            
            # Target directories for analysis
            target_dirs = ['processors', 'pipeline', 'extractors', 'utils', 'database']
            
            total_files = 0
            for target_dir in target_dirs:
                dir_path = self.project_root / target_dir
                if dir_path.exists():
                    files_processed = self._analyze_directory(dir_path)
                    total_files += files_processed
                    self.logger.info(f"Analyzed {files_processed} files in {target_dir}/")
            
            # Extract property-specific patterns
            self._extract_property_analysis_patterns()
            
            # Analyze naming conventions
            self._analyze_naming_conventions()
            
            # Generate pattern summary
            summary = self._generate_pattern_summary()
            
            self.logger.info(f"Pattern extraction complete: {total_files} files analyzed")
            self.logger.info(f"Found {len(self.patterns['classes'])} class patterns, "
                           f"{len(self.patterns['functions'])} function patterns")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error extracting patterns: {e}")
            raise
    
    def _analyze_directory(self, directory: Path) -> int:
        """
        Analyze all Python files in a directory
        
        Args:
            directory: Directory to analyze
            
        Returns:
            Number of files processed
        """
        files_processed = 0
        
        for py_file in directory.rglob("*.py"):
            if py_file.name.startswith('__'):
                continue
                
            try:
                analysis = self._analyze_file(py_file)
                if analysis:
                    self.module_analyses[analysis.module_name] = analysis
                    files_processed += 1
                    
            except Exception as e:
                self.logger.warning(f"Failed to analyze {py_file}: {e}")
                continue
        
        return files_processed
    
    def _analyze_file(self, file_path: Path) -> Optional[ModuleAnalysis]:
        """
        Analyze a single Python file
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            ModuleAnalysis object or None if analysis failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Create module analysis
            module_name = file_path.stem
            analysis = ModuleAnalysis(
                module_name=module_name,
                file_path=str(file_path.relative_to(self.project_root))
            )
            
            # Extract module docstring
            if (tree.body and isinstance(tree.body[0], ast.Expr) and 
                isinstance(tree.body[0].value, ast.Constant) and 
                isinstance(tree.body[0].value.value, str)):
                analysis.docstring = tree.body[0].value.value
            
            # Analyze AST nodes
            for node in ast.walk(tree):
                self._analyze_ast_node(node, analysis, content)
            
            # Calculate complexity score
            analysis.complexity_score = self._calculate_complexity(tree)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
            return None
    
    def _analyze_ast_node(self, node: ast.AST, analysis: ModuleAnalysis, content: str) -> None:
        """
        Analyze a single AST node
        
        Args:
            node: AST node to analyze
            analysis: Module analysis to update
            content: File content for extracting source code
        """
        try:
            if isinstance(node, ast.ClassDef):
                self._extract_class_pattern(node, analysis, content)
            elif isinstance(node, ast.FunctionDef):
                self._extract_function_pattern(node, analysis, content)
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                self._extract_import_pattern(node, analysis)
            elif isinstance(node, ast.Assign):
                self._extract_constant_pattern(node, analysis)
                
        except Exception as e:
            self.logger.debug(f"Error analyzing AST node: {e}")
    
    def _extract_class_pattern(self, node: ast.ClassDef, analysis: ModuleAnalysis, content: str) -> None:
        """Extract class pattern from AST node"""
        try:
            # Get class signature
            bases = [self._get_node_name(base) for base in node.bases]
            signature = f"class {node.name}"
            if bases:
                signature += f"({', '.join(bases)})"
            
            # Extract docstring
            docstring = None
            if (node.body and isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, ast.Constant)):
                docstring = node.body[0].value.value
            
            # Create pattern
            pattern = CodePattern(
                pattern_type='class',
                name=node.name,
                signature=signature,
                docstring=docstring,
                file_path=analysis.file_path,
                line_number=node.lineno
            )
            
            # Extract methods
            methods = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods.append(item.name)
            
            if methods:
                pattern.examples = [f"Methods: {', '.join(methods)}"]
            
            analysis.classes.append(pattern)
            self.patterns['classes'].append(pattern)
            
            # Track naming convention
            self.naming_conventions['class_names'].add(node.name)
            
        except Exception as e:
            self.logger.debug(f"Error extracting class pattern: {e}")
    
    def _extract_function_pattern(self, node: ast.FunctionDef, analysis: ModuleAnalysis, content: str) -> None:
        """Extract function pattern from AST node"""
        try:
            # Get function signature
            args = []
            for arg in node.args.args:
                args.append(arg.arg)
            
            signature = f"def {node.name}({', '.join(args)})"
            
            # Extract docstring
            docstring = None
            if (node.body and isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, ast.Constant)):
                docstring = node.body[0].value.value
            
            # Create pattern
            pattern = CodePattern(
                pattern_type='function',
                name=node.name,
                signature=signature,
                docstring=docstring,
                file_path=analysis.file_path,
                line_number=node.lineno
            )
            
            # Check for decorators
            if node.decorator_list:
                decorators = [self._get_node_name(dec) for dec in node.decorator_list]
                pattern.examples = [f"Decorators: {', '.join(decorators)}"]
                analysis.decorators.extend(decorators)
            
            analysis.functions.append(pattern)
            self.patterns['functions'].append(pattern)
            
            # Track naming convention
            self.naming_conventions['function_names'].add(node.name)
            
        except Exception as e:
            self.logger.debug(f"Error extracting function pattern: {e}")
    
    def _extract_import_pattern(self, node: ast.AST, analysis: ModuleAnalysis) -> None:
        """Extract import pattern from AST node"""
        try:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_name = alias.name
                    analysis.imports.append(import_name)
                    
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    import_name = f"from {module} import {alias.name}"
                    analysis.imports.append(import_name)
                    
        except Exception as e:
            self.logger.debug(f"Error extracting import pattern: {e}")
    
    def _extract_constant_pattern(self, node: ast.Assign, analysis: ModuleAnalysis) -> None:
        """Extract constant pattern from assignment node"""
        try:
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    # This looks like a constant
                    pattern = CodePattern(
                        pattern_type='constant',
                        name=target.id,
                        signature=f"{target.id} = ...",
                        file_path=analysis.file_path,
                        line_number=node.lineno
                    )
                    
                    analysis.constants.append(pattern)
                    self.patterns['constants'].append(pattern)
                    self.naming_conventions['constant_names'].add(target.id)
                    
        except Exception as e:
            self.logger.debug(f"Error extracting constant pattern: {e}")
    
    def _get_node_name(self, node: ast.AST) -> str:
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_node_name(node.value)}.{node.attr}"
        else:
            return str(node)
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate complexity score for AST tree"""
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 1
            elif isinstance(node, ast.FunctionDef):
                complexity += 1
            elif isinstance(node, ast.ClassDef):
                complexity += 2
        return complexity
    
    def _extract_property_analysis_patterns(self) -> None:
        """Extract property analysis specific patterns"""
        try:
            # Data validation patterns
            for pattern in self.patterns['functions']:
                if any(keyword in pattern.name.lower() for keyword in ['validate', 'check', 'verify']):
                    self.property_patterns['data_validation'].append(pattern)
            
            # Scoring algorithm patterns
            for pattern in self.patterns['functions']:
                if any(keyword in pattern.name.lower() for keyword in ['score', 'calculate', 'rate']):
                    self.property_patterns['scoring_algorithms'].append(pattern)
            
            # Pipeline component patterns
            for pattern in self.patterns['classes']:
                if any(keyword in pattern.name.lower() for keyword in ['pipeline', 'processor', 'extractor']):
                    self.property_patterns['pipeline_components'].append(pattern)
            
            # Data processing patterns
            for pattern in self.patterns['functions']:
                if any(keyword in pattern.name.lower() for keyword in ['process', 'transform', 'filter', 'clean']):
                    self.property_patterns['data_processing'].append(pattern)
            
            # Export format patterns
            for pattern in self.patterns['functions']:
                if any(keyword in pattern.name.lower() for keyword in ['export', 'save', 'write', 'output']):
                    self.property_patterns['export_formats'].append(pattern)
                    
        except Exception as e:
            self.logger.error(f"Error extracting property analysis patterns: {e}")
    
    def _analyze_naming_conventions(self) -> None:
        """Analyze naming conventions used in the codebase"""
        try:
            # Analyze class naming patterns
            class_patterns = {}
            for name in self.naming_conventions['class_names']:
                if name.endswith('Classifier'):
                    class_patterns['classifier'] = class_patterns.get('classifier', 0) + 1
                elif name.endswith('Processor'):
                    class_patterns['processor'] = class_patterns.get('processor', 0) + 1
                elif name.endswith('Extractor'):
                    class_patterns['extractor'] = class_patterns.get('extractor', 0) + 1
                elif name.endswith('Manager'):
                    class_patterns['manager'] = class_patterns.get('manager', 0) + 1
            
            # Analyze function naming patterns
            function_patterns = {}
            for name in self.naming_conventions['function_names']:
                if name.startswith('get_'):
                    function_patterns['getter'] = function_patterns.get('getter', 0) + 1
                elif name.startswith('set_'):
                    function_patterns['setter'] = function_patterns.get('setter', 0) + 1
                elif name.startswith('calculate_'):
                    function_patterns['calculator'] = function_patterns.get('calculator', 0) + 1
                elif name.startswith('validate_'):
                    function_patterns['validator'] = function_patterns.get('validator', 0) + 1
            
            self.logger.info(f"Class naming patterns: {class_patterns}")
            self.logger.info(f"Function naming patterns: {function_patterns}")
            
        except Exception as e:
            self.logger.error(f"Error analyzing naming conventions: {e}")
    
    def _generate_pattern_summary(self) -> Dict[str, Any]:
        """Generate comprehensive pattern summary"""
        try:
            summary = {
                'extraction_timestamp': str(Path.cwd()),
                'total_modules': len(self.module_analyses),
                'pattern_counts': {
                    'classes': len(self.patterns['classes']),
                    'functions': len(self.patterns['functions']),
                    'constants': len(self.patterns['constants'])
                },
                'property_analysis_patterns': {
                    'data_validation': len(self.property_patterns['data_validation']),
                    'scoring_algorithms': len(self.property_patterns['scoring_algorithms']),
                    'pipeline_components': len(self.property_patterns['pipeline_components']),
                    'data_processing': len(self.property_patterns['data_processing']),
                    'export_formats': len(self.property_patterns['export_formats'])
                },
                'naming_conventions': {
                    'class_names': len(self.naming_conventions['class_names']),
                    'function_names': len(self.naming_conventions['function_names']),
                    'constant_names': len(self.naming_conventions['constant_names'])
                },
                'module_complexity': {
                    module: analysis.complexity_score 
                    for module, analysis in self.module_analyses.items()
                },
                'common_imports': self._get_common_imports(),
                'pattern_examples': self._get_pattern_examples()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating pattern summary: {e}")
            return {}
    
    def _get_common_imports(self) -> List[str]:
        """Get most common imports across modules"""
        import_counts = {}
        for analysis in self.module_analyses.values():
            for import_stmt in analysis.imports:
                import_counts[import_stmt] = import_counts.get(import_stmt, 0) + 1
        
        # Return top 10 most common imports
        return sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def _get_pattern_examples(self) -> Dict[str, List[str]]:
        """Get examples of each pattern type"""
        examples = {}
        
        # Class examples
        examples['classes'] = [
            pattern.signature for pattern in self.patterns['classes'][:5]
        ]
        
        # Function examples
        examples['functions'] = [
            pattern.signature for pattern in self.patterns['functions'][:5]
        ]
        
        # Property analysis examples
        examples['property_analysis'] = []
        for category, patterns in self.property_patterns.items():
            if patterns:
                examples['property_analysis'].append(f"{category}: {patterns[0].name}")
        
        return examples
    
    def save_patterns_to_file(self, output_path: str = "data/patterns/extracted_patterns.json") -> str:
        """
        Save extracted patterns to JSON file
        
        Args:
            output_path: Path to save the patterns file
            
        Returns:
            Path to the saved file
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for JSON serialization
            patterns_data = {
                'summary': self._generate_pattern_summary(),
                'modules': {
                    name: {
                        'file_path': analysis.file_path,
                        'docstring': analysis.docstring,
                        'complexity_score': analysis.complexity_score,
                        'classes': [
                            {
                                'name': p.name,
                                'signature': p.signature,
                                'docstring': p.docstring,
                                'line_number': p.line_number
                            } for p in analysis.classes
                        ],
                        'functions': [
                            {
                                'name': p.name,
                                'signature': p.signature,
                                'docstring': p.docstring,
                                'line_number': p.line_number
                            } for p in analysis.functions
                        ],
                        'imports': analysis.imports,
                        'constants': [
                            {
                                'name': p.name,
                                'signature': p.signature,
                                'line_number': p.line_number
                            } for p in analysis.constants
                        ]
                    } for name, analysis in self.module_analyses.items()
                },
                'property_patterns': {
                    category: [
                        {
                            'name': p.name,
                            'signature': p.signature,
                            'file_path': p.file_path
                        } for p in patterns
                    ] for category, patterns in self.property_patterns.items()
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(patterns_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Patterns saved to {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Error saving patterns to file: {e}")
            raise
    
    def get_patterns_for_context(self, task_type: str) -> Dict[str, Any]:
        """
        Get relevant patterns for a specific task type
        
        Args:
            task_type: Type of task ('classifier', 'pipeline', 'processor', etc.)
            
        Returns:
            Dictionary containing relevant patterns for the task
        """
        try:
            context_patterns = {
                'relevant_classes': [],
                'relevant_functions': [],
                'common_imports': [],
                'naming_conventions': [],
                'examples': []
            }
            
            # Get task-specific patterns
            if task_type.lower() in ['classifier', 'classification']:
                context_patterns['relevant_classes'] = [
                    p for p in self.patterns['classes'] 
                    if 'classifier' in p.name.lower()
                ]
                context_patterns['relevant_functions'] = [
                    p for p in self.property_patterns['scoring_algorithms']
                ]
                
            elif task_type.lower() in ['pipeline', 'processor']:
                context_patterns['relevant_classes'] = [
                    p for p in self.property_patterns['pipeline_components']
                ]
                context_patterns['relevant_functions'] = [
                    p for p in self.property_patterns['data_processing']
                ]
                
            elif task_type.lower() in ['validation', 'validator']:
                context_patterns['relevant_functions'] = [
                    p for p in self.property_patterns['data_validation']
                ]
            
            # Add common imports
            context_patterns['common_imports'] = [
                import_stmt for import_stmt, count in self._get_common_imports()[:5]
            ]
            
            # Add naming conventions
            context_patterns['naming_conventions'] = list(
                self.naming_conventions['class_names']
            )[:10]
            
            return context_patterns
            
        except Exception as e:
            self.logger.error(f"Error getting patterns for context: {e}")
            return {}


if __name__ == "__main__":
    # Example usage
    extractor = CodePatternExtractor()
    patterns = extractor.extract_patterns_from_project()
    
    print("Pattern extraction complete!")
    print(f"Found {patterns['pattern_counts']['classes']} class patterns")
    print(f"Found {patterns['pattern_counts']['functions']} function patterns")
    
    # Save patterns
    output_file = extractor.save_patterns_to_file()
    print(f"Patterns saved to: {output_file}")