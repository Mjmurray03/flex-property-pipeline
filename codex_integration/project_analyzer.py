"""
Project Structure Analyzer
Analyzes the property analysis codebase to provide context for Codex CLI
"""

import ast
import os
import json
from typing import Dict, List, Optional, Set, Any
from pathlib import Path
import logging
from dataclasses import dataclass, field
import importlib.util
import inspect

from utils.logger import setup_logging


@dataclass
class ModuleInfo:
    """Information about a Python module"""
    name: str
    path: str
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    size_lines: int = 0
    complexity_score: float = 0.0


@dataclass
class ProjectStructure:
    """Complete project structure information"""
    root_path: str
    modules: Dict[str, ModuleInfo] = field(default_factory=dict)
    directories: List[str] = field(default_factory=list)
    data_models: List[str] = field(default_factory=list)
    api_endpoints: List[str] = field(default_factory=list)
    config_files: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    documentation: List[str] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    patterns: Dict[str, List[str]] = field(default_factory=dict)


class ProjectStructureAnalyzer:
    """
    Analyzer for understanding the property analysis project structure
    """
    
    def __init__(self, project_root: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize project analyzer
        
        Args:
            project_root: Root directory of the project (defaults to current directory)
            logger: Optional logger instance
        """
        self.logger = logger or setup_logging(name='project_analyzer')
        self.project_root = Path(project_root or os.getcwd())
        self.cache_file = self.project_root / '.codex_integration' / 'project_cache.json'
        self.cache_file.parent.mkdir(exist_ok=True)
        
        # Property analysis specific patterns
        self.property_keywords = [
            'property', 'flex', 'industrial', 'commercial', 'real_estate',
            'parcel', 'zoning', 'assessment', 'appraisal', 'gis'
        ]
        
        self.data_processing_patterns = [
            'classifier', 'scorer', 'processor', 'extractor', 'pipeline',
            'analyzer', 'validator', 'transformer', 'aggregator'
        ]
        
        # Directories to analyze
        self.target_directories = [
            'processors', 'pipeline', 'extractors', 'utils', 'database',
            'codex_integration', 'tests', 'docs', 'data', 'config'
        ]
        
        # File patterns to ignore
        self.ignore_patterns = {
            '__pycache__', '.git', '.pytest_cache', 'node_modules',
            '.venv', 'venv', '.env', '*.pyc', '*.pyo', '*.pyd'
        }
    
    def analyze_project_structure(self, use_cache: bool = True) -> ProjectStructure:
        """
        Analyze the complete project structure
        
        Args:
            use_cache: Whether to use cached results if available
            
        Returns:
            ProjectStructure object with complete analysis
        """
        # Check cache first
        if use_cache and self.cache_file.exists():
            try:
                cached_structure = self._load_cached_structure()
                if cached_structure and self._is_cache_valid(cached_structure):
                    self.logger.info("Using cached project structure analysis")
                    return cached_structure
            except Exception as e:
                self.logger.warning(f"Cache loading failed: {e}")
        
        self.logger.info(f"Analyzing project structure at: {self.project_root}")
        
        structure = ProjectStructure(root_path=str(self.project_root))
        
        try:
            # Analyze directory structure
            structure.directories = self._analyze_directories()
            
            # Analyze Python modules
            structure.modules = self._analyze_python_modules()
            
            # Extract data models
            structure.data_models = self._extract_data_models(structure.modules)
            
            # Find configuration files
            structure.config_files = self._find_config_files()
            
            # Find test files
            structure.test_files = self._find_test_files()
            
            # Find documentation
            structure.documentation = self._find_documentation()
            
            # Analyze dependencies
            structure.dependencies = self._analyze_dependencies(structure.modules)
            
            # Extract coding patterns
            structure.patterns = self._extract_coding_patterns(structure.modules)
            
            # Cache the results
            self._cache_structure(structure)
            
            self.logger.info(f"Project analysis complete: {len(structure.modules)} modules analyzed")
            
        except Exception as e:
            self.logger.error(f"Project analysis failed: {e}")
            raise
        
        return structure
    
    def _analyze_directories(self) -> List[str]:
        """Analyze directory structure"""
        directories = []
        
        for root, dirs, files in os.walk(self.project_root):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not any(
                d.startswith(pattern.rstrip('*')) for pattern in self.ignore_patterns
            )]
            
            rel_path = os.path.relpath(root, self.project_root)
            if rel_path != '.':
                directories.append(rel_path)
        
        return sorted(directories)
    
    def _analyze_python_modules(self) -> Dict[str, ModuleInfo]:
        """Analyze all Python modules in the project"""
        modules = {}
        
        for root, dirs, files in os.walk(self.project_root):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not any(
                d.startswith(pattern.rstrip('*')) for pattern in self.ignore_patterns
            )]
            
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(self.project_root)
                    
                    try:
                        module_info = self._analyze_python_file(file_path)
                        if module_info:
                            modules[str(rel_path)] = module_info
                    except Exception as e:
                        self.logger.warning(f"Failed to analyze {rel_path}: {e}")
        
        return modules
    
    def _analyze_python_file(self, file_path: Path) -> Optional[ModuleInfo]:
        """
        Analyze a single Python file
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            ModuleInfo object or None if analysis fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Create module info
            module_info = ModuleInfo(
                name=file_path.stem,
                path=str(file_path),
                size_lines=len(content.splitlines())
            )
            
            # Extract information from AST
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    module_info.classes.append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    if not node.name.startswith('_'):  # Skip private functions
                        module_info.functions.append(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            module_info.imports.append(alias.name)
                    else:  # ImportFrom
                        if node.module:
                            module_info.imports.append(node.module)
            
            # Extract docstring
            if (tree.body and isinstance(tree.body[0], ast.Expr) and 
                isinstance(tree.body[0].value, ast.Constant) and 
                isinstance(tree.body[0].value.value, str)):
                module_info.docstring = tree.body[0].value.value
            
            # Calculate complexity score (simple heuristic)
            module_info.complexity_score = self._calculate_complexity_score(tree)
            
            return module_info
            
        except Exception as e:
            self.logger.warning(f"Error analyzing {file_path}: {e}")
            return None
    
    def _calculate_complexity_score(self, tree: ast.AST) -> float:
        """
        Calculate a simple complexity score for the module
        
        Args:
            tree: AST tree
            
        Returns:
            Complexity score (higher = more complex)
        """
        complexity = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(node, ast.Try):
                complexity += 2
            elif isinstance(node, ast.ClassDef):
                complexity += len(node.body) * 0.5
            elif isinstance(node, ast.FunctionDef):
                complexity += len(node.body) * 0.3
        
        return complexity
    
    def _extract_data_models(self, modules: Dict[str, ModuleInfo]) -> List[str]:
        """Extract data model classes from modules"""
        data_models = []
        
        for module_path, module_info in modules.items():
            for class_name in module_info.classes:
                # Check if it's likely a data model
                if any(keyword in class_name.lower() for keyword in 
                      ['property', 'data', 'model', 'config', 'result']):
                    data_models.append(f"{module_path}:{class_name}")
        
        return data_models
    
    def _find_config_files(self) -> List[str]:
        """Find configuration files"""
        config_files = []
        config_extensions = {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg'}
        
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if not any(
                d.startswith(pattern.rstrip('*')) for pattern in self.ignore_patterns
            )]
            
            for file in files:
                file_path = Path(root) / file
                if (file_path.suffix in config_extensions or 
                    'config' in file.lower() or 
                    file.startswith('.env')):
                    rel_path = file_path.relative_to(self.project_root)
                    config_files.append(str(rel_path))
        
        return sorted(config_files)
    
    def _find_test_files(self) -> List[str]:
        """Find test files"""
        test_files = []
        
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if not any(
                d.startswith(pattern.rstrip('*')) for pattern in self.ignore_patterns
            )]
            
            for file in files:
                if (file.startswith('test_') or file.endswith('_test.py') or 
                    'test' in Path(root).name):
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(self.project_root)
                    test_files.append(str(rel_path))
        
        return sorted(test_files)
    
    def _find_documentation(self) -> List[str]:
        """Find documentation files"""
        doc_files = []
        doc_extensions = {'.md', '.rst', '.txt'}
        
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if not any(
                d.startswith(pattern.rstrip('*')) for pattern in self.ignore_patterns
            )]
            
            for file in files:
                file_path = Path(root) / file
                if (file_path.suffix in doc_extensions or 
                    file.upper() in ['README', 'CHANGELOG', 'LICENSE']):
                    rel_path = file_path.relative_to(self.project_root)
                    doc_files.append(str(rel_path))
        
        return sorted(doc_files)
    
    def _analyze_dependencies(self, modules: Dict[str, ModuleInfo]) -> Dict[str, List[str]]:
        """Analyze module dependencies"""
        dependencies = {}
        
        for module_path, module_info in modules.items():
            module_deps = []
            
            for import_name in module_info.imports:
                # Filter for project-internal dependencies
                if any(target_dir in import_name for target_dir in self.target_directories):
                    module_deps.append(import_name)
            
            if module_deps:
                dependencies[module_path] = module_deps
        
        return dependencies
    
    def _extract_coding_patterns(self, modules: Dict[str, ModuleInfo]) -> Dict[str, List[str]]:
        """Extract common coding patterns from the codebase"""
        patterns = {
            'property_analysis': [],
            'data_processing': [],
            'error_handling': [],
            'logging': [],
            'configuration': [],
            'testing': []
        }
        
        for module_path, module_info in modules.items():
            # Property analysis patterns
            if any(keyword in module_path.lower() for keyword in self.property_keywords):
                patterns['property_analysis'].append(module_path)
            
            # Data processing patterns
            if any(pattern in module_path.lower() for pattern in self.data_processing_patterns):
                patterns['data_processing'].append(module_path)
            
            # Error handling patterns
            if 'error' in module_info.docstring.lower() if module_info.docstring else False:
                patterns['error_handling'].append(module_path)
            
            # Logging patterns
            if 'logging' in module_info.imports or 'logger' in module_info.functions:
                patterns['logging'].append(module_path)
            
            # Configuration patterns
            if 'config' in module_path.lower() or any('config' in cls.lower() for cls in module_info.classes):
                patterns['configuration'].append(module_path)
            
            # Testing patterns
            if 'test' in module_path.lower():
                patterns['testing'].append(module_path)
        
        return patterns
    
    def get_relevant_modules(self, query: str) -> List[str]:
        """
        Get modules relevant to a specific query
        
        Args:
            query: Search query or task description
            
        Returns:
            List of relevant module paths
        """
        structure = self.analyze_project_structure()
        relevant_modules = []
        query_lower = query.lower()
        
        # Score modules based on relevance
        module_scores = {}
        
        for module_path, module_info in structure.modules.items():
            score = 0
            
            # Check module name
            if any(word in module_path.lower() for word in query_lower.split()):
                score += 3
            
            # Check class names
            for class_name in module_info.classes:
                if any(word in class_name.lower() for word in query_lower.split()):
                    score += 2
            
            # Check function names
            for func_name in module_info.functions:
                if any(word in func_name.lower() for word in query_lower.split()):
                    score += 1
            
            # Check docstring
            if module_info.docstring:
                if any(word in module_info.docstring.lower() for word in query_lower.split()):
                    score += 1
            
            if score > 0:
                module_scores[module_path] = score
        
        # Sort by relevance score
        relevant_modules = sorted(module_scores.keys(), 
                                key=lambda x: module_scores[x], 
                                reverse=True)
        
        return relevant_modules[:10]  # Return top 10 most relevant
    
    def _cache_structure(self, structure: ProjectStructure) -> None:
        """Cache the project structure analysis"""
        try:
            cache_data = {
                'timestamp': os.path.getmtime(self.project_root),
                'structure': self._structure_to_dict(structure)
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Failed to cache structure: {e}")
    
    def _load_cached_structure(self) -> Optional[ProjectStructure]:
        """Load cached project structure"""
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            return self._dict_to_structure(cache_data['structure'])
            
        except Exception as e:
            self.logger.warning(f"Failed to load cached structure: {e}")
            return None
    
    def _is_cache_valid(self, cached_structure: ProjectStructure) -> bool:
        """Check if cached structure is still valid"""
        try:
            cache_time = os.path.getmtime(self.cache_file)
            project_time = os.path.getmtime(self.project_root)
            
            # Cache is valid if it's newer than the project directory
            return cache_time >= project_time
            
        except Exception:
            return False
    
    def _structure_to_dict(self, structure: ProjectStructure) -> Dict:
        """Convert ProjectStructure to dictionary for JSON serialization"""
        return {
            'root_path': structure.root_path,
            'modules': {k: {
                'name': v.name,
                'path': v.path,
                'classes': v.classes,
                'functions': v.functions,
                'imports': v.imports,
                'docstring': v.docstring,
                'dependencies': v.dependencies,
                'size_lines': v.size_lines,
                'complexity_score': v.complexity_score
            } for k, v in structure.modules.items()},
            'directories': structure.directories,
            'data_models': structure.data_models,
            'api_endpoints': structure.api_endpoints,
            'config_files': structure.config_files,
            'test_files': structure.test_files,
            'documentation': structure.documentation,
            'dependencies': structure.dependencies,
            'patterns': structure.patterns
        }
    
    def _dict_to_structure(self, data: Dict) -> ProjectStructure:
        """Convert dictionary back to ProjectStructure"""
        structure = ProjectStructure(root_path=data['root_path'])
        
        # Convert modules
        for k, v in data['modules'].items():
            structure.modules[k] = ModuleInfo(**v)
        
        # Set other attributes
        structure.directories = data.get('directories', [])
        structure.data_models = data.get('data_models', [])
        structure.api_endpoints = data.get('api_endpoints', [])
        structure.config_files = data.get('config_files', [])
        structure.test_files = data.get('test_files', [])
        structure.documentation = data.get('documentation', [])
        structure.dependencies = data.get('dependencies', {})
        structure.patterns = data.get('patterns', {})
        
        return structure


def main():
    """Test the project analyzer"""
    analyzer = ProjectStructureAnalyzer()
    structure = analyzer.analyze_project_structure()
    
    print("Project Structure Analysis:")
    print("=" * 40)
    print(f"Root: {structure.root_path}")
    print(f"Modules: {len(structure.modules)}")
    print(f"Directories: {len(structure.directories)}")
    print(f"Data Models: {len(structure.data_models)}")
    print(f"Config Files: {len(structure.config_files)}")
    print(f"Test Files: {len(structure.test_files)}")
    print(f"Documentation: {len(structure.documentation)}")
    
    print("\nTop Modules by Complexity:")
    sorted_modules = sorted(structure.modules.items(), 
                          key=lambda x: x[1].complexity_score, 
                          reverse=True)
    for module_path, module_info in sorted_modules[:5]:
        print(f"  {module_path}: {module_info.complexity_score:.1f}")
    
    print("\nCoding Patterns:")
    for pattern_type, modules in structure.patterns.items():
        if modules:
            print(f"  {pattern_type}: {len(modules)} modules")


if __name__ == "__main__":
    main()