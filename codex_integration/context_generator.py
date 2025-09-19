"""
Context Prompt Generator
Generates domain-specific context prompts for Codex based on project analysis
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging

from codex_integration.pattern_extractor import CodePatternExtractor
from utils.logger import setup_logging


@dataclass
class ContextTemplate:
    """Template for generating context prompts"""
    name: str
    description: str
    base_prompt: str
    required_sections: List[str] = field(default_factory=list)
    optional_sections: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


@dataclass
class GeneratedContext:
    """Generated context for Codex"""
    task_type: str
    prompt: str
    relevant_patterns: Dict[str, Any]
    examples: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextPromptGenerator:
    """
    Generates context-aware prompts for Codex based on project patterns
    
    Creates domain-specific prompts that help Codex understand:
    - Property analysis domain context
    - Existing code patterns and conventions
    - Integration requirements
    - Best practices for the codebase
    """
    
    def __init__(self, project_root: str = ".", logger: Optional[logging.Logger] = None):
        """
        Initialize the context prompt generator
        
        Args:
            project_root: Root directory of the project
            logger: Optional logger instance
        """
        self.project_root = Path(project_root)
        self.logger = logger or setup_logging(name='context_generator')
        
        # Initialize pattern extractor
        self.pattern_extractor = CodePatternExtractor(project_root, logger)
        
        # Context templates for different task types
        self.templates = self._initialize_templates()
        
        # Domain-specific knowledge
        self.domain_context = {
            'name': 'Flex Property Intelligence Platform',
            'domain': 'Commercial Real Estate Analysis',
            'focus': 'Flex industrial property identification and scoring',
            'data_sources': ['Palm Beach County Property Appraiser', 'GIS data', 'Tax records'],
            'key_concepts': [
                'flex properties', 'industrial classification', 'property scoring',
                'data pipeline', 'property validation', 'export formats'
            ],
            'business_rules': [
                'Minimum building size: 20,000 sqft',
                'Lot size range: 0.5-20 acres',
                'Flex score range: 0-10',
                'Industrial property types preferred'
            ]
        }
        
        # Extracted patterns (will be populated when needed)
        self.extracted_patterns = None
        
        self.logger.info("ContextPromptGenerator initialized")
    
    def _initialize_templates(self) -> Dict[str, ContextTemplate]:
        """Initialize context templates for different task types"""
        templates = {}
        
        # Classifier template
        templates['classifier'] = ContextTemplate(
            name='Property Classifier',
            description='Generate code for property classification and scoring',
            base_prompt="""You are working on the Flex Property Intelligence Platform, a commercial real estate analysis system that identifies and scores flex industrial properties.

DOMAIN CONTEXT:
- Focus: Flex industrial property identification from 600,000+ property records
- Key metrics: Building size (≥20k sqft), lot size (0.5-20 acres), flex score (0-10)
- Data sources: Property appraiser records, GIS data, tax information

EXISTING PATTERNS:
{patterns}

TASK: Generate code for property classification that follows existing patterns and integrates with the current FlexPropertyClassifier architecture.

REQUIREMENTS:
- Follow existing naming conventions
- Use pandas DataFrame for data processing
- Implement proper error handling and logging
- Include comprehensive docstrings
- Maintain compatibility with existing interfaces""",
            required_sections=['patterns', 'examples', 'constraints'],
            optional_sections=['related_modules', 'test_cases']
        )
        
        # Pipeline template
        templates['pipeline'] = ContextTemplate(
            name='Pipeline Component',
            description='Generate pipeline components for data processing',
            base_prompt="""You are working on the Flex Property Intelligence Platform pipeline system that processes property data through multiple phases.

DOMAIN CONTEXT:
- Pipeline phases: GIS extraction, property enrichment, analysis
- Data flow: Raw data → Processing → Scoring → Export
- Architecture: Async processing, MongoDB storage, batch operations

EXISTING PATTERNS:
{patterns}

TASK: Generate pipeline component code that integrates with the existing scalable pipeline architecture.

REQUIREMENTS:
- Use async/await patterns for I/O operations
- Implement proper error handling and recovery
- Follow existing configuration patterns
- Include progress reporting and logging
- Maintain compatibility with existing pipeline interfaces""",
            required_sections=['patterns', 'architecture', 'constraints'],
            optional_sections=['configuration', 'monitoring']
        )
        
        # Validation template
        templates['validation'] = ContextTemplate(
            name='Data Validation',
            description='Generate data validation and verification code',
            base_prompt="""You are working on the Flex Property Intelligence Platform data validation system that ensures data quality and integrity.

DOMAIN CONTEXT:
- Data types: Property records, GIS data, tax information
- Validation rules: Required fields, data types, business rules
- Error handling: Graceful degradation, detailed error reporting

EXISTING PATTERNS:
{patterns}

TASK: Generate data validation code that follows existing validation patterns and maintains data quality standards.

REQUIREMENTS:
- Implement comprehensive validation rules
- Provide detailed error messages
- Follow existing error handling patterns
- Include logging for validation results
- Maintain performance for large datasets""",
            required_sections=['patterns', 'validation_rules', 'constraints'],
            optional_sections=['error_handling', 'performance']
        )
        
        # Test template
        templates['test'] = ContextTemplate(
            name='Test Code',
            description='Generate test code for existing modules',
            base_prompt="""You are writing tests for the Flex Property Intelligence Platform to ensure code quality and reliability.

DOMAIN CONTEXT:
- Testing framework: pytest with unittest
- Test types: Unit tests, integration tests, property analysis tests
- Test data: Mock property records, sample datasets

EXISTING PATTERNS:
{patterns}

TASK: Generate comprehensive test code that validates functionality and maintains code quality.

REQUIREMENTS:
- Use pytest framework with appropriate fixtures
- Include both positive and negative test cases
- Test edge cases and error conditions
- Follow existing test naming conventions
- Include property analysis domain-specific test scenarios""",
            required_sections=['patterns', 'test_cases', 'fixtures'],
            optional_sections=['mock_data', 'integration_tests']
        )
        
        # Debug template
        templates['debug'] = ContextTemplate(
            name='Debug Assistant',
            description='Help debug and optimize existing code',
            base_prompt="""You are helping debug and optimize code in the Flex Property Intelligence Platform.

DOMAIN CONTEXT:
- Common issues: Data processing errors, performance bottlenecks, integration problems
- Performance considerations: Large datasets (600k+ records), memory usage, processing time
- Integration points: MongoDB, file I/O, external APIs

EXISTING PATTERNS:
{patterns}

TASK: Analyze the provided code and suggest improvements, fixes, or optimizations.

REQUIREMENTS:
- Identify potential issues and provide solutions
- Suggest performance optimizations for large datasets
- Maintain compatibility with existing code
- Follow established error handling patterns
- Consider property analysis domain requirements""",
            required_sections=['patterns', 'analysis', 'suggestions'],
            optional_sections=['performance', 'alternatives']
        )
        
        return templates
    
    def generate_context(self, task_type: str, specific_requirements: Optional[Dict[str, Any]] = None) -> GeneratedContext:
        """
        Generate context prompt for a specific task type
        
        Args:
            task_type: Type of task ('classifier', 'pipeline', 'validation', 'test', 'debug')
            specific_requirements: Additional requirements for the task
            
        Returns:
            GeneratedContext object with prompt and metadata
        """
        try:
            self.logger.info(f"Generating context for task type: {task_type}")
            
            # Get or extract patterns if not already done
            if self.extracted_patterns is None:
                self.logger.info("Extracting patterns from project...")
                self.extracted_patterns = self.pattern_extractor.extract_patterns_from_project()
            
            # Get template for task type
            template = self.templates.get(task_type.lower())
            if not template:
                raise ValueError(f"Unknown task type: {task_type}")
            
            # Get relevant patterns for this task type
            relevant_patterns = self.pattern_extractor.get_patterns_for_context(task_type)
            
            # Build context sections
            context_sections = self._build_context_sections(task_type, relevant_patterns, specific_requirements)
            
            # Generate the prompt
            prompt = self._generate_prompt(template, context_sections)
            
            # Create generated context
            context = GeneratedContext(
                task_type=task_type,
                prompt=prompt,
                relevant_patterns=relevant_patterns,
                examples=context_sections.get('examples', []),
                constraints=context_sections.get('constraints', []),
                metadata={
                    'template_name': template.name,
                    'domain': self.domain_context['domain'],
                    'patterns_count': len(relevant_patterns.get('relevant_classes', [])) + len(relevant_patterns.get('relevant_functions', [])),
                    'generated_at': str(Path.cwd())
                }
            )
            
            self.logger.info(f"Context generated successfully for {task_type}")
            return context
            
        except Exception as e:
            self.logger.error(f"Error generating context for {task_type}: {e}")
            raise
    
    def _build_context_sections(self, task_type: str, relevant_patterns: Dict[str, Any], 
                               specific_requirements: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build context sections for prompt generation"""
        sections = {}
        
        # Patterns section
        sections['patterns'] = self._format_patterns_section(relevant_patterns)
        
        # Examples section
        sections['examples'] = self._generate_examples(task_type, relevant_patterns)
        
        # Constraints section
        sections['constraints'] = self._generate_constraints(task_type, specific_requirements)
        
        # Architecture section (for pipeline tasks)
        if task_type.lower() == 'pipeline':
            sections['architecture'] = self._generate_architecture_section()
        
        # Validation rules section (for validation tasks)
        if task_type.lower() == 'validation':
            sections['validation_rules'] = self._generate_validation_rules()
        
        # Test cases section (for test tasks)
        if task_type.lower() == 'test':
            sections['test_cases'] = self._generate_test_cases(relevant_patterns)
            sections['fixtures'] = self._generate_test_fixtures()
        
        # Analysis section (for debug tasks)
        if task_type.lower() == 'debug':
            sections['analysis'] = self._generate_debug_analysis()
            sections['suggestions'] = self._generate_debug_suggestions()
        
        return sections
    
    def _format_patterns_section(self, relevant_patterns: Dict[str, Any]) -> str:
        """Format patterns section for prompt"""
        patterns_text = []
        
        # Add relevant classes
        if relevant_patterns.get('relevant_classes'):
            patterns_text.append("EXISTING CLASSES:")
            for pattern in relevant_patterns['relevant_classes'][:3]:  # Limit to top 3
                patterns_text.append(f"- {pattern.signature}")
                if pattern.docstring:
                    patterns_text.append(f"  Purpose: {pattern.docstring.split('.')[0]}")
        
        # Add relevant functions
        if relevant_patterns.get('relevant_functions'):
            patterns_text.append("\nEXISTING FUNCTIONS:")
            for pattern in relevant_patterns['relevant_functions'][:5]:  # Limit to top 5
                patterns_text.append(f"- {pattern.signature}")
                if pattern.docstring:
                    patterns_text.append(f"  Purpose: {pattern.docstring.split('.')[0]}")
        
        # Add common imports
        if relevant_patterns.get('common_imports'):
            patterns_text.append("\nCOMMON IMPORTS:")
            for import_stmt in relevant_patterns['common_imports'][:5]:
                patterns_text.append(f"- {import_stmt}")
        
        # Add naming conventions
        if relevant_patterns.get('naming_conventions'):
            patterns_text.append("\nNAMING CONVENTIONS:")
            conventions = relevant_patterns['naming_conventions'][:5]
            patterns_text.append(f"- Class names: {', '.join(conventions)}")
        
        return '\n'.join(patterns_text) if patterns_text else "No specific patterns found."
    
    def _generate_examples(self, task_type: str, relevant_patterns: Dict[str, Any]) -> List[str]:
        """Generate examples for the task type"""
        examples = []
        
        if task_type.lower() == 'classifier':
            examples = [
                "def calculate_flex_score(self, property_data: pd.Series) -> float:",
                "def validate_property_data(self, df: pd.DataFrame) -> bool:",
                "def filter_by_building_size(self, df: pd.DataFrame, min_sqft: int) -> pd.DataFrame:"
            ]
        elif task_type.lower() == 'pipeline':
            examples = [
                "async def process_batch(self, batch_data: List[Dict]) -> List[Dict]:",
                "def configure_pipeline(self, config: PipelineConfig) -> None:",
                "async def extract_property_details(self, parcel_ids: List[str]) -> List[Dict]:"
            ]
        elif task_type.lower() == 'validation':
            examples = [
                "def validate_parcel_id(self, parcel_id: str) -> bool:",
                "def check_required_fields(self, data: Dict) -> List[str]:",
                "def validate_numeric_range(self, value: float, min_val: float, max_val: float) -> bool:"
            ]
        elif task_type.lower() == 'test':
            examples = [
                "def test_flex_score_calculation(self):",
                "def test_property_validation_with_invalid_data(self):",
                "@pytest.fixture\ndef sample_property_data(self):"
            ]
        
        return examples
    
    def _generate_constraints(self, task_type: str, specific_requirements: Optional[Dict[str, Any]]) -> List[str]:
        """Generate constraints for the task type"""
        base_constraints = [
            "Follow existing code style and conventions",
            "Include comprehensive error handling",
            "Add detailed docstrings with type hints",
            "Maintain compatibility with existing interfaces",
            "Consider performance for large datasets (600k+ records)"
        ]
        
        task_specific = {
            'classifier': [
                "Use pandas DataFrame for data processing",
                "Implement scoring algorithm with 0-10 range",
                "Include validation for required property fields"
            ],
            'pipeline': [
                "Use async/await for I/O operations",
                "Implement batch processing for large datasets",
                "Include progress reporting and logging"
            ],
            'validation': [
                "Provide detailed error messages",
                "Handle missing or invalid data gracefully",
                "Validate business rules (building size, lot size, etc.)"
            ],
            'test': [
                "Use pytest framework with appropriate fixtures",
                "Include both positive and negative test cases",
                "Test edge cases and error conditions"
            ],
            'debug': [
                "Identify root cause of issues",
                "Suggest performance optimizations",
                "Maintain backward compatibility"
            ]
        }
        
        constraints = base_constraints + task_specific.get(task_type.lower(), [])
        
        # Add specific requirements if provided
        if specific_requirements:
            if 'constraints' in specific_requirements:
                constraints.extend(specific_requirements['constraints'])
        
        return constraints
    
    def _generate_architecture_section(self) -> str:
        """Generate architecture section for pipeline tasks"""
        return """PIPELINE ARCHITECTURE:
- Phase 1: GIS Data Extraction (industrial-zoned parcels)
- Phase 2: Property Detail Enrichment (appraiser data)
- Phase 3: Flex Property Analysis (scoring and classification)
- Storage: MongoDB with batch operations
- Processing: Async with configurable batch sizes
- Error Handling: Graceful degradation with detailed logging"""
    
    def _generate_validation_rules(self) -> str:
        """Generate validation rules section"""
        return """VALIDATION RULES:
- Parcel ID: Required, non-empty string
- Building Size: Numeric, ≥ 20,000 sqft for flex candidates
- Lot Size: Numeric, 0.5-20 acres range
- Property Type: Must contain industrial keywords
- Market Value: Numeric, > 0
- Address: Required for property identification
- Municipality: Required for location analysis"""
    
    def _generate_test_cases(self, relevant_patterns: Dict[str, Any]) -> str:
        """Generate test cases section"""
        return """TEST SCENARIOS:
- Valid property data processing
- Invalid/missing data handling
- Edge cases (minimum/maximum values)
- Large dataset performance
- Error condition handling
- Integration with existing components
- Property analysis domain-specific scenarios"""
    
    def _generate_test_fixtures(self) -> str:
        """Generate test fixtures section"""
        return """TEST FIXTURES:
- sample_property_data: Valid property DataFrame
- invalid_property_data: Data with missing/invalid fields
- large_dataset: Performance testing dataset
- mock_api_responses: External API response mocks
- test_configuration: Pipeline configuration for testing"""
    
    def _generate_debug_analysis(self) -> str:
        """Generate debug analysis section"""
        return """DEBUG ANALYSIS APPROACH:
1. Identify error patterns and frequency
2. Analyze performance bottlenecks
3. Check data quality and validation issues
4. Review integration points and dependencies
5. Examine memory usage and resource consumption
6. Validate business logic implementation"""
    
    def _generate_debug_suggestions(self) -> str:
        """Generate debug suggestions section"""
        return """COMMON SOLUTIONS:
- Add input validation and error handling
- Implement batch processing for large datasets
- Use connection pooling for database operations
- Add caching for frequently accessed data
- Optimize pandas operations for memory efficiency
- Implement retry logic for external API calls"""
    
    def _generate_prompt(self, template: ContextTemplate, sections: Dict[str, Any]) -> str:
        """Generate the final prompt from template and sections"""
        try:
            # Start with base prompt
            prompt = template.base_prompt
            
            # Replace placeholders with actual content
            for key, value in sections.items():
                placeholder = f"{{{key}}}"
                if placeholder in prompt:
                    if isinstance(value, list):
                        value = '\n'.join(f"- {item}" for item in value)
                    prompt = prompt.replace(placeholder, str(value))
            
            # Add domain context
            prompt += f"\n\nDOMAIN KNOWLEDGE:\n"
            prompt += f"Platform: {self.domain_context['name']}\n"
            prompt += f"Focus: {self.domain_context['focus']}\n"
            prompt += f"Key Concepts: {', '.join(self.domain_context['key_concepts'])}\n"
            prompt += f"Business Rules:\n"
            for rule in self.domain_context['business_rules']:
                prompt += f"- {rule}\n"
            
            return prompt
            
        except Exception as e:
            self.logger.error(f"Error generating prompt: {e}")
            return template.base_prompt
    
    def save_context_to_file(self, context: GeneratedContext, output_path: str) -> str:
        """
        Save generated context to file
        
        Args:
            context: Generated context to save
            output_path: Path to save the context file
            
        Returns:
            Path to the saved file
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare context data
            context_data = {
                'task_type': context.task_type,
                'prompt': context.prompt,
                'examples': context.examples,
                'constraints': context.constraints,
                'metadata': context.metadata,
                'relevant_patterns': {
                    'classes_count': len(context.relevant_patterns.get('relevant_classes', [])),
                    'functions_count': len(context.relevant_patterns.get('relevant_functions', [])),
                    'common_imports': context.relevant_patterns.get('common_imports', []),
                    'naming_conventions': context.relevant_patterns.get('naming_conventions', [])
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(context_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Context saved to {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Error saving context to file: {e}")
            raise
    
    def get_available_task_types(self) -> List[str]:
        """Get list of available task types"""
        return list(self.templates.keys())
    
    def get_template_info(self, task_type: str) -> Dict[str, Any]:
        """Get information about a specific template"""
        template = self.templates.get(task_type.lower())
        if not template:
            return {}
        
        return {
            'name': template.name,
            'description': template.description,
            'required_sections': template.required_sections,
            'optional_sections': template.optional_sections,
            'examples': template.examples
        }


if __name__ == "__main__":
    # Example usage
    generator = ContextPromptGenerator()
    
    # Generate context for classifier task
    context = generator.generate_context('classifier')
    print("Generated context for classifier:")
    print(context.prompt[:500] + "...")
    
    # Save context to file
    output_file = generator.save_context_to_file(
        context, 
        "data/context/classifier_context.json"
    )
    print(f"Context saved to: {output_file}")
    
    # Show available task types
    print(f"Available task types: {generator.get_available_task_types()}")