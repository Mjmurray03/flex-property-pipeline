"""
Main Scalable Multi-File Pipeline
Orchestrates the complete pipeline from file discovery to final expperty classification
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import yaml
import json
from datetime import datetime

from utils.logger import setup_logging
from processors.flex_property_classifier import FlexPropertyClassifier
from processors.flex_scorer import FlexPropertyScorer


@dataclass
class PipelineConfiguration:
    """Configuration settings for the scalable flex pipeline"""
    
    # Input/Output paths
    input_folder: str = "data/raw"
    output_file: str = "data/exports/all_flex_properties.xlsx"
    
    # Processing settings
    batch_size: int = 10
    max_workers: int = 4
    enable_deduplication: bool = True
    min_flex_score: float = 4.0
    
    # Progress and logging
    progress_reporting: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # File processing
    file_pattern: str = "*.xlsx"
    recursive_scan: bool = False
    
    # Memory and performance
    memory_limit_gb: float = 4.0
    timeout_minutes: int = 30
    
    # Output options
    enable_csv_export: bool = True
    backup_existing: bool = True
    
    # Filtering options
    duplicate_fields: List[str] = field(default_factory=lambda: ["Address", "City", "State"])
    
    @classmethod
    def from_file(cls, config_path: str) -> 'PipelineConfiguration':
        """Load configuration from YAML or JSON file"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            elif config_file.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")
        
        return cls(**config_data)
    
    def to_file(self, config_path: str) -> None:
        """Save configuration to YAML or JSON file"""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        config_dict = {
            'input_folder': self.input_folder,
            'output_file': self.output_file,
            'batch_size': self.batch_size,
            'max_workers': self.max_workers,
            'enable_deduplication': self.enable_deduplication,
            'min_flex_score': self.min_flex_score,
            'progress_reporting': self.progress_reporting,
            'log_level': self.log_level,
            'log_file': self.log_file,
            'file_pattern': self.file_pattern,
            'recursive_scan': self.recursive_scan,
            'memory_limit_gb': self.memory_limit_gb,
            'timeout_minutes': self.timeout_minutes,
            'enable_csv_export': self.enable_csv_export,
            'backup_existing': self.backup_existing,
            'duplicate_fields': self.duplicate_fields
        }
        
        with open(config_file, 'w') as f:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif config_file.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")


class ScalableFlexPipeline:
    """
    Main pipeline class for processing multiple Excel files with flex property classification
    
    Handles file discovery, batch processing, result aggregation, and reporting
    """
    
    def __init__(self, config: Optional[PipelineConfiguration] = None, config_file: Optional[str] = None):
        """
        Initialize the scalable flex pipeline
        
        Args:
            config: Pipeline configuration object
            config_file: Path to configuration file (YAML or JSON)
        """
        # Load configuration
        if config_file:
            self.config = PipelineConfiguration.from_file(config_file)
        elif config:
            self.config = config
        else:
            self.config = PipelineConfiguration()
        
        # Set up logging
        self.logger = setup_logging(
            name='scalable_flex_pipeline',
            level=self.config.log_level,
            log_file=self.config.log_file,
            console=True,
            file_logging=True
        )
        
        # Initialize components (will be set up by integration function)
        self.file_discovery = None
        self.batch_processor = None
        self.result_aggregator = None
        self.report_generator = None
        self.output_manager = None
        self.data_validator = None
        self.pipeline_logger = None
        
        # Processing state
        self.discovered_files: List[Path] = []
        self.processing_results: List[Dict] = []
        self.aggregated_results = None
        self.processing_stats = {}
        
        self.logger.info(f"ScalableFlexPipeline initialized with configuration:")
        self.logger.info(f"  Input folder: {self.config.input_folder}")
        self.logger.info(f"  Output file: {self.config.output_file}")
        self.logger.info(f"  Max workers: {self.config.max_workers}")
        self.logger.info(f"  Deduplication: {self.config.enable_deduplication}")
    
    def create_default_config(self, config_path: str = "config/pipeline_config.yaml") -> str:
        """
        Create a default configuration file
        
        Args:
            config_path: Path where to save the configuration file
            
        Returns:
            Path to the created configuration file
        """
        try:
            config_file = Path(config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create default configuration
            default_config = PipelineConfiguration()
            default_config.to_file(config_path)
            
            self.logger.info(f"Default configuration created at: {config_path}")
            return str(config_file)
            
        except Exception as e:
            self.logger.error(f"Failed to create default configuration: {e}")
            raise
    
    def validate_configuration(self) -> bool:
        """
        Validate the current configuration
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check input folder
            input_path = Path(self.config.input_folder)
            if not input_path.exists():
                self.logger.warning(f"Input folder does not exist: {input_path}")
                self.logger.info(f"Creating input folder: {input_path}")
                input_path.mkdir(parents=True, exist_ok=True)
            
            # Check output folder
            output_path = Path(self.config.output_file)
            if not output_path.parent.exists():
                self.logger.info(f"Creating output directory: {output_path.parent}")
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Validate numeric settings
            if self.config.max_workers <= 0:
                self.logger.error("max_workers must be greater than 0")
                return False
            
            if self.config.min_flex_score < 0 or self.config.min_flex_score > 10:
                self.logger.error("min_flex_score must be between 0 and 10")
                return False
            
            if self.config.memory_limit_gb <= 0:
                self.logger.error("memory_limit_gb must be greater than 0")
                return False
            
            if self.config.timeout_minutes <= 0:
                self.logger.error("timeout_minutes must be greater than 0")
                return False
            
            # Validate duplicate fields
            if not self.config.duplicate_fields:
                self.logger.warning("No duplicate fields specified - deduplication will be disabled")
            
            self.logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current configuration
        
        Returns:
            Dictionary containing configuration summary
        """
        return {
            'input_folder': self.config.input_folder,
            'output_file': self.config.output_file,
            'processing': {
                'batch_size': self.config.batch_size,
                'max_workers': self.config.max_workers,
                'timeout_minutes': self.config.timeout_minutes,
                'memory_limit_gb': self.config.memory_limit_gb
            },
            'filtering': {
                'min_flex_score': self.config.min_flex_score,
                'enable_deduplication': self.config.enable_deduplication,
                'duplicate_fields': self.config.duplicate_fields
            },
            'output_options': {
                'enable_csv_export': self.config.enable_csv_export,
                'backup_existing': self.config.backup_existing
            },
            'file_discovery': {
                'file_pattern': self.config.file_pattern,
                'recursive_scan': self.config.recursive_scan
            }
        }
    
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete pipeline process
        
        Returns:
            Dictionary containing pipeline execution results
        """
        try:
            start_time = datetime.now()
            self.logger.info("Starting scalable flex pipeline execution...")
            
            # Validate configuration
            if not self.validate_configuration():
                raise ValueError("Configuration validation failed")
            
            # Check if components are initialized
            if not all([self.file_discovery, self.batch_processor, self.result_aggregator, 
                       self.report_generator, self.output_manager]):
                self.logger.warning("Pipeline components not fully initialized - using basic execution")
                return self._run_basic_pipeline(start_time)
            
            # Use the integrated pipeline execution from run_scalable_pipeline.py
            from run_scalable_pipeline import run_complete_pipeline
            return run_complete_pipeline(self)
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'configuration': self.get_configuration_summary()
            }
    
    def _run_basic_pipeline(self, start_time: datetime) -> Dict[str, Any]:
        """
        Run basic pipeline without full component integration
        
        Args:
            start_time: Pipeline start time
            
        Returns:
            Basic pipeline execution results
        """
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Basic file discovery
        input_path = Path(self.config.input_folder)
        if input_path.exists():
            if self.config.recursive_scan:
                discovered_files = list(input_path.rglob(self.config.file_pattern))
            else:
                discovered_files = list(input_path.glob(self.config.file_pattern))
            self.discovered_files = discovered_files
        else:
            discovered_files = []
        
        results = {
            'success': True,
            'execution_time': execution_time,
            'files_discovered': len(discovered_files),
            'files_processed': 0,  # Would need full integration for processing
            'flex_properties_found': 0,  # Would need full integration for processing
            'output_file': self.config.output_file,
            'configuration': self.get_configuration_summary(),
            'message': 'Basic pipeline execution - use run_scalable_pipeline.py for full functionality'
        }
        
        self.logger.info(f"Basic pipeline execution completed in {execution_time:.2f} seconds")
        return results


# Convenience function for quick pipeline setup
def create_pipeline(input_folder: str = "data/raw", 
                   output_file: str = "data/exports/all_flex_properties.xlsx",
                   **kwargs) -> ScalableFlexPipeline:
    """
    Create a pipeline with custom settings
    
    Args:
        input_folder: Path to folder containing Excel files
        output_file: Path for output Excel file
        **kwargs: Additional configuration options
        
    Returns:
        Configured ScalableFlexPipeline instance
    """
    config = PipelineConfiguration(
        input_folder=input_folder,
        output_file=output_file,
        **kwargs
    )
    
    return ScalableFlexPipeline(config=config)


if __name__ == "__main__":
    # Example usage
    pipeline = create_pipeline()
    
    # Create default configuration file
    pipeline.create_default_config()
    
    # Run pipeline
    results = pipeline.run_pipeline()
    print(f"Pipeline completed: {results}")