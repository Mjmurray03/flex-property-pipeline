"""
Flex Property Classifier Configuration Manager
Handles configurable scoring criteria, weights, and advanced settings
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import logging


@dataclass
class ScoringWeights:
    """Configuration for scoring weights and criteria"""
    building_size_weight: float = 1.0
    property_type_weight: float = 1.0
    lot_size_weight: float = 1.0
    age_condition_weight: float = 1.0
    occupancy_weight: float = 1.0
    
    # Building size scoring ranges
    building_size_ranges: Dict[str, Dict[str, float]] = None
    
    # Property type scoring
    property_type_scores: Dict[str, float] = None
    
    # Lot size scoring ranges
    lot_size_ranges: Dict[str, Dict[str, float]] = None
    
    # Age/condition scoring
    age_thresholds: Dict[str, int] = None
    
    def __post_init__(self):
        """Initialize default values if not provided"""
        if self.building_size_ranges is None:
            self.building_size_ranges = {
                "20k_to_50k": {"min": 20000, "max": 50000, "score": 3.0},
                "50k_to_100k": {"min": 50000, "max": 100000, "score": 2.0},
                "100k_to_200k": {"min": 100000, "max": 200000, "score": 1.0}
            }
        
        if self.property_type_scores is None:
            self.property_type_scores = {
                "flex": 3.0,
                "warehouse": 2.5,
                "distribution": 2.5,
                "light industrial": 2.0,
                "industrial": 1.5,
                "manufacturing": 1.0,
                "storage": 1.0,
                "logistics": 1.0
            }
        
        if self.lot_size_ranges is None:
            self.lot_size_ranges = {
                "ideal": {"min": 1.0, "max": 5.0, "score": 2.0},
                "good": {"min": 5.0, "max": 10.0, "score": 1.5},
                "acceptable_small": {"min": 0.5, "max": 1.0, "score": 1.0},
                "acceptable_large": {"min": 10.0, "max": 20.0, "score": 1.0}
            }
        
        if self.age_thresholds is None:
            self.age_thresholds = {
                "modern": 1990,
                "decent": 1980
            }


@dataclass
class FilteringCriteria:
    """Configuration for filtering criteria"""
    min_building_sqft: int = 20000
    min_lot_acres: float = 0.5
    max_lot_acres: float = 20.0
    industrial_keywords: List[str] = None
    
    def __post_init__(self):
        """Initialize default industrial keywords if not provided"""
        if self.industrial_keywords is None:
            self.industrial_keywords = [
                'industrial', 'warehouse', 'distribution', 'flex',
                'manufacturing', 'storage', 'logistics', 'light industrial'
            ]


@dataclass
class AdvancedSettings:
    """Configuration for advanced features"""
    enable_batch_processing: bool = True
    batch_size: int = 1000
    enable_progress_tracking: bool = True
    enable_performance_monitoring: bool = True
    enable_geographic_analysis: bool = True
    enable_size_distribution_analysis: bool = True
    parallel_processing: bool = False
    max_workers: int = 4
    cache_results: bool = True
    export_formats: List[str] = None
    
    def __post_init__(self):
        """Initialize default export formats if not provided"""
        if self.export_formats is None:
            self.export_formats = ['xlsx', 'csv', 'json']


@dataclass
class FlexClassifierConfig:
    """Complete configuration for Flex Property Classifier"""
    scoring_weights: ScoringWeights
    filtering_criteria: FilteringCriteria
    advanced_settings: AdvancedSettings
    max_flex_score: float = 10.0
    version: str = "1.0"


class FlexConfigManager:
    """Manager for Flex Property Classifier configuration"""
    
    DEFAULT_CONFIG_PATH = Path("config/flex_classifier_config.yaml")
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.logger = logging.getLogger(__name__)
        self._config: Optional[FlexClassifierConfig] = None
    
    def load_config(self, config_path: Optional[Path] = None) -> FlexClassifierConfig:
        """
        Load configuration from file or create default
        
        Args:
            config_path: Optional path to configuration file
            
        Returns:
            FlexClassifierConfig instance
        """
        if config_path:
            self.config_path = config_path
        
        try:
            if self.config_path.exists():
                self.logger.info(f"Loading configuration from {self.config_path}")
                return self._load_from_file()
            else:
                self.logger.info("Configuration file not found, creating default configuration")
                return self._create_default_config()
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self.logger.info("Falling back to default configuration")
            return self._create_default_config()
    
    def _load_from_file(self) -> FlexClassifierConfig:
        """Load configuration from YAML or JSON file"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        # Convert dict to dataclass instances
        scoring_weights = ScoringWeights(**data.get('scoring_weights', {}))
        filtering_criteria = FilteringCriteria(**data.get('filtering_criteria', {}))
        advanced_settings = AdvancedSettings(**data.get('advanced_settings', {}))
        
        config = FlexClassifierConfig(
            scoring_weights=scoring_weights,
            filtering_criteria=filtering_criteria,
            advanced_settings=advanced_settings,
            max_flex_score=data.get('max_flex_score', 10.0),
            version=data.get('version', '1.0')
        )
        
        self._config = config
        return config
    
    def _create_default_config(self) -> FlexClassifierConfig:
        """Create default configuration"""
        config = FlexClassifierConfig(
            scoring_weights=ScoringWeights(),
            filtering_criteria=FilteringCriteria(),
            advanced_settings=AdvancedSettings()
        )
        
        self._config = config
        return config
    
    def save_config(self, config: FlexClassifierConfig, config_path: Optional[Path] = None) -> None:
        """
        Save configuration to file
        
        Args:
            config: Configuration to save
            config_path: Optional path to save configuration
        """
        if config_path:
            self.config_path = config_path
        
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict
        config_dict = {
            'scoring_weights': asdict(config.scoring_weights),
            'filtering_criteria': asdict(config.filtering_criteria),
            'advanced_settings': asdict(config.advanced_settings),
            'max_flex_score': config.max_flex_score,
            'version': config.version
        }
        
        # Save as YAML
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Configuration saved to {self.config_path}")
        self._config = config
    
    def get_config(self) -> FlexClassifierConfig:
        """Get current configuration, loading if necessary"""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def update_scoring_weights(self, **kwargs) -> None:
        """
        Update scoring weights
        
        Args:
            **kwargs: Scoring weight parameters to update
        """
        config = self.get_config()
        
        for key, value in kwargs.items():
            if hasattr(config.scoring_weights, key):
                setattr(config.scoring_weights, key, value)
                self.logger.info(f"Updated scoring weight {key} to {value}")
        
        self.save_config(config)
    
    def update_filtering_criteria(self, **kwargs) -> None:
        """
        Update filtering criteria
        
        Args:
            **kwargs: Filtering criteria parameters to update
        """
        config = self.get_config()
        
        for key, value in kwargs.items():
            if hasattr(config.filtering_criteria, key):
                setattr(config.filtering_criteria, key, value)
                self.logger.info(f"Updated filtering criteria {key} to {value}")
        
        self.save_config(config)
    
    def update_advanced_settings(self, **kwargs) -> None:
        """
        Update advanced settings
        
        Args:
            **kwargs: Advanced settings parameters to update
        """
        config = self.get_config()
        
        for key, value in kwargs.items():
            if hasattr(config.advanced_settings, key):
                setattr(config.advanced_settings, key, value)
                self.logger.info(f"Updated advanced setting {key} to {value}")
        
        self.save_config(config)
    
    def reset_to_defaults(self) -> FlexClassifierConfig:
        """Reset configuration to defaults"""
        self.logger.info("Resetting configuration to defaults")
        config = self._create_default_config()
        self.save_config(config)
        return config
    
    def validate_config(self, config: FlexClassifierConfig) -> List[str]:
        """
        Validate configuration and return list of issues
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        # Validate scoring weights
        weights = config.scoring_weights
        if not (0 <= weights.building_size_weight <= 5):
            issues.append("Building size weight should be between 0 and 5")
        
        if not (0 <= weights.property_type_weight <= 5):
            issues.append("Property type weight should be between 0 and 5")
        
        # Validate filtering criteria
        criteria = config.filtering_criteria
        if criteria.min_building_sqft < 0:
            issues.append("Minimum building sqft cannot be negative")
        
        if criteria.min_lot_acres < 0:
            issues.append("Minimum lot acres cannot be negative")
        
        if criteria.max_lot_acres <= criteria.min_lot_acres:
            issues.append("Maximum lot acres must be greater than minimum")
        
        # Validate advanced settings
        settings = config.advanced_settings
        if settings.batch_size <= 0:
            issues.append("Batch size must be positive")
        
        if settings.max_workers <= 0:
            issues.append("Max workers must be positive")
        
        return issues
    
    def create_custom_config(self, 
                           scoring_adjustments: Optional[Dict[str, float]] = None,
                           filtering_adjustments: Optional[Dict[str, Any]] = None,
                           advanced_adjustments: Optional[Dict[str, Any]] = None) -> FlexClassifierConfig:
        """
        Create custom configuration with adjustments
        
        Args:
            scoring_adjustments: Scoring weight adjustments
            filtering_adjustments: Filtering criteria adjustments
            advanced_adjustments: Advanced settings adjustments
            
        Returns:
            Custom FlexClassifierConfig
        """
        # Start with default config
        config = self._create_default_config()
        
        # Apply scoring adjustments
        if scoring_adjustments:
            for key, value in scoring_adjustments.items():
                if hasattr(config.scoring_weights, key):
                    setattr(config.scoring_weights, key, value)
        
        # Apply filtering adjustments
        if filtering_adjustments:
            for key, value in filtering_adjustments.items():
                if hasattr(config.filtering_criteria, key):
                    setattr(config.filtering_criteria, key, value)
        
        # Apply advanced adjustments
        if advanced_adjustments:
            for key, value in advanced_adjustments.items():
                if hasattr(config.advanced_settings, key):
                    setattr(config.advanced_settings, key, value)
        
        # Validate the custom config
        issues = self.validate_config(config)
        if issues:
            self.logger.warning(f"Custom configuration has issues: {issues}")
        
        return config


def create_sample_configs() -> Dict[str, FlexClassifierConfig]:
    """Create sample configurations for different use cases"""
    manager = FlexConfigManager()
    
    configs = {}
    
    # Conservative configuration (stricter criteria)
    configs['conservative'] = manager.create_custom_config(
        scoring_adjustments={
            'building_size_weight': 1.5,
            'property_type_weight': 2.0
        },
        filtering_adjustments={
            'min_building_sqft': 30000,
            'min_lot_acres': 1.0,
            'max_lot_acres': 15.0
        }
    )
    
    # Aggressive configuration (more lenient criteria)
    configs['aggressive'] = manager.create_custom_config(
        scoring_adjustments={
            'building_size_weight': 0.8,
            'lot_size_weight': 1.2
        },
        filtering_adjustments={
            'min_building_sqft': 15000,
            'min_lot_acres': 0.3,
            'max_lot_acres': 25.0
        }
    )
    
    # Performance-optimized configuration
    configs['performance'] = manager.create_custom_config(
        advanced_adjustments={
            'enable_batch_processing': True,
            'batch_size': 2000,
            'parallel_processing': True,
            'max_workers': 8,
            'cache_results': True
        }
    )
    
    # Analysis-focused configuration
    configs['analysis'] = manager.create_custom_config(
        advanced_adjustments={
            'enable_geographic_analysis': True,
            'enable_size_distribution_analysis': True,
            'enable_performance_monitoring': True,
            'export_formats': ['xlsx', 'csv', 'json', 'parquet']
        }
    )
    
    return configs


if __name__ == '__main__':
    # Example usage
    manager = FlexConfigManager()
    
    # Create and save default config
    config = manager.load_config()
    print("Default configuration loaded")
    
    # Create sample configs
    samples = create_sample_configs()
    
    # Save sample configs
    config_dir = Path("config/samples")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    for name, sample_config in samples.items():
        sample_path = config_dir / f"flex_classifier_{name}.yaml"
        manager.save_config(sample_config, sample_path)
        print(f"Saved {name} configuration to {sample_path}")