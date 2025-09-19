"""
Centralized configuration management with environment-specific settings
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "flex_property_db"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600


@dataclass
class CacheConfig:
    """Cache configuration"""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    ttl_default: int = 3600
    max_connections: int = 10


@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    password_min_length: int = 8
    session_timeout_minutes: int = 30
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15


@dataclass
class MLConfig:
    """Machine Learning configuration"""
    anomaly_contamination: float = 0.1
    clustering_n_clusters: int = 5
    similarity_threshold: float = 0.8
    model_cache_ttl: int = 7200
    batch_size: int = 1000


@dataclass
class AppConfig:
    """Main application configuration"""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    host: str = "localhost"
    port: int = 8501
    max_upload_size_mb: int = 100
    allowed_file_types: list = field(default_factory=lambda: ['.csv', '.xlsx', '.xls'])
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    
    # Feature flags
    enable_ml_features: bool = True
    enable_audit_logging: bool = True
    enable_caching: bool = True
    enable_api: bool = True


class ConfigManager:
    """Centralized configuration manager"""
    
    def __init__(self):
        self._config: Optional[AppConfig] = None
        self._environment = self._detect_environment()
    
    def _detect_environment(self) -> Environment:
        """Detect current environment from environment variables"""
        env_name = os.getenv('FLASK_ENV', os.getenv('APP_ENV', 'development')).lower()
        try:
            return Environment(env_name)
        except ValueError:
            return Environment.DEVELOPMENT
    
    def load_config(self) -> AppConfig:
        """Load configuration based on environment"""
        if self._config is None:
            self._config = self._create_config_for_environment(self._environment)
        return self._config
    
    def _create_config_for_environment(self, env: Environment) -> AppConfig:
        """Create configuration for specific environment"""
        config = AppConfig(environment=env)
        
        if env == Environment.DEVELOPMENT:
            config = self._configure_development(config)
        elif env == Environment.STAGING:
            config = self._configure_staging(config)
        elif env == Environment.PRODUCTION:
            config = self._configure_production(config)
        elif env == Environment.TESTING:
            config = self._configure_testing(config)
        
        # Override with environment variables
        self._apply_environment_overrides(config)
        
        return config
    
    def _configure_development(self, config: AppConfig) -> AppConfig:
        """Configure for development environment"""
        config.debug = True
        config.database.database = "flex_property_dev"
        config.cache.database = 0
        config.security.jwt_expiration_hours = 24
        return config
    
    def _configure_staging(self, config: AppConfig) -> AppConfig:
        """Configure for staging environment"""
        config.debug = False
        config.database.database = "flex_property_staging"
        config.cache.database = 1
        config.security.jwt_expiration_hours = 12
        return config
    
    def _configure_production(self, config: AppConfig) -> AppConfig:
        """Configure for production environment"""
        config.debug = False
        config.database.database = "flex_property_prod"
        config.database.pool_size = 20
        config.cache.database = 2
        config.cache.max_connections = 20
        config.security.jwt_expiration_hours = 8
        config.security.session_timeout_minutes = 15
        return config
    
    def _configure_testing(self, config: AppConfig) -> AppConfig:
        """Configure for testing environment"""
        config.debug = True
        config.database.database = "flex_property_test"
        config.cache.database = 3
        config.enable_audit_logging = False
        return config
    
    def _apply_environment_overrides(self, config: AppConfig) -> None:
        """Apply environment variable overrides"""
        # Database overrides
        if os.getenv('DB_HOST'):
            config.database.host = os.getenv('DB_HOST')
        if os.getenv('DB_PORT'):
            config.database.port = int(os.getenv('DB_PORT'))
        if os.getenv('DB_NAME'):
            config.database.database = os.getenv('DB_NAME')
        if os.getenv('DB_USER'):
            config.database.username = os.getenv('DB_USER')
        if os.getenv('DB_PASSWORD'):
            config.database.password = os.getenv('DB_PASSWORD')
        
        # Cache overrides
        if os.getenv('REDIS_HOST'):
            config.cache.host = os.getenv('REDIS_HOST')
        if os.getenv('REDIS_PORT'):
            config.cache.port = int(os.getenv('REDIS_PORT'))
        if os.getenv('REDIS_PASSWORD'):
            config.cache.password = os.getenv('REDIS_PASSWORD')
        
        # Security overrides
        if os.getenv('SECRET_KEY'):
            config.security.secret_key = os.getenv('SECRET_KEY')
        
        # App overrides
        if os.getenv('APP_HOST'):
            config.host = os.getenv('APP_HOST')
        if os.getenv('APP_PORT'):
            config.port = int(os.getenv('APP_PORT'))
    
    def get_config(self) -> AppConfig:
        """Get current configuration"""
        return self.load_config()
    
    def reload_config(self) -> AppConfig:
        """Reload configuration"""
        self._config = None
        return self.load_config()


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Get application configuration"""
    return config_manager.get_config()