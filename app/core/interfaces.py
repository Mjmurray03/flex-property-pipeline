"""
Base interfaces and abstract classes for component architecture
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


@dataclass
class User:
    """User model"""
    id: str
    username: str
    email: str
    role: str
    permissions: List[str]
    last_login: Optional[datetime] = None
    created_at: Optional[datetime] = None
    is_active: bool = True


@dataclass
class AuthResult:
    """Authentication result"""
    success: bool
    user: Optional[User] = None
    token: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class ValidationResult:
    """Data validation result"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    quality_score: float


@dataclass
class ProcessingReport:
    """Data processing report"""
    file_fingerprint: str
    processing_time: float
    memory_usage: float
    rows_processed: int
    columns_cleaned: List[str]
    quality_score: float
    warnings: List[str]
    recommendations: List[str]
    created_at: datetime


@dataclass
class FilterResult:
    """Filter operation result"""
    filtered_data: pd.DataFrame
    filter_summary: Dict[str, Any]
    performance_metrics: Dict[str, float]
    recommendations: List[str]


@dataclass
class ExportResult:
    """Export operation result"""
    success: bool
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    export_time: Optional[float] = None
    error_message: Optional[str] = None


class IAuthenticationService(ABC):
    """Authentication service interface"""
    
    @abstractmethod
    def authenticate_user(self, credentials: Dict[str, Any]) -> AuthResult:
        """Authenticate a user with credentials"""
        pass
    
    @abstractmethod
    def authorize_action(self, user: User, action: str) -> bool:
        """Check if user is authorized for an action"""
        pass
    
    @abstractmethod
    def get_current_user(self) -> Optional[User]:
        """Get the current authenticated user"""
        pass
    
    @abstractmethod
    def logout(self) -> None:
        """Logout the current user"""
        pass
    
    @abstractmethod
    def refresh_token(self, token: str) -> str:
        """Refresh an authentication token"""
        pass


class IDataProcessor(ABC):
    """Data processor interface"""
    
    @abstractmethod
    def process_upload(self, file_data: Any) -> Tuple[pd.DataFrame, ProcessingReport]:
        """Process uploaded file data"""
        pass
    
    @abstractmethod
    def validate_structure(self, df: pd.DataFrame) -> ValidationResult:
        """Validate data structure and quality"""
        pass
    
    @abstractmethod
    def clean_and_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and transform data"""
        pass
    
    @abstractmethod
    def generate_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality report"""
        pass


class IFilterEngine(ABC):
    """Filter engine interface"""
    
    @abstractmethod
    def apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> FilterResult:
        """Apply filters to data"""
        pass
    
    @abstractmethod
    def apply_ml_filters(self, df: pd.DataFrame, ml_options: Dict[str, Any]) -> FilterResult:
        """Apply ML-powered filters"""
        pass
    
    @abstractmethod
    def get_filter_recommendations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get AI-powered filter recommendations"""
        pass


class IAnalyticsEngine(ABC):
    """Analytics engine interface"""
    
    @abstractmethod
    def generate_market_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate market analysis"""
        pass
    
    @abstractmethod
    def create_visualizations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create data visualizations"""
        pass
    
    @abstractmethod
    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics"""
        pass


class IExportManager(ABC):
    """Export manager interface"""
    
    @abstractmethod
    def export_data(self, df: pd.DataFrame, format: str, options: Dict[str, Any]) -> ExportResult:
        """Export data in specified format"""
        pass
    
    @abstractmethod
    def generate_report(self, df: pd.DataFrame, template: str) -> ExportResult:
        """Generate formatted report"""
        pass


class ICacheManager(ABC):
    """Cache manager interface"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete cached value"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cached values"""
        pass


class IAuditLogger(ABC):
    """Audit logger interface"""
    
    @abstractmethod
    def log_user_action(self, user: User, action: str, details: Dict[str, Any]) -> None:
        """Log user action"""
        pass
    
    @abstractmethod
    def log_data_access(self, user: User, data_info: Dict[str, Any]) -> None:
        """Log data access"""
        pass
    
    @abstractmethod
    def generate_audit_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate audit report"""
        pass


class ISessionManager(ABC):
    """Session manager interface"""
    
    @abstractmethod
    def create_session(self, user: User) -> str:
        """Create a new session"""
        pass
    
    @abstractmethod
    def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        pass
    
    @abstractmethod
    def update_session_data(self, session_id: str, data: Dict[str, Any]) -> None:
        """Update session data"""
        pass
    
    @abstractmethod
    def destroy_session(self, session_id: str) -> None:
        """Destroy a session"""
        pass


class BaseComponent(ABC):
    """Base component class"""
    
    def __init__(self):
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the component"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup component resources"""
        pass
    
    def is_initialized(self) -> bool:
        """Check if component is initialized"""
        return self._initialized


class BaseService(ABC):
    """Base service class"""
    
    def __init__(self):
        self._logger = None
    
    @abstractmethod
    def start(self) -> None:
        """Start the service"""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the service"""
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        pass