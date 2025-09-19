"""
Authentication component with multi-provider support
"""

import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import bcrypt

from ..core.interfaces import IAuthenticationService, User, AuthResult
from ..core.base_classes import ComponentBase
from config.settings import get_config


class AuthProvider(Enum):
    """Authentication provider types"""
    LOCAL = "local"
    OAUTH2 = "oauth2"
    SAML = "saml"
    LDAP = "ldap"


class UserRole(Enum):
    """User roles"""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


@dataclass
class Permission:
    """Permission model"""
    name: str
    description: str
    resource: str
    action: str


@dataclass
class LoginAttempt:
    """Login attempt tracking"""
    username: str
    timestamp: datetime
    success: bool
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class AuthenticationComponent(ComponentBase, IAuthenticationService):
    """Authentication component with multi-provider support"""
    
    def __init__(self):
        super().__init__("AuthenticationComponent")
        self.config = get_config()
        self.current_user: Optional[User] = None
        self.login_attempts: List[LoginAttempt] = []
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.permissions_cache: Dict[str, List[Permission]] = {}
        
        # Define role permissions
        self._setup_role_permissions()
    
    def _do_initialize(self) -> None:
        """Initialize authentication component"""
        self.logger.info("Setting up authentication providers")
        # Initialize JWT settings
        if not self.config.security.secret_key:
            self.config.security.secret_key = secrets.token_urlsafe(32)
            self.logger.warning("Generated new secret key - should be set in production")
    
    def _do_cleanup(self) -> None:
        """Cleanup authentication resources"""
        self.active_sessions.clear()
        self.permissions_cache.clear()
    
    def _setup_role_permissions(self) -> None:
        """Setup role-based permissions"""
        admin_permissions = [
            Permission("user_management", "Manage users", "users", "create,read,update,delete"),
            Permission("system_config", "System configuration", "system", "read,update"),
            Permission("audit_logs", "View audit logs", "audit", "read"),
            Permission("data_management", "Full data access", "data", "create,read,update,delete"),
            Permission("export_data", "Export data", "export", "create"),
            Permission("analytics", "View analytics", "analytics", "read"),
            Permission("ml_features", "Use ML features", "ml", "read,execute")
        ]
        
        analyst_permissions = [
            Permission("data_analysis", "Analyze data", "data", "read,update"),
            Permission("export_data", "Export data", "export", "create"),
            Permission("analytics", "View analytics", "analytics", "read"),
            Permission("ml_features", "Use ML features", "ml", "read,execute"),
            Permission("filter_data", "Filter data", "filters", "create,read,update")
        ]
        
        viewer_permissions = [
            Permission("data_view", "View data", "data", "read"),
            Permission("analytics_view", "View analytics", "analytics", "read"),
            Permission("export_limited", "Limited export", "export", "create")
        ]
        
        self.permissions_cache = {
            UserRole.ADMIN.value: admin_permissions,
            UserRole.ANALYST.value: analyst_permissions,
            UserRole.VIEWER.value: viewer_permissions
        }
    
    def authenticate_user(self, credentials: Dict[str, Any]) -> AuthResult:
        """Authenticate a user with credentials"""
        try:
            provider = credentials.get("provider", AuthProvider.LOCAL.value)
            username = credentials.get("username", "")
            
            # Check for account lockout
            if self._is_account_locked(username):
                return AuthResult(
                    success=False,
                    error_message="Account is temporarily locked due to too many failed attempts"
                )
            
            if provider == AuthProvider.LOCAL.value:
                return self._authenticate_local(credentials)
            elif provider == AuthProvider.OAUTH2.value:
                return self._authenticate_oauth2(credentials)
            elif provider == AuthProvider.SAML.value:
                return self._authenticate_saml(credentials)
            elif provider == AuthProvider.LDAP.value:
                return self._authenticate_ldap(credentials)
            else:
                return AuthResult(success=False, error_message="Unsupported authentication provider")
        
        except Exception as e:
            self.logger.error(f"Authentication error: {str(e)}")
            return AuthResult(success=False, error_message="Authentication failed")
    
    def _authenticate_local(self, credentials: Dict[str, Any]) -> AuthResult:
        """Authenticate using local credentials"""
        username = credentials.get("username", "")
        password = credentials.get("password", "")
        
        # For demo purposes, create some default users
        default_users = {
            "admin": {
                "password_hash": bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()),
                "role": UserRole.ADMIN.value,
                "email": "admin@example.com"
            },
            "analyst": {
                "password_hash": bcrypt.hashpw("analyst123".encode(), bcrypt.gensalt()),
                "role": UserRole.ANALYST.value,
                "email": "analyst@example.com"
            },
            "viewer": {
                "password_hash": bcrypt.hashpw("viewer123".encode(), bcrypt.gensalt()),
                "role": UserRole.VIEWER.value,
                "email": "viewer@example.com"
            }
        }
        
        if username in default_users:
            user_data = default_users[username]
            if bcrypt.checkpw(password.encode(), user_data["password_hash"]):
                user = User(
                    id=username,
                    username=username,
                    email=user_data["email"],
                    role=user_data["role"],
                    permissions=[p.name for p in self.permissions_cache.get(user_data["role"], [])],
                    last_login=datetime.now(),
                    created_at=datetime.now(),
                    is_active=True
                )
                
                token = self._generate_jwt_token(user)
                self._record_login_attempt(username, True)
                self.current_user = user
                
                return AuthResult(success=True, user=user, token=token)
        
        self._record_login_attempt(username, False)
        return AuthResult(success=False, error_message="Invalid username or password")
    
    def _authenticate_oauth2(self, credentials: Dict[str, Any]) -> AuthResult:
        """Authenticate using OAuth2"""
        # Placeholder for OAuth2 implementation
        self.logger.info("OAuth2 authentication requested")
        return AuthResult(success=False, error_message="OAuth2 not implemented yet")
    
    def _authenticate_saml(self, credentials: Dict[str, Any]) -> AuthResult:
        """Authenticate using SAML"""
        # Placeholder for SAML implementation
        self.logger.info("SAML authentication requested")
        return AuthResult(success=False, error_message="SAML not implemented yet")
    
    def _authenticate_ldap(self, credentials: Dict[str, Any]) -> AuthResult:
        """Authenticate using LDAP"""
        # Placeholder for LDAP implementation
        self.logger.info("LDAP authentication requested")
        return AuthResult(success=False, error_message="LDAP not implemented yet")
    
    def authorize_action(self, user: User, action: str) -> bool:
        """Check if user is authorized for an action"""
        if not user or not user.is_active:
            return False
        
        user_permissions = self.permissions_cache.get(user.role, [])
        
        # Check if user has permission for the action
        for permission in user_permissions:
            if action in permission.action.split(","):
                return True
        
        return False
    
    def get_current_user(self) -> Optional[User]:
        """Get the current authenticated user"""
        return self.current_user
    
    def logout(self) -> None:
        """Logout the current user"""
        if self.current_user:
            self.logger.info(f"User logged out: {self.current_user.username}")
            self.current_user = None
    
    def refresh_token(self, token: str) -> str:
        """Refresh an authentication token"""
        try:
            payload = jwt.decode(
                token,
                self.config.security.secret_key,
                algorithms=[self.config.security.jwt_algorithm]
            )
            
            # Create new token with extended expiration
            new_payload = payload.copy()
            new_payload["exp"] = datetime.utcnow() + timedelta(
                hours=self.config.security.jwt_expiration_hours
            )
            
            return jwt.encode(
                new_payload,
                self.config.security.secret_key,
                algorithm=self.config.security.jwt_algorithm
            )
        
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")
    
    def _generate_jwt_token(self, user: User) -> str:
        """Generate JWT token for user"""
        payload = {
            "user_id": user.id,
            "username": user.username,
            "role": user.role,
            "permissions": user.permissions,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=self.config.security.jwt_expiration_hours)
        }
        
        return jwt.encode(
            payload,
            self.config.security.secret_key,
            algorithm=self.config.security.jwt_algorithm
        )
    
    def _record_login_attempt(self, username: str, success: bool) -> None:
        """Record login attempt"""
        attempt = LoginAttempt(
            username=username,
            timestamp=datetime.now(),
            success=success
        )
        
        self.login_attempts.append(attempt)
        
        # Keep only recent attempts (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        self.login_attempts = [
            attempt for attempt in self.login_attempts
            if attempt.timestamp > cutoff
        ]
    
    def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to failed attempts"""
        cutoff = datetime.now() - timedelta(minutes=self.config.security.lockout_duration_minutes)
        
        recent_failures = [
            attempt for attempt in self.login_attempts
            if (attempt.username == username and 
                not attempt.success and 
                attempt.timestamp > cutoff)
        ]
        
        return len(recent_failures) >= self.config.security.max_login_attempts
    
    def validate_token(self, token: str) -> Optional[User]:
        """Validate JWT token and return user"""
        try:
            payload = jwt.decode(
                token,
                self.config.security.secret_key,
                algorithms=[self.config.security.jwt_algorithm]
            )
            
            user = User(
                id=payload["user_id"],
                username=payload["username"],
                email="",  # Would be loaded from database in real implementation
                role=payload["role"],
                permissions=payload["permissions"],
                is_active=True
            )
            
            return user
        
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("Invalid token")
            return None
    
    def get_user_permissions(self, user: User) -> List[Permission]:
        """Get user permissions"""
        return self.permissions_cache.get(user.role, [])
    
    def create_session(self, user: User) -> str:
        """Create a new user session"""
        session_id = secrets.token_urlsafe(32)
        self.active_sessions[session_id] = {
            "user_id": user.id,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "user": user
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        session = self.active_sessions.get(session_id)
        if session:
            # Check session timeout
            timeout = timedelta(minutes=self.config.security.session_timeout_minutes)
            if datetime.now() - session["last_activity"] > timeout:
                del self.active_sessions[session_id]
                return None
            
            # Update last activity
            session["last_activity"] = datetime.now()
        
        return session
    
    def destroy_session(self, session_id: str) -> None:
        """Destroy a session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

    def create_test_user(self, email: str, name: str, roles: List[str]) -> Optional[User]:
        """Create a test user for testing purposes"""
        try:
            user = User(
                id=f"test_{secrets.token_urlsafe(8)}",
                username=email,
                email=email,
                role=roles[0] if roles else "viewer",  # Use first role or default to viewer
                permissions=[],
                is_active=True,
                last_login=datetime.now(),
                created_at=datetime.now()
            )

            self.logger.info(f"Created test user: {email} with roles: {roles}")
            return user

        except Exception as e:
            self.logger.error(f"Failed to create test user: {e}")
            return None