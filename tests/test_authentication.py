"""
Unit tests for authentication component
"""

import pytest
from datetime import datetime, timedelta
import jwt

from app.components.authentication import AuthenticationComponent, UserRole, AuthProvider
from app.core.interfaces import User
from config.settings import get_config


class TestAuthenticationComponent:
    """Test authentication component"""
    
    def setup_method(self):
        """Setup test method"""
        self.auth_component = AuthenticationComponent()
        self.auth_component.initialize()
    
    def teardown_method(self):
        """Teardown test method"""
        self.auth_component.cleanup()
    
    def test_local_authentication_success(self):
        """Test successful local authentication"""
        credentials = {
            "provider": AuthProvider.LOCAL.value,
            "username": "admin",
            "password": "admin123"
        }
        
        result = self.auth_component.authenticate_user(credentials)
        
        assert result.success is True
        assert result.user is not None
        assert result.user.username == "admin"
        assert result.user.role == UserRole.ADMIN.value
        assert result.token is not None
    
    def test_local_authentication_failure(self):
        """Test failed local authentication"""
        credentials = {
            "provider": AuthProvider.LOCAL.value,
            "username": "admin",
            "password": "wrongpassword"
        }
        
        result = self.auth_component.authenticate_user(credentials)
        
        assert result.success is False
        assert result.user is None
        assert result.error_message == "Invalid username or password"
    
    def test_authorization_admin_user(self):
        """Test authorization for admin user"""
        user = User(
            id="admin",
            username="admin",
            email="admin@example.com",
            role=UserRole.ADMIN.value,
            permissions=["user_management", "system_config"],
            is_active=True
        )
        
        assert self.auth_component.authorize_action(user, "create") is True
        assert self.auth_component.authorize_action(user, "delete") is True
    
    def test_authorization_viewer_user(self):
        """Test authorization for viewer user"""
        user = User(
            id="viewer",
            username="viewer",
            email="viewer@example.com",
            role=UserRole.VIEWER.value,
            permissions=["data_view"],
            is_active=True
        )
        
        assert self.auth_component.authorize_action(user, "read") is True
        assert self.auth_component.authorize_action(user, "delete") is False
    
    def test_jwt_token_generation_and_validation(self):
        """Test JWT token generation and validation"""
        user = User(
            id="test_user",
            username="testuser",
            email="test@example.com",
            role=UserRole.ANALYST.value,
            permissions=["data_analysis"],
            is_active=True
        )
        
        token = self.auth_component._generate_jwt_token(user)
        assert token is not None
        
        validated_user = self.auth_component.validate_token(token)
        assert validated_user is not None
        assert validated_user.username == user.username
        assert validated_user.role == user.role
    
    def test_account_lockout(self):
        """Test account lockout after failed attempts"""
        credentials = {
            "provider": AuthProvider.LOCAL.value,
            "username": "admin",
            "password": "wrongpassword"
        }
        
        # Make multiple failed attempts
        for _ in range(6):  # More than max_login_attempts (5)
            self.auth_component.authenticate_user(credentials)
        
        # Account should be locked
        assert self.auth_component._is_account_locked("admin") is True
        
        # Even correct credentials should fail
        correct_credentials = {
            "provider": AuthProvider.LOCAL.value,
            "username": "admin",
            "password": "admin123"
        }
        
        result = self.auth_component.authenticate_user(correct_credentials)
        assert result.success is False
        assert "locked" in result.error_message.lower()
    
    def test_session_management(self):
        """Test session creation and management"""
        user = User(
            id="test_user",
            username="testuser",
            email="test@example.com",
            role=UserRole.ANALYST.value,
            permissions=["data_analysis"],
            is_active=True
        )
        
        # Create session
        session_id = self.auth_component.create_session(user)
        assert session_id is not None
        
        # Get session
        session = self.auth_component.get_session(session_id)
        assert session is not None
        assert session["user_id"] == user.id
        
        # Destroy session
        self.auth_component.destroy_session(session_id)
        session = self.auth_component.get_session(session_id)
        assert session is None
    
    def test_token_refresh(self):
        """Test token refresh functionality"""
        user = User(
            id="test_user",
            username="testuser",
            email="test@example.com",
            role=UserRole.ANALYST.value,
            permissions=["data_analysis"],
            is_active=True
        )
        
        original_token = self.auth_component._generate_jwt_token(user)
        refreshed_token = self.auth_component.refresh_token(original_token)
        
        assert refreshed_token != original_token
        
        # Both tokens should be valid
        original_user = self.auth_component.validate_token(original_token)
        refreshed_user = self.auth_component.validate_token(refreshed_token)
        
        assert original_user.username == refreshed_user.username
    
    def test_logout(self):
        """Test user logout"""
        # First authenticate a user
        credentials = {
            "provider": AuthProvider.LOCAL.value,
            "username": "admin",
            "password": "admin123"
        }
        
        result = self.auth_component.authenticate_user(credentials)
        assert result.success is True
        assert self.auth_component.current_user is not None
        
        # Logout
        self.auth_component.logout()
        assert self.auth_component.current_user is None