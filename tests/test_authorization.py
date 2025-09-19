"""
Unit tests for role-based access control system
"""

import pytest
from datetime import datetime, timedelta

from app.components.authorization import (
    RoleBasedAccessControl, ResourceType, ActionType, 
    AccessRequest, Permission, Role
)
from app.core.interfaces import User


class TestRoleBasedAccessControl:
    """Test role-based access control system"""
    
    def setup_method(self):
        """Setup test method"""
        self.rbac = RoleBasedAccessControl()
        self.rbac.initialize()
        
        # Create test user
        self.test_user = User(
            id="test_user",
            username="testuser",
            email="test@example.com",
            role="analyst",
            permissions=[],
            is_active=True
        )
    
    def teardown_method(self):
        """Teardown test method"""
        self.rbac.cleanup()
    
    def test_default_roles_creation(self):
        """Test that default roles are created"""
        assert "admin" in self.rbac.roles
        assert "analyst" in self.rbac.roles
        assert "viewer" in self.rbac.roles
        
        admin_role = self.rbac.roles["admin"]
        assert admin_role.name == "Administrator"
        assert admin_role.is_system_role is True
    
    def test_default_permissions_creation(self):
        """Test that default permissions are created"""
        assert "data_read" in self.rbac.permissions
        assert "data_create" in self.rbac.permissions
        assert "user_create" in self.rbac.permissions
        assert "export_basic" in self.rbac.permissions
        
        data_read_perm = self.rbac.permissions["data_read"]
        assert data_read_perm.resource_type == ResourceType.DATA
        assert ActionType.READ in data_read_perm.actions
    
    def test_assign_role_to_user(self):
        """Test assigning role to user"""
        success = self.rbac.assign_role_to_user("test_user", "analyst", "admin")
        assert success is True
        
        user_roles = self.rbac.get_user_roles("test_user")
        assert len(user_roles) == 1
        assert user_roles[0].id == "analyst"
    
    def test_assign_duplicate_role(self):
        """Test assigning duplicate role to user"""
        self.rbac.assign_role_to_user("test_user", "analyst", "admin")
        success = self.rbac.assign_role_to_user("test_user", "analyst", "admin")
        assert success is False
    
    def test_revoke_role_from_user(self):
        """Test revoking role from user"""
        self.rbac.assign_role_to_user("test_user", "analyst", "admin")
        success = self.rbac.revoke_role_from_user("test_user", "analyst")
        assert success is True
        
        user_roles = self.rbac.get_user_roles("test_user")
        assert len(user_roles) == 0
    
    def test_get_user_permissions(self):
        """Test getting user permissions"""
        self.rbac.assign_role_to_user("test_user", "analyst", "admin")
        permissions = self.rbac.get_user_permissions("test_user")
        
        assert len(permissions) > 0
        permission_ids = [p.id for p in permissions]
        assert "data_read" in permission_ids
        assert "ml_execute" in permission_ids
    
    def test_check_access_granted(self):
        """Test access check with granted permission"""
        self.rbac.assign_role_to_user("test_user", "analyst", "admin")
        
        request = AccessRequest(
            user=self.test_user,
            resource_type=ResourceType.DATA,
            action=ActionType.READ
        )
        
        result = self.rbac.check_access(request)
        assert result.granted is True
        assert "data_read" in result.permissions_used
    
    def test_check_access_denied(self):
        """Test access check with denied permission"""
        self.rbac.assign_role_to_user("test_user", "viewer", "admin")
        
        request = AccessRequest(
            user=self.test_user,
            resource_type=ResourceType.USER,
            action=ActionType.CREATE
        )
        
        result = self.rbac.check_access(request)
        assert result.granted is False
        assert "No permission found" in result.reason
    
    def test_check_access_with_conditions(self):
        """Test access check with permission conditions"""
        self.rbac.assign_role_to_user("test_user", "viewer", "admin")
        
        # Test export with record limit
        request = AccessRequest(
            user=self.test_user,
            resource_type=ResourceType.EXPORT,
            action=ActionType.EXPORT,
            context={"record_count": 500, "format": "csv"}
        )
        
        result = self.rbac.check_access(request)
        assert result.granted is True
        
        # Test export exceeding limit
        request.context["record_count"] = 2000
        result = self.rbac.check_access(request)
        assert result.granted is False
        assert result.conditions_met is False
    
    def test_create_custom_role(self):
        """Test creating custom role"""
        permission_ids = ["data_read", "filter_read", "analytics_read"]
        success = self.rbac.create_role(
            "custom_role", 
            "Custom Role", 
            "Custom role for testing",
            permission_ids
        )
        
        assert success is True
        assert "custom_role" in self.rbac.roles
        
        custom_role = self.rbac.roles["custom_role"]
        assert len(custom_role.permissions) == 3
        assert custom_role.is_system_role is False
    
    def test_update_role_permissions(self):
        """Test updating role permissions"""
        # Create custom role first
        self.rbac.create_role("test_role", "Test Role", "Test", ["data_read"])
        
        # Update permissions
        success = self.rbac.update_role("test_role", ["data_read", "data_create"])
        assert success is True
        
        role = self.rbac.roles["test_role"]
        assert len(role.permissions) == 2
    
    def test_update_system_role_fails(self):
        """Test that updating system role fails"""
        success = self.rbac.update_role("admin", ["data_read"])
        assert success is False
    
    def test_delete_custom_role(self):
        """Test deleting custom role"""
        self.rbac.create_role("temp_role", "Temp Role", "Temporary", ["data_read"])
        success = self.rbac.delete_role("temp_role")
        assert success is True
        assert "temp_role" not in self.rbac.roles
    
    def test_delete_system_role_fails(self):
        """Test that deleting system role fails"""
        success = self.rbac.delete_role("admin")
        assert success is False
        assert "admin" in self.rbac.roles
    
    def test_access_logging(self):
        """Test access attempt logging"""
        self.rbac.assign_role_to_user("test_user", "analyst", "admin")
        
        request = AccessRequest(
            user=self.test_user,
            resource_type=ResourceType.DATA,
            action=ActionType.READ,
            resource_id="dataset_1"
        )
        
        self.rbac.check_access(request)
        
        logs = self.rbac.get_access_log()
        assert len(logs) > 0
        
        last_log = logs[-1]
        assert last_log["user_id"] == "test_user"
        assert last_log["resource_type"] == "data"
        assert last_log["action"] == "read"
        assert last_log["resource_id"] == "dataset_1"
    
    def test_export_configuration(self):
        """Test exporting roles and permissions configuration"""
        config = self.rbac.export_roles_and_permissions()
        
        assert "permissions" in config
        assert "roles" in config
        
        assert "data_read" in config["permissions"]
        assert "admin" in config["roles"]
        
        admin_role = config["roles"]["admin"]
        assert admin_role["name"] == "Administrator"
        assert admin_role["is_system_role"] is True