"""
Role-based access control system
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from ..core.interfaces import User
from ..core.base_classes import ComponentBase


class ResourceType(Enum):
    """Resource types in the system"""
    USER = "user"
    DATA = "data"
    FILTER = "filter"
    EXPORT = "export"
    ANALYTICS = "analytics"
    SYSTEM = "system"
    AUDIT = "audit"
    ML = "ml"


class ActionType(Enum):
    """Action types"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    EXPORT = "export"
    IMPORT = "import"
    CONFIGURE = "configure"


@dataclass
class Permission:
    """Permission model with detailed access control"""
    id: str
    name: str
    description: str
    resource_type: ResourceType
    actions: Set[ActionType]
    conditions: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Role:
    """Role model with permissions"""
    id: str
    name: str
    description: str
    permissions: List[Permission]
    is_system_role: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class UserRole:
    """User role assignment"""
    user_id: str
    role_id: str
    assigned_by: str
    assigned_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    is_active: bool = True


@dataclass
class AccessRequest:
    """Access request for authorization"""
    user: User
    resource_type: ResourceType
    action: ActionType
    resource_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessResult:
    """Result of access authorization"""
    granted: bool
    reason: str
    permissions_used: List[str] = field(default_factory=list)
    conditions_met: bool = True


class RoleBasedAccessControl(ComponentBase):
    """Role-based access control system"""
    
    def __init__(self):
        super().__init__("RoleBasedAccessControl")
        self.roles: Dict[str, Role] = {}
        self.permissions: Dict[str, Permission] = {}
        self.user_roles: Dict[str, List[UserRole]] = {}
        self.access_log: List[Dict[str, Any]] = []
        
        self._setup_default_permissions()
        self._setup_default_roles()
    
    def _do_initialize(self) -> None:
        """Initialize RBAC system"""
        self.logger.info("Initializing Role-Based Access Control system")
        self._validate_role_permissions()
    
    def _do_cleanup(self) -> None:
        """Cleanup RBAC resources"""
        self.access_log.clear()
    
    def _setup_default_permissions(self) -> None:
        """Setup default system permissions"""
        permissions = [
            # User management permissions
            Permission(
                id="user_create",
                name="Create Users",
                description="Create new user accounts",
                resource_type=ResourceType.USER,
                actions={ActionType.CREATE}
            ),
            Permission(
                id="user_read",
                name="Read Users",
                description="View user information",
                resource_type=ResourceType.USER,
                actions={ActionType.READ}
            ),
            Permission(
                id="user_update",
                name="Update Users",
                description="Modify user accounts",
                resource_type=ResourceType.USER,
                actions={ActionType.UPDATE}
            ),
            Permission(
                id="user_delete",
                name="Delete Users",
                description="Delete user accounts",
                resource_type=ResourceType.USER,
                actions={ActionType.DELETE}
            ),
            
            # Data permissions
            Permission(
                id="data_read",
                name="Read Data",
                description="View property data",
                resource_type=ResourceType.DATA,
                actions={ActionType.READ}
            ),
            Permission(
                id="data_create",
                name="Create Data",
                description="Upload and create new data",
                resource_type=ResourceType.DATA,
                actions={ActionType.CREATE, ActionType.IMPORT}
            ),
            Permission(
                id="data_update",
                name="Update Data",
                description="Modify existing data",
                resource_type=ResourceType.DATA,
                actions={ActionType.UPDATE}
            ),
            Permission(
                id="data_delete",
                name="Delete Data",
                description="Delete data records",
                resource_type=ResourceType.DATA,
                actions={ActionType.DELETE}
            ),
            
            # Filter permissions
            Permission(
                id="filter_create",
                name="Create Filters",
                description="Create data filters",
                resource_type=ResourceType.FILTER,
                actions={ActionType.CREATE}
            ),
            Permission(
                id="filter_read",
                name="Read Filters",
                description="View and use filters",
                resource_type=ResourceType.FILTER,
                actions={ActionType.READ}
            ),
            Permission(
                id="filter_update",
                name="Update Filters",
                description="Modify existing filters",
                resource_type=ResourceType.FILTER,
                actions={ActionType.UPDATE}
            ),
            Permission(
                id="filter_delete",
                name="Delete Filters",
                description="Delete filters",
                resource_type=ResourceType.FILTER,
                actions={ActionType.DELETE}
            ),
            
            # Export permissions
            Permission(
                id="export_basic",
                name="Basic Export",
                description="Export data in basic formats",
                resource_type=ResourceType.EXPORT,
                actions={ActionType.EXPORT},
                conditions={"max_records": 1000, "formats": ["csv"]}
            ),
            Permission(
                id="export_advanced",
                name="Advanced Export",
                description="Export data in all formats without limits",
                resource_type=ResourceType.EXPORT,
                actions={ActionType.EXPORT}
            ),
            
            # Analytics permissions
            Permission(
                id="analytics_read",
                name="View Analytics",
                description="View analytics dashboards",
                resource_type=ResourceType.ANALYTICS,
                actions={ActionType.READ}
            ),
            Permission(
                id="analytics_create",
                name="Create Analytics",
                description="Create custom analytics",
                resource_type=ResourceType.ANALYTICS,
                actions={ActionType.CREATE}
            ),
            
            # ML permissions
            Permission(
                id="ml_execute",
                name="Execute ML",
                description="Use machine learning features",
                resource_type=ResourceType.ML,
                actions={ActionType.EXECUTE}
            ),
            Permission(
                id="ml_configure",
                name="Configure ML",
                description="Configure ML models and parameters",
                resource_type=ResourceType.ML,
                actions={ActionType.CONFIGURE}
            ),
            
            # System permissions
            Permission(
                id="system_configure",
                name="System Configuration",
                description="Configure system settings",
                resource_type=ResourceType.SYSTEM,
                actions={ActionType.CONFIGURE, ActionType.UPDATE}
            ),
            Permission(
                id="system_read",
                name="System Read",
                description="View system information",
                resource_type=ResourceType.SYSTEM,
                actions={ActionType.READ}
            ),
            
            # Audit permissions
            Permission(
                id="audit_read",
                name="Read Audit Logs",
                description="View audit logs and reports",
                resource_type=ResourceType.AUDIT,
                actions={ActionType.READ}
            )
        ]
        
        for permission in permissions:
            self.permissions[permission.id] = permission
    
    def _setup_default_roles(self) -> None:
        """Setup default system roles"""
        # Admin role - full access
        admin_permissions = list(self.permissions.values())
        admin_role = Role(
            id="admin",
            name="Administrator",
            description="Full system access",
            permissions=admin_permissions,
            is_system_role=True
        )
        
        # Analyst role - data analysis and ML
        analyst_permission_ids = [
            "data_read", "data_create", "data_update",
            "filter_create", "filter_read", "filter_update", "filter_delete",
            "export_advanced", "analytics_read", "analytics_create",
            "ml_execute", "system_read"
        ]
        analyst_permissions = [self.permissions[pid] for pid in analyst_permission_ids]
        analyst_role = Role(
            id="analyst",
            name="Data Analyst",
            description="Data analysis and ML capabilities",
            permissions=analyst_permissions,
            is_system_role=True
        )
        
        # Viewer role - read-only access
        viewer_permission_ids = [
            "data_read", "filter_read", "export_basic",
            "analytics_read", "system_read"
        ]
        viewer_permissions = [self.permissions[pid] for pid in viewer_permission_ids]
        viewer_role = Role(
            id="viewer",
            name="Viewer",
            description="Read-only access to data and analytics",
            permissions=viewer_permissions,
            is_system_role=True
        )
        
        self.roles = {
            "admin": admin_role,
            "analyst": analyst_role,
            "viewer": viewer_role
        }
    
    def _validate_role_permissions(self) -> None:
        """Validate that all role permissions exist"""
        for role in self.roles.values():
            for permission in role.permissions:
                if permission.id not in self.permissions:
                    self.logger.warning(f"Role {role.name} references unknown permission {permission.id}")
    
    def assign_role_to_user(self, user_id: str, role_id: str, assigned_by: str) -> bool:
        """Assign a role to a user"""
        if role_id not in self.roles:
            self.logger.error(f"Role {role_id} does not exist")
            return False
        
        if user_id not in self.user_roles:
            self.user_roles[user_id] = []
        
        # Check if user already has this role
        for user_role in self.user_roles[user_id]:
            if user_role.role_id == role_id and user_role.is_active:
                self.logger.warning(f"User {user_id} already has role {role_id}")
                return False
        
        user_role = UserRole(
            user_id=user_id,
            role_id=role_id,
            assigned_by=assigned_by
        )
        
        self.user_roles[user_id].append(user_role)
        self.logger.info(f"Assigned role {role_id} to user {user_id}")
        return True
    
    def revoke_role_from_user(self, user_id: str, role_id: str) -> bool:
        """Revoke a role from a user"""
        if user_id not in self.user_roles:
            return False
        
        for user_role in self.user_roles[user_id]:
            if user_role.role_id == role_id and user_role.is_active:
                user_role.is_active = False
                self.logger.info(f"Revoked role {role_id} from user {user_id}")
                return True
        
        return False
    
    def get_user_roles(self, user_id: str) -> List[Role]:
        """Get all active roles for a user"""
        if user_id not in self.user_roles:
            return []
        
        active_roles = []
        for user_role in self.user_roles[user_id]:
            if user_role.is_active:
                if user_role.expires_at is None or user_role.expires_at > datetime.now():
                    role = self.roles.get(user_role.role_id)
                    if role:
                        active_roles.append(role)
        
        return active_roles
    
    def get_user_permissions(self, user_id: str) -> List[Permission]:
        """Get all permissions for a user based on their roles"""
        user_roles = self.get_user_roles(user_id)
        permissions = set()
        
        for role in user_roles:
            for permission in role.permissions:
                permissions.add(permission)
        
        return list(permissions)
    
    def check_access(self, request: AccessRequest) -> AccessResult:
        """Check if user has access to perform an action"""
        user_permissions = self.get_user_permissions(request.user.id)
        
        # Log access attempt
        self._log_access_attempt(request)
        
        # Check if user has required permission
        for permission in user_permissions:
            if (permission.resource_type == request.resource_type and 
                request.action in permission.actions):
                
                # Check conditions if any
                if permission.conditions:
                    conditions_met = self._check_conditions(permission.conditions, request)
                    if not conditions_met:
                        return AccessResult(
                            granted=False,
                            reason=f"Conditions not met for permission {permission.name}",
                            permissions_used=[permission.id],
                            conditions_met=False
                        )
                
                return AccessResult(
                    granted=True,
                    reason=f"Access granted via permission {permission.name}",
                    permissions_used=[permission.id]
                )
        
        return AccessResult(
            granted=False,
            reason=f"No permission found for {request.action.value} on {request.resource_type.value}"
        )
    
    def _check_conditions(self, conditions: Dict[str, Any], request: AccessRequest) -> bool:
        """Check if permission conditions are met"""
        # Example condition checks
        if "max_records" in conditions:
            requested_records = request.context.get("record_count", 0)
            if requested_records > conditions["max_records"]:
                return False
        
        if "formats" in conditions:
            requested_format = request.context.get("format")
            if requested_format and requested_format not in conditions["formats"]:
                return False
        
        if "time_restriction" in conditions:
            # Check time-based restrictions
            current_hour = datetime.now().hour
            allowed_hours = conditions["time_restriction"].get("allowed_hours", [])
            if allowed_hours and current_hour not in allowed_hours:
                return False
        
        return True
    
    def _log_access_attempt(self, request: AccessRequest) -> None:
        """Log access attempt for audit purposes"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": request.user.id,
            "username": request.user.username,
            "resource_type": request.resource_type.value,
            "action": request.action.value,
            "resource_id": request.resource_id,
            "context": request.context
        }
        
        self.access_log.append(log_entry)
        
        # Keep only recent logs (last 1000 entries)
        if len(self.access_log) > 1000:
            self.access_log = self.access_log[-1000:]
    
    def create_role(self, role_id: str, name: str, description: str, permission_ids: List[str]) -> bool:
        """Create a new role"""
        if role_id in self.roles:
            self.logger.error(f"Role {role_id} already exists")
            return False
        
        # Validate permissions
        permissions = []
        for permission_id in permission_ids:
            if permission_id in self.permissions:
                permissions.append(self.permissions[permission_id])
            else:
                self.logger.error(f"Permission {permission_id} does not exist")
                return False
        
        role = Role(
            id=role_id,
            name=name,
            description=description,
            permissions=permissions
        )
        
        self.roles[role_id] = role
        self.logger.info(f"Created role {role_id}")
        return True
    
    def update_role(self, role_id: str, permission_ids: List[str]) -> bool:
        """Update role permissions"""
        if role_id not in self.roles:
            self.logger.error(f"Role {role_id} does not exist")
            return False
        
        role = self.roles[role_id]
        if role.is_system_role:
            self.logger.error(f"Cannot modify system role {role_id}")
            return False
        
        # Validate permissions
        permissions = []
        for permission_id in permission_ids:
            if permission_id in self.permissions:
                permissions.append(self.permissions[permission_id])
            else:
                self.logger.error(f"Permission {permission_id} does not exist")
                return False
        
        role.permissions = permissions
        role.updated_at = datetime.now()
        self.logger.info(f"Updated role {role_id}")
        return True
    
    def delete_role(self, role_id: str) -> bool:
        """Delete a role"""
        if role_id not in self.roles:
            return False
        
        role = self.roles[role_id]
        if role.is_system_role:
            self.logger.error(f"Cannot delete system role {role_id}")
            return False
        
        # Revoke role from all users
        for user_id, user_roles in self.user_roles.items():
            for user_role in user_roles:
                if user_role.role_id == role_id:
                    user_role.is_active = False
        
        del self.roles[role_id]
        self.logger.info(f"Deleted role {role_id}")
        return True
    
    def get_access_log(self, user_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get access log entries"""
        logs = self.access_log
        
        if user_id:
            logs = [log for log in logs if log["user_id"] == user_id]
        
        return logs[-limit:] if limit else logs
    
    def export_roles_and_permissions(self) -> Dict[str, Any]:
        """Export roles and permissions configuration"""
        return {
            "permissions": {
                pid: {
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "resource_type": p.resource_type.value,
                    "actions": [a.value for a in p.actions],
                    "conditions": p.conditions
                }
                for pid, p in self.permissions.items()
            },
            "roles": {
                rid: {
                    "id": r.id,
                    "name": r.name,
                    "description": r.description,
                    "permissions": [p.id for p in r.permissions],
                    "is_system_role": r.is_system_role
                }
                for rid, r in self.roles.items()
            }
        }