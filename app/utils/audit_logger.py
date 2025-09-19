"""
Comprehensive audit logging system
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import threading
from pathlib import Path

from ..core.interfaces import User, IAuditLogger
from ..core.base_classes import ComponentBase
from config.settings import get_config


class AuditEventType(Enum):
    """Types of audit events"""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_ACTION = "user_action"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    FILTER_OPERATION = "filter_operation"
    EXPORT_OPERATION = "export_operation"
    SYSTEM_CONFIGURATION = "system_configuration"
    SECURITY_EVENT = "security_event"
    ERROR_EVENT = "error_event"


class AuditSeverity(Enum):
    """Audit event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event model"""
    id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    username: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    action: str
    details: Dict[str, Any]
    severity: AuditSeverity = AuditSeverity.LOW
    success: bool = True
    error_message: Optional[str] = None
    checksum: Optional[str] = field(default=None, init=False)
    
    def __post_init__(self):
        """Calculate checksum for integrity"""
        self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum for event integrity"""
        event_data = {
            "id": self.id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "action": self.action,
            "details": self.details
        }
        
        event_json = json.dumps(event_data, sort_keys=True)
        return hashlib.sha256(event_json.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify event integrity using checksum"""
        return self.checksum == self._calculate_checksum()


@dataclass
class AuditQuery:
    """Audit query parameters"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    user_id: Optional[str] = None
    event_types: Optional[List[AuditEventType]] = None
    severity: Optional[AuditSeverity] = None
    resource_type: Optional[str] = None
    action: Optional[str] = None
    success: Optional[bool] = None
    limit: int = 1000
    offset: int = 0


@dataclass
class AuditReport:
    """Audit report model"""
    report_id: str
    generated_at: datetime
    query: AuditQuery
    total_events: int
    events: List[AuditEvent]
    summary: Dict[str, Any]
    generated_by: str


class AuditLogger(ComponentBase, IAuditLogger):
    """Comprehensive audit logging system"""
    
    def __init__(self):
        super().__init__("AuditLogger")
        self.config = get_config()
        self.events: List[AuditEvent] = []
        self.event_counter = 0
        self.lock = threading.Lock()
        
        # Setup file logging
        self.log_file_path = Path("logs/audit.log")
        self.log_file_path.parent.mkdir(exist_ok=True)
        
        # Setup structured logger
        self.audit_logger = logging.getLogger("audit")
        self.audit_logger.setLevel(logging.INFO)
        
        # File handler for audit logs
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.INFO)
        
        # JSON formatter
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}'
        )
        file_handler.setFormatter(formatter)
        self.audit_logger.addHandler(file_handler)
    
    def _do_initialize(self) -> None:
        """Initialize audit logger"""
        self.logger.info("Initializing audit logging system")
        self._log_system_event("audit_system_started", {"component": "AuditLogger"})
    
    def _do_cleanup(self) -> None:
        """Cleanup audit logger"""
        self._log_system_event("audit_system_stopped", {"component": "AuditLogger"})
        
        # Close file handlers
        for handler in self.audit_logger.handlers:
            handler.close()
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        with self.lock:
            self.event_counter += 1
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            return f"AE_{timestamp}_{self.event_counter:06d}"
    
    def _create_audit_event(
        self,
        event_type: AuditEventType,
        action: str,
        user: Optional[User] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: AuditSeverity = AuditSeverity.LOW,
        success: bool = True,
        error_message: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> AuditEvent:
        """Create audit event"""
        event = AuditEvent(
            id=self._generate_event_id(),
            event_type=event_type,
            timestamp=datetime.now(),
            user_id=user.id if user else None,
            username=user.username if user else None,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            details=details or {},
            severity=severity,
            success=success,
            error_message=error_message
        )
        
        return event
    
    def _store_event(self, event: AuditEvent) -> None:
        """Store audit event"""
        with self.lock:
            self.events.append(event)
            
            # Keep only recent events in memory (last 10000)
            if len(self.events) > 10000:
                self.events = self.events[-10000:]
        
        # Log to file
        event_json = json.dumps(asdict(event), default=str)
        self.audit_logger.info(event_json)
    
    def log_user_action(self, user: User, action: str, details: Dict[str, Any]) -> None:
        """Log user action"""
        event = self._create_audit_event(
            event_type=AuditEventType.USER_ACTION,
            action=action,
            user=user,
            details=details,
            severity=AuditSeverity.LOW
        )
        
        self._store_event(event)
    
    def log_data_access(self, user: User, data_info: Dict[str, Any]) -> None:
        """Log data access"""
        event = self._create_audit_event(
            event_type=AuditEventType.DATA_ACCESS,
            action="data_access",
            user=user,
            details=data_info,
            severity=AuditSeverity.MEDIUM,
            resource_type="data",
            resource_id=data_info.get("dataset_id")
        )
        
        self._store_event(event)
    
    def log_data_modification(
        self, 
        user: User, 
        operation: str, 
        data_info: Dict[str, Any],
        changes: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log data modification"""
        details = data_info.copy()
        if changes:
            details["changes"] = changes
        
        event = self._create_audit_event(
            event_type=AuditEventType.DATA_MODIFICATION,
            action=operation,
            user=user,
            details=details,
            severity=AuditSeverity.HIGH,
            resource_type="data",
            resource_id=data_info.get("dataset_id")
        )
        
        self._store_event(event)
    
    def log_filter_operation(
        self, 
        user: User, 
        filters: Dict[str, Any], 
        results: Dict[str, Any]
    ) -> None:
        """Log filter operation"""
        details = {
            "filters_applied": filters,
            "results_summary": results,
            "records_before": results.get("records_before", 0),
            "records_after": results.get("records_after", 0)
        }
        
        event = self._create_audit_event(
            event_type=AuditEventType.FILTER_OPERATION,
            action="apply_filters",
            user=user,
            details=details,
            severity=AuditSeverity.LOW
        )
        
        self._store_event(event)
    
    def log_export_operation(
        self, 
        user: User, 
        export_info: Dict[str, Any]
    ) -> None:
        """Log export operation"""
        event = self._create_audit_event(
            event_type=AuditEventType.EXPORT_OPERATION,
            action="export_data",
            user=user,
            details=export_info,
            severity=AuditSeverity.MEDIUM,
            resource_type="export"
        )
        
        self._store_event(event)
    
    def log_login(
        self, 
        user: User, 
        success: bool, 
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> None:
        """Log user login attempt"""
        event = self._create_audit_event(
            event_type=AuditEventType.USER_LOGIN,
            action="login",
            user=user,
            details={"login_method": "local"},
            severity=AuditSeverity.MEDIUM if success else AuditSeverity.HIGH,
            success=success,
            error_message=error_message,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self._store_event(event)
    
    def log_logout(self, user: User) -> None:
        """Log user logout"""
        event = self._create_audit_event(
            event_type=AuditEventType.USER_LOGOUT,
            action="logout",
            user=user,
            details={},
            severity=AuditSeverity.LOW
        )
        
        self._store_event(event)
    
    def log_security_event(
        self, 
        event_description: str, 
        details: Dict[str, Any],
        user: Optional[User] = None,
        severity: AuditSeverity = AuditSeverity.HIGH
    ) -> None:
        """Log security event"""
        event = self._create_audit_event(
            event_type=AuditEventType.SECURITY_EVENT,
            action=event_description,
            user=user,
            details=details,
            severity=severity
        )
        
        self._store_event(event)
    
    def log_system_configuration(
        self, 
        user: User, 
        configuration_change: str, 
        details: Dict[str, Any]
    ) -> None:
        """Log system configuration change"""
        event = self._create_audit_event(
            event_type=AuditEventType.SYSTEM_CONFIGURATION,
            action=configuration_change,
            user=user,
            details=details,
            severity=AuditSeverity.HIGH,
            resource_type="system"
        )
        
        self._store_event(event)
    
    def log_error(
        self, 
        error_description: str, 
        details: Dict[str, Any],
        user: Optional[User] = None,
        severity: AuditSeverity = AuditSeverity.MEDIUM
    ) -> None:
        """Log error event"""
        event = self._create_audit_event(
            event_type=AuditEventType.ERROR_EVENT,
            action=error_description,
            user=user,
            details=details,
            severity=severity,
            success=False
        )
        
        self._store_event(event)
    
    def _log_system_event(self, action: str, details: Dict[str, Any]) -> None:
        """Log system event (internal)"""
        event = self._create_audit_event(
            event_type=AuditEventType.SYSTEM_CONFIGURATION,
            action=action,
            details=details,
            severity=AuditSeverity.LOW
        )
        
        self._store_event(event)
    
    def query_events(self, query: AuditQuery) -> List[AuditEvent]:
        """Query audit events"""
        with self.lock:
            filtered_events = self.events.copy()
        
        # Apply filters
        if query.start_date:
            filtered_events = [e for e in filtered_events if e.timestamp >= query.start_date]
        
        if query.end_date:
            filtered_events = [e for e in filtered_events if e.timestamp <= query.end_date]
        
        if query.user_id:
            filtered_events = [e for e in filtered_events if e.user_id == query.user_id]
        
        if query.event_types:
            filtered_events = [e for e in filtered_events if e.event_type in query.event_types]
        
        if query.severity:
            filtered_events = [e for e in filtered_events if e.severity == query.severity]
        
        if query.resource_type:
            filtered_events = [e for e in filtered_events if e.resource_type == query.resource_type]
        
        if query.action:
            filtered_events = [e for e in filtered_events if query.action.lower() in e.action.lower()]
        
        if query.success is not None:
            filtered_events = [e for e in filtered_events if e.success == query.success]
        
        # Sort by timestamp (newest first)
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply pagination
        start_idx = query.offset
        end_idx = start_idx + query.limit
        
        return filtered_events[start_idx:end_idx]
    
    def generate_audit_report(self, start_date: datetime, end_date: datetime) -> AuditReport:
        """Generate comprehensive audit report"""
        query = AuditQuery(
            start_date=start_date,
            end_date=end_date,
            limit=10000  # Get all events for report
        )
        
        events = self.query_events(query)
        
        # Generate summary statistics
        summary = self._generate_report_summary(events)
        
        report = AuditReport(
            report_id=self._generate_event_id().replace("AE_", "AR_"),
            generated_at=datetime.now(),
            query=query,
            total_events=len(events),
            events=events,
            summary=summary,
            generated_by="system"
        )
        
        return report
    
    def _generate_report_summary(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Generate summary statistics for events"""
        if not events:
            return {}
        
        # Event type distribution
        event_types = {}
        for event in events:
            event_type = event.event_type.value
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        # Severity distribution
        severities = {}
        for event in events:
            severity = event.severity.value
            severities[severity] = severities.get(severity, 0) + 1
        
        # User activity
        user_activity = {}
        for event in events:
            if event.user_id:
                user_activity[event.user_id] = user_activity.get(event.user_id, 0) + 1
        
        # Success/failure rates
        successful_events = sum(1 for e in events if e.success)
        failed_events = len(events) - successful_events
        
        # Time range
        timestamps = [e.timestamp for e in events]
        time_range = {
            "start": min(timestamps).isoformat(),
            "end": max(timestamps).isoformat()
        }
        
        return {
            "total_events": len(events),
            "time_range": time_range,
            "event_types": event_types,
            "severities": severities,
            "user_activity": user_activity,
            "success_rate": {
                "successful": successful_events,
                "failed": failed_events,
                "success_percentage": (successful_events / len(events)) * 100 if events else 0
            },
            "top_users": sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10],
            "security_events": sum(1 for e in events if e.event_type == AuditEventType.SECURITY_EVENT),
            "data_modifications": sum(1 for e in events if e.event_type == AuditEventType.DATA_MODIFICATION)
        }
    
    def export_audit_log(
        self, 
        query: AuditQuery, 
        format: str = "json"
    ) -> str:
        """Export audit log in specified format"""
        events = self.query_events(query)
        
        if format.lower() == "json":
            return json.dumps([asdict(event) for event in events], default=str, indent=2)
        elif format.lower() == "csv":
            import csv
            import io
            
            output = io.StringIO()
            if events:
                fieldnames = asdict(events[0]).keys()
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                
                for event in events:
                    row = asdict(event)
                    # Convert complex objects to strings
                    for key, value in row.items():
                        if isinstance(value, (dict, list)):
                            row[key] = json.dumps(value)
                        elif hasattr(value, 'value'):  # Enum
                            row[key] = value.value
                    writer.writerow(row)
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def verify_log_integrity(self) -> Dict[str, Any]:
        """Verify integrity of audit logs"""
        with self.lock:
            events = self.events.copy()
        
        total_events = len(events)
        corrupted_events = []
        
        for event in events:
            if not event.verify_integrity():
                corrupted_events.append(event.id)
        
        return {
            "total_events": total_events,
            "corrupted_events": len(corrupted_events),
            "corrupted_event_ids": corrupted_events,
            "integrity_percentage": ((total_events - len(corrupted_events)) / total_events * 100) if total_events > 0 else 100
        }
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit system statistics"""
        with self.lock:
            events = self.events.copy()
        
        if not events:
            return {"total_events": 0}
        
        recent_events = [e for e in events if e.timestamp > datetime.now() - timedelta(hours=24)]
        
        return {
            "total_events": len(events),
            "events_last_24h": len(recent_events),
            "oldest_event": min(e.timestamp for e in events).isoformat(),
            "newest_event": max(e.timestamp for e in events).isoformat(),
            "memory_usage_events": len(events),
            "log_file_path": str(self.log_file_path)
        }