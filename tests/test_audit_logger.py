"""
Unit tests for audit logging system
"""

import pytest
from datetime import datetime, timedelta
import json
import tempfile
from pathlib import Path

from app.utils.audit_logger import (
    AuditLogger, AuditEventType, AuditSeverity, 
    AuditEvent, AuditQuery
)
from app.core.interfaces import User


class TestAuditLogger:
    """Test audit logging system"""
    
    def setup_method(self):
        """Setup test method"""
        self.audit_logger = AuditLogger()
        self.audit_logger.initialize()
        
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
        self.audit_logger.cleanup()
    
    def test_audit_event_creation(self):
        """Test audit event creation and integrity"""
        event = AuditEvent(
            id="TEST_001",
            event_type=AuditEventType.USER_ACTION,
            timestamp=datetime.now(),
            user_id="test_user",
            username="testuser",
            session_id=None,
            ip_address=None,
            user_agent=None,
            resource_type="data",
            resource_id="dataset_1",
            action="view_data",
            details={"table": "properties"},
            severity=AuditSeverity.LOW
        )
        
        assert event.id == "TEST_001"
        assert event.event_type == AuditEventType.USER_ACTION
        assert event.checksum is not None
        assert event.verify_integrity() is True
    
    def test_log_user_action(self):
        """Test logging user action"""
        details = {"action": "view_dashboard", "page": "analytics"}
        
        self.audit_logger.log_user_action(self.test_user, "view_dashboard", details)
        
        events = self.audit_logger.events
        assert len(events) == 1
        
        event = events[0]
        assert event.event_type == AuditEventType.USER_ACTION
        assert event.user_id == "test_user"
        assert event.action == "view_dashboard"
        assert event.details == details
    
    def test_log_data_access(self):
        """Test logging data access"""
        data_info = {
            "dataset_id": "properties_2024",
            "table_name": "properties",
            "record_count": 1500,
            "columns_accessed": ["address", "price", "sqft"]
        }
        
        self.audit_logger.log_data_access(self.test_user, data_info)
        
        events = self.audit_logger.events
        assert len(events) == 1
        
        event = events[0]
        assert event.event_type == AuditEventType.DATA_ACCESS
        assert event.resource_type == "data"
        assert event.resource_id == "properties_2024"
        assert event.severity == AuditSeverity.MEDIUM
    
    def test_log_data_modification(self):
        """Test logging data modification"""
        data_info = {"dataset_id": "properties_2024", "table_name": "properties"}
        changes = {
            "records_updated": 5,
            "columns_modified": ["price", "status"],
            "operation": "bulk_update"
        }
        
        self.audit_logger.log_data_modification(
            self.test_user, "update_records", data_info, changes
        )
        
        events = self.audit_logger.events
        assert len(events) == 1
        
        event = events[0]
        assert event.event_type == AuditEventType.DATA_MODIFICATION
        assert event.severity == AuditSeverity.HIGH
        assert "changes" in event.details
        assert event.details["changes"] == changes
    
    def test_log_filter_operation(self):
        """Test logging filter operation"""
        filters = {
            "price_min": 100000,
            "price_max": 500000,
            "property_type": "residential",
            "city": "Seattle"
        }
        
        results = {
            "records_before": 10000,
            "records_after": 1250,
            "processing_time": 2.5,
            "filters_applied": 4
        }
        
        self.audit_logger.log_filter_operation(self.test_user, filters, results)
        
        events = self.audit_logger.events
        assert len(events) == 1
        
        event = events[0]
        assert event.event_type == AuditEventType.FILTER_OPERATION
        assert event.action == "apply_filters"
        assert event.details["filters_applied"] == filters
        assert event.details["results_summary"] == results
    
    def test_log_export_operation(self):
        """Test logging export operation"""
        export_info = {
            "format": "csv",
            "record_count": 500,
            "file_size": "2.5MB",
            "columns_exported": ["address", "price", "sqft", "year_built"],
            "export_time": 3.2
        }
        
        self.audit_logger.log_export_operation(self.test_user, export_info)
        
        events = self.audit_logger.events
        assert len(events) == 1
        
        event = events[0]
        assert event.event_type == AuditEventType.EXPORT_OPERATION
        assert event.action == "export_data"
        assert event.resource_type == "export"
        assert event.severity == AuditSeverity.MEDIUM
    
    def test_log_login_success(self):
        """Test logging successful login"""
        self.audit_logger.log_login(
            self.test_user, 
            success=True, 
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0..."
        )
        
        events = self.audit_logger.events
        assert len(events) == 1
        
        event = events[0]
        assert event.event_type == AuditEventType.USER_LOGIN
        assert event.success is True
        assert event.ip_address == "192.168.1.100"
        assert event.severity == AuditSeverity.MEDIUM
    
    def test_log_login_failure(self):
        """Test logging failed login"""
        self.audit_logger.log_login(
            self.test_user,
            success=False,
            error_message="Invalid password",
            ip_address="192.168.1.100"
        )
        
        events = self.audit_logger.events
        assert len(events) == 1
        
        event = events[0]
        assert event.event_type == AuditEventType.USER_LOGIN
        assert event.success is False
        assert event.error_message == "Invalid password"
        assert event.severity == AuditSeverity.HIGH
    
    def test_log_security_event(self):
        """Test logging security event"""
        details = {
            "event": "multiple_failed_logins",
            "attempts": 5,
            "ip_address": "192.168.1.100",
            "time_window": "5 minutes"
        }
        
        self.audit_logger.log_security_event(
            "account_lockout_triggered",
            details,
            self.test_user,
            AuditSeverity.CRITICAL
        )
        
        events = self.audit_logger.events
        assert len(events) == 1
        
        event = events[0]
        assert event.event_type == AuditEventType.SECURITY_EVENT
        assert event.severity == AuditSeverity.CRITICAL
        assert event.details == details
    
    def test_query_events_by_date_range(self):
        """Test querying events by date range"""
        # Create events with different timestamps
        now = datetime.now()
        
        # Event from yesterday
        self.audit_logger.log_user_action(self.test_user, "action1", {})
        self.audit_logger.events[-1].timestamp = now - timedelta(days=1)
        
        # Event from today
        self.audit_logger.log_user_action(self.test_user, "action2", {})
        
        # Event from tomorrow (future)
        self.audit_logger.log_user_action(self.test_user, "action3", {})
        self.audit_logger.events[-1].timestamp = now + timedelta(days=1)
        
        # Query for today's events
        query = AuditQuery(
            start_date=now.replace(hour=0, minute=0, second=0, microsecond=0),
            end_date=now.replace(hour=23, minute=59, second=59, microsecond=999999)
        )
        
        results = self.audit_logger.query_events(query)
        assert len(results) == 1
        assert results[0].action == "action2"
    
    def test_query_events_by_user(self):
        """Test querying events by user"""
        # Create another user
        other_user = User(
            id="other_user",
            username="otheruser",
            email="other@example.com",
            role="viewer",
            permissions=[],
            is_active=True
        )
        
        # Create events for different users
        self.audit_logger.log_user_action(self.test_user, "action1", {})
        self.audit_logger.log_user_action(other_user, "action2", {})
        self.audit_logger.log_user_action(self.test_user, "action3", {})
        
        # Query for test_user events
        query = AuditQuery(user_id="test_user")
        results = self.audit_logger.query_events(query)
        
        assert len(results) == 2
        assert all(event.user_id == "test_user" for event in results)
    
    def test_query_events_by_event_type(self):
        """Test querying events by event type"""
        # Create different types of events
        self.audit_logger.log_user_action(self.test_user, "action1", {})
        self.audit_logger.log_data_access(self.test_user, {"dataset_id": "test"})
        self.audit_logger.log_export_operation(self.test_user, {"format": "csv"})
        
        # Query for data access events only
        query = AuditQuery(event_types=[AuditEventType.DATA_ACCESS])
        results = self.audit_logger.query_events(query)
        
        assert len(results) == 1
        assert results[0].event_type == AuditEventType.DATA_ACCESS
    
    def test_generate_audit_report(self):
        """Test generating audit report"""
        # Create some test events
        self.audit_logger.log_user_action(self.test_user, "action1", {})
        self.audit_logger.log_data_access(self.test_user, {"dataset_id": "test"})
        self.audit_logger.log_security_event("test_event", {}, self.test_user)
        
        # Generate report
        start_date = datetime.now() - timedelta(hours=1)
        end_date = datetime.now() + timedelta(hours=1)
        
        report = self.audit_logger.generate_audit_report(start_date, end_date)
        
        assert report.total_events == 3
        assert len(report.events) == 3
        assert "total_events" in report.summary
        assert "event_types" in report.summary
        assert "severities" in report.summary
        assert "user_activity" in report.summary
        assert report.summary["security_events"] == 1
    
    def test_export_audit_log_json(self):
        """Test exporting audit log as JSON"""
        self.audit_logger.log_user_action(self.test_user, "test_action", {"key": "value"})
        
        query = AuditQuery(limit=10)
        json_export = self.audit_logger.export_audit_log(query, "json")
        
        # Parse JSON to verify it's valid
        data = json.loads(json_export)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["action"] == "test_action"
    
    def test_export_audit_log_csv(self):
        """Test exporting audit log as CSV"""
        self.audit_logger.log_user_action(self.test_user, "test_action", {"key": "value"})
        
        query = AuditQuery(limit=10)
        csv_export = self.audit_logger.export_audit_log(query, "csv")
        
        # Verify CSV format
        lines = csv_export.strip().split('\n')
        assert len(lines) >= 2  # Header + at least one data row
        assert "action" in lines[0]  # Header should contain action column
    
    def test_verify_log_integrity(self):
        """Test log integrity verification"""
        # Create some events
        self.audit_logger.log_user_action(self.test_user, "action1", {})
        self.audit_logger.log_user_action(self.test_user, "action2", {})
        
        # Verify integrity
        integrity_report = self.audit_logger.verify_log_integrity()
        
        assert integrity_report["total_events"] == 2
        assert integrity_report["corrupted_events"] == 0
        assert integrity_report["integrity_percentage"] == 100.0
    
    def test_get_audit_statistics(self):
        """Test getting audit statistics"""
        # Create some events
        self.audit_logger.log_user_action(self.test_user, "action1", {})
        self.audit_logger.log_data_access(self.test_user, {"dataset_id": "test"})
        
        stats = self.audit_logger.get_audit_statistics()
        
        assert stats["total_events"] == 2
        assert stats["events_last_24h"] == 2
        assert "oldest_event" in stats
        assert "newest_event" in stats
        assert "log_file_path" in stats
    
    def test_event_id_generation(self):
        """Test unique event ID generation"""
        ids = set()
        
        # Generate multiple events and check ID uniqueness
        for i in range(10):
            self.audit_logger.log_user_action(self.test_user, f"action_{i}", {})
        
        for event in self.audit_logger.events:
            assert event.id not in ids, f"Duplicate ID found: {event.id}"
            ids.add(event.id)
            assert event.id.startswith("AE_")
    
    def test_pagination_in_query(self):
        """Test pagination in event queries"""
        # Create multiple events
        for i in range(25):
            self.audit_logger.log_user_action(self.test_user, f"action_{i}", {})
        
        # Test pagination
        query = AuditQuery(limit=10, offset=0)
        page1 = self.audit_logger.query_events(query)
        assert len(page1) == 10
        
        query.offset = 10
        page2 = self.audit_logger.query_events(query)
        assert len(page2) == 10
        
        query.offset = 20
        page3 = self.audit_logger.query_events(query)
        assert len(page3) == 5
        
        # Verify no overlap
        page1_ids = {event.id for event in page1}
        page2_ids = {event.id for event in page2}
        assert len(page1_ids.intersection(page2_ids)) == 0