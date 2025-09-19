"""
Base abstract classes for component architecture
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from .interfaces import BaseComponent, BaseService


class ComponentBase(BaseComponent):
    """Base class for all application components"""
    
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.created_at = datetime.now()
        self.last_activity = None
    
    def initialize(self) -> None:
        """Initialize the component"""
        if self._initialized:
            return
        
        self.logger.info(f"Initializing component: {self.name}")
        self._do_initialize()
        self._initialized = True
        self.logger.info(f"Component initialized: {self.name}")
    
    def cleanup(self) -> None:
        """Cleanup component resources"""
        if not self._initialized:
            return
        
        self.logger.info(f"Cleaning up component: {self.name}")
        self._do_cleanup()
        self._initialized = False
        self.logger.info(f"Component cleaned up: {self.name}")
    
    @abstractmethod
    def _do_initialize(self) -> None:
        """Perform actual initialization"""
        pass
    
    @abstractmethod
    def _do_cleanup(self) -> None:
        """Perform actual cleanup"""
        pass
    
    def update_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def get_status(self) -> Dict[str, Any]:
        """Get component status"""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat() if self.last_activity else None
        }


class ServiceBase(BaseService):
    """Base class for all application services"""
    
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._running = False
        self.started_at = None
        self.metrics = {
            "requests_processed": 0,
            "errors_count": 0,
            "last_error": None
        }
    
    def start(self) -> None:
        """Start the service"""
        if self._running:
            return
        
        self.logger.info(f"Starting service: {self.name}")
        self._do_start()
        self._running = True
        self.started_at = datetime.now()
        self.logger.info(f"Service started: {self.name}")
    
    def stop(self) -> None:
        """Stop the service"""
        if not self._running:
            return
        
        self.logger.info(f"Stopping service: {self.name}")
        self._do_stop()
        self._running = False
        self.logger.info(f"Service stopped: {self.name}")
    
    @abstractmethod
    def _do_start(self) -> None:
        """Perform actual service start"""
        pass
    
    @abstractmethod
    def _do_stop(self) -> None:
        """Perform actual service stop"""
        pass
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            "name": self.name,
            "status": "healthy" if self._running else "stopped",
            "running": self._running,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "metrics": self.metrics.copy()
        }
    
    def is_running(self) -> bool:
        """Check if service is running"""
        return self._running
    
    def increment_requests(self) -> None:
        """Increment request counter"""
        self.metrics["requests_processed"] += 1
    
    def increment_errors(self, error: Optional[str] = None) -> None:
        """Increment error counter"""
        self.metrics["errors_count"] += 1
        if error:
            self.metrics["last_error"] = error


class ProcessorBase(ComponentBase):
    """Base class for data processors"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.processing_stats = {
            "total_processed": 0,
            "total_errors": 0,
            "average_processing_time": 0.0,
            "last_processed": None
        }
    
    def _record_processing(self, processing_time: float, success: bool = True) -> None:
        """Record processing statistics"""
        self.processing_stats["total_processed"] += 1
        if not success:
            self.processing_stats["total_errors"] += 1
        
        # Update average processing time
        current_avg = self.processing_stats["average_processing_time"]
        total = self.processing_stats["total_processed"]
        self.processing_stats["average_processing_time"] = (
            (current_avg * (total - 1) + processing_time) / total
        )
        
        self.processing_stats["last_processed"] = datetime.now().isoformat()
        self.update_activity()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.processing_stats.copy()


class ManagerBase(ServiceBase):
    """Base class for manager services"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.managed_resources = {}
        self.resource_stats = {
            "total_resources": 0,
            "active_resources": 0,
            "failed_resources": 0
        }
    
    def add_resource(self, resource_id: str, resource: Any) -> None:
        """Add a managed resource"""
        self.managed_resources[resource_id] = {
            "resource": resource,
            "created_at": datetime.now(),
            "status": "active"
        }
        self.resource_stats["total_resources"] += 1
        self.resource_stats["active_resources"] += 1
    
    def remove_resource(self, resource_id: str) -> None:
        """Remove a managed resource"""
        if resource_id in self.managed_resources:
            del self.managed_resources[resource_id]
            self.resource_stats["active_resources"] -= 1
    
    def mark_resource_failed(self, resource_id: str) -> None:
        """Mark a resource as failed"""
        if resource_id in self.managed_resources:
            self.managed_resources[resource_id]["status"] = "failed"
            self.resource_stats["active_resources"] -= 1
            self.resource_stats["failed_resources"] += 1
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource statistics"""
        return self.resource_stats.copy()
    
    def list_resources(self) -> Dict[str, Any]:
        """List all managed resources"""
        return {
            resource_id: {
                "created_at": info["created_at"].isoformat(),
                "status": info["status"]
            }
            for resource_id, info in self.managed_resources.items()
        }