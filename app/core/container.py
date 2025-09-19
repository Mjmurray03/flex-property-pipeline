"""
Dependency injection container for service management
"""

from typing import Dict, Any, Type, TypeVar, Callable, Optional
from abc import ABC, abstractmethod
import inspect
from functools import wraps

T = TypeVar('T')


class ServiceLifetime:
    """Service lifetime constants"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class ServiceDescriptor:
    """Describes a service registration"""
    
    def __init__(self, service_type: Type, implementation: Type, lifetime: str):
        self.service_type = service_type
        self.implementation = implementation
        self.lifetime = lifetime
        self.instance = None


class DIContainer:
    """Dependency injection container"""
    
    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._scoped_instances: Dict[Type, Any] = {}
    
    def register_singleton(self, service_type: Type[T], implementation: Type[T]) -> 'DIContainer':
        """Register a singleton service"""
        self._services[service_type] = ServiceDescriptor(
            service_type, implementation, ServiceLifetime.SINGLETON
        )
        return self
    
    def register_transient(self, service_type: Type[T], implementation: Type[T]) -> 'DIContainer':
        """Register a transient service"""
        self._services[service_type] = ServiceDescriptor(
            service_type, implementation, ServiceLifetime.TRANSIENT
        )
        return self
    
    def register_scoped(self, service_type: Type[T], implementation: Type[T]) -> 'DIContainer':
        """Register a scoped service"""
        self._services[service_type] = ServiceDescriptor(
            service_type, implementation, ServiceLifetime.SCOPED
        )
        return self
    
    def register_instance(self, service_type: Type[T], instance: T) -> 'DIContainer':
        """Register a specific instance"""
        self._singletons[service_type] = instance
        self._services[service_type] = ServiceDescriptor(
            service_type, type(instance), ServiceLifetime.SINGLETON
        )
        return self
    
    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service instance"""
        if service_type not in self._services:
            raise ValueError(f"Service {service_type.__name__} is not registered")
        
        descriptor = self._services[service_type]
        
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            if service_type in self._singletons:
                return self._singletons[service_type]
            
            instance = self._create_instance(descriptor.implementation)
            self._singletons[service_type] = instance
            return instance
        
        elif descriptor.lifetime == ServiceLifetime.SCOPED:
            if service_type in self._scoped_instances:
                return self._scoped_instances[service_type]
            
            instance = self._create_instance(descriptor.implementation)
            self._scoped_instances[service_type] = instance
            return instance
        
        else:  # TRANSIENT
            return self._create_instance(descriptor.implementation)
    
    def _create_instance(self, implementation_type: Type[T]) -> T:
        """Create an instance with dependency injection"""
        constructor = implementation_type.__init__
        sig = inspect.signature(constructor)
        
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            if param.annotation != inspect.Parameter.empty:
                dependency = self.resolve(param.annotation)
                kwargs[param_name] = dependency
        
        return implementation_type(**kwargs)
    
    def clear_scoped(self):
        """Clear scoped instances"""
        self._scoped_instances.clear()
    
    def is_registered(self, service_type: Type) -> bool:
        """Check if a service is registered"""
        return service_type in self._services


# Global container instance
container = DIContainer()


def inject(service_type: Type[T]) -> Callable:
    """Decorator for dependency injection"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            service = container.resolve(service_type)
            return func(service, *args, **kwargs)
        return wrapper
    return decorator


def get_service(service_type: Type[T]) -> T:
    """Get a service from the container"""
    return container.resolve(service_type)