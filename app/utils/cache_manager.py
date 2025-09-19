"""
Intelligent caching system with data fingerprinting
"""

import hashlib
import json
import pickle
import time
from typing import Any, Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import pandas as pd

from ..core.interfaces import ICacheManager
from ..core.base_classes import ManagerBase
from config.settings import get_config


class CacheStrategy(Enum):
    """Cache strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In First Out


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[int] = None
    size_bytes: int = 0
    fingerprint: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl)
    
    def touch(self) -> None:
        """Update access metadata"""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics"""
    total_entries: int = 0
    total_size_bytes: int = 0
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    expired_count: int = 0
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    average_access_time: float = 0.0
    memory_usage_mb: float = 0.0


class CacheManager(ManagerBase, ICacheManager):
    """Intelligent cache manager with data fingerprinting"""
    
    def __init__(self, max_size_mb: int = 100, default_ttl: int = 3600):
        super().__init__("CacheManager")
        self.config = get_config()
        
        # Cache configuration
        self.max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes
        self.default_ttl = default_ttl
        self.strategy = CacheStrategy.LRU
        
        # Cache storage
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = CacheStats()
        self.access_times: List[float] = []
        
        # Data fingerprinting
        self.fingerprint_cache: Dict[str, str] = {}
    
    def _do_start(self) -> None:
        """Start cache manager"""
        self.logger.info(f"Starting cache manager with {self.max_size_bytes // 1024 // 1024}MB limit")
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self.cleanup_thread.start()
    
    def _do_stop(self) -> None:
        """Stop cache manager"""
        self.logger.info("Stopping cache manager")
        with self.lock:
            self.cache.clear()
            self.fingerprint_cache.clear()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        start_time = time.time()
        
        with self.lock:
            if key not in self.cache:
                self.stats.miss_count += 1
                self._update_access_time(time.time() - start_time)
                return None
            
            entry = self.cache[key]
            
            # Check if expired
            if entry.is_expired():
                del self.cache[key]
                self.stats.expired_count += 1
                self.stats.miss_count += 1
                self._update_access_time(time.time() - start_time)
                return None
            
            # Update access metadata
            entry.touch()
            self.stats.hit_count += 1
            self._update_access_time(time.time() - start_time)
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value"""
        with self.lock:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Use default TTL if not specified
            if ttl is None:
                ttl = self.default_ttl
            
            # Generate fingerprint for data
            fingerprint = self._generate_fingerprint(value)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl=ttl,
                size_bytes=size_bytes,
                fingerprint=fingerprint
            )
            
            # Check if we need to evict entries
            self._ensure_capacity(size_bytes)
            
            # Store entry
            self.cache[key] = entry
            self.fingerprint_cache[fingerprint] = key
            
            # Update statistics
            self.stats.total_entries = len(self.cache)
            self.stats.total_size_bytes = sum(e.size_bytes for e in self.cache.values())
    
    def delete(self, key: str) -> None:
        """Delete cached value"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if entry.fingerprint and entry.fingerprint in self.fingerprint_cache:
                    del self.fingerprint_cache[entry.fingerprint]
                del self.cache[key]
                
                # Update statistics
                self.stats.total_entries = len(self.cache)
                self.stats.total_size_bytes = sum(e.size_bytes for e in self.cache.values())
    
    def clear(self) -> None:
        """Clear all cached values"""
        with self.lock:
            self.cache.clear()
            self.fingerprint_cache.clear()
            self.stats = CacheStats()
    
    def get_by_fingerprint(self, fingerprint: str) -> Optional[Any]:
        """Get cached value by data fingerprint"""
        if fingerprint in self.fingerprint_cache:
            key = self.fingerprint_cache[fingerprint]
            return self.get(key)
        return None
    
    def cache_dataframe(self, df: pd.DataFrame, key_prefix: str = "df", ttl: Optional[int] = None) -> str:
        """Cache DataFrame with automatic key generation"""
        fingerprint = self._generate_dataframe_fingerprint(df)
        cache_key = f"{key_prefix}_{fingerprint}"
        
        # Check if already cached
        if self.get(cache_key) is not None:
            return cache_key
        
        # Cache the DataFrame
        self.set(cache_key, df, ttl)
        return cache_key
    
    def get_cached_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached DataFrame"""
        return self.get(key)
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        invalidated = 0
        
        with self.lock:
            keys_to_delete = [key for key in self.cache.keys() if pattern in key]
            
            for key in keys_to_delete:
                self.delete(key)
                invalidated += 1
        
        return invalidated
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self.lock:
            # Update calculated stats
            total_requests = self.stats.hit_count + self.stats.miss_count
            if total_requests > 0:
                self.stats.hit_rate = self.stats.hit_count / total_requests
                self.stats.miss_rate = self.stats.miss_count / total_requests
            
            if self.access_times:
                self.stats.average_access_time = sum(self.access_times) / len(self.access_times)
            
            self.stats.memory_usage_mb = self.stats.total_size_bytes / 1024 / 1024
            
            return self.stats
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information"""
        with self.lock:
            entries_info = []
            
            for key, entry in self.cache.items():
                entries_info.append({
                    'key': key,
                    'size_bytes': entry.size_bytes,
                    'created_at': entry.created_at.isoformat(),
                    'last_accessed': entry.last_accessed.isoformat(),
                    'access_count': entry.access_count,
                    'ttl': entry.ttl,
                    'is_expired': entry.is_expired()
                })
            
            return {
                'total_entries': len(self.cache),
                'total_size_mb': self.stats.total_size_bytes / 1024 / 1024,
                'max_size_mb': self.max_size_bytes / 1024 / 1024,
                'utilization_percent': (self.stats.total_size_bytes / self.max_size_bytes) * 100,
                'entries': entries_info,
                'stats': self.get_stats().__dict__
            }
    
    def _generate_fingerprint(self, value: Any) -> str:
        """Generate fingerprint for any value"""
        if isinstance(value, pd.DataFrame):
            return self._generate_dataframe_fingerprint(value)
        elif isinstance(value, (dict, list)):
            return hashlib.md5(json.dumps(value, sort_keys=True).encode()).hexdigest()
        else:
            return hashlib.md5(str(value).encode()).hexdigest()
    
    def _generate_dataframe_fingerprint(self, df: pd.DataFrame) -> str:
        """Generate fingerprint for DataFrame"""
        # Create fingerprint based on structure and sample data
        fingerprint_data = {
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'shape': df.shape,
            'sample_hash': hashlib.md5(str(df.head().values).encode()).hexdigest()
        }
        
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes"""
        try:
            if isinstance(value, pd.DataFrame):
                return value.memory_usage(deep=True).sum()
            else:
                return len(pickle.dumps(value))
        except Exception:
            # Fallback estimation
            return len(str(value)) * 2  # Rough estimate
    
    def _ensure_capacity(self, new_size: int) -> None:
        """Ensure cache has capacity for new entry"""
        current_size = sum(e.size_bytes for e in self.cache.values())
        
        while current_size + new_size > self.max_size_bytes and self.cache:
            evicted_key = self._select_eviction_candidate()
            if evicted_key:
                self.delete(evicted_key)
                self.stats.eviction_count += 1
                current_size = sum(e.size_bytes for e in self.cache.values())
            else:
                break
    
    def _select_eviction_candidate(self) -> Optional[str]:
        """Select entry for eviction based on strategy"""
        if not self.cache:
            return None
        
        if self.strategy == CacheStrategy.LRU:
            # Least Recently Used
            return min(self.cache.keys(), 
                      key=lambda k: self.cache[k].last_accessed)
        
        elif self.strategy == CacheStrategy.LFU:
            # Least Frequently Used
            return min(self.cache.keys(), 
                      key=lambda k: self.cache[k].access_count)
        
        elif self.strategy == CacheStrategy.TTL:
            # Shortest TTL remaining
            now = datetime.now()
            candidates = []
            
            for key, entry in self.cache.items():
                if entry.ttl:
                    remaining = (entry.created_at + timedelta(seconds=entry.ttl)) - now
                    candidates.append((key, remaining.total_seconds()))
            
            if candidates:
                return min(candidates, key=lambda x: x[1])[0]
            else:
                # Fallback to LRU
                return min(self.cache.keys(), 
                          key=lambda k: self.cache[k].last_accessed)
        
        elif self.strategy == CacheStrategy.FIFO:
            # First In First Out
            return min(self.cache.keys(), 
                      key=lambda k: self.cache[k].created_at)
        
        return None
    
    def _cleanup_expired(self) -> None:
        """Background thread to cleanup expired entries"""
        while self.is_running():
            try:
                with self.lock:
                    expired_keys = [
                        key for key, entry in self.cache.items()
                        if entry.is_expired()
                    ]
                    
                    for key in expired_keys:
                        self.delete(key)
                        self.stats.expired_count += 1
                
                # Sleep for 60 seconds before next cleanup
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in cache cleanup: {str(e)}")
                time.sleep(60)
    
    def _update_access_time(self, access_time: float) -> None:
        """Update access time statistics"""
        self.access_times.append(access_time)
        
        # Keep only recent access times (last 1000)
        if len(self.access_times) > 1000:
            self.access_times = self.access_times[-1000:]
    
    def set_strategy(self, strategy: CacheStrategy) -> None:
        """Set cache eviction strategy"""
        self.strategy = strategy
        self.logger.info(f"Cache strategy set to {strategy.value}")
    
    def warm_cache(self, data_items: List[Tuple[str, Any, Optional[int]]]) -> None:
        """Warm cache with initial data"""
        self.logger.info(f"Warming cache with {len(data_items)} items")
        
        for key, value, ttl in data_items:
            self.set(key, value, ttl)
    
    def export_cache_metrics(self) -> Dict[str, Any]:
        """Export cache metrics for monitoring"""
        stats = self.get_stats()
        
        return {
            'cache_hit_rate': stats.hit_rate,
            'cache_miss_rate': stats.miss_rate,
            'cache_size_mb': stats.memory_usage_mb,
            'cache_utilization_percent': (stats.total_size_bytes / self.max_size_bytes) * 100,
            'total_entries': stats.total_entries,
            'eviction_count': stats.eviction_count,
            'expired_count': stats.expired_count,
            'average_access_time_ms': stats.average_access_time * 1000
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform cache health check"""
        base_health = super().health_check()
        
        stats = self.get_stats()
        utilization = (stats.total_size_bytes / self.max_size_bytes) * 100
        
        # Determine health status
        if utilization > 90:
            status = "warning"
            message = "Cache utilization is high"
        elif stats.hit_rate < 0.5 and stats.hit_count + stats.miss_count > 100:
            status = "warning"
            message = "Cache hit rate is low"
        else:
            status = "healthy"
            message = "Cache is operating normally"
        
        base_health.update({
            'cache_status': status,
            'cache_message': message,
            'cache_utilization_percent': utilization,
            'cache_hit_rate': stats.hit_rate,
            'cache_entries': stats.total_entries
        })
        
        return base_health