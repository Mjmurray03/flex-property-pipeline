"""
MongoDB Client with Storage Management
Handles connection pooling and storage monitoring for 512MB limit
"""
import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

load_dotenv()

class MongoDBManager:
    """Manages MongoDB connections with storage monitoring"""
    
    def __init__(self):
        self.uri = os.getenv('MONGODB_URI')
        self.database_name = os.getenv('MONGODB_DATABASE', 'flexfilter')
        self.max_storage_mb = int(os.getenv('MAX_STORAGE_MB', 500))
        self.cleanup_threshold_mb = int(os.getenv('CLEANUP_THRESHOLD_MB', 450))
        
        self.client = None
        self.db = None
        self.logger = logging.getLogger(__name__)
        
        self.connect()
        self._create_indexes()
    
    def connect(self):
        """Establish MongoDB connection"""
        try:
            self.client = MongoClient(self.uri)
            self.db = self.client[self.database_name]
            # Test connection
            self.client.admin.command('ping')
            self.logger.info(f"Connected to MongoDB: {self.database_name}")
        except ConnectionFailure as e:
            self.logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def _create_indexes(self):
        """Create essential indexes for performance"""
        indexes = [
            {
                'collection': 'enriched_properties',
                'index': [('flex_score', DESCENDING), ('parcel_id', ASCENDING)]
            },
            {
                'collection': 'enriched_properties',
                'index': [('zoning', ASCENDING), ('acres', ASCENDING)]
            },
            {
                'collection': 'zoning_data',
                'index': [('PCN', ASCENDING)]
            },
            {
                'collection': 'staging_parcels',
                'index': [('parcel_id', ASCENDING)],
                'unique': True
            }
        ]
        
        for idx in indexes:
            try:
                self.db[idx['collection']].create_index(
                    idx['index'],
                    unique=idx.get('unique', False)
                )
                self.logger.info(f"Created index on {idx['collection']}")
            except Exception as e:
                self.logger.warning(f"Index creation failed: {e}")
    
    def check_storage(self) -> Dict[str, float]:
        """Check current database storage usage"""
        stats = self.db.command("dbStats")
        
        storage_mb = stats['dataSize'] / (1024 * 1024)
        index_mb = stats['indexSize'] / (1024 * 1024)
        total_mb = storage_mb + index_mb
        
        return {
            'data_size_mb': round(storage_mb, 2),
            'index_size_mb': round(index_mb, 2),
            'total_size_mb': round(total_mb, 2),
            'available_mb': round(self.max_storage_mb - total_mb, 2),
            'usage_percent': round((total_mb / self.max_storage_mb) * 100, 2)
        }
    
    def cleanup_staging(self, days_old: int = 7):
        """Remove old staging data to free space"""
        cutoff_date = datetime.utcnow() - datetime.timedelta(days=days_old)
        
        result = self.db.staging_parcels.delete_many({
            'created_at': {'$lt': cutoff_date}
        })
        
        self.logger.info(f"Cleaned up {result.deleted_count} old staging records")
        return result.deleted_count
    
    def get_collection_stats(self) -> List[Dict]:
        """Get size statistics for all collections"""
        collections = self.db.list_collection_names()
        stats = []
        
        for coll_name in collections:
            coll_stats = self.db.command("collStats", coll_name)
            stats.append({
                'name': coll_name,
                'count': coll_stats['count'],
                'size_mb': round(coll_stats['size'] / (1024 * 1024), 2),
                'avg_doc_size_kb': round(coll_stats.get('avgObjSize', 0) / 1024, 2)
            })
        
        return sorted(stats, key=lambda x: x['size_mb'], reverse=True)
    
    def optimize_storage(self):
        """Compact collections and rebuild indexes if needed"""
        storage = self.check_storage()
        
        if storage['total_size_mb'] > self.cleanup_threshold_mb:
            self.logger.warning(f"Storage threshold exceeded: {storage['total_size_mb']}MB")
            
            # Clean staging
            self.cleanup_staging()
            
            # Compact collections
            for collection in ['staging_parcels', 'enriched_properties']:
                try:
                    self.db.command('compact', collection)
                    self.logger.info(f"Compacted {collection}")
                except Exception as e:
                    self.logger.error(f"Compact failed for {collection}: {e}")
            
            # Recheck storage
            new_storage = self.check_storage()
            self.logger.info(f"Storage after optimization: {new_storage['total_size_mb']}MB")
            
            return new_storage
        
        return storage
    
    def batch_insert(self, collection: str, documents: List[Dict], 
                    batch_size: int = 100) -> int:
        """Insert documents in batches with storage checking"""
        
        # Check storage before insert
        storage = self.check_storage()
        if storage['available_mb'] < 10:
            self.logger.error("Insufficient storage for insert")
            self.optimize_storage()
            storage = self.check_storage()
            if storage['available_mb'] < 10:
                raise Exception("Cannot insert: storage limit reached")
        
        inserted_count = 0
        collection_obj = self.db[collection]
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Add metadata to each document
            for doc in batch:
                doc['inserted_at'] = datetime.utcnow()
                doc['batch_id'] = f"{datetime.utcnow().isoformat()}_{i}"
            
            try:
                self.logger.info(f"Attempting to insert batch of {len(batch)} documents into {collection}")
                self.logger.debug(f"Sample document keys: {list(batch[0].keys()) if batch else 'No documents'}")
                result = collection_obj.insert_many(batch, ordered=False)
                inserted_count += len(result.inserted_ids)
                self.logger.info(f"Successfully inserted {len(result.inserted_ids)} documents")
            except Exception as e:
                self.logger.error(f"Batch insert failed: {e}")
                self.logger.error(f"Sample document for debugging: {batch[0] if batch else 'No documents'}")
                # Don't raise, continue with other batches
        
        self.logger.info(f"Inserted {inserted_count} documents into {collection}")
        return inserted_count
    
    def find_flex_candidates(self, min_score: int = 5) -> List[Dict]:
        """Query for flex property candidates"""
        
        pipeline = [
            {
                '$match': {
                    'flex_score': {'$gte': min_score},
                    'zoning': {'$in': ['IL', 'IG', 'IP', 'IND', 'M-1', 'MUPD']}
                }
            },
            {
                '$sort': {'flex_score': -1}
            },
            {
                '$limit': 1000  # Limit results to manage memory
            },
            {
                '$project': {
                    'parcel_id': 1,
                    'address': 1,
                    'owner_name': 1,
                    'flex_score': 1,
                    'zoning': 1,
                    'acres': 1,
                    'building_sqft': 1,
                    'indicators': 1
                }
            }
        ]
        
        return list(self.db.enriched_properties.aggregate(pipeline))
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.logger.info("MongoDB connection closed")

# Singleton instance
_db_manager = None

def get_db_manager() -> MongoDBManager:
    """Get or create database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = MongoDBManager()
    return _db_manager
