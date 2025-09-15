#!/usr/bin/env python3
"""
Setup New Collections Script
Creates the required collections with proper indexes since database is empty
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, OperationFailure

# Load environment variables
load_dotenv()

def setup_collections():
    """Create new collections with proper indexes"""
    print("Setting up MongoDB Collections")
    print("=" * 40)
    
    # Connect to MongoDB
    uri = os.getenv('MONGODB_URI')
    client = MongoClient(uri)
    db_name = os.getenv('DATABASE_NAME', 'flexfilter-cluster')
    db = client[db_name]
    
    try:
        # Test connection
        client.admin.command('ping')
        print("[SUCCESS] Connected to MongoDB Atlas")
        
        collections_config = {
            'staging_parcels': {
                'description': 'Initial parcel data from GIS extraction',
                'indexes': [
                    ('parcel_id', ASCENDING),
                    ('municipality', ASCENDING),
                    ('zoning', ASCENDING),
                    ('created_at', DESCENDING)
                ]
            },
            'zoning_data': {
                'description': 'Processed zoning and parcel information',
                'indexes': [
                    ('parcel_id', ASCENDING),
                    ('zoning', ASCENDING),
                    ('municipality', ASCENDING),
                    ('acres', DESCENDING)
                ]
            },
            'enriched_properties': {
                'description': 'Properties with additional data and scoring',
                'indexes': [
                    ('parcel_id', ASCENDING),
                    ('flex_score', DESCENDING),
                    ('municipality', ASCENDING),
                    ('enriched_at', DESCENDING)
                ]
            },
            'flex_candidates': {
                'description': 'Top scoring flex property candidates',
                'indexes': [
                    ('parcel_id', ASCENDING),
                    ('flex_score', DESCENDING),
                    ('municipality', ASCENDING),
                    ('acres', DESCENDING)
                ]
            }
        }
        
        print(f"\nCreating {len(collections_config)} collections...")
        
        for collection_name, config in collections_config.items():
            print(f"\n  Creating '{collection_name}'...")
            print(f"    Purpose: {config['description']}")
            
            # Get collection reference (creates it when we add an index)
            collection = db[collection_name]
            
            # Create indexes
            for field, direction in config['indexes']:
                collection.create_index([(field, direction)])
                print(f"    [OK] Index on '{field}' ({direction})")
        
        # Show final status
        print(f"\n[SUCCESS] All collections created!")
        
        # Get updated stats
        stats = db.command("dbstats")
        collections = db.list_collection_names()
        
        print(f"\nDatabase Status:")
        print(f"  Collections: {len(collections)}")
        print(f"  Collection names: {collections}")
        print(f"  Total size: {stats.get('storageSize', 0) / (1024 * 1024):.2f} MB")
        
        # Atlas storage info
        data_size_mb = stats.get('dataSize', 0) / (1024 * 1024)
        index_size_mb = stats.get('indexSize', 0) / (1024 * 1024)
        total_size_mb = data_size_mb + index_size_mb
        available_mb = 512 - total_size_mb
        
        print(f"\nAtlas Free Tier (512 MB):")
        print(f"  Used: {total_size_mb:.2f} MB ({(total_size_mb/512)*100:.1f}%)")
        print(f"  Available: {available_mb:.2f} MB")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"[ERROR] {e}")
        client.close()
        return False

if __name__ == "__main__":
    success = setup_collections()
    if success:
        print("\n[SUCCESS] MongoDB setup complete!")
        print("\nNext steps:")
        print("1. Run: python main.py --phases 1 --test")
        print("2. This will test the GIS extraction pipeline")
    else:
        print("\n[ERROR] Setup failed. Check the error messages above.")