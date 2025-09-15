#!/usr/bin/env python3
"""
MongoDB Atlas Cleanup Script
This script connects to your MongoDB Atlas cluster and:
1. Exports current parcels collection as backup
2. Drops all collections to free up space
3. Creates new collections with proper indexes
4. Shows available storage
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, OperationFailure
import sys

# Load environment variables
load_dotenv()

def get_mongodb_client():
    """Connect to MongoDB Atlas"""
    uri = os.getenv('MONGODB_URI')
    if not uri or '<db_username>' in uri or '<db_password>' in uri:
        print("❌ ERROR: Please update your .env file with actual MongoDB credentials")
        print("Replace <db_username> and <db_password> with your actual values")
        sys.exit(1)
    
    try:
        client = MongoClient(uri)
        # Test connection
        client.admin.command('ping')
        print("[SUCCESS] Successfully connected to MongoDB Atlas")
        return client
    except ConnectionFailure as e:
        print(f"[ERROR] Failed to connect to MongoDB: {e}")
        sys.exit(1)

def backup_parcels_collection(db):
    """Export current parcels collection as backup"""
    print("\n📦 Backing up parcels collection...")
    
    try:
        collection_names = db.list_collection_names()
        print(f"Available collections: {collection_names}")
        
        # Look for parcels collection (could be named differently)
        parcels_collection = None
        for name in collection_names:
            if 'parcel' in name.lower():
                parcels_collection = name
                break
        
        if not parcels_collection:
            print("ℹ️  No parcels collection found - nothing to backup")
            return
        
        # Get collection stats
        collection = db[parcels_collection]
        count = collection.count_documents({})
        print(f"Found {count} documents in '{parcels_collection}' collection")
        
        if count == 0:
            print("ℹ️  Collection is empty - nothing to backup")
            return
        
        # Create backup directory
        backup_dir = "data/backups"
        os.makedirs(backup_dir, exist_ok=True)
        
        # Export to JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = f"{backup_dir}/parcels_backup_{timestamp}.json"
        
        print(f"Exporting to {backup_file}...")
        documents = list(collection.find())
        
        # Convert ObjectId to string for JSON serialization
        for doc in documents:
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
        
        with open(backup_file, 'w') as f:
            json.dump(documents, f, indent=2, default=str)
        
        print(f"✅ Backup completed: {backup_file}")
        print(f"   Exported {len(documents)} documents")
        
    except Exception as e:
        print(f"❌ Error during backup: {e}")

def drop_all_collections(db):
    """Drop all existing collections to free up space"""
    print("\n🗑️  Dropping all existing collections...")
    
    try:
        collection_names = db.list_collection_names()
        
        if not collection_names:
            print("ℹ️  No collections found - nothing to drop")
            return
        
        print(f"Found collections: {collection_names}")
        
        for collection_name in collection_names:
            print(f"  Dropping '{collection_name}'...")
            db[collection_name].drop()
        
        print("✅ All collections dropped successfully")
        
    except Exception as e:
        print(f"❌ Error dropping collections: {e}")

def create_new_collections(db):
    """Create new collections with proper indexes"""
    print("\n🏗️  Creating new collections with indexes...")
    
    collections_config = {
        'staging_parcels': {
            'indexes': [
                ('parcel_id', ASCENDING),
                ('municipality', ASCENDING),
                ('zoning', ASCENDING),
                ('created_at', DESCENDING)
            ]
        },
        'zoning_data': {
            'indexes': [
                ('parcel_id', ASCENDING),
                ('zoning', ASCENDING),
                ('municipality', ASCENDING),
                ('acres', DESCENDING),
                ('geometry', '2dsphere')  # For geospatial queries
            ]
        },
        'enriched_properties': {
            'indexes': [
                ('parcel_id', ASCENDING),
                ('flex_score', DESCENDING),
                ('municipality', ASCENDING),
                ('enriched_at', DESCENDING)
            ]
        },
        'flex_candidates': {
            'indexes': [
                ('parcel_id', ASCENDING),
                ('flex_score', DESCENDING),
                ('municipality', ASCENDING),
                ('acres', DESCENDING),
                ('indicators.size_score', DESCENDING),
                ('indicators.zoning_score', DESCENDING)
            ]
        }
    }
    
    try:
        for collection_name, config in collections_config.items():
            print(f"  Creating '{collection_name}'...")
            
            # Create collection (happens automatically when we create an index)
            collection = db[collection_name]
            
            # Create indexes
            for index_spec in config['indexes']:
                if isinstance(index_spec, tuple) and len(index_spec) == 2:
                    field, direction = index_spec
                    if direction == '2dsphere':
                        collection.create_index([(field, direction)])
                    else:
                        collection.create_index([(field, direction)])
                    print(f"    Created index on '{field}'")
        
        print("✅ All collections and indexes created successfully")
        
    except Exception as e:
        print(f"❌ Error creating collections: {e}")

def show_storage_info(client, db_name):
    """Show available storage information"""
    print("\n💾 Storage Information:")
    
    try:
        # Get database stats
        stats = client[db_name].command("dbstats")
        
        # Convert bytes to MB
        data_size_mb = stats.get('dataSize', 0) / (1024 * 1024)
        storage_size_mb = stats.get('storageSize', 0) / (1024 * 1024)
        index_size_mb = stats.get('indexSize', 0) / (1024 * 1024)
        
        print(f"Database: {db_name}")
        print(f"Collections: {stats.get('collections', 0)}")
        print(f"Documents: {stats.get('objects', 0)}")
        print(f"Data Size: {data_size_mb:.2f} MB")
        print(f"Storage Size: {storage_size_mb:.2f} MB")
        print(f"Index Size: {index_size_mb:.2f} MB")
        print(f"Total Size: {(data_size_mb + index_size_mb):.2f} MB")
        
        # Atlas Free Tier is 512MB
        free_tier_limit = 512
        used_percentage = ((data_size_mb + index_size_mb) / free_tier_limit) * 100
        available_mb = free_tier_limit - (data_size_mb + index_size_mb)
        
        print(f"\n📊 Atlas Free Tier Usage:")
        print(f"Used: {used_percentage:.1f}% of {free_tier_limit} MB")
        print(f"Available: {available_mb:.2f} MB")
        
        if used_percentage > 80:
            print("⚠️  Warning: High storage usage!")
        elif used_percentage < 20:
            print("✅ Good: Low storage usage")
        
        # List collections
        collections = client[db_name].list_collection_names()
        if collections:
            print(f"\nActive Collections:")
            for collection_name in collections:
                try:
                    count = client[db_name][collection_name].count_documents({})
                    print(f"  {collection_name}: {count} documents")
                except:
                    print(f"  {collection_name}: (unable to count)")
        
    except Exception as e:
        print(f"❌ Error getting storage info: {e}")

def main():
    """Main function to orchestrate the cleanup process"""
    print("🚀 MongoDB Atlas Cleanup Script")
    print("=" * 50)
    
    # Connect to MongoDB
    client = get_mongodb_client()
    db_name = os.getenv('DATABASE_NAME', 'flexfilter-cluster')
    db = client[db_name]
    
    try:
        # Show initial storage
        print("\n📊 Initial Storage Status:")
        show_storage_info(client, db_name)
        
        # Ask for confirmation before proceeding
        print("\n⚠️  WARNING: This will delete all existing data!")
        response = input("Do you want to continue? (yes/no): ").lower().strip()
        
        if response not in ['yes', 'y']:
            print("❌ Operation cancelled by user")
            return
        
        # Step 1: Backup parcels collection
        backup_parcels_collection(db)
        
        # Step 2: Drop all collections
        drop_all_collections(db)
        
        # Step 3: Create new collections
        create_new_collections(db)
        
        # Step 4: Show final storage
        print("\n📊 Final Storage Status:")
        show_storage_info(client, db_name)
        
        print("\n✅ MongoDB cleanup completed successfully!")
        print("\nNext steps:")
        print("1. Update your .env file with the correct MongoDB credentials if needed")
        print("2. Run the flex property pipeline: python main.py --test")
        
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    main()