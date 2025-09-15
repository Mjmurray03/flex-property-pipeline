#!/usr/bin/env python3
"""
Fix MongoDB schema validation for property_use instead of zoning_code
"""

import os
import sys
from pymongo import MongoClient
from dotenv import load_dotenv
import json

def connect_to_mongodb():
    """Connect to MongoDB Atlas"""
    load_dotenv()
    
    uri = os.getenv('MONGODB_URI')
    if not uri:
        print("Error: MONGODB_URI not found in .env file")
        return None
    
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        
        # Use the database name from config
        database_name = os.getenv('DATABASE_NAME', 'flexfilter')
        db = client[database_name]
        
        print(f"[SUCCESS] Connected to MongoDB: {database_name}")
        return db, client
        
    except Exception as e:
        print(f"[ERROR] MongoDB connection failed: {e}")
        return None, None

def fix_zoning_data_collection(db):
    """Fix the zoning_data collection schema"""
    
    collection_name = 'zoning_data'
    collection = db[collection_name]
    
    print(f"\n=== Fixing {collection_name} Collection ===")
    
    # 1. Check current validation rules
    try:
        collection_info = db.command('listCollections', filter={'name': collection_name})
        current_validation = None
        
        for coll_info in collection_info['cursor']['firstBatch']:
            if 'options' in coll_info and 'validator' in coll_info['options']:
                current_validation = coll_info['options']['validator']
                print(f"Current validation rules found:")
                print(json.dumps(current_validation, indent=2))
                break
        
        if not current_validation:
            print("No validation rules found on collection")
            
    except Exception as e:
        print(f"Could not check validation rules: {e}")
    
    # 2. Drop current validation
    try:
        db.command('collMod', collection_name, validator={})
        print("[SUCCESS] Dropped existing validation rules")
    except Exception as e:
        print(f"[INFO] Could not drop validation (may not exist): {e}")
    
    # 3. Clear existing data that might have wrong structure
    try:
        count_before = collection.count_documents({})
        print(f"Documents in collection before clear: {count_before}")
        
        if count_before > 0:
            result = collection.delete_many({})
            print(f"[SUCCESS] Cleared {result.deleted_count} existing documents")
        else:
            print("[INFO] Collection was already empty")
            
    except Exception as e:
        print(f"[ERROR] Could not clear collection: {e}")
    
    # 4. Set new validation schema for property_use structure
    new_validator = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["parcel_id", "property_use", "source", "extracted_at"],
            "properties": {
                "parcel_id": {
                    "bsonType": "string",
                    "description": "Unique parcel identifier"
                },
                "property_use": {
                    "bsonType": "string", 
                    "description": "Property use classification (replaces zoning_code)"
                },
                "acres": {
                    "bsonType": ["double", "int", "null"],
                    "description": "Property size in acres"
                },
                "owner_name": {
                    "bsonType": ["string", "null"],
                    "description": "Property owner name"
                },
                "street_address": {
                    "bsonType": ["string", "null"],
                    "description": "Street address of property"
                },
                "municipality": {
                    "bsonType": ["string", "null"],
                    "description": "Municipality/city"
                },
                "market_value": {
                    "bsonType": ["double", "int", "null"],
                    "description": "Market value of property"
                },
                "assessed_value": {
                    "bsonType": ["double", "int", "null"],
                    "description": "Assessed value of property"
                },
                "sale_date": {
                    "bsonType": ["string", "null"],
                    "description": "Last sale date"
                },
                "sale_price": {
                    "bsonType": ["double", "int", "null"],
                    "description": "Last sale price"
                },
                "source": {
                    "bsonType": "string",
                    "description": "Data source identifier"
                },
                "extracted_at": {
                    "bsonType": "string",
                    "description": "Extraction timestamp"
                }
            }
        }
    }
    
    try:
        db.command('collMod', collection_name, validator=new_validator)
        print("[SUCCESS] Applied new validation schema for property_use structure")
        
        # Show the new schema
        print("\nNew validation schema:")
        print(json.dumps(new_validator, indent=2))
        
    except Exception as e:
        print(f"[ERROR] Could not apply new validation: {e}")
        return False
    
    return True

def show_collection_stats(db):
    """Show stats for all collections"""
    print(f"\n=== Collection Statistics ===")
    
    collections = ['staging_parcels', 'zoning_data', 'enriched_properties', 'flex_candidates']
    
    for coll_name in collections:
        try:
            collection = db[coll_name]
            count = collection.count_documents({})
            
            # Get collection stats
            try:
                stats = db.command('collStats', coll_name)
                size_bytes = stats.get('size', 0)
                size_mb = size_bytes / (1024 * 1024)
                
                print(f"  {coll_name}: {count} documents, {size_mb:.2f} MB")
                
            except:
                print(f"  {coll_name}: {count} documents")
                
        except Exception as e:
            print(f"  {coll_name}: Error - {e}")
    
    # Overall database stats
    try:
        db_stats = db.command('dbStats')
        data_size_mb = db_stats['dataSize'] / (1024 * 1024)
        index_size_mb = db_stats['indexSize'] / (1024 * 1024) 
        total_size_mb = data_size_mb + index_size_mb
        
        print(f"\nDatabase Total:")
        print(f"  Data: {data_size_mb:.2f} MB")
        print(f"  Indexes: {index_size_mb:.2f} MB") 
        print(f"  Total: {total_size_mb:.2f} MB")
        print(f"  Available: {512 - total_size_mb:.2f} MB / 512 MB")
        
    except Exception as e:
        print(f"Could not get database stats: {e}")

def main():
    """Main function"""
    print("=== MongoDB Schema Fix for Property Use Structure ===")
    
    # Connect to MongoDB
    db, client = connect_to_mongodb()
    if db is None:
        return
    
    try:
        # Fix the zoning_data collection
        if fix_zoning_data_collection(db):
            print("\n[SUCCESS] Schema validation updated successfully")
        else:
            print("\n[ERROR] Failed to update schema validation")
            return
        
        # Show updated stats
        show_collection_stats(db)
        
        print(f"\n=== Schema Fix Complete ===")
        print("The zoning_data collection now accepts the new structure with:")
        print("- property_use (instead of zoning_code)")
        print("- All the required fields for extracted GIS data")
        print("- Flexible null handling for optional fields")
        print("\nYou can now run the pipeline again to insert the extracted data.")
        
    finally:
        client.close()

if __name__ == "__main__":
    main()