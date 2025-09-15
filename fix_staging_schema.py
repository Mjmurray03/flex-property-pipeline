#!/usr/bin/env python3
"""
Fix MongoDB schema validation for staging_parcels collection
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
        database_name = os.getenv('MONGODB_DATABASE', 'flexfilter')
        db = client[database_name]
        
        print(f"[SUCCESS] Connected to MongoDB: {database_name}")
        return db, client
        
    except Exception as e:
        print(f"[ERROR] MongoDB connection failed: {e}")
        return None, None

def fix_staging_parcels_collection(db):
    """Fix the staging_parcels collection schema"""
    
    collection_name = 'staging_parcels'
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
    
    # 4. Set new validation schema for staging_parcels with property_use structure
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
                    "bsonType": ["string", "double", "int", "null"],
                    "description": "Last sale date"
                },
                "sale_price": {
                    "bsonType": ["string", "double", "int", "null"],
                    "description": "Last sale price"
                },
                "subdivision": {
                    "bsonType": ["string", "null"],
                    "description": "Subdivision name"
                },
                "shape_area": {
                    "bsonType": ["double", "int", "null"],
                    "description": "GIS shape area"
                },
                "objectid": {
                    "bsonType": ["int", "null"],
                    "description": "GIS object ID"
                },
                "raw_attributes": {
                    "bsonType": ["object", "null"],
                    "description": "Raw GIS attributes"
                },
                "source": {
                    "bsonType": "string",
                    "description": "Data source identifier"
                },
                "extracted_at": {
                    "bsonType": "string",
                    "description": "Extraction timestamp"
                },
                "inserted_at": {
                    "bsonType": ["date", "null"],
                    "description": "Database insertion timestamp"
                },
                "batch_id": {
                    "bsonType": ["string", "null"],
                    "description": "Batch insertion identifier"
                }
            }
        }
    }
    
    try:
        db.command('collMod', collection_name, validator=new_validator)
        print("[SUCCESS] Applied new validation schema for staging_parcels with property_use structure")
        
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

def main():
    """Main function"""
    print("=== MongoDB Schema Fix for staging_parcels Collection ===")
    
    # Connect to MongoDB
    db, client = connect_to_mongodb()
    if db is None:
        return
    
    try:
        # Fix the staging_parcels collection
        if fix_staging_parcels_collection(db):
            print("\n[SUCCESS] staging_parcels schema validation updated successfully")
        else:
            print("\n[ERROR] Failed to update staging_parcels schema validation")
            return
        
        # Show updated stats
        show_collection_stats(db)
        
        print(f"\n=== Schema Fix Complete ===")
        print("The staging_parcels collection now accepts the new structure with:")
        print("- property_use (instead of zoning_code)")
        print("- All the required fields for extracted GIS data")
        print("- Flexible null handling for optional fields")
        print("\nYou can now run the pipeline again to insert the extracted data.")
        
    finally:
        client.close()

if __name__ == "__main__":
    main()