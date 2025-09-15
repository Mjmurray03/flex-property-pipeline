#!/usr/bin/env python3
"""
Remove ALL validation rules from zoning_data collection temporarily
"""

import os
import sys
from pymongo import MongoClient
from dotenv import load_dotenv
import json
from datetime import datetime

def connect_to_mongodb():
    """Connect to MongoDB Atlas"""
    load_dotenv()
    
    uri = os.getenv('MONGODB_URI')
    if not uri:
        print("[ERROR] MONGODB_URI not found in .env file")
        return None, None
    
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

def remove_validation_from_collection(db, collection_name):
    """Remove ALL validation rules from specified collection"""
    
    print(f"\n=== Removing Validation from {collection_name} Collection ===")
    
    # 1. Check current validation rules
    try:
        collection_info = db.command('listCollections', filter={'name': collection_name})
        current_validation = None
        
        for coll_info in collection_info['cursor']['firstBatch']:
            if 'options' in coll_info and 'validator' in coll_info['options']:
                current_validation = coll_info['options']['validator']
                print(f"[INFO] Current validation rules found:")
                print(json.dumps(current_validation, indent=2))
                break
        
        if not current_validation:
            print("[INFO] No validation rules found on collection")
            
    except Exception as e:
        print(f"[ERROR] Could not check validation rules: {e}")
        return False
    
    # 2. Remove ALL validation rules using collMod with empty validator
    try:
        result = db.command('collMod', collection_name, validator={})
        print(f"[SUCCESS] Removed ALL validation rules from {collection_name}")
        print(f"[INFO] collMod result: {result}")
        
    except Exception as e:
        print(f"[ERROR] Could not remove validation: {e}")
        return False
    
    return True

def show_collection_options(db, collection_name):
    """Show collection options after validation removal"""
    
    print(f"\n=== {collection_name} Collection Options After Validation Removal ===")
    
    try:
        collection_infos = db.command('listCollections', filter={'name': collection_name})
        
        for coll_info in collection_infos['cursor']['firstBatch']:
            options = coll_info.get('options', {})
            print(f"Collection options:")
            print(json.dumps(options, indent=2, default=str))
            
            # Check if validator still exists
            if 'validator' in options:
                if options['validator']:
                    print(f"[WARNING] Validator still exists: {options['validator']}")
                else:
                    print(f"[SUCCESS] Validator is now empty - validation removed")
            else:
                print(f"[SUCCESS] No validator key - validation completely removed")
            
            return options
            
    except Exception as e:
        print(f"[ERROR] Could not get collection options: {e}")
        return None

def test_insert_and_cleanup(db, collection_name):
    """Test insert to verify validation is removed, then clean up"""
    
    print(f"\n=== Testing Insert on {collection_name} ===")
    
    collection = db[collection_name]
    
    # Test document with property_use (not zoning_code)
    test_doc = {
        "parcel_id": "TEST123",
        "property_use": "TEST", 
        "source": "test",
        "extracted_at": datetime.utcnow().isoformat()
    }
    
    try:
        # Try to insert test document
        result = collection.insert_one(test_doc)
        print(f"[SUCCESS] Test insert successful - inserted_id: {result.inserted_id}")
        
        # Verify it was inserted
        found_doc = collection.find_one({"parcel_id": "TEST123"})
        if found_doc:
            print(f"[SUCCESS] Test document found in database")
            
            # Clean up - delete the test document
            delete_result = collection.delete_one({"parcel_id": "TEST123"})
            if delete_result.deleted_count == 1:
                print(f"[SUCCESS] Test document deleted successfully")
            else:
                print(f"[WARNING] Could not delete test document")
                
        else:
            print(f"[ERROR] Test document not found after insert")
            
        return True
        
    except Exception as e:
        print(f"[ERROR] Test insert failed: {e}")
        return False

def main():
    """Main function"""
    print("=== Remove ALL Validation Rules from zoning_data Collection ===")
    
    # Connect to MongoDB
    db, client = connect_to_mongodb()
    if db is None:
        return
    
    target_collection = 'zoning_data'
    
    try:
        # Remove validation from zoning_data collection
        if remove_validation_from_collection(db, target_collection):
            print(f"\n[SUCCESS] Validation removal completed for {target_collection}")
        else:
            print(f"\n[ERROR] Failed to remove validation from {target_collection}")
            return
        
        # Show collection options after removal
        options = show_collection_options(db, target_collection)
        
        # Test insert to verify validation is removed
        if test_insert_and_cleanup(db, target_collection):
            print(f"\n[SUCCESS] Validation completely removed and verified")
        else:
            print(f"\n[ERROR] Validation removal verification failed")
            return
        
        print(f"\n=== VALIDATION REMOVAL COMPLETE ===")
        print(f"The {target_collection} collection now has:")
        print("- NO validation rules")
        print("- Can accept documents with property_use field")
        print("- Ready for GIS data insertion")
        print("\nYou can now run: python main.py --phases 1")
        
    finally:
        client.close()

if __name__ == "__main__":
    main()