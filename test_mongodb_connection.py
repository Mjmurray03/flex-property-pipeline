#!/usr/bin/env python3
"""
Simple MongoDB Connection Test
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

# Load environment variables
load_dotenv()

def test_connection():
    """Test MongoDB connection with detailed error reporting"""
    print("Testing MongoDB Atlas Connection")
    print("=" * 40)
    
    uri = os.getenv('MONGODB_URI')
    db_name = os.getenv('DATABASE_NAME', 'flexfilter-cluster')
    
    print(f"Database Name: {db_name}")
    print(f"Connection URI: {uri[:50]}...{uri[-30:]}")  # Show partial URI for security
    
    try:
        print("\n1. Creating MongoDB client...")
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        
        print("2. Testing connection (ping)...")
        client.admin.command('ping')
        print("[SUCCESS] Connection successful!")
        
        print("3. Getting database info...")
        db = client[db_name]
        collections = db.list_collection_names()
        print(f"Available collections: {collections}")
        
        print("4. Getting database stats...")
        try:
            stats = db.command("dbstats")
            data_size_mb = stats.get('dataSize', 0) / (1024 * 1024)
            print(f"Database size: {data_size_mb:.2f} MB")
            print(f"Collections: {stats.get('collections', 0)}")
            print(f"Documents: {stats.get('objects', 0)}")
        except Exception as e:
            print(f"[WARNING] Could not get database stats: {e}")
        
        client.close()
        print("\n[SUCCESS] All tests passed!")
        return True
        
    except OperationFailure as e:
        if "Authentication failed" in str(e) or "bad auth" in str(e):
            print(f"[ERROR] Authentication failed: {e}")
            print("\nPossible solutions:")
            print("1. Check username and password in .env file")
            print("2. Ensure database user exists in MongoDB Atlas")
            print("3. Verify user has proper permissions")
            return False
        else:
            print(f"[ERROR] Operation failed: {e}")
            return False
        
    except ConnectionFailure as e:
        print(f"[ERROR] Connection failed: {e}")
        print("\nPossible solutions:")
        print("1. Check your internet connection")
        print("2. Verify MongoDB Atlas cluster is running")
        print("3. Check if your IP is whitelisted in Atlas")
        return False
        
        
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_connection()
    if not success:
        print("\n" + "=" * 40)
        print("CONNECTION TROUBLESHOOTING GUIDE")
        print("=" * 40)
        print("1. MongoDB Atlas Dashboard -> Database Access")
        print("   - Verify user exists and has correct password")
        print("   - User should have 'Atlas admin' or 'readWriteAnyDatabase' role")
        print("2. MongoDB Atlas Dashboard -> Network Access")
        print("   - Add your current IP address or use 0.0.0.0/0 for testing")
        print("3. Check .env file format:")
        print("   MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/...")
        print("   DATABASE_NAME=your_database_name")