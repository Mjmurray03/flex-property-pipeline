#!/usr/bin/env python
"""
QUICK START SCRIPT - Run this first to test your setup
This will verify MongoDB connection and extract initial data
"""

import os
import sys
import asyncio
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_requirements():
    """Check if required packages are installed"""
    required = ['pymongo', 'aiohttp', 'dotenv']  # dotenv not python-dotenv for import
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print(f"Run: pip install {' '.join(missing)}")
        return False
    
    print("[SUCCESS] All required packages installed")
    return True

def check_mongodb_connection():
    """Test MongoDB connection"""
    try:
        from pymongo import MongoClient
        from dotenv import load_dotenv
        
        load_dotenv()
        
        uri = os.getenv('MONGODB_URI')
        if not uri:
            print("[ERROR] MONGODB_URI not found in .env file")
            print("Please add your MongoDB connection string to .env")
            return False
        
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        
        db = client.flexfilter
        collections = db.list_collection_names()
        
        print(f"[SUCCESS] Connected to MongoDB")
        print(f"   Database: flexfilter")
        print(f"   Collections: {collections}")
        
        # Check storage
        stats = db.command("dbStats")
        size_mb = stats['dataSize'] / (1024 * 1024)
        print(f"   Storage used: {size_mb:.2f} MB / 512 MB")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"[ERROR] MongoDB connection failed: {e}")
        return False

async def test_gis_extraction():
    """Test GIS data extraction"""
    print("\n[TESTING] Testing GIS extraction...")
    
    try:
        from extractors.gis_extractor import GISExtractor
        
        async with GISExtractor() as extractor:
            # Just get count first
            count = await extractor.get_record_count('zoning', "ZONE_ IN ('IL', 'IG', 'IP')")
            print(f"[SUCCESS] Found {count} industrial parcels available")
            
            if count > 0:
                # Extract a small sample
                print("   Extracting sample of 10 parcels...")
                
                url = "https://services2.arcgis.com/HsXtOCMp1Nis1Ogr/arcgis/rest/services/Zoning/FeatureServer/0/query"
                params = {
                    'where': "ZONE_ = 'IL'",
                    'outFields': 'PCN,ZONE_,ACRES',
                    'f': 'json',
                    'resultRecordCount': 10
                }
                
                result = await extractor.fetch(url, params)
                
                if result and 'features' in result:
                    parcels = result['features']
                    print(f"[SUCCESS] Successfully extracted {len(parcels)} sample parcels")
                    
                    # Save sample
                    Path('data/raw').mkdir(parents=True, exist_ok=True)
                    with open('data/raw/sample_parcels.json', 'w') as f:
                        json.dump(parcels, f, indent=2)
                    
                    print("   Sample saved to data/raw/sample_parcels.json")
                    return True
                    
    except ImportError:
        print("[ERROR] GIS extractor not found. Please add gis_extractor.py to extractors/")
        return False
    except Exception as e:
        print(f"[ERROR] GIS extraction failed: {e}")
        return False

def create_project_structure():
    """Create necessary directories"""
    directories = [
        'config',
        'extractors', 
        'processors',
        'database',
        'utils',
        'data/raw',
        'data/processed',
        'logs',
        'tests'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create __init__.py files
    for dir_path in ['config', 'extractors', 'processors', 'database', 'utils']:
        init_file = Path(dir_path) / '__init__.py'
        if not init_file.exists():
            init_file.touch()
    
    print("[SUCCESS] Project structure created")

def create_env_file():
    """Create .env file if it doesn't exist"""
    if not Path('.env').exists():
        env_content = """# MongoDB Configuration
MONGODB_URI=mongodb+srv://username:password@flexfilter-cluster.xxxxx.mongodb.net/
MONGODB_DATABASE=flexfilter

# API Rate Limits
GIS_RATE_LIMIT=5
PROPERTY_APPRAISER_RATE_LIMIT=2

# Processing
MAX_WORKERS=4
CHUNK_SIZE=100

# Storage
MAX_STORAGE_MB=500
CLEANUP_THRESHOLD_MB=450

# Logging
LOG_LEVEL=INFO
"""
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("[SUCCESS] Created .env file - Please add your MongoDB connection string")
        return False
    
    return True

def main():
    """Run all checks"""
    print("=" * 50)
    print("FLEX PROPERTY PIPELINE - QUICK START")
    print("=" * 50)
    
    # Create project structure
    create_project_structure()
    
    # Check environment
    if not create_env_file():
        print("\n[WARNING] Please edit .env file with your MongoDB connection string")
        return
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check MongoDB
    if not check_mongodb_connection():
        return
    
    # Test GIS extraction
    asyncio.run(test_gis_extraction())
    
    print("\n" + "=" * 50)
    print("[SUCCESS] SETUP COMPLETE!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Review the IMPLEMENTATION_GUIDE.md")
    print("2. Copy the provided files to their directories:")
    print("   - mongodb_client.py -> database/")
    print("   - gis_extractor.py -> extractors/")
    print("   - flex_scorer.py -> processors/")
    print("   - main_pipeline.py -> root directory")
    print("3. Run: python main_pipeline.py --phases 1 --test")
    print("\nYour system is ready to identify flex properties!")

if __name__ == "__main__":
    main()
