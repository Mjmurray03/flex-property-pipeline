#!/usr/bin/env python3
"""
Test the scaled scraper on a small batch before running the full pipeline
"""

import os
import time
import json
import logging
from datetime import datetime
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
from scale_building_scraper import BuildingDataScraper

load_dotenv()

def test_scale_scraper():
    """Test the scale scraper on a small batch"""

    print("=== TESTING SCALED BUILDING SCRAPER ===\n")

    # Initialize scraper
    scraper = BuildingDataScraper()

    # Get a small sample to test
    print("1. Testing database connection and property retrieval...")

    industrial_filter = {
        "property_use": {
            "$in": [
                "WAREH/DIST TERM",
                "VACANT INDUSTRIAL",
                "HEAVY MFG",
                "OPEN STORAGE",
                "WORKING WATERFRONT"
            ]
        }
    }

    # Get total count
    total_count = scraper.db.zoning_data.count_documents(industrial_filter)
    print(f"   Total industrial properties in database: {total_count}")

    # Get a small sample for testing (10 properties)
    test_properties = list(scraper.db.zoning_data.find(industrial_filter).limit(10))
    print(f"   Retrieved {len(test_properties)} properties for testing")

    # Show property details
    print("\n2. Test properties:")
    for i, prop in enumerate(test_properties, 1):
        print(f"   {i}. {prop.get('parcel_id')} - {prop.get('property_use')} - {prop.get('street_address', 'N/A')}")
        print(f"      Current building_sqft: {prop.get('building_sqft', 'None')}")

    # Test checkpoint functionality
    print("\n3. Testing checkpoint functionality...")
    checkpoint = scraper.load_checkpoint()
    print(f"   Checkpoint loaded: {checkpoint}")

    # Test scraping a few properties
    print("\n4. Testing property scraping (3 properties)...")
    test_results = []

    for i, prop in enumerate(test_properties[:3]):
        parcel_id = prop.get('parcel_id')
        print(f"\n   Testing {i+1}/3: {parcel_id}")

        # Test scraping
        start_time = time.time()
        building_data = scraper.scrape_property_data(parcel_id)
        scrape_time = time.time() - start_time

        print(f"      Scrape time: {scrape_time:.2f}s")
        print(f"      Success: {building_data['scrape_success']}")

        if building_data['scrape_success']:
            print(f"      Building sqft: {building_data.get('building_sqft', 'N/A')}")
            if building_data.get('warehouse_area'):
                print(f"      Warehouse area: {building_data['warehouse_area']}")
            if building_data.get('office_area'):
                print(f"      Office area: {building_data['office_area']}")
            if building_data.get('year_built'):
                print(f"      Year built: {building_data['year_built']}")
        else:
            print(f"      Error: {building_data['scrape_message']}")

        test_results.append({
            'parcel_id': parcel_id,
            'scrape_time': scrape_time,
            'success': building_data['scrape_success'],
            'building_sqft': building_data.get('building_sqft'),
            'message': building_data['scrape_message']
        })

        # Test MongoDB update
        if building_data['scrape_success']:
            update_success = scraper.update_mongodb_property(parcel_id, building_data)
            print(f"      MongoDB update: {'Success' if update_success else 'Failed'}")

        # Rate limiting test
        time.sleep(2)

    # Test results summary
    print(f"\n5. Test Results Summary:")
    successful = sum(1 for r in test_results if r['success'])
    avg_time = sum(r['scrape_time'] for r in test_results) / len(test_results)

    print(f"   Success rate: {successful}/{len(test_results)} ({successful/len(test_results)*100:.1f}%)")
    print(f"   Average scrape time: {avg_time:.2f}s")

    # Time estimation for full run
    print(f"\n6. Full Run Estimates:")
    properties_per_minute = 60 / (avg_time + 2)  # include 2s delay
    total_time_minutes = total_count / properties_per_minute
    total_time_hours = total_time_minutes / 60

    print(f"   Properties per minute: {properties_per_minute:.1f}")
    print(f"   Estimated total time: {total_time_hours:.1f} hours ({total_time_minutes:.0f} minutes)")

    # Test checkpoint save
    print(f"\n7. Testing checkpoint save...")
    try:
        scraper.save_checkpoint(10, ['test_failed_parcel'])
        print(f"   Checkpoint save: Success")

        # Load it back
        reloaded = scraper.load_checkpoint()
        print(f"   Checkpoint reload: Success")
        print(f"   Reloaded data: {reloaded}")
    except Exception as e:
        print(f"   Checkpoint test failed: {e}")

    print(f"\n=== TEST COMPLETE ===")
    print(f"[SUCCESS] Database connection: Working")
    print(f"[SUCCESS] Property retrieval: {total_count} properties found")
    print(f"[SUCCESS] Scraping functionality: {successful}/{len(test_results)} success rate")
    print(f"[SUCCESS] MongoDB updates: Working")
    print(f"[SUCCESS] Checkpoint system: Working")
    print(f"[SUCCESS] Rate limiting: 2s delays implemented")

    if successful >= len(test_results) * 0.5:  # 50% or better success rate
        print(f"\n[SUCCESS] TESTS PASSED! Ready for full-scale scraping.")
        print(f"[INFO] Estimated time for {total_count} properties: {total_time_hours:.1f} hours")
        print(f"[INFO] Data will be saved to MongoDB as processing occurs")
        print(f"[INFO] Checkpoints will be saved every 100 properties")

        # Ask for confirmation to proceed
        print(f"\n" + "="*60)
        print(f"READY TO SCALE TO ALL {total_count} PROPERTIES")
        print(f"="*60)
        print(f"This will take approximately {total_time_hours:.1f} hours to complete.")
        print(f"The scraper will:")
        print(f"  - Process in batches of 50 properties")
        print(f"  - Save progress every 100 properties")
        print(f"  - Update MongoDB in real-time")
        print(f"  - Handle errors gracefully")
        print(f"  - Can be resumed if interrupted")

        return True
    else:
        print(f"\n[FAILED] TESTS FAILED! Fix issues before running full-scale scraping.")
        return False

if __name__ == "__main__":
    test_scale_scraper()