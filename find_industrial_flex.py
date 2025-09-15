#!/usr/bin/env python3

import os
import pymongo
import json
from pymongo import MongoClient
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

def connect_to_mongodb():
    """Connect to MongoDB and return database"""
    try:
        uri = os.getenv('MONGODB_URI')
        # Try both database names to find data
        possible_names = ['flexfilter-cluster', 'flexfilter', 'flex_properties']

        client = MongoClient(uri, serverSelectionTimeoutMS=5000)

        # Test both databases to find the one with data
        db = None
        for db_name in possible_names:
            test_db = client[db_name]
            collections = test_db.list_collection_names()
            if collections:
                total_docs = sum(test_db[coll].count_documents({}) for coll in collections)
                if total_docs > 0:
                    print(f"Found data in database: {db_name} ({total_docs} documents)")
                    db = test_db
                    break

        if db is None:
            # Default to first name if no data found
            db = client[possible_names[0]]

        # Test connection
        client.admin.command('ping')
        print("Connected to MongoDB")
        return db
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return None

def analyze_industrial_properties():
    """Find and analyze industrial flex properties"""
    db = connect_to_mongodb()
    if db is None:
        return

    # Debug: Check what collections exist and have data
    print(f"\n=== DEBUG: Collection Data Check ===")
    collections = db.list_collection_names()
    print(f"Available collections: {collections}")

    property_data_found = False

    for coll_name in collections:
        count = db[coll_name].count_documents({})
        print(f"  {coll_name}: {count} documents")

        if count > 0:
            sample = db[coll_name].find_one()
            if sample:
                print(f"    Sample keys: {list(sample.keys())[:10]}")  # Show first 10 keys

                # Check for property_use or similar fields
                if 'property_use' in sample or 'use_code' in sample or 'USE_CODE' in sample or 'zoning' in sample or 'ZONE_' in sample:
                    property_data_found = True
                    print(f"    ** {coll_name} contains property data!")

                    # Show property use distribution
                    if count < 1000:  # Only if not too many documents
                        use_field = None
                        for field in ['property_use', 'use_code', 'USE_CODE', 'ZONE_', 'zoning', 'land_use']:
                            if field in sample:
                                use_field = field
                                break

                        if use_field:
                            pipeline = [
                                {'$group': {'_id': f'${use_field}', 'count': {'$sum': 1}}},
                                {'$sort': {'count': -1}},
                                {'$limit': 10}
                            ]
                            use_distribution = list(db[coll_name].aggregate(pipeline))
                            print(f"    Top property uses ({use_field}):")
                            for item in use_distribution:
                                print(f"      {item['_id']}: {item['count']}")

    if not property_data_found:
        print(f"\n[WARNING] No property use data found in any collection!")
        print(f"This may mean the database needs to be populated with property data first.")
        return

    # Define industrial property types
    industrial_types = [
        "WAREH/DIST TERM",
        "VACANT INDUSTRIAL",
        "HEAVY MFG",
        "OPEN STORAGE",
        "WORKING WATERFRONT"
    ]

    print("\n=== STEP 1: Query Industrial Properties ===")

    # Query for industrial properties
    industrial_query = {
        "property_use": {"$in": industrial_types}
    }

    industrial_properties = list(db.enriched_properties.find(industrial_query))
    print(f"Found {len(industrial_properties)} industrial properties")

    # Count by property use type
    use_counts = Counter([prop.get('property_use') for prop in industrial_properties])
    print("\nBreakdown by property use:")
    for use_type, count in use_counts.items():
        print(f"  {use_type}: {count}")

    print("\n=== STEP 2: Check Building Square Footage Data ===")

    # Check how many have building data
    properties_with_building_data = 0
    properties_with_total_area = 0
    building_areas = []

    for prop in industrial_properties:
        prop_id = prop.get('_id')

        # Check if building_data exists for this property
        building_data = db.building_data.find_one({"property_id": prop_id})

        if building_data:
            properties_with_building_data += 1

            total_area = building_data.get('total_area')
            if total_area and total_area > 0:
                properties_with_total_area += 1
                building_areas.append(total_area)

    print(f"Properties with building_data records: {properties_with_building_data}")
    print(f"Properties with total_area > 0: {properties_with_total_area}")

    if building_areas:
        print(f"Building area stats:")
        print(f"  Min: {min(building_areas):,.0f} SF")
        print(f"  Max: {max(building_areas):,.0f} SF")
        print(f"  Average: {sum(building_areas)/len(building_areas):,.0f} SF")
        print(f"  Properties ≥20,000 SF: {sum(1 for area in building_areas if area >= 20000)}")

        # Show some examples
        print(f"\nExample properties with building areas:")
        count = 0
        for prop in industrial_properties:
            if count >= 5:
                break

            prop_id = prop.get('_id')
            building_data = db.building_data.find_one({"property_id": prop_id})

            if building_data and building_data.get('total_area', 0) > 0:
                print(f"  {prop.get('property_use')} - {building_data.get('total_area'):,.0f} SF - Market Value: ${prop.get('market_value', 0):,.0f}")
                count += 1

    print("\n=== STEP 3: Flex Property Scoring Criteria ===")

    flex_candidates = []

    for prop in industrial_properties:
        prop_id = prop.get('_id')
        building_data = db.building_data.find_one({"property_id": prop_id})

        # Initialize scoring
        score = 0
        criteria_met = []

        # Required: Is it industrial/warehouse?
        property_use = prop.get('property_use', '')
        if property_use in industrial_types:
            score += 10
            criteria_met.append(f"Industrial type: {property_use}")
        else:
            continue  # Skip if not industrial

        # Required: Building ≥20,000 SF?
        building_area = 0
        if building_data and building_data.get('total_area'):
            building_area = building_data.get('total_area')
            if building_area >= 20000:
                score += 10
                criteria_met.append(f"Building >=20K SF: {building_area:,.0f}")
            else:
                continue  # Skip if too small
        else:
            continue  # Skip if no building data

        # Property use type scoring (warehouse/distribution scores highest)
        if "WAREH" in property_use or "DIST" in property_use:
            score += 5
            criteria_met.append("Warehouse/Distribution (highest)")
        elif "VACANT INDUSTRIAL" in property_use:
            score += 4
            criteria_met.append("Vacant Industrial (good)")
        elif "MFG" in property_use:
            score += 3
            criteria_met.append("Manufacturing (moderate)")
        else:
            score += 2
            criteria_met.append("Other industrial (basic)")

        # Acres between 1-10 (ideal for flex)
        acres = prop.get('acres', 0)
        if 1 <= acres <= 10:
            score += 3
            criteria_met.append(f"Ideal acreage: {acres:.1f}")
        elif acres > 0:
            score += 1
            criteria_met.append(f"Acreage: {acres:.1f}")

        # Market value $500K-$10M (typical flex range)
        market_value = prop.get('market_value', 0)
        if 500000 <= market_value <= 10000000:
            score += 3
            criteria_met.append(f"Flex value range: ${market_value:,.0f}")
        elif market_value > 0:
            score += 1
            criteria_met.append(f"Market value: ${market_value:,.0f}")

        flex_candidates.append({
            'property_id': prop_id,
            'property_use': property_use,
            'building_area': building_area,
            'acres': acres,
            'market_value': market_value,
            'score': score,
            'criteria_met': criteria_met
        })

    print(f"\nFound {len(flex_candidates)} properties meeting minimum flex criteria (industrial + >=20K SF)")

    # Sort by score
    flex_candidates.sort(key=lambda x: x['score'], reverse=True)

    print(f"\nTop 10 Flex Property Candidates:")
    for i, candidate in enumerate(flex_candidates[:10]):
        print(f"\n{i+1}. Score: {candidate['score']}")
        print(f"   Type: {candidate['property_use']}")
        print(f"   Building: {candidate['building_area']:,.0f} SF")
        print(f"   Acres: {candidate['acres']:.1f}")
        print(f"   Value: ${candidate['market_value']:,.0f}")
        print(f"   Criteria: {', '.join(candidate['criteria_met'])}")

    return flex_candidates

if __name__ == "__main__":
    candidates = analyze_industrial_properties()