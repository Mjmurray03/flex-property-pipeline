#!/usr/bin/env python3
"""
Final flex property analysis combining scraped building data with property data
"""

import os
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
import json

load_dotenv()

def connect_to_mongodb():
    """Connect to MongoDB and return database"""
    try:
        uri = os.getenv('MONGODB_URI')
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)

        # Find database with data
        db = None
        for db_name in ['flexfilter', 'flexfilter-cluster', 'flex_properties']:
            test_db = client[db_name]
            collections = test_db.list_collection_names()
            if collections:
                total_docs = sum(test_db[coll].count_documents({}) for coll in collections)
                if total_docs > 0:
                    print(f"Using database: {db_name}")
                    db = test_db
                    break

        if db is None:
            db = client['flexfilter']

        client.admin.command('ping')
        return db
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return None

def analyze_flex_properties_with_building_data():
    """Comprehensive flex property analysis with building square footage"""

    db = connect_to_mongodb()
    if db is None:
        return

    print("=== FINAL FLEX PROPERTY ANALYSIS WITH BUILDING DATA ===\n")

    # Get all industrial properties with building data
    pipeline = [
        {
            '$lookup': {
                'from': 'building_data',
                'localField': 'parcel_id',
                'foreignField': 'parcel_id',
                'as': 'building_info'
            }
        },
        {
            '$match': {
                'property_use': {
                    '$in': [
                        'WAREH/DIST TERM',
                        'VACANT INDUSTRIAL',
                        'HEAVY MFG',
                        'OPEN STORAGE',
                        'WORKING WATERFRONT'
                    ]
                }
            }
        },
        {
            '$addFields': {
                'has_building_data': {'$gt': [{'$size': '$building_info'}, 0]},
                'building_sqft': {
                    '$ifNull': [
                        {'$arrayElemAt': ['$building_info.building_sqft', 0]},
                        0
                    ]
                },
                'year_built': {
                    '$ifNull': [
                        {'$arrayElemAt': ['$building_info.year_built', 0]},
                        None
                    ]
                },
                'warehouse_area': {
                    '$ifNull': [
                        {'$arrayElemAt': ['$building_info.warehouse_area', 0]},
                        None
                    ]
                },
                'office_area': {
                    '$ifNull': [
                        {'$arrayElemAt': ['$building_info.office_area', 0]},
                        None
                    ]
                }
            }
        }
    ]

    properties = list(db.zoning_data.aggregate(pipeline))

    print(f"1. INDUSTRIAL PROPERTY INVENTORY WITH BUILDING DATA")
    print("-" * 60)
    print(f"Total industrial properties: {len(properties)}")

    # Separate properties with/without building data
    with_building_data = [p for p in properties if p.get('has_building_data', False)]
    without_building_data = [p for p in properties if not p.get('has_building_data', False)]

    print(f"Properties with building data: {len(with_building_data)}")
    print(f"Properties without building data: {len(without_building_data)}")

    # Analyze properties with building data
    if with_building_data:
        print(f"\n2. BUILDING SIZE ANALYSIS (Properties with Data)")
        print("-" * 60)

        # Extract building sizes
        building_sizes = [p['building_sqft'] for p in with_building_data if p['building_sqft'] > 0]

        if building_sizes:
            building_sizes.sort(reverse=True)
            print(f"Properties with square footage: {len(building_sizes)}")
            print(f"Largest building: {max(building_sizes):,} sqft")
            print(f"Smallest building: {min(building_sizes):,} sqft")
            print(f"Average building size: {sum(building_sizes)//len(building_sizes):,} sqft")
            print(f"Median building size: {building_sizes[len(building_sizes)//2]:,} sqft")

            # Categorize by size
            mega_facilities = [s for s in building_sizes if s >= 100000]    # 100K+ sqft
            large_flex = [s for s in building_sizes if 50000 <= s < 100000] # 50-100K sqft
            standard_flex = [s for s in building_sizes if 20000 <= s < 50000] # 20-50K sqft
            medium_buildings = [s for s in building_sizes if 10000 <= s < 20000] # 10-20K sqft
            small_buildings = [s for s in building_sizes if s < 10000]      # <10K sqft

            print(f"\nBuilding Size Categories:")
            print(f"  Mega Facilities (100K+ sqft): {len(mega_facilities)}")
            print(f"  Large Flex (50-100K sqft): {len(large_flex)}")
            print(f"  Standard Flex (20-50K sqft): {len(standard_flex)}")
            print(f"  Medium (10-20K sqft): {len(medium_buildings)}")
            print(f"  Small (<10K sqft): {len(small_buildings)}")

            # Total flex-qualified properties (>=20K sqft)
            flex_qualified = len(mega_facilities) + len(large_flex) + len(standard_flex)
            print(f"\nFLEX-QUALIFIED PROPERTIES (>=20K sqft): {flex_qualified}")
            print(f"Percentage of analyzed properties: {flex_qualified/len(building_sizes)*100:.1f}%")

    print(f"\n3. DETAILED FLEX CANDIDATE ANALYSIS")
    print("-" * 60)

    # Score all properties (including those without building data)
    scored_properties = []

    for prop in properties:
        score = 0
        criteria_met = []
        building_sqft = prop.get('building_sqft', 0) or 0

        # Base industrial property score
        property_use = prop.get('property_use', '')
        if property_use:
            score += 5
            criteria_met.append(f"Industrial: {property_use}")

        # Property use type scoring
        if "WAREH" in property_use or "DIST" in property_use:
            score += 4
            criteria_met.append("Warehouse/Distribution (highest)")
        elif "VACANT INDUSTRIAL" in property_use:
            score += 3
            criteria_met.append("Vacant Industrial (flexible)")
        elif "WORKING WATERFRONT" in property_use:
            score += 3
            criteria_met.append("Working Waterfront (flexible)")
        elif "MFG" in property_use:
            score += 2
            criteria_met.append("Manufacturing")
        else:
            score += 1
            criteria_met.append("Other industrial")

        # CRITICAL: Building square footage scoring
        if building_sqft >= 100000:
            score += 10
            criteria_met.append(f"Mega facility: {building_sqft:,} sqft")
        elif building_sqft >= 50000:
            score += 8
            criteria_met.append(f"Large flex: {building_sqft:,} sqft")
        elif building_sqft >= 20000:
            score += 6
            criteria_met.append(f"Standard flex: {building_sqft:,} sqft")
        elif building_sqft >= 10000:
            score += 3
            criteria_met.append(f"Medium building: {building_sqft:,} sqft")
        elif building_sqft > 0:
            score += 1
            criteria_met.append(f"Small building: {building_sqft:,} sqft")
        else:
            score += 0
            criteria_met.append("No building data")

        # Acreage scoring
        acres = prop.get('acres', 0) or 0
        if 1 <= acres <= 10:
            score += 3
            criteria_met.append(f"Ideal acres: {acres:.1f}")
        elif 0.5 <= acres <= 25:
            score += 2
            criteria_met.append(f"Good acres: {acres:.1f}")
        elif acres > 0:
            score += 1
            criteria_met.append(f"Acres: {acres:.1f}")

        # Market value scoring
        market_value = prop.get('market_value', 0) or 0
        if 500000 <= market_value <= 10000000:
            score += 3
            criteria_met.append(f"Ideal value: ${market_value:,.0f}")
        elif market_value > 0:
            score += 1
            criteria_met.append(f"Value: ${market_value:,.0f}")

        # Year built bonus (newer buildings)
        year_built = prop.get('year_built')
        if year_built and year_built >= 1990:
            score += 2
            criteria_met.append(f"Modern: {year_built}")
        elif year_built and year_built >= 1970:
            score += 1
            criteria_met.append(f"Updated: {year_built}")

        # Office/warehouse mix bonus
        warehouse_area = prop.get('warehouse_area', 0) or 0
        office_area = prop.get('office_area', 0) or 0
        if warehouse_area > 0 and office_area > 0:
            ratio = office_area / (warehouse_area + office_area)
            if 0.05 <= ratio <= 0.30:  # 5-30% office space is ideal for flex
                score += 2
                criteria_met.append(f"Flex mix: {ratio*100:.0f}% office")

        scored_properties.append({
            'parcel_id': prop.get('parcel_id'),
            'property_use': property_use,
            'building_sqft': building_sqft,
            'warehouse_area': warehouse_area,
            'office_area': office_area,
            'acres': acres,
            'market_value': market_value,
            'year_built': year_built,
            'municipality': prop.get('municipality', ''),
            'address': prop.get('street_address', ''),
            'has_building_data': prop.get('has_building_data', False),
            'score': score,
            'criteria_met': criteria_met
        })

    # Sort by score
    scored_properties.sort(key=lambda x: x['score'], reverse=True)

    print(f"Analyzed {len(scored_properties)} total industrial properties")

    # Show top flex candidates
    print(f"\nTOP 20 FLEX PROPERTY CANDIDATES:")
    print("=" * 100)

    for i, prop in enumerate(scored_properties[:20]):
        print(f"\n{i+1}. SCORE: {prop['score']}/30")
        print(f"   Property: {prop['property_use']}")
        if prop['building_sqft'] > 0:
            print(f"   Building: {prop['building_sqft']:,} sqft")
            if prop['warehouse_area']:
                print(f"   Warehouse: {prop['warehouse_area']:,} sqft")
            if prop['office_area']:
                print(f"   Office: {prop['office_area']:,} sqft")
        else:
            print(f"   Building: NO DATA")
        print(f"   Size: {prop['acres']:.1f} acres")
        print(f"   Value: ${prop['market_value']:,.0f}")
        if prop['year_built']:
            print(f"   Built: {prop['year_built']}")
        print(f"   Location: {prop['address']}, {prop['municipality']}")
        print(f"   Parcel: {prop['parcel_id']}")
        print(f"   Data: {'YES' if prop['has_building_data'] else 'NO'}")
        print(f"   Criteria: {', '.join(prop['criteria_met'][:3])}...")

    # Summary statistics
    high_score = [p for p in scored_properties if p['score'] >= 20]
    with_building = [p for p in scored_properties if p['has_building_data']]
    flex_qualified = [p for p in scored_properties if p['building_sqft'] >= 20000]

    print(f"\n4. FINAL SUMMARY")
    print("-" * 60)
    print(f"Total industrial properties analyzed: {len(scored_properties)}")
    print(f"Properties with building data: {len(with_building)}")
    print(f"High-scoring candidates (20+ points): {len(high_score)}")
    print(f"Flex-qualified buildings (>=20K sqft): {len(flex_qualified)}")

    if len(with_building) > 0:
        flex_rate = len(flex_qualified) / len(with_building) * 100
        print(f"Flex qualification rate: {flex_rate:.1f}%")

    # Export results
    results = {
        'summary': {
            'total_properties': len(scored_properties),
            'with_building_data': len(with_building),
            'high_scoring_candidates': len(high_score),
            'flex_qualified': len(flex_qualified),
            'analysis_date': '2025-09-12'
        },
        'top_candidates': scored_properties[:50],
        'flex_qualified_properties': flex_qualified
    }

    with open('final_flex_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[SUCCESS] Analysis complete! Results saved to 'final_flex_analysis_results.json'")
    print(f"[SUCCESS] Found {len(flex_qualified)} flex-qualified industrial properties")

if __name__ == "__main__":
    analyze_flex_properties_with_building_data()