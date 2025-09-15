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
        possible_names = ['flexfilter-cluster', 'flexfilter', 'flex_properties']

        client = MongoClient(uri, serverSelectionTimeoutMS=5000)

        # Find database with data
        db = None
        for db_name in possible_names:
            test_db = client[db_name]
            collections = test_db.list_collection_names()
            if collections:
                total_docs = sum(test_db[coll].count_documents({}) for coll in collections)
                if total_docs > 0:
                    print(f"Using database: {db_name} ({total_docs} documents)")
                    db = test_db
                    break

        if db is None:
            db = client[possible_names[0]]

        client.admin.command('ping')
        return db
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return None

def analyze_industrial_flex_properties():
    """Analyze industrial flex properties with available data"""
    db = connect_to_mongodb()
    if db is None:
        return

    print("\n=== FOCUSED INDUSTRIAL FLEX PROPERTY ANALYSIS ===")

    # Define industrial property types
    industrial_types = [
        "WAREH/DIST TERM",
        "VACANT INDUSTRIAL",
        "HEAVY MFG",
        "OPEN STORAGE",
        "WORKING WATERFRONT"
    ]

    # Query for industrial properties in both collections
    print("\n1. INDUSTRIAL PROPERTY INVENTORY")
    print("-" * 40)

    # Check enriched_properties first (has 99 properties)
    enriched_industrial = list(db.enriched_properties.find({
        "property_use": {"$in": industrial_types}
    }))

    # Check zoning_data (has 5360 properties)
    zoning_industrial = list(db.zoning_data.find({
        "property_use": {"$in": industrial_types}
    }))

    print(f"Industrial properties in enriched_properties: {len(enriched_industrial)}")
    print(f"Industrial properties in zoning_data: {len(zoning_industrial)}")

    # Combine and deduplicate by parcel_id
    all_industrial = {}

    for prop in enriched_industrial:
        parcel_id = prop.get('parcel_id')
        if parcel_id:
            all_industrial[parcel_id] = {**prop, 'source': 'enriched'}

    for prop in zoning_industrial:
        parcel_id = prop.get('parcel_id')
        if parcel_id and parcel_id not in all_industrial:
            all_industrial[parcel_id] = {**prop, 'source': 'zoning'}

    print(f"Total unique industrial properties: {len(all_industrial)}")

    # Analyze by property type
    print("\n2. BREAKDOWN BY PROPERTY TYPE")
    print("-" * 40)
    use_counts = Counter([prop.get('property_use') for prop in all_industrial.values()])
    for use_type, count in use_counts.most_common():
        print(f"  {use_type}: {count}")

    # Analyze by size (acres)
    print("\n3. PROPERTY SIZE ANALYSIS")
    print("-" * 40)

    properties_with_acres = [prop for prop in all_industrial.values() if prop.get('acres') is not None and prop.get('acres', 0) > 0]
    if properties_with_acres:
        acres_list = [prop['acres'] for prop in properties_with_acres]
        print(f"Properties with acreage data: {len(properties_with_acres)}")
        print(f"Acreage range: {min(acres_list):.1f} - {max(acres_list):.1f} acres")
        print(f"Average acreage: {sum(acres_list)/len(acres_list):.1f} acres")

        # Ideal flex range (1-10 acres)
        flex_size = [prop for prop in properties_with_acres if 1 <= prop['acres'] <= 10]
        print(f"Properties in flex size range (1-10 acres): {len(flex_size)}")

    # Analyze by market value
    print("\n4. MARKET VALUE ANALYSIS")
    print("-" * 40)

    properties_with_value = [prop for prop in all_industrial.values() if prop.get('market_value') is not None and prop.get('market_value', 0) > 0]
    if properties_with_value:
        values = [prop['market_value'] for prop in properties_with_value]
        print(f"Properties with market value data: {len(properties_with_value)}")
        print(f"Value range: ${min(values):,.0f} - ${max(values):,.0f}")
        print(f"Average value: ${sum(values)/len(values):,.0f}")

        # Flex value range ($500K - $10M)
        flex_value = [prop for prop in properties_with_value if 500000 <= prop['market_value'] <= 10000000]
        print(f"Properties in flex value range ($500K-$10M): {len(flex_value)}")

    # Create simplified flex scoring without building square footage
    print("\n5. SIMPLIFIED FLEX SCORING (WITHOUT BUILDING DATA)")
    print("-" * 50)

    flex_candidates = []

    for prop in all_industrial.values():
        score = 0
        criteria_met = []

        # Base industrial property score
        property_use = prop.get('property_use', '')
        if property_use in industrial_types:
            score += 5
            criteria_met.append(f"Industrial type: {property_use}")

        # Property use type scoring (warehouse/distribution scores highest)
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
            criteria_met.append("Manufacturing (moderate)")
        else:
            score += 1
            criteria_met.append("Other industrial")

        # Acres scoring (ideal for flex)
        acres = prop.get('acres', 0) or 0
        if acres is None:
            acres = 0
        if 1 <= acres <= 10:
            score += 3
            criteria_met.append(f"Ideal flex acreage: {acres:.1f}")
        elif 0.5 <= acres < 1:
            score += 2
            criteria_met.append(f"Small but usable: {acres:.1f} acres")
        elif 10 < acres <= 25:
            score += 2
            criteria_met.append(f"Large but possible: {acres:.1f} acres")
        elif acres > 0:
            score += 1
            criteria_met.append(f"Acreage: {acres:.1f}")

        # Market value scoring
        market_value = prop.get('market_value', 0) or 0
        if market_value is None:
            market_value = 0
        if 500000 <= market_value <= 10000000:
            score += 3
            criteria_met.append(f"Ideal flex value: ${market_value:,.0f}")
        elif 200000 <= market_value < 500000:
            score += 2
            criteria_met.append(f"Lower value range: ${market_value:,.0f}")
        elif 10000000 < market_value <= 20000000:
            score += 2
            criteria_met.append(f"Higher value range: ${market_value:,.0f}")
        elif market_value > 0:
            score += 1
            criteria_met.append(f"Market value: ${market_value:,.0f}")

        # Municipality bonus (some areas better for flex)
        municipality = prop.get('municipality', '') or ''
        if municipality and any(city in municipality.upper() for city in ['WEST PALM BEACH', 'POMPANO', 'FORT LAUDERDALE', 'HOLLYWOOD']):
            score += 1
            criteria_met.append(f"Prime location: {municipality}")

        flex_candidates.append({
            'parcel_id': prop.get('parcel_id'),
            'property_use': property_use,
            'acres': acres,
            'market_value': market_value,
            'municipality': prop.get('municipality', ''),
            'address': prop.get('street_address') or prop.get('address', ''),
            'owner_name': prop.get('owner_name', ''),
            'score': score,
            'criteria_met': criteria_met,
            'source': prop.get('source')
        })

    # Sort by score
    flex_candidates.sort(key=lambda x: x['score'], reverse=True)

    print(f"Analyzed {len(flex_candidates)} industrial properties")
    print(f"Average score: {sum(c['score'] for c in flex_candidates)/len(flex_candidates):.1f}")

    # Show top candidates
    print(f"\nTOP 15 FLEX PROPERTY CANDIDATES:")
    print("=" * 80)

    for i, candidate in enumerate(flex_candidates[:15]):
        print(f"\n{i+1}. SCORE: {candidate['score']}/15")
        print(f"   Property: {candidate['property_use']}")
        print(f"   Size: {candidate['acres']:.1f} acres")
        print(f"   Value: ${candidate['market_value']:,.0f}")
        print(f"   Location: {candidate['address']}, {candidate['municipality']}")
        print(f"   Owner: {candidate['owner_name']}")
        print(f"   Parcel ID: {candidate['parcel_id']}")
        print(f"   Source: {candidate['source']}")
        print(f"   Criteria: {', '.join(candidate['criteria_met'])}")

    # Score distribution
    print(f"\n6. SCORE DISTRIBUTION")
    print("-" * 30)
    score_counts = Counter([c['score'] for c in flex_candidates])
    for score in sorted(score_counts.keys(), reverse=True):
        count = score_counts[score]
        print(f"Score {score}: {count} properties")

    # High-scoring candidates (8+ out of 15)
    high_score = [c for c in flex_candidates if c['score'] >= 8]
    print(f"\nHIGH-POTENTIAL CANDIDATES (Score >= 8): {len(high_score)}")

    # Warehouse/Distribution focus
    warehouse_candidates = [c for c in flex_candidates if "WAREH" in c['property_use'] or "DIST" in c['property_use']]
    print(f"WAREHOUSE/DISTRIBUTION PROPERTIES: {len(warehouse_candidates)}")
    if warehouse_candidates:
        avg_warehouse_score = sum(c['score'] for c in warehouse_candidates) / len(warehouse_candidates)
        print(f"Average warehouse score: {avg_warehouse_score:.1f}")

    # Export results
    results = {
        'summary': {
            'total_industrial_properties': len(flex_candidates),
            'high_potential_candidates': len(high_score),
            'warehouse_distribution_count': len(warehouse_candidates),
            'average_score': sum(c['score'] for c in flex_candidates)/len(flex_candidates) if flex_candidates else 0
        },
        'top_15_candidates': flex_candidates[:15],
        'all_candidates': flex_candidates
    }

    with open('industrial_flex_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[SUCCESS] Analysis complete! Results saved to 'industrial_flex_analysis_results.json'")
    print(f"[SUCCESS] Found {len(high_score)} high-potential industrial flex properties")

    return flex_candidates

if __name__ == "__main__":
    candidates = analyze_industrial_flex_properties()