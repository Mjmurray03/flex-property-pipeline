#!/usr/bin/env python3
"""
Comprehensive Flex Property Aggregator
Combines ALL data sources for qualified industrial properties ≥20,000 sqft
"""

import os
import pandas as pd
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
import json
from datetime import datetime

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
                    print(f"Connected to database: {db_name}")
                    db = test_db
                    break

        if db is None:
            db = client['flexfilter']

        client.admin.command('ping')
        return db
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return None

def calculate_simple_score(property_data):
    """
    Simple scoring based on fundamentals only
    Total possible: 10 points
    """
    score = 0

    # 1. Building Size (0-3 points)
    sqft = property_data.get('building_sqft', 0) or 0
    if 20000 <= sqft <= 50000:
        score += 3  # Perfect flex size
    elif 50000 < sqft <= 100000:
        score += 2  # Still good for multi-tenant
    elif 100000 < sqft <= 200000:
        score += 1  # Large but workable
    elif sqft > 200000:
        score += 0.5  # Very large facilities

    # 2. Property Type (0-3 points)
    use_type = property_data.get('property_use', '')
    if 'WAREH' in use_type or 'DIST' in use_type:
        score += 3  # Best for flex
    elif 'VACANT INDUSTRIAL' in use_type:
        score += 2  # Good potential
    elif 'MFG' in use_type or 'STORAGE' in use_type:
        score += 1  # Acceptable
    elif 'WORKING WATERFRONT' in use_type:
        score += 1.5  # Specialty flex

    # 3. Acreage (0-2 points)
    acres = property_data.get('acres', 0) or 0
    if 1 <= acres <= 5:
        score += 2  # Ideal
    elif 5 < acres <= 10:
        score += 1  # Good
    elif 0.5 <= acres < 1:
        score += 0.5  # Small but workable
    elif 10 < acres <= 25:
        score += 0.5  # Large but possible

    # 4. Market Value (0-2 points)
    value = property_data.get('market_value', 0) or 0
    if 500000 <= value <= 5000000:
        score += 2  # Typical flex range
    elif 5000000 < value <= 10000000:
        score += 1  # Higher end
    elif 250000 <= value < 500000:
        score += 0.5  # Lower end
    elif 10000000 < value <= 25000000:
        score += 0.5  # Premium range

    return round(score, 1)

def aggregate_flex_properties():
    """
    Aggregate all industrial properties with buildings ≥20,000 sqft
    Match data across collections using parcel_id
    """

    db = connect_to_mongodb()
    if db is None:
        print("Failed to connect to MongoDB")
        return []

    print("=== AGGREGATING QUALIFIED FLEX PROPERTIES ===\n")

    # Step 1: Get all industrial properties with building ≥20,000 sqft from zoning_data
    qualified_filter = {
        "property_use": {
            "$in": [
                "WAREH/DIST TERM",
                "VACANT INDUSTRIAL",
                "HEAVY MFG",
                "OPEN STORAGE",
                "WORKING WATERFRONT"
            ]
        },
        "building_sqft": {"$gte": 20000}
    }

    qualified_properties = list(db.zoning_data.find(qualified_filter))

    print(f"Found {len(qualified_properties)} properties with buildings >=20,000 sqft")

    flex_properties = []

    for i, property_data in enumerate(qualified_properties, 1):
        parcel_id = property_data.get('parcel_id')

        print(f"Processing {i}/{len(qualified_properties)}: {parcel_id}")

        # Step 2: Get enriched data if it exists
        enriched = db.enriched_properties.find_one({"parcel_id": parcel_id})

        # Step 3: Get building data if it exists
        building = db.building_data.find_one({"parcel_id": parcel_id})

        # Step 4: Combine ALL available data
        combined = {
            # CORE IDENTIFIERS
            'parcel_id': parcel_id,
            'address': property_data.get('street_address', ''),
            'municipality': property_data.get('municipality', ''),

            # BUILDING INFO
            'building_sqft': property_data.get('building_sqft', 0),
            'year_built': None,
            'warehouse_sqft': None,
            'office_sqft': None,

            # PROPERTY INFO
            'property_use': property_data.get('property_use', ''),
            'acres': property_data.get('acres', 0) or 0,

            # OWNERSHIP
            'owner_name': property_data.get('owner_name', ''),
            'owner_address': '',

            # VALUE INFO
            'market_value': property_data.get('market_value', 0) or 0,
            'assessed_value': property_data.get('assessed_value', 0) or 0,
            'improvement_value': 0,
            'land_value': 0,

            # SALE INFO
            'sale_date': property_data.get('sale_date', ''),
            'sale_price': property_data.get('sale_price', 0) or 0,

            # DATA SOURCES
            'has_enriched_data': enriched is not None,
            'has_building_data': building is not None
        }

        # Build owner address from available fields
        addr_parts = []
        if property_data.get('PADDR1'):
            addr_parts.append(property_data['PADDR1'])
        if property_data.get('CITYNAME'):
            addr_parts.append(property_data['CITYNAME'])
        if property_data.get('STATE'):
            addr_parts.append(property_data['STATE'])
        if property_data.get('ZIP1'):
            addr_parts.append(str(property_data['ZIP1']))
        combined['owner_address'] = ' '.join(addr_parts)

        # Add enriched data if available
        if enriched:
            if enriched.get('building_data'):
                if enriched['building_data'].get('year_built'):
                    combined['year_built'] = enriched['building_data']['year_built']
                combined['improvement_value'] = enriched['building_data'].get('improvement_value', 0) or 0
                combined['land_value'] = enriched['building_data'].get('land_value', 0) or 0

        # Add scraped building details if available (prioritize over enriched data)
        if building:
            if building.get('warehouse_area'):
                combined['warehouse_sqft'] = building['warehouse_area']
            if building.get('office_area'):
                combined['office_sqft'] = building['office_area']
            if building.get('year_built'):
                combined['year_built'] = building['year_built']  # Use scraped if available

        # Calculate office percentage if both office and warehouse areas are available
        if combined['warehouse_sqft'] and combined['office_sqft']:
            total_defined = combined['warehouse_sqft'] + combined['office_sqft']
            combined['office_percentage'] = round((combined['office_sqft'] / total_defined) * 100, 1)
        else:
            combined['office_percentage'] = None

        # Step 5: Calculate SIMPLE flex score
        score = calculate_simple_score(combined)
        combined['flex_score'] = score

        # Add score category
        if score >= 8:
            combined['score_category'] = 'Excellent'
        elif score >= 6:
            combined['score_category'] = 'Very Good'
        elif score >= 4:
            combined['score_category'] = 'Good'
        else:
            combined['score_category'] = 'Fair'

        flex_properties.append(combined)

    return flex_properties

def create_exports(flex_properties):
    """Create CSV and Excel exports with all data"""

    print(f"\n=== CREATING EXPORTS ===")

    # Create data directory if it doesn't exist
    os.makedirs('data/exports', exist_ok=True)

    # Convert to DataFrame
    df = pd.DataFrame(flex_properties)

    # Reorder columns for better readability
    column_order = [
        'flex_score', 'score_category', 'property_use', 'building_sqft',
        'address', 'municipality', 'acres', 'market_value',
        'owner_name', 'year_built', 'warehouse_sqft', 'office_sqft', 'office_percentage',
        'parcel_id', 'assessed_value', 'improvement_value', 'land_value',
        'sale_date', 'sale_price', 'owner_address',
        'has_enriched_data', 'has_building_data'
    ]

    # Reorder columns (keep any extra columns at the end)
    available_columns = [col for col in column_order if col in df.columns]
    remaining_columns = [col for col in df.columns if col not in column_order]
    final_columns = available_columns + remaining_columns

    df = df[final_columns]

    # Export to CSV
    csv_path = 'data/exports/complete_flex_properties.csv'
    df.to_csv(csv_path, index=False)
    print(f"CSV exported to: {csv_path}")

    # Export to Excel with formatting
    excel_path = 'data/exports/complete_flex_properties.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='All Flex Properties', index=False)

        # Get the worksheet
        worksheet = writer.sheets['All Flex Properties']

        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
            worksheet.column_dimensions[column_letter].width = adjusted_width

    print(f"Excel exported to: {excel_path}")

    return csv_path, excel_path

def generate_summary_report(flex_properties):
    """Generate comprehensive summary statistics"""

    print(f"\n=== FLEX PROPERTY AGGREGATION COMPLETE ===")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=" * 60)

    total = len(flex_properties)
    print(f"Total Flex Properties (>=20K sqft): {total}")

    if total == 0:
        print("No qualified properties found!")
        return

    # Score distribution
    excellent = [p for p in flex_properties if p['flex_score'] >= 8]
    very_good = [p for p in flex_properties if 6 <= p['flex_score'] < 8]
    good = [p for p in flex_properties if 4 <= p['flex_score'] < 6]
    fair = [p for p in flex_properties if p['flex_score'] < 4]

    print(f"\nSCORE DISTRIBUTION:")
    print(f"  Excellent (8-10): {len(excellent)} ({len(excellent)/total*100:.1f}%)")
    print(f"  Very Good (6-8):  {len(very_good)} ({len(very_good)/total*100:.1f}%)")
    print(f"  Good (4-6):       {len(good)} ({len(good)/total*100:.1f}%)")
    print(f"  Fair (<4):        {len(fair)} ({len(fair)/total*100:.1f}%)")

    # Property type distribution
    print(f"\nPROPERTY TYPE BREAKDOWN:")
    type_counts = {}
    for prop in flex_properties:
        prop_type = prop['property_use']
        type_counts[prop_type] = type_counts.get(prop_type, 0) + 1

    for prop_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {prop_type}: {count} ({count/total*100:.1f}%)")

    # Building size distribution
    building_sizes = [p['building_sqft'] for p in flex_properties]
    print(f"\nBUILDING SIZE ANALYSIS:")
    print(f"  Average: {sum(building_sizes)/len(building_sizes):,.0f} sqft")
    print(f"  Median: {sorted(building_sizes)[len(building_sizes)//2]:,.0f} sqft")
    print(f"  Largest: {max(building_sizes):,.0f} sqft")
    print(f"  Smallest: {min(building_sizes):,.0f} sqft")

    # Size categories
    small_flex = len([s for s in building_sizes if 20000 <= s <= 50000])
    medium_flex = len([s for s in building_sizes if 50000 < s <= 100000])
    large_flex = len([s for s in building_sizes if 100000 < s <= 200000])
    mega_flex = len([s for s in building_sizes if s > 200000])

    print(f"\nSIZE CATEGORIES:")
    print(f"  Small Flex (20-50K sqft): {small_flex}")
    print(f"  Medium Flex (50-100K sqft): {medium_flex}")
    print(f"  Large Flex (100-200K sqft): {large_flex}")
    print(f"  Mega Flex (200K+ sqft): {mega_flex}")

    # Market value analysis
    values = [p['market_value'] for p in flex_properties if p['market_value'] > 0]
    if values:
        print(f"\nMARKET VALUE ANALYSIS:")
        print(f"  Average: ${sum(values)/len(values):,.0f}")
        print(f"  Median: ${sorted(values)[len(values)//2]:,.0f}")
        print(f"  Highest: ${max(values):,.0f}")
        print(f"  Lowest: ${min(values):,.0f}")

    # Data completeness
    with_enriched = len([p for p in flex_properties if p['has_enriched_data']])
    with_building = len([p for p in flex_properties if p['has_building_data']])
    with_year = len([p for p in flex_properties if p['year_built']])
    with_office_breakdown = len([p for p in flex_properties if p['office_percentage'] is not None])

    print(f"\nDATA COMPLETENESS:")
    print(f"  Properties with enriched data: {with_enriched} ({with_enriched/total*100:.1f}%)")
    print(f"  Properties with building details: {with_building} ({with_building/total*100:.1f}%)")
    print(f"  Properties with year built: {with_year} ({with_year/total*100:.1f}%)")
    print(f"  Properties with office/warehouse breakdown: {with_office_breakdown} ({with_office_breakdown/total*100:.1f}%)")

    # TOP 10 PROPERTIES
    print(f"\nTOP 10 FLEX PROPERTIES:")
    print(f"=" * 100)

    for i, prop in enumerate(flex_properties[:10], 1):
        address = prop['address'] or 'No Address'
        municipality = prop['municipality'] or 'Unknown'

        print(f"{i:2d}. Score {prop['flex_score']}: {address}, {municipality}")
        print(f"    {prop['property_use']} | {prop['building_sqft']:,} sqft | ${prop['market_value']:,.0f}")

        details = []
        if prop['year_built']:
            details.append(f"Built {prop['year_built']}")
        if prop['warehouse_sqft']:
            details.append(f"Warehouse: {prop['warehouse_sqft']:,} sqft")
        if prop['office_sqft']:
            details.append(f"Office: {prop['office_sqft']:,} sqft")
        if prop['office_percentage']:
            details.append(f"Office: {prop['office_percentage']}%")

        if details:
            print(f"    {' | '.join(details)}")
        print()

def main():
    """Main aggregation function"""

    # Run aggregation
    all_flex = aggregate_flex_properties()

    if not all_flex:
        print("No flex properties found!")
        return

    # Sort by score (highest first)
    all_flex.sort(key=lambda x: x['flex_score'], reverse=True)

    # Create exports
    csv_path, excel_path = create_exports(all_flex)

    # Generate summary report
    generate_summary_report(all_flex)

    # Save detailed JSON for further analysis
    json_path = 'data/exports/complete_flex_properties.json'
    with open(json_path, 'w') as f:
        json.dump(all_flex, f, indent=2, default=str)

    print(f"\nFILES CREATED:")
    print(f"  Excel: {excel_path}")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")

    print(f"\n[SUCCESS] AGGREGATION COMPLETE!")
    print(f"Found {len(all_flex)} qualified flex properties ready for analysis!")

if __name__ == "__main__":
    main()