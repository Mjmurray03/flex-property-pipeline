#!/usr/bin/env python3
"""
Improved building data scraper with corrected parsing logic
"""

import os
import time
import requests
from bs4 import BeautifulSoup
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
import json
import re
from collections import Counter

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

def get_sample_parcels(db, limit=50):
    """Get a larger sample of industrial parcels"""
    print(f"\n=== Getting {limit} Industrial Property Parcels ===")

    # Get a mix of different industrial types
    pipeline = [
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
        {'$sample': {'size': limit}},  # Random sample
        {'$project': {
            'parcel_id': 1,
            'property_use': 1,
            'address': {'$ifNull': ['$street_address', '$address', 'N/A']},
            'market_value': 1,
            'acres': 1,
            'municipality': 1
        }}
    ]

    parcels = list(db.zoning_data.aggregate(pipeline))

    print(f"Found {len(parcels)} industrial parcels")

    # Show breakdown by type
    type_counts = Counter([p.get('property_use') for p in parcels])
    print("\nProperty type breakdown:")
    for prop_type, count in type_counts.most_common():
        print(f"  {prop_type}: {count}")

    return parcels

def scrape_property_data(parcel_id):
    """Improved scraper with corrected parsing logic"""

    clean_parcel_id = parcel_id.replace('-', '').replace(' ', '')

    # Use the working URL pattern
    url = f"https://pbcpao.gov/Property/RenderPrintSum?parcelId={clean_parcel_id}&flag=ALL"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    scraped_data = {
        'parcel_id': parcel_id,
        'building_sqft': None,
        'total_living_area': None,
        'heated_area': None,
        'warehouse_area': None,
        'office_area': None,
        'year_built': None,
        'building_count': 0,
        'scrape_success': False,
        'scrape_message': '',
        'raw_areas': {}
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            page_text = soup.get_text()

            # Extract all numeric values that could be square footage
            # Look for patterns in the actual HTML structure
            tables = soup.find_all('table')

            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        label = cells[0].get_text().strip().lower()
                        value_text = cells[1].get_text().strip()

                        # Extract numeric values
                        numbers = re.findall(r'[\d,]+', value_text)
                        if numbers:
                            try:
                                value = int(numbers[0].replace(',', ''))
                                if 50 < value < 10000000:  # Reasonable range
                                    scraped_data['raw_areas'][label] = value

                                    # Map specific fields
                                    if 'year' in label and ('built' in label or 'added' in label):
                                        if 1800 <= value <= 2025:
                                            scraped_data['year_built'] = value
                            except ValueError:
                                continue

            # Also extract from general page text patterns
            # Look for the specific patterns we found in our test
            area_patterns = [
                (r'\*total square feet\s*:\s*([0-9,]+)', 'total_sqft_star'),
                (r'total square footage\s*:?\s*([0-9,]+)', 'total_square_footage'),
                (r'warehouse storage\s*:?\s*([0-9,]+)', 'warehouse_storage'),
                (r'warehouse\s*:?\s*([0-9,]+)', 'warehouse'),
                (r'whse\s+office\s*:?\s*([0-9,]+)', 'warehouse_office'),
                (r'office\s*:?\s*([0-9,]+)', 'office'),
                (r'retail store\s*:?\s*([0-9,]+)', 'retail'),
                (r'year built\s*:?\s*([0-9]{4})', 'year_built'),
            ]

            for pattern, field_name in area_patterns:
                matches = re.findall(pattern, page_text, re.IGNORECASE)
                if matches:
                    for match in matches:
                        try:
                            value = int(match.replace(',', ''))

                            if field_name == 'year_built' and 1800 <= value <= 2025:
                                scraped_data['year_built'] = value
                            elif field_name != 'year_built' and 50 < value < 10000000:
                                scraped_data['raw_areas'][field_name] = value
                        except ValueError:
                            continue

            # Now intelligently extract the best building square footage
            if scraped_data['raw_areas']:
                # Priority order for building square footage
                priority_fields = [
                    'total_sqft_star',         # "*total square feet"
                    'total_square_footage',    # "total square footage"
                    'warehouse_storage',       # "warehouse storage"
                    'warehouse',               # "warehouse"
                ]

                # Find the best square footage value
                best_sqft = None
                best_field = None

                for field in priority_fields:
                    if field in scraped_data['raw_areas']:
                        best_sqft = scraped_data['raw_areas'][field]
                        best_field = field
                        break

                # If no priority field found, take the largest reasonable area
                if best_sqft is None:
                    area_values = {k: v for k, v in scraped_data['raw_areas'].items()
                                 if isinstance(v, int) and 100 <= v <= 1000000}
                    if area_values:
                        best_field = max(area_values, key=area_values.get)
                        best_sqft = area_values[best_field]

                if best_sqft:
                    scraped_data['building_sqft'] = best_sqft
                    scraped_data['primary_area_source'] = best_field

                # Extract component areas
                if 'warehouse' in scraped_data['raw_areas']:
                    scraped_data['warehouse_area'] = scraped_data['raw_areas']['warehouse']
                if 'warehouse_office' in scraped_data['raw_areas']:
                    scraped_data['office_area'] = scraped_data['raw_areas']['warehouse_office']
                elif 'office' in scraped_data['raw_areas']:
                    scraped_data['office_area'] = scraped_data['raw_areas']['office']

                scraped_data['scrape_success'] = True
                scraped_data['scrape_message'] = f"Successfully extracted {len(scraped_data['raw_areas'])} area fields"
            else:
                scraped_data['scrape_message'] = "Page found but no area data extracted"

        elif response.status_code == 404:
            scraped_data['scrape_message'] = "Property page not found (404)"
        else:
            scraped_data['scrape_message'] = f"HTTP {response.status_code}"

    except requests.exceptions.Timeout:
        scraped_data['scrape_message'] = "Request timeout"
    except requests.exceptions.RequestException as e:
        scraped_data['scrape_message'] = f"Request error: {str(e)[:50]}"
    except Exception as e:
        scraped_data['scrape_message'] = f"Scraping error: {str(e)[:50]}"

    return scraped_data

def save_to_mongodb(db, building_data):
    """Save building data to MongoDB"""
    print(f"\n=== Saving Building Data to MongoDB ===")

    # Create building_data collection if it doesn't exist
    collection = db.building_data

    # Prepare documents for insertion
    documents = []
    for data in building_data:
        if data['scrape_success'] and data['building_sqft']:
            doc = {
                'parcel_id': data['parcel_id'],
                'building_sqft': data['building_sqft'],
                'warehouse_area': data.get('warehouse_area'),
                'office_area': data.get('office_area'),
                'year_built': data.get('year_built'),
                'primary_area_source': data.get('primary_area_source'),
                'raw_areas': data.get('raw_areas', {}),
                'scraped_at': time.time(),
                'scrape_url': f"https://pbcpao.gov/Property/RenderPrintSum?parcelId={data['parcel_id']}&flag=ALL"
            }
            documents.append(doc)

    if documents:
        # Insert with upsert to avoid duplicates
        for doc in documents:
            collection.replace_one(
                {'parcel_id': doc['parcel_id']},
                doc,
                upsert=True
            )

        print(f"Saved {len(documents)} building records to MongoDB")
    else:
        print("No valid building data to save")

def main():
    """Main function for improved scraping"""

    # Connect to MongoDB
    db = connect_to_mongodb()
    if db is None:
        print("Failed to connect to MongoDB")
        return

    # Get sample parcels (start with 50)
    parcels = get_sample_parcels(db, limit=50)

    if not parcels:
        print("No industrial parcels found")
        return

    print(f"\n=== Starting Building Data Scraping for {len(parcels)} Properties ===")

    results = []
    sqft_distribution = []

    for i, parcel in enumerate(parcels, 1):
        parcel_id = parcel.get('parcel_id')
        print(f"\n{i}/{len(parcels)}. Scraping: {parcel_id}")
        print(f"   Type: {parcel.get('property_use')}")
        print(f"   Address: {parcel.get('address')}")

        # Scrape the data
        scraped = scrape_property_data(parcel_id)
        scraped['original_data'] = parcel

        # Print results
        if scraped['scrape_success']:
            sqft = scraped.get('building_sqft')
            if sqft:
                print(f"   [SUCCESS] Building: {sqft:,} sqft")
                sqft_distribution.append(sqft)

                # Check flex requirements
                if sqft >= 20000:
                    print(f"   [FLEX] Meets 20K+ sqft requirement!")
                elif sqft >= 10000:
                    print(f"   [GOOD] Large building (10K+ sqft)")
                elif sqft >= 5000:
                    print(f"   [OK] Medium building (5K+ sqft)")

                if scraped.get('warehouse_area'):
                    print(f"   [INFO] Warehouse: {scraped['warehouse_area']:,} sqft")
                if scraped.get('office_area'):
                    print(f"   [INFO] Office: {scraped['office_area']:,} sqft")
                if scraped.get('year_built'):
                    print(f"   [INFO] Built: {scraped['year_built']}")
            else:
                print(f"   [SUCCESS] Scraped but no sqft found")
        else:
            print(f"   [FAILED] {scraped['scrape_message']}")

        results.append(scraped)

        # Be polite - 2 second delay between requests
        if i < len(parcels):
            time.sleep(2)

    # Save all results
    output_file = f'scraped_building_data_{len(parcels)}_properties.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save to MongoDB
    save_to_mongodb(db, results)

    # Analysis
    print(f"\n=== SCRAPING ANALYSIS ===")
    successful = sum(1 for r in results if r['scrape_success'])
    with_sqft = sum(1 for r in results if r.get('building_sqft') and r['building_sqft'] > 0)

    print(f"Total parcels scraped: {len(results)}")
    print(f"Successfully scraped: {successful}")
    print(f"With building sqft: {with_sqft}")
    print(f"Success rate: {successful/len(results)*100:.1f}%")

    if sqft_distribution:
        sqft_distribution.sort(reverse=True)
        print(f"\nBuilding Size Distribution:")
        print(f"  Largest: {max(sqft_distribution):,} sqft")
        print(f"  Smallest: {min(sqft_distribution):,} sqft")
        print(f"  Average: {sum(sqft_distribution)//len(sqft_distribution):,} sqft")
        print(f"  Median: {sqft_distribution[len(sqft_distribution)//2]:,} sqft")

        # Flex analysis
        flex_candidates = [s for s in sqft_distribution if s >= 20000]
        large_buildings = [s for s in sqft_distribution if s >= 10000]
        medium_buildings = [s for s in sqft_distribution if 5000 <= s < 10000]

        print(f"\nFlex Property Analysis:")
        print(f"  >=20,000 sqft (flex requirement): {len(flex_candidates)} properties")
        print(f"  >=10,000 sqft (large): {len(large_buildings)} properties")
        print(f"  5,000-9,999 sqft (medium): {len(medium_buildings)} properties")

        if flex_candidates:
            print(f"\nTop Flex Candidates (>=20K sqft):")
            top_flex = [r for r in results if r.get('building_sqft', 0) >= 20000]
            for candidate in sorted(top_flex, key=lambda x: x.get('building_sqft', 0), reverse=True)[:10]:
                sqft = candidate['building_sqft']
                parcel_data = candidate['original_data']
                print(f"  - {sqft:,} sqft | {parcel_data.get('property_use')} | {parcel_data.get('address')}")

    print(f"\nResults saved to: {output_file}")
    print(f"Building data saved to MongoDB collection: building_data")

if __name__ == "__main__":
    main()