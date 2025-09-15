#!/usr/bin/env python3
"""
Scrape building data from Palm Beach County Property Appraiser website
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

def get_sample_parcels(db, limit=5):
    """Get sample warehouse parcel IDs from MongoDB"""
    print("\n=== Getting Sample Warehouse Parcels ===")

    # Query for warehouse properties
    pipeline = [
        {'$match': {'property_use': {'$regex': 'WAREH'}}},
        {'$limit': limit},
        {'$project': {
            'parcel_id': 1,
            'property_use': 1,
            'address': {'$ifNull': ['$street_address', '$address', 'N/A']},
            'market_value': 1,
            'acres': 1
        }}
    ]

    parcels = list(db.zoning_data.aggregate(pipeline))

    print(f"Found {len(parcels)} warehouse parcels")
    for i, parcel in enumerate(parcels, 1):
        print(f"\n{i}. Parcel ID: {parcel.get('parcel_id')}")
        print(f"   Type: {parcel.get('property_use')}")
        print(f"   Address: {parcel.get('address')}")
        print(f"   Value: ${parcel.get('market_value', 0):,.0f}")
        print(f"   Acres: {parcel.get('acres', 0):.2f}")

    return parcels

def scrape_property_data(parcel_id):
    """Scrape building data from Property Appraiser website"""

    # Clean parcel ID (remove special characters if any)
    clean_parcel_id = parcel_id.replace('-', '').replace(' ', '')

    # Try different URL patterns - using the working RenderPrintSum pattern
    urls_to_try = [
        f"https://pbcpao.gov/Property/RenderPrintSum?parcelId={clean_parcel_id}&flag=ALL",
        f"https://pbcpao.gov/Property/RenderPrintSum?parcelId={parcel_id}&flag=ALL",
        f"https://www.pbcpao.gov/Property/RenderPrintSum?parcelId={clean_parcel_id}&flag=ALL",
        f"https://www.pbcpao.gov/Property/RenderPrintSum?parcelId={parcel_id}&flag=ALL"
    ]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    scraped_data = {
        'parcel_id': parcel_id,
        'building_sqft': None,
        'total_living_area': None,
        'heated_area': None,
        'gross_area': None,
        'effective_area': None,
        'building_count': 0,
        'year_built': None,
        'scrape_success': False,
        'scrape_message': '',
        'raw_areas': {}
    }

    for url in urls_to_try:
        try:
            print(f"\n   Trying URL: {url}")
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Look for various patterns of square footage data
                # Common patterns in property appraiser sites

                # Pattern 1: Look for tables with building information
                tables = soup.find_all('table')
                for table in tables:
                    # Check if this is a building information table
                    text = table.get_text().lower()
                    if any(term in text for term in ['building', 'square', 'area', 'sqft', 'sq ft', 'living']):
                        rows = table.find_all('tr')
                        for row in rows:
                            cells = row.find_all(['td', 'th'])
                            if len(cells) >= 2:
                                label = cells[0].get_text().strip().lower()
                                value = cells[1].get_text().strip()

                                # Extract numeric value
                                numeric_value = re.findall(r'[\d,]+', value)
                                if numeric_value:
                                    numeric_value = int(numeric_value[0].replace(',', ''))

                                    # Store in raw areas for debugging
                                    scraped_data['raw_areas'][label] = numeric_value

                                    # Map to our standard fields
                                    if 'total' in label and 'area' in label:
                                        scraped_data['total_living_area'] = numeric_value
                                    elif 'living' in label and 'area' in label:
                                        scraped_data['total_living_area'] = numeric_value
                                    elif 'heated' in label:
                                        scraped_data['heated_area'] = numeric_value
                                    elif 'gross' in label:
                                        scraped_data['gross_area'] = numeric_value
                                    elif 'effective' in label:
                                        scraped_data['effective_area'] = numeric_value
                                    elif 'building' in label and ('sqft' in label or 'square' in label):
                                        scraped_data['building_sqft'] = numeric_value
                                    elif 'year' in label and 'built' in label:
                                        try:
                                            scraped_data['year_built'] = int(value)
                                        except:
                                            pass

                # Pattern 2: Look for divs/spans with class names containing area/sqft
                area_elements = soup.find_all(['div', 'span'], class_=re.compile(r'(area|sqft|square)', re.I))
                for elem in area_elements:
                    text = elem.get_text().strip()
                    numbers = re.findall(r'[\d,]+', text)
                    if numbers:
                        value = int(numbers[0].replace(',', ''))
                        label = elem.get('class', [''])[0] if elem.get('class') else 'unknown'
                        scraped_data['raw_areas'][label] = value

                # Pattern 3: Look for specific text patterns for PBCPAO
                page_text = soup.get_text()

                # Specific patterns for Palm Beach County Property Appraiser
                pbcpao_patterns = [
                    r'Total Square Footage[:\s]+([0-9,]+)\s*sq\s*ft',
                    r'Area Under Air[:\s]+([0-9,]+)\s*sq\s*ft',
                    r'Base Area[:\s]+([0-9,]+)\s*sq\s*ft',
                    r'Building\s+Area[:\s]+([0-9,]+)',
                    r'Total\s+Area[:\s]+([0-9,]+)',
                    r'Living\s+Area[:\s]+([0-9,]+)',
                    r'Heated\s+Area[:\s]+([0-9,]+)',
                    r'Gross\s+Area[:\s]+([0-9,]+)',
                ]

                for pattern in pbcpao_patterns:
                    matches = re.findall(pattern, page_text, re.IGNORECASE)
                    if matches:
                        for match in matches:
                            try:
                                value = int(match.replace(',', ''))
                                if 100 < value < 1000000:  # Reasonable range for sqft
                                    # Store the area type based on pattern
                                    if 'Total Square Footage' in pattern:
                                        scraped_data['building_sqft'] = value
                                        scraped_data['raw_areas']['total_sqft'] = value
                                    elif 'Area Under Air' in pattern:
                                        scraped_data['heated_area'] = value
                                        scraped_data['raw_areas']['area_under_air'] = value
                                    elif 'Base Area' in pattern:
                                        scraped_data['total_living_area'] = value
                                        scraped_data['raw_areas']['base_area'] = value
                                    else:
                                        scraped_data['raw_areas'][f'pattern_{pattern[:20]}'] = value
                            except ValueError:
                                continue

                # Look for Year Built
                year_patterns = [
                    r'Year Built[:\s]+([0-9]{4})',
                    r'Built[:\s]+([0-9]{4})',
                ]

                for pattern in year_patterns:
                    matches = re.findall(pattern, page_text, re.IGNORECASE)
                    if matches:
                        try:
                            year = int(matches[0])
                            if 1800 <= year <= 2025:
                                scraped_data['year_built'] = year
                                break
                        except ValueError:
                            continue

                # Check if we found any data
                if scraped_data['raw_areas'] or scraped_data['building_sqft']:
                    scraped_data['scrape_success'] = True
                    scraped_data['scrape_message'] = f"Successfully scraped from {url}"

                    # If we have any area data but no building_sqft, use the largest area
                    if not scraped_data['building_sqft']:
                        area_values = [
                            scraped_data['total_living_area'],
                            scraped_data['heated_area'],
                            scraped_data['gross_area'],
                            scraped_data['effective_area']
                        ]
                        valid_areas = [a for a in area_values if a and a > 0]
                        if valid_areas:
                            scraped_data['building_sqft'] = max(valid_areas)

                    print(f"   SUCCESS: Found building data!")
                    break
                else:
                    scraped_data['scrape_message'] = "Page found but no building data extracted"

            elif response.status_code == 404:
                scraped_data['scrape_message'] = f"Property page not found (404)"
            else:
                scraped_data['scrape_message'] = f"HTTP {response.status_code}"

        except requests.exceptions.Timeout:
            scraped_data['scrape_message'] = "Request timeout"
        except requests.exceptions.RequestException as e:
            scraped_data['scrape_message'] = f"Request error: {str(e)[:50]}"
        except Exception as e:
            scraped_data['scrape_message'] = f"Scraping error: {str(e)[:50]}"

    return scraped_data

def main():
    """Main function to test scraping"""

    # Connect to MongoDB
    db = connect_to_mongodb()
    if db is None:
        print("Failed to connect to MongoDB")
        return

    # Get sample parcels
    parcels = get_sample_parcels(db, limit=5)

    if not parcels:
        print("No warehouse parcels found")
        return

    print("\n=== Starting Building Data Scraping ===")

    results = []

    for i, parcel in enumerate(parcels, 1):
        parcel_id = parcel.get('parcel_id')
        print(f"\n{i}. Scraping parcel: {parcel_id}")
        print(f"   Address: {parcel.get('address')}")

        # Scrape the data
        scraped = scrape_property_data(parcel_id)

        # Add original parcel info
        scraped['original_data'] = parcel

        # Print results
        if scraped['scrape_success']:
            print(f"   [SUCCESS] Building SqFt: {scraped['building_sqft']:,}" if scraped['building_sqft'] else "   - No building sqft found")
            print(f"   [SUCCESS] Total Living Area: {scraped['total_living_area']:,}" if scraped['total_living_area'] else "")
            print(f"   [SUCCESS] Heated Area: {scraped['heated_area']:,}" if scraped['heated_area'] else "")
            print(f"   [SUCCESS] Year Built: {scraped['year_built']}" if scraped['year_built'] else "")
            if scraped['raw_areas']:
                print(f"   [SUCCESS] Raw areas found: {scraped['raw_areas']}")
        else:
            print(f"   [FAILED] {scraped['scrape_message']}")

        results.append(scraped)

        # Be polite to the server
        time.sleep(2)

    # Save results
    output_file = 'scraped_building_data_test.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n=== Scraping Complete ===")
    print(f"Results saved to: {output_file}")

    # Summary
    successful = sum(1 for r in results if r['scrape_success'])
    with_sqft = sum(1 for r in results if r.get('building_sqft'))

    print(f"\nSummary:")
    print(f"  Total parcels tested: {len(results)}")
    print(f"  Successfully scraped: {successful}")
    print(f"  With building sqft: {with_sqft}")

    if successful > 0:
        print(f"\n[SUCCESS] Scraping appears to be working!")
        print(f"  We can scale this up to scrape all {len(parcels)} industrial properties")
        print(f"  Estimated time for 3,168 properties: ~{(3168 * 3) // 60} minutes")
    else:
        print(f"\n[FAILED] Scraping did not work. May need to:")
        print(f"  1. Check if the website requires authentication")
        print(f"  2. Try different URL patterns")
        print(f"  3. Use selenium for JavaScript-rendered content")
        print(f"  4. Request data directly from the Property Appraiser")

if __name__ == "__main__":
    main()