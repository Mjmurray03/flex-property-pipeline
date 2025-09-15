#!/usr/bin/env python3
"""
Fix the building square footage extraction from scraped data
"""

import json
import re

def fix_scraped_data():
    """Fix the building square footage extraction from the test data"""

    with open('scraped_building_data_test.json', 'r') as f:
        data = json.load(f)

    print("=== Fixing Building Square Footage Extraction ===\n")

    for i, property_data in enumerate(data):
        parcel_id = property_data['parcel_id']
        raw_areas = property_data.get('raw_areas', {})

        print(f"{i+1}. Parcel: {parcel_id}")
        print(f"   Current building_sqft: {property_data.get('building_sqft')}")

        # Extract the correct building square footage
        correct_sqft = None

        # Look for the key with "*total square feet :"
        if '*total square feet :' in raw_areas:
            correct_sqft = raw_areas['*total square feet :']
            print(f"   Found total sqft: {correct_sqft}")

        # Also check other potential keys
        potential_keys = [
            'total square footage',
            'warehouse storage',
            'warehouse',
            'building area'
        ]

        areas_found = {}
        for key in raw_areas:
            if isinstance(raw_areas[key], (int, float)) and raw_areas[key] > 100:
                for potential in potential_keys:
                    if potential in key.lower():
                        areas_found[key] = raw_areas[key]

        if areas_found:
            print(f"   Areas found: {areas_found}")

            # Priority order for selecting the best area value
            if 'total square footage' in areas_found:
                correct_sqft = areas_found['total square footage']
            elif 'warehouse storage' in areas_found:
                correct_sqft = areas_found['warehouse storage']
            elif correct_sqft is None and areas_found:
                # Take the largest reasonable area
                correct_sqft = max(areas_found.values())

        # Update the property data
        if correct_sqft and correct_sqft > 100:
            property_data['building_sqft'] = correct_sqft
            property_data['corrected_sqft'] = True
            print(f"   CORRECTED: {correct_sqft:,} sqft")
        else:
            property_data['corrected_sqft'] = False
            print(f"   NO CORRECTION POSSIBLE")

        print()

    # Save the corrected data
    with open('scraped_building_data_corrected.json', 'w') as f:
        json.dump(data, f, indent=2, default=str)

    print("=== Summary ===")
    corrected_count = sum(1 for p in data if p.get('corrected_sqft', False))
    print(f"Properties corrected: {corrected_count}/{len(data)}")

    # Show the corrected building sizes
    print("\nCorrected Building Sizes:")
    for i, property_data in enumerate(data):
        sqft = property_data.get('building_sqft', 0)
        if sqft > 100:
            print(f"  {i+1}. {property_data['parcel_id']}: {sqft:,} sqft")

            # Check if this meets our >=20K sqft requirement
            if sqft >= 20000:
                print(f"      [SUCCESS] MEETS FLEX REQUIREMENT (>=20K sqft)")
            else:
                print(f"      [BELOW] Below flex requirement ({sqft:,} < 20,000 sqft)")

    # Check how many would meet the 20K sqft requirement
    large_enough = [p for p in data if p.get('building_sqft', 0) >= 20000]
    print(f"\nProperties meeting >=20K sqft requirement: {len(large_enough)}/{len(data)}")

    return data

if __name__ == "__main__":
    corrected_data = fix_scraped_data()