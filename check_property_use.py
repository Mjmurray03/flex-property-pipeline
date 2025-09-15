#!/usr/bin/env python3
"""
Check actual property use codes in Palm Beach County
"""

import requests
import json

def check_property_use_codes():
    """Check what property use codes actually exist"""
    
    print("Checking Palm Beach County Property Use Codes")
    print("=" * 50)
    
    property_use_url = "https://gis.pbcgov.org/arcgis/rest/services/Parcels/PROPERTY_USE/FeatureServer/0"
    query_url = f"{property_use_url}/query"
    
    # Get a larger sample to see actual property use values
    params = {
        'where': '1=1',
        'outFields': 'PARID,PROPERTY_USE,ACRES',
        'resultRecordCount': 100,
        'f': 'json'
    }
    
    try:
        response = requests.get(query_url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            features = data.get('features', [])
            
            print(f"Found {len(features)} sample records")
            
            # Collect unique property use codes
            property_use_codes = set()
            sample_records = []
            
            for feature in features:
                attrs = feature.get('attributes', {})
                prop_use = attrs.get('PROPERTY_USE')
                acres = attrs.get('ACRES', 0)
                
                if prop_use:
                    property_use_codes.add(prop_use)
                    sample_records.append({
                        'parid': attrs.get('PARID'),
                        'property_use': prop_use,
                        'acres': acres
                    })
            
            print(f"\nUnique Property Use Codes found: {len(property_use_codes)}")
            sorted_codes = sorted(list(property_use_codes))
            for code in sorted_codes:
                print(f"  - {code}")
            
            print(f"\nSample records:")
            for i, record in enumerate(sample_records[:10]):
                print(f"  {i+1}. Use: {record['property_use']}, Acres: {record['acres']}")
            
            # Now search for specific codes that might be industrial/commercial
            print(f"\nTesting counts for different property use codes...")
            
            for code in sorted_codes:
                params = {
                    'where': f"PROPERTY_USE = '{code}'",
                    'returnCountOnly': 'true',
                    'f': 'json'
                }
                
                try:
                    count_response = requests.get(query_url, params=params, timeout=10)
                    if count_response.status_code == 200:
                        count_data = count_response.json()
                        count = count_data.get('count', 0)
                        print(f"  {code}: {count} records")
                except:
                    print(f"  {code}: Could not get count")
                    
        else:
            print(f"Error: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_property_use_codes()