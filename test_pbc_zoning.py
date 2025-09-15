#!/usr/bin/env python3
"""
Test Palm Beach County zoning codes to find industrial properties
"""

import requests
import json

def test_pbc_zoning_codes():
    """Test actual zoning codes in Palm Beach County system"""
    
    print("Testing Palm Beach County Zoning Codes")
    print("=" * 45)
    
    # Test the zoning applications service
    zoning_url = "https://gis.pbcgov.org/arcgis/rest/services/PZB/ZONING_APPS_SHAPE/FeatureServer/0"
    
    print("1. Testing ZONING_APPS_SHAPE service...")
    
    # Get sample records first
    query_url = f"{zoning_url}/query"
    params = {
        'where': '1=1',
        'outFields': '*',
        'resultRecordCount': 10,
        'f': 'json'
    }
    
    try:
        response = requests.get(query_url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            features = data.get('features', [])
            
            print(f"   Found {len(features)} sample records")
            
            if features:
                # Show field names
                sample_attrs = features[0].get('attributes', {})
                print(f"   Available fields: {list(sample_attrs.keys())}")
                
                # Show sample data
                print(f"   Sample records:")
                for i, feature in enumerate(features[:5]):
                    attrs = feature.get('attributes', {})
                    print(f"     {i+1}. {dict(list(attrs.items())[:5])}...")
        else:
            print(f"   Error: {response.status_code}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test property use service for better zoning info
    print(f"\n2. Testing PROPERTY_USE service...")
    
    property_use_url = "https://gis.pbcgov.org/arcgis/rest/services/Parcels/PROPERTY_USE/FeatureServer/0"
    query_url = f"{property_use_url}/query"
    
    params = {
        'where': '1=1',
        'outFields': '*',
        'resultRecordCount': 10,
        'f': 'json'
    }
    
    try:
        response = requests.get(query_url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            features = data.get('features', [])
            
            print(f"   Found {len(features)} sample records")
            
            if features:
                # Show field names
                sample_attrs = features[0].get('attributes', {})
                print(f"   Available fields: {list(sample_attrs.keys())}")
                
                # Show sample data
                print(f"   Sample records:")
                for i, feature in enumerate(features[:5]):
                    attrs = feature.get('attributes', {})
                    print(f"     {i+1}. {dict(list(attrs.items())[:5])}...")
        else:
            print(f"   Error: {response.status_code}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Try to find industrial/commercial property types
    print(f"\n3. Looking for industrial/commercial properties...")
    
    # Common property use codes for industrial/commercial
    industrial_terms = ['IND', 'COMM', 'WAREHOUSE', 'MANUFACT', 'INDUSTRIAL', 'COMMERCIAL']
    
    for term in industrial_terms:
        print(f"\n   Searching for '{term}'...")
        
        # Search in property use
        params = {
            'where': f"UPPER(PROPERTY_USE_DESC) LIKE UPPER('%{term}%')",
            'returnCountOnly': 'true',
            'f': 'json'
        }
        
        try:
            response = requests.get(f"{property_use_url}/query", params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                count = data.get('count', 0)
                print(f"     Property Use - {term}: {count} records")
                
                if count > 0 and count < 50:  # Get sample if manageable number
                    sample_params = {
                        'where': f"UPPER(PROPERTY_USE_DESC) LIKE UPPER('%{term}%')",
                        'outFields': 'PCN,PROPERTY_USE_DESC',
                        'resultRecordCount': 5,
                        'f': 'json'
                    }
                    
                    sample_response = requests.get(f"{property_use_url}/query", params=sample_params, timeout=10)
                    if sample_response.status_code == 200:
                        sample_data = sample_response.json()
                        sample_features = sample_data.get('features', [])
                        for sample in sample_features:
                            attrs = sample.get('attributes', {})
                            print(f"       - {attrs.get('PROPERTY_USE_DESC', 'N/A')}")
            else:
                print(f"     Error searching {term}: {response.status_code}")
        except Exception as e:
            print(f"     Error searching {term}: {e}")

    # Get unique property use descriptions to see what's available
    print(f"\n4. Getting unique property use descriptions...")
    
    params = {
        'where': '1=1',
        'returnDistinctValues': 'true',
        'outFields': 'PROPERTY_USE_DESC',
        'f': 'json'
    }
    
    try:
        response = requests.get(f"{property_use_url}/query", params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            features = data.get('features', [])
            
            use_codes = set()
            for feature in features:
                use_desc = feature.get('attributes', {}).get('PROPERTY_USE_DESC')
                if use_desc:
                    use_codes.add(use_desc)
            
            use_codes = sorted(list(use_codes))
            print(f"   Found {len(use_codes)} unique property use descriptions")
            
            # Look for industrial/commercial related codes
            relevant_codes = []
            for code in use_codes:
                code_upper = code.upper()
                if any(term in code_upper for term in ['IND', 'COMM', 'WAREHOUSE', 'MANUFACT', 'OFFICE', 'RETAIL']):
                    relevant_codes.append(code)
            
            print(f"   Industrial/Commercial related codes ({len(relevant_codes)}):")
            for code in relevant_codes[:20]:  # Show first 20
                print(f"     - {code}")
            
            if len(relevant_codes) > 20:
                print(f"     ... and {len(relevant_codes) - 20} more")
                
        else:
            print(f"   Error: {response.status_code}")
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    test_pbc_zoning_codes()