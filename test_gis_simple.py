#!/usr/bin/env python3
"""
Simple test to check Palm Beach County GIS accessibility
"""

import requests
import json

def test_gis_access():
    """Test basic GIS service access"""
    
    print("Testing Palm Beach County GIS Access")
    print("=" * 40)
    
    # Try different base URLs
    base_urls = [
        "https://services2.arcgis.com/HsXtOCMp1Nis1Ogr/arcgis/rest/services",
        "https://gis.pbcgov.org/arcgis/rest/services",
        "https://maps.pbcgov.org/arcgis/rest/services"
    ]
    
    for base_url in base_urls:
        print(f"\nTesting: {base_url}")
        try:
            response = requests.get(f"{base_url}?f=json", timeout=10)
            if response.status_code == 200:
                data = response.json()
                services = data.get('services', [])
                print(f"  [SUCCESS] Found {len(services)} services")
                
                # Look for zoning-related services
                zoning_services = []
                for service in services:
                    name = service.get('name', '').lower()
                    if 'zon' in name or 'parcel' in name or 'land' in name:
                        zoning_services.append(service.get('name'))
                
                if zoning_services:
                    print(f"  Relevant services: {zoning_services[:5]}")
                else:
                    print("  No zoning/parcel services found")
                    
            else:
                print(f"  [ERROR] Status: {response.status_code}")
        except Exception as e:
            print(f"  [ERROR] {e}")
    
    # Test specific zoning service
    print(f"\nTesting specific zoning service...")
    zoning_url = "https://services2.arcgis.com/HsXtOCMp1Nis1Ogr/arcgis/rest/services/Zoning/FeatureServer/0"
    
    try:
        response = requests.get(f"{zoning_url}?f=json", timeout=10)
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Service exists: {data.get('name', 'Unknown')}")
            
            # Test a simple query
            query_url = f"{zoning_url}/query"
            params = {
                'where': '1=1',
                'returnCountOnly': 'true',
                'f': 'json'
            }
            
            response = requests.get(query_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                count = data.get('count', 0)
                print(f"  Total records: {count}")
            else:
                print(f"  Query failed: {response.status_code}")
        else:
            print(f"  Service not accessible")
            
    except Exception as e:
        print(f"  [ERROR] {e}")

if __name__ == "__main__":
    test_gis_access()