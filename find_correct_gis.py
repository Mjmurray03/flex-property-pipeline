#!/usr/bin/env python3
"""
Find the correct Palm Beach County GIS service with zoning data
"""

import requests
import json

def explore_pbc_gis():
    """Explore Palm Beach County's actual GIS services"""
    
    print("Exploring Palm Beach County Official GIS")
    print("=" * 45)
    
    # Test the official PBC GIS
    base_url = "https://gis.pbcgov.org/arcgis/rest/services"
    
    try:
        response = requests.get(f"{base_url}?f=json", timeout=15)
        if response.status_code == 200:
            data = response.json()
            services = data.get('services', [])
            folders = data.get('folders', [])
            
            print(f"Found {len(services)} services and {len(folders)} folders")
            print(f"Services: {[s.get('name') for s in services]}")
            print(f"Folders: {folders}")
            
            # Check folders for more services
            for folder in folders:
                print(f"\nExploring folder: {folder}")
                try:
                    folder_response = requests.get(f"{base_url}/{folder}?f=json", timeout=10)
                    if folder_response.status_code == 200:
                        folder_data = folder_response.json()
                        folder_services = folder_data.get('services', [])
                        print(f"  Found {len(folder_services)} services in {folder}:")
                        
                        for service in folder_services:
                            service_name = service.get('name', '')
                            service_type = service.get('type', '')
                            print(f"    - {service_name} ({service_type})")
                            
                            # Look for zoning/parcel related services
                            if any(keyword in service_name.lower() for keyword in ['zon', 'parcel', 'land', 'property']):
                                print(f"      *** POTENTIAL MATCH: {service_name} ***")
                                
                                # Test this service
                                if service_type == 'FeatureServer':
                                    test_service_url = f"{base_url}/{service_name}/FeatureServer"
                                    try:
                                        test_response = requests.get(f"{test_service_url}?f=json", timeout=10)
                                        if test_response.status_code == 200:
                                            test_data = test_response.json()
                                            layers = test_data.get('layers', [])
                                            print(f"        Layers: {len(layers)}")
                                            
                                            for layer in layers:
                                                layer_id = layer.get('id')
                                                layer_name = layer.get('name', '')
                                                print(f"          Layer {layer_id}: {layer_name}")
                                                
                                                # Test record count
                                                layer_url = f"{test_service_url}/{layer_id}/query"
                                                count_params = {
                                                    'where': '1=1',
                                                    'returnCountOnly': 'true',
                                                    'f': 'json'
                                                }
                                                
                                                count_response = requests.get(layer_url, params=count_params, timeout=10)
                                                if count_response.status_code == 200:
                                                    count_data = count_response.json()
                                                    record_count = count_data.get('count', 0)
                                                    print(f"            Records: {record_count}")
                                    except:
                                        print(f"        Could not test service")
                except:
                    print(f"  Could not access folder: {folder}")
                    
        else:
            print(f"Error accessing PBC GIS: {response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")

    # Also test some common GIS endpoints
    print(f"\nTesting common Palm Beach County GIS endpoints...")
    
    common_endpoints = [
        "https://maps.pbcgov.org/arcgis/rest/services",
        "https://gis-public.pbcgov.org/arcgis/rest/services",
        "https://webgis.pbcgov.org/arcgis/rest/services"
    ]
    
    for endpoint in common_endpoints:
        print(f"Testing: {endpoint}")
        try:
            response = requests.get(f"{endpoint}?f=json", timeout=10)
            if response.status_code == 200:
                data = response.json()
                services = data.get('services', [])
                print(f"  [SUCCESS] {len(services)} services found")
            else:
                print(f"  [ERROR] Status: {response.status_code}")
        except Exception as e:
            print(f"  [ERROR] {e}")

if __name__ == "__main__":
    explore_pbc_gis()