#!/usr/bin/env python3
"""
Debug script to explore Palm Beach County GIS data
"""

import asyncio
import aiohttp
import json

async def explore_gis_service():
    """Explore the GIS service to understand the data structure"""
    
    base_url = "https://services2.arcgis.com/HsXtOCMp1Nis1Ogr/arcgis/rest/services"
    
    async with aiohttp.ClientSession() as session:
        
        # 1. Get service info
        print("=== EXPLORING PALM BEACH COUNTY GIS SERVICES ===\n")
        
        # Test zoning service
        zoning_url = f"{base_url}/Zoning/FeatureServer/0"
        
        print("1. Getting zoning service metadata...")
        async with session.get(f"{zoning_url}?f=json") as response:
            if response.status == 200:
                data = await response.json()
                print(f"   Service Name: {data.get('name', 'Unknown')}")
                print(f"   Type: {data.get('type', 'Unknown')}")
                print(f"   Geometry Type: {data.get('geometryType', 'Unknown')}")
                
                # Check fields
                fields = data.get('fields', [])
                print(f"\n   Available Fields ({len(fields)}):")
                for field in fields[:10]:  # Show first 10 fields
                    print(f"     - {field.get('name', 'Unknown')} ({field.get('type', 'Unknown')})")
                if len(fields) > 10:
                    print(f"     ... and {len(fields) - 10} more fields")
            else:
                print(f"   Error: {response.status}")
        
        # 2. Get sample records to see actual data
        print("\n2. Getting sample zoning records...")
        
        query_url = f"{zoning_url}/query"
        params = {
            'where': '1=1',
            'outFields': 'OBJECTID,ZONE_,ZONING,ACRES,PCN',
            'resultRecordCount': 10,
            'f': 'json'
        }
        
        async with session.get(query_url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                features = data.get('features', [])
                
                print(f"   Found {len(features)} sample records:")
                for i, feature in enumerate(features):
                    attrs = feature.get('attributes', {})
                    print(f"     {i+1}. Zone: {attrs.get('ZONE_', 'N/A')}, "
                          f"Zoning: {attrs.get('ZONING', 'N/A')}, "
                          f"Acres: {attrs.get('ACRES', 'N/A')}, "
                          f"PCN: {attrs.get('PCN', 'N/A')}")
            else:
                print(f"   Error: {response.status}")
        
        # 3. Get unique zoning codes
        print("\n3. Getting unique zoning codes...")
        
        # Get distinct values for ZONE_ field
        params = {
            'where': '1=1',
            'returnDistinctValues': 'true',
            'outFields': 'ZONE_',
            'f': 'json'
        }
        
        async with session.get(query_url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                features = data.get('features', [])
                
                zones = set()
                for feature in features:
                    zone = feature.get('attributes', {}).get('ZONE_')
                    if zone:
                        zones.add(zone)
                
                zones = sorted(list(zones))
                print(f"   Found {len(zones)} unique zoning codes:")
                
                # Look for potential industrial zones
                industrial_zones = []
                for zone in zones:
                    zone_upper = zone.upper()
                    if any(ind in zone_upper for ind in ['I', 'IND', 'MAN', 'COMM']):
                        industrial_zones.append(zone)
                
                print(f"   All zones: {zones[:20]}{'...' if len(zones) > 20 else ''}")
                print(f"   Potential industrial/commercial zones: {industrial_zones}")
                
            else:
                print(f"   Error: {response.status}")
        
        # 4. Test count with our current query
        print("\n4. Testing current industrial zone query...")
        
        industrial_zones = ['IL', 'IG', 'IP', 'IND', 'M-1', 'M-2', 'MUPD', 'PIPD', 'AGR/IND']
        zones_str = "','".join(industrial_zones)
        where_clause = f"ZONE_ IN ('{zones_str}')"
        
        params = {
            'where': where_clause,
            'returnCountOnly': 'true',
            'f': 'json'
        }
        
        async with session.get(query_url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                count = data.get('count', 0)
                print(f"   Query: {where_clause}")
                print(f"   Count: {count} records")
                
                if count == 0:
                    print("   [ERROR] No records found with current zone codes")
                else:
                    print(f"   [SUCCESS] Found {count} industrial parcels")
            else:
                print(f"   Error: {response.status}")

async def main():
    await explore_gis_service()

if __name__ == "__main__":
    asyncio.run(main())