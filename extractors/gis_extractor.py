"""
Palm Beach County GIS Data Extractor
Extracts zoning, parcels, and building footprint data
"""
import asyncio
import aiohttp
import logging
from typing import List, Dict, Optional, Set
from datetime import datetime
import json
import os
from dataclasses import dataclass

@dataclass
class GISConfig:
    """Configuration for Palm Beach County GIS services"""
    BASE_URL = "https://gis.pbcgov.org/arcgis/rest/services"
    
    # Service endpoints - Using actual Palm Beach County GIS
    SERVICES = {
        'zoning': f"{BASE_URL}/PZB/ZONING_APPS_SHAPE/FeatureServer/0",
        'parcels': f"{BASE_URL}/Parcels/PARCELS/FeatureServer/0", 
        'property_use': f"{BASE_URL}/Parcels/PROPERTY_USE/FeatureServer/0",
        'parcel_details': f"{BASE_URL}/Parcels/PARCEL_INFO/FeatureServer/4",
        'sales_data': f"{BASE_URL}/Parcels/PARCEL_SALES/FeatureServer/30"
    }
    
    # Industrial and flex-compatible property use codes for Palm Beach County
    INDUSTRIAL_PROPERTY_USES = [
        'HEAVY MFG',           # Heavy Manufacturing (5 records)
        'VACANT INDUSTRIAL',   # Vacant Industrial land (339 records)  
        'WAREH/DIST TERM',     # Warehouse/Distribution Terminal (2,520 records)
        'OPEN STORAGE',        # Open Storage (278 records)
        'WORKING WATERFRONT'   # Working waterfront (62 records)
    ]
    
    # Flex-compatible commercial property use codes
    FLEX_COMMERCIAL_USES = [
        'STORES',              # Retail stores (1,777 records)
        'SHOPPING CENTER CMMITY', # Shopping centers (333 records)
        'SMALL DISCOUNT STORE < 25000 SF', # Small stores (120 records)
    ]

class GISExtractor:
    """Extract data from Palm Beach County GIS services"""
    
    def __init__(self, rate_limit: int = 5):
        self.config = GISConfig()
        self.rate_limit = rate_limit
        self.semaphore = asyncio.Semaphore(rate_limit)
        self.logger = logging.getLogger(__name__)
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch(self, url: str, params: Dict) -> Optional[Dict]:
        """Fetch data from GIS service with rate limiting"""
        async with self.semaphore:
            try:
                async with self.session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        self.logger.error(f"GIS request failed: {response.status}")
                        return None
            except asyncio.TimeoutError:
                self.logger.error(f"Request timeout: {url}")
                return None
            except Exception as e:
                self.logger.error(f"Request error: {e}")
                return None
    
    async def get_record_count(self, service: str, where_clause: str = "1=1") -> int:
        """Get total record count for a query"""
        url = f"{self.config.SERVICES[service]}/query"
        params = {
            'where': where_clause,
            'returnCountOnly': 'true',
            'f': 'json'
        }
        
        result = await self.fetch(url, params)
        return result.get('count', 0) if result else 0
    
    async def extract_industrial_parcels(self) -> List[Dict]:
        """Extract all industrial and flex-compatible parcels by property use"""
        
        all_parcels = []
        url = f"{self.config.SERVICES['property_use']}/query"
        
        # Build WHERE clause for industrial and flex commercial property uses
        all_target_uses = self.config.INDUSTRIAL_PROPERTY_USES + self.config.FLEX_COMMERCIAL_USES
        uses_str = "','".join(all_target_uses)
        where_clause = f"PROPERTY_USE IN ('{uses_str}')"
        
        # Get total count first
        total_count = await self.get_record_count('property_use', where_clause)
        self.logger.info(f"Found {total_count} industrial/flex parcels to extract")
        
        # Extract in batches (ArcGIS limit is usually 1000-2000 records)
        batch_size = 1000
        offset = 0
        
        while offset < total_count:
            params = {
                'where': where_clause,
                'outFields': '*',  # Get all fields
                'f': 'json',
                'resultOffset': offset,
                'resultRecordCount': batch_size,
                'orderByFields': 'OBJECTID'
            }
            
            result = await self.fetch(url, params)
            
            if result and 'features' in result:
                features = result['features']
                
                # Process each feature
                for feature in features:
                    parcel_data = feature.get('attributes', {})
                    
                    # Add geometry if available
                    if 'geometry' in feature:
                        parcel_data['geometry'] = feature['geometry']
                    
                    # Standardize field names for our database schema
                    processed_parcel = {
                        'parcel_id': parcel_data.get('PARID', ''),
                        'property_use': parcel_data.get('PROPERTY_USE', ''),
                        'acres': parcel_data.get('ACRES', 0),
                        'owner_name': parcel_data.get('OWNER_NAME1', ''),
                        'street_address': parcel_data.get('SITE_ADDR_STR', ''),
                        'municipality': parcel_data.get('MUNICIPALITY', ''),
                        'market_value': parcel_data.get('TOTAL_MARKET', 0),
                        'assessed_value': parcel_data.get('ASSESSED_VAL', 0),
                        'sale_date': parcel_data.get('SALE_DATE', ''),
                        'sale_price': parcel_data.get('PRICE', 0),
                        'subdivision': parcel_data.get('SUBDIV_NAME', ''),
                        'shape_area': parcel_data.get('Shape__Area', 0),
                        'objectid': parcel_data.get('OBJECTID'),
                        'source': 'pbc_gis_property_use',
                        'extracted_at': datetime.utcnow().isoformat(),
                        'raw_attributes': parcel_data
                    }
                    
                    all_parcels.append(processed_parcel)
                
                self.logger.info(f"Extracted {len(features)} parcels (offset: {offset})")
            
            offset += batch_size
            await asyncio.sleep(0.5)  # Rate limiting between batches
        
        return all_parcels
    
    async def extract_parcel_details(self, parcel_ids: List[str], 
                                   batch_size: int = 50) -> List[Dict]:
        """Extract detailed parcel information for specific parcels"""
        
        url = f"{self.config.SERVICES['parcels']}/query"
        detailed_parcels = []
        
        # Process in batches
        for i in range(0, len(parcel_ids), batch_size):
            batch = parcel_ids[i:i + batch_size]
            
            # Build WHERE clause
            pcn_list = "','".join(batch)
            where_clause = f"PCN IN ('{pcn_list}')"
            
            params = {
                'where': where_clause,
                'outFields': '*',
                'f': 'json',
                'returnGeometry': 'true'
            }
            
            result = await self.fetch(url, params)
            
            if result and 'features' in result:
                for feature in result['features']:
                    attrs = feature.get('attributes', {})
                    
                    parcel_detail = {
                        'parcel_id': attrs.get('PCN'),
                        'owner_name': attrs.get('OWNER_NAME'),
                        'mailing_address': attrs.get('MAILING_ADDRESS'),
                        'site_address': attrs.get('SITE_ADDRESS'),
                        'municipality': attrs.get('MUNICIPALITY'),
                        'subdivision': attrs.get('SUBDIVISION'),
                        'legal_desc': attrs.get('LEGAL_DESC'),
                        'land_use': attrs.get('LAND_USE'),
                        'geometry': feature.get('geometry'),
                        'source': 'pbc_gis_parcels',
                        'extracted_at': datetime.utcnow().isoformat()
                    }
                    
                    detailed_parcels.append(parcel_detail)
            
            await asyncio.sleep(0.5)  # Rate limiting
            
        return detailed_parcels
    
    async def extract_building_footprints(self, parcel_ids: List[str]) -> Dict[str, List]:
        """Extract building footprints for parcels"""
        
        url = f"{self.config.SERVICES['building_footprints']}/query"
        footprints_by_parcel = {}
        
        # Process in smaller batches for building data
        batch_size = 25
        
        for i in range(0, len(parcel_ids), batch_size):
            batch = parcel_ids[i:i + batch_size]
            pcn_list = "','".join(batch)
            
            params = {
                'where': f"PCN IN ('{pcn_list}')",
                'outFields': 'PCN,BLDG_ID,BLDG_TYPE,YEAR_BUILT,TOTAL_SQFT,HEIGHT',
                'f': 'json',
                'returnGeometry': 'true'
            }
            
            result = await self.fetch(url, params)
            
            if result and 'features' in result:
                for feature in result['features']:
                    attrs = feature.get('attributes', {})
                    pcn = attrs.get('PCN')
                    
                    if pcn:
                        if pcn not in footprints_by_parcel:
                            footprints_by_parcel[pcn] = []
                        
                        footprints_by_parcel[pcn].append({
                            'building_id': attrs.get('BLDG_ID'),
                            'building_type': attrs.get('BLDG_TYPE'),
                            'year_built': attrs.get('YEAR_BUILT'),
                            'total_sqft': attrs.get('TOTAL_SQFT'),
                            'height': attrs.get('HEIGHT'),
                            'geometry': feature.get('geometry')
                        })
            
            await asyncio.sleep(0.5)
        
        return footprints_by_parcel
    
    async def extract_flex_candidates(self) -> List[Dict]:
        """
        Extract potential flex properties including industrial 
        and compatible commercial property uses
        """
        
        # Get industrial and commercial parcels in one query
        # (the extract_industrial_parcels method now handles both)
        all_parcels = await self.extract_industrial_parcels()
        
        self.logger.info(f"Total flex candidates: {len(all_parcels)}")
        
        return all_parcels
    
    def save_to_file(self, data: List[Dict], filename: str):
        """Save extracted data to JSON file for backup"""
        
        output_dir = 'data/raw'
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Saved {len(data)} records to {filepath}")
        return filepath

# Standalone function for testing
async def test_extractor():
    """Test the GIS extractor"""
    
    logging.basicConfig(level=logging.INFO)
    
    async with GISExtractor() as extractor:
        # Test getting industrial parcels
        print("Extracting industrial parcels...")
        parcels = await extractor.extract_flex_candidates()
        
        print(f"Found {len(parcels)} potential flex parcels")
        
        if parcels:
            # Save to file
            filepath = extractor.save_to_file(parcels, 'industrial_parcels')
            print(f"Data saved to {filepath}")
            
            # Show sample
            print("\nSample parcel:")
            print(json.dumps(parcels[0], indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(test_extractor())
