"""
Palm Beach County Property Detail Extractor - Using ArcGIS REST Services
Fetches real property data from Palm Beach County's official ArcGIS endpoints
"""
import asyncio
import aiohttp
import logging
from typing import List, Dict, Optional
from datetime import datetime
import json

class PropertyDetailExtractor:
    """Extract property details from Palm Beach County ArcGIS services"""
    
    def __init__(self, rate_limit: int = 5):
        # Correct ArcGIS endpoints from Palm Beach County
        self.base_url = "https://services1.arcgis.com/ZWOoUZbtaYePLlPw/arcgis/rest/services"
        
        self.endpoints = {
            'property_table': f"{self.base_url}/Property_Information_Table/FeatureServer/0",
            'parcels_details': f"{self.base_url}/Parcels_and_Property_Details_WebMercator/FeatureServer/0",
            'situs_addresses': f"{self.base_url}/Situs_Addresses/FeatureServer/0"
        }
        
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
    
    async def fetch(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Fetch data from ArcGIS endpoint"""
        async with self.semaphore:
            try:
                async with self.session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        self.logger.warning(f"Request failed {response.status}: {url}")
                        return None
            except asyncio.TimeoutError:
                self.logger.error(f"Request timeout: {url}")
                return None
            except Exception as e:
                self.logger.error(f"Request error: {e}")
                return None
    
    async def get_property_details(self, parcel_id: str) -> Dict:
        """Get comprehensive property details from ArcGIS"""
        
        details = {
            'parcel_id': parcel_id,
            'property_info': None,
            'building_info': None,
            'commercial_info': None,
            'sales_history': None,
            'flex_indicators': {},
            'extracted_at': datetime.utcnow().isoformat()
        }
        
        # Query parameters for ArcGIS
        params = {
            'where': f"PARCEL_NUMBER='{parcel_id}'",
            'outFields': '*',
            'f': 'json',
            'returnGeometry': 'false'
        }
        
        # Fetch from Property Information Table
        self.logger.info(f"Fetching property details for {parcel_id}")
        self.logger.info(f"Query URL: {self.endpoints['property_table']}/query")
        self.logger.info(f"Query params: {params}")
        
        result = await self.fetch(
            f"{self.endpoints['property_table']}/query",
            params
        )
        
        self.logger.info(f"Query result: {result is not None}")
        if result:
            self.logger.info(f"Features found: {len(result.get('features', []))}")
        
        if result and 'features' in result and len(result['features']) > 0:
            # Extract the first matching property
            property_data = result['features'][0].get('attributes', {})
            
            # Store raw property information
            details['property_info'] = property_data
            
            # Extract building information
            details['building_info'] = self.extract_building_info(property_data)
            
            # Extract commercial indicators
            details['commercial_info'] = self.extract_commercial_info(property_data)
            
            # Extract sales history
            details['sales_history'] = self.extract_sales_info(property_data)
            
            # Calculate flex indicators
            details['flex_indicators'] = self.calculate_flex_indicators(property_data)
            
            self.logger.info(f"Successfully retrieved details for {parcel_id}")
        else:
            self.logger.warning(f"No data found for parcel {parcel_id}")
        
        return details
    
    def extract_building_info(self, data: Dict) -> Dict:
        """Extract building information from property data"""
        
        year_built = data.get('YEAR_ADDED')
        
        # Convert year built to integer if it's valid
        if year_built:
            try:
                year_built = int(float(year_built)) if year_built != 0 else None
            except:
                year_built = None
        
        return {
            'year_built': year_built,
            'acres': data.get('ACRES', 0),
            'property_use': data.get('PROPERTY_USE', ''),
            'municipality': data.get('MUNICIPALITY', ''),
            'site_address': data.get('SITE_ADDR_STR', ''),
            'improvement_value': data.get('IMPRV_MRKT', 0),
            'land_value': data.get('LAND_MARKET', 0),
            'total_value': data.get('TOTAL_MARKET', 0),
            'assessed_value': data.get('ASSESSED_VAL', 0),
            'owner_name': data.get('OWNER_NAME1', '')
        }
    
    def extract_commercial_info(self, data: Dict) -> Dict:
        """Extract commercial property indicators"""
        
        property_use = str(data.get('PROPERTY_USE', '')).upper()
        
        # Identify commercial/industrial property types
        is_warehouse = 'WAREH' in property_use or 'DIST' in property_use
        is_manufacturing = 'MFG' in property_use or 'MANUFACT' in property_use
        is_industrial = 'INDUSTRIAL' in property_use
        is_storage = 'STORAGE' in property_use
        is_commercial = 'STORE' in property_use or 'SHOPPING' in property_use
        
        # For flex properties, we can't get exact office/warehouse split from this data
        # But we can infer from property use codes
        property_type = 'UNKNOWN'
        
        if is_warehouse:
            property_type = 'WAREHOUSE/DISTRIBUTION'
        elif is_manufacturing:
            property_type = 'MANUFACTURING'
        elif is_industrial:
            property_type = 'INDUSTRIAL'
        elif is_storage:
            property_type = 'STORAGE'
        elif is_commercial:
            property_type = 'COMMERCIAL'
        
        return {
            'property_type': property_type,
            'property_use_code': property_use,
            'is_warehouse': is_warehouse,
            'is_manufacturing': is_manufacturing,
            'is_industrial': is_industrial,
            'is_flex_compatible': is_warehouse or is_industrial or is_manufacturing,
            'acres': data.get('ACRES', 0),
            'improvement_value': data.get('IMPRV_MRKT', 0)
        }
    
    def extract_sales_info(self, data: Dict) -> List[Dict]:
        """Extract sales information"""
        
        sales = []
        
        # Get the most recent sale
        sale_date = data.get('SALE_DATE')
        sale_price = data.get('PRICE', 0)
        
        if sale_date:
            # Convert epoch timestamp to datetime if needed
            if isinstance(sale_date, (int, float)):
                try:
                    # Assuming milliseconds timestamp
                    sale_dt = datetime.fromtimestamp(sale_date / 1000)
                    sale_date_str = sale_dt.isoformat()
                except:
                    sale_date_str = str(sale_date)
            else:
                sale_date_str = str(sale_date)
            
            sales.append({
                'sale_date': sale_date_str,
                'sale_price': sale_price,
                'book': data.get('BOOK'),
                'page': data.get('PAGE')
            })
        
        return sales
    
    def calculate_flex_indicators(self, data: Dict) -> Dict:
        """Calculate flex property indicators from available data"""
        
        indicators = {
            'flex_score_adjustment': 0,
            'is_flex_compatible': False,
            'building_age': 0,
            'recent_sale': False,
            'value_ratio': 0,
            'property_type_score': 0
        }
        
        property_use = str(data.get('PROPERTY_USE', '')).upper()
        
        # Property type scoring
        if 'WAREH' in property_use or 'DIST' in property_use:
            indicators['property_type_score'] = 3
            indicators['is_flex_compatible'] = True
            indicators['flex_score_adjustment'] += 3
        elif 'INDUSTRIAL' in property_use:
            indicators['property_type_score'] = 2.5
            indicators['is_flex_compatible'] = True
            indicators['flex_score_adjustment'] += 2.5
        elif 'MFG' in property_use:
            indicators['property_type_score'] = 2
            indicators['is_flex_compatible'] = True
            indicators['flex_score_adjustment'] += 2
        elif 'STORAGE' in property_use:
            indicators['property_type_score'] = 1.5
            indicators['is_flex_compatible'] = True
            indicators['flex_score_adjustment'] += 1.5
        
        # Building age scoring
        year_built = data.get('YRBLT')
        if year_built:
            try:
                year = int(float(year_built))
                if year > 0:
                    age = datetime.now().year - year
                    indicators['building_age'] = age if age >= 0 else 0
                    
                    # Newer buildings better for flex
                    if age <= 10:
                        indicators['flex_score_adjustment'] += 2
                    elif age <= 20:
                        indicators['flex_score_adjustment'] += 1
            except:
                pass
        
        # Recent sale activity
        sale_date = data.get('SALE_DATE')
        if sale_date:
            try:
                if isinstance(sale_date, (int, float)):
                    # Milliseconds timestamp
                    sale_dt = datetime.fromtimestamp(sale_date / 1000)
                else:
                    sale_dt = datetime.fromisoformat(str(sale_date))
                
                days_ago = (datetime.now() - sale_dt).days
                if days_ago <= 730:  # Within 2 years
                    indicators['recent_sale'] = True
                    indicators['flex_score_adjustment'] += 1
            except:
                pass
        
        # Value ratio (improvement to land)
        improvement_value = data.get('IMPRV_MRKT', 0)
        land_value = data.get('LAND_MARKET', 1)
        
        if land_value > 0:
            ratio = improvement_value / land_value
            indicators['value_ratio'] = ratio if ratio >= 0 else 0
            
            # Good ratio for flex (substantial improvements)
            if ratio >= 2:
                indicators['flex_score_adjustment'] += 2
            elif ratio >= 1:
                indicators['flex_score_adjustment'] += 1
        
        # Size scoring
        acres = data.get('ACRES', 0) or 0  # Ensure not None
        if acres > 0 and 0.5 <= acres <= 10:  # Ideal flex size
            indicators['flex_score_adjustment'] += 1
        
        return indicators
    
    async def batch_extract_details(self, parcel_ids: List[str], 
                                  batch_size: int = 10) -> List[Dict]:
        """Extract details for multiple parcels in batches"""
        
        all_details = []
        total = len(parcel_ids)
        
        self.logger.info(f"Extracting details for {total} parcels from ArcGIS")
        
        for i in range(0, total, batch_size):
            batch = parcel_ids[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [self.get_property_details(pid) for pid in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and empty results
            for result in batch_results:
                if not isinstance(result, Exception) and result.get('property_info'):
                    all_details.append(result)
            
            # Progress update
            processed = min(i + batch_size, total)
            self.logger.info(f"Processed {processed}/{total} parcels ({processed*100/total:.1f}%)")
            
            # Rate limiting between batches
            await asyncio.sleep(0.5)
        
        return all_details

# Test function
async def test_extractor():
    """Test the property detail extractor with real Palm Beach parcels"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Test with actual parcel IDs from GIS data
    test_parcels = [
        "38434421020290030",  # WAREH/DIST TERM - Composite Systems Inc
        "38434421020120060",  # WAREH/DIST TERM - 1925 10th Avenue LLC
        "00404113000003010",  # HEAVY MFG - RTX Corporation
    ]
    
    async with PropertyDetailExtractor() as extractor:
        for parcel_id in test_parcels:
            print(f"\n{'='*50}")
            print(f"Testing parcel: {parcel_id}")
            print('='*50)
            
            details = await extractor.get_property_details(parcel_id)
            
            if details.get('property_info'):
                print(f"[SUCCESS] Property found!")
                print(f"Address: {details['building_info'].get('site_address')}")
                print(f"Property Use: {details['building_info'].get('property_use')}")
                print(f"Year Built: {details['building_info'].get('year_built')}")
                print(f"Total Value: ${details['building_info'].get('total_value'):,.0f}")
                print(f"Commercial Type: {details['commercial_info'].get('property_type')}")
                print(f"Flex Compatible: {details['commercial_info'].get('is_flex_compatible')}")
                print(f"Flex Score Adjustment: {details['flex_indicators'].get('flex_score_adjustment')}")
            else:
                print(f"[ERROR] No data found")

if __name__ == "__main__":
    asyncio.run(test_extractor())