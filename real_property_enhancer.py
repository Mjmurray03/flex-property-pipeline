import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import json
import os
from datetime import datetime
from database.mongodb_client import get_db_manager
import logging
import re

class RealPropertyEnhancer:
    def __init__(self):
        self.db = get_db_manager()
        self.logger = logging.getLogger(__name__)

    def get_qualified_properties(self):
        """Get the 611 properties we've already identified"""
        # First try the Excel file
        if os.path.exists('data/exports/complete_flex_properties.xlsx'):
            df = pd.read_excel('data/exports/complete_flex_properties.xlsx')
            return df.to_dict('records')

        # Fallback to MongoDB
        return list(self.db.db.building_data.find(
            {"building_sqft": {"$gte": 20000}},
            {"parcel_id": 1, "address": 1, "building_sqft": 1}
        ))

    def scrape_real_property_data(self, parcel_id, address):
        """
        GET REAL DATA FROM PALM BEACH COUNTY BULK DATA FILES
        NO FAKE DATA, NO ESTIMATES, NO SHORTCUTS
        """

        self.logger.info(f"Getting REAL data for {parcel_id} at {address}")

        # First ensure we have the CAMA data downloaded
        cama_df = self._get_cama_data()
        if cama_df is None:
            return {'parcel_id': parcel_id, 'status': 'CAMA_DOWNLOAD_FAILED', 'data_quality': 'FAILED'}

        # Search for the parcel in CAMA data
        parcel_variations = [
            str(parcel_id),
            str(parcel_id).zfill(17),
            f"{str(parcel_id)[:2]}-{str(parcel_id)[2:4]}-{str(parcel_id)[4:6]}-{str(parcel_id)[6:8]}-{str(parcel_id)[8:10]}-{str(parcel_id)[10:13]}-{str(parcel_id)[13:]}"
        ]

        for parcel_variant in parcel_variations:
            # Try different possible field names for parcel ID
            parcel_fields = ['PARID', 'PCN', 'PARCEL_ID', 'PARCEL_NUMBER']

            for field in parcel_fields:
                if field in cama_df.columns:
                    matches = cama_df[cama_df[field].astype(str) == str(parcel_variant)]
                    if not matches.empty:
                        self.logger.info(f"Found REAL data for {parcel_variant} in CAMA file")

                        result = {
                            'parcel_id': parcel_id,
                            'address': address,
                            'scrape_timestamp': datetime.now().isoformat(),
                            'data_source': 'pbcpao_cama_bulk_data'
                        }

                        # Extract REAL data from CAMA record
                        cama_record = matches.iloc[0]
                        result.update(self._extract_cama_data(cama_record))
                        result['data_quality'] = 'COMPLETE'
                        return result

        self.logger.warning(f"Parcel {parcel_id} not found in CAMA data")
        return {'parcel_id': parcel_id, 'status': 'NOT_FOUND_IN_CAMA', 'data_quality': 'FAILED'}

    def _get_cama_data(self):
        """Download and load CAMA CSV data from Palm Beach County"""
        cama_file = 'data/raw/pbcpao_cama.csv'

        # Check if we already have the file
        if os.path.exists(cama_file):
            try:
                self.logger.info(f"Loading existing CAMA file: {cama_file}")
                return pd.read_csv(cama_file, dtype=str, low_memory=False, on_bad_lines='skip')
            except Exception as e:
                self.logger.error(f"Error reading existing CAMA file: {e}")

        # Download CAMA CSV file - try the direct download URL
        CAMA_CSV_URL = "https://pbcclouddrive.pbcgov.org/invitations/?share=d7743c5acc778a9958e6&dl=0"

        try:
            self.logger.info("Downloading CAMA CSV file from Palm Beach County...")

            # Ensure directory exists
            os.makedirs('data/raw', exist_ok=True)

            response = requests.get(CAMA_CSV_URL, timeout=300, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })

            if response.status_code == 200:
                with open(cama_file, 'wb') as f:
                    f.write(response.content)

                self.logger.info(f"Downloaded CAMA file to {cama_file}")

                # Load and return the dataframe with error handling
                return pd.read_csv(cama_file, dtype=str, low_memory=False, on_bad_lines='skip')
            else:
                self.logger.error(f"Failed to download CAMA file: HTTP {response.status_code}")
                return None

        except Exception as e:
            self.logger.error(f"Error downloading CAMA file: {e}")
            return None

    def _extract_cama_data(self, cama_record):
        """Extract REAL property data from CAMA record"""
        data = {}

        # Map CAMA fields to our schema - these are the REAL field names from Palm Beach County
        cama_field_mapping = {
            'JV': 'market_value_cama',           # Just Value (Market Value)
            'LND_VAL': 'land_value_cama',        # Land Value
            'DOR_UC': 'property_use_code_cama',  # Property Use Code
            'OWNER_NAME': 'owner_name_cama',     # Owner Name
            'OWNER_ADDR1': 'owner_address_cama', # Owner Address
            'ACREAGE': 'acreage_cama',           # Acreage
            'YR_BLT': 'year_built_cama',         # Year Built
            'TOT_LVG_AREA': 'living_area_cama',  # Total Living Area
            'GROSS_AREA': 'gross_area_cama',     # Gross Area
            'EFFYR': 'effective_year_cama',      # Effective Year
            'NET_AREA': 'net_area_cama',         # Net Area
        }

        # Extract values from CAMA record
        for cama_field, our_field in cama_field_mapping.items():
            if cama_field in cama_record.index and pd.notna(cama_record[cama_field]):
                value = cama_record[cama_field]

                # Convert numeric fields
                if cama_field in ['JV', 'LND_VAL', 'ACREAGE', 'YR_BLT', 'TOT_LVG_AREA', 'GROSS_AREA', 'EFFYR', 'NET_AREA']:
                    try:
                        data[our_field] = float(value) if value and str(value).strip() != '' else None
                    except (ValueError, TypeError):
                        data[our_field] = None
                else:
                    data[our_field] = str(value).strip() if value else None

        # Calculate improvement value if we have market and land values
        if data.get('market_value_cama') and data.get('land_value_cama'):
            data['improvement_value_cama'] = data['market_value_cama'] - data['land_value_cama']

        return data

    def _extract_arcgis_data(self, feature):
        """Extract real data from ArcGIS feature attributes"""
        data = {}

        # REAL ZONING - NO FAKE DATA
        if 'ZONING' in feature:
            data['zoning_code'] = feature['ZONING']
        elif 'ZONE' in feature:
            data['zoning_code'] = feature['ZONE']

        # REAL PROPERTY VALUES
        if 'JV' in feature:  # Just Value/Market Value
            data['market_value_arcgis'] = feature['JV']
        if 'LND_VAL' in feature:  # Land Value
            data['land_value_arcgis'] = feature['LND_VAL']
        if 'JV' in feature and 'LND_VAL' in feature:
            data['improvement_value_arcgis'] = feature['JV'] - feature['LND_VAL']

        # PROPERTY USE CODE
        if 'DOR_UC' in feature:
            data['property_use_code'] = feature['DOR_UC']

        # OWNER INFORMATION
        if 'OWNER_NAME' in feature:
            data['owner_name_arcgis'] = feature['OWNER_NAME']
        if 'OWNER_ADDR' in feature:
            data['owner_address_arcgis'] = feature['OWNER_ADDR']

        # PROPERTY DETAILS
        if 'ACREAGE' in feature:
            data['acreage_arcgis'] = feature['ACREAGE']
        if 'GROSS_AREA' in feature:
            data['gross_area_sqft'] = feature['GROSS_AREA']

        # TAX INFORMATION
        if 'TAX_YEAR' in feature:
            data['tax_year'] = feature['TAX_YEAR']

        return data

    def _try_opendata_portal(self, parcel_id, address):
        """Fallback to use existing enriched data"""
        self.logger.info(f"Using existing data for {parcel_id}")

        # Get enriched data from our zoning collection which has REAL scraped data
        try:
            zoning_data = self.db.db.zoning_data.find_one({'parcel_id': str(parcel_id)})
            if zoning_data:
                result = {
                    'parcel_id': parcel_id,
                    'address': address,
                    'scrape_timestamp': datetime.now().isoformat(),
                    'data_source': 'mongodb_zoning_collection'
                }

                # Extract REAL zoning and property data
                if 'zoning_code' in zoning_data:
                    result['zoning_code'] = zoning_data['zoning_code']
                if 'market_value' in zoning_data:
                    result['market_value_real'] = zoning_data['market_value']
                if 'assessed_value' in zoning_data:
                    result['assessed_value_real'] = zoning_data['assessed_value']
                if 'property_use' in zoning_data:
                    result['property_use_real'] = zoning_data['property_use']

                result['data_quality'] = 'COMPLETE'
                return result

        except Exception as e:
            self.logger.error(f"MongoDB query failed: {e}")

        return {'parcel_id': parcel_id, 'status': 'NOT_FOUND', 'data_quality': 'FAILED'}

    def _extract_opendata(self, props):
        """Extract data from OpenData properties"""
        data = {}

        # Map OpenData fields to our schema
        field_mapping = {
            'ZONING': 'zoning_code',
            'MARKET_VAL': 'market_value_opendata',
            'LAND_VAL': 'land_value_opendata',
            'OWNER_NAME': 'owner_name_opendata',
            'USE_CODE': 'property_use_code',
            'ACREAGE': 'acreage_opendata'
        }

        for opendata_field, our_field in field_mapping.items():
            if opendata_field in props and props[opendata_field] is not None:
                data[our_field] = props[opendata_field]

        return data

    def verify_against_known_values(self):
        """Verify scraper works with known property"""
        # Test with property from screenshot: 00-43-42-30-01-003-0000
        test_parcel = '00-43-42-30-01-003-0000'
        test_address = '3608 E INDUSTRIAL WAY'

        result = self.scrape_real_property_data(test_parcel, test_address)

        # Known values from screenshot
        assert result.get('zoning_code') == 'C4', f"Wrong zoning: {result.get('zoning_code')}"
        assert abs(result.get('total_annual_tax', 0) - 47830) < 100, f"Wrong tax: {result.get('total_annual_tax')}"

        print("✓ Verification passed - scraper is getting REAL data")
        return True

    def enhance_all_properties(self):
        """Enhance all 611 properties with real data"""
        properties = self.get_qualified_properties()
        print(f"Enhancing {len(properties)} properties with REAL data...")

        enhanced = []
        batch_size = 10

        for i in range(0, len(properties), batch_size):
            batch = properties[i:i+batch_size]

            for prop in batch:
                parcel_id = prop.get('parcel_id')
                address = prop.get('address', '')

                # Scrape real data
                enhanced_data = self.scrape_real_property_data(parcel_id, address)

                # Merge with existing data
                enhanced_prop = {**prop, **enhanced_data}
                enhanced.append(enhanced_prop)

                # Update MongoDB
                self.db.db.building_data.update_one(
                    {'parcel_id': parcel_id},
                    {'$set': enhanced_data}
                )

                # Rate limiting
                time.sleep(2)

            print(f"Processed {min(i+batch_size, len(properties))}/{len(properties)}")

            # Save checkpoint
            pd.DataFrame(enhanced).to_csv('enhanced_checkpoint.csv', index=False)

        # Save final enhanced dataset
        df = pd.DataFrame(enhanced)
        df.to_excel('data/exports/complete_flex_properties_ENHANCED_REAL.xlsx', index=False)

        # Verify no fake data
        self.verify_no_fake_data(df)

        return df

    def verify_no_fake_data(self, df):
        """Check that we didn't generate fake data"""
        # All same zoning = FAKE
        if df['zoning_code'].nunique() == 1:
            raise ValueError("ALL PROPERTIES HAVE SAME ZONING - THIS IS FAKE DATA")

        # Check for obvious estimates
        tax_rate = df['total_annual_tax'] / df['market_value']
        if tax_rate.std() < 0.001:  # All same tax rate = FAKE
            raise ValueError("ALL PROPERTIES HAVE SAME TAX RATE - THIS IS FAKE DATA")

        print(f"✓ Data verification passed")
        print(f"  - {df['zoning_code'].nunique()} unique zoning codes")
        print(f"  - Tax rates vary from {tax_rate.min():.3f} to {tax_rate.max():.3f}")

if __name__ == "__main__":
    # RUN THE REAL SCRAPER
    enhancer = RealPropertyEnhancer()

    # First verify it works
    print("STEP 1: Verifying scraper with known property...")
    enhancer.verify_against_known_values()

    # Then run on all properties
    print("\nSTEP 2: Enhancing all 611 properties...")
    enhanced_df = enhancer.enhance_all_properties()

    print(f"\nCOMPLETE: {len(enhanced_df)} properties enhanced with REAL data")
    print(f"Saved to: complete_flex_properties_ENHANCED_REAL.xlsx")