#!/usr/bin/env python3
"""
Full-Scale Building Data Scraper for ALL Industrial Properties
Processes all 3,168+ industrial properties with robust error handling and progress tracking
"""

import os
import time
import json
import logging
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
import re
from collections import Counter

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('building_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BuildingDataScraper:
    """Full-scale building data scraper with batch processing and error handling"""

    def __init__(self):
        self.db = self.connect_to_mongodb()
        self.batch_size = 50
        self.retry_limit = 3
        self.request_delay = 2  # seconds between requests
        self.checkpoint_interval = 100  # save checkpoint every N properties

        self.stats = {
            'total_properties': 0,
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'errors': [],
            'start_time': None,
            'batch_times': []
        }

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def connect_to_mongodb(self):
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
                        logger.info(f"Connected to database: {db_name}")
                        db = test_db
                        break

            if db is None:
                db = client['flexfilter']

            client.admin.command('ping')
            return db
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise

    def get_all_industrial_properties(self):
        """Get ALL industrial properties from zoning_data collection"""
        logger.info("=== RETRIEVING ALL INDUSTRIAL PROPERTIES ===")

        industrial_filter = {
            "property_use": {
                "$in": [
                    "WAREH/DIST TERM",
                    "VACANT INDUSTRIAL",
                    "HEAVY MFG",
                    "OPEN STORAGE",
                    "WORKING WATERFRONT"
                ]
            }
        }

        # Get all properties
        properties = list(self.db.zoning_data.find(industrial_filter))

        logger.info(f"Retrieved {len(properties)} industrial properties")

        # Show breakdown by type
        type_counts = Counter([p.get('property_use') for p in properties])
        logger.info("Property type breakdown:")
        for prop_type, count in type_counts.most_common():
            logger.info(f"  {prop_type}: {count}")

        self.stats['total_properties'] = len(properties)
        return properties

    def load_checkpoint(self):
        """Load checkpoint data to resume interrupted processing"""
        checkpoint_file = 'scraping_checkpoint.json'

        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                logger.info(f"Loaded checkpoint: resuming from property {checkpoint.get('last_index', 0)}")
                return checkpoint
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")

        return {'last_index': 0, 'failed_properties': [], 'processed_count': 0}

    def save_checkpoint(self, index, failed_properties):
        """Save checkpoint data"""
        checkpoint = {
            'last_index': index,
            'failed_properties': failed_properties,
            'processed_count': self.stats['processed'],
            'successful_count': self.stats['successful'],
            'failed_count': self.stats['failed'],
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats
        }

        with open('scraping_checkpoint.json', 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)

        logger.info(f"Checkpoint saved at index {index}")

    def scrape_property_data(self, parcel_id):
        """Scrape building data for a single property (improved version)"""

        clean_parcel_id = parcel_id.replace('-', '').replace(' ', '')
        url = f"https://pbcpao.gov/Property/RenderPrintSum?parcelId={clean_parcel_id}&flag=ALL"

        scraped_data = {
            'parcel_id': parcel_id,
            'building_sqft': None,
            'warehouse_area': None,
            'office_area': None,
            'year_built': None,
            'scrape_success': False,
            'scrape_message': '',
            'raw_areas': {}
        }

        try:
            response = self.session.get(url, timeout=15)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                page_text = soup.get_text()

                # Extract building data using improved patterns
                area_patterns = [
                    (r'\*total square feet\s*:\s*([0-9,]+)', 'total_sqft_star'),
                    (r'total square footage\s*:?\s*([0-9,]+)', 'total_square_footage'),
                    (r'warehouse storage\s*:?\s*([0-9,]+)', 'warehouse_storage'),
                    (r'warehouse\s*:?\s*([0-9,]+)', 'warehouse'),
                    (r'whse\s+office\s*:?\s*([0-9,]+)', 'warehouse_office'),
                    (r'office\s*:?\s*([0-9,]+)', 'office'),
                    (r'year built\s*:?\s*([0-9]{4})', 'year_built'),
                ]

                for pattern, field_name in area_patterns:
                    matches = re.findall(pattern, page_text, re.IGNORECASE)
                    if matches:
                        for match in matches:
                            try:
                                value = int(match.replace(',', ''))

                                if field_name == 'year_built' and 1800 <= value <= 2025:
                                    scraped_data['year_built'] = value
                                elif field_name != 'year_built' and 50 < value < 10000000:
                                    scraped_data['raw_areas'][field_name] = value
                            except ValueError:
                                continue

                # Determine best building square footage
                if scraped_data['raw_areas']:
                    priority_fields = [
                        'total_sqft_star',
                        'total_square_footage',
                        'warehouse_storage',
                        'warehouse'
                    ]

                    best_sqft = None
                    for field in priority_fields:
                        if field in scraped_data['raw_areas']:
                            best_sqft = scraped_data['raw_areas'][field]
                            break

                    if best_sqft is None and scraped_data['raw_areas']:
                        # Take largest reasonable area
                        area_values = {k: v for k, v in scraped_data['raw_areas'].items()
                                     if isinstance(v, int) and 100 <= v <= 1000000}
                        if area_values:
                            best_sqft = max(area_values.values())

                    if best_sqft:
                        scraped_data['building_sqft'] = best_sqft

                        # Extract component areas
                        if 'warehouse' in scraped_data['raw_areas']:
                            scraped_data['warehouse_area'] = scraped_data['raw_areas']['warehouse']
                        if 'warehouse_office' in scraped_data['raw_areas']:
                            scraped_data['office_area'] = scraped_data['raw_areas']['warehouse_office']
                        elif 'office' in scraped_data['raw_areas']:
                            scraped_data['office_area'] = scraped_data['raw_areas']['office']

                        scraped_data['scrape_success'] = True
                        scraped_data['scrape_message'] = "Successfully extracted building data"
                    else:
                        scraped_data['scrape_message'] = "Page found but no valid building data"
                else:
                    scraped_data['scrape_message'] = "Page found but no area data extracted"

            elif response.status_code == 404:
                scraped_data['scrape_message'] = "Property page not found (404)"
            else:
                scraped_data['scrape_message'] = f"HTTP {response.status_code}"

        except requests.exceptions.Timeout:
            scraped_data['scrape_message'] = "Request timeout"
        except requests.exceptions.RequestException as e:
            scraped_data['scrape_message'] = f"Request error: {str(e)[:50]}"
        except Exception as e:
            scraped_data['scrape_message'] = f"Scraping error: {str(e)[:50]}"

        return scraped_data

    def update_mongodb_property(self, parcel_id, building_data):
        """Update both zoning_data and building_data collections"""

        # Update zoning_data with building square footage
        if building_data['scrape_success'] and building_data['building_sqft']:
            update_data = {
                'building_sqft': building_data['building_sqft'],
                'building_data_scraped': True,
                'building_data_timestamp': datetime.now()
            }

            if building_data.get('year_built'):
                update_data['year_built'] = building_data['year_built']

            result = self.db.zoning_data.update_one(
                {'parcel_id': parcel_id},
                {'$set': update_data}
            )

            # Add to building_data collection
            building_doc = {
                'parcel_id': parcel_id,
                'building_sqft': building_data['building_sqft'],
                'warehouse_area': building_data.get('warehouse_area'),
                'office_area': building_data.get('office_area'),
                'year_built': building_data.get('year_built'),
                'raw_areas': building_data.get('raw_areas', {}),
                'scraped_at': time.time(),
                'scrape_url': f"https://pbcpao.gov/Property/RenderPrintSum?parcelId={parcel_id}&flag=ALL",
                'scrape_success': building_data['scrape_success'],
                'scrape_message': building_data['scrape_message']
            }

            self.db.building_data.replace_one(
                {'parcel_id': parcel_id},
                building_doc,
                upsert=True
            )

            return True
        else:
            # Still log the failed attempt
            failed_doc = {
                'parcel_id': parcel_id,
                'scrape_success': False,
                'scrape_message': building_data['scrape_message'],
                'scraped_at': time.time()
            }

            self.db.building_data.replace_one(
                {'parcel_id': parcel_id},
                failed_doc,
                upsert=True
            )

            return False

    def estimate_time_remaining(self, processed, total, batch_times):
        """Estimate time remaining based on batch processing times"""
        if not batch_times or processed == 0:
            return "Calculating..."

        avg_batch_time = sum(batch_times[-5:]) / len(batch_times[-5:])  # Use last 5 batches
        remaining_properties = total - processed
        remaining_batches = remaining_properties / self.batch_size
        remaining_seconds = remaining_batches * avg_batch_time

        hours = int(remaining_seconds // 3600)
        minutes = int((remaining_seconds % 3600) // 60)

        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"

    def scale_building_scraper(self):
        """Main function to scale scraping to ALL industrial properties"""

        logger.info("=== STARTING FULL-SCALE BUILDING DATA SCRAPING ===")
        self.stats['start_time'] = time.time()

        # Get all industrial properties
        all_properties = self.get_all_industrial_properties()
        total = len(all_properties)

        if total == 0:
            logger.error("No industrial properties found!")
            return

        # Load checkpoint to resume if interrupted
        checkpoint = self.load_checkpoint()
        start_index = checkpoint.get('last_index', 0)
        failed_properties = checkpoint.get('failed_properties', [])

        logger.info(f"Starting processing from index {start_index}")
        logger.info(f"Previous failures to retry: {len(failed_properties)}")

        # Process in batches
        batch_count = 0
        processed_this_run = 0

        for i in range(start_index, total, self.batch_size):
            batch_start_time = time.time()
            batch_count += 1

            batch = all_properties[i:i+self.batch_size]
            batch_successful = 0
            batch_failed = 0

            logger.info(f"\n--- BATCH {batch_count} ---")
            logger.info(f"Processing properties {i+1} to {min(i+self.batch_size, total)} of {total}")

            # Process each property in batch
            for j, property_data in enumerate(batch):
                parcel_id = property_data.get('parcel_id')
                property_index = i + j + 1

                logger.info(f"  {property_index}/{total}: {parcel_id}")

                # Check if already processed (has building_sqft)
                if property_data.get('building_sqft') is not None:
                    logger.info(f"    SKIPPED: Already has building data")
                    self.stats['processed'] += 1
                    self.stats['successful'] += 1
                    processed_this_run += 1
                    continue

                # Scrape with retry logic
                success = False
                for attempt in range(self.retry_limit):
                    try:
                        building_data = self.scrape_property_data(parcel_id)

                        if building_data['scrape_success'] and building_data['building_sqft']:
                            # Update MongoDB immediately
                            if self.update_mongodb_property(parcel_id, building_data):
                                logger.info(f"    SUCCESS: {building_data['building_sqft']:,} sqft")
                                self.stats['successful'] += 1
                                batch_successful += 1
                                success = True
                                break
                            else:
                                logger.warning(f"    WARNING: Scraped but no valid building data")
                        else:
                            logger.warning(f"    FAILED: {building_data['scrape_message']}")

                        # Always record the attempt
                        self.update_mongodb_property(parcel_id, building_data)

                    except Exception as e:
                        logger.error(f"    ERROR (attempt {attempt+1}): {str(e)[:100]}")
                        if attempt < self.retry_limit - 1:
                            time.sleep(1)  # Brief pause before retry
                        continue

                if not success:
                    self.stats['failed'] += 1
                    batch_failed += 1
                    if parcel_id not in failed_properties:
                        failed_properties.append(parcel_id)

                self.stats['processed'] += 1
                processed_this_run += 1

                # Rate limiting
                time.sleep(self.request_delay)

            # Batch completed
            batch_time = time.time() - batch_start_time
            self.stats['batch_times'].append(batch_time)

            # Progress update
            progress_pct = (self.stats['processed'] / total) * 100
            time_remaining = self.estimate_time_remaining(
                self.stats['processed'],
                total,
                self.stats['batch_times']
            )

            logger.info(f"BATCH {batch_count} COMPLETE:")
            logger.info(f"  Successful: {batch_successful}")
            logger.info(f"  Failed: {batch_failed}")
            logger.info(f"  Batch time: {batch_time:.1f}s")
            logger.info(f"  Overall progress: {self.stats['processed']}/{total} ({progress_pct:.1f}%)")
            logger.info(f"  Success rate: {(self.stats['successful']/self.stats['processed']*100):.1f}%")
            logger.info(f"  Estimated time remaining: {time_remaining}")

            # Save checkpoint
            if self.stats['processed'] % self.checkpoint_interval == 0 or i + self.batch_size >= total:
                self.save_checkpoint(i + self.batch_size, failed_properties)

        # Final summary
        total_time = time.time() - self.stats['start_time']
        logger.info("\n=== SCRAPING COMPLETE ===")
        logger.info(f"Total time: {total_time/3600:.1f} hours")
        logger.info(f"Total processed: {self.stats['processed']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Success rate: {(self.stats['successful']/self.stats['processed']*100):.1f}%")

        # Analyze results
        self.analyze_scraped_data()

        # Clean up checkpoint file
        if os.path.exists('scraping_checkpoint.json'):
            os.rename('scraping_checkpoint.json', f'scraping_checkpoint_completed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

        logger.info("Full-scale scraping pipeline completed successfully!")

    def analyze_scraped_data(self):
        """Analyze the complete scraped dataset"""
        logger.info("\n=== ANALYZING COMPLETE DATASET ===")

        # Get all properties with building data
        pipeline = [
            {
                '$match': {
                    'property_use': {
                        '$in': [
                            'WAREH/DIST TERM',
                            'VACANT INDUSTRIAL',
                            'HEAVY MFG',
                            'OPEN STORAGE',
                            'WORKING WATERFRONT'
                        ]
                    },
                    'building_sqft': {'$exists': True, '$gt': 0}
                }
            }
        ]

        properties_with_buildings = list(self.db.zoning_data.aggregate(pipeline))

        if properties_with_buildings:
            building_sizes = [p['building_sqft'] for p in properties_with_buildings]
            building_sizes.sort(reverse=True)

            # Size categories
            mega_facilities = len([s for s in building_sizes if s >= 100000])
            large_flex = len([s for s in building_sizes if 50000 <= s < 100000])
            standard_flex = len([s for s in building_sizes if 20000 <= s < 50000])
            medium_buildings = len([s for s in building_sizes if 10000 <= s < 20000])
            small_buildings = len([s for s in building_sizes if s < 10000])

            flex_qualified = mega_facilities + large_flex + standard_flex

            logger.info(f"Properties with building data: {len(properties_with_buildings)}")
            logger.info(f"Building size distribution:")
            logger.info(f"  Mega Facilities (100K+ sqft): {mega_facilities}")
            logger.info(f"  Large Flex (50-100K sqft): {large_flex}")
            logger.info(f"  Standard Flex (20-50K sqft): {standard_flex}")
            logger.info(f"  Medium (10-20K sqft): {medium_buildings}")
            logger.info(f"  Small (<10K sqft): {small_buildings}")
            logger.info(f"TOTAL FLEX-QUALIFIED (>=20K sqft): {flex_qualified}")
            logger.info(f"Flex qualification rate: {(flex_qualified/len(properties_with_buildings)*100):.1f}%")

if __name__ == "__main__":
    scraper = BuildingDataScraper()
    scraper.scale_building_scraper()