#!/usr/bin/env python3
"""
Enhanced Property Data Scraper for Flex Properties
Extracts comprehensive property data from pbcpao.gov using the same successful methodology
"""

import os
import time
import json
import logging
import re
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedPropertyScraper:
    """Enhanced property scraper for comprehensive data extraction"""

    def __init__(self):
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

    def load_base_properties(self):
        """Load complete_flex_properties.xlsx to get parcel_ids"""
        logger.info("=== LOADING BASE FLEX PROPERTIES ===")

        excel_path = 'data/exports/complete_flex_properties.xlsx'
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Base properties file not found: {excel_path}")

        df = pd.read_excel(excel_path)
        logger.info(f"Loaded {len(df)} properties from {excel_path}")

        # Get parcel_ids for processing
        parcel_ids = df['parcel_id'].tolist()
        self.stats['total_properties'] = len(parcel_ids)

        return df, parcel_ids

    def load_checkpoint(self):
        """Load checkpoint data to resume interrupted processing"""
        checkpoint_file = 'enhanced_scraping_checkpoint.json'

        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                logger.info(f"Loaded checkpoint: resuming from property {checkpoint.get('last_index', 0)}")
                return checkpoint
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")

        return {'last_index': 0, 'failed_properties': [], 'processed_count': 0, 'enhanced_data': []}

    def save_checkpoint(self, index, enhanced_data, failed_properties):
        """Save checkpoint data"""
        checkpoint = {
            'last_index': index,
            'enhanced_data': enhanced_data,
            'failed_properties': failed_properties,
            'processed_count': self.stats['processed'],
            'successful_count': self.stats['successful'],
            'failed_count': self.stats['failed'],
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats
        }

        with open('enhanced_scraping_checkpoint.json', 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)

        logger.info(f"Checkpoint saved at index {index}")

    def extract_enhanced_property_data(self, parcel_id):
        """Extract comprehensive property data from pbcpao.gov"""

        # Convert to string and clean - use same URL format as successful scraper
        parcel_id_str = str(parcel_id)
        clean_parcel_id = parcel_id_str.replace('-', '').replace(' ', '')
        url = f"https://pbcpao.gov/Property/RenderPrintSum?parcelId={clean_parcel_id}&flag=ALL"

        enhanced_data = {
            'parcel_id': parcel_id_str,
            'extraction_timestamp': datetime.now().isoformat(),
            'scrape_success': False,
            'scrape_message': '',

            # Property Information
            'subarea_warehouse_sqft': None,
            'subarea_office_sqft': None,
            'zoning_code': None,
            'property_use_code_detail': None,
            'number_of_units': None,

            # Appraisals (5 years)
            'improvement_value_2025': None,
            'improvement_value_2024': None,
            'improvement_value_2023': None,
            'improvement_value_2022': None,
            'improvement_value_2021': None,
            'land_value_2025': None,
            'land_value_2024': None,
            'land_value_2023': None,
            'land_value_2022': None,
            'land_value_2021': None,
            'market_value_2025': None,
            'market_value_2024': None,
            'market_value_2023': None,
            'market_value_2022': None,
            'market_value_2021': None,

            # Assessed & Taxable Values
            'assessed_value_current': None,
            'exemption_amount': None,
            'taxable_value_current': None,

            # Taxes
            'ad_valorem_tax': None,
            'non_ad_valorem_tax': None,
            'total_annual_tax': None,

            # Sales History
            'sales_history': []
        }

        try:
            response = self.session.get(url, timeout=30)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                page_text = soup.get_text()

                # Extract Property Information
                self._extract_property_info(soup, page_text, enhanced_data)

                # Extract Subarea Information
                self._extract_subarea_data(soup, page_text, enhanced_data)

                # Extract Appraisal History
                self._extract_appraisal_data(soup, page_text, enhanced_data)

                # Extract Tax Information
                self._extract_tax_data(soup, page_text, enhanced_data)

                # Extract Sales History
                self._extract_sales_history(soup, page_text, enhanced_data)

                enhanced_data['scrape_success'] = True
                enhanced_data['scrape_message'] = "Successfully extracted enhanced property data"

            elif response.status_code == 404:
                enhanced_data['scrape_message'] = "Property page not found (404)"
            else:
                enhanced_data['scrape_message'] = f"HTTP {response.status_code}"

        except requests.exceptions.Timeout:
            enhanced_data['scrape_message'] = "Request timeout"
        except requests.exceptions.RequestException as e:
            enhanced_data['scrape_message'] = f"Request error: {str(e)[:50]}"
        except Exception as e:
            enhanced_data['scrape_message'] = f"Scraping error: {str(e)[:50]}"

        return enhanced_data

    def _extract_property_info(self, soup, page_text, data):
        """Extract basic property information"""

        # Zoning Code (e.g., "PID—PID PLANNED COMMERCIAL DEV")
        zoning_patterns = [
            r'zoning[:\s]*([A-Z][A-Z0-9\s\-—]+(?:PLANNED|COMMERCIAL|INDUSTRIAL|DEV|DISTRICT)[A-Z0-9\s\-—]*)',
            r'(PID[—\-\s]*[A-Z0-9\s\-—]*)',
            r'([A-Z]{2,3}[—\-\s]*[A-Z0-9\s\-—]*(?:PLANNED|COMMERCIAL|INDUSTRIAL|DEV|DISTRICT)[A-Z0-9\s\-—]*)'
        ]

        for pattern in zoning_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            if matches:
                data['zoning_code'] = matches[0].strip()
                break

        # Property Use Code Detail (e.g., "4800—WAREH/DIST TERM")
        use_code_patterns = [
            r'(\d{4}[—\-\s]*[A-Z/\s]+(?:WAREH|DIST|TERM|MFG|STORAGE))',
            r'property\s+use[:\s]*(\d{4}[—\-\s]*[A-Z/\s]+)',
            r'use\s+code[:\s]*(\d{4}[—\-\s]*[A-Z/\s]+)'
        ]

        for pattern in use_code_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            if matches:
                data['property_use_code_detail'] = matches[0].strip()
                break

        # Number of Units
        unit_patterns = [
            r'number\s+of\s+units[:\s]*(\d+)',
            r'units[:\s]*(\d+)',
            r'(\d+)\s+units'
        ]

        for pattern in unit_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            if matches:
                try:
                    data['number_of_units'] = int(matches[0])
                    break
                except ValueError:
                    continue

    def _extract_subarea_data(self, soup, page_text, data):
        """Extract subarea square footage data"""

        # Look for subarea table or section
        subarea_sections = [
            soup.find('table', string=re.compile('subarea', re.IGNORECASE)),
            soup.find(text=re.compile('SUBAREA.*SQUARE.*FOOTAGE', re.IGNORECASE)),
            soup.find('div', {'class': re.compile('subarea', re.IGNORECASE)})
        ]

        # Extract warehouse and office areas
        warehouse_patterns = [
            r'warehouse[:\s]*([0-9,]+)',
            r'whse[:\s]*([0-9,]+)',
            r'warehouse\s+storage[:\s]*([0-9,]+)'
        ]

        office_patterns = [
            r'whse\s+office[:\s]*([0-9,]+)',
            r'warehouse\s+office[:\s]*([0-9,]+)',
            r'office[:\s]*([0-9,]+)'
        ]

        for pattern in warehouse_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            if matches:
                try:
                    value = int(matches[0].replace(',', ''))
                    if 10 < value < 10000000:  # Reasonable bounds
                        data['subarea_warehouse_sqft'] = value
                        break
                except ValueError:
                    continue

        for pattern in office_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            if matches:
                try:
                    value = int(matches[0].replace(',', ''))
                    if 10 < value < 10000000:  # Reasonable bounds
                        data['subarea_office_sqft'] = value
                        break
                except ValueError:
                    continue

    def _extract_appraisal_data(self, soup, page_text, data):
        """Extract 5 years of appraisal data"""

        # Find appraisal section
        appraisal_section = soup.find('div', {'id': 'appraisals'}) or soup.find(text=re.compile('APPRAISALS', re.IGNORECASE))

        years = [2025, 2024, 2023, 2022, 2021]
        value_types = ['improvement', 'land', 'market']

        for year in years:
            for value_type in value_types:
                patterns = [
                    rf'{value_type}.*value.*{year}[:\s]*\$?([0-9,]+)',
                    rf'{year}.*{value_type}[:\s]*\$?([0-9,]+)',
                    rf'{value_type}[:\s]*{year}[:\s]*\$?([0-9,]+)'
                ]

                for pattern in patterns:
                    matches = re.findall(pattern, page_text, re.IGNORECASE)
                    if matches:
                        try:
                            value = int(matches[0].replace(',', ''))
                            field_name = f'{value_type}_value_{year}'
                            data[field_name] = value
                            break
                        except ValueError:
                            continue

    def _extract_tax_data(self, soup, page_text, data):
        """Extract tax information"""

        # Assessed Value
        assessed_patterns = [
            r'assessed\s+value[:\s]*\$?([0-9,]+)',
            r'current\s+assessed[:\s]*\$?([0-9,]+)'
        ]

        for pattern in assessed_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            if matches:
                try:
                    data['assessed_value_current'] = int(matches[0].replace(',', ''))
                    break
                except ValueError:
                    continue

        # Exemption Amount
        exemption_patterns = [
            r'exemption[:\s]*\$?([0-9,]+)',
            r'exempt\s+amount[:\s]*\$?([0-9,]+)'
        ]

        for pattern in exemption_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            if matches:
                try:
                    data['exemption_amount'] = int(matches[0].replace(',', ''))
                    break
                except ValueError:
                    continue

        # Taxable Value
        taxable_patterns = [
            r'taxable\s+value[:\s]*\$?([0-9,]+)',
            r'current\s+taxable[:\s]*\$?([0-9,]+)'
        ]

        for pattern in taxable_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            if matches:
                try:
                    data['taxable_value_current'] = int(matches[0].replace(',', ''))
                    break
                except ValueError:
                    continue

        # Tax amounts
        ad_valorem_patterns = [
            r'ad\s+valorem[:\s]*\$?([0-9,.]+)',
            r'ad\s+valorem\s+tax[:\s]*\$?([0-9,.]+)'
        ]

        non_ad_valorem_patterns = [
            r'non\s+ad\s+valorem[:\s]*\$?([0-9,.]+)',
            r'non-ad\s+valorem[:\s]*\$?([0-9,.]+)'
        ]

        total_tax_patterns = [
            r'total\s+annual\s+tax[:\s]*\$?([0-9,.]+)',
            r'total\s+tax[:\s]*\$?([0-9,.]+)'
        ]

        for pattern in ad_valorem_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            if matches:
                try:
                    data['ad_valorem_tax'] = float(matches[0].replace(',', ''))
                    break
                except ValueError:
                    continue

        for pattern in non_ad_valorem_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            if matches:
                try:
                    data['non_ad_valorem_tax'] = float(matches[0].replace(',', ''))
                    break
                except ValueError:
                    continue

        for pattern in total_tax_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            if matches:
                try:
                    data['total_annual_tax'] = float(matches[0].replace(',', ''))
                    break
                except ValueError:
                    continue

    def _extract_sales_history(self, soup, page_text, data):
        """Extract sales history records"""

        # Look for sales section
        sales_section = soup.find('div', {'id': 'sales'}) or soup.find(text=re.compile('SALES.*HISTORY', re.IGNORECASE))

        # Sales record patterns
        sales_patterns = [
            r'(\d{1,2}/\d{1,2}/\d{4})[:\s]*\$?([0-9,]+)[:\s]*([A-Z\s]+)[:\s]*(\d+[-/]\d+)',
            r'sale\s+date[:\s]*(\d{1,2}/\d{1,2}/\d{4})[:\s]*price[:\s]*\$?([0-9,]+)'
        ]

        sales_history = []

        for pattern in sales_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            for match in matches:
                try:
                    if len(match) >= 4:
                        sale_record = {
                            'date': match[0],
                            'price': int(match[1].replace(',', '')),
                            'deed_type': match[2].strip(),
                            'book_page': match[3]
                        }
                    else:
                        sale_record = {
                            'date': match[0],
                            'price': int(match[1].replace(',', '')),
                            'deed_type': None,
                            'book_page': None
                        }
                    sales_history.append(sale_record)
                except (ValueError, IndexError):
                    continue

        data['sales_history'] = sales_history

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

    def enhance_property_data(self, test_mode=False):
        """Main function to enhance property data"""

        logger.info("=== STARTING ENHANCED PROPERTY DATA EXTRACTION ===")
        self.stats['start_time'] = time.time()

        # Load base properties
        base_df, parcel_ids = self.load_base_properties()

        if test_mode:
            parcel_ids = parcel_ids[:5]  # Test with first 5 properties
            logger.info(f"TEST MODE: Processing first {len(parcel_ids)} properties")

        total = len(parcel_ids)

        if total == 0:
            logger.error("No properties found to process!")
            return

        # Load checkpoint to resume if interrupted
        checkpoint = self.load_checkpoint()
        start_index = checkpoint.get('last_index', 0)
        enhanced_data = checkpoint.get('enhanced_data', [])
        failed_properties = checkpoint.get('failed_properties', [])

        logger.info(f"Starting processing from index {start_index}")
        logger.info(f"Previous enhanced records: {len(enhanced_data)}")
        logger.info(f"Previous failures to retry: {len(failed_properties)}")

        # Process in batches
        batch_count = 0
        processed_this_run = 0

        for i in range(start_index, total, self.batch_size):
            batch_start_time = time.time()
            batch_count += 1

            batch = parcel_ids[i:i+self.batch_size]
            batch_successful = 0
            batch_failed = 0

            logger.info(f"\n--- BATCH {batch_count} ---")
            logger.info(f"Processing properties {i+1} to {min(i+self.batch_size, total)} of {total}")

            # Process each property in batch
            for j, parcel_id in enumerate(batch):
                property_index = i + j + 1

                logger.info(f"  {property_index}/{total}: {parcel_id}")

                # Scrape with retry logic
                success = False
                for attempt in range(self.retry_limit):
                    try:
                        property_data = self.extract_enhanced_property_data(parcel_id)

                        if property_data['scrape_success']:
                            enhanced_data.append(property_data)
                            logger.info(f"    SUCCESS: Enhanced data extracted")
                            self.stats['successful'] += 1
                            batch_successful += 1
                            success = True
                            break
                        else:
                            logger.warning(f"    FAILED: {property_data['scrape_message']}")

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
            batch_success_rate = (self.stats['successful']/self.stats['processed']*100) if self.stats['processed'] > 0 else 0
            logger.info(f"  Success rate: {batch_success_rate:.1f}%")
            logger.info(f"  Estimated time remaining: {time_remaining}")

            # Save checkpoint
            if self.stats['processed'] % self.checkpoint_interval == 0 or i + self.batch_size >= total:
                self.save_checkpoint(i + self.batch_size, enhanced_data, failed_properties)

        # Final summary
        total_time = time.time() - self.stats['start_time']
        logger.info("\n=== ENHANCEMENT COMPLETE ===")
        logger.info(f"Total time: {total_time/3600:.1f} hours")
        logger.info(f"Total processed: {self.stats['processed']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        success_rate = (self.stats['successful']/self.stats['processed']*100) if self.stats['processed'] > 0 else 0
        logger.info(f"Success rate: {success_rate:.1f}%")

        return enhanced_data, base_df

    def merge_enhanced_with_base(self, enhanced_data, base_df):
        """Merge enhanced data with base properties"""
        logger.info("=== MERGING ENHANCED DATA WITH BASE PROPERTIES ===")

        # Convert enhanced data to DataFrame
        enhanced_df = pd.DataFrame(enhanced_data)

        if enhanced_df.empty:
            logger.warning("No enhanced data to merge!")
            return base_df

        # Merge on parcel_id
        final_df = base_df.merge(enhanced_df, on='parcel_id', how='left')

        # Add calculated fields
        self._add_calculated_fields(final_df)

        # Save enhanced dataset
        output_dir = 'data/exports'
        os.makedirs(output_dir, exist_ok=True)

        excel_path = os.path.join(output_dir, 'complete_flex_properties_ENHANCED.xlsx')
        csv_path = os.path.join(output_dir, 'complete_flex_properties_ENHANCED.csv')

        final_df.to_excel(excel_path, index=False)
        final_df.to_csv(csv_path, index=False)

        logger.info(f"Enhanced dataset saved to:")
        logger.info(f"  Excel: {excel_path}")
        logger.info(f"  CSV: {csv_path}")

        return final_df

    def _add_calculated_fields(self, df):
        """Add calculated fields to the merged dataset"""

        # 5-year appreciation
        df['5yr_appreciation'] = (
            (df['market_value_2025'] - df['market_value_2021']) /
            df['market_value_2021'] * 100
        ).round(2)

        # Office to warehouse ratio
        df['office_warehouse_ratio'] = (
            df['subarea_office_sqft'] /
            (df['subarea_warehouse_sqft'] + df['subarea_office_sqft'])
        ).round(3)

        # Tax rate (percentage)
        df['tax_rate'] = (
            df['total_annual_tax'] / df['market_value_2025'] * 100
        ).round(3)

        # Sales count
        df['sales_count'] = df['sales_history'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    def create_summary_report(self, final_df):
        """Create summary report of enhanced data"""
        logger.info("=== CREATING SUMMARY REPORT ===")

        report = {
            'generation_timestamp': datetime.now().isoformat(),
            'total_properties': len(final_df),
            'enhanced_properties': final_df['scrape_success'].sum() if 'scrape_success' in final_df.columns else 0,
            'enhancement_rate': f"{(final_df['scrape_success'].sum() / len(final_df) * 100):.1f}%" if 'scrape_success' in final_df.columns else "0%",

            'top_appreciation_properties': [],
            'lowest_tax_rate_properties': [],
            'best_office_warehouse_ratio': [],
            'average_metrics': {}
        }

        # Calculate average metrics
        numeric_columns = [
            'total_annual_tax', 'market_value_2025', 'tax_rate',
            '5yr_appreciation', 'office_warehouse_ratio'
        ]

        for col in numeric_columns:
            if col in final_df.columns:
                report['average_metrics'][col] = float(final_df[col].mean()) if not final_df[col].isna().all() else None

        # Top properties by various metrics
        try:
            # Highest 5-year appreciation
            if '5yr_appreciation' in final_df.columns:
                top_appreciation = final_df.nlargest(10, '5yr_appreciation')[
                    ['parcel_id', 'address', '5yr_appreciation', 'market_value_2025']
                ].to_dict('records')
                report['top_appreciation_properties'] = top_appreciation

            # Lowest tax rates
            if 'tax_rate' in final_df.columns:
                lowest_tax = final_df.nsmallest(10, 'tax_rate')[
                    ['parcel_id', 'address', 'tax_rate', 'total_annual_tax']
                ].to_dict('records')
                report['lowest_tax_rate_properties'] = lowest_tax

            # Best office/warehouse ratios for flex
            if 'office_warehouse_ratio' in final_df.columns:
                best_ratio = final_df.dropna(subset=['office_warehouse_ratio']).nlargest(10, 'office_warehouse_ratio')[
                    ['parcel_id', 'address', 'office_warehouse_ratio', 'subarea_office_sqft', 'subarea_warehouse_sqft']
                ].to_dict('records')
                report['best_office_warehouse_ratio'] = best_ratio

        except Exception as e:
            logger.warning(f"Error creating detailed report sections: {e}")

        # Save report
        report_path = 'data/exports/enhancement_summary_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Summary report saved to: {report_path}")
        return report

def main():
    """Main execution function"""
    scraper = EnhancedPropertyScraper()

    # Test mode first (5 properties)
    logger.info("Starting with TEST MODE (5 properties)")
    enhanced_data, base_df = scraper.enhance_property_data(test_mode=True)

    if enhanced_data:
        logger.info(f"Test successful! Enhanced {len(enhanced_data)} properties")

        # Ask user to continue with full run
        user_input = input("\nTest completed successfully. Continue with full enhancement of all properties? (y/n): ")

        if user_input.lower() == 'y':
            # Reset stats for full run
            scraper.stats = {
                'total_properties': 0,
                'processed': 0,
                'successful': 0,
                'failed': 0,
                'errors': [],
                'start_time': None,
                'batch_times': []
            }

            # Full enhancement
            enhanced_data, base_df = scraper.enhance_property_data(test_mode=False)

            # Merge and create outputs
            final_df = scraper.merge_enhanced_with_base(enhanced_data, base_df)
            report = scraper.create_summary_report(final_df)

            logger.info("Enhanced property scraping completed successfully!")
        else:
            logger.info("Full enhancement cancelled by user")
    else:
        logger.error("Test failed - no data extracted")

if __name__ == "__main__":
    main()