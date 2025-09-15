#!/usr/bin/env python3
"""
Run Full Enhancement - Non-interactive version for all 611 properties
"""

import logging
from enhanced_property_scraper import EnhancedPropertyScraper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('full_enhancement.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Run full enhancement on all properties"""

    logger.info("=== STARTING FULL ENHANCEMENT FOR ALL 611 PROPERTIES ===")

    scraper = EnhancedPropertyScraper()

    # Run full enhancement (not test mode)
    enhanced_data, base_df = scraper.enhance_property_data(test_mode=False)

    if enhanced_data:
        logger.info(f"Enhancement completed! Enhanced {len(enhanced_data)} properties")

        # Merge and create outputs
        final_df = scraper.merge_enhanced_with_base(enhanced_data, base_df)
        report = scraper.create_summary_report(final_df)

        logger.info("=== FULL ENHANCEMENT PIPELINE COMPLETED SUCCESSFULLY ===")
        logger.info(f"Total properties processed: {scraper.stats['processed']}")
        logger.info(f"Successfully enhanced: {scraper.stats['successful']}")
        logger.info(f"Failed: {scraper.stats['failed']}")

        # Output file summary
        logger.info("\nOutput files created:")
        logger.info("  - data/exports/complete_flex_properties_ENHANCED.xlsx")
        logger.info("  - data/exports/complete_flex_properties_ENHANCED.csv")
        logger.info("  - data/exports/enhancement_summary_report.json")
        logger.info("  - full_enhancement.log")

    else:
        logger.error("Enhancement failed - no data extracted")

if __name__ == "__main__":
    main()