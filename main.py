"""
Main Pipeline Orchestrator
Coordinates data extraction, processing, and storage
"""
import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.mongodb_client import get_db_manager
from extractors.gis_extractor import GISExtractor, GISConfig
from extractors.property_detail_extractor import PropertyDetailExtractor
from processors.flex_scorer import FlexPropertyScorer
from utils.logger import setup_logging

class FlexPropertyPipeline:
    """Main pipeline for flex property identification"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.db_manager = get_db_manager()
        self.scorer = FlexPropertyScorer()
        self.stats = {
            'start_time': None,
            'parcels_processed': 0,
            'flex_candidates_found': 0,
            'errors': []
        }
    
    async def phase1_gis_extraction(self) -> Dict:
        """
        Phase 1: Extract GIS zoning and parcel data
        This is the foundation - gets all industrial-zoned parcels
        """
        
        self.logger.info("=" * 50)
        self.logger.info("PHASE 1: GIS Data Extraction")
        self.logger.info("=" * 50)
        
        results = {
            'parcels_extracted': 0,
            'storage_used_mb': 0,
            'errors': []
        }
        
        try:
            # Check storage before starting
            storage = self.db_manager.check_storage()
            self.logger.info(f"Storage before extraction: {storage}")
            
            if storage['available_mb'] < 50:
                self.logger.warning("Low storage - running cleanup")
                self.db_manager.optimize_storage()
            
            # Extract GIS data
            async with GISExtractor() as extractor:
                # Get industrial parcels
                self.logger.info("Extracting industrial-zoned parcels...")
                parcels = await extractor.extract_flex_candidates()
                
                results['parcels_extracted'] = len(parcels)
                self.logger.info(f"Extracted {len(parcels)} parcels from GIS")
                
                if parcels:
                    # Save backup to file
                    backup_file = f"data/raw/gis_parcels_{datetime.now().strftime('%Y%m%d')}.json"
                    os.makedirs(os.path.dirname(backup_file), exist_ok=True)
                    
                    with open(backup_file, 'w') as f:
                        json.dump(parcels, f, default=str)
                    self.logger.info(f"Backup saved to {backup_file}")
                    
                    # Store in MongoDB in batches
                    self.logger.info("Storing parcels in MongoDB...")
                    inserted = self.db_manager.batch_insert(
                        'zoning_data',
                        parcels,
                        batch_size=100
                    )
                    
                    self.logger.info(f"Inserted {inserted} parcels into MongoDB")
                    
                    # Get detailed parcel info for top candidates
                    parcel_ids = [p['parcel_id'] for p in parcels[:500]]  # Start with first 500
                    
                    self.logger.info("Extracting detailed parcel information...")
                    detailed = await extractor.extract_parcel_details(parcel_ids)
                    
                    if detailed:
                        self.db_manager.batch_insert('staging_parcels', detailed)
                        self.logger.info(f"Added {len(detailed)} detailed parcels")
                    
                    # Check final storage
                    storage = self.db_manager.check_storage()
                    results['storage_used_mb'] = storage['total_size_mb']
                    
        except Exception as e:
            self.logger.error(f"Phase 1 error: {e}")
            results['errors'].append(str(e))
        
        return results
    
    async def phase2_enrichment(self) -> Dict:
        """
        Phase 2: Enrich parcels with property appraiser data
        Get building details, commercial info, sales history
        """
        import traceback
        
        self.logger.info("=" * 50)
        self.logger.info("PHASE 2: Property Detail Enrichment")
        self.logger.info("=" * 50)
        
        results = {
            'parcels_enriched': 0,
            'details_found': 0,
            'flex_candidates': 0,
            'successful_parcels': 0,
            'failed_parcels': 0,
            'errors': []
        }
        
        enriched_parcels = []
        failed_parcels = []
        
        try:
            # Get parcels from zoning_data that need enrichment
            # Start with top properties by value
            pipeline = [
                {'$match': {'market_value': {'$gt': 500000}}},  # Properties worth >$500k
                {'$sort': {'market_value': -1}},
                {'$limit': 100}  # Start with top 100 properties
            ]
            
            parcels = list(self.db_manager.db.zoning_data.aggregate(pipeline))
            
            self.logger.info(f"Enriching {len(parcels)} high-value parcels")
            
            if not parcels:
                self.logger.warning("No parcels found for enrichment")
                return results
            
            # Extract parcel IDs
            parcel_ids = [p['parcel_id'] for p in parcels if p.get('parcel_id')]
            
            # Get detailed property information
            async with PropertyDetailExtractor() as extractor:
                self.logger.info("Fetching property details from appraiser...")
                property_details = await extractor.batch_extract_details(
                    parcel_ids[:50],  # Limit to 50 for initial test
                    batch_size=5
                )
                
                self.logger.info(f"Retrieved details for {len(property_details)} properties")
                results['details_found'] = len(property_details)
                
                # Process each parcel individually with error handling
                for i, parcel in enumerate(parcels):
                    parcel_id = parcel.get('parcel_id', 'unknown')
                    
                    try:
                        self.logger.debug(f"Processing parcel {i+1}/{len(parcels)}: {parcel_id}")
                        
                        # Find matching details
                        details = next((d for d in property_details 
                                      if d['parcel_id'] == parcel_id), None)
                        
                        # Create enriched document
                        enriched = {
                            'parcel_id': parcel_id,
                            'property_use': parcel.get('property_use', ''),
                            'acres': parcel.get('acres', 0) or 0,
                            'owner_name': parcel.get('owner_name', ''),
                            'address': parcel.get('street_address', ''),
                            'municipality': parcel.get('municipality', ''),
                            'market_value': parcel.get('market_value', 0) or 0,
                            'assessed_value': parcel.get('assessed_value', 0) or 0,
                            'source_data': {
                                'gis': parcel,
                                'appraiser': details
                            }
                        }
                        
                        # Add building info if available
                        if details and details.get('building_info'):
                            enriched['building_data'] = details['building_info']
                        
                        # Add commercial info if available
                        if details and details.get('commercial_info'):
                            enriched['commercial_data'] = details['commercial_info']
                            
                            # Check for flex indicators
                            comm = details['commercial_info']
                            if comm.get('loading_docks', 0) > 0 or comm.get('overhead_doors', 0) > 0:
                                enriched['has_loading'] = True
                            
                            # Calculate office/warehouse ratio
                            office = comm.get('office_area', 0) or 0
                            warehouse = comm.get('warehouse_area', 0) or 0
                            if office > 0 and warehouse > 0:
                                enriched['office_warehouse_ratio'] = office / (office + warehouse)
                        
                        # Add sales history
                        if details and details.get('sales_history'):
                            enriched['sales_history'] = details['sales_history']
                        
                        # Calculate flex score with explicit None checks
                        if details:
                            indicators = extractor.calculate_flex_indicators(details)
                            base_score_result = self.scorer.calculate_flex_score(enriched)
                            base_score = base_score_result[0] if base_score_result and base_score_result[0] is not None else 0
                            
                            # Ensure base_score and adjustment are valid numbers
                            base_score = base_score if base_score is not None else 0
                            adjustment = indicators.get('flex_score_adjustment', 0)
                            adjustment = adjustment if adjustment is not None else 0
                            
                            # Explicit None checks before comparison
                            if base_score is None:
                                base_score = 0
                            if adjustment is None:
                                adjustment = 0
                            
                            # Adjust score based on detailed indicators
                            final_score = base_score + adjustment
                            final_score = min(10, max(0, final_score))  # Cap between 0-10
                            
                            enriched['flex_score'] = final_score
                            enriched['flex_indicators'] = indicators
                            
                            # Explicit None check before comparison
                            if final_score is not None and final_score >= 5:
                                results['flex_candidates'] += 1
                        else:
                            # Basic scoring without details
                            score_result = self.scorer.calculate_flex_score(enriched)
                            score = score_result[0] if score_result and score_result[0] is not None else 0
                            indicators = score_result[1] if score_result and len(score_result) > 1 else {}
                            
                            # Ensure score is not None
                            score = score if score is not None else 0
                            enriched['flex_score'] = score
                            enriched['indicators'] = indicators
                            
                            # Explicit None check before comparison
                            if score is not None and score >= 5:
                                results['flex_candidates'] += 1
                        
                        enriched['processed_at'] = datetime.utcnow()
                        enriched_parcels.append(enriched)
                        
                        results['successful_parcels'] += 1
                        
                    except Exception as e:
                        self.logger.error(f"Phase 2 error on parcel {parcel_id} (line {traceback.extract_tb(e.__traceback__)[-1].lineno}): {e}")
                        self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
                        
                        failed_parcels.append({
                            'parcel_id': parcel_id,
                            'error': str(e),
                            'traceback': traceback.format_exc()
                        })
                        results['failed_parcels'] += 1
                        results['errors'].append(f"Parcel {parcel_id}: {traceback.format_exc()}")
                        continue
        
        except Exception as e:
            self.logger.error(f"Phase 2 critical error: {e}")
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            results['errors'].append(f"Critical error: {traceback.format_exc()}")
        
        # ALWAYS attempt to save whatever data was successfully processed
        if enriched_parcels:
            try:
                inserted = self.db_manager.batch_insert(
                    'enriched_properties',
                    enriched_parcels,
                    batch_size=50
                )
                results['parcels_enriched'] = inserted
                
                self.logger.info(f"Saved {inserted} properties despite {len(failed_parcels)} failures")
                self.logger.info(f"Found {results['flex_candidates']} flex candidates")
                self.logger.info(f"Success rate: {results['successful_parcels']}/{results['successful_parcels'] + results['failed_parcels']} parcels")
            except Exception as e:
                self.logger.error(f"Failed to save enriched parcels: {e}")
                self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
                results['errors'].append(f"Save error: {traceback.format_exc()}")
        else:
            self.logger.warning("No parcels were successfully processed")
        
        # Log failed parcels for debugging
        if failed_parcels:
            self.logger.error(f"Failed to process {len(failed_parcels)} parcels:")
            for failed in failed_parcels[:5]:  # Show first 5 failures
                self.logger.error(f"  - {failed['parcel_id']}: {failed['error']}")
        
        return results
    
    async def phase3_analysis(self) -> Dict:
        """
        Phase 3: Analyze and identify top flex candidates
        """
        
        self.logger.info("=" * 50)
        self.logger.info("PHASE 3: Flex Property Analysis")
        self.logger.info("=" * 50)
        
        results = {
            'top_candidates': [],
            'statistics': {},
            'errors': []
        }
        
        try:
            # Get flex candidates
            candidates = self.db_manager.find_flex_candidates(min_score=5)
            
            self.logger.info(f"Analyzing {len(candidates)} flex candidates")
            
            # Group by municipality
            by_municipality = {}
            for candidate in candidates:
                muni = candidate.get('municipality', 'Unknown')
                if muni not in by_municipality:
                    by_municipality[muni] = []
                by_municipality[muni].append(candidate)
            
            # Statistics
            results['statistics'] = {
                'total_candidates': len(candidates),
                'by_municipality': {k: len(v) for k, v in by_municipality.items()},
                'average_flex_score': sum(c.get('flex_score', 0) for c in candidates) / len(candidates) if candidates else 0,
                'top_10_parcels': candidates[:10]
            }
            
            # Save results
            output_file = f"data/processed/flex_candidates_{datetime.now().strftime('%Y%m%d')}.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(results['statistics'], f, indent=2, default=str)
            
            self.logger.info(f"Analysis complete. Results saved to {output_file}")
            
            # Store top candidates in separate collection
            if candidates[:100]:  # Top 100
                self.db_manager.batch_insert('flex_candidates', candidates[:100])
            
        except Exception as e:
            self.logger.error(f"Phase 3 error: {e}")
            results['errors'].append(str(e))
        
        return results
    
    async def run_pipeline(self, phases: List[int] = None):
        """
        Run the complete pipeline or specific phases
        
        Args:
            phases: List of phase numbers to run (1, 2, 3) or None for all
        """
        
        self.stats['start_time'] = datetime.now()
        
        if phases is None:
            phases = [1, 2, 3]
        
        self.logger.info(f"Starting Flex Property Pipeline - Phases: {phases}")
        self.logger.info(f"MongoDB Database: {self.db_manager.database_name}")
        
        # Check initial storage
        storage = self.db_manager.check_storage()
        self.logger.info(f"Initial storage: {storage}")
        
        results = {}
        
        try:
            # Run requested phases
            if 1 in phases:
                results['phase1'] = await self.phase1_gis_extraction()
                self.stats['parcels_processed'] = results['phase1'].get('parcels_extracted', 0)
            
            if 2 in phases:
                results['phase2'] = await self.phase2_enrichment()
                self.stats['flex_candidates_found'] = results['phase2'].get('flex_candidates', 0)
            
            if 3 in phases:
                results['phase3'] = await self.phase3_analysis()
            
            # Final storage check
            final_storage = self.db_manager.check_storage()
            
            # Print summary
            self.logger.info("=" * 50)
            self.logger.info("PIPELINE COMPLETE")
            self.logger.info("=" * 50)
            self.logger.info(f"Duration: {datetime.now() - self.stats['start_time']}")
            self.logger.info(f"Parcels Processed: {self.stats['parcels_processed']}")
            self.logger.info(f"Flex Candidates Found: {self.stats['flex_candidates_found']}")
            self.logger.info(f"Final Storage: {final_storage['total_size_mb']}MB / {final_storage['available_mb']}MB available")
            
            # Save pipeline results
            results_file = f"data/processed/pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            
            with open(results_file, 'w') as f:
                json.dump({
                    'stats': self.stats,
                    'results': results,
                    'storage': final_storage
                }, f, indent=2, default=str)
            
            self.logger.info(f"Results saved to {results_file}")
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            self.stats['errors'].append(str(e))
            raise
        
        finally:
            # Cleanup
            self.db_manager.close()
        
        return results

async def main():
    """Main entry point"""
    
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Flex Property Pipeline')
    parser.add_argument('--phases', nargs='+', type=int, choices=[1, 2, 3],
                       help='Phases to run (1=GIS, 2=Enrichment, 3=Analysis)')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with limited data')
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = FlexPropertyPipeline()
    
    if args.test:
        pipeline.logger.info("Running in TEST mode")
        # Limit data for testing
        # You can modify the extractors to limit data in test mode
    
    results = await pipeline.run_pipeline(phases=args.phases)
    
    return results

if __name__ == "__main__":
    # Run the pipeline
    asyncio.run(main())
