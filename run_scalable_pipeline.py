#!/usr/bin/env python3
"""
Main execution script for the Scalable Multi-File Pipeline
Integrates all pipeline components for complete execution
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from pipeline.scalable_flex_pipeline import ScalableFlexPipeline, PipelineConfiguration
from pipeline.file_discovery import FileDiscovery
from pipeline.batch_processor import BatchProcessor
from pipeline.result_aggregator import ResultAggregator
from pipeline.report_generator import ReportGenerator
from pipeline.output_manager import OutputManager
from pipeline.data_validator import DataValidator
from pipeline.pipeline_logger import PipelineLogger
from utils.logger import setup_logging


def integrate_pipeline_components(pipeline: ScalableFlexPipeline) -> None:
    """
    Initialize and integrate all pipeline components
    
    Args:
        pipeline: Main pipeline instance to configure
    """
    try:
        # Initialize file discovery
        pipeline.file_discovery = FileDiscovery(
            input_folder=pipeline.config.input_folder,
            file_pattern=pipeline.config.file_pattern,
            recursive_scan=pipeline.config.recursive_scan
        )
        
        # Initialize batch processor
        pipeline.batch_processor = BatchProcessor(
            max_workers=pipeline.config.max_workers,
            batch_size=pipeline.config.batch_size,
            timeout_minutes=pipeline.config.timeout_minutes
        )
        
        # Initialize result aggregator
        pipeline.result_aggregator = ResultAggregator(
            enable_deduplication=pipeline.config.enable_deduplication,
            duplicate_fields=pipeline.config.duplicate_fields,
            min_flex_score=pipeline.config.min_flex_score
        )
        
        # Initialize report generator
        pipeline.report_generator = ReportGenerator()
        
        # Initialize output manager
        output_dir = Path(pipeline.config.output_file).parent
        pipeline.output_manager = OutputManager(
            base_output_dir=str(output_dir),
            enable_backup=pipeline.config.backup_existing
        )
        
        # Initialize data validator
        pipeline.data_validator = DataValidator()
        
        # Initialize pipeline logger
        pipeline.pipeline_logger = PipelineLogger(
            log_level=pipeline.config.log_level,
            log_file=pipeline.config.log_file,
            enable_progress=pipeline.config.progress_reporting
        )
        
        pipeline.logger.info("All pipeline components initialized successfully")
        
    except Exception as e:
        pipeline.logger.error(f"Failed to initialize pipeline components: {e}")
        raise


def run_complete_pipeline(pipeline: ScalableFlexPipeline) -> Dict[str, Any]:
    """
    Execute the complete pipeline with all integrated components
    
    Args:
        pipeline: Configured pipeline instance
        
    Returns:
        Dictionary containing complete execution results
    """
    start_time = datetime.now()
    pipeline.logger.info("Starting complete pipeline execution...")
    
    try:
        # Step 1: File Discovery
        pipeline.logger.info("Step 1: Discovering files...")
        discovered_files = pipeline.file_discovery.discover_files()
        pipeline.discovered_files = discovered_files
        
        if not discovered_files:
            pipeline.logger.warning("No files found to process")
            return {
                'success': True,
                'files_discovered': 0,
                'files_processed': 0,
                'flex_properties_found': 0,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'message': 'No files found to process'
            }
        
        pipeline.logger.info(f"Discovered {len(discovered_files)} files for processing")
        
        # Step 2: Data Validation
        pipeline.logger.info("Step 2: Validating file schemas...")
        validation_results = pipeline.data_validator.validate_files(discovered_files)
        
        valid_files = [f for f, result in validation_results.items() if result['valid']]
        invalid_files = [f for f, result in validation_results.items() if not result['valid']]
        
        if invalid_files:
            pipeline.logger.warning(f"Found {len(invalid_files)} invalid files - they will be skipped")
            for invalid_file in invalid_files:
                pipeline.logger.warning(f"Invalid file: {invalid_file}")
        
        if not valid_files:
            pipeline.logger.error("No valid files found to process")
            return {
                'success': False,
                'error': 'No valid files found to process',
                'files_discovered': len(discovered_files),
                'files_processed': 0,
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
        
        # Step 3: Batch Processing
        pipeline.logger.info(f"Step 3: Processing {len(valid_files)} valid files...")
        processing_results = pipeline.batch_processor.process_files(valid_files)
        pipeline.processing_results = processing_results
        
        successful_results = [r for r in processing_results if r.success]
        failed_results = [r for r in processing_results if not r.success]
        
        pipeline.logger.info(f"Processing completed: {len(successful_results)} successful, {len(failed_results)} failed")
        
        if not successful_results:
            pipeline.logger.error("No files processed successfully")
            return {
                'success': False,
                'error': 'No files processed successfully',
                'files_discovered': len(discovered_files),
                'files_processed': 0,
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
        
        # Step 4: Result Aggregation
        pipeline.logger.info("Step 4: Aggregating results...")
        aggregated_df = pipeline.result_aggregator.aggregate_results(successful_results)
        pipeline.aggregated_results = aggregated_df
        
        if aggregated_df is None or aggregated_df.empty:
            pipeline.logger.warning("No flex properties found in processed files")
            flex_properties_count = 0
        else:
            flex_properties_count = len(aggregated_df)
            pipeline.logger.info(f"Found {flex_properties_count} unique flex properties")
        
        # Step 5: Report Generation
        pipeline.logger.info("Step 5: Generating reports...")
        report_data = pipeline.report_generator.generate_comprehensive_report(
            aggregated_df if aggregated_df is not None else None,
            processing_results,
            validation_results
        )
        
        # Step 6: Output Export
        pipeline.logger.info("Step 6: Exporting results...")
        export_results = []
        
        if aggregated_df is not None and not aggregated_df.empty:
            # Export master Excel file
            excel_result, csv_result = pipeline.output_manager.export_master_file(
                aggregated_df,
                Path(pipeline.config.output_file).stem
            )
            export_results.extend([excel_result, csv_result])
            
            # Save processing metadata
            processing_stats = {
                'files_discovered': len(discovered_files),
                'files_processed': len(successful_results),
                'files_failed': len(failed_results),
                'flex_properties_found': flex_properties_count,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'report_data': report_data
            }
            
            metadata_path = pipeline.output_manager.save_export_metadata(
                export_results,
                processing_stats
            )
            pipeline.logger.info(f"Export metadata saved to: {metadata_path}")
        
        # Step 7: Final Statistics
        execution_time = (datetime.now() - start_time).total_seconds()
        pipeline.processing_stats = {
            'files_discovered': len(discovered_files),
            'files_processed': len(successful_results),
            'files_failed': len(failed_results),
            'flex_properties_found': flex_properties_count,
            'execution_time': execution_time,
            'processing_rate': len(successful_results) / (execution_time / 60) if execution_time > 0 else 0,
            'report_data': report_data,
            'export_results': export_results
        }
        
        pipeline.logger.info(f"Pipeline execution completed successfully in {execution_time:.2f} seconds")
        pipeline.logger.info(f"Processing rate: {pipeline.processing_stats['processing_rate']:.1f} files/minute")
        
        return {
            'success': True,
            'files_discovered': len(discovered_files),
            'files_processed': len(successful_results),
            'files_failed': len(failed_results),
            'flex_properties_found': flex_properties_count,
            'execution_time': execution_time,
            'output_file': pipeline.config.output_file,
            'report_data': report_data,
            'export_results': export_results,
            'configuration': pipeline.get_configuration_summary()
        }
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        pipeline.logger.error(f"Pipeline execution failed after {execution_time:.2f} seconds: {e}")
        
        return {
            'success': False,
            'error': str(e),
            'files_discovered': len(pipeline.discovered_files),
            'files_processed': 0,
            'execution_time': execution_time,
            'configuration': pipeline.get_configuration_summary()
        }


def main():
    """Main execution function"""
    try:
        # Create default configuration
        config = PipelineConfiguration()
        
        # Create pipeline
        pipeline = ScalableFlexPipeline(config=config)
        
        # Integrate all components
        integrate_pipeline_components(pipeline)
        
        # Run complete pipeline
        results = run_complete_pipeline(pipeline)
        
        # Print results summary
        print("\n" + "="*80)
        print("SCALABLE MULTI-FILE PIPELINE EXECUTION SUMMARY")
        print("="*80)
        
        if results['success']:
            print(f"✅ Status: SUCCESS")
            print(f"📁 Files Discovered: {results['files_discovered']}")
            print(f"✅ Files Processed: {results['files_processed']}")
            if results.get('files_failed', 0) > 0:
                print(f"❌ Files Failed: {results['files_failed']}")
            print(f"🏠 Flex Properties Found: {results['flex_properties_found']}")
            print(f"⏱️  Execution Time: {results['execution_time']:.2f} seconds")
            print(f"📄 Output File: {results['output_file']}")
        else:
            print(f"❌ Status: FAILED")
            print(f"❌ Error: {results['error']}")
            print(f"⏱️  Execution Time: {results['execution_time']:.2f} seconds")
        
        print("="*80)
        
        return 0 if results['success'] else 1
        
    except KeyboardInterrupt:
        print("\n❌ Pipeline execution interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())