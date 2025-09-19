"""
Batch Flex Property Analysis Script
Command-line interface for processing multiple Excel files with advanced features
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from processors.flex_classifier_advanced import AdvancedFlexClassifier, ScoringConfiguration
from utils.logger import setup_logging


def find_excel_files(directory: str, recursive: bool = True) -> List[str]:
    """
    Find all Excel files in a directory
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        List of Excel file paths
    """
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    patterns = ['*.xlsx', '*.xls']
    excel_files = []
    
    for pattern in patterns:
        if recursive:
            excel_files.extend(directory_path.rglob(pattern))
        else:
            excel_files.extend(directory_path.glob(pattern))
    
    return [str(f) for f in excel_files]


def create_progress_callback(logger: logging.Logger):
    """Create a progress callback function"""
    def progress_callback(message: str, progress: float):
        logger.info(f"Progress: {progress*100:.1f}% - {message}")
    return progress_callback


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Batch Flex Property Analysis with Advanced Features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all Excel files in a directory
  python batch_flex_analysis.py --input-dir data/excel_files --output-dir results

  # Process specific files with custom configuration
  python batch_flex_analysis.py --files file1.xlsx file2.xlsx --config custom_config.json

  # Process with custom workers and show detailed analytics
  python batch_flex_analysis.py --input-dir data --workers 8 --analytics --verbose
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input-dir', '-d',
        help='Directory containing Excel files to process'
    )
    input_group.add_argument(
        '--files', '-f',
        nargs='+',
        help='Specific Excel files to process'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir', '-o',
        default='data/exports/batch_analysis',
        help='Output directory for results (default: data/exports/batch_analysis)'
    )
    
    # Processing options
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of concurrent workers (default: 4)'
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Path to scoring configuration JSON file'
    )
    
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Search for Excel files recursively in subdirectories'
    )
    
    # Analysis options
    parser.add_argument(
        '--analytics', '-a',
        action='store_true',
        help='Generate advanced analytics'
    )
    
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bar'
    )
    
    # Logging options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--log-file',
        help='Log file path (default: logs/batch_analysis.log)'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    log_file = args.log_file or 'logs/batch_analysis.log'
    
    logger = setup_logging(
        name='batch_flex_analysis',
        level=log_level,
        log_file=log_file
    )
    
    try:
        # Get list of files to process
        if args.input_dir:
            logger.info(f"Searching for Excel files in: {args.input_dir}")
            file_paths = find_excel_files(args.input_dir, args.recursive)
            logger.info(f"Found {len(file_paths)} Excel files")
        else:
            file_paths = args.files
            logger.info(f"Processing {len(file_paths)} specified files")
        
        if not file_paths:
            logger.error("No Excel files found to process")
            return 1
        
        # Load scoring configuration
        scoring_config = None
        if args.config:
            logger.info(f"Loading scoring configuration from: {args.config}")
            scoring_config = ScoringConfiguration.load_from_file(args.config)
        else:
            logger.info("Using default scoring configuration")
            scoring_config = ScoringConfiguration()
        
        # Create advanced classifier
        classifier = AdvancedFlexClassifier(scoring_config=scoring_config, logger=logger)
        
        # Set up progress callback
        if not args.no_progress:
            progress_callback = create_progress_callback(logger)
            classifier.set_progress_callback(progress_callback)
        
        # Process files
        logger.info(f"Starting batch processing with {args.workers} workers")
        
        results = classifier.process_multiple_files(
            file_paths=file_paths,
            max_workers=args.workers,
            show_progress=not args.no_progress
        )
        
        # Extract results
        aggregated_results = results['aggregated_results']
        processing_metrics = results['processing_metrics']
        failed_files = results['failed_files']
        
        # Log processing summary
        logger.info("=" * 60)
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total files processed: {processing_metrics['processed_files']}")
        logger.info(f"Failed files: {processing_metrics['failed_files']}")
        logger.info(f"Total properties analyzed: {processing_metrics['total_properties']:,}")
        logger.info(f"Total flex candidates found: {processing_metrics['total_candidates']:,}")
        logger.info(f"Processing time: {processing_metrics['processing_time']:.2f} seconds")
        logger.info(f"Properties per second: {processing_metrics['properties_per_second']:.1f}")
        
        if failed_files:
            logger.warning(f"Failed files ({len(failed_files)}):")
            for failed_file in failed_files:
                logger.warning(f"  - {failed_file['file_path']}: {failed_file['error']}")
        
        # Generate advanced analytics if requested
        analytics = {}
        if args.analytics and not aggregated_results.empty:
            logger.info("Generating advanced analytics...")
            analytics = classifier.generate_advanced_analytics(aggregated_results)
            
            # Log key insights
            if 'overview' in analytics:
                overview = analytics['overview']
                logger.info(f"Analytics Overview:")
                logger.info(f"  - Average flex score: {overview.get('average_score', 0):.2f}")
                logger.info(f"  - Score standard deviation: {overview.get('score_std', 0):.2f}")
            
            if 'market_insights' in analytics and 'recommendations' in analytics['market_insights']:
                recommendations = analytics['market_insights']['recommendations']
                if recommendations:
                    logger.info("Market Insights:")
                    for rec in recommendations[:3]:  # Show top 3 recommendations
                        logger.info(f"  - {rec}")
        
        # Export results
        logger.info(f"Exporting results to: {args.output_dir}")
        
        exported_files = classifier.export_advanced_results(
            results_df=aggregated_results,
            analytics=analytics,
            output_dir=args.output_dir
        )
        
        # Log exported files
        logger.info("Exported files:")
        for file_type, file_path in exported_files.items():
            logger.info(f"  - {file_type}: {file_path}")
        
        # Create summary report
        summary_report = {
            'processing_summary': processing_metrics,
            'results_summary': {
                'total_candidates': len(aggregated_results) if not aggregated_results.empty else 0,
                'high_score_candidates': len(aggregated_results[aggregated_results.get('flex_score', 0) >= 7.0]) if not aggregated_results.empty and 'flex_score' in aggregated_results.columns else 0,
                'average_score': float(aggregated_results['flex_score'].mean()) if not aggregated_results.empty and 'flex_score' in aggregated_results.columns else 0
            },
            'failed_files': failed_files,
            'exported_files': exported_files
        }
        
        # Save summary report
        summary_path = Path(args.output_dir) / 'batch_processing_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)
        
        logger.info(f"Summary report saved to: {summary_path}")
        
        logger.info("=" * 60)
        logger.info("BATCH PROCESSING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        if args.verbose:
            logger.exception("Full error details:")
        return 1


if __name__ == '__main__':
    sys.exit(main())