#!/usr/bin/env python3
"""
Private Property Data Analyzer CLI
Command-line interface for analyzing private property data from Excel files

Usage:
    python analyze_private_properties.py <excel_file> [options]

Examples:
    # Basic analysis
    python analyze_private_properties.py data/properties.xlsx
    
    # Full analysis with scoring and database storage
    python analyze_private_properties.py data/properties.xlsx --score --store-db --export-all
    
    # Analysis with custom output directory
    python analyze_private_properties.py data/properties.xlsx --output-dir results/analysis_2024
    
    # Quiet mode with JSON export only
    python analyze_private_properties.py data/properties.xlsx --quiet --export json
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from processors.private_property_analyzer import PrivatePropertyAnalyzer
from utils.logger import setup_logging


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Analyze private property data from Excel files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/properties.xlsx
  %(prog)s data/properties.xlsx --score --export-all
  %(prog)s data/properties.xlsx --output-dir results --quiet
  %(prog)s data/properties.xlsx --store-db --compare-historical
        """
    )
    
    # Required arguments
    parser.add_argument(
        'excel_file',
        help='Path to Excel file containing property data'
    )
    
    # Analysis options
    parser.add_argument(
        '--score',
        action='store_true',
        help='Add flex scores to industrial properties using FlexPropertyScorer'
    )
    
    parser.add_argument(
        '--score-all',
        action='store_true',
        help='Add flex scores to ALL properties (not just industrial)'
    )
    
    # Export options
    parser.add_argument(
        '--export',
        nargs='+',
        choices=['json', 'excel', 'csv'],
        default=['json'],
        help='Export formats (default: json)'
    )
    
    parser.add_argument(
        '--export-all',
        action='store_true',
        help='Export in all formats (json, excel, csv)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='data/analysis_results',
        help='Output directory for analysis results (default: data/analysis_results)'
    )
    
    # Database options
    parser.add_argument(
        '--store-db',
        action='store_true',
        help='Store analysis results in MongoDB database'
    )
    
    parser.add_argument(
        '--compare-historical',
        action='store_true',
        help='Compare current analysis with historical data from database'
    )
    
    # Display options
    parser.add_argument(
        '--sample-limit',
        type=int,
        default=10,
        help='Number of sample industrial properties to display (default: 10)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce console output (errors and warnings only)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Increase console output (debug level)'
    )
    
    # Logging options
    parser.add_argument(
        '--log-file',
        help='Custom log file path (default: logs/private_analysis_YYYYMMDD.log)'
    )
    
    parser.add_argument(
        '--no-file-logging',
        action='store_true',
        help='Disable file logging (console only)'
    )
    
    return parser.parse_args()


def setup_cli_logging(args):
    """Set up logging based on CLI arguments"""
    
    # Determine log level
    if args.quiet:
        level = 'WARNING'
    elif args.verbose:
        level = 'DEBUG'
    else:
        level = 'INFO'
    
    # Set up logger
    logger = setup_logging(
        name='private_property_cli',
        level=level,
        log_file=args.log_file,
        file_logging=not args.no_file_logging
    )
    
    return logger


def print_analysis_summary(report, logger):
    """Print a formatted summary of analysis results"""
    
    print("\n" + "="*60)
    print("PRIVATE PROPERTY ANALYSIS SUMMARY")
    print("="*60)
    
    # Dataset overview
    overview = report.get('dataset_overview', {})
    print(f"📊 Dataset Overview:")
    print(f"   Total Properties: {overview.get('total_properties', 'N/A'):,}")
    print(f"   Total Columns: {overview.get('total_columns', 'N/A')}")
    print(f"   Memory Usage: {overview.get('memory_usage_mb', 'N/A')} MB")
    
    # Property type analysis
    prop_analysis = report.get('property_type_analysis', {})
    if 'total_industrial_properties' in prop_analysis:
        print(f"\n🏭 Industrial Properties:")
        print(f"   Industrial Count: {prop_analysis.get('total_industrial_properties', 0):,}")
        print(f"   Industrial Percentage: {prop_analysis.get('industrial_percentage', 0):.1f}%")
        
        industrial_types = prop_analysis.get('industrial_types_found', [])
        if industrial_types:
            print(f"   Types Found: {', '.join(industrial_types)}")
    
    # Data quality
    quality = report.get('data_quality_metrics', {})
    if 'average_completeness' in quality:
        print(f"\n📈 Data Quality:")
        print(f"   Average Completeness: {quality.get('average_completeness', 0):.1f}%")
        print(f"   Complete Fields: {quality.get('complete_fields', 0)}")
        print(f"   Incomplete Fields: {quality.get('incomplete_fields', 0)}")
    
    # Industrial sample
    industrial_summary = report.get('industrial_property_summary', {})
    if industrial_summary.get('has_industrial_properties'):
        print(f"\n🏢 Industrial Sample:")
        print(f"   Sample Properties: {industrial_summary.get('sample_count', 0)}")
    
    # Analysis status
    status = report.get('analysis_status', 'unknown')
    errors = report.get('errors', [])
    
    print(f"\n✅ Analysis Status: {status.upper()}")
    if errors:
        print(f"⚠️  Errors Encountered: {len(errors)}")
    
    print("="*60)


def main():
    """Main CLI function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Set up logging
    logger = setup_cli_logging(args)
    
    try:
        # Validate input file
        excel_file = Path(args.excel_file)
        if not excel_file.exists():
            logger.error(f"Excel file not found: {excel_file}")
            sys.exit(1)
        
        logger.info(f"Starting analysis of: {excel_file}")
        logger.info(f"Output directory: {args.output_dir}")
        
        # Initialize analyzer
        analyzer = PrivatePropertyAnalyzer(str(excel_file), logger=logger)
        
        # Load data
        logger.info("Loading property data...")
        data = analyzer.load_data()
        logger.info(f"Loaded {len(data):,} properties with {len(data.columns)} columns")
        
        # Run core analysis
        logger.info("Running property type analysis...")
        industrial_types = analyzer.analyze_property_types()
        
        logger.info("Checking data completeness...")
        completeness = analyzer.check_data_completeness()
        
        logger.info(f"Getting industrial property sample (limit: {args.sample_limit})...")
        sample = analyzer.get_industrial_sample(limit=args.sample_limit)
        
        # Generate summary report
        logger.info("Generating comprehensive summary report...")
        report = analyzer.generate_summary_report()
        
        # Add flex scoring if requested
        scored_properties = None
        if args.score or args.score_all:
            logger.info("Adding flex scores to properties...")
            try:
                scored_properties = analyzer.add_flex_scoring(include_all_properties=args.score_all)
                
                if not scored_properties.empty:
                    logger.info(f"Successfully scored {len(scored_properties)} properties")
                    
                    # Log top scoring properties
                    top_scores = scored_properties.nlargest(5, 'flex_score')
                    logger.info("Top 5 flex scoring properties:")
                    for idx, (_, prop) in enumerate(top_scores.iterrows(), 1):
                        name = prop.get('Property Name', 'Unknown')[:30]
                        score = prop.get('flex_score', 0)
                        prop_type = prop.get('Property Type', 'Unknown')
                        logger.info(f"  {idx}. {name} - Score: {score:.2f} ({prop_type})")
                else:
                    logger.warning("No properties were scored")
                    
            except Exception as e:
                logger.error(f"Flex scoring failed: {e}")
        
        # Export results
        export_formats = ['json', 'excel', 'csv'] if args.export_all else args.export
        logger.info(f"Exporting results in formats: {export_formats}")
        
        exported_files = analyzer.export_results(
            output_dir=args.output_dir,
            formats=export_formats
        )
        
        logger.info("Export complete:")
        for format_name, file_path in exported_files.items():
            logger.info(f"  {format_name.upper()}: {file_path}")
        
        # Database operations
        if args.store_db:
            logger.info("Storing analysis results in database...")
            success = analyzer.store_results_in_database()
            
            if success:
                logger.info("✅ Analysis results stored in database")
                
                # Store scored properties if available
                if scored_properties is not None and not scored_properties.empty:
                    stored_count = analyzer.store_scored_properties(scored_properties)
                    logger.info(f"✅ Stored {stored_count} scored properties in database")
            else:
                logger.warning("❌ Failed to store results in database")
        
        # Historical comparison
        if args.compare_historical:
            logger.info("Comparing with historical analyses...")
            comparison = analyzer.compare_with_historical()
            
            if comparison.get('comparison_available'):
                trends = comparison.get('trends', {})
                logger.info("Historical comparison results:")
                logger.info(f"  Properties vs avg: {trends.get('properties_vs_avg', 0):+.0f}")
                logger.info(f"  Industrial vs avg: {trends.get('industrial_vs_avg', 0):+.0f}")
                logger.info(f"  Quality vs avg: {trends.get('quality_vs_avg', 0):+.1f}%")
            else:
                logger.info("No historical data available for comparison")
        
        # Print summary (unless quiet mode)
        if not args.quiet:
            print_analysis_summary(report, logger)
        
        # Final success message
        logger.info("🎉 Analysis completed successfully!")
        
        # Return success
        return 0
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return 1


if __name__ == '__main__':
    sys.exit(main())