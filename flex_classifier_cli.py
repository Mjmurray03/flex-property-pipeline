#!/usr/bin/env python3
"""
Flex Property Classifier Command Line Interface
Main CLI script for running flex property classification
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

from processors.flex_property_classifier import FlexPropertyClassifier
from utils.flex_data_utils import load_property_data, preprocess_property_data, create_sample_dataset
from utils.logger import setup_logging


def main():
    """Main CLI function"""
    
    parser = argparse.ArgumentParser(
        description="Flex Property Classifier - Identify flex industrial properties from Excel data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python flex_classifier_cli.py --file properties.xlsx
  
  # Create sample data and analyze
  python flex_classifier_cli.py --create-sample --analyze-sample
  
  # Custom output and top candidates
  python flex_classifier_cli.py --file properties.xlsx --output results.xlsx --top-n 50
  
  # Preprocessing and verbose output
  python flex_classifier_cli.py --file properties.xlsx --preprocess --verbose
        """
    )
    
    # Input options
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Path to Excel file containing property data'
    )
    
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Create sample dataset for testing'
    )
    
    parser.add_argument(
        '--analyze-sample',
        action='store_true',
        help='Analyze sample dataset after creation'
    )
    
    # Processing options
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Apply data preprocessing before analysis'
    )
    
    parser.add_argument(
        '--top-n', '-n',
        type=int,
        default=100,
        help='Number of top candidates to export (default: 100)'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output path for results (default: data/exports/private_flex_candidates.xlsx)'
    )
    
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Show statistics only, do not export results'
    )
    
    # Logging options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    if args.quiet:
        log_level = 'WARNING'
    elif args.verbose:
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'
    
    logger = setup_logging('flex_classifier_cli', level=log_level)
    
    try:
        # Handle sample creation
        if args.create_sample:
            logger.info("Creating sample dataset...")
            sample_file = create_sample_dataset(logger=logger)
            
            if not args.quiet:
                print(f"✅ Sample dataset created: {sample_file}")
            
            if args.analyze_sample:
                args.file = sample_file
            elif not args.file:
                return 0
        
        # Validate input
        if not args.file:
            logger.error("No input file specified. Use --file or --create-sample")
            parser.print_help()
            return 1
        
        if not Path(args.file).exists():
            logger.error(f"Input file not found: {args.file}")
            return 1
        
        # Load data
        logger.info(f"Loading data from: {args.file}")
        df = load_property_data(args.file, logger=logger)
        
        # Preprocess if requested
        if args.preprocess:
            logger.info("Preprocessing data...")
            df = preprocess_property_data(df, logger=logger)
        
        # Initialize classifier
        logger.info("Initializing classifier...")
        classifier = FlexPropertyClassifier(df, logger=logger)
        
        # Validate data quality
        quality_report = classifier.validate_data_quality()
        quality_score = quality_report.get('quality_score', 0)
        
        if not args.quiet:
            print(f"Data Quality Score: {quality_score:.1f}/10")
        
        # Run classification
        logger.info("Running classification...")
        candidates = classifier.classify_flex_properties()
        
        if len(candidates) == 0:
            if not args.quiet:
                print("❌ No flex candidates found")
            return 0
        
        # Calculate scores
        logger.info("Calculating flex scores...")
        classifier._apply_scoring_to_candidates()
        
        # Get statistics
        stats = classifier.get_analysis_statistics()
        
        # Display results
        if not args.quiet:
            print("\n" + "="*50)
            print("FLEX PROPERTY ANALYSIS RESULTS")
            print("="*50)
            print(f"Properties Analyzed: {stats['total_properties_analyzed']:,}")
            print(f"Flex Candidates: {stats['total_flex_candidates']:,}")
            print(f"Success Rate: {stats['candidate_percentage']:.1f}%")
            
            if stats.get('score_statistics'):
                score_stats = stats['score_statistics']
                print(f"Average Score: {score_stats['average_flex_score']:.2f}")
                print(f"Score Range: {score_stats['min_score']:.1f} - {score_stats['max_score']:.1f}")
        
        # Export results
        if not args.stats_only:
            top_candidates = classifier.get_top_candidates(n=args.top_n)
            
            if len(top_candidates) > 0:
                # Set up export
                original_candidates = classifier.flex_candidates
                classifier.flex_candidates = top_candidates
                
                output_path = classifier.export_results(args.output)
                
                # Restore
                classifier.flex_candidates = original_candidates
                
                if not args.quiet:
                    print(f"✅ Exported {len(top_candidates)} candidates to: {output_path}")
            else:
                if not args.quiet:
                    print("❌ No candidates to export")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())