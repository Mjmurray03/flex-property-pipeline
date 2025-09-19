#!/usr/bin/env python3
"""
Flex Property Classifier Example Usage
Demonstrates complete workflow from Excel loading to analysis and export
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from processors.flex_property_classifier import FlexPropertyClassifier
from utils.flex_data_utils import load_property_data, preprocess_property_data, create_sample_dataset
from utils.logger import setup_logging


def main():
    """Main CLI interface for Flex Property Classifier"""
    
    parser = argparse.ArgumentParser(
        description="Flex Property Classifier - Identify flex industrial properties from Excel data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze properties from Excel file
  python flex_classifier_example.py --file data/properties.xlsx
  
  # Create and analyze sample dataset
  python flex_classifier_example.py --create-sample
  
  # Analyze with custom output path
  python flex_classifier_example.py --file data/properties.xlsx --output data/my_results.xlsx
  
  # Show top 50 candidates only
  python flex_classifier_example.py --file data/properties.xlsx --top-n 50
  
  # Enable debug logging
  python flex_classifier_example.py --file data/properties.xlsx --verbose
        """
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Path to Excel file containing property data'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/exports/flex_candidates.xlsx',
        help='Output path for results (default: data/exports/flex_candidates.xlsx)'
    )
    
    parser.add_argument(
        '--top-n', '-n',
        type=int,
        default=100,
        help='Number of top candidates to export (default: 100)'
    )
    
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Create sample dataset for testing'
    )
    
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Apply data preprocessing before analysis'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Show statistics only, do not export results'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logging('flex_classifier_cli', level=log_level)
    
    try:
        # Handle sample creation
        if args.create_sample:
            logger.info("Creating sample dataset...")
            sample_file = create_sample_dataset(logger=logger)
            logger.info(f"✅ Sample dataset created: {sample_file}")
            
            # Ask if user wants to analyze the sample
            if not args.file:
                response = input("Analyze the sample dataset? (y/n): ").lower().strip()
                if response in ['y', 'yes']:
                    args.file = sample_file
                else:
                    logger.info("Sample created. Use --file to analyze it later.")
                    return
        
        # Validate input file
        if not args.file:
            logger.error("No input file specified. Use --file or --create-sample")
            parser.print_help()
            return
        
        if not Path(args.file).exists():
            logger.error(f"Input file not found: {args.file}")
            return
        
        # Load and preprocess data
        logger.info(f"Loading property data from: {args.file}")
        df = load_property_data(args.file, logger=logger)
        
        if args.preprocess:
            logger.info("Applying data preprocessing...")
            df = preprocess_property_data(df, logger=logger)
        
        # Initialize classifier
        logger.info("Initializing Flex Property Classifier...")
        classifier = FlexPropertyClassifier(df, logger=logger)
        
        # Validate data quality
        logger.info("Validating data quality...")
        quality_report = classifier.validate_data_quality()
        quality_score = quality_report.get('quality_score', 0)
        
        logger.info(f"Data quality score: {quality_score:.1f}/10")
        
        if quality_score < 5:
            logger.warning("Low data quality detected. Consider data cleaning or preprocessing.")
            response = input("Continue with analysis? (y/n): ").lower().strip()
            if response not in ['y', 'yes']:
                logger.info("Analysis cancelled due to data quality concerns.")
                return
        
        # Run classification
        logger.info("Running flex property classification...")
        candidates = classifier.classify_flex_properties()
        
        if len(candidates) == 0:
            logger.warning("No flex candidates found. Check your data and criteria.")
            return
        
        # Apply scoring
        logger.info("Calculating flex scores...")
        classifier._apply_scoring_to_candidates()
        
        # Get analysis statistics
        logger.info("Generating analysis statistics...")
        stats = classifier.get_analysis_statistics()
        
        # Display results
        print("\n" + "="*60)
        print("FLEX PROPERTY ANALYSIS RESULTS")
        print("="*60)
        
        print(f"Total Properties Analyzed: {stats['total_properties_analyzed']:,}")
        print(f"Flex Candidates Found: {stats['total_flex_candidates']:,}")
        print(f"Candidate Rate: {stats['candidate_percentage']:.1f}%")
        
        if stats.get('score_statistics'):
            score_stats = stats['score_statistics']
            print(f"\nScore Statistics:")
            print(f"  Average Score: {score_stats['average_flex_score']:.2f}")
            print(f"  Score Range: {score_stats['min_score']:.1f} - {score_stats['max_score']:.1f}")
            print(f"  High Scores (8+): {score_stats['high_score_count']}")
            print(f"  Medium Scores (6-8): {score_stats['medium_score_count']}")
            print(f"  Low Scores (<6): {score_stats['low_score_count']}")
        
        # Show top candidates
        top_candidates = classifier.get_top_candidates(n=min(10, args.top_n))
        
        if len(top_candidates) > 0:
            print(f"\nTop {len(top_candidates)} Candidates:")
            print("-" * 80)
            
            for idx, (_, candidate) in enumerate(top_candidates.iterrows(), 1):
                score = candidate.get('flex_score', 0)
                prop_name = candidate.get('Property Name', candidate.get('property_name', 'N/A'))
                prop_type = candidate.get('Property Type', candidate.get('property_type', 'N/A'))
                building_size = candidate.get('Building SqFt', candidate.get('building_sqft', 'N/A'))
                
                print(f"{idx:2d}. {prop_name} (Score: {score:.1f})")
                print(f"    Type: {prop_type}, Size: {building_size:,} sqft" if building_size != 'N/A' else f"    Type: {prop_type}")
        
        # Property type distribution
        if stats.get('property_type_distribution'):
            print(f"\nProperty Type Distribution (Top 100):")
            for prop_type, count in list(stats['property_type_distribution'].items())[:5]:
                print(f"  {prop_type}: {count}")
        
        # Export results unless stats-only
        if not args.stats_only:
            logger.info(f"Exporting top {args.top_n} candidates...")
            
            # Get top N candidates for export
            export_candidates = classifier.get_top_candidates(n=args.top_n)
            
            if len(export_candidates) > 0:
                # Temporarily store the candidates for export
                original_candidates = classifier.flex_candidates
                classifier.flex_candidates = export_candidates
                
                # Export
                output_path = classifier.export_results(args.output)
                
                # Restore original candidates
                classifier.flex_candidates = original_candidates
                
                print(f"\n✅ Results exported to: {output_path}")
                print(f"   Exported {len(export_candidates)} candidates")
            else:
                logger.warning("No candidates to export")
        
        print("\n" + "="*60)
        logger.info("Analysis complete!")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_example_workflow():
    """Run a complete example workflow for demonstration"""
    
    logger = setup_logging('flex_example_workflow', level='INFO')
    
    try:
        logger.info("🏭 Starting Flex Property Classifier Example Workflow")
        
        # Step 1: Create sample dataset
        logger.info("Step 1: Creating sample dataset...")
        sample_file = create_sample_dataset(logger=logger)
        
        # Step 2: Load and preprocess data
        logger.info("Step 2: Loading and preprocessing data...")
        df = load_property_data(sample_file, logger=logger)
        processed_df = preprocess_property_data(df, logger=logger)
        
        # Step 3: Initialize classifier
        logger.info("Step 3: Initializing classifier...")
        classifier = FlexPropertyClassifier(processed_df, logger=logger)
        
        # Step 4: Run classification
        logger.info("Step 4: Running classification...")
        candidates = classifier.classify_flex_properties()
        
        # Step 5: Calculate scores and get top candidates
        logger.info("Step 5: Calculating scores...")
        top_candidates = classifier.get_top_candidates(n=5)
        
        # Step 6: Generate statistics
        logger.info("Step 6: Generating statistics...")
        stats = classifier.get_analysis_statistics()
        
        # Step 7: Export results
        logger.info("Step 7: Exporting results...")
        output_path = classifier.export_results()
        
        # Display summary
        print("\n🎉 Example Workflow Complete!")
        print(f"📊 Found {len(candidates)} flex candidates")
        print(f"⭐ Top score: {stats['score_statistics']['max_score']:.1f}")
        print(f"📁 Results saved to: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Example workflow failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Check if running as example workflow
    if len(sys.argv) == 1:
        print("Running example workflow...")
        success = run_example_workflow()
        if not success:
            sys.exit(1)
    else:
        main()