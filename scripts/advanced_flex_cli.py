"""
Advanced Flex Property Classifier Command Line Interface
Provides comprehensive CLI for advanced features and batch processing
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logging
from utils.flex_config_manager import FlexConfigManager, create_sample_configs
from processors.advanced_flex_classifier import AdvancedFlexClassifier, create_advanced_classifier_from_config
from utils.batch_processor import FlexBatchProcessor, create_batch_processor_from_config


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description='Advanced Flex Property Classifier - Enhanced property analysis with configurable scoring',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single Excel file with default configuration
  python advanced_flex_cli.py single data/properties.xlsx
  
  # Process multiple files in batch mode
  python advanced_flex_cli.py batch data/input_folder/ --output data/results/
  
  # Create custom configuration
  python advanced_flex_cli.py config create --type conservative
  
  # Process with custom configuration and analytics
  python advanced_flex_cli.py single data/properties.xlsx --config config/custom.yaml --analytics
  
  # Batch process with parallel processing
  python advanced_flex_cli.py batch data/input/ --parallel --workers 8
        """
    )
    
    # Global arguments
    parser.add_argument('--config', '-c', type=Path, 
                       help='Path to configuration file')
    parser.add_argument('--output', '-o', type=Path, default=Path('data/exports/advanced'),
                       help='Output directory (default: data/exports/advanced)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress non-error output')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single file processing
    single_parser = subparsers.add_parser('single', help='Process single Excel file')
    single_parser.add_argument('file', type=Path, help='Path to Excel file')
    single_parser.add_argument('--analytics', action='store_true',
                              help='Perform advanced analytics (geographic, size distribution)')
    single_parser.add_argument('--export-formats', nargs='+', 
                              choices=['xlsx', 'csv', 'json', 'parquet'],
                              default=['xlsx'], help='Export formats')
    single_parser.add_argument('--batch-size', type=int,
                              help='Batch size for processing (overrides config)')
    
    # Batch processing
    batch_parser = subparsers.add_parser('batch', help='Process multiple Excel files')
    batch_parser.add_argument('input_paths', nargs='+', type=Path,
                             help='Input file or directory paths')
    batch_parser.add_argument('--recursive', '-r', action='store_true', default=True,
                             help='Search directories recursively')
    batch_parser.add_argument('--parallel', action='store_true',
                             help='Enable parallel processing')
    batch_parser.add_argument('--workers', type=int,
                             help='Number of worker threads (overrides config)')
    batch_parser.add_argument('--retry-failed', action='store_true',
                             help='Retry failed files after initial processing')
    batch_parser.add_argument('--file-patterns', nargs='+', 
                             default=['*.xlsx', '*.xls', '*.xlsm'],
                             help='File patterns to match')
    
    # Configuration management
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_subparsers = config_parser.add_subparsers(dest='config_action')
    
    # Create configuration
    create_config_parser = config_subparsers.add_parser('create', help='Create configuration')
    create_config_parser.add_argument('--type', choices=['default', 'conservative', 'aggressive', 'performance', 'analysis'],
                                     default='default', help='Configuration type')
    create_config_parser.add_argument('--output-path', type=Path,
                                     help='Output path for configuration file')
    
    # Validate configuration
    validate_config_parser = config_subparsers.add_parser('validate', help='Validate configuration')
    validate_config_parser.add_argument('config_file', type=Path, help='Configuration file to validate')
    
    # Show configuration
    show_config_parser = config_subparsers.add_parser('show', help='Show configuration')
    show_config_parser.add_argument('config_file', nargs='?', type=Path, help='Configuration file to show')
    
    # Performance testing
    perf_parser = subparsers.add_parser('performance', help='Performance testing and benchmarking')
    perf_parser.add_argument('--dataset-size', type=int, default=10000,
                            help='Size of synthetic dataset for testing')
    perf_parser.add_argument('--iterations', type=int, default=3,
                            help='Number of test iterations')
    perf_parser.add_argument('--test-batch', action='store_true',
                            help='Test batch processing performance')
    
    return parser


def setup_logging_from_args(args) -> None:
    """Set up logging based on command line arguments"""
    if args.quiet:
        level = 'ERROR'
    elif args.verbose:
        level = 'DEBUG'
    else:
        level = 'INFO'
    
    setup_logging(name='advanced_flex_cli', level=level)


def progress_callback(completed: int, total: int, current_item: str, elapsed_time: float):
    """Progress callback for batch processing"""
    percentage = (completed / total * 100) if total > 0 else 0
    rate = completed / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\rProgress: {completed}/{total} ({percentage:.1f}%) - {rate:.1f} files/sec - {current_item}", 
          end='', flush=True)
    
    if completed == total:
        print()  # New line when complete


def handle_single_file_processing(args) -> int:
    """Handle single file processing command"""
    try:
        logger = setup_logging(name='single_file', level='INFO')
        
        if not args.file.exists():
            print(f"Error: File {args.file} does not exist")
            return 1
        
        print(f"Processing file: {args.file}")
        
        # Load data
        data = pd.read_excel(args.file)
        print(f"Loaded {len(data)} properties from {args.file}")
        
        # Create classifier
        classifier = create_advanced_classifier_from_config(data, args.config)
        
        # Override batch size if specified
        if args.batch_size:
            classifier.config.advanced_settings.batch_size = args.batch_size
        
        # Override export formats if specified
        if args.export_formats:
            classifier.config.advanced_settings.export_formats = args.export_formats
        
        # Set progress callback
        def single_progress(current, total, message):
            if total > 1:  # Only show progress for batch processing
                percentage = (current / total * 100) if total > 0 else 0
                print(f"\rProcessing: {current}/{total} ({percentage:.1f}%) - {message}", 
                      end='', flush=True)
                if current == total:
                    print()
        
        classifier.set_progress_callback(single_progress)
        
        # Process
        start_time = time.time()
        candidates = classifier.classify_flex_properties_batch()
        processing_time = time.time() - start_time
        
        print(f"Found {len(candidates)} flex candidates in {processing_time:.2f} seconds")
        
        # Perform analytics if requested
        if args.analytics and len(candidates) > 0:
            print("Performing advanced analytics...")
            
            geo_analysis = classifier.perform_geographic_analysis()
            size_analysis = classifier.perform_size_distribution_analysis()
            
            print(f"Geographic analysis: {len(geo_analysis.state_distribution)} states")
            print(f"Size analysis: Building size distribution calculated")
        
        # Export results
        args.output.mkdir(parents=True, exist_ok=True)
        exported_files = classifier.export_advanced_results(args.output, include_analytics=args.analytics)
        
        print(f"Results exported to {args.output}:")
        for format_type, file_path in exported_files.items():
            print(f"  {format_type}: {file_path}")
        
        # Show performance report
        performance = classifier.get_performance_report()
        if performance.get('processing_metrics', {}).get('processing_rate', 0) > 0:
            rate = performance['processing_metrics']['processing_rate']
            print(f"Processing rate: {rate:.1f} properties/second")
        
        return 0
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return 1


def handle_batch_processing(args) -> int:
    """Handle batch processing command"""
    try:
        logger = setup_logging(name='batch_processing', level='INFO')
        
        print(f"Starting batch processing of {len(args.input_paths)} input paths")
        
        # Create batch processor
        processor = create_batch_processor_from_config(args.config)
        
        # Override settings if specified
        if args.workers:
            processor.config.advanced_settings.max_workers = args.workers
        
        if args.parallel:
            processor.config.advanced_settings.parallel_processing = True
        
        # Discover files
        print("Discovering Excel files...")
        files = processor.discover_excel_files(
            args.input_paths, 
            recursive=args.recursive,
            file_patterns=args.file_patterns
        )
        
        if not files:
            print("No Excel files found in specified paths")
            return 1
        
        print(f"Found {len(files)} Excel files to process")
        
        # Process files
        args.output.mkdir(parents=True, exist_ok=True)
        
        summary = processor.process_files(
            files,
            output_dir=args.output,
            max_workers=args.workers,
            progress_callback=progress_callback
        )
        
        # Print summary
        print(f"\nBatch processing complete:")
        print(f"  Total files: {summary.total_files}")
        print(f"  Successful: {summary.successful_files}")
        print(f"  Failed: {summary.failed_files}")
        print(f"  Total properties: {summary.total_properties:,}")
        print(f"  Total candidates: {summary.total_candidates:,}")
        print(f"  Processing time: {summary.total_processing_time:.2f} seconds")
        print(f"  Processing rate: {summary.properties_per_second:.1f} properties/second")
        
        if summary.error_summary:
            print(f"  Error summary: {dict(summary.error_summary)}")
        
        # Retry failed files if requested
        if args.retry_failed and summary.failed_files > 0:
            print(f"\nRetrying {summary.failed_files} failed files...")
            retry_summary = processor.retry_failed_files(progress_callback=progress_callback)
            
            print(f"Retry results:")
            print(f"  Additional successful: {retry_summary.successful_files}")
            print(f"  Still failed: {retry_summary.failed_files}")
        
        return 0 if summary.failed_files == 0 else 1
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return 1


def handle_config_management(args) -> int:
    """Handle configuration management commands"""
    try:
        config_manager = FlexConfigManager(args.config)
        
        if args.config_action == 'create':
            print(f"Creating {args.type} configuration...")
            
            if args.type == 'default':
                config = config_manager.load_config()
            else:
                # Create sample configurations
                samples = create_sample_configs()
                if args.type in samples:
                    config = samples[args.type]
                else:
                    print(f"Unknown configuration type: {args.type}")
                    return 1
            
            # Determine output path
            if args.output_path:
                output_path = args.output_path
            else:
                output_path = Path(f"config/flex_classifier_{args.type}.yaml")
            
            config_manager.save_config(config, output_path)
            print(f"Configuration saved to {output_path}")
            
        elif args.config_action == 'validate':
            print(f"Validating configuration: {args.config_file}")
            
            if not args.config_file.exists():
                print(f"Configuration file {args.config_file} does not exist")
                return 1
            
            config = config_manager.load_config(args.config_file)
            issues = config_manager.validate_config(config)
            
            if issues:
                print("Configuration validation issues:")
                for issue in issues:
                    print(f"  - {issue}")
                return 1
            else:
                print("Configuration is valid")
                
        elif args.config_action == 'show':
            config_file = args.config_file or config_manager.config_path
            
            if config_file and config_file.exists():
                config = config_manager.load_config(config_file)
                print(f"Configuration from {config_file}:")
                print(f"  Version: {config.version}")
                print(f"  Max flex score: {config.max_flex_score}")
                print(f"  Min building sqft: {config.filtering_criteria.min_building_sqft:,}")
                print(f"  Lot size range: {config.filtering_criteria.min_lot_acres}-{config.filtering_criteria.max_lot_acres} acres")
                print(f"  Batch processing: {config.advanced_settings.enable_batch_processing}")
                print(f"  Parallel processing: {config.advanced_settings.parallel_processing}")
                print(f"  Export formats: {config.advanced_settings.export_formats}")
            else:
                print("No configuration file found")
                return 1
        
        return 0
        
    except Exception as e:
        print(f"Error in configuration management: {e}")
        return 1


def handle_performance_testing(args) -> int:
    """Handle performance testing command"""
    try:
        import numpy as np
        
        print(f"Running performance tests with {args.dataset_size} properties")
        
        # Create synthetic dataset
        np.random.seed(42)
        data = pd.DataFrame({
            'Property Type': np.random.choice(['Industrial', 'Warehouse', 'Flex', 'Office'], args.dataset_size),
            'Building SqFt': np.random.randint(15000, 200000, args.dataset_size),
            'Lot Size Acres': np.random.uniform(0.3, 25.0, args.dataset_size),
            'Year Built': np.random.randint(1970, 2020, args.dataset_size),
            'Occupancy': np.random.uniform(60, 100, args.dataset_size),
            'City': np.random.choice(['Dallas', 'Houston', 'Austin'], args.dataset_size),
            'State': ['TX'] * args.dataset_size
        })
        
        # Test standard processing
        print("Testing standard processing...")
        times = []
        
        for i in range(args.iterations):
            classifier = create_advanced_classifier_from_config(data, args.config)
            
            start_time = time.time()
            candidates = classifier.classify_flex_properties()
            end_time = time.time()
            
            processing_time = end_time - start_time
            times.append(processing_time)
            
            rate = len(data) / processing_time
            print(f"  Iteration {i+1}: {processing_time:.2f}s, {rate:.1f} properties/sec, {len(candidates)} candidates")
        
        avg_time = sum(times) / len(times)
        avg_rate = len(data) / avg_time
        print(f"Standard processing average: {avg_time:.2f}s, {avg_rate:.1f} properties/sec")
        
        # Test batch processing if requested
        if args.test_batch:
            print("\nTesting batch processing...")
            batch_times = []
            
            for i in range(args.iterations):
                classifier = create_advanced_classifier_from_config(data, args.config)
                
                start_time = time.time()
                candidates = classifier.classify_flex_properties_batch()
                end_time = time.time()
                
                processing_time = end_time - start_time
                batch_times.append(processing_time)
                
                rate = len(data) / processing_time
                print(f"  Iteration {i+1}: {processing_time:.2f}s, {rate:.1f} properties/sec, {len(candidates)} candidates")
            
            avg_batch_time = sum(batch_times) / len(batch_times)
            avg_batch_rate = len(data) / avg_batch_time
            print(f"Batch processing average: {avg_batch_time:.2f}s, {avg_batch_rate:.1f} properties/sec")
            
            # Compare performance
            improvement = (avg_time - avg_batch_time) / avg_time * 100
            print(f"Batch processing improvement: {improvement:.1f}%")
        
        return 0
        
    except Exception as e:
        print(f"Error in performance testing: {e}")
        return 1


def main() -> int:
    """Main CLI entry point"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    setup_logging_from_args(args)
    
    try:
        if args.command == 'single':
            return handle_single_file_processing(args)
        elif args.command == 'batch':
            return handle_batch_processing(args)
        elif args.command == 'config':
            return handle_config_management(args)
        elif args.command == 'performance':
            return handle_performance_testing(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())