#!/usr/bin/env python3
"""
Command-Line Interface for Scalable Multi-File Pipeline
Provides CLI access to the scalable flex property classification pipeline
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any

from pipeline.scalable_flex_pipeline import ScalableFlexPipeline, PipelineConfiguration
from utils.logger import setup_logging


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser"""
    parser = argparse.ArgumentParser(
        description="Scalable Multi-File Flex Property Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python scalable_pipeline_cli.py

  # Specify input and output paths
  python scalable_pipeline_cli.py -i data/raw -o data/exports/results.xlsx

  # Use configuration file
  python scalable_pipeline_cli.py -c config/my_pipeline.yaml

  # Dry run to validate without processing
  python scalable_pipeline_cli.py --dry-run

  # Run with custom workers and score threshold
  python scalable_pipeline_cli.py -w 8 --min-score 6.0

  # Generate default configuration file
  python scalable_pipeline_cli.py --create-config config/pipeline.yaml

  # Resume processing of failed files
  python scalable_pipeline_cli.py --resume-failed

  # Generate error report from previous run
  python scalable_pipeline_cli.py --error-report

  # Disable error recovery for faster processing
  python scalable_pipeline_cli.py --disable-error-recovery

  # Use memory-efficient optimization for large datasets
  python scalable_pipeline_cli.py --optimization-level memory_efficient

  # Disable performance optimization for debugging
  python scalable_pipeline_cli.py --disable-performance-optimization
        """
    )

    # Input/Output options
    io_group = parser.add_argument_group('Input/Output Options')
    io_group.add_argument(
        '-i', '--input-folder',
        type=str,
        default='data/raw',
        help='Input folder containing Excel files (default: data/raw)'
    )
    io_group.add_argument(
        '-o', '--output-file',
        type=str,
        default='data/exports/all_flex_properties.xlsx',
        help='Output Excel file path (default: data/exports/all_flex_properties.xlsx)'
    )
    io_group.add_argument(
        '--file-pattern',
        type=str,
        default='*.xlsx',
        help='File pattern to match (default: *.xlsx)'
    )
    io_group.add_argument(
        '--recursive',
        action='store_true',
        help='Scan input folder recursively'
    )

    # Processing options
    proc_group = parser.add_argument_group('Processing Options')
    proc_group.add_argument(
        '-w', '--workers',
        type=int,
        default=4,
        help='Number of worker threads (default: 4)'
    )
    proc_group.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Batch size for processing (default: 10)'
    )
    proc_group.add_argument(
        '--min-score',
        type=float,
        default=4.0,
        help='Minimum flex score threshold (default: 4.0)'
    )
    proc_group.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Processing timeout in minutes (default: 30)'
    )
    proc_group.add_argument(
        '--memory-limit',
        type=float,
        default=4.0,
        help='Memory limit in GB (default: 4.0)'
    )

    # Performance options
    perf_group = parser.add_argument_group('Performance Options')
    perf_group.add_argument(
        '--disable-performance-optimization',
        action='store_true',
        help='Disable performance optimization and memory management'
    )
    perf_group.add_argument(
        '--optimization-level',
        choices=['fast', 'balanced', 'memory_efficient'],
        default='balanced',
        help='Performance optimization level (default: balanced)'
    )
    perf_group.add_argument(
        '--chunk-size',
        type=int,
        default=10000,
        help='Chunk size for large file processing (default: 10000)'
    )

    # Filtering options
    filter_group = parser.add_argument_group('Filtering Options')
    filter_group.add_argument(
        '--no-deduplication',
        action='store_true',
        help='Disable duplicate property removal'
    )
    filter_group.add_argument(
        '--duplicate-fields',
        nargs='+',
        default=['Address', 'City', 'State'],
        help='Fields to use for duplicate detection (default: Address City State)'
    )

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--no-csv',
        action='store_true',
        help='Disable CSV export'
    )
    output_group.add_argument(
        '--no-backup',
        action='store_true',
        help='Disable backup of existing output files'
    )

    # Configuration options
    config_group = parser.add_argument_group('Configuration Options')
    config_group.add_argument(
        '-c', '--config',
        type=str,
        help='Configuration file path (YAML or JSON)'
    )
    config_group.add_argument(
        '--create-config',
        type=str,
        metavar='PATH',
        help='Create default configuration file at specified path'
    )
    config_group.add_argument(
        '--save-config',
        type=str,
        metavar='PATH',
        help='Save current configuration to file'
    )

    # Error Recovery options
    recovery_group = parser.add_argument_group('Error Recovery Options')
    recovery_group.add_argument(
        '--disable-error-recovery',
        action='store_true',
        help='Disable error recovery and retry mechanisms'
    )
    recovery_group.add_argument(
        '--resume-failed',
        action='store_true',
        help='Resume processing of previously failed files'
    )
    recovery_group.add_argument(
        '--error-report',
        action='store_true',
        help='Generate error report from previous run'
    )
    recovery_group.add_argument(
        '--error-log-path',
        type=str,
        default='logs/pipeline_errors.json',
        help='Path to error log file (default: logs/pipeline_errors.json)'
    )

    # Execution options
    exec_group = parser.add_argument_group('Execution Options')
    exec_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration and discover files without processing'
    )
    exec_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    exec_group.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    exec_group.add_argument(
        '--log-file',
        type=str,
        help='Log file path'
    )

    return parser


def create_configuration_from_args(args: argparse.Namespace) -> PipelineConfiguration:
    """Create pipeline configuration from command line arguments"""
    config = PipelineConfiguration(
        input_folder=args.input_folder,
        output_file=args.output_file,
        batch_size=args.batch_size,
        max_workers=args.workers,
        enable_deduplication=not args.no_deduplication,
        min_flex_score=args.min_score,
        progress_reporting=not args.quiet,
        log_level='DEBUG' if args.verbose else 'INFO',
        log_file=args.log_file,
        file_pattern=args.file_pattern,
        recursive_scan=args.recursive,
        memory_limit_gb=args.memory_limit,
        timeout_minutes=args.timeout,
        enable_csv_export=not args.no_csv,
        backup_existing=not args.no_backup,
        duplicate_fields=args.duplicate_fields
    )
    
    # Add error recovery settings (not part of PipelineConfiguration but used by CLI)
    config._enable_error_recovery = not args.disable_error_recovery
    config._error_log_path = args.error_log_path
    
    # Add performance optimization settings
    config._enable_performance_optimization = not args.disable_performance_optimization
    config._optimization_level = args.optimization_level
    config._chunk_size = args.chunk_size
    
    return config


def print_configuration_summary(config: PipelineConfiguration) -> None:
    """Print a summary of the pipeline configuration"""
    print("\n" + "="*60)
    print("PIPELINE CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Input Folder:      {config.input_folder}")
    print(f"Output File:       {config.output_file}")
    print(f"File Pattern:      {config.file_pattern}")
    print(f"Recursive Scan:    {config.recursive_scan}")
    print(f"Workers:           {config.max_workers}")
    print(f"Batch Size:        {config.batch_size}")
    print(f"Min Flex Score:    {config.min_flex_score}")
    print(f"Deduplication:     {config.enable_deduplication}")
    print(f"CSV Export:        {config.enable_csv_export}")
    print(f"Backup Files:      {config.backup_existing}")
    print(f"Memory Limit:      {config.memory_limit_gb} GB")
    print(f"Timeout:           {config.timeout_minutes} minutes")
    print("="*60)


def print_results_summary(results: Dict[str, Any]) -> None:
    """Print a summary of pipeline execution results"""
    print("\n" + "="*60)
    print("PIPELINE EXECUTION RESULTS")
    print("="*60)
    
    if results.get('success', False):
        print(f"Status:            SUCCESS")
        print(f"Execution Time:    {results.get('execution_time', 0):.2f} seconds")
        print(f"Files Discovered:  {results.get('files_discovered', 0)}")
        print(f"Files Processed:   {results.get('files_processed', 0)}")
        print(f"Flex Properties:   {results.get('flex_properties_found', 0)}")
        print(f"Output File:       {results.get('output_file', 'N/A')}")
    else:
        print(f"Status:            FAILED")
        print(f"Error:             {results.get('error', 'Unknown error')}")
    
    print("="*60)


def run_dry_run(pipeline: ScalableFlexPipeline) -> bool:
    """
    Run dry-run validation without processing files
    
    Args:
        pipeline: Configured pipeline instance
        
    Returns:
        True if validation passes, False otherwise
    """
    print("\n" + "="*60)
    print("DRY RUN - VALIDATION ONLY")
    print("="*60)
    
    try:
        # Validate configuration
        if not pipeline.validate_configuration():
            print("❌ Configuration validation failed")
            return False
        
        print("✅ Configuration validation passed")
        
        # Check input folder
        input_path = Path(pipeline.config.input_folder)
        if not input_path.exists():
            print(f"❌ Input folder does not exist: {input_path}")
            return False
        
        print(f"✅ Input folder exists: {input_path}")
        
        # Discover files (this would be implemented in file discovery task)
        excel_files = list(input_path.glob(pipeline.config.file_pattern))
        if pipeline.config.recursive_scan:
            excel_files = list(input_path.rglob(pipeline.config.file_pattern))
        
        print(f"✅ Found {len(excel_files)} Excel files to process")
        
        if excel_files:
            print("\nFiles to be processed:")
            for i, file_path in enumerate(excel_files[:10], 1):  # Show first 10
                print(f"  {i}. {file_path}")
            if len(excel_files) > 10:
                print(f"  ... and {len(excel_files) - 10} more files")
        
        # Check output directory
        output_path = Path(pipeline.config.output_file)
        if not output_path.parent.exists():
            print(f"ℹ️  Output directory will be created: {output_path.parent}")
        else:
            print(f"✅ Output directory exists: {output_path.parent}")
        
        print("\n✅ Dry run completed successfully - ready to process!")
        return True
        
    except Exception as e:
        print(f"❌ Dry run failed: {e}")
        return False


def handle_error_report(error_log_path: str, quiet: bool) -> int:
    """
    Handle error report generation
    
    Args:
        error_log_path: Path to error log file
        quiet: Whether to suppress output
        
    Returns:
        Exit code
    """
    try:
        from pipeline.error_recovery import ErrorRecoveryManager
        
        recovery_manager = ErrorRecoveryManager(error_log_path=error_log_path)
        error_data = recovery_manager.load_error_log()
        
        if not error_data:
            print("❌ No error log found or error log is empty")
            return 1
        
        if not quiet:
            print("\n" + "="*60)
            print("ERROR REPORT")
            print("="*60)
            
            summary = error_data.get('summary', {})
            print(f"Total Errors:      {summary.get('total_errors', 0)}")
            print(f"Recovered Files:   {summary.get('recovered_files', 0)}")
            print(f"Permanent Failures: {summary.get('permanent_failures', 0)}")
            print(f"Recovery Rate:     {summary.get('recovery_rate', 0):.1%}")
            
            # Show error breakdown
            breakdown = error_data.get('error_breakdown', {})
            if breakdown.get('by_category'):
                print(f"\nError Categories:")
                for category, count in breakdown['by_category'].items():
                    print(f"  {category}: {count}")
            
            # Show recommendations
            recommendations = error_data.get('troubleshooting_recommendations', [])
            if recommendations:
                print(f"\nTroubleshooting Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec['category']}: {rec['recommendation']}")
            
            print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to generate error report: {e}")
        return 1


def handle_resume_failed(pipeline: 'ScalableFlexPipeline', error_log_path: str, quiet: bool) -> int:
    """
    Handle resuming failed files
    
    Args:
        pipeline: Pipeline instance
        error_log_path: Path to error log file
        quiet: Whether to suppress output
        
    Returns:
        Exit code
    """
    try:
        from pipeline.error_recovery import ErrorRecoveryManager
        from run_scalable_pipeline import integrate_pipeline_components
        
        # Integrate pipeline components
        integrate_pipeline_components(pipeline)
        
        # Load error log to get failed files
        recovery_manager = ErrorRecoveryManager(error_log_path=error_log_path)
        error_data = recovery_manager.load_error_log()
        
        if not error_data or not error_data.get('failed_files'):
            print("❌ No failed files found to resume")
            return 1
        
        failed_files = [f['file_path'] for f in error_data['failed_files']]
        
        if not quiet:
            print(f"\n🔄 Resuming processing of {len(failed_files)} failed files...")
        
        # Resume processing using batch processor
        if hasattr(pipeline, 'batch_processor') and pipeline.batch_processor:
            resume_results = pipeline.batch_processor.resume_failed_files()
            
            if not quiet:
                print(f"✅ Resume completed:")
                print(f"   Files resumed: {resume_results.get('resumed_files', 0)}")
                print(f"   Successful: {resume_results.get('successful', 0)}")
                print(f"   Still failed: {resume_results.get('failed', 0)}")
                print(f"   Recovery rate: {resume_results.get('recovery_rate', 0):.1%}")
            
            return 0 if resume_results.get('successful', 0) > 0 else 1
        else:
            print("❌ Pipeline components not properly initialized")
            return 1
        
    except Exception as e:
        print(f"❌ Failed to resume failed files: {e}")
        return 1


def main() -> int:
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Handle configuration file creation
        if args.create_config:
            config = PipelineConfiguration()
            config.to_file(args.create_config)
            print(f"✅ Default configuration created at: {args.create_config}")
            return 0
        
        # Load configuration
        if args.config:
            # Load from configuration file
            config = PipelineConfiguration.from_file(args.config)
            print(f"✅ Configuration loaded from: {args.config}")
        else:
            # Create from command line arguments
            config = create_configuration_from_args(args)
        
        # Save configuration if requested
        if args.save_config:
            config.to_file(args.save_config)
            print(f"✅ Configuration saved to: {args.save_config}")
        
        # Create pipeline
        pipeline = ScalableFlexPipeline(config=config)
        
        # Print configuration summary
        if not args.quiet:
            print_configuration_summary(config)
        
        # Handle error report generation
        if args.error_report:
            return handle_error_report(args.error_log_path, args.quiet)
        
        # Handle resume failed files
        if args.resume_failed:
            return handle_resume_failed(pipeline, args.error_log_path, args.quiet)
        
        # Handle dry run
        if args.dry_run:
            success = run_dry_run(pipeline)
            return 0 if success else 1
        
        # Run the pipeline
        print(f"\n🚀 Starting pipeline execution...")
        results = pipeline.run_pipeline()
        
        # Print results
        if not args.quiet:
            print_results_summary(results)
        
        # Return appropriate exit code
        return 0 if results.get('success', False) else 1
        
    except KeyboardInterrupt:
        print("\n❌ Pipeline execution interrupted by user")
        return 130
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return 2
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return 3
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())