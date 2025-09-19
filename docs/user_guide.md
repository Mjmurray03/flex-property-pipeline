# Scalable Multi-File Pipeline User Guide

## Overview

The Scalable Multi-File Pipeline is a powerful system for processing multiple Excel files containing property data and identifying flex property investment opportunities. It automatically discovers files, processes them in parallel, aggregates results, removes duplicates, and generates comprehensive reports.

## Key Features

- **Batch Processing**: Process multiple Excel files simultaneously
- **Flex Property Classification**: Automatically identify properties with development potential
- **Deduplication**: Remove duplicate properties across multiple files
- **Error Recovery**: Retry failed files and continue processing
- **Performance Optimization**: Memory-efficient processing for large datasets
- **Comprehensive Reporting**: Detailed statistics and analysis
- **Multiple Output Formats**: Excel and CSV export options

## Quick Start

### 1. Basic Usage

The simplest way to run the pipeline:

```bash
python scalable_pipeline_cli.py
```

This will:
- Look for Excel files in `data/raw/`
- Process all found files
- Save results to `data/exports/all_flex_properties.xlsx`

### 2. Specify Input and Output

```bash
python scalable_pipeline_cli.py -i /path/to/excel/files -o /path/to/output.xlsx
```

### 3. Configuration File

Create a configuration file for repeated use:

```bash
# Generate default configuration
python scalable_pipeline_cli.py --create-config config/my_pipeline.yaml

# Use configuration file
python scalable_pipeline_cli.py -c config/my_pipeline.yaml
```

## Command Line Options

### Input/Output Options

- `-i, --input-folder`: Input folder containing Excel files (default: `data/raw`)
- `-o, --output-file`: Output Excel file path (default: `data/exports/all_flex_properties.xlsx`)
- `--file-pattern`: File pattern to match (default: `*.xlsx`)
- `--recursive`: Scan input folder recursively

### Processing Options

- `-w, --workers`: Number of worker threads (default: 4)
- `--batch-size`: Batch size for processing (default: 10)
- `--min-score`: Minimum flex score threshold (default: 4.0)
- `--timeout`: Processing timeout in minutes (default: 30)
- `--memory-limit`: Memory limit in GB (default: 4.0)

### Performance Options

- `--disable-performance-optimization`: Disable performance optimization
- `--optimization-level`: Choose from `fast`, `balanced`, `memory_efficient` (default: balanced)
- `--chunk-size`: Chunk size for large file processing (default: 10000)

### Error Recovery Options

- `--disable-error-recovery`: Disable error recovery and retry mechanisms
- `--resume-failed`: Resume processing of previously failed files
- `--error-report`: Generate error report from previous run
- `--error-log-path`: Path to error log file

### Filtering Options

- `--no-deduplication`: Disable duplicate property removal
- `--duplicate-fields`: Fields to use for duplicate detection (default: Address City State)

### Output Options

- `--no-csv`: Disable CSV export
- `--no-backup`: Disable backup of existing output files

## Configuration File Format

Configuration files can be in YAML or JSON format:

```yaml
# pipeline_config.yaml
input_folder: "data/raw"
output_file: "data/exports/all_flex_properties.xlsx"
file_pattern: "*.xlsx"
recursive_scan: false

processing:
  batch_size: 10
  max_workers: 4
  timeout_minutes: 30
  memory_limit_gb: 4.0

filtering:
  min_flex_score: 4.0
  enable_deduplication: true
  duplicate_fields: ["Address", "City", "State"]

output_options:
  enable_csv_export: true
  backup_existing: true

logging:
  level: "INFO"
  file: "logs/pipeline.log"
```

## Input File Requirements

### Required Columns

Your Excel files should contain these columns (case-insensitive):

- **Address/Site Address**: Property address
- **City**: Property city
- **State**: Property state
- **Acres/Lot Size**: Lot size in acres
- **Zoning**: Zoning classification
- **Improvement Value**: Building/improvement value
- **Land Market Value**: Land value
- **Total Market Value**: Total assessed value

### Optional Columns

- **Parcel ID**: Unique property identifier
- **ZIP Code**: Property ZIP code
- **Property Type**: Type of property
- **Year Built**: Construction year

### Example File Structure

```
| Parcel ID | Site Address    | City        | State | Acres | Zoning | Improvement Value | Land Market Value | Total Market Value |
|-----------|-----------------|-------------|-------|-------|--------|-------------------|-------------------|-------------------|
| P001234   | 123 Main St     | Springfield | IL    | 0.25  | R1     | 150000           | 75000            | 225000           |
| P001235   | 456 Oak Ave     | Springfield | IL    | 0.30  | C1     | 80000            | 120000           | 200000           |
```

## Understanding Flex Properties

The pipeline identifies "flex properties" - properties with development or redevelopment potential based on:

### Scoring Criteria

1. **Zoning Flexibility**: Commercial, mixed-use, or PUD zoning
2. **Land-to-Improvement Ratio**: Higher land value relative to improvements
3. **Lot Size**: Adequate size for development
4. **Location Factors**: Urban or suburban locations
5. **Market Conditions**: Assessed vs. market value ratios

### Flex Score Ranges

- **8-10**: Excellent flex potential
- **6-8**: Good flex potential  
- **4-6**: Moderate flex potential
- **Below 4**: Limited flex potential (filtered out by default)

## Output Files

### Main Output File

The primary output is an Excel file containing all identified flex properties with:

- Original property data
- Calculated flex score
- Source file information
- Processing metadata

### CSV Export

When enabled, creates a CSV version of the Excel output for compatibility with other tools.

### Reports and Logs

- **Processing Log**: Detailed execution log in `logs/`
- **Error Report**: Summary of any processing errors
- **Export Metadata**: JSON file with processing statistics

## Common Use Cases

### 1. Regular Property Analysis

```bash
# Daily processing of new property exports
python scalable_pipeline_cli.py \
  -i /data/daily_exports \
  -o /reports/daily_flex_analysis.xlsx \
  --min-score 6.0
```

### 2. Large Dataset Processing

```bash
# Process large datasets with memory optimization
python scalable_pipeline_cli.py \
  --optimization-level memory_efficient \
  --memory-limit 8.0 \
  --workers 8
```

### 3. Error Recovery

```bash
# Resume failed files from previous run
python scalable_pipeline_cli.py --resume-failed

# Generate error report
python scalable_pipeline_cli.py --error-report
```

### 4. Custom Filtering

```bash
# High-quality candidates only, no duplicates
python scalable_pipeline_cli.py \
  --min-score 7.0 \
  --duplicate-fields Address City State ZIP
```

## Troubleshooting

### Common Issues

#### "No files found to process"
- Check input folder path
- Verify file pattern (default: `*.xlsx`)
- Use `--recursive` for subdirectories

#### "Memory usage critical"
- Reduce `--workers` count
- Use `--optimization-level memory_efficient`
- Increase `--memory-limit`
- Process files in smaller batches

#### "Processing timeout"
- Increase `--timeout` value
- Reduce concurrent workers
- Check for corrupted files

#### "Required columns not found"
- Verify Excel file column names
- Check for merged headers or empty rows
- Ensure consistent column naming across files

### Performance Optimization

#### For Speed
```bash
python scalable_pipeline_cli.py \
  --optimization-level fast \
  --workers 8 \
  --disable-error-recovery
```

#### For Memory Efficiency
```bash
python scalable_pipeline_cli.py \
  --optimization-level memory_efficient \
  --chunk-size 5000 \
  --memory-limit 2.0
```

#### For Large Datasets
```bash
python scalable_pipeline_cli.py \
  --workers 6 \
  --batch-size 20 \
  --timeout 60 \
  --memory-limit 8.0
```

### Error Recovery

The pipeline includes robust error recovery:

1. **Automatic Retry**: Failed files are automatically retried with exponential backoff
2. **Error Categorization**: Errors are classified and handled appropriately
3. **Graceful Degradation**: Processing continues even if some files fail
4. **Resume Capability**: Resume processing from where it left off

### Monitoring and Logging

- **Progress Tracking**: Real-time progress updates during processing
- **Memory Monitoring**: Automatic memory usage warnings
- **Performance Metrics**: Detailed timing and throughput statistics
- **Error Logging**: Comprehensive error tracking and reporting

## Best Practices

### File Organization

1. **Consistent Naming**: Use consistent file naming conventions
2. **Clean Data**: Remove empty rows and merged cells from Excel files
3. **Backup Originals**: Keep backups of original files
4. **Organize by Region**: Group files by geographic region for better deduplication

### Performance

1. **Right-size Workers**: Use 1-2 workers per CPU core
2. **Monitor Memory**: Watch memory usage with large datasets
3. **Batch Processing**: Process files in reasonable batch sizes
4. **Regular Cleanup**: Clean up old output files and logs

### Data Quality

1. **Validate Inputs**: Check file formats before processing
2. **Review Results**: Manually review high-scoring properties
3. **Update Regularly**: Process new data regularly for current market conditions
4. **Cross-reference**: Compare results across different data sources

## Advanced Usage

### Python API

For programmatic use:

```python
from pipeline.scalable_flex_pipeline import ScalableFlexPipeline, PipelineConfiguration
from run_scalable_pipeline import integrate_pipeline_components, run_complete_pipeline

# Create configuration
config = PipelineConfiguration(
    input_folder="data/raw",
    output_file="data/exports/results.xlsx",
    max_workers=4,
    min_flex_score=5.0
)

# Create and run pipeline
pipeline = ScalableFlexPipeline(config=config)
integrate_pipeline_components(pipeline)
results = run_complete_pipeline(pipeline)

print(f"Processed {results['files_processed']} files")
print(f"Found {results['flex_properties_found']} flex properties")
```

### Custom Processing

Extend the pipeline for custom analysis:

```python
from pipeline.batch_processor import BatchProcessor

def custom_progress_callback(progress_info):
    print(f"Progress: {progress_info['progress_pct']:.1f}%")

processor = BatchProcessor(
    max_workers=4,
    progress_callback=custom_progress_callback
)

results = processor.process_files(file_paths)
```

## Support and Resources

### Getting Help

1. **Dry Run**: Use `--dry-run` to validate configuration
2. **Verbose Logging**: Use `--verbose` for detailed output
3. **Error Reports**: Generate error reports for troubleshooting
4. **Documentation**: Refer to this guide and inline help

### Performance Monitoring

Monitor pipeline performance with:

```bash
# Get memory usage
python -c "from pipeline.performance_optimizer import get_memory_usage; print(get_memory_usage())"

# Performance summary after processing
python scalable_pipeline_cli.py --verbose
```

This guide covers the essential aspects of using the Scalable Multi-File Pipeline. For specific technical details, refer to the developer documentation and API reference.