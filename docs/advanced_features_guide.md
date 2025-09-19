# Advanced Flex Property Classifier Features Guide

## Overview

The Advanced Flex Property Classifier extends the basic classifier with powerful features for enterprise-scale property analysis:

- **Configurable Scoring**: Customize scoring criteria and weights
- **Batch Processing**: Process multiple Excel files efficiently
- **Advanced Analytics**: Geographic and size distribution analysis
- **Performance Optimization**: Parallel processing and memory management
- **Multiple Export Formats**: Excel, CSV, JSON, Parquet support

## Configuration Management

### Configuration Structure

The classifier uses a hierarchical configuration system:

```yaml
# Scoring weights and criteria
scoring_weights:
  building_size_weight: 1.0
  property_type_weight: 1.0
  lot_size_weight: 1.0
  age_condition_weight: 1.0
  occupancy_weight: 1.0
  
  # Configurable scoring ranges
  building_size_ranges:
    20k_to_50k: {min: 20000, max: 50000, score: 3.0}
    50k_to_100k: {min: 50000, max: 100000, score: 2.0}
    100k_to_200k: {min: 100000, max: 200000, score: 1.0}
  
  property_type_scores:
    flex: 3.0
    warehouse: 2.5
    distribution: 2.5
    light industrial: 2.0
    industrial: 1.5

# Filtering criteria
filtering_criteria:
  min_building_sqft: 20000
  min_lot_acres: 0.5
  max_lot_acres: 20.0
  industrial_keywords:
    - industrial
    - warehouse
    - distribution
    - flex

# Advanced settings
advanced_settings:
  enable_batch_processing: true
  batch_size: 1000
  parallel_processing: false
  max_workers: 4
  export_formats: [xlsx, csv, json]
```

### Creating Configurations

#### Using the CLI

```bash
# Create default configuration
python scripts/advanced_flex_cli.py config create --type default

# Create specialized configurations
python scripts/advanced_flex_cli.py config create --type conservative
python scripts/advanced_flex_cli.py config create --type aggressive
python scripts/advanced_flex_cli.py config create --type performance
```

#### Programmatically

```python
from utils.flex_config_manager import FlexConfigManager

# Create configuration manager
config_manager = FlexConfigManager()

# Create custom configuration
config = config_manager.create_custom_config(
    scoring_adjustments={
        'building_size_weight': 1.5,
        'property_type_weight': 2.0
    },
    filtering_adjustments={
        'min_building_sqft': 30000
    },
    advanced_adjustments={
        'enable_batch_processing': True,
        'parallel_processing': True
    }
)

# Save configuration
config_manager.save_config(config, Path('config/my_custom_config.yaml'))
```

### Configuration Types

#### Conservative Configuration
- Stricter filtering criteria
- Higher minimum building size (30k+ sqft)
- Smaller lot size range (1-15 acres)
- Emphasis on building size and property type scoring

#### Aggressive Configuration
- More lenient filtering criteria
- Lower minimum building size (15k+ sqft)
- Wider lot size range (0.3-25 acres)
- Balanced scoring weights

#### Performance Configuration
- Optimized for large datasets
- Larger batch sizes (2000+ properties)
- Parallel processing enabled
- Result caching enabled

#### Analysis Configuration
- All analytics features enabled
- Multiple export formats
- Detailed performance monitoring

## Advanced Classifier Usage

### Basic Usage with Configuration

```python
from processors.advanced_flex_classifier import create_advanced_classifier_from_config
import pandas as pd

# Load your data
data = pd.read_excel('properties.xlsx')

# Create classifier with custom config
classifier = create_advanced_classifier_from_config(
    data, 
    config_path=Path('config/my_config.yaml')
)

# Process with batch processing
candidates = classifier.classify_flex_properties_batch()

# Perform analytics
geo_analysis = classifier.perform_geographic_analysis()
size_analysis = classifier.perform_size_distribution_analysis()

# Export results
exported_files = classifier.export_advanced_results(
    output_dir=Path('results'),
    include_analytics=True
)
```

### Progress Tracking

```python
def progress_callback(current, total, message):
    percentage = (current / total * 100) if total > 0 else 0
    print(f"Progress: {current}/{total} ({percentage:.1f}%) - {message}")

classifier.set_progress_callback(progress_callback)
candidates = classifier.classify_flex_properties_batch()
```

### Advanced Scoring with Breakdown

```python
# Get detailed scoring breakdown
for _, property_row in candidates.head(10).iterrows():
    score, breakdown = classifier.calculate_flex_score_advanced(property_row)
    
    print(f"Property: {property_row.get('Property Name', 'Unknown')}")
    print(f"Total Score: {score:.2f}")
    print("Score Breakdown:")
    for factor, points in breakdown.items():
        print(f"  {factor}: {points:.2f}")
    print()
```

## Batch Processing

### Processing Multiple Files

#### Using the CLI

```bash
# Process all Excel files in a directory
python scripts/advanced_flex_cli.py batch data/input_folder/ --output results/

# Parallel processing with custom worker count
python scripts/advanced_flex_cli.py batch data/input/ --parallel --workers 8

# Retry failed files
python scripts/advanced_flex_cli.py batch data/input/ --retry-failed
```

#### Programmatically

```python
from utils.batch_processor import FlexBatchProcessor

# Create batch processor
processor = FlexBatchProcessor()

# Discover Excel files
files = processor.discover_excel_files(
    ['data/input1', 'data/input2'], 
    recursive=True
)

# Process with progress tracking
def progress_callback(completed, total, current_file, elapsed_time):
    print(f"Processing: {completed}/{total} - {current_file}")

summary = processor.process_files(
    files,
    output_dir=Path('results'),
    progress_callback=progress_callback
)

print(f"Processed {summary.successful_files}/{summary.total_files} files")
print(f"Found {summary.total_candidates:,} total candidates")
```

### Batch Processing Features

#### Parallel Processing
- Configurable worker threads
- Automatic load balancing
- Thread-safe progress tracking

#### Error Recovery
- Continue processing on individual file failures
- Detailed error categorization
- Retry failed files functionality

#### Result Aggregation
- Combine results from multiple files
- Automatic deduplication based on address/city/state
- Preserve highest scoring duplicates

## Advanced Analytics

### Geographic Analysis

Analyzes the geographic distribution of flex candidates:

```python
geo_analysis = classifier.perform_geographic_analysis()

print("State Distribution:")
for state, count in geo_analysis.state_distribution.items():
    print(f"  {state}: {count} properties")

print(f"Geographic Concentration Index: {geo_analysis.geographic_concentration:.3f}")
print("Top Markets:")
for market, count in geo_analysis.top_markets[:5]:
    print(f"  {market}: {count} properties")
```

#### Geographic Metrics
- **State/City/County Distribution**: Property counts by geographic area
- **Top Markets**: Highest concentration areas
- **Geographic Concentration**: Herfindahl index measuring market concentration

### Size Distribution Analysis

Analyzes building and lot size patterns:

```python
size_analysis = classifier.perform_size_distribution_analysis()

print("Building Size Distribution:")
for size_range, count in size_analysis.building_size_distribution.items():
    print(f"  {size_range}: {count} properties")

print("Optimal Size Ranges (for high-scoring properties):")
if 'building_size' in size_analysis.optimal_size_ranges:
    min_size, max_size = size_analysis.optimal_size_ranges['building_size']
    print(f"  Building: {min_size:,.0f} - {max_size:,.0f} sqft")
```

#### Size Metrics
- **Size Distribution**: Property counts by size buckets
- **Optimal Ranges**: Size ranges for highest-scoring properties
- **Correlations**: Relationships between size factors and scores

## Performance Optimization

### Batch Processing Settings

```python
# Optimize for large datasets
config.advanced_settings.enable_batch_processing = True
config.advanced_settings.batch_size = 2000  # Larger batches for better performance
config.advanced_settings.parallel_processing = True
config.advanced_settings.max_workers = 8  # Use more CPU cores
```

### Memory Management

```python
# Enable result caching for repeated operations
config.advanced_settings.cache_results = True

# Monitor memory usage
performance_report = classifier.get_performance_report()
print(f"Processing rate: {performance_report['processing_metrics']['processing_rate']:.1f} properties/sec")
```

### Performance Monitoring

```python
# Get detailed performance metrics
metrics = classifier.processing_metrics

print(f"Properties processed: {metrics.properties_processed:,}")
print(f"Processing time: {metrics.end_time - metrics.start_time:.2f} seconds")
print(f"Processing rate: {metrics.processing_rate:.1f} properties/second")
print(f"Memory usage: {metrics.memory_usage_mb:.1f} MB")
```

## Export Formats

### Multiple Format Export

```python
# Configure export formats
config.advanced_settings.export_formats = ['xlsx', 'csv', 'json', 'parquet']

# Export in all configured formats
exported_files = classifier.export_advanced_results(
    output_dir=Path('results'),
    include_analytics=True
)

for format_type, file_path in exported_files.items():
    print(f"{format_type.upper()}: {file_path}")
```

### Export Options

#### Excel (.xlsx)
- Formatted spreadsheet with all candidate data
- Compatible with business analysis tools
- Includes all available property columns

#### CSV (.csv)
- Plain text format for data analysis
- Compatible with all spreadsheet applications
- Lightweight and portable

#### JSON (.json)
- Structured data format for APIs
- Includes nested analytics data
- Machine-readable format

#### Parquet (.parquet)
- Columnar storage format
- Optimized for big data analytics
- Compressed and efficient

## Command Line Interface

### Complete CLI Reference

```bash
# Single file processing
python scripts/advanced_flex_cli.py single properties.xlsx --analytics --config config/custom.yaml

# Batch processing
python scripts/advanced_flex_cli.py batch input_folder/ --output results/ --parallel --workers 4

# Configuration management
python scripts/advanced_flex_cli.py config create --type performance
python scripts/advanced_flex_cli.py config validate config/my_config.yaml
python scripts/advanced_flex_cli.py config show config/my_config.yaml

# Performance testing
python scripts/advanced_flex_cli.py performance --dataset-size 5000 --test-batch
```

### CLI Options

#### Global Options
- `--config, -c`: Configuration file path
- `--output, -o`: Output directory
- `--verbose, -v`: Enable verbose logging
- `--quiet, -q`: Suppress non-error output

#### Single File Options
- `--analytics`: Perform advanced analytics
- `--export-formats`: Specify export formats
- `--batch-size`: Override batch size

#### Batch Processing Options
- `--recursive, -r`: Search directories recursively
- `--parallel`: Enable parallel processing
- `--workers`: Number of worker threads
- `--retry-failed`: Retry failed files
- `--file-patterns`: File patterns to match

## Best Practices

### Configuration Management
1. **Start with sample configurations** and customize as needed
2. **Validate configurations** before production use
3. **Version control** configuration files
4. **Test configuration changes** with small datasets first

### Performance Optimization
1. **Use batch processing** for datasets > 1000 properties
2. **Enable parallel processing** for multiple files
3. **Adjust batch size** based on available memory
4. **Monitor processing rates** and optimize accordingly

### Data Quality
1. **Validate input data** before processing
2. **Handle missing columns** gracefully
3. **Review analytics results** for data quality insights
4. **Use deduplication** for multi-file processing

### Production Deployment
1. **Use performance configurations** for large-scale processing
2. **Implement error handling** and retry logic
3. **Monitor resource usage** and adjust settings
4. **Archive results** with timestamps and metadata

## Troubleshooting

### Common Issues

#### Performance Issues
- **Slow processing**: Increase batch size, enable parallel processing
- **High memory usage**: Reduce batch size, disable result caching
- **CPU bottlenecks**: Adjust worker count based on CPU cores

#### Configuration Issues
- **Invalid configuration**: Use validation command to check
- **Missing columns**: Review column mapping in configuration
- **Unexpected results**: Verify scoring weights and criteria

#### File Processing Issues
- **File format errors**: Ensure Excel files are valid
- **Permission errors**: Check file and directory permissions
- **Large file issues**: Increase memory limits or use chunked processing

### Getting Help

1. **Check logs** for detailed error information
2. **Use validation tools** to verify configurations
3. **Run performance tests** to identify bottlenecks
4. **Review analytics results** for data quality insights

## Examples

See the `examples/` directory for complete working examples:

- `basic_advanced_usage.py`: Basic advanced classifier usage
- `batch_processing_example.py`: Multi-file batch processing
- `custom_configuration.py`: Creating custom configurations
- `analytics_example.py`: Advanced analytics and reporting
- `performance_optimization.py`: Performance tuning examples