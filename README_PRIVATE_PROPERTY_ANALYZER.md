# Private Property Data Analyzer

A comprehensive tool for analyzing private property data from Excel files to identify and categorize flex industrial properties. This analyzer integrates seamlessly with the existing Flex Property Pipeline infrastructure.

## Features

- **Excel Data Loading**: Load and analyze property data from Excel files with comprehensive error handling
- **Property Type Analysis**: Automatically identify industrial properties using configurable keyword matching
- **Data Quality Assessment**: Analyze completeness and quality of key property fields
- **Flex Scoring Integration**: Calculate flex industrial scores using the existing FlexPropertyScorer
- **Multiple Export Formats**: Export results to JSON, Excel, and CSV formats
- **Database Integration**: Store and retrieve analysis results using MongoDB infrastructure
- **Historical Comparison**: Compare current analysis with historical data trends
- **Comprehensive Logging**: Detailed logging with existing pipeline logger integration
- **Command-Line Interface**: Easy-to-use CLI for batch processing and automation

## Installation

The analyzer is part of the Flex Property Pipeline project. Ensure you have the required dependencies:

```bash
pip install pandas openpyxl pymongo
```

## Quick Start

### Command Line Usage

```bash
# Basic analysis
python analyze_private_properties.py data/properties.xlsx

# Full analysis with scoring and exports
python analyze_private_properties.py data/properties.xlsx --score --export-all --store-db

# Custom output directory
python analyze_private_properties.py data/properties.xlsx --output-dir results/analysis_2024
```

### Python API Usage

```python
from processors.private_property_analyzer import PrivatePropertyAnalyzer
from utils.logger import setup_logging

# Initialize
logger = setup_logging('property_analysis')
analyzer = PrivatePropertyAnalyzer('data/properties.xlsx', logger=logger)

# Load and analyze
data = analyzer.load_data()
industrial_types = analyzer.analyze_property_types()
completeness = analyzer.check_data_completeness()
sample = analyzer.get_industrial_sample(limit=10)

# Generate comprehensive report
report = analyzer.generate_summary_report()

# Export results
exported_files = analyzer.export_results(
    output_dir='results',
    formats=['json', 'excel', 'csv']
)
```

## Excel File Requirements

The analyzer expects Excel files with property data. Required and optional columns:

### Required Columns (flexible naming)
- **Property Type** (or "Type"): Property classification
- **Property Name** (or "Name"): Property identifier

### Optional Columns (enhance analysis when present)
- **Building SqFt** (or "Building Sq Ft"): Building square footage
- **Lot Size Acres** (or "Acres"): Property size in acres
- **City**: Property location
- **State**: Property state
- **Owner Name**: Property owner
- **Market Value**: Property market value
- **Year Built**: Construction year
- **Zoning Code**: Zoning classification

### Example Excel Structure

| Property Name | Property Type | Building SqFt | Lot Size Acres | City | Market Value |
|---------------|---------------|---------------|----------------|------|--------------|
| Industrial Complex A | Industrial | 50000 | 5.2 | West Palm Beach | 2500000 |
| Warehouse B | Warehouse | 75000 | 8.1 | Boca Raton | 3200000 |
| Flex Space C | Flex Industrial | 40000 | 4.5 | Delray Beach | 2100000 |

## Industrial Property Detection

The analyzer automatically identifies industrial properties using these keywords:
- industrial
- warehouse  
- distribution
- flex
- manufacturing
- storage
- logistics

Properties matching these keywords are classified as industrial and included in specialized analysis.

## Analysis Components

### 1. Dataset Overview
- Total property count
- Column analysis
- Memory usage assessment
- Data type identification

### 2. Property Type Analysis
- Unique property type identification
- Industrial property detection
- Property type distribution
- Industrial percentage calculation

### 3. Data Quality Metrics
- Field completeness percentages
- Missing data identification
- Data quality scoring
- Field availability assessment

### 4. Industrial Property Sample
- Sample industrial property display
- Key field extraction
- Property detail formatting
- Industrial property filtering

### 5. Flex Scoring (Optional)
- Integration with FlexPropertyScorer
- Automated flex score calculation
- Score-based property ranking
- Detailed scoring indicators

## Export Formats

### JSON Export
Comprehensive analysis results in structured JSON format:
```json
{
  "metadata": {
    "analysis_timestamp": "2024-01-15T10:30:00",
    "file_path": "data/properties.xlsx"
  },
  "dataset_overview": {
    "total_properties": 1000,
    "total_columns": 12
  },
  "property_type_analysis": {
    "industrial_types_found": ["Industrial", "Warehouse"],
    "total_industrial_properties": 250
  }
}
```

### Excel Export
Multi-sheet Excel workbook with:
- **Summary**: Key metrics and overview
- **Property Types**: Property type distribution
- **Data Completeness**: Field quality analysis
- **Industrial Sample**: Sample industrial properties

### CSV Export
Industrial properties in CSV format for further analysis and integration.

## Database Integration

### Storing Results
```python
# Store analysis results
success = analyzer.store_results_in_database()

# Store scored properties
scored_properties = analyzer.add_flex_scoring()
stored_count = analyzer.store_scored_properties(scored_properties)
```

### Historical Analysis
```python
# Retrieve historical analyses
historical = analyzer.retrieve_historical_analysis(limit=10)

# Compare with historical data
comparison = analyzer.compare_with_historical()
trends = comparison.get('trends', {})
```

## Command Line Options

### Basic Options
- `excel_file`: Path to Excel file (required)
- `--output-dir`: Output directory for results
- `--quiet`: Reduce console output
- `--verbose`: Increase console output

### Analysis Options
- `--score`: Add flex scores to industrial properties
- `--score-all`: Add flex scores to all properties
- `--sample-limit`: Number of sample properties to display

### Export Options
- `--export`: Specify export formats (json, excel, csv)
- `--export-all`: Export in all formats

### Database Options
- `--store-db`: Store results in MongoDB
- `--compare-historical`: Compare with historical data

### Logging Options
- `--log-file`: Custom log file path
- `--no-file-logging`: Disable file logging

## Integration with Pipeline

### FlexPropertyScorer Integration
```python
# Convert Excel data to pipeline format
pipeline_property = analyzer.convert_to_pipeline_format(excel_row)

# Add flex scoring
scored_properties = analyzer.add_flex_scoring(include_all_properties=False)
```

### MongoDB Integration
```python
# Uses existing database infrastructure
from database.mongodb_client import get_db_manager

# Automatic integration with existing collections
analyzer.store_results_in_database()
```

### Logger Integration
```python
# Uses existing logger infrastructure
from utils.logger import setup_logging

logger = setup_logging('private_analysis')
analyzer = PrivatePropertyAnalyzer(file_path, logger=logger)
```

## Error Handling

The analyzer provides comprehensive error handling:

### File Validation
- File existence checking
- Excel format validation
- Permission error handling
- Corruption detection

### Data Processing
- Missing column handling
- Data type conversion errors
- Null value management
- Partial analysis completion

### Integration Errors
- Database connection issues
- Scorer availability
- Export permission problems
- Memory management

## Performance Considerations

### Large Datasets
- Efficient pandas operations
- Memory usage monitoring
- Batch processing support
- Progress logging

### Optimization Tips
- Use `--quiet` mode for batch processing
- Limit sample sizes for large datasets
- Export only needed formats
- Monitor memory usage with large files

## Examples

See `examples/private_property_analysis_example.py` for comprehensive usage examples including:

1. Basic analysis workflow
2. Flex scoring integration
3. Export functionality
4. Database integration
5. Error handling patterns
6. Pipeline integration

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/test_private_property_analyzer.py -v

# Run specific test categories
python -m pytest tests/test_private_property_analyzer.py::TestPrivatePropertyAnalyzer::test_load_data_perfect_dataset -v
```

## Troubleshooting

### Common Issues

**File Not Found Error**
```
FileNotFoundError: File not found: data/properties.xlsx
```
- Verify file path is correct
- Ensure file exists and is accessible
- Check file permissions

**Excel Format Error**
```
ValueError: File must be an Excel file (.xlsx or .xls)
```
- Ensure file has .xlsx or .xls extension
- Verify file is valid Excel format
- Try opening file in Excel to confirm

**No Industrial Properties Found**
```
No industrial properties detected based on keywords
```
- Check Property Type column values
- Verify industrial keywords match your data
- Consider customizing industrial_keywords list

**Memory Issues with Large Files**
```
MemoryError: Unable to allocate array
```
- Process file in smaller chunks
- Increase available system memory
- Use `--quiet` mode to reduce memory usage

### Performance Optimization

For large datasets (>10,000 properties):
- Use `--sample-limit` to reduce sample size
- Export only essential formats
- Enable `--quiet` mode
- Monitor system memory usage

## Contributing

When extending the analyzer:

1. Follow existing code patterns
2. Add comprehensive error handling
3. Include logging for all operations
4. Write tests for new functionality
5. Update documentation

## License

Part of the Flex Property Pipeline project. See main project license for details.