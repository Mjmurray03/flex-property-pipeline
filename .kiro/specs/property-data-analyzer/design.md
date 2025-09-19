# Design Document

## Overview

The Property Data Analyzer is a comprehensive data analysis component that integrates with the existing Flex Property Pipeline to load, analyze, and categorize private property data from Excel files. The system leverages the existing logging infrastructure, scoring algorithms, and data processing patterns while providing a focused interface for Excel-based property data analysis.

## Architecture

### Component Integration

The Property Data Analyzer integrates with the existing pipeline architecture:

```
Excel Data Source
       ↓
PrivatePropertyAnalyzer
       ↓
┌─────────────────────────────────────┐
│  Data Loading & Validation Layer    │
│  - Excel file reader               │
│  - Data structure analysis         │
│  - Quality assessment              │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│  Property Classification Layer      │
│  - Industrial property detection    │
│  - Keyword-based filtering         │
│  - Type categorization             │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│  Analysis & Reporting Layer         │
│  - Data completeness metrics       │
│  - Sample property display         │
│  - Summary statistics              │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│  Integration Layer (Optional)       │
│  - FlexPropertyScorer integration  │
│  - MongoDB storage                 │
│  - Pipeline data format            │
└─────────────────────────────────────┘
```

### Design Patterns

1. **Single Responsibility**: Each method handles one specific aspect of analysis
2. **Dependency Injection**: Logger and file path are configurable
3. **Error Resilience**: Graceful handling of missing columns and data issues
4. **Extensibility**: Easy to add new analysis methods and metrics

## Components and Interfaces

### Core Class: PrivatePropertyAnalyzer

```python
class PrivatePropertyAnalyzer:
    def __init__(self, file_path: str, logger: Optional[logging.Logger] = None)
    def load_data(self) -> pd.DataFrame
    def analyze_property_types(self) -> List[str]
    def check_data_completeness(self) -> Dict[str, float]
    def get_industrial_sample(self, limit: int = 10) -> pd.DataFrame
    def generate_summary_report(self) -> Dict
    def export_results(self, output_path: str, format: str = 'excel') -> None
```

### Data Structures

#### Input Data Schema (Expected Excel Columns)
- **Property Type**: Primary classification field
- **Building SqFt**: Building size information
- **Sale Date**: Transaction timing
- **Sold Price**: Transaction value
- **Year Built**: Property age
- **Lot Size Acres**: Land area
- **Zoning Code**: Land use classification
- **County**: Geographic location
- **Property Name**: Identifier/description
- **City**: Municipal location
- **State**: State location

#### Analysis Results Schema
```python
{
    "dataset_overview": {
        "total_properties": int,
        "total_columns": int,
        "file_path": str,
        "load_timestamp": datetime
    },
    "property_types": {
        "all_types": Dict[str, int],
        "industrial_types": List[str],
        "industrial_count": int
    },
    "data_quality": {
        "field_completeness": Dict[str, float],
        "overall_completeness": float
    },
    "industrial_sample": List[Dict]
}
```

### Integration Interfaces

#### Logger Integration
- Uses existing `utils.logger.setup_logging()` infrastructure
- Consistent logging format with pipeline components
- Progress tracking for large datasets

#### FlexPropertyScorer Integration (Optional)
- Convert Excel data to pipeline format
- Apply existing scoring algorithms
- Generate flex scores for industrial properties

#### Database Integration (Optional)
- Store analysis results in MongoDB
- Use existing `database.mongodb_client` patterns
- Enable pipeline integration workflows

## Data Models

### Property Data Model
```python
@dataclass
class PropertyRecord:
    property_name: Optional[str]
    property_type: str
    building_sqft: Optional[float]
    city: Optional[str]
    state: Optional[str]
    county: Optional[str]
    sale_date: Optional[datetime]
    sold_price: Optional[float]
    year_built: Optional[int]
    lot_size_acres: Optional[float]
    zoning_code: Optional[str]
    raw_data: Dict  # Original Excel row data
```

### Analysis Results Model
```python
@dataclass
class AnalysisResults:
    total_properties: int
    industrial_properties: List[PropertyRecord]
    property_type_distribution: Dict[str, int]
    data_completeness: Dict[str, float]
    industrial_keywords_found: List[str]
    summary_statistics: Dict[str, Any]
```

## Error Handling

### File Loading Errors
- **FileNotFoundError**: Clear message with file path validation
- **PermissionError**: Guidance on file access issues
- **pandas.errors.ParserError**: Excel format validation and suggestions
- **Memory errors**: Chunked reading for large files

### Data Processing Errors
- **Missing columns**: Continue with available columns, log warnings
- **Data type mismatches**: Attempt conversion, fallback to string
- **Empty datasets**: Graceful handling with informative messages
- **Encoding issues**: Multiple encoding attempts (utf-8, latin-1, cp1252)

### Analysis Errors
- **No industrial properties found**: Clear reporting, suggest keyword expansion
- **Incomplete data**: Partial analysis with quality warnings
- **Invalid data ranges**: Outlier detection and reporting

### Error Recovery Strategies
1. **Partial Success**: Complete analysis with available data
2. **Fallback Methods**: Alternative approaches for missing data
3. **User Guidance**: Specific recommendations for data issues
4. **Logging**: Detailed error context for troubleshooting

## Testing Strategy

### Unit Tests
- **Data Loading**: Various Excel formats, edge cases, error conditions
- **Property Classification**: Keyword matching accuracy, case sensitivity
- **Data Quality**: Completeness calculations, missing data handling
- **Industrial Detection**: Known datasets with expected results

### Integration Tests
- **Logger Integration**: Verify logging output and formatting
- **FlexScorer Integration**: End-to-end scoring workflow
- **Database Integration**: MongoDB storage and retrieval
- **Pipeline Integration**: Data format compatibility

### Test Data Sets
1. **Perfect Dataset**: All columns present, clean data
2. **Missing Columns**: Subset of expected columns
3. **Dirty Data**: Mixed data types, nulls, inconsistent formats
4. **Large Dataset**: Performance testing with 100k+ records
5. **No Industrial**: Dataset with no industrial properties
6. **Edge Cases**: Empty files, single row, malformed Excel

### Performance Tests
- **Memory Usage**: Large file handling without memory issues
- **Processing Speed**: Acceptable performance for typical datasets
- **Scalability**: Performance degradation patterns

### Validation Tests
- **Keyword Accuracy**: Industrial property detection precision/recall
- **Data Quality Metrics**: Completeness calculation accuracy
- **Export Functionality**: Output format validation

## Implementation Phases

### Phase 1: Core Functionality
1. Basic Excel loading with pandas
2. Property type analysis and industrial detection
3. Data completeness assessment
4. Simple console output and logging

### Phase 2: Enhanced Analysis
1. Advanced industrial property classification
2. Statistical analysis and summary metrics
3. Sample property display with formatting
4. Error handling and edge case management

### Phase 3: Integration Features
1. FlexPropertyScorer integration
2. MongoDB storage capabilities
3. Pipeline data format compatibility
4. Export functionality (Excel, JSON, CSV)

### Phase 4: Advanced Features
1. Configurable industrial keywords
2. Custom analysis metrics
3. Batch processing capabilities
4. API integration endpoints