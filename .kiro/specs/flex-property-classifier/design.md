# Design Document

## Overview

The Flex Property Classifier is a specialized component that analyzes private property datasets from Excel files to identify flex industrial property candidates using specific criteria and scoring algorithms. The classifier integrates with the existing Flex Property Pipeline architecture while providing a focused interface for Excel-based property analysis and candidate identification.

## Architecture

### Component Integration

The Flex Property Classifier integrates with the existing pipeline infrastructure:

```
Excel Data Source
       ↓
FlexPropertyClassifier
       ↓
┌─────────────────────────────────────┐
│  Data Loading & Validation Layer    │
│  - Excel file reader               │
│  - Data type validation            │
│  - Column mapping                  │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│  Property Filtering Layer           │
│  - Industrial keyword matching     │
│  - Building size filtering         │
│  - Lot size filtering              │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│  Flex Scoring Engine                │
│  - Multi-factor scoring algorithm  │
│  - Building size scoring           │
│  - Property type scoring           │
│  - Lot size scoring                │
│  - Age/condition scoring           │
│  - Occupancy bonus calculation     │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│  Results Processing Layer           │
│  - Score ranking and sorting       │
│  - Top candidate selection         │
│  - Statistics calculation          │
│  - Export formatting               │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│  Output Layer                       │
│  - Excel export                    │
│  - Analysis statistics             │
│  - Progress reporting              │
└─────────────────────────────────────┘
```

### Design Patterns

1. **Single Responsibility**: Each method handles one specific aspect of classification
2. **Fluent Interface**: Method chaining for analysis workflow
3. **Error Resilience**: Graceful handling of missing data and columns
4. **Extensibility**: Easy to modify scoring criteria and add new factors
5. **Integration Ready**: Compatible with existing pipeline components

## Components and Interfaces

### Core Class: FlexPropertyClassifier

```python
class FlexPropertyClassifier:
    def __init__(self, df: pd.DataFrame, logger: Optional[logging.Logger] = None)
    def classify_flex_properties(self) -> pd.DataFrame
    def calculate_flex_score(self, row: pd.Series) -> float
    def get_top_candidates(self, n: int = 100) -> pd.DataFrame
    def export_results(self, output_path: Optional[str] = None) -> pd.DataFrame
    def get_analysis_statistics(self) -> Dict[str, Any]
```

### Supporting Functions

```python
def load_property_data(file_path: str) -> pd.DataFrame
def validate_required_columns(df: pd.DataFrame) -> List[str]
def normalize_property_types(df: pd.DataFrame) -> pd.DataFrame
def calculate_statistics(candidates: pd.DataFrame) -> Dict[str, Any]
```

## Data Models

### Input Data Schema (Expected Excel Columns)

**Required Columns:**
- **Property Type**: Primary classification field for industrial filtering
- **Building SqFt**: Building size for size-based filtering and scoring
- **Lot Size Acres**: Land area for lot size filtering and scoring

**Optional Columns (for enhanced scoring):**
- **Year Built**: Property age for condition scoring
- **Occupancy**: Occupancy percentage for bonus scoring
- **Property Name**: Property identifier
- **Address**: Property location
- **City**: Municipal location
- **State**: State location
- **County**: County location
- **Zoning Code**: Land use classification
- **Sale Date**: Transaction timing
- **Sold Price**: Transaction value
- **Sold Price/SqFt**: Price per square foot
- **Owner Name**: Property owner information

### Flex Scoring Criteria

```python
SCORING_CRITERIA = {
    "building_size": {
        "20k_to_50k": 3.0,      # Ideal flex size
        "50k_to_100k": 2.0,     # Good flex size
        "100k_to_200k": 1.0,    # Acceptable flex size
        "over_200k": 0.0        # Too large for typical flex
    },
    "property_type": {
        "flex": 3.0,            # Perfect match
        "warehouse": 2.5,       # Very good
        "distribution": 2.5,    # Very good
        "light_industrial": 2.0, # Good
        "industrial": 1.5,      # Acceptable
        "manufacturing": 1.0,   # Possible
        "storage": 1.0,         # Possible
        "logistics": 1.0        # Possible
    },
    "lot_size": {
        "1_to_5_acres": 2.0,    # Ideal range
        "5_to_10_acres": 1.5,   # Good range
        "0.5_to_1_acres": 1.0,  # Acceptable (small)
        "10_to_20_acres": 1.0   # Acceptable (large)
    },
    "age_condition": {
        "built_after_1990": 1.0,  # Modern construction
        "built_after_1980": 0.5   # Decent condition
    },
    "occupancy_bonus": {
        "under_100_percent": 1.0  # Flex opportunity
    }
}
```

### Output Data Schema

```python
@dataclass
class FlexCandidate:
    property_name: Optional[str]
    property_type: str
    address: Optional[str]
    city: Optional[str]
    state: Optional[str]
    county: Optional[str]
    building_sqft: float
    lot_size_acres: float
    year_built: Optional[int]
    zoning_code: Optional[str]
    sale_date: Optional[str]
    sold_price: Optional[float]
    sold_price_per_sqft: Optional[float]
    owner_name: Optional[str]
    occupancy: Optional[float]
    flex_score: float
    score_breakdown: Dict[str, float]
```

### Analysis Statistics Schema

```python
@dataclass
class AnalysisStatistics:
    total_properties: int
    industrial_properties: int
    flex_candidates: int
    average_flex_score: float
    score_range: Tuple[float, float]
    top_property_types: Dict[str, int]
    size_distribution: Dict[str, int]
    geographic_distribution: Dict[str, int]
```

## Error Handling

### Data Loading Errors
- **FileNotFoundError**: Clear message with file path validation
- **pandas.errors.ParserError**: Excel format validation and suggestions
- **PermissionError**: File access guidance
- **Memory errors**: Chunked reading recommendations for large files

### Data Quality Issues
- **Missing required columns**: Skip missing criteria, continue with available data
- **Invalid data types**: Attempt conversion, log warnings for failures
- **Null values in scoring fields**: Assign 0 points for that factor
- **Negative or invalid values**: Data validation with outlier detection

### Processing Errors
- **Empty datasets**: Graceful handling with informative messages
- **No industrial properties found**: Clear reporting with suggestions
- **Scoring calculation errors**: Fallback to partial scoring
- **Export failures**: Alternative export formats and error recovery

### Error Recovery Strategies
1. **Partial Processing**: Complete analysis with available data
2. **Graceful Degradation**: Reduce scoring complexity if data is limited
3. **User Guidance**: Specific recommendations for data quality improvements
4. **Detailed Logging**: Comprehensive error context for troubleshooting

## Testing Strategy

### Unit Tests

**Data Loading Tests:**
- Various Excel formats and structures
- Missing column scenarios
- Invalid file paths and formats
- Large file handling

**Filtering Tests:**
- Industrial keyword matching accuracy
- Building size threshold validation
- Lot size range filtering
- Edge cases and boundary conditions

**Scoring Tests:**
- Individual scoring factor calculations
- Score aggregation and capping
- Missing data handling in scoring
- Score consistency and reproducibility

**Export Tests:**
- Excel output format validation
- Column selection and ordering
- Large dataset export performance
- File path and permission handling

### Integration Tests

**Pipeline Integration:**
- Compatibility with existing FlexPropertyScorer
- Logger integration and output formatting
- Database integration for result storage
- Data format consistency across components

**End-to-End Tests:**
- Complete workflow from Excel input to export
- Real dataset processing validation
- Performance benchmarking
- Memory usage optimization

### Test Data Sets

1. **Perfect Dataset**: All columns present, clean industrial data
2. **Minimal Dataset**: Only required columns, mixed property types
3. **Dirty Dataset**: Missing values, inconsistent formats, outliers
4. **Large Dataset**: 10k+ properties for performance testing
5. **No Industrial Dataset**: No qualifying properties
6. **Edge Cases**: Single property, empty file, malformed data

### Performance Requirements

- **Processing Speed**: Handle 10,000 properties in under 30 seconds
- **Memory Usage**: Process 50,000 properties without exceeding 1GB RAM
- **Export Performance**: Generate Excel output for 5,000 candidates in under 10 seconds
- **Accuracy**: 95%+ precision in industrial property identification

## Implementation Phases

### Phase 1: Core Classification Engine
1. Basic Excel loading and validation
2. Industrial property filtering with keyword matching
3. Size-based filtering (building and lot size)
4. Basic flex scoring algorithm implementation
5. Console output and basic logging

### Phase 2: Enhanced Scoring and Analysis
1. Multi-factor scoring system with all criteria
2. Score breakdown and detailed analysis
3. Top candidate selection and ranking
4. Analysis statistics calculation
5. Comprehensive error handling

### Phase 3: Export and Integration Features
1. Excel export functionality with formatting
2. Integration with existing pipeline components
3. Database storage capabilities
4. Batch processing optimization
5. Performance monitoring and optimization

### Phase 4: Advanced Features and Optimization
1. Configurable scoring criteria and weights
2. Custom export formats and templates
3. API integration capabilities
4. Advanced analytics and reporting
5. Machine learning enhancement opportunities

## Integration Points

### Existing Pipeline Components

**FlexPropertyScorer Integration:**
- Leverage existing scoring algorithms for validation
- Compare classification results with pipeline scores
- Use existing scoring weights and criteria as baseline

**Database Integration:**
- Store classification results in MongoDB
- Use existing database patterns and collections
- Enable historical analysis and comparison

**Logger Integration:**
- Use existing logging infrastructure
- Consistent log formatting and levels
- Progress tracking for large datasets

**Utils Integration:**
- Leverage existing utility functions
- Consistent error handling patterns
- Shared configuration management