# Design Document

## Overview

The Interactive Filter Dashboard is a Streamlit-based web application that provides comprehensive property data filtering, analysis, and visualization capabilities. The application follows a modular architecture with clear separation between data processing, user interface, and visualization components. The design emphasizes performance through data caching, user experience through intuitive controls, and reliability through robust error handling.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web Interface                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Sidebar   │  │  Main Area  │  │   Tabs      │         │
│  │   Filters   │  │   Metrics   │  │   Content   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                    Application Logic Layer                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │    Data     │  │   Filter    │  │Visualization│         │
│  │  Processing │  │   Engine    │  │   Engine    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                      Data Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Excel     │  │   Cached    │  │  Filtered   │         │
│  │   Source    │  │    Data     │  │   Results   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### Component Architecture

The application consists of four main components:

1. **Data Management Component**: Handles data loading, cleaning, and caching
2. **Filter Management Component**: Manages filter state and application logic
3. **Visualization Component**: Generates charts and displays data tables
4. **Export Component**: Handles data export functionality

## Components and Interfaces

### Data Management Component

**Purpose**: Load, clean, and cache property data from Excel sources

**Key Functions**:
- `load_data(file_path)`: Main data loading function with caching
- `clean_numeric_column(series)`: Standardizes numeric data formatting

**Interface**:
```python
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and cache property data from Excel file
    Returns: Cleaned DataFrame with standardized numeric columns
    """

def clean_numeric_column(series: pd.Series) -> pd.Series:
    """
    Clean text-based numeric columns by removing formatting
    Returns: Numeric series with proper data types
    """
```

**Data Cleaning Logic**:
- Remove currency symbols ($), commas, and percentage signs
- Convert various null representations to proper null values
- Handle data type conversion with error handling
- Standardize numeric columns for consistent filtering

### Filter Management Component

**Purpose**: Manage filter state and apply filtering logic to datasets

**Filter Types**:
1. **Property Type Filter**: Multi-select keyword matching
2. **Building Size Filter**: Range slider (square footage)
3. **Lot Size Filter**: Range slider (acres)
4. **Price Filter**: Optional range slider with toggle
5. **Year Built Filter**: Optional range slider with toggle
6. **Advanced Filters**: Occupancy, county, and state selection

**Filter Application Logic**:
```python
def apply_filters(df: pd.DataFrame, filter_params: dict) -> pd.DataFrame:
    """
    Apply all active filters to the dataset
    Returns: Filtered DataFrame based on current filter state
    """
```

**State Management**:
- Use Streamlit session state to persist filtered results
- Trigger re-filtering only when "Apply Filters" button is clicked
- Maintain filter parameter validation and bounds checking

### Visualization Component

**Purpose**: Generate interactive charts and data displays using Plotly

**Chart Types**:
1. **Building Size Distribution**: Histogram with 30 bins
2. **Property Type Distribution**: Pie chart (top 10 types)
3. **Year Built Distribution**: Histogram with 20 bins
4. **Sale Price Distribution**: Box plot for price analysis
5. **Geographic Distribution**: Horizontal and vertical bar charts

**Visualization Interface**:
```python
def create_size_distribution_chart(df: pd.DataFrame) -> plotly.graph_objects.Figure:
    """Generate building size histogram"""

def create_property_type_chart(df: pd.DataFrame) -> plotly.graph_objects.Figure:
    """Generate property type pie chart"""

def create_geographic_charts(df: pd.DataFrame) -> tuple:
    """Generate county and city distribution charts"""
```

**Chart Configuration**:
- Consistent height settings (300-500px) for uniform layout
- Responsive design with `use_container_width=True`
- Proper axis labeling without special characters
- Graceful handling of missing or insufficient data

### Export Component

**Purpose**: Provide data export functionality in multiple formats

**Export Formats**:
1. **CSV Export**: Standard comma-separated values
2. **Excel Export**: XLSX format with proper formatting

**Export Interface**:
```python
def generate_csv_export(df: pd.DataFrame) -> str:
    """Generate CSV string for download"""

def generate_excel_export(df: pd.DataFrame) -> bytes:
    """Generate Excel file buffer for download"""

def create_filter_summary(filter_params: dict) -> dict:
    """Generate summary of applied filters"""
```

**File Naming Convention**:
- Include timestamp: `filtered_properties_YYYYMMDD_HHMMSS.{ext}`
- Prevent filename conflicts through unique timestamps

## Data Models

### Property Data Schema

The application expects property data with the following key columns:

```python
PropertyDataSchema = {
    'Property Name': str,           # Property identifier
    'Property Type': str,           # Type classification
    'Address': str,                 # Street address
    'City': str,                    # City name
    'County': str,                  # County name
    'State': str,                   # State abbreviation
    'Building SqFt': float,         # Building square footage
    'Lot Size Acres': float,        # Lot size in acres
    'Lot Size SqFt': float,         # Lot size in square feet
    'Year Built': int,              # Construction year
    'Sold Price': float,            # Sale price
    'Loan Amount': float,           # Loan amount
    'Interest Rate': float,         # Interest rate percentage
    'Number of Units': int,         # Unit count
    'Occupancy': float              # Occupancy percentage
}
```

### Filter State Model

```python
FilterState = {
    'industrial_keywords': List[str],     # Selected industrial keywords
    'size_range': Tuple[int, int],        # Building size range
    'lot_range': Tuple[float, float],     # Lot size range
    'price_range': Tuple[int, int],       # Price range
    'year_range': Tuple[int, int],        # Year built range
    'occupancy_range': Tuple[int, int],   # Occupancy percentage range
    'selected_counties': List[str],       # Selected counties
    'selected_states': List[str],         # Selected states
    'use_price_filter': bool,             # Price filter toggle
    'use_year_filter': bool,              # Year filter toggle
    'use_occupancy_filter': bool          # Occupancy filter toggle
}
```

## Error Handling

### Data Loading Errors

1. **File Not Found**: Display user-friendly error message with file path
2. **Invalid Excel Format**: Provide guidance on expected file format
3. **Missing Columns**: Gracefully handle missing optional columns
4. **Data Type Conversion**: Use `errors='coerce'` for robust numeric conversion

### Runtime Error Handling

1. **Empty Datasets**: Display appropriate messages when no data matches filters
2. **Visualization Errors**: Skip charts when insufficient data is available
3. **Export Errors**: Provide fallback options and error messaging
4. **Memory Issues**: Implement data pagination and chunking for large datasets

### User Input Validation

1. **Filter Range Validation**: Ensure min <= max for all range inputs
2. **File Path Validation**: Verify file existence before processing
3. **Data Bounds Checking**: Set reasonable limits on slider ranges

## Testing Strategy

### Unit Testing

1. **Data Cleaning Functions**: Test numeric column cleaning with various input formats
2. **Filter Logic**: Verify filter application produces expected results
3. **Export Functions**: Validate CSV and Excel generation
4. **Visualization Functions**: Test chart generation with edge cases

### Integration Testing

1. **End-to-End Workflow**: Test complete filter-to-export workflow
2. **Data Loading Pipeline**: Verify Excel loading and caching behavior
3. **UI Component Integration**: Test filter interactions and state management

### Performance Testing

1. **Large Dataset Handling**: Test with datasets of 10,000+ properties
2. **Filter Performance**: Measure response times for complex filter combinations
3. **Memory Usage**: Monitor memory consumption during data processing
4. **Caching Effectiveness**: Verify cache hit rates and performance improvements

### User Acceptance Testing

1. **Filter Usability**: Verify intuitive filter controls and feedback
2. **Visualization Clarity**: Ensure charts are readable and informative
3. **Export Functionality**: Validate exported data matches filtered results
4. **Error Recovery**: Test graceful handling of various error conditions

## Performance Considerations

### Data Caching Strategy

- Use `@st.cache_data` decorator for expensive data loading operations
- Cache cleaned and processed data to avoid repeated transformations
- Implement cache invalidation when source data changes

### Memory Optimization

- Limit initial data table display to 100 rows
- Use efficient pandas operations for filtering
- Implement lazy loading for large datasets

### UI Responsiveness

- Use Streamlit's built-in progress indicators
- Implement asynchronous data processing where possible
- Optimize chart rendering with appropriate bin sizes and data limits

## Security Considerations

### Data Privacy

- Ensure no sensitive data is logged or cached inappropriately
- Implement proper session management for multi-user scenarios
- Validate file paths to prevent directory traversal attacks

### Input Validation

- Sanitize all user inputs before processing
- Validate file uploads and data formats
- Implement proper error handling to prevent information disclosure