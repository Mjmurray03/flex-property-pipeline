# Interactive Filter Dashboard

A comprehensive Streamlit-based web application for filtering, analyzing, and visualizing property data. This dashboard provides real estate professionals and analysts with powerful tools to explore property datasets interactively.

## Features

### Data Loading & Processing
- Automatic Excel file loading with error handling
- Data cleaning for currency, percentage, and text-formatted numeric columns
- Caching for improved performance
- Graceful handling of missing or malformed data

### Interactive Filtering
- **Property Type Filter**: Multi-select keyword matching for industrial properties
- **Building Size Filter**: Range slider for square footage filtering
- **Lot Size Filter**: Range slider for acreage-based filtering
- **Price Filter**: Optional price range filtering with toggle
- **Year Built Filter**: Optional construction year filtering with toggle
- **Advanced Filters**: Occupancy rate, county, and state selection

### Analytics & Visualizations
- Building size distribution histogram
- Property type distribution pie chart
- Year built distribution histogram
- Sale price distribution box plot
- Geographic distribution charts (county and city)

### Data Export
- CSV export with timestamp-based filenames
- Excel export with proper formatting
- Filter summary documentation
- Download buttons with proper MIME types

### User Interface
- Clean, emoji-free interface for universal compatibility
- Tabbed organization (Data Table, Analytics, Geographic, Export)
- Responsive design with column selection
- Real-time metrics and feedback
- Performance-optimized data display (100-row pagination)

## Installation

### Prerequisites
- Python 3.7+
- Required packages:
  ```
  streamlit
  pandas
  plotly
  openpyxl
  numpy
  ```

### Setup
1. Install required packages:
   ```bash
   pip install streamlit pandas plotly openpyxl numpy
   ```

2. Place your Excel data file at:
   ```
   C:\flex-property-pipeline\data\raw\Full Property Export.xlsx
   ```
   (Or modify the file path in the code)

3. Run the dashboard:
   ```bash
   streamlit run flex_filter_dashboard.py
   ```

## Usage

### Basic Workflow
1. **Load Data**: The dashboard automatically loads your property data on startup
2. **Apply Filters**: Use the sidebar controls to set your filtering criteria
3. **View Results**: Examine filtered properties in the main area
4. **Analyze Data**: Switch to the Analytics tab for visualizations
5. **Export Results**: Use the Export tab to download filtered data

### Filter Controls

#### Property Type
- Select industrial keywords to filter by property type
- Default includes: industrial, warehouse, distribution, flex

#### Building Size
- Use the slider to set minimum and maximum square footage
- Default range: 20,000 - 100,000 sqft

#### Lot Size
- Set acreage range using the slider
- Default range: 0.5 - 20.0 acres

#### Price Range (Optional)
- Enable with checkbox and set price range
- Default range: $150,000 - $2,000,000

#### Year Built (Optional)
- Enable with checkbox and set construction year range
- Default range: 1980 - current year

#### Advanced Filters
- **Occupancy Rate**: Filter by occupancy percentage
- **Counties**: Multi-select county filtering
- **States**: Multi-select state filtering

### Data Table Features
- Column selection for customized views
- Responsive table display
- Performance-optimized (shows first 100 rows)
- Real-time filtering results

### Analytics Features
- **Building Size Distribution**: Histogram showing size patterns
- **Property Type Distribution**: Pie chart of top 10 property types
- **Year Built Distribution**: Histogram of construction years
- **Price Distribution**: Box plot showing price ranges
- **Geographic Distribution**: Bar charts for counties and cities

### Export Options
- **CSV Export**: Standard comma-separated format
- **Excel Export**: XLSX format with proper formatting
- **Filter Summary**: Documentation of applied filters
- **Timestamped Filenames**: Prevents file overwrites

## Data Requirements

### Expected Columns
The dashboard expects the following columns in your Excel file:

**Required:**
- Property Name
- Property Type
- City
- County
- State

**Optional (with graceful fallback):**
- Address
- Building SqFt
- Lot Size Acres
- Lot Size SqFt
- Year Built
- Sold Price
- Loan Amount
- Interest Rate
- Number of Units
- Occupancy

### Data Formats
The dashboard automatically cleans:
- Currency symbols ($)
- Thousands separators (,)
- Percentage symbols (%)
- Various null representations (N/A, n/a, NA, na, None, none, empty strings)

## Performance Considerations

### Optimization Features
- Streamlit caching for data loading and processing
- Efficient pandas operations for filtering
- Lazy loading for large datasets
- Optimized chart rendering

### Recommended Limits
- Dataset size: Up to 100,000 properties
- Display limit: 100 rows in data table
- Chart data: Automatically binned for performance

## Error Handling

### Data Loading Errors
- File not found: Clear error message with path
- Invalid format: Guidance on expected file format
- Permission errors: Instructions for file access

### Runtime Error Handling
- Missing columns: Graceful feature disabling
- Invalid data: Automatic cleaning and conversion
- Empty results: Appropriate user messaging
- Filter validation: Range checking and error prevention

## Testing

### Unit Tests
Run the test suite:
```bash
python test_dashboard.py
```

### Validation
Run comprehensive validation:
```bash
python validate_dashboard.py
```

### Test Coverage
- Data cleaning functions
- Filter application logic
- Export functionality
- Error handling scenarios
- Performance with large datasets

## Troubleshooting

### Common Issues

**Dashboard won't start:**
- Check Python version (3.7+ required)
- Verify all packages are installed
- Ensure file path is correct

**No data loaded:**
- Verify Excel file exists at specified path
- Check file permissions
- Ensure file is not open in another application

**Filters not working:**
- Check for required columns in your data
- Verify data types are correct
- Look for error messages in the interface

**Performance issues:**
- Reduce dataset size if over 100,000 rows
- Clear Streamlit cache: `streamlit cache clear`
- Check available system memory

### Support
For issues or questions:
1. Check the error messages in the dashboard
2. Review the console output for detailed errors
3. Validate your data format matches expectations
4. Run the validation script to identify issues

## Architecture

### Components
- **Data Management**: Loading, cleaning, and caching
- **Filter Engine**: Multi-criteria filtering logic
- **Visualization Engine**: Plotly-based charts and graphs
- **Export Engine**: CSV and Excel generation
- **UI Layer**: Streamlit interface components

### Design Principles
- Performance through caching and optimization
- Reliability through comprehensive error handling
- Usability through intuitive interface design
- Maintainability through modular architecture

## Version History

### v1.0.0
- Initial release with full filtering functionality
- Analytics visualizations
- Export capabilities
- Comprehensive error handling
- Performance optimizations
- Complete test suite