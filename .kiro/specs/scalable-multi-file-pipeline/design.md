# Design Document

## Overview

The Scalable Multi-File Pipeline extends the existing flex property classification system to handle batch processing of multiple Excel files. The design leverages the current `FlexPropertyScorer` and property analysis components while adding orchestration, aggregation, and reporting capabilities for large-scale data processing.

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                 Scalable Multi-File Pipeline                │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ File Discovery  │  │ Batch Processor │  │ Aggregator   │ │
│  │ & Validation    │  │                 │  │ & Deduper   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│           │                     │                    │       │
│           ▼                     ▼                    ▼       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Progress        │  │ Error Handler   │  │ Report       │ │
│  │ Tracker         │  │ & Logger        │  │ Generator    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Existing Flex Classification System            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ FlexProperty    │  │ Property Data   │  │ Flex         │ │
│  │ Classifier      │  │ Analyzer        │  │ Scorer       │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Core Classes

#### 1. ScalableFlexPipeline
- **Purpose**: Main orchestrator for multi-file processing
- **Responsibilities**: 
  - File discovery and validation
  - Batch processing coordination
  - Progress tracking and logging
  - Result aggregation and export

#### 2. FileProcessor
- **Purpose**: Individual file processing handler
- **Responsibilities**:
  - Load Excel files using pandas
  - Apply flex classification using existing `FlexPropertyScorer`
  - Handle file-specific errors gracefully
  - Add source file metadata

#### 3. ResultAggregator
- **Purpose**: Combine and deduplicate results across files
- **Responsibilities**:
  - Merge DataFrames from multiple files
  - Remove duplicates based on address matching
  - Sort by flex score
  - Generate master dataset

#### 4. ReportGenerator
- **Purpose**: Create comprehensive summary reports
- **Responsibilities**:
  - Calculate aggregate statistics
  - Generate score distribution analysis
  - Create top candidates lists
  - Export formatted reports

## Components and Interfaces

### File Discovery Interface
```python
class FileDiscovery:
    def scan_input_folder(self, folder_path: str) -> List[Path]
    def validate_file_format(self, file_path: Path) -> bool
    def get_file_metadata(self, file_path: Path) -> Dict
```

### Batch Processing Interface
```python
class BatchProcessor:
    def process_file(self, file_path: Path) -> ProcessingResult
    def process_batch(self, file_paths: List[Path]) -> List[ProcessingResult]
    def handle_processing_error(self, file_path: Path, error: Exception) -> None
```

### Aggregation Interface
```python
class ResultAggregator:
    def combine_results(self, results: List[DataFrame]) -> DataFrame
    def deduplicate_properties(self, df: DataFrame) -> DataFrame
    def sort_by_score(self, df: DataFrame) -> DataFrame
```

## Data Models

### ProcessingResult
```python
@dataclass
class ProcessingResult:
    file_path: str
    success: bool
    flex_properties: Optional[pd.DataFrame]
    property_count: int
    flex_candidate_count: int
    processing_time: float
    error_message: Optional[str]
    source_file_info: Dict
```

### PipelineConfiguration
```python
@dataclass
class PipelineConfiguration:
    input_folder: str = "data/raw"
    output_file: str = "data/exports/all_flex_properties.xlsx"
    batch_size: int = 10
    max_workers: int = 4
    enable_deduplication: bool = True
    min_flex_score: float = 4.0
    progress_reporting: bool = True
```

### AggregateStatistics
```python
@dataclass
class AggregateStatistics:
    total_files_processed: int
    successful_files: int
    failed_files: int
    total_properties: int
    unique_flex_properties: int
    average_flex_score: float
    score_distribution: Dict[str, int]
    geographic_coverage: Dict[str, int]
    processing_duration: float
```

## Error Handling

### Error Categories
1. **File Access Errors**: Missing files, permission issues, corrupted files
2. **Data Format Errors**: Invalid Excel format, missing required columns
3. **Processing Errors**: Classification failures, memory issues
4. **Aggregation Errors**: Merge conflicts, data type mismatches

### Error Handling Strategy
- **Graceful Degradation**: Continue processing remaining files when individual files fail
- **Detailed Logging**: Log all errors with context for troubleshooting
- **Error Reporting**: Include error summary in final reports
- **Recovery Options**: Provide mechanisms to retry failed files

### Error Recovery Mechanisms
```python
class ErrorHandler:
    def handle_file_error(self, file_path: Path, error: Exception) -> ErrorAction
    def log_processing_error(self, context: Dict, error: Exception) -> None
    def generate_error_report(self, errors: List[ProcessingError]) -> Dict
```

## Testing Strategy

### Unit Testing
- **File Discovery**: Test folder scanning, file validation, metadata extraction
- **File Processing**: Test individual file processing with various data formats
- **Aggregation**: Test result combination, deduplication logic, sorting
- **Report Generation**: Test statistics calculation, export functionality

### Integration Testing
- **End-to-End Pipeline**: Test complete pipeline with sample datasets
- **Error Scenarios**: Test pipeline behavior with corrupted/missing files
- **Performance Testing**: Test with large datasets and multiple files
- **Memory Management**: Test memory usage with large file sets

### Test Data Strategy
```python
# Test file structures
test_files/
├── valid_properties_1.xlsx      # Standard format with flex candidates
├── valid_properties_2.xlsx      # Different regions, some duplicates
├── empty_file.xlsx              # Empty dataset
├── invalid_format.xlsx          # Missing required columns
├── corrupted_file.xlsx          # Corrupted Excel file
└── large_dataset.xlsx           # Performance testing (10k+ properties)
```

### Performance Benchmarks
- **Processing Speed**: Target 1000 properties per minute per file
- **Memory Usage**: Maximum 2GB RAM for processing 100k properties
- **Concurrent Processing**: Support 4-8 concurrent file processors
- **Scalability**: Handle up to 100 files in single batch

## Implementation Phases

### Phase 1: Core Pipeline Structure
- Implement `ScalableFlexPipeline` main class
- Create file discovery and validation logic
- Integrate with existing `FlexPropertyScorer`
- Basic error handling and logging

### Phase 2: Batch Processing
- Implement concurrent file processing
- Add progress tracking and reporting
- Implement result aggregation and deduplication
- Memory optimization for large datasets

### Phase 3: Advanced Features
- Comprehensive report generation
- Performance monitoring and optimization
- Advanced error recovery mechanisms
- Configuration management system

## Integration Points

### Existing System Integration
- **FlexPropertyScorer**: Use existing scoring algorithm without modification
- **Property Data Models**: Leverage existing data structures
- **Logging System**: Extend existing logging infrastructure
- **Export Utilities**: Build on existing Excel export capabilities

### External Dependencies
- **pandas**: DataFrame operations and Excel I/O
- **pathlib**: File system operations
- **concurrent.futures**: Parallel processing
- **logging**: Progress tracking and error reporting

## Configuration Management

### Pipeline Configuration File
```yaml
# pipeline_config.yaml
input:
  folder: "data/raw"
  file_pattern: "*.xlsx"
  recursive_scan: false

processing:
  batch_size: 10
  max_workers: 4
  memory_limit_gb: 4
  timeout_minutes: 30

output:
  master_file: "data/exports/all_flex_properties.xlsx"
  report_file: "data/reports/pipeline_summary.json"
  enable_csv_export: true

filtering:
  min_flex_score: 4.0
  enable_deduplication: true
  duplicate_fields: ["Address", "City", "State"]

logging:
  level: "INFO"
  file: "logs/pipeline.log"
  progress_interval: 100
```

## Performance Considerations

### Memory Management
- **Streaming Processing**: Process files individually to avoid loading all data into memory
- **Chunked Operations**: Process large files in chunks for memory efficiency
- **Garbage Collection**: Explicit cleanup of processed DataFrames

### Parallel Processing
- **File-Level Parallelism**: Process multiple files concurrently
- **Thread Safety**: Ensure thread-safe operations for shared resources
- **Resource Limits**: Respect system memory and CPU constraints

### Optimization Strategies
- **Lazy Loading**: Load file metadata before full processing
- **Early Filtering**: Apply basic filters before expensive operations
- **Caching**: Cache frequently accessed configuration and lookup data