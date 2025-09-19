# Scalable Multi-File Pipeline Developer Guide

## Architecture Overview

The Scalable Multi-File Pipeline is designed with a modular architecture that separates concerns and enables extensibility. The system follows the principles of clean architecture with clear separation between data processing, business logic, and infrastructure concerns.

### Core Components

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

## Module Structure

### Core Pipeline Modules

#### `pipeline/scalable_flex_pipeline.py`
Main orchestrator class that coordinates all pipeline components.

**Key Classes:**
- `ScalableFlexPipeline`: Main pipeline orchestrator
- `PipelineConfiguration`: Configuration management

**Responsibilities:**
- Configuration validation
- Component initialization
- Pipeline execution coordination

#### `pipeline/file_discovery.py`
Handles file system scanning and validation.

**Key Classes:**
- `FileDiscovery`: File scanning and metadata extraction

**Features:**
- Recursive directory scanning
- File format validation
- Metadata extraction

#### `pipeline/batch_processor.py`
Manages concurrent processing of multiple files.

**Key Classes:**
- `BatchProcessor`: Concurrent file processing orchestrator
- `ProgressTracker`: Thread-safe progress tracking
- `BatchProcessingStats`: Processing statistics

**Features:**
- ThreadPoolExecutor-based concurrency
- Progress tracking and reporting
- Error handling and recovery
- Performance monitoring

#### `pipeline/file_processor.py`
Processes individual Excel files.

**Key Classes:**
- `FileProcessor`: Individual file processing
- `ProcessingResult`: Standardized result structure

**Features:**
- Excel file loading and validation
- Column mapping and normalization
- Integration with flex classification
- Performance optimization

#### `pipeline/result_aggregator.py`
Combines and deduplicates results from multiple files.

**Key Classes:**
- `ResultAggregator`: Result combination and deduplication

**Features:**
- DataFrame merging
- Duplicate detection and removal
- Score-based sorting
- Data validation

#### `pipeline/report_generator.py`
Generates comprehensive reports and statistics.

**Key Classes:**
- `ReportGenerator`: Report generation and statistics

**Features:**
- Summary statistics calculation
- Score distribution analysis
- Geographic coverage analysis
- Top candidates identification

#### `pipeline/output_manager.py`
Manages file output operations.

**Key Classes:**
- `OutputManager`: File export and management
- `ExportResult`: Export operation results

**Features:**
- Excel and CSV export
- Backup and versioning
- Directory management
- Export metadata tracking

### Advanced Features

#### `pipeline/error_recovery.py`
Comprehensive error handling and retry system.

**Key Classes:**
- `ErrorRecoveryManager`: Main error recovery system
- `RetryManager`: Retry logic and backoff strategies
- `ErrorClassifier`: Error categorization
- `ProcessingError`: Detailed error information

**Features:**
- Exponential backoff retry
- Error categorization and severity assessment
- Recovery statistics and reporting
- Resume failed files capability

#### `pipeline/performance_optimizer.py`
Performance optimization and memory management.

**Key Classes:**
- `PerformanceOptimizer`: Main optimization system
- `MemoryMonitor`: Memory usage monitoring
- `ChunkedDataProcessor`: Large dataset processing

**Features:**
- Memory usage monitoring and warnings
- DataFrame optimization
- Chunked processing for large files
- Performance metrics tracking

#### `pipeline/data_validator.py`
Data validation and quality checks.

**Key Classes:**
- `DataValidator`: Schema and data validation

**Features:**
- Schema validation across files
- Data type consistency checking
- Quality metrics calculation
- Validation reporting

## Extension Points

### Adding New Data Sources

To add support for new file formats:

1. **Create a new file processor:**

```python
from pipeline.file_processor import FileProcessor, ProcessingResult

class CSVFileProcessor(FileProcessor):
    def _load_file(self, file_path: Path) -> pd.DataFrame:
        return pd.read_csv(file_path)
    
    def _validate_format(self, df: pd.DataFrame) -> bool:
        # Custom validation logic
        return True
```

2. **Register the processor:**

```python
from pipeline.file_discovery import FileDiscovery

# Extend FileDiscovery to handle CSV files
class ExtendedFileDiscovery(FileDiscovery):
    def __init__(self):
        super().__init__()
        self.processors = {
            '.xlsx': FileProcessor,
            '.csv': CSVFileProcessor
        }
```

### Custom Scoring Algorithms

To implement custom flex property scoring:

1. **Create a custom scorer:**

```python
from processors.flex_scorer import FlexPropertyScorer

class CustomFlexScorer(FlexPropertyScorer):
    def calculate_flex_score(self, property_data: Dict[str, Any]) -> float:
        # Custom scoring logic
        score = 0.0
        
        # Add your scoring criteria
        if property_data.get('zoning') in ['C1', 'C2']:
            score += 3.0
        
        # Land to improvement ratio
        land_value = property_data.get('land_market_value', 0)
        improvement_value = property_data.get('improvement_value', 0)
        
        if improvement_value > 0:
            ratio = land_value / improvement_value
            score += min(ratio * 2, 5.0)
        
        return min(score, 10.0)
```

2. **Use the custom scorer:**

```python
from pipeline.file_processor import FileProcessor

class CustomFileProcessor(FileProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flex_scorer = CustomFlexScorer()
```

### Custom Report Generators

To create custom reports:

```python
from pipeline.report_generator import ReportGenerator

class CustomReportGenerator(ReportGenerator):
    def generate_custom_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate custom analysis report"""
        
        # Your custom analysis logic
        analysis = {
            'custom_metric_1': self._calculate_custom_metric_1(df),
            'custom_metric_2': self._calculate_custom_metric_2(df),
            'regional_breakdown': self._analyze_by_region(df)
        }
        
        return analysis
    
    def _calculate_custom_metric_1(self, df: pd.DataFrame) -> float:
        # Custom calculation
        return df['flex_score'].quantile(0.75)
```

## Testing Framework

### Test Structure

The testing framework includes several types of tests:

#### Unit Tests
- Individual component testing
- Mock-based isolation
- Fast execution

#### Integration Tests
- Component interaction testing
- Database integration
- File system operations

#### End-to-End Tests
- Complete pipeline execution
- Real data scenarios
- Performance benchmarks

### Test Data Generation

Use the `TestDataGenerator` for creating test scenarios:

```python
from tests.test_data_generator import TestDataGenerator

# Create test scenario
with TestDataGenerator() as generator:
    scenario = generator.create_test_scenario('basic_processing')
    
    # Run your tests with the generated data
    results = run_pipeline_test(scenario['files'])
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test category
python -m pytest tests/test_batch_processor.py -v

# Run with coverage
python -m pytest tests/ --cov=pipeline --cov-report=html

# Run performance benchmarks
python -m pytest tests/ -m benchmark -v
```

## Performance Considerations

### Memory Management

1. **Use chunked processing for large files:**

```python
from pipeline.performance_optimizer import ChunkedDataProcessor

processor = ChunkedDataProcessor(chunk_size=10000)
result = processor.process_dataframe_chunked(large_df, processing_func)
```

2. **Monitor memory usage:**

```python
from pipeline.performance_optimizer import MemoryMonitor

with MemoryMonitor().monitor_operation("my_operation") as stats:
    # Your memory-intensive operation
    process_large_dataset()
```

3. **Optimize DataFrames:**

```python
from pipeline.performance_optimizer import optimize_dataframe_memory

optimized_df = optimize_dataframe_memory(df)
```

### Concurrency

1. **Right-size worker threads:**
   - Use 1-2 workers per CPU core
   - Consider I/O vs CPU bound operations
   - Monitor system resources

2. **Batch size optimization:**
   - Larger batches for I/O bound operations
   - Smaller batches for memory-intensive operations
   - Balance throughput vs memory usage

### Database Integration

For database integration:

```python
from database.mongodb_client import MongoDBClient

class DatabaseIntegratedPipeline(ScalableFlexPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db_client = MongoDBClient()
    
    def save_results_to_database(self, results: pd.DataFrame):
        """Save results to database"""
        records = results.to_dict('records')
        self.db_client.insert_many('flex_properties', records)
```

## Configuration Management

### Environment-Specific Configurations

Create environment-specific configuration files:

```yaml
# config/development.yaml
input_folder: "test_data"
max_workers: 2
log_level: "DEBUG"

# config/production.yaml
input_folder: "/data/production"
max_workers: 8
log_level: "INFO"
```

### Configuration Validation

Implement custom configuration validation:

```python
from pipeline.scalable_flex_pipeline import PipelineConfiguration

class CustomPipelineConfiguration(PipelineConfiguration):
    def validate_custom_settings(self) -> bool:
        """Add custom validation logic"""
        
        # Validate custom business rules
        if self.min_flex_score < 0 or self.min_flex_score > 10:
            return False
        
        # Validate file paths exist
        if not Path(self.input_folder).exists():
            return False
        
        return True
```

## Logging and Monitoring

### Custom Logging

Implement custom logging strategies:

```python
import logging
from pipeline.pipeline_logger import PipelineLogger

class CustomPipelineLogger(PipelineLogger):
    def __init__(self):
        super().__init__()
        
        # Add custom handlers
        self.add_database_handler()
        self.add_metrics_handler()
    
    def add_database_handler(self):
        """Add database logging handler"""
        # Implementation for database logging
        pass
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics"""
        self.logger.info(f"Performance metrics: {metrics}")
```

### Monitoring Integration

Integrate with monitoring systems:

```python
from pipeline.performance_optimizer import PerformanceOptimizer

class MonitoredPerformanceOptimizer(PerformanceOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_client = MetricsClient()  # Your metrics system
    
    def track_performance(self, operation_name: str):
        context = super().track_performance(operation_name)
        
        # Send metrics to monitoring system
        @contextmanager
        def monitored_context():
            with context as metrics:
                yield metrics
                
                # Send metrics
                self.metrics_client.send_metrics({
                    'operation': operation_name,
                    'duration': metrics.duration_seconds,
                    'throughput': metrics.throughput_records_per_second
                })
        
        return monitored_context()
```

## Deployment Considerations

### Docker Deployment

Create a Dockerfile for containerized deployment:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/exports logs

# Set environment variables
ENV PYTHONPATH=/app

# Run the pipeline
CMD ["python", "scalable_pipeline_cli.py"]
```

### Kubernetes Deployment

Example Kubernetes deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flex-pipeline
spec:
  replicas: 1
  selector:
    matchLabels:
      app: flex-pipeline
  template:
    metadata:
      labels:
        app: flex-pipeline
    spec:
      containers:
      - name: pipeline
        image: flex-pipeline:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        env:
        - name: PIPELINE_CONFIG
          value: "/app/config/production.yaml"
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: pipeline-data-pvc
```

### Scaling Considerations

1. **Horizontal Scaling:**
   - Use message queues for distributed processing
   - Implement stateless processing components
   - Use shared storage for input/output files

2. **Vertical Scaling:**
   - Increase memory limits for large datasets
   - Add more CPU cores for concurrent processing
   - Use SSD storage for better I/O performance

## Security Considerations

### Input Validation

Always validate input files:

```python
from pipeline.data_validator import DataValidator

class SecureDataValidator(DataValidator):
    def validate_file_security(self, file_path: Path) -> bool:
        """Validate file security"""
        
        # Check file size limits
        if file_path.stat().st_size > 100 * 1024 * 1024:  # 100MB limit
            return False
        
        # Check file permissions
        if not os.access(file_path, os.R_OK):
            return False
        
        # Validate file content
        try:
            df = pd.read_excel(file_path, nrows=1)  # Read only first row
            return True
        except Exception:
            return False
```

### Access Control

Implement access control for sensitive operations:

```python
from functools import wraps

def require_permission(permission: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check user permissions
            if not check_user_permission(permission):
                raise PermissionError(f"Permission required: {permission}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

class SecurePipeline(ScalableFlexPipeline):
    @require_permission("pipeline.execute")
    def run_pipeline(self):
        return super().run_pipeline()
```

## Contributing Guidelines

### Code Style

Follow PEP 8 and use these tools:

```bash
# Format code
black pipeline/ tests/

# Check style
flake8 pipeline/ tests/

# Type checking
mypy pipeline/
```

### Pull Request Process

1. Create feature branch from `main`
2. Implement changes with tests
3. Update documentation
4. Run full test suite
5. Submit pull request with description

### Adding New Features

1. **Design Document**: Create design document for significant features
2. **Tests First**: Write tests before implementation
3. **Documentation**: Update user and developer guides
4. **Performance**: Consider performance implications
5. **Backward Compatibility**: Maintain API compatibility

This developer guide provides the foundation for extending and maintaining the Scalable Multi-File Pipeline. For specific implementation details, refer to the inline code documentation and test examples.