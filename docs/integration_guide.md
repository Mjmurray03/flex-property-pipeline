# Integration Guide for Scalable Multi-File Pipeline

## Overview

The Scalable Multi-File Pipeline includes comprehensive integration utilities to connect with existing pipeline components and systems. This guide explains how to use these integration features to maintain compatibility with your current workflows while leveraging the new scalable processing capabilities.

## Integration Components

### 1. Component Compatibility Checker

Verifies compatibility with existing pipeline components before integration.

```python
from pipeline.integration_manager import ComponentCompatibilityChecker

checker = ComponentCompatibilityChecker()

# Check individual components
flex_compatible = checker.check_flex_classifier_compatibility()
db_compatible = checker.check_database_compatibility()
utils_compatible = checker.check_data_utils_compatibility()

# Run full compatibility check
results = checker.run_full_compatibility_check()
print(f"Compatibility results: {results}")
```

### 2. Legacy Data Migrator

Migrates data from single-file workflows to scalable pipeline format.

```python
from pipeline.integration_manager import LegacyDataMigrator

migrator = LegacyDataMigrator()

# Migrate legacy results file
result = migrator.migrate_single_file_results(
    legacy_file_path="legacy_results.xlsx",
    output_path="migrated_results.xlsx"
)

print(f"Migration success: {result.success}")
print(f"Records migrated: {result.records_integrated}")
```

### 3. Existing System Integrator

Integrates with existing FlexPropertyClassifier, database, and analysis components.

```python
from pipeline.integration_manager import ExistingSystemIntegrator
import pandas as pd

integrator = ExistingSystemIntegrator()

# Sample data from scalable pipeline
df = pd.DataFrame({
    'Address': ['123 Industrial St'],
    'City': ['Springfield'],
    'State': ['IL'],
    'Building SqFt': [25000],
    'Lot Size Acres': [2.5]
})

# Integrate with existing FlexPropertyClassifier
processed_df, result = integrator.integrate_with_flex_classifier(df)

# Integrate with database
db_result = integrator.integrate_with_database(df, 'flex_properties')

# Integrate with PrivatePropertyAnalyzer
analyzed_df, analysis_result = integrator.integrate_with_private_analyzer(df)
```

### 4. Output Format Converter

Converts scalable pipeline output to formats compatible with existing tools.

```python
from pipeline.integration_manager import OutputFormatConverter

converter = OutputFormatConverter()

# Convert to different legacy formats
formats = ['classic', 'enhanced', 'minimal']

for format_type in formats:
    result = converter.convert_to_legacy_format(
        df=pipeline_results,
        output_path=f"output_{format_type}.xlsx",
        format_type=format_type
    )
    print(f"{format_type} conversion: {result.success}")
```

## Complete Integration Workflow

Use the `IntegrationManager` for comprehensive integration workflows:

```python
from pipeline.integration_manager import IntegrationManager

# Create integration manager
manager = IntegrationManager()

# Define integration configuration
integration_config = {
    'check_compatibility': True,
    'integrate_flex_classifier': True,
    'integrate_database': True,
    'integrate_private_analyzer': False,
    'convert_legacy_format': True,
    'legacy_output_path': 'output/legacy_format.xlsx',
    'legacy_format_type': 'classic',
    'collection_name': 'scalable_pipeline_results'
}

# Run complete integration workflow
results = manager.run_integration_workflow(
    df=scalable_pipeline_results,
    integration_config=integration_config
)

# Check results
print(f"Integration success: {results['success']}")
print(f"Steps completed: {results['steps_completed']}")
print(f"Steps failed: {results['steps_failed']}")

# Get integration summary
summary = manager.get_integration_summary()
print(f"Success rate: {summary['success_rate']:.1f}%")
```

## Integration Scenarios

### Scenario 1: Gradual Migration

Gradually migrate from single-file to multi-file processing:

1. **Phase 1**: Use compatibility checker to verify existing components
2. **Phase 2**: Migrate existing results to new format
3. **Phase 3**: Run scalable pipeline alongside existing workflow
4. **Phase 4**: Integrate outputs with existing database and analysis tools

### Scenario 2: Hybrid Processing

Use both old and new systems simultaneously:

1. Process new data with scalable pipeline
2. Convert output to legacy format for existing tools
3. Integrate results with existing database
4. Generate reports using existing report generators

### Scenario 3: Complete Integration

Fully integrate scalable pipeline with existing ecosystem:

1. Replace single-file processor with scalable pipeline
2. Maintain existing database schema and connections
3. Keep existing analysis and reporting tools
4. Migrate all historical data to new format

## Column Mapping

The integration utilities automatically handle column name mapping between formats:

### Legacy to Scalable Pipeline Format

| Legacy Column | Scalable Pipeline Column |
|---------------|-------------------------|
| site_address | Address |
| property_address | Address |
| property_city | City |
| property_state | State |
| building_size | Building SqFt |
| building_sqft | Building SqFt |
| lot_acres | Lot Size Acres |
| lot_size | Lot Size Acres |
| property_type | Property Type |
| year_built | Year Built |
| score | Flex Score |
| flex_score | Flex Score |

### Scalable Pipeline to Legacy Format

| Scalable Pipeline Column | Classic Legacy Column |
|-------------------------|----------------------|
| Address | site_address |
| City | property_city |
| State | property_state |
| Building SqFt | building_size |
| Lot Size Acres | lot_acres |
| Property Type | property_type |
| Flex Score | flex_score |

## Error Handling

Integration utilities include comprehensive error handling:

```python
# Check integration results
if not result.success:
    print(f"Integration failed: {result.error_message}")
    print(f"Component: {result.component_name}")
    print(f"Processing time: {result.integration_time:.2f}s")
else:
    print(f"Successfully integrated {result.records_integrated} records")
```

## Best Practices

### 1. Always Check Compatibility First

```python
# Verify compatibility before running integration
checker = ComponentCompatibilityChecker()
compatibility = checker.run_full_compatibility_check()

if not all(compatibility.values()):
    print("Some components are not compatible")
    print("Check compatibility results before proceeding")
```

### 2. Test with Small Datasets

```python
# Test integration with a small sample first
sample_df = full_df.head(10)
test_results = manager.run_integration_workflow(sample_df, config)

if test_results['success']:
    # Proceed with full dataset
    full_results = manager.run_integration_workflow(full_df, config)
```

### 3. Monitor Integration Performance

```python
# Track integration performance
summary = manager.get_integration_summary()
print(f"Average integration time: {summary['total_integration_time']:.2f}s")
print(f"Records per second: {summary['total_records_integrated'] / summary['total_integration_time']:.0f}")
```

### 4. Handle Partial Failures Gracefully

```python
# Check for partial failures
if results['steps_failed']:
    print(f"Some integration steps failed: {results['steps_failed']}")
    print("Review failed steps and retry if necessary")

# Continue with successful integrations
successful_results = [r for r in results['integration_results'] if r.success]
```

## Migration Utilities

### Batch Migration Script

For migrating multiple legacy files:

```python
from pathlib import Path
from pipeline.integration_manager import LegacyDataMigrator

migrator = LegacyDataMigrator()
legacy_folder = Path("legacy_results")
output_folder = Path("migrated_results")

output_folder.mkdir(exist_ok=True)

for legacy_file in legacy_folder.glob("*.xlsx"):
    output_file = output_folder / f"migrated_{legacy_file.name}"
    
    result = migrator.migrate_single_file_results(
        str(legacy_file),
        str(output_file)
    )
    
    if result.success:
        print(f"✅ Migrated {legacy_file.name}")
    else:
        print(f"❌ Failed to migrate {legacy_file.name}: {result.error_message}")
```

### Database Migration

For migrating existing database records:

```python
# Load existing database records
existing_records = db_client.find_all('legacy_flex_properties')

# Convert to DataFrame
df = pd.DataFrame(existing_records)

# Standardize column names
migrator = LegacyDataMigrator()
standardized_df = migrator._standardize_legacy_columns(df)

# Add pipeline metadata
migrated_df = migrator._add_pipeline_metadata(standardized_df, 'database_migration')

# Save to new collection
integrator = ExistingSystemIntegrator()
result = integrator.integrate_with_database(migrated_df, 'scalable_pipeline_results')
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Some existing components may not be available
   - Solution: Use compatibility checker to identify missing components
   - Disable integration for unavailable components

2. **Column Name Mismatches**: Legacy data has different column names
   - Solution: Update column mappings in `LegacyDataMigrator`
   - Add custom mapping rules for your specific format

3. **Data Type Conflicts**: Different data types between systems
   - Solution: Use data validation and type conversion utilities
   - Implement custom data transformation functions

4. **Performance Issues**: Large datasets cause memory problems
   - Solution: Process data in chunks using performance optimizer
   - Use memory-efficient integration settings

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Run integration with detailed logging
manager = IntegrationManager()
results = manager.run_integration_workflow(df, config)
```

## Example Integration Scripts

See the `examples/integration_example.py` file for complete working examples of:

- Basic integration workflow
- Legacy data migration
- Format conversion
- Compatibility checking

Run the example:

```bash
python examples/integration_example.py
```

This will demonstrate all integration capabilities and create sample output files in the `output/` directory.

## Support

For integration support:

1. Check compatibility first using `ComponentCompatibilityChecker`
2. Review error messages in `IntegrationResult` objects
3. Use debug logging for detailed troubleshooting
4. Test with small datasets before full integration
5. Refer to existing component documentation for specific integration requirements

The integration utilities are designed to be flexible and extensible. You can customize the integration logic for your specific use case by extending the base classes or implementing custom integration workflows.