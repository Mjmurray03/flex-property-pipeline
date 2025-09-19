#!/usr/bin/env python3
"""
Integration Example for Scalable Multi-File Pipeline
Demonstrates how to integrate with existing pipeline components
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.integration_manager import IntegrationManager
from pipeline.scalable_flex_pipeline import ScalableFlexPipeline, PipelineConfiguration
from run_scalable_pipeline import integrate_pipeline_components, run_complete_pipeline


def demonstrate_basic_integration():
    """Demonstrate basic integration workflow"""
    print("=" * 80)
    print("BASIC INTEGRATION EXAMPLE")
    print("=" * 80)
    
    # Create sample data that would come from scalable pipeline
    sample_results = pd.DataFrame({
        'Address': ['123 Industrial Blvd', '456 Warehouse Way', '789 Flex Street'],
        'City': ['Springfield', 'Riverside', 'Lakewood'],
        'State': ['IL', 'CA', 'CO'],
        'Property Type': ['Industrial', 'Warehouse', 'Flex'],
        'Building SqFt': [25000, 18000, 32000],
        'Lot Size Acres': [2.5, 1.8, 3.2],
        'Flex Score': [8.5, 7.2, 9.1],
        'Source_File': ['properties_1.xlsx', 'properties_2.xlsx', 'properties_3.xlsx']
    })
    
    print(f"Sample pipeline results: {len(sample_results)} properties")
    print(sample_results[['Address', 'City', 'State', 'Flex Score']].to_string(index=False))
    
    # Create integration manager
    integration_manager = IntegrationManager()
    
    # Define integration configuration
    integration_config = {
        'check_compatibility': True,
        'integrate_flex_classifier': False,  # Set to True if you have the existing classifier
        'integrate_database': False,         # Set to True if you have database setup
        'integrate_private_analyzer': False, # Set to True if you have the analyzer
        'convert_legacy_format': True,
        'legacy_output_path': 'output/legacy_format_example.xlsx',
        'legacy_format_type': 'classic'
    }
    
    # Run integration workflow
    print(f"\nRunning integration workflow...")
    workflow_results = integration_manager.run_integration_workflow(
        sample_results, 
        integration_config
    )
    
    # Display results
    print(f"\nIntegration Workflow Results:")
    print(f"Success: {workflow_results['success']}")
    print(f"Steps Completed: {workflow_results['steps_completed']}")
    print(f"Steps Failed: {workflow_results['steps_failed']}")
    print(f"Total Records: {workflow_results['total_records']}")
    print(f"Workflow Time: {workflow_results['workflow_time']:.2f} seconds")
    
    # Show integration summary
    summary = integration_manager.get_integration_summary()
    print(f"\nIntegration Summary:")
    print(f"Total Operations: {summary.get('total_operations', 0)}")
    print(f"Success Rate: {summary.get('success_rate', 0):.1f}%")
    
    return workflow_results


def demonstrate_legacy_migration():
    """Demonstrate legacy data migration"""
    print("\n" + "=" * 80)
    print("LEGACY DATA MIGRATION EXAMPLE")
    print("=" * 80)
    
    # Create sample legacy data
    legacy_data = pd.DataFrame({
        'site_address': ['100 Old Industrial Way', '200 Legacy Blvd'],
        'property_city': ['Old Town', 'Legacy City'],
        'property_state': ['TX', 'FL'],
        'building_size': [15000, 22000],
        'lot_acres': [1.2, 2.0],
        'property_type': ['Manufacturing', 'Distribution'],
        'score': [6.8, 7.5]
    })
    
    print(f"Legacy data format:")
    print(legacy_data.to_string(index=False))
    
    # Save legacy data to temporary file
    legacy_file = Path('temp_legacy_data.xlsx')
    legacy_data.to_excel(legacy_file, index=False)
    
    try:
        # Create migrator and migrate data
        from pipeline.integration_manager import LegacyDataMigrator
        
        migrator = LegacyDataMigrator()
        migration_result = migrator.migrate_single_file_results(
            str(legacy_file),
            'output/migrated_legacy_data.xlsx'
        )
        
        print(f"\nMigration Results:")
        print(f"Success: {migration_result.success}")
        print(f"Records Processed: {migration_result.records_processed}")
        print(f"Records Integrated: {migration_result.records_integrated}")
        print(f"Migration Time: {migration_result.integration_time:.2f} seconds")
        
        if migration_result.success:
            # Load and display migrated data
            migrated_data = pd.read_excel('output/migrated_legacy_data.xlsx')
            print(f"\nMigrated data format:")
            print(migrated_data[['Address', 'City', 'State', 'Building SqFt', 'Flex Score']].to_string(index=False))
        
    finally:
        # Cleanup temporary file
        if legacy_file.exists():
            legacy_file.unlink()


def demonstrate_format_conversion():
    """Demonstrate output format conversion"""
    print("\n" + "=" * 80)
    print("FORMAT CONVERSION EXAMPLE")
    print("=" * 80)
    
    # Create sample scalable pipeline output
    pipeline_output = pd.DataFrame({
        'Address': ['300 Modern Ave', '400 New Industrial Pkwy'],
        'City': ['Tech City', 'Innovation Hub'],
        'State': ['CA', 'WA'],
        'Building SqFt': [35000, 28000],
        'Lot Size Acres': [3.5, 2.8],
        'Property Type': ['Tech Flex', 'R&D'],
        'Flex Score': [9.2, 8.7],
        'Source_File': ['modern_properties.xlsx', 'tech_properties.xlsx'],
        'Processing_Date': ['2024-01-15', '2024-01-15'],
        'Pipeline_Version': ['scalable_v1.0', 'scalable_v1.0']
    })
    
    print(f"Scalable pipeline output:")
    print(pipeline_output[['Address', 'City', 'State', 'Flex Score']].to_string(index=False))
    
    # Create format converter
    from pipeline.integration_manager import OutputFormatConverter
    
    converter = OutputFormatConverter()
    
    # Convert to different formats
    formats_to_test = ['classic', 'enhanced', 'minimal']
    
    for format_type in formats_to_test:
        output_file = f'output/converted_{format_type}_format.xlsx'
        
        conversion_result = converter.convert_to_legacy_format(
            pipeline_output,
            output_file,
            format_type
        )
        
        print(f"\n{format_type.title()} Format Conversion:")
        print(f"Success: {conversion_result.success}")
        print(f"Records Processed: {conversion_result.records_processed}")
        print(f"Output File: {output_file}")
        
        if conversion_result.success:
            # Load and show sample of converted data
            converted_data = pd.read_excel(output_file)
            print(f"Converted columns: {list(converted_data.columns)}")


def demonstrate_compatibility_check():
    """Demonstrate compatibility checking"""
    print("\n" + "=" * 80)
    print("COMPATIBILITY CHECK EXAMPLE")
    print("=" * 80)
    
    from pipeline.integration_manager import ComponentCompatibilityChecker
    
    checker = ComponentCompatibilityChecker()
    
    print("Checking compatibility with existing components...")
    
    # Run individual compatibility checks
    components = [
        ('FlexPropertyClassifier', checker.check_flex_classifier_compatibility),
        ('Database', checker.check_database_compatibility),
        ('Data Utils', checker.check_data_utils_compatibility)
    ]
    
    for component_name, check_function in components:
        try:
            is_compatible = check_function()
            status = "✅ COMPATIBLE" if is_compatible else "❌ NOT COMPATIBLE"
            print(f"{component_name}: {status}")
        except Exception as e:
            print(f"{component_name}: ❌ ERROR - {e}")
    
    # Run full compatibility check
    print(f"\nRunning full compatibility check...")
    full_results = checker.run_full_compatibility_check()
    
    print(f"Full Compatibility Results:")
    for component, is_compatible in full_results.items():
        status = "✅ COMPATIBLE" if is_compatible else "❌ NOT COMPATIBLE"
        print(f"  {component}: {status}")
    
    # Show detailed compatibility results
    if checker.compatibility_results:
        print(f"\nDetailed Compatibility Information:")
        for component, details in checker.compatibility_results.items():
            print(f"  {component}:")
            if details.get('compatible'):
                print(f"    Version: {details.get('version', 'unknown')}")
                if 'methods' in details:
                    print(f"    Available methods: {len(details['methods'])}")
            else:
                print(f"    Error: {details.get('error', 'unknown')}")


def main():
    """Main demonstration function"""
    print("Scalable Multi-File Pipeline Integration Examples")
    print("This script demonstrates various integration capabilities")
    
    # Ensure output directory exists
    Path('output').mkdir(exist_ok=True)
    
    try:
        # Run demonstrations
        demonstrate_compatibility_check()
        demonstrate_basic_integration()
        demonstrate_legacy_migration()
        demonstrate_format_conversion()
        
        print("\n" + "=" * 80)
        print("INTEGRATION EXAMPLES COMPLETED")
        print("=" * 80)
        print("Check the 'output/' directory for generated files:")
        
        output_files = list(Path('output').glob('*.xlsx'))
        for file in output_files:
            print(f"  - {file}")
        
        print(f"\nIntegration utilities are ready for use with your existing pipeline components!")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())