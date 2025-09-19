"""
Advanced Flex Property Classifier Demo
Demonstrates all advanced features with sample data
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.flex_config_manager import FlexConfigManager, create_sample_configs
from processors.advanced_flex_classifier import AdvancedFlexClassifier
from utils.batch_processor import FlexBatchProcessor
from utils.logger import setup_logging


def create_sample_data(n_properties: int = 100) -> pd.DataFrame:
    """Create sample property data for demonstration"""
    np.random.seed(42)
    
    property_types = ['Industrial', 'Warehouse', 'Flex Industrial', 'Distribution Center', 
                     'Light Industrial', 'Manufacturing', 'Office', 'Retail']
    
    cities = ['Dallas', 'Houston', 'Austin', 'San Antonio', 'Fort Worth', 'El Paso']
    counties = ['Dallas County', 'Harris County', 'Travis County', 'Bexar County', 'Tarrant County']
    
    data = pd.DataFrame({
        'Property Name': [f'Property {i+1}' for i in range(n_properties)],
        'Property Type': np.random.choice(property_types, n_properties),
        'Building SqFt': np.random.randint(10000, 300000, n_properties),
        'Lot Size Acres': np.random.uniform(0.2, 30.0, n_properties),
        'Year Built': np.random.randint(1960, 2023, n_properties),
        'Occupancy': np.random.uniform(50, 100, n_properties),
        'Address': [f'{1000+i} Industrial Blvd' for i in range(n_properties)],
        'City': np.random.choice(cities, n_properties),
        'State': ['TX'] * n_properties,
        'County': np.random.choice(counties, n_properties),
        'Zoning Code': np.random.choice(['I-1', 'I-2', 'I-3', 'M-1', 'M-2'], n_properties),
        'Sale Date': pd.date_range('2020-01-01', periods=n_properties, freq='3D').strftime('%Y-%m-%d'),
        'Sold Price': np.random.randint(500000, 15000000, n_properties),
        'Sold Price/SqFt': np.random.uniform(25, 150, n_properties),
        'Owner Name': [f'Owner {i+1} LLC' for i in range(n_properties)]
    })
    
    return data


def demo_configuration_management():
    """Demonstrate configuration management features"""
    print("=" * 60)
    print("CONFIGURATION MANAGEMENT DEMO")
    print("=" * 60)
    
    # Create configuration manager
    config_manager = FlexConfigManager()
    
    # Load default configuration
    print("1. Loading default configuration...")
    default_config = config_manager.load_config()
    print(f"   Default min building sqft: {default_config.filtering_criteria.min_building_sqft:,}")
    print(f"   Default batch size: {default_config.advanced_settings.batch_size}")
    
    # Create sample configurations
    print("\n2. Creating sample configurations...")
    sample_configs = create_sample_configs()
    
    for config_name, config in sample_configs.items():
        print(f"   {config_name.title()} Configuration:")
        print(f"     Min building sqft: {config.filtering_criteria.min_building_sqft:,}")
        print(f"     Lot size range: {config.filtering_criteria.min_lot_acres}-{config.filtering_criteria.max_lot_acres} acres")
        print(f"     Batch processing: {config.advanced_settings.enable_batch_processing}")
    
    # Create custom configuration
    print("\n3. Creating custom configuration...")
    custom_config = config_manager.create_custom_config(
        scoring_adjustments={
            'building_size_weight': 1.5,
            'property_type_weight': 2.0
        },
        filtering_adjustments={
            'min_building_sqft': 25000,
            'min_lot_acres': 1.0
        },
        advanced_adjustments={
            'enable_batch_processing': True,
            'batch_size': 500,
            'enable_geographic_analysis': True,
            'export_formats': ['xlsx', 'csv', 'json']
        }
    )
    
    print(f"   Custom building size weight: {custom_config.scoring_weights.building_size_weight}")
    print(f"   Custom property type weight: {custom_config.scoring_weights.property_type_weight}")
    
    # Validate configuration
    print("\n4. Validating configuration...")
    issues = config_manager.validate_config(custom_config)
    if issues:
        print(f"   Validation issues found: {issues}")
    else:
        print("   Configuration is valid!")
    
    return custom_config


def demo_advanced_classifier(config):
    """Demonstrate advanced classifier features"""
    print("\n" + "=" * 60)
    print("ADVANCED CLASSIFIER DEMO")
    print("=" * 60)
    
    # Create sample data
    print("1. Creating sample property data...")
    data = create_sample_data(200)
    print(f"   Created {len(data)} properties")
    
    # Create advanced classifier
    print("\n2. Initializing advanced classifier...")
    classifier = AdvancedFlexClassifier(data, config)
    
    # Set up progress tracking
    progress_updates = []
    def progress_callback(current, total, message):
        progress_updates.append((current, total, message))
        if total > 1:  # Only show for batch processing
            percentage = (current / total * 100) if total > 0 else 0
            print(f"   Progress: {current}/{total} ({percentage:.1f}%) - {message}")
    
    classifier.set_progress_callback(progress_callback)
    
    # Process with batch processing
    print("\n3. Processing with batch processing...")
    candidates = classifier.classify_flex_properties_batch()
    print(f"   Found {len(candidates)} flex candidates")
    
    if len(candidates) > 0:
        # Demonstrate advanced scoring
        print("\n4. Advanced scoring breakdown (top 3 candidates):")
        for i, (_, row) in enumerate(candidates.head(3).iterrows()):
            score, breakdown = classifier.calculate_flex_score_advanced(row)
            print(f"   Property {i+1}: {row.get('Property Name', 'Unknown')}")
            print(f"     Total Score: {score:.2f}")
            print(f"     Building Size: {breakdown.get('building_size', 0):.2f}")
            print(f"     Property Type: {breakdown.get('property_type', 0):.2f}")
            print(f"     Lot Size: {breakdown.get('lot_size', 0):.2f}")
        
        # Perform analytics
        print("\n5. Performing advanced analytics...")
        
        # Geographic analysis
        geo_analysis = classifier.perform_geographic_analysis()
        print(f"   Geographic Analysis:")
        print(f"     States: {len(geo_analysis.state_distribution)}")
        print(f"     Cities: {len(geo_analysis.city_distribution)}")
        print(f"     Top markets: {geo_analysis.top_markets[:3]}")
        
        # Size distribution analysis
        size_analysis = classifier.perform_size_distribution_analysis()
        print(f"   Size Distribution Analysis:")
        print(f"     Building size buckets: {len(size_analysis.building_size_distribution)}")
        print(f"     Lot size buckets: {len(size_analysis.lot_size_distribution)}")
        
        # Export results
        print("\n6. Exporting results...")
        with tempfile.TemporaryDirectory() as temp_dir:
            exported_files = classifier.export_advanced_results(
                Path(temp_dir),
                include_analytics=True
            )
            
            print(f"   Exported {len(exported_files)} files:")
            for format_type, file_path in exported_files.items():
                file_size = Path(file_path).stat().st_size / 1024  # KB
                print(f"     {format_type}: {Path(file_path).name} ({file_size:.1f} KB)")
        
        # Performance report
        print("\n7. Performance report:")
        performance = classifier.get_performance_report()
        metrics = performance['processing_metrics']
        print(f"   Processing rate: {metrics.get('processing_rate', 0):.1f} properties/second")
        print(f"   Total processing time: {metrics.get('total_processing_time', 0):.2f} seconds")
        print(f"   Conversion rate: {performance['data_summary']['conversion_rate']:.1f}%")
        
        if performance['recommendations']:
            print(f"   Recommendations:")
            for rec in performance['recommendations']:
                print(f"     - {rec}")
    
    return candidates


def demo_batch_processing():
    """Demonstrate batch processing features"""
    print("\n" + "=" * 60)
    print("BATCH PROCESSING DEMO")
    print("=" * 60)
    
    # Create temporary Excel files for batch processing
    print("1. Creating sample Excel files...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create multiple Excel files
        file_paths = []
        for i in range(3):
            data = create_sample_data(50)  # Smaller datasets for demo
            file_path = temp_path / f"properties_{i+1}.xlsx"
            data.to_excel(file_path, index=False, engine='openpyxl')
            file_paths.append(file_path)
        
        print(f"   Created {len(file_paths)} Excel files")
        
        # Create batch processor
        print("\n2. Initializing batch processor...")
        processor = FlexBatchProcessor()
        
        # Discover files
        print("\n3. Discovering Excel files...")
        discovered_files = processor.discover_excel_files([temp_path])
        print(f"   Discovered {len(discovered_files)} files")
        
        # Set up progress tracking
        def batch_progress_callback(completed, total, current_file, elapsed_time):
            percentage = (completed / total * 100) if total > 0 else 0
            rate = completed / elapsed_time if elapsed_time > 0 else 0
            print(f"   Progress: {completed}/{total} ({percentage:.1f}%) - {rate:.1f} files/sec - {Path(current_file).name}")
        
        # Process files
        print("\n4. Processing files in batch...")
        summary = processor.process_files(
            discovered_files,
            progress_callback=batch_progress_callback
        )
        
        print(f"\n5. Batch processing results:")
        print(f"   Total files: {summary.total_files}")
        print(f"   Successful: {summary.successful_files}")
        print(f"   Failed: {summary.failed_files}")
        print(f"   Total properties: {summary.total_properties:,}")
        print(f"   Total candidates: {summary.total_candidates:,}")
        print(f"   Processing time: {summary.total_processing_time:.2f} seconds")
        print(f"   Processing rate: {summary.properties_per_second:.1f} properties/second")
        
        if summary.error_summary:
            print(f"   Errors: {dict(summary.error_summary)}")
        
        # Show aggregated results
        if processor.aggregated_results is not None and len(processor.aggregated_results) > 0:
            print(f"\n6. Aggregated results:")
            print(f"   Total aggregated candidates: {len(processor.aggregated_results)}")
            
            # Show top candidates
            if 'flex_score' in processor.aggregated_results.columns:
                top_candidates = processor.aggregated_results.nlargest(3, 'flex_score')
                print(f"   Top 3 candidates by score:")
                for i, (_, row) in enumerate(top_candidates.iterrows()):
                    print(f"     {i+1}. {row.get('Property Name', 'Unknown')} - Score: {row.get('flex_score', 0):.2f}")


def demo_performance_comparison():
    """Demonstrate performance comparison between standard and advanced processing"""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON DEMO")
    print("=" * 60)
    
    import time
    
    # Create larger dataset for performance testing
    print("1. Creating large dataset for performance testing...")
    large_data = create_sample_data(1000)
    print(f"   Created {len(large_data)} properties")
    
    # Test standard processing
    print("\n2. Testing standard processing...")
    from processors.flex_property_classifier import FlexPropertyClassifier
    
    standard_classifier = FlexPropertyClassifier(large_data)
    
    start_time = time.time()
    standard_candidates = standard_classifier.classify_flex_properties()
    standard_time = time.time() - start_time
    
    standard_rate = len(large_data) / standard_time
    print(f"   Standard processing: {standard_time:.2f}s, {standard_rate:.1f} properties/sec")
    print(f"   Found {len(standard_candidates)} candidates")
    
    # Test advanced processing with batch processing
    print("\n3. Testing advanced batch processing...")
    config_manager = FlexConfigManager()
    config = config_manager.create_custom_config(
        advanced_adjustments={
            'enable_batch_processing': True,
            'batch_size': 200
        }
    )
    
    advanced_classifier = AdvancedFlexClassifier(large_data, config)
    
    start_time = time.time()
    advanced_candidates = advanced_classifier.classify_flex_properties_batch()
    advanced_time = time.time() - start_time
    
    advanced_rate = len(large_data) / advanced_time
    print(f"   Advanced batch processing: {advanced_time:.2f}s, {advanced_rate:.1f} properties/sec")
    print(f"   Found {len(advanced_candidates)} candidates")
    
    # Compare results
    print(f"\n4. Performance comparison:")
    improvement = (standard_time - advanced_time) / standard_time * 100
    print(f"   Time improvement: {improvement:.1f}%")
    print(f"   Rate improvement: {(advanced_rate - standard_rate) / standard_rate * 100:.1f}%")
    
    # Verify results are consistent
    if len(standard_candidates) == len(advanced_candidates):
        print(f"   Results consistency: ✓ Same number of candidates found")
    else:
        print(f"   Results consistency: ⚠ Different candidate counts (standard: {len(standard_candidates)}, advanced: {len(advanced_candidates)})")


def main():
    """Run all demonstrations"""
    print("Advanced Flex Property Classifier - Feature Demonstration")
    print("=" * 60)
    
    # Set up logging
    logger = setup_logging(name='advanced_demo', level='INFO')
    
    try:
        # Demo 1: Configuration Management
        custom_config = demo_configuration_management()
        
        # Demo 2: Advanced Classifier
        candidates = demo_advanced_classifier(custom_config)
        
        # Demo 3: Batch Processing
        demo_batch_processing()
        
        # Demo 4: Performance Comparison
        demo_performance_comparison()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("All advanced features demonstrated successfully!")
        print("\nKey features shown:")
        print("✓ Configurable scoring criteria and weights")
        print("✓ Batch processing with progress tracking")
        print("✓ Advanced analytics (geographic and size distribution)")
        print("✓ Multiple export formats")
        print("✓ Performance optimization")
        print("✓ Multi-file processing capabilities")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()