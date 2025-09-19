"""
Private Property Analysis Example
Demonstrates usage of PrivatePropertyAnalyzer with various scenarios and integrations

This example shows:
1. Basic property analysis workflow
2. Integration with FlexPropertyScorer
3. Database storage and retrieval
4. Export functionality
5. Error handling patterns
"""

import sys
import os
from pathlib import Path
import pandas as pd
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.private_property_analyzer import PrivatePropertyAnalyzer
from utils.logger import setup_logging


def create_sample_data():
    """Create sample property data for demonstration"""
    
    sample_properties = pd.DataFrame({
        'Property Name': [
            'Industrial Park West Building A',
            'Flex Warehouse Complex B', 
            'Manufacturing Facility C',
            'Office Building Downtown',
            'Retail Shopping Center',
            'Distribution Center North',
            'Mixed Use Development',
            'Logistics Hub South'
        ],
        'Property Type': [
            'Industrial',
            'Flex Industrial', 
            'Manufacturing',
            'Office',
            'Retail',
            'Warehouse',
            'Mixed Use',
            'Distribution'
        ],
        'Building SqFt': [
            75000,
            45000,
            120000,
            35000,
            25000,
            95000,
            60000,
            85000
        ],
        'Lot Size Acres': [
            8.5,
            4.2,
            15.3,
            2.1,
            3.8,
            12.7,
            5.5,
            10.2
        ],
        'City': [
            'West Palm Beach',
            'Boca Raton',
            'Jupiter',
            'Delray Beach',
            'Boynton Beach',
            'Lake Worth',
            'Palm Beach Gardens',
            'Wellington'
        ],
        'State': ['FL'] * 8,
        'Owner Name': [
            'Industrial Properties LLC',
            'Flex Development Corp',
            'Manufacturing Holdings Inc',
            'Office Real Estate Group',
            'Retail Investment Trust',
            'Logistics Properties LP',
            'Mixed Use Developers',
            'Distribution Holdings LLC'
        ],
        'Market Value': [
            3500000,
            2100000,
            5800000,
            1800000,
            1200000,
            4200000,
            2800000,
            3900000
        ],
        'Year Built': [
            1998,
            2005,
            1985,
            2010,
            1995,
            2001,
            2008,
            1992
        ],
        'Zoning Code': [
            'IL',
            'IP',
            'IG',
            'CG',
            'CR',
            'IL',
            'MUPD',
            'IG'
        ]
    })
    
    return sample_properties


def example_basic_analysis():
    """Example 1: Basic property analysis workflow"""
    
    print("\n" + "="*60)
    print("EXAMPLE 1: BASIC PROPERTY ANALYSIS")
    print("="*60)
    
    # Set up logging
    logger = setup_logging('example_basic', level='INFO')
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Save to temporary Excel file
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
        sample_data.to_excel(tmp_file.name, index=False)
        excel_file = tmp_file.name
    
    try:
        # Initialize analyzer
        logger.info("Initializing PrivatePropertyAnalyzer...")
        analyzer = PrivatePropertyAnalyzer(excel_file, logger=logger)
        
        # Load and analyze data
        logger.info("Loading property data...")
        data = analyzer.load_data()
        
        logger.info("Analyzing property types...")
        industrial_types = analyzer.analyze_property_types()
        
        logger.info("Checking data completeness...")
        completeness = analyzer.check_data_completeness()
        
        logger.info("Getting industrial property sample...")
        sample = analyzer.get_industrial_sample(limit=5)
        
        logger.info("Generating summary report...")
        report = analyzer.generate_summary_report()
        
        # Display results
        print(f"\n📊 Analysis Results:")
        print(f"   Total Properties: {len(data):,}")
        print(f"   Industrial Types Found: {len(industrial_types)}")
        print(f"   Industrial Properties: {report['property_type_analysis']['total_industrial_properties']}")
        print(f"   Average Data Quality: {report['data_quality_metrics']['average_completeness']:.1f}%")
        
        logger.info("✅ Basic analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Basic analysis failed: {e}")
        
    finally:
        # Cleanup
        os.unlink(excel_file)


def example_flex_scoring_integration():
    """Example 2: Integration with FlexPropertyScorer"""
    
    print("\n" + "="*60)
    print("EXAMPLE 2: FLEX SCORING INTEGRATION")
    print("="*60)
    
    # Set up logging
    logger = setup_logging('example_scoring', level='INFO')
    
    # Create sample data with more detailed property information
    enhanced_data = create_sample_data()
    
    # Save to temporary Excel file
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
        enhanced_data.to_excel(tmp_file.name, index=False)
        excel_file = tmp_file.name
    
    try:
        # Initialize analyzer
        analyzer = PrivatePropertyAnalyzer(excel_file, logger=logger)
        
        # Load data
        analyzer.load_data()
        
        # Add flex scoring to industrial properties
        logger.info("Adding flex scores to industrial properties...")
        scored_properties = analyzer.add_flex_scoring(include_all_properties=False)
        
        if not scored_properties.empty:
            # Display top scoring properties
            top_properties = scored_properties.nlargest(3, 'flex_score')
            
            print(f"\n🏆 Top 3 Flex Scoring Properties:")
            for idx, (_, prop) in enumerate(top_properties.iterrows(), 1):
                name = prop.get('Property Name', 'Unknown')
                score = prop.get('flex_score', 0)
                prop_type = prop.get('Property Type', 'Unknown')
                sqft = prop.get('Building SqFt', 0)
                acres = prop.get('Lot Size Acres', 0)
                
                print(f"   {idx}. {name}")
                print(f"      Score: {score:.2f}/10 | Type: {prop_type}")
                print(f"      Size: {sqft:,} sq ft | Lot: {acres:.1f} acres")
            
            logger.info(f"✅ Successfully scored {len(scored_properties)} properties")
        else:
            logger.warning("No properties were scored")
            
    except Exception as e:
        logger.error(f"Flex scoring integration failed: {e}")
        
    finally:
        # Cleanup
        os.unlink(excel_file)


def example_export_functionality():
    """Example 3: Export functionality demonstration"""
    
    print("\n" + "="*60)
    print("EXAMPLE 3: EXPORT FUNCTIONALITY")
    print("="*60)
    
    # Set up logging
    logger = setup_logging('example_export', level='INFO')
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Save to temporary Excel file
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
        sample_data.to_excel(tmp_file.name, index=False)
        excel_file = tmp_file.name
    
    try:
        # Initialize analyzer
        analyzer = PrivatePropertyAnalyzer(excel_file, logger=logger)
        
        # Run analysis
        analyzer.load_data()
        analyzer.generate_summary_report()
        
        # Export in multiple formats
        logger.info("Exporting analysis results...")
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as output_dir:
            exported_files = analyzer.export_results(
                output_dir=output_dir,
                formats=['json', 'excel', 'csv']
            )
            
            print(f"\n📁 Exported Files:")
            for format_name, file_path in exported_files.items():
                file_size = Path(file_path).stat().st_size
                print(f"   {format_name.upper()}: {Path(file_path).name} ({file_size:,} bytes)")
            
            logger.info("✅ Export completed successfully")
            
    except Exception as e:
        logger.error(f"Export functionality failed: {e}")
        
    finally:
        # Cleanup
        os.unlink(excel_file)


def example_database_integration():
    """Example 4: Database storage and retrieval"""
    
    print("\n" + "="*60)
    print("EXAMPLE 4: DATABASE INTEGRATION")
    print("="*60)
    
    # Set up logging
    logger = setup_logging('example_database', level='INFO')
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Save to temporary Excel file
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
        sample_data.to_excel(tmp_file.name, index=False)
        excel_file = tmp_file.name
    
    try:
        # Initialize analyzer
        analyzer = PrivatePropertyAnalyzer(excel_file, logger=logger)
        
        # Run analysis
        analyzer.load_data()
        report = analyzer.generate_summary_report()
        
        # Store results in database
        logger.info("Storing analysis results in database...")
        success = analyzer.store_results_in_database()
        
        if success:
            print(f"\n💾 Database Storage:")
            print(f"   ✅ Analysis results stored successfully")
            
            # Retrieve historical analyses
            logger.info("Retrieving historical analyses...")
            historical = analyzer.retrieve_historical_analysis(limit=3)
            
            print(f"   📚 Historical analyses found: {len(historical)}")
            
            # Compare with historical data
            logger.info("Comparing with historical data...")
            comparison = analyzer.compare_with_historical()
            
            if comparison.get('comparison_available'):
                trends = comparison.get('trends', {})
                print(f"   📈 Trends vs historical average:")
                print(f"      Properties: {trends.get('properties_vs_avg', 0):+.0f}")
                print(f"      Industrial: {trends.get('industrial_vs_avg', 0):+.0f}")
                print(f"      Quality: {trends.get('quality_vs_avg', 0):+.1f}%")
            
            logger.info("✅ Database integration completed successfully")
        else:
            logger.warning("Database storage not available")
            print(f"\n💾 Database Storage:")
            print(f"   ⚠️  Database not available (MongoDB may not be running)")
            
    except Exception as e:
        logger.error(f"Database integration failed: {e}")
        
    finally:
        # Cleanup
        os.unlink(excel_file)


def example_error_handling():
    """Example 5: Error handling patterns"""
    
    print("\n" + "="*60)
    print("EXAMPLE 5: ERROR HANDLING PATTERNS")
    print("="*60)
    
    # Set up logging
    logger = setup_logging('example_errors', level='INFO')
    
    # Test 1: Invalid file path
    print(f"\n🧪 Test 1: Invalid file path")
    try:
        analyzer = PrivatePropertyAnalyzer('nonexistent_file.xlsx', logger=logger)
        print(f"   ❌ Should have failed")
    except FileNotFoundError as e:
        print(f"   ✅ Correctly caught FileNotFoundError: {e}")
    
    # Test 2: Dirty data handling
    print(f"\n🧪 Test 2: Dirty data handling")
    
    # Create problematic data
    dirty_data = pd.DataFrame({
        'Property Name': ['Good Property', None, 'Property with Issues'],
        'Property Type': ['Industrial', None, 'Manufacturing'],
        'Building SqFt': [25000, 'N/A', '35,000'],  # Mixed formats
        'City': ['Valid City', '', None]
    })
    
    # Save to temporary Excel file
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
        dirty_data.to_excel(tmp_file.name, index=False)
        excel_file = tmp_file.name
    
    try:
        analyzer = PrivatePropertyAnalyzer(excel_file, logger=logger)
        analyzer.load_data()
        
        # Should handle dirty data gracefully
        industrial_types = analyzer.analyze_property_types()
        completeness = analyzer.check_data_completeness()
        sample = analyzer.get_industrial_sample()
        report = analyzer.generate_summary_report()
        
        print(f"   ✅ Successfully handled dirty data")
        print(f"      Industrial types found: {len(industrial_types)}")
        print(f"      Analysis completed with partial data")
        
    except Exception as e:
        print(f"   ❌ Error handling failed: {e}")
        
    finally:
        # Cleanup
        os.unlink(excel_file)
    
    logger.info("✅ Error handling examples completed")


def example_pipeline_integration():
    """Example 6: Integration with existing pipeline components"""
    
    print("\n" + "="*60)
    print("EXAMPLE 6: PIPELINE INTEGRATION")
    print("="*60)
    
    # Set up logging
    logger = setup_logging('example_pipeline', level='INFO')
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Save to temporary Excel file
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
        sample_data.to_excel(tmp_file.name, index=False)
        excel_file = tmp_file.name
    
    try:
        # Initialize analyzer
        analyzer = PrivatePropertyAnalyzer(excel_file, logger=logger)
        
        # Load data
        analyzer.load_data()
        
        # Demonstrate data format conversion
        logger.info("Converting Excel data to pipeline format...")
        first_property = analyzer.data.iloc[0]
        pipeline_format = analyzer.convert_to_pipeline_format(first_property)
        
        print(f"\n🔄 Data Format Conversion:")
        print(f"   Original Excel columns: {list(analyzer.data.columns)}")
        print(f"   Pipeline format keys: {list(pipeline_format.keys())}")
        print(f"   Example conversion:")
        print(f"      Property Type: '{first_property.get('Property Type')}' → property_use: '{pipeline_format['property_use']}'")
        print(f"      Building SqFt: {first_property.get('Building SqFt')} → building_sqft: {pipeline_format['building_sqft']}")
        print(f"      Lot Size Acres: {first_property.get('Lot Size Acres')} → acres: {pipeline_format['acres']}")
        
        # Add flex scoring (integrates with FlexPropertyScorer)
        logger.info("Integrating with FlexPropertyScorer...")
        scored_properties = analyzer.add_flex_scoring(include_all_properties=False)
        
        if not scored_properties.empty:
            print(f"   ✅ Successfully integrated with FlexPropertyScorer")
            print(f"   📊 Scored {len(scored_properties)} properties")
        
        # Store in database (integrates with MongoDB client)
        logger.info("Integrating with database infrastructure...")
        success = analyzer.store_results_in_database()
        
        if success:
            print(f"   ✅ Successfully integrated with MongoDB infrastructure")
        else:
            print(f"   ⚠️  Database integration not available")
        
        logger.info("✅ Pipeline integration examples completed")
        
    except Exception as e:
        logger.error(f"Pipeline integration failed: {e}")
        
    finally:
        # Cleanup
        os.unlink(excel_file)


def main():
    """Run all examples"""
    
    print("🚀 PRIVATE PROPERTY ANALYZER EXAMPLES")
    print("This script demonstrates various usage scenarios and integrations")
    
    try:
        # Run all examples
        example_basic_analysis()
        example_flex_scoring_integration()
        example_export_functionality()
        example_database_integration()
        example_error_handling()
        example_pipeline_integration()
        
        print("\n" + "="*60)
        print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("1. Try the CLI: python analyze_private_properties.py <your_excel_file>")
        print("2. Integrate with your existing pipeline workflows")
        print("3. Customize analysis parameters for your specific needs")
        print("4. Set up MongoDB for historical analysis tracking")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Examples interrupted by user")
        
    except Exception as e:
        print(f"\n\n❌ Examples failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()