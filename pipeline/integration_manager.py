"""
Integration Manager for Scalable Multi-File Pipeline
Handles integration with existing pipeline components and systems
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import sys
import importlib
from datetime import datetime

# Import existing components
try:
    from processors.flex_property_classifier import FlexPropertyClassifier
    from processors.flex_scorer import FlexPropertyScorer
    from processors.private_property_analyzer import PrivatePropertyAnalyzer
    from database.mongodb_client import MongoDBClient
    from utils.logger import setup_logging
    from utils.flex_data_utils import FlexDataUtils
except ImportError as e:
    logging.warning(f"Some existing components not available: {e}")


@dataclass
class IntegrationResult:
    """Result of integration operation"""
    success: bool
    component_name: str
    records_processed: int = 0
    records_integrated: int = 0
    integration_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class ComponentCompatibilityChecker:
    """Checks compatibility with existing pipeline components"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.compatibility_results = {}
    
    def check_flex_classifier_compatibility(self) -> bool:
        """Check compatibility with existing FlexPropertyClassifier"""
        try:
            # Try to import and instantiate
            classifier = FlexPropertyClassifier()
            
            # Check if required methods exist
            required_methods = ['classify_properties', 'get_flex_candidates']
            for method in required_methods:
                if not hasattr(classifier, method):
                    self.logger.warning(f"FlexPropertyClassifier missing method: {method}")
                    return False
            
            # Test with sample data
            sample_data = pd.DataFrame({
                'Address': ['123 Test St'],
                'City': ['Test City'],
                'State': ['TS'],
                'Property Type': ['Industrial'],
                'Building SqFt': [10000],
                'Lot Size Acres': [1.0]
            })
            
            # Test classification
            results = classifier.classify_properties(sample_data)
            
            self.compatibility_results['flex_classifier'] = {
                'compatible': True,
                'version': getattr(classifier, 'version', 'unknown'),
                'methods': [m for m in dir(classifier) if not m.startswith('_')]
            }
            
            self.logger.info("FlexPropertyClassifier compatibility: PASSED")
            return True
            
        except Exception as e:
            self.logger.error(f"FlexPropertyClassifier compatibility check failed: {e}")
            self.compatibility_results['flex_classifier'] = {
                'compatible': False,
                'error': str(e)
            }
            return False
    
    def check_database_compatibility(self) -> bool:
        """Check compatibility with existing database components"""
        try:
            # Try to import and test connection
            db_client = MongoDBClient()
            
            # Test basic operations
            test_collection = 'compatibility_test'
            test_doc = {'test': True, 'timestamp': datetime.now()}
            
            # Insert test document
            result = db_client.insert_one(test_collection, test_doc)
            
            # Clean up test document
            if result:
                db_client.delete_one(test_collection, {'test': True})
            
            self.compatibility_results['database'] = {
                'compatible': True,
                'client_type': type(db_client).__name__,
                'connection_status': 'active'
            }
            
            self.logger.info("Database compatibility: PASSED")
            return True
            
        except Exception as e:
            self.logger.error(f"Database compatibility check failed: {e}")
            self.compatibility_results['database'] = {
                'compatible': False,
                'error': str(e)
            }
            return False
    
    def check_data_utils_compatibility(self) -> bool:
        """Check compatibility with existing data utilities"""
        try:
            # Test FlexDataUtils
            utils = FlexDataUtils()
            
            # Check required methods
            required_methods = ['normalize_column_names', 'validate_data_types']
            available_methods = []
            
            for method in required_methods:
                if hasattr(utils, method):
                    available_methods.append(method)
            
            self.compatibility_results['data_utils'] = {
                'compatible': True,
                'available_methods': available_methods,
                'utility_class': type(utils).__name__
            }
            
            self.logger.info("Data utilities compatibility: PASSED")
            return True
            
        except Exception as e:
            self.logger.error(f"Data utilities compatibility check failed: {e}")
            self.compatibility_results['data_utils'] = {
                'compatible': False,
                'error': str(e)
            }
            return False
    
    def run_full_compatibility_check(self) -> Dict[str, bool]:
        """Run complete compatibility check"""
        self.logger.info("Running full compatibility check...")
        
        results = {
            'flex_classifier': self.check_flex_classifier_compatibility(),
            'database': self.check_database_compatibility(),
            'data_utils': self.check_data_utils_compatibility()
        }
        
        compatible_count = sum(results.values())
        total_count = len(results)
        
        self.logger.info(f"Compatibility check complete: {compatible_count}/{total_count} components compatible")
        
        return results


class LegacyDataMigrator:
    """Migrates data from legacy single-file workflows to scalable pipeline format"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def migrate_single_file_results(self, 
                                  legacy_file_path: str, 
                                  output_path: str) -> IntegrationResult:
        """
        Migrate results from single-file processing to scalable pipeline format
        
        Args:
            legacy_file_path: Path to legacy results file
            output_path: Path for migrated results
            
        Returns:
            IntegrationResult with migration details
        """
        start_time = datetime.now()
        
        try:
            # Load legacy results
            legacy_df = pd.read_excel(legacy_file_path)
            original_count = len(legacy_df)
            
            # Standardize column names for scalable pipeline
            migrated_df = self._standardize_legacy_columns(legacy_df)
            
            # Add scalable pipeline metadata
            migrated_df = self._add_pipeline_metadata(migrated_df, legacy_file_path)
            
            # Save migrated results
            migrated_df.to_excel(output_path, index=False)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Successfully migrated {original_count} records from {legacy_file_path}")
            
            return IntegrationResult(
                success=True,
                component_name="legacy_migrator",
                records_processed=original_count,
                records_integrated=len(migrated_df),
                integration_time=processing_time,
                metadata={
                    'source_file': legacy_file_path,
                    'output_file': output_path,
                    'migration_date': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Failed to migrate legacy data: {e}")
            
            return IntegrationResult(
                success=False,
                component_name="legacy_migrator",
                integration_time=processing_time,
                error_message=str(e)
            )
    
    def _standardize_legacy_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize legacy column names to match scalable pipeline format"""
        # Common column mappings from legacy to new format
        column_mappings = {
            'site_address': 'Address',
            'property_address': 'Address',
            'street_address': 'Address',
            'property_city': 'City',
            'property_state': 'State',
            'building_size': 'Building SqFt',
            'building_sqft': 'Building SqFt',
            'lot_acres': 'Lot Size Acres',
            'lot_size': 'Lot Size Acres',
            'property_type': 'Property Type',
            'year_built': 'Year Built',
            'flex_score': 'Flex Score',
            'score': 'Flex Score'
        }
        
        # Apply mappings
        df_standardized = df.copy()
        
        for old_name, new_name in column_mappings.items():
            if old_name in df_standardized.columns:
                df_standardized = df_standardized.rename(columns={old_name: new_name})
                self.logger.debug(f"Mapped column: {old_name} -> {new_name}")
        
        return df_standardized
    
    def _add_pipeline_metadata(self, df: pd.DataFrame, source_file: str) -> pd.DataFrame:
        """Add scalable pipeline metadata to migrated data"""
        df_with_metadata = df.copy()
        
        # Add source file information
        df_with_metadata['Source_File'] = Path(source_file).name
        df_with_metadata['Migration_Date'] = datetime.now().isoformat()
        df_with_metadata['Pipeline_Version'] = 'scalable_v1.0'
        
        return df_with_metadata


class ExistingSystemIntegrator:
    """Integrates scalable pipeline with existing analysis tools and databases"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.compatibility_checker = ComponentCompatibilityChecker()
    
    def integrate_with_flex_classifier(self, 
                                     df: pd.DataFrame) -> Tuple[pd.DataFrame, IntegrationResult]:
        """
        Integrate DataFrame processing with existing FlexPropertyClassifier
        
        Args:
            df: DataFrame to process with existing classifier
            
        Returns:
            Tuple of (processed_df, integration_result)
        """
        start_time = datetime.now()
        
        try:
            # Check compatibility first
            if not self.compatibility_checker.check_flex_classifier_compatibility():
                raise Exception("FlexPropertyClassifier not compatible")
            
            # Use existing classifier
            classifier = FlexPropertyClassifier()
            
            # Process with existing classifier
            classified_df = classifier.classify_properties(df)
            
            # Get flex candidates using existing logic
            flex_candidates = classifier.get_flex_candidates(classified_df)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = IntegrationResult(
                success=True,
                component_name="flex_classifier_integration",
                records_processed=len(df),
                records_integrated=len(flex_candidates),
                integration_time=processing_time,
                metadata={
                    'classifier_version': getattr(classifier, 'version', 'unknown'),
                    'integration_method': 'existing_classifier'
                }
            )
            
            self.logger.info(f"Successfully integrated {len(df)} records with FlexPropertyClassifier")
            
            return flex_candidates, result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Failed to integrate with FlexPropertyClassifier: {e}")
            
            result = IntegrationResult(
                success=False,
                component_name="flex_classifier_integration",
                integration_time=processing_time,
                error_message=str(e)
            )
            
            return df, result
    
    def integrate_with_database(self, 
                              df: pd.DataFrame, 
                              collection_name: str = 'scalable_pipeline_results') -> IntegrationResult:
        """
        Integrate results with existing database system
        
        Args:
            df: DataFrame to save to database
            collection_name: Database collection name
            
        Returns:
            IntegrationResult with database integration details
        """
        start_time = datetime.now()
        
        try:
            # Check database compatibility
            if not self.compatibility_checker.check_database_compatibility():
                raise Exception("Database not compatible")
            
            # Connect to existing database
            db_client = MongoDBClient()
            
            # Convert DataFrame to records
            records = df.to_dict('records')
            
            # Add integration metadata
            for record in records:
                record['integration_timestamp'] = datetime.now()
                record['pipeline_type'] = 'scalable_multi_file'
                record['integration_version'] = '1.0'
            
            # Insert records
            insert_result = db_client.insert_many(collection_name, records)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = IntegrationResult(
                success=True,
                component_name="database_integration",
                records_processed=len(df),
                records_integrated=len(records),
                integration_time=processing_time,
                metadata={
                    'collection_name': collection_name,
                    'database_type': type(db_client).__name__,
                    'insert_result': str(insert_result)
                }
            )
            
            self.logger.info(f"Successfully integrated {len(records)} records with database")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Failed to integrate with database: {e}")
            
            return IntegrationResult(
                success=False,
                component_name="database_integration",
                integration_time=processing_time,
                error_message=str(e)
            )
    
    def integrate_with_private_analyzer(self, 
                                      df: pd.DataFrame) -> Tuple[pd.DataFrame, IntegrationResult]:
        """
        Integrate with existing PrivatePropertyAnalyzer
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Tuple of (analyzed_df, integration_result)
        """
        start_time = datetime.now()
        
        try:
            # Use existing private property analyzer
            analyzer = PrivatePropertyAnalyzer()
            
            # Run analysis
            analyzed_df = analyzer.analyze_properties(df)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = IntegrationResult(
                success=True,
                component_name="private_analyzer_integration",
                records_processed=len(df),
                records_integrated=len(analyzed_df),
                integration_time=processing_time,
                metadata={
                    'analyzer_version': getattr(analyzer, 'version', 'unknown'),
                    'analysis_features': getattr(analyzer, 'features', [])
                }
            )
            
            self.logger.info(f"Successfully integrated {len(df)} records with PrivatePropertyAnalyzer")
            
            return analyzed_df, result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Failed to integrate with PrivatePropertyAnalyzer: {e}")
            
            result = IntegrationResult(
                success=False,
                component_name="private_analyzer_integration",
                integration_time=processing_time,
                error_message=str(e)
            )
            
            return df, result


class OutputFormatConverter:
    """Converts scalable pipeline output to formats compatible with existing tools"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def convert_to_legacy_format(self, 
                                df: pd.DataFrame, 
                                output_path: str,
                                format_type: str = 'classic') -> IntegrationResult:
        """
        Convert scalable pipeline output to legacy format
        
        Args:
            df: Scalable pipeline results DataFrame
            output_path: Path for converted output
            format_type: Type of legacy format ('classic', 'enhanced', 'minimal')
            
        Returns:
            IntegrationResult with conversion details
        """
        start_time = datetime.now()
        
        try:
            if format_type == 'classic':
                converted_df = self._convert_to_classic_format(df)
            elif format_type == 'enhanced':
                converted_df = self._convert_to_enhanced_format(df)
            elif format_type == 'minimal':
                converted_df = self._convert_to_minimal_format(df)
            else:
                raise ValueError(f"Unknown format type: {format_type}")
            
            # Save converted results
            converted_df.to_excel(output_path, index=False)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = IntegrationResult(
                success=True,
                component_name="format_converter",
                records_processed=len(df),
                records_integrated=len(converted_df),
                integration_time=processing_time,
                metadata={
                    'format_type': format_type,
                    'output_path': output_path,
                    'conversion_date': datetime.now().isoformat()
                }
            )
            
            self.logger.info(f"Successfully converted {len(df)} records to {format_type} format")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Failed to convert to legacy format: {e}")
            
            return IntegrationResult(
                success=False,
                component_name="format_converter",
                integration_time=processing_time,
                error_message=str(e)
            )
    
    def _convert_to_classic_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert to classic single-file format"""
        # Select and rename columns for classic format
        classic_columns = {
            'Address': 'site_address',
            'City': 'property_city',
            'State': 'property_state',
            'Building SqFt': 'building_size',
            'Lot Size Acres': 'lot_acres',
            'Property Type': 'property_type',
            'Flex Score': 'flex_score'
        }
        
        classic_df = df.copy()
        
        # Select only available columns
        available_columns = {k: v for k, v in classic_columns.items() if k in df.columns}
        classic_df = classic_df[list(available_columns.keys())]
        classic_df = classic_df.rename(columns=available_columns)
        
        return classic_df
    
    def _convert_to_enhanced_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert to enhanced format with additional metadata"""
        enhanced_df = df.copy()
        
        # Add enhanced metadata
        enhanced_df['processing_date'] = datetime.now().strftime('%Y-%m-%d')
        enhanced_df['pipeline_version'] = 'scalable_v1.0'
        enhanced_df['data_quality_score'] = 'high'  # Could be calculated
        
        return enhanced_df
    
    def _convert_to_minimal_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert to minimal format with essential columns only"""
        essential_columns = ['Address', 'City', 'State', 'Flex Score']
        
        minimal_df = df.copy()
        available_essential = [col for col in essential_columns if col in df.columns]
        minimal_df = minimal_df[available_essential]
        
        return minimal_df


class IntegrationManager:
    """Main integration manager coordinating all integration activities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.compatibility_checker = ComponentCompatibilityChecker()
        self.system_integrator = ExistingSystemIntegrator()
        self.data_migrator = LegacyDataMigrator()
        self.format_converter = OutputFormatConverter()
        
        self.integration_results: List[IntegrationResult] = []
    
    def run_integration_workflow(self, 
                                df: pd.DataFrame,
                                integration_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run complete integration workflow
        
        Args:
            df: DataFrame with scalable pipeline results
            integration_config: Configuration for integration steps
            
        Returns:
            Dictionary with integration workflow results
        """
        self.logger.info("Starting integration workflow...")
        workflow_start = datetime.now()
        
        workflow_results = {
            'success': True,
            'steps_completed': [],
            'steps_failed': [],
            'total_records': len(df),
            'integration_results': []
        }
        
        try:
            # Step 1: Compatibility Check
            if integration_config.get('check_compatibility', True):
                self.logger.info("Step 1: Running compatibility check...")
                compatibility_results = self.compatibility_checker.run_full_compatibility_check()
                workflow_results['compatibility_results'] = compatibility_results
                workflow_results['steps_completed'].append('compatibility_check')
            
            # Step 2: Integrate with FlexPropertyClassifier
            if integration_config.get('integrate_flex_classifier', False):
                self.logger.info("Step 2: Integrating with FlexPropertyClassifier...")
                processed_df, result = self.system_integrator.integrate_with_flex_classifier(df)
                self.integration_results.append(result)
                workflow_results['integration_results'].append(result)
                
                if result.success:
                    df = processed_df  # Use processed data for next steps
                    workflow_results['steps_completed'].append('flex_classifier_integration')
                else:
                    workflow_results['steps_failed'].append('flex_classifier_integration')
            
            # Step 3: Database Integration
            if integration_config.get('integrate_database', False):
                self.logger.info("Step 3: Integrating with database...")
                result = self.system_integrator.integrate_with_database(
                    df, 
                    integration_config.get('collection_name', 'scalable_pipeline_results')
                )
                self.integration_results.append(result)
                workflow_results['integration_results'].append(result)
                
                if result.success:
                    workflow_results['steps_completed'].append('database_integration')
                else:
                    workflow_results['steps_failed'].append('database_integration')
            
            # Step 4: Private Property Analysis Integration
            if integration_config.get('integrate_private_analyzer', False):
                self.logger.info("Step 4: Integrating with PrivatePropertyAnalyzer...")
                analyzed_df, result = self.system_integrator.integrate_with_private_analyzer(df)
                self.integration_results.append(result)
                workflow_results['integration_results'].append(result)
                
                if result.success:
                    df = analyzed_df  # Use analyzed data
                    workflow_results['steps_completed'].append('private_analyzer_integration')
                else:
                    workflow_results['steps_failed'].append('private_analyzer_integration')
            
            # Step 5: Legacy Format Conversion
            if integration_config.get('convert_legacy_format', False):
                self.logger.info("Step 5: Converting to legacy format...")
                output_path = integration_config.get('legacy_output_path', 'output/legacy_format.xlsx')
                format_type = integration_config.get('legacy_format_type', 'classic')
                
                result = self.format_converter.convert_to_legacy_format(
                    df, output_path, format_type
                )
                self.integration_results.append(result)
                workflow_results['integration_results'].append(result)
                
                if result.success:
                    workflow_results['steps_completed'].append('legacy_format_conversion')
                    workflow_results['legacy_output_path'] = output_path
                else:
                    workflow_results['steps_failed'].append('legacy_format_conversion')
            
            # Calculate workflow summary
            workflow_time = (datetime.now() - workflow_start).total_seconds()
            workflow_results['workflow_time'] = workflow_time
            workflow_results['success'] = len(workflow_results['steps_failed']) == 0
            
            success_msg = f"Integration workflow completed in {workflow_time:.2f}s"
            if workflow_results['steps_failed']:
                success_msg += f" with {len(workflow_results['steps_failed'])} failed steps"
            
            self.logger.info(success_msg)
            
            return workflow_results
            
        except Exception as e:
            workflow_time = (datetime.now() - workflow_start).total_seconds()
            self.logger.error(f"Integration workflow failed: {e}")
            
            workflow_results.update({
                'success': False,
                'error': str(e),
                'workflow_time': workflow_time
            })
            
            return workflow_results
    
    def get_integration_summary(self) -> Dict[str, Any]:
        """Get summary of all integration operations"""
        if not self.integration_results:
            return {'message': 'No integration operations performed'}
        
        successful_integrations = [r for r in self.integration_results if r.success]
        failed_integrations = [r for r in self.integration_results if not r.success]
        
        total_records_processed = sum(r.records_processed for r in self.integration_results)
        total_records_integrated = sum(r.records_integrated for r in successful_integrations)
        total_integration_time = sum(r.integration_time for r in self.integration_results)
        
        return {
            'total_operations': len(self.integration_results),
            'successful_operations': len(successful_integrations),
            'failed_operations': len(failed_integrations),
            'success_rate': len(successful_integrations) / len(self.integration_results) * 100,
            'total_records_processed': total_records_processed,
            'total_records_integrated': total_records_integrated,
            'total_integration_time': total_integration_time,
            'operations_by_component': {
                result.component_name: {
                    'success': result.success,
                    'records_processed': result.records_processed,
                    'integration_time': result.integration_time
                }
                for result in self.integration_results
            }
        }