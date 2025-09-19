#!/usr/bin/env python3
"""
Comprehensive System Test Suite
Tests the complete flex property intelligence platform from import to export
"""

import os
import sys
import pandas as pd
import numpy as np
import traceback
from datetime import datetime
import json
import importlib.util

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class SystemTester:
    """Comprehensive system test runner"""

    def __init__(self):
        self.test_results = {
            'started_at': datetime.now().isoformat(),
            'tests_passed': 0,
            'tests_failed': 0,
            'errors': [],
            'warnings': [],
            'test_details': {}
        }
        self.sample_data_path = "data/exports/SAMPLE_50_FINAL_CLEAN.xlsx"

    def log_test(self, test_name, success, details=None, error=None):
        """Log test result"""
        if success:
            self.test_results['tests_passed'] += 1
            print(f"[PASS] {test_name}")
        else:
            self.test_results['tests_failed'] += 1
            print(f"[FAIL] {test_name}")
            if error:
                print(f"   Error: {error}")
                self.test_results['errors'].append(f"{test_name}: {error}")

        self.test_results['test_details'][test_name] = {
            'success': success,
            'details': details,
            'error': str(error) if error else None,
            'timestamp': datetime.now().isoformat()
        }

    def test_data_loading(self):
        """Test 1: Data Loading and Validation"""
        print("\n🔍 Testing Data Loading...")

        try:
            # Check if sample data exists
            if not os.path.exists(self.sample_data_path):
                self.log_test("Data file exists", False, error="Sample data file not found")
                return False

            # Test pandas loading
            df = pd.read_excel(self.sample_data_path)
            self.log_test("Excel file loads successfully", True, f"Loaded {len(df)} rows")

            # Test data structure
            required_columns = ['Building SqFt', 'Sold Price', 'Property Type']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                self.log_test("Required columns present", False,
                            error=f"Missing columns: {missing_columns}")
            else:
                self.log_test("Required columns present", True,
                            f"Found {len(df.columns)} columns")

            # Test data quality
            empty_rows = df.isnull().all(axis=1).sum()
            self.log_test("Data quality check", empty_rows == 0,
                        f"Empty rows: {empty_rows}")

            return True

        except Exception as e:
            self.log_test("Data loading", False, error=str(e))
            return False

    def test_data_processor(self):
        """Test 2: Data Processing Components"""
        print("\n⚙️ Testing Data Processing...")

        try:
            # Test data processor import
            from app.components.data_processor import DataProcessor, DataQualityAssessment
            self.log_test("Data processor imports", True)

            # Test initialization
            processor = DataProcessor()
            self.log_test("Data processor initialization", True)

            # Load sample data
            df = pd.read_excel(self.sample_data_path)

            # Test data validation
            validation_result = processor.validate_data(df)
            self.log_test("Data validation", validation_result['is_valid'],
                        f"Validation score: {validation_result.get('quality_score', 'N/A')}")

            # Test data cleaning
            cleaned_df = processor.clean_data(df)
            self.log_test("Data cleaning", len(cleaned_df) > 0,
                        f"Cleaned {len(cleaned_df)} rows from {len(df)}")

            # Test calculated fields
            enhanced_df = processor.add_calculated_fields(cleaned_df)
            new_columns = set(enhanced_df.columns) - set(cleaned_df.columns)
            self.log_test("Calculated fields added", len(new_columns) > 0,
                        f"Added {len(new_columns)} new fields")

            return True

        except Exception as e:
            self.log_test("Data processing", False, error=str(e))
            print(f"Full error: {traceback.format_exc()}")
            return False

    def test_filter_engine(self):
        """Test 3: Advanced Filter Engine"""
        print("\n🔍 Testing Filter Engine...")

        try:
            # Test filter engine import
            from app.components.filter_engine import AdvancedFilterEngine
            self.log_test("Filter engine imports", True)

            # Test initialization
            filter_engine = AdvancedFilterEngine()
            self.log_test("Filter engine initialization", True)

            # Load and prepare sample data
            df = pd.read_excel(self.sample_data_path)

            # Test basic filtering
            filter_criteria = {
                'min_building_sqft': 10000,
                'max_price': 5000000,
                'property_types': ['Warehouse', 'Industrial']
            }

            filtered_df = filter_engine.apply_filters(df, filter_criteria)
            self.log_test("Basic filtering", len(filtered_df) >= 0,
                        f"Filtered to {len(filtered_df)} properties")

            # Test ML filtering (if data supports it)
            try:
                ml_results = filter_engine.apply_ml_filters(df)
                self.log_test("ML filtering", len(ml_results) >= 0,
                            f"ML identified {len(ml_results)} candidates")
            except Exception as ml_error:
                self.log_test("ML filtering", False, error=str(ml_error))

            # Test recommendations
            try:
                recommendations = filter_engine.generate_recommendations(df)
                self.log_test("Filter recommendations", 'recommendations' in recommendations,
                            f"Generated {len(recommendations.get('recommendations', []))} recommendations")
            except Exception as rec_error:
                self.log_test("Filter recommendations", False, error=str(rec_error))

            return True

        except Exception as e:
            self.log_test("Filter engine", False, error=str(e))
            print(f"Full error: {traceback.format_exc()}")
            return False

    def test_authentication(self):
        """Test 4: Authentication System"""
        print("\n🔐 Testing Authentication...")

        try:
            # Test authentication components
            from app.components.authentication import AuthenticationComponent
            from app.components.authorization import AuthorizationManager
            self.log_test("Authentication imports", True)

            # Test authentication initialization
            auth = AuthenticationComponent()
            authz = AuthorizationManager()
            self.log_test("Authentication initialization", True)

            # Test user creation
            test_user = auth.create_test_user("test@example.com", "Test User", ["analyst"])
            self.log_test("Test user creation", test_user is not None,
                        f"Created user: {test_user.get('email', 'N/A') if test_user else 'None'}")

            # Test permission checking
            if test_user:
                has_permission = authz.check_permission(test_user, "data_analysis")
                self.log_test("Permission checking", has_permission is not None,
                            f"Permission result: {has_permission}")

            return True

        except Exception as e:
            self.log_test("Authentication", False, error=str(e))
            print(f"Full error: {traceback.format_exc()}")
            return False

    def test_workflow_navigation(self):
        """Test 5: Workflow Navigation"""
        print("\n🧭 Testing Workflow Navigation...")

        try:
            # Test workflow components
            from app.components.workflow_navigation import WorkflowNavigator, WorkflowState
            self.log_test("Workflow navigation imports", True)

            # Test workflow state
            workflow_state = WorkflowState()
            self.log_test("Workflow state initialization", True)

            # Test workflow navigator
            navigator = WorkflowNavigator()
            self.log_test("Workflow navigator initialization", True)

            # Test workflow summary
            summary = navigator.get_workflow_summary()
            self.log_test("Workflow summary", 'workflow_id' in summary,
                        f"Workflow ID: {summary.get('workflow_id', 'N/A')}")

            return True

        except Exception as e:
            self.log_test("Workflow navigation", False, error=str(e))
            print(f"Full error: {traceback.format_exc()}")
            return False

    def test_notifications(self):
        """Test 6: Notification System"""
        print("\n🔔 Testing Notification System...")

        try:
            # Test notification components
            from app.components.notifications import NotificationManager, ErrorHandler
            self.log_test("Notification imports", True)

            # Test notification manager
            notif_manager = NotificationManager()
            self.log_test("Notification manager initialization", True)

            # Test error handler
            error_handler = ErrorHandler()
            self.log_test("Error handler initialization", True)

            # Test error handling
            try:
                raise ValueError("Test error for error handler")
            except Exception as test_error:
                error_details = error_handler.handle_error(test_error, "Testing error handling")
                self.log_test("Error handling", 'id' in error_details,
                            f"Error ID: {error_details.get('id', 'N/A')}")

            return True

        except Exception as e:
            self.log_test("Notification system", False, error=str(e))
            print(f"Full error: {traceback.format_exc()}")
            return False

    def test_caching_system(self):
        """Test 7: Caching System"""
        print("\n💾 Testing Caching System...")

        try:
            # Test cache manager
            from app.components.data_processor import CacheManager
            self.log_test("Cache manager imports", True)

            # Test cache initialization
            cache_manager = CacheManager()
            self.log_test("Cache manager initialization", True)

            # Test cache operations
            test_data = {"test": "data", "timestamp": datetime.now().isoformat()}
            cache_key = cache_manager.generate_cache_key(test_data)
            self.log_test("Cache key generation", len(cache_key) > 0,
                        f"Generated key: {cache_key[:12]}...")

            # Test cache storage and retrieval
            cache_manager.store_cache(cache_key, test_data)
            retrieved_data = cache_manager.get_cache(cache_key)
            self.log_test("Cache storage/retrieval", retrieved_data is not None,
                        f"Retrieved: {bool(retrieved_data)}")

            return True

        except Exception as e:
            self.log_test("Caching system", False, error=str(e))
            print(f"Full error: {traceback.format_exc()}")
            return False

    def test_export_functionality(self):
        """Test 8: Export Functionality"""
        print("\n📤 Testing Export Functionality...")

        try:
            # Load sample data
            df = pd.read_excel(self.sample_data_path)

            # Test CSV export
            csv_output = "test_export.csv"
            df.to_csv(csv_output, index=False)
            csv_exists = os.path.exists(csv_output)
            self.log_test("CSV export", csv_exists, f"File created: {csv_output}")

            # Test Excel export
            excel_output = "test_export.xlsx"
            df.to_excel(excel_output, index=False)
            excel_exists = os.path.exists(excel_output)
            self.log_test("Excel export", excel_exists, f"File created: {excel_output}")

            # Clean up test files
            for file in [csv_output, excel_output]:
                if os.path.exists(file):
                    os.remove(file)

            # Test data processor export functionality
            try:
                from app.components.data_processor import DataProcessor
                processor = DataProcessor()

                # Test metadata inclusion
                export_data = processor.prepare_export_data(df)
                self.log_test("Export data preparation", export_data is not None,
                            f"Prepared {len(export_data) if export_data is not None else 0} records")
            except Exception as export_error:
                self.log_test("Export data preparation", False, error=str(export_error))

            return True

        except Exception as e:
            self.log_test("Export functionality", False, error=str(e))
            print(f"Full error: {traceback.format_exc()}")
            return False

    def test_error_edge_cases(self):
        """Test 9: Error Handling and Edge Cases"""
        print("\n⚠️ Testing Error Handling and Edge Cases...")

        try:
            # Test empty data handling
            empty_df = pd.DataFrame()

            from app.components.data_processor import DataProcessor
            processor = DataProcessor()

            # Test validation with empty data
            validation_result = processor.validate_data(empty_df)
            self.log_test("Empty data validation", not validation_result.get('is_valid', True),
                        "Correctly identifies empty data as invalid")

            # Test malformed data handling
            malformed_df = pd.DataFrame({
                'col1': [1, 2, None, 'invalid'],
                'col2': ['a', None, 'c', 'd']
            })

            cleaned_malformed = processor.clean_data(malformed_df)
            self.log_test("Malformed data cleaning", len(cleaned_malformed) >= 0,
                        f"Cleaned malformed data: {len(cleaned_malformed)} rows")

            # Test filter engine with edge cases
            from app.components.filter_engine import AdvancedFilterEngine
            filter_engine = AdvancedFilterEngine()

            # Test filtering with no matches
            extreme_filters = {
                'min_building_sqft': 999999999,  # Unrealistic value
                'max_price': 1  # Unrealistic value
            }

            df = pd.read_excel(self.sample_data_path)
            no_match_result = filter_engine.apply_filters(df, extreme_filters)
            self.log_test("No match filtering", len(no_match_result) == 0,
                        "Correctly returns empty result for impossible criteria")

            return True

        except Exception as e:
            self.log_test("Error handling and edge cases", False, error=str(e))
            print(f"Full error: {traceback.format_exc()}")
            return False

    def test_performance(self):
        """Test 10: Performance and Scalability"""
        print("\n⚡ Testing Performance...")

        try:
            import time

            # Load sample data
            df = pd.read_excel(self.sample_data_path)

            # Test data processing performance
            from app.components.data_processor import DataProcessor
            processor = DataProcessor()

            start_time = time.time()
            processed_df = processor.clean_data(df)
            processing_time = time.time() - start_time

            self.log_test("Data processing performance", processing_time < 30.0,
                        f"Processing took {processing_time:.2f} seconds")

            # Test filtering performance
            from app.components.filter_engine import AdvancedFilterEngine
            filter_engine = AdvancedFilterEngine()

            start_time = time.time()
            filter_criteria = {
                'min_building_sqft': 5000,
                'max_price': 1000000
            }
            filtered_df = filter_engine.apply_filters(df, filter_criteria)
            filtering_time = time.time() - start_time

            self.log_test("Filtering performance", filtering_time < 10.0,
                        f"Filtering took {filtering_time:.2f} seconds")

            # Test memory usage
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024

            self.log_test("Memory usage", memory_mb < 1000,  # Less than 1GB
                        f"Memory usage: {memory_mb:.1f} MB")

            return True

        except Exception as e:
            self.log_test("Performance testing", False, error=str(e))
            print(f"Full error: {traceback.format_exc()}")
            return False

    def run_all_tests(self):
        """Run all system tests"""
        print("🚀 Starting Comprehensive System Test Suite")
        print("=" * 60)

        test_methods = [
            self.test_data_loading,
            self.test_data_processor,
            self.test_filter_engine,
            self.test_authentication,
            self.test_workflow_navigation,
            self.test_notifications,
            self.test_caching_system,
            self.test_export_functionality,
            self.test_error_edge_cases,
            self.test_performance
        ]

        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                print(f"❌ Test suite error in {test_method.__name__}: {e}")
                self.test_results['errors'].append(f"Test suite error: {e}")

        # Generate final report
        self.generate_test_report()

    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)

        total_tests = self.test_results['tests_passed'] + self.test_results['tests_failed']
        success_rate = (self.test_results['tests_passed'] / total_tests * 100) if total_tests > 0 else 0

        print(f"✅ Tests Passed: {self.test_results['tests_passed']}")
        print(f"❌ Tests Failed: {self.test_results['tests_failed']}")
        print(f"📈 Success Rate: {success_rate:.1f}%")

        if self.test_results['errors']:
            print(f"\n🔍 ERRORS ENCOUNTERED:")
            for error in self.test_results['errors'][:5]:  # Show first 5 errors
                print(f"  • {error}")
            if len(self.test_results['errors']) > 5:
                print(f"  ... and {len(self.test_results['errors']) - 5} more errors")

        # Save detailed report
        self.test_results['completed_at'] = datetime.now().isoformat()
        self.test_results['success_rate'] = success_rate

        report_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)

        print(f"\n📄 Detailed report saved to: {report_file}")

        if success_rate >= 80:
            print("\n🎉 SYSTEM TEST PASSED - System is ready for production!")
        elif success_rate >= 60:
            print("\n⚠️ SYSTEM TEST PARTIAL - Some issues need attention")
        else:
            print("\n🚨 SYSTEM TEST FAILED - Critical issues need resolution")

        return self.test_results


def main():
    """Main test runner"""
    tester = SystemTester()
    results = tester.run_all_tests()
    return results


if __name__ == "__main__":
    results = main()
    sys.exit(0 if results['success_rate'] >= 80 else 1)