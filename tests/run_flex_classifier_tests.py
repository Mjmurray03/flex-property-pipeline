"""
Test runner for Flex Property Classifier
Runs comprehensive test suite with detailed reporting
"""

import unittest
import sys
import time
import logging
from pathlib import Path
from io import StringIO

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from test_flex_property_classifier import TestFlexPropertyClassifier, TestFlexPropertyClassifierIntegration
from test_data_generator import FlexTestDataGenerator


class FlexClassifierTestRunner:
    """Custom test runner with detailed reporting"""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def run_all_tests(self, verbosity=2):
        """
        Run all test suites with comprehensive reporting
        
        Args:
            verbosity: Test output verbosity level
        """
        print("=" * 80)
        print("FLEX PROPERTY CLASSIFIER - COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        
        self.start_time = time.time()
        
        # Create test suite
        test_suite = unittest.TestSuite()
        
        # Add all test classes
        test_classes = [
            TestFlexPropertyClassifier,
            TestFlexPropertyClassifierIntegration
        ]
        
        for test_class in test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            test_suite.addTests(tests)
        
        # Run tests with custom result collector
        stream = StringIO()
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=verbosity,
            buffer=True
        )
        
        print(f"Running {test_suite.countTestCases()} tests...")
        print("-" * 80)
        
        result = runner.run(test_suite)
        
        self.end_time = time.time()
        
        # Process and display results
        self._process_results(result, stream.getvalue())
        
        return result.wasSuccessful()
    
    def _process_results(self, result, output):
        """Process and display test results"""
        total_time = self.end_time - self.start_time
        
        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)
        
        # Basic statistics
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
        print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per test: {total_time / result.testsRun:.3f} seconds")
        
        # Detailed failure/error reporting
        if result.failures:
            print("\n" + "-" * 40)
            print("FAILURES:")
            print("-" * 40)
            for test, traceback in result.failures:
                print(f"\nFAILED: {test}")
                print(traceback)
        
        if result.errors:
            print("\n" + "-" * 40)
            print("ERRORS:")
            print("-" * 40)
            for test, traceback in result.errors:
                print(f"\nERROR: {test}")
                print(traceback)
        
        # Success message
        if result.wasSuccessful():
            print("\n" + "=" * 80)
            print("ALL TESTS PASSED SUCCESSFULLY!")
            print("The Flex Property Classifier is ready for production use.")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("SOME TESTS FAILED!")
            print("Please review the failures above and fix issues before deployment.")
            print("=" * 80)
    
    def run_performance_tests(self):
        """Run specific performance tests"""
        print("\n" + "=" * 80)
        print("PERFORMANCE TEST SUITE")
        print("=" * 80)
        
        # Create large dataset for performance testing
        generator = FlexTestDataGenerator()
        large_data = generator.create_large_dataset(5000)
        
        print(f"Testing with {len(large_data)} properties...")
        
        # Import here to avoid circular imports
        from processors.flex_property_classifier import FlexPropertyClassifier
        from utils.logger import setup_logging
        
        logger = setup_logging(name='performance_test', level='WARNING')  # Reduce log noise
        
        # Test classification performance
        start_time = time.time()
        classifier = FlexPropertyClassifier(large_data, logger)
        init_time = time.time() - start_time
        
        start_time = time.time()
        candidates = classifier.classify_flex_properties()
        classification_time = time.time() - start_time
        
        start_time = time.time()
        if len(candidates) > 0:
            top_candidates = classifier.get_top_candidates(100)
            scoring_time = time.time() - start_time
        else:
            scoring_time = 0
            top_candidates = candidates
        
        # Report performance results
        print(f"Initialization time: {init_time:.3f} seconds")
        print(f"Classification time: {classification_time:.3f} seconds")
        print(f"Scoring time: {scoring_time:.3f} seconds")
        print(f"Properties processed: {len(large_data):,}")
        print(f"Candidates found: {len(candidates):,}")
        print(f"Processing rate: {len(large_data) / classification_time:.0f} properties/second")
        
        # Performance thresholds
        performance_ok = True
        if classification_time > 30:  # Should process 5k properties in under 30 seconds
            print("WARNING: Classification performance is slower than expected")
            performance_ok = False
        
        if scoring_time > 10:  # Should score candidates in under 10 seconds
            print("WARNING: Scoring performance is slower than expected")
            performance_ok = False
        
        if performance_ok:
            print("PERFORMANCE: All performance tests passed!")
        
        return performance_ok
    
    def run_memory_tests(self):
        """Run memory usage tests"""
        print("\n" + "=" * 80)
        print("MEMORY USAGE TEST SUITE")
        print("=" * 80)
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            print(f"Initial memory usage: {initial_memory:.1f} MB")
            
            # Create and process large dataset
            generator = FlexTestDataGenerator()
            large_data = generator.create_large_dataset(10000)
            
            from processors.flex_property_classifier import FlexPropertyClassifier
            from utils.logger import setup_logging
            
            logger = setup_logging(name='memory_test', level='ERROR')  # Minimal logging
            
            classifier = FlexPropertyClassifier(large_data, logger)
            candidates = classifier.classify_flex_properties()
            
            if len(candidates) > 0:
                top_candidates = classifier.get_top_candidates(1000)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            print(f"Final memory usage: {final_memory:.1f} MB")
            print(f"Memory increase: {memory_increase:.1f} MB")
            print(f"Memory per property: {memory_increase / len(large_data) * 1024:.1f} KB")
            
            # Memory threshold check
            memory_ok = memory_increase < 200  # Should use less than 200MB for 10k properties
            
            if memory_ok:
                print("MEMORY: Memory usage is within acceptable limits!")
            else:
                print("WARNING: Memory usage is higher than expected")
            
            return memory_ok
            
        except ImportError:
            print("psutil not available - skipping memory tests")
            return True


def main():
    """Main test execution function"""
    runner = FlexClassifierTestRunner()
    
    # Run main test suite
    success = runner.run_all_tests(verbosity=2)
    
    # Run performance tests
    performance_ok = runner.run_performance_tests()
    
    # Run memory tests
    memory_ok = runner.run_memory_tests()
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL TEST SUMMARY")
    print("=" * 80)
    print(f"Unit/Integration Tests: {'PASS' if success else 'FAIL'}")
    print(f"Performance Tests: {'PASS' if performance_ok else 'FAIL'}")
    print(f"Memory Tests: {'PASS' if memory_ok else 'FAIL'}")
    
    overall_success = success and performance_ok and memory_ok
    print(f"Overall Result: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'}")
    
    if overall_success:
        print("\nThe Flex Property Classifier is ready for production deployment!")
    else:
        print("\nPlease address the failing tests before deployment.")
    
    return 0 if overall_success else 1


if __name__ == '__main__':
    sys.exit(main())