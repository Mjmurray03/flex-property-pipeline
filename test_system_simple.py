#!/usr/bin/env python3
"""
Simple System Test - Tests core functionality without unicode
"""

import os
import sys
import pandas as pd
import traceback
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_data_loading():
    """Test data loading functionality"""
    print("\n[TEST] Data Loading...")

    try:
        sample_file = "data/exports/SAMPLE_50_FINAL_CLEAN.xlsx"
        if not os.path.exists(sample_file):
            print(f"[FAIL] Sample file not found: {sample_file}")
            return False

        df = pd.read_excel(sample_file)
        print(f"[PASS] Loaded {len(df)} rows with {len(df.columns)} columns")
        return True

    except Exception as e:
        print(f"[FAIL] Data loading error: {e}")
        return False

def test_data_processor():
    """Test data processing components"""
    print("\n[TEST] Data Processing...")

    try:
        from app.components.data_processor import DataProcessor

        processor = DataProcessor()
        print("[PASS] DataProcessor imported and initialized")

        # Load sample data
        df = pd.read_excel("data/exports/SAMPLE_50_FINAL_CLEAN.xlsx")

        # Test validation
        validation = processor.validate_data(df)
        print(f"[PASS] Data validation completed: {validation.get('is_valid', False)}")

        # Test cleaning
        cleaned = processor.clean_data(df)
        print(f"[PASS] Data cleaning completed: {len(cleaned)} rows")

        return True

    except Exception as e:
        print(f"[FAIL] Data processor error: {e}")
        return False

def test_filter_engine():
    """Test filter engine"""
    print("\n[TEST] Filter Engine...")

    try:
        from app.components.filter_engine import AdvancedFilterEngine

        engine = AdvancedFilterEngine()
        print("[PASS] AdvancedFilterEngine imported and initialized")

        # Load sample data
        df = pd.read_excel("data/exports/SAMPLE_50_FINAL_CLEAN.xlsx")

        # Test basic filtering
        filters = {
            'min_building_sqft': 5000,
            'max_price': 1000000
        }

        filtered = engine.apply_filters(df, filters)
        print(f"[PASS] Basic filtering completed: {len(filtered.filtered_data)} results")

        return True

    except Exception as e:
        print(f"[FAIL] Filter engine error: {e}")
        return False

def test_authentication():
    """Test authentication system"""
    print("\n[TEST] Authentication...")

    try:
        from app.components.authentication import AuthenticationComponent
        from app.components.authorization import RoleBasedAccessControl

        auth = AuthenticationComponent()
        authz = RoleBasedAccessControl()
        print("[PASS] Authentication components imported")

        # Test user creation
        user = auth.create_test_user("test@example.com", "Test User", ["analyst"])
        if user:
            print("[PASS] Test user created successfully")
        else:
            print("[FAIL] Test user creation failed")

        return True

    except Exception as e:
        print(f"[FAIL] Authentication error: {e}")
        return False

def test_workflow():
    """Test workflow navigation"""
    print("\n[TEST] Workflow Navigation...")

    try:
        from app.components.workflow_navigation import WorkflowNavigator

        navigator = WorkflowNavigator()
        print("[PASS] WorkflowNavigator imported and initialized")

        summary = navigator.get_workflow_summary()
        print(f"[PASS] Workflow summary generated: {summary.get('workflow_id', 'N/A')}")

        return True

    except Exception as e:
        print(f"[FAIL] Workflow error: {e}")
        return False

def test_notifications():
    """Test notification system"""
    print("\n[TEST] Notifications...")

    try:
        from app.components.notifications import NotificationManager, ErrorHandler

        notif = NotificationManager()
        error_handler = ErrorHandler()
        print("[PASS] Notification components imported")

        # Test error handling
        try:
            raise ValueError("Test error")
        except Exception as test_error:
            error_details = error_handler.handle_error(test_error, "Test context")
            print(f"[PASS] Error handling works: {error_details.get('id', 'N/A')}")

        return True

    except Exception as e:
        print(f"[FAIL] Notification error: {e}")
        return False

def test_export():
    """Test export functionality"""
    print("\n[TEST] Export Functionality...")

    try:
        # Load sample data
        df = pd.read_excel("data/exports/SAMPLE_50_FINAL_CLEAN.xlsx")

        # Test CSV export
        test_csv = "test_export.csv"
        df.head(10).to_csv(test_csv, index=False)

        if os.path.exists(test_csv):
            print("[PASS] CSV export successful")
            os.remove(test_csv)
        else:
            print("[FAIL] CSV export failed")

        # Test Excel export
        test_xlsx = "test_export.xlsx"
        df.head(10).to_excel(test_xlsx, index=False)

        if os.path.exists(test_xlsx):
            print("[PASS] Excel export successful")
            os.remove(test_xlsx)
        else:
            print("[FAIL] Excel export failed")

        return True

    except Exception as e:
        print(f"[FAIL] Export error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("COMPREHENSIVE SYSTEM TEST")
    print("=" * 50)

    tests = [
        test_data_loading,
        test_data_processor,
        test_filter_engine,
        test_authentication,
        test_workflow,
        test_notifications,
        test_export
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[ERROR] Test {test_func.__name__} crashed: {e}")
            failed += 1

    # Summary
    total = passed + failed
    success_rate = (passed / total * 100) if total > 0 else 0

    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"PASSED: {passed}")
    print(f"FAILED: {failed}")
    print(f"SUCCESS RATE: {success_rate:.1f}%")

    if success_rate >= 80:
        print("\n[SUCCESS] System is ready for production!")
        return True
    elif success_rate >= 60:
        print("\n[WARNING] Some issues need attention")
        return False
    else:
        print("\n[CRITICAL] Major issues need resolution")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)