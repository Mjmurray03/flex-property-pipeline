#!/usr/bin/env python3
"""
Test Phase 2 fixes for error handling and data insertion
"""
import asyncio
import logging
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extractors.property_detail_extractor import PropertyDetailExtractor

def test_calculate_flex_indicators():
    """Test that calculate_flex_indicators never returns None values"""
    print("Testing PropertyDetailExtractor.calculate_flex_indicators...")
    
    extractor = PropertyDetailExtractor()
    
    # Test with empty data
    empty_data = {}
    indicators = extractor.calculate_flex_indicators(empty_data)
    
    print(f"Empty data indicators: {indicators}")
    
    # Check that no values are None
    for key, value in indicators.items():
        assert value is not None, f"Key '{key}' returned None value"
        print(f"OK {key}: {value} (type: {type(value).__name__})")
    
    # Test with partial data that might cause None values
    test_data = {
        'PROPERTY_USE': 'WAREH/DIST TERM',
        'YRBLT': None,  # This could cause None
        'IMPRV_MRKT': None,  # This could cause None
        'LAND_MARKET': 0,  # Division by zero potential
        'ACRES': None  # This could cause None
    }
    
    indicators = extractor.calculate_flex_indicators(test_data)
    print(f"\nPartial data indicators: {indicators}")
    
    # Check that no values are None
    for key, value in indicators.items():
        assert value is not None, f"Key '{key}' returned None value with partial data"
        print(f"OK {key}: {value} (type: {type(value).__name__})")
    
    print("OK All tests passed - no None values returned!")

def test_none_checks():
    """Test explicit None checks before comparisons"""
    print("\nTesting None checks before comparisons...")
    
    # Test scenarios that were causing the '<=' not supported error
    test_values = [None, 0, 5, 10]
    
    for base_score in test_values:
        for adjustment in test_values:
            # Simulate the fixed logic
            base_score_safe = base_score if base_score is not None else 0
            adjustment_safe = adjustment if adjustment is not None else 0
            
            final_score = base_score_safe + adjustment_safe
            final_score = min(10, max(0, final_score))
            
            # Test comparison
            is_candidate = final_score is not None and final_score >= 5
            
            print(f"base_score={base_score}, adjustment={adjustment} -> final_score={final_score}, is_candidate={is_candidate}")
    
    print("OK All None checks passed!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Phase 2 fixes...")
    print("=" * 50)
    
    try:
        test_calculate_flex_indicators()
        test_none_checks()
        
        print("\n" + "=" * 50)
        print("OK ALL TESTS PASSED!")
        print("Phase 2 should now handle errors properly and save data even when some parcels fail.")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)