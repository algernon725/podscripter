#!/usr/bin/env python3
"""
Test to verify no deprecation warnings occur.
"""

import warnings
import sys


def test_no_deprecation_warning():
    """Test that no deprecation warnings occur when importing transformers."""
    
    print("Testing for deprecation warnings...")
    print("=" * 40)
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Import transformers (this is where the warning would occur)
        try:
            import transformers
            print("✅ Successfully imported transformers")
        except ImportError as e:
            print(f"❌ Failed to import transformers: {e}")
            return
        
        # Check for any warnings
        deprecation_warnings = [warning for warning in w if 'deprecated' in str(warning.message).lower()]
        
        if deprecation_warnings:
            print("❌ Found deprecation warnings:")
            for warning in deprecation_warnings:
                print(f"   - {warning.message}")
        else:
            print("✅ No deprecation warnings found")
        
        # Check for specific TRANSFORMERS_CACHE warning
        transformers_cache_warnings = [warning for warning in w if 'TRANSFORMERS_CACHE' in str(warning.message)]
        
        if transformers_cache_warnings:
            print("❌ Found TRANSFORMERS_CACHE deprecation warnings:")
            for warning in transformers_cache_warnings:
                print(f"   - {warning.message}")
        else:
            print("✅ No TRANSFORMERS_CACHE deprecation warnings found")
    
    print("\n" + "=" * 40)
    print("SUMMARY:")
    print("=" * 40)
    
    if not deprecation_warnings:
        print("✅ Environment variables are correctly configured")
        print("✅ No deprecation warnings should appear during normal operation")
    else:
        print("❌ Deprecation warnings found - check environment variable configuration")


if __name__ == "__main__":
    test_no_deprecation_warning()
