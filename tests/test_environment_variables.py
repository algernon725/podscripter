#!/usr/bin/env python3
"""
Test to verify environment variables are set correctly.
"""

import os
import sys


def test_environment_variables():
    """Test that environment variables are set correctly."""
    
    print("Testing environment variables...")
    print("=" * 40)
    
    # Check for deprecated variables
    deprecated_vars = ['TRANSFORMERS_CACHE', 'HF_DATASETS_CACHE']
    current_vars = ['HF_HOME', 'WHISPER_CACHE_DIR']
    
    print("Checking for deprecated environment variables:")
    for var in deprecated_vars:
        if var in os.environ:
            print(f"  ❌ {var} is set (should be removed)")
        else:
            print(f"  ✅ {var} is not set (good)")
    
    print("\nChecking for current environment variables:")
    for var in current_vars:
        if var in os.environ:
            print(f"  ✅ {var} is set: {os.environ[var]}")
        else:
            print(f"  ⚠️  {var} is not set")
    
    print("\n" + "=" * 40)
    print("RECOMMENDATIONS:")
    print("=" * 40)
    
    if any(var in os.environ for var in deprecated_vars):
        print("❌ Remove deprecated environment variables from Dockerfile:")
        for var in deprecated_vars:
            if var in os.environ:
                print(f"   - {var}")
    else:
        print("✅ No deprecated environment variables found")
    
    if 'HF_HOME' in os.environ:
        print("✅ HF_HOME is set correctly")
    else:
        print("⚠️  Consider setting HF_HOME for HuggingFace caching")
    
    if 'WHISPER_CACHE_DIR' in os.environ:
        print("✅ WHISPER_CACHE_DIR is set correctly")
    else:
        print("⚠️  Consider setting WHISPER_CACHE_DIR for Whisper caching")


if __name__ == "__main__":
    test_environment_variables()
