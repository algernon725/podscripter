#!/usr/bin/env python3
"""
Test runner for all punctuation restoration tests.
Run this script to execute all tests.
"""

import os
import sys
import subprocess
import importlib.util

def run_test_file(test_file):
    """Run a single test file and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {test_file}")
    print(f"{'='*60}")
    
    try:
        # Add the parent directory to the path so tests can import modules
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Import and run the test module
        spec = importlib.util.spec_from_file_location("test_module", test_file)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
        
        print(f"✅ {test_file} completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ {test_file} failed: {e}")
        return False

def main():
    """Run all test files in the tests directory."""
    print("Running All Punctuation Restoration Tests")
    print("=" * 60)
    
    # Get all test files
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_files = [f for f in os.listdir(test_dir) if f.startswith('test_') and f.endswith('.py')]
    test_files.sort()
    
    if not test_files:
        print("No test files found!")
        return
    
    print(f"Found {len(test_files)} test files:")
    for test_file in test_files:
        print(f"  - {test_file}")
    
    # Run each test
    successful_tests = 0
    total_tests = len(test_files)
    
    for test_file in test_files:
        test_path = os.path.join(test_dir, test_file)
        if run_test_file(test_path):
            successful_tests += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    
    if successful_tests == total_tests:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 