#!/usr/bin/env python3
"""
Test runner for punctuation restoration tests.

Defaults:
  - Runs primary language tests (English, Spanish, French, German) and core/env checks
  - Skips multilingual aggregates and debug/benchmark tests unless enabled via env flags

Env flags:
  RUN_MULTILINGUAL=1     include test_multilingual_*.py
  RUN_TRANSCRIPTION=1    include transcription integration tests
  RUN_DEBUG=1            include debug/benchmark/experimental tests
  RUN_ALL=1              include everything
"""

import os
import sys
import subprocess
import importlib.util
from typing import List, Set

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
    print("Running Punctuation Restoration Tests")
    print("=" * 60)

    run_all = os.getenv("RUN_ALL") == "1"
    run_multilingual = run_all or os.getenv("RUN_MULTILINGUAL") == "1"
    run_transcription = run_all or os.getenv("RUN_TRANSCRIPTION") == "1"
    run_debug = run_all or os.getenv("RUN_DEBUG") == "1"

    test_dir = os.path.dirname(os.path.abspath(__file__))
    all_files: List[str] = [f for f in os.listdir(test_dir) if f.startswith('test_') and f.endswith('.py')]

    # Buckets
    primary_prefixes = (
        'test_spanish_', 'test_english_', 'test_french_', 'test_german_'
    )
    core_files: Set[str] = set(f for f in all_files if f.startswith(primary_prefixes))

    # Core generic tests always included
    core_files |= set(x for x in all_files if x in {
        'test_improved_punctuation.py',
        'test_punctuation.py',
        'test_environment_variables.py',
        'test_no_deprecation_warning.py',
        'test_past_tense_questions.py',
        'test_punctuation_preservation.py',
        'test_human_vs_program_intro.py',
        'test_spanish_embedded_questions.py',
        'test_sentence_assembly_unit.py',
        'test_chunk_merge_helpers.py',
    })

    multilingual_files = sorted(x for x in all_files if x.startswith('test_multilingual_'))

    transcription_files = sorted(x for x in all_files if x in {
        'test_transcription.py', 'test_transcription_logic.py'
    })

    # Debug/benchmark/experimental
    debug_files = sorted(x for x in all_files if x in {
        'test_question_detection_debug.py', 'test_transcription_debug.py', 'test_transcription_specific.py',
        'model_comparison.py', 'test_model_change.py'
    })

    selected: List[str] = sorted(core_files)
    if run_multilingual:
        selected += multilingual_files
    if run_transcription:
        selected += transcription_files
    if run_debug:
        selected += debug_files

    if not selected:
        print("No tests selected. Set RUN_ALL=1 to run everything.")
        sys.exit(1)

    print("Selected test files:")
    for f in selected:
        print(f"  - {f}")

    # Run tests
    successful_tests = 0
    for test_file in selected:
        test_path = os.path.join(test_dir, test_file)
        if run_test_file(test_path):
            successful_tests += 1

    # Summary
    total_tests = len(selected)
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")

    if successful_tests == total_tests:
        print("🎉 All selected tests passed!")
    else:
        print("⚠️ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 