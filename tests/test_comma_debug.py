#!/usr/bin/env python3
"""
Debug test to trace comma spacing issue.
"""

import sys
import os
import re

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_regex_patterns():
    """Test the regex patterns directly."""
    print("\n" + "="*60)
    print("TESTING REGEX PATTERNS DIRECTLY")
    print("="*60 + "\n")
    
    # Test input
    text = "episodio 147,151,156,164,170,177 y 184"
    print(f"Input:  '{text}'")
    
    # Step 1: Add space after ALL commas
    text = re.sub(r",\s*", ", ", text)
    print(f"After add spaces:  '{text}'")
    
    # Step 2: Remove spaces from 2-group thousands (with lookahead/lookbehind, NO comma in lookahead)
    text = re.sub(r"(?<=\s)(\d{1,3}),\s+(\d{3}),\s+(\d{3})(?=\s|[.!?]|$)", r"\1,\2,\3", text)
    print(f"After remove 2-group thousands:  '{text}'")
    
    # Step 3: Remove spaces from 1-group thousands (with lookahead/lookbehind, NO comma in lookahead)
    text = re.sub(r"(?<=\s)(\d{1,3}),\s+(\d{3})(?=\s|[.!?]|$)", r"\1,\2", text)
    print(f"After remove 1-group thousands:  '{text}'")
    
    print()
    
    # Test thousands
    text2 = "hay 1,000 personas y 25,000 dólares"
    print(f"\nThousands input:  '{text2}'")
    text2 = re.sub(r",\s*", ", ", text2)
    print(f"After add spaces:  '{text2}'")
    text2 = re.sub(r"(?<=\s)(\d{1,3}),\s+(\d{3}),\s+(\d{3})(?=\s|[.!?]|$)", r"\1,\2,\3", text2)
    print(f"After remove 2-group thousands:  '{text2}'")
    text2 = re.sub(r"(?<=\s)(\d{1,3}),\s+(\d{3})(?=\s|[.!?]|$)", r"\1,\2", text2)
    print(f"After remove 1-group thousands:  '{text2}'")
    
    print()


if __name__ == "__main__":
    test_regex_patterns()

