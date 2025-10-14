#!/usr/bin/env python3
"""
Unit tests for the centralized _normalize_comma_spacing() function.

This test ensures the refactored comma spacing logic works correctly
and consistently across all call sites.
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from punctuation_restorer import _normalize_comma_spacing


def test_basic_comma_spacing():
    """Test basic comma spacing normalization."""
    # Add space after commas
    assert _normalize_comma_spacing("a,b,c") == "a, b, c"
    assert _normalize_comma_spacing("uno,dos,tres") == "uno, dos, tres"
    
    # Already has spaces
    assert _normalize_comma_spacing("a, b, c") == "a, b, c"
    
    print("✓ Basic comma spacing test passed")


def test_remove_spaces_before_commas():
    """Test that spaces before commas are removed."""
    result = _normalize_comma_spacing("palabra ,otra")
    assert result == "palabra, otra", f"Expected 'palabra, otra' but got '{result}'"
    
    result = _normalize_comma_spacing("test  ,  value")
    assert result == "test, value", f"Expected 'test, value' but got '{result}'"
    
    result = _normalize_comma_spacing("a , b , c")
    assert result == "a, b, c", f"Expected 'a, b, c' but got '{result}'"
    
    print("✓ Remove spaces before commas test passed")


def test_deduplicate_commas():
    """Test that multiple commas are deduplicated."""
    assert _normalize_comma_spacing("test,,doble") == "test, doble"
    assert _normalize_comma_spacing("test, ,doble") == "test, doble"
    assert _normalize_comma_spacing("test,,,triple") == "test, triple"
    assert _normalize_comma_spacing("a, , , ,b") == "a, b"
    
    print("✓ Deduplicate commas test passed")


def test_number_lists():
    """Test that number lists get proper spacing."""
    # Episode numbers (the original bug)
    assert _normalize_comma_spacing("episodio 147,151,156") == "episodio 147, 151, 156"
    assert _normalize_comma_spacing("147,151,156,164,170,177") == "147, 151, 156, 164, 170, 177"
    
    # Various digit counts
    assert _normalize_comma_spacing("1,2,3,4,5") == "1, 2, 3, 4, 5"
    assert _normalize_comma_spacing("10,20,30,40") == "10, 20, 30, 40"
    assert _normalize_comma_spacing("100,200,300") == "100, 200, 300"
    
    print("✓ Number lists test passed")


def test_thousands_get_spaces():
    """Test that thousands separators also get spaces (documented trade-off)."""
    # Thousands now get spaces (acceptable trade-off)
    assert _normalize_comma_spacing("1,000") == "1, 000"
    assert _normalize_comma_spacing("25,000") == "25, 000"
    assert _normalize_comma_spacing("1,000,000") == "1, 000, 000"
    
    print("✓ Thousands separator test passed (with documented trade-off)")


def test_mixed_content():
    """Test text with mixed commas (lists, phrases, etc)."""
    text = "episodios 147,151,156 con María,Juan,Pedro y otros"
    result = _normalize_comma_spacing(text)
    assert result == "episodios 147, 151, 156 con María, Juan, Pedro y otros"
    
    text = "en el año 2,000 había más de 1,000 personas"
    result = _normalize_comma_spacing(text)
    assert result == "en el año 2, 000 había más de 1, 000 personas"
    
    print("✓ Mixed content test passed")


def test_preserves_already_correct():
    """Test that already correctly spaced text is preserved."""
    text = "This is correct, with proper spacing, throughout the sentence."
    assert _normalize_comma_spacing(text) == text
    
    text = "uno, dos, tres, cuatro"
    assert _normalize_comma_spacing(text) == text
    
    print("✓ Preserves already correct test passed")


def test_empty_and_edge_cases():
    """Test edge cases like empty strings."""
    assert _normalize_comma_spacing("") == ""
    assert _normalize_comma_spacing(None) == ""  # None -> empty string for safety
    assert _normalize_comma_spacing(",") == ","  # Lone comma doesn't get trailing space
    assert _normalize_comma_spacing(",,") == ", "  # Double comma becomes single with space
    assert _normalize_comma_spacing(" , ") == ", "  # Spaces normalized
    
    print("✓ Empty and edge cases test passed")


def run_all_tests():
    """Run all comma spacing normalization tests."""
    print("\n" + "="*60)
    print("TESTING CENTRALIZED _normalize_comma_spacing() FUNCTION")
    print("="*60 + "\n")
    
    try:
        test_basic_comma_spacing()
        test_remove_spaces_before_commas()
        test_deduplicate_commas()
        test_number_lists()
        test_thousands_get_spaces()
        test_mixed_content()
        test_preserves_already_correct()
        test_empty_and_edge_cases()
        
        print("\n" + "="*60)
        print("✅ ALL COMMA SPACING NORMALIZATION TESTS PASSED!")
        print("="*60 + "\n")
        return True
        
    except AssertionError as e:
        print("\n" + "="*60)
        print(f"❌ TEST FAILED: {e}")
        print("="*60 + "\n")
        return False
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ UNEXPECTED ERROR: {e}")
        print("="*60 + "\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

