#!/usr/bin/env python3
"""
Test suite for domain_utils module.

Tests the centralized domain detection, masking, and unmasking utilities.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from domain_utils import (
    mask_domains, unmask_domains, fix_spaced_domains,
    apply_safe_text_processing, create_domain_aware_regex,
    _get_domain_safe_split_pattern, _is_spanish_word
)
import re


def test_spanish_word_detection():
    """Test Spanish word exclusion logic."""
    print("Testing Spanish word detection...")
    
    # Spanish words that should be excluded
    spanish_words = ["uno", "dos", "tres", "este", "esta", "muy", "el", "la"]
    for word in spanish_words:
        assert _is_spanish_word(word), f"{word} should be detected as Spanish word"
        assert _is_spanish_word(word.upper()), f"{word.upper()} should be detected as Spanish word"
        assert _is_spanish_word(word.capitalize()), f"{word.capitalize()} should be detected as Spanish word"
    
    # Non-Spanish words that should not be excluded
    non_spanish = ["google", "github", "amazon", "microsoft", "facebook"]
    for word in non_spanish:
        assert not _is_spanish_word(word), f"{word} should NOT be detected as Spanish word"
    
    print("✅ Spanish word detection working correctly")


def test_domain_masking():
    """Test domain masking functionality."""
    print("\nTesting domain masking...")
    
    test_cases = [
        # (input, expected_with_exclusions, expected_without_exclusions)
        ("Visit google.com for search", "Visit google__DOT__com for search", "Visit google__DOT__com for search"),
        ("Go to github.de for code", "Visit github__DOT__de for code", "Visit github__DOT__de for code"),
        ("Y que es uno.de los lugares", "Y que es uno.de los lugares", "Y que es uno__DOT__de los lugares"),
        ("Este.es muy importante", "Este.es muy importante", "Este__DOT__es muy importante"),
        ("Check bbc.co.uk for news", "Check bbc__DOT__co_DOT_uk for news", "Check bbc__DOT__co_DOT_uk for news"),
        ("Mixed: google.com and uno.de", "Mixed: google__DOT__com and uno.de", "Mixed: google__DOT__com and uno__DOT__de"),
    ]
    
    for input_text, expected_with_exclusions, expected_without_exclusions in test_cases:
        # Test with exclusions (default)
        result_with = mask_domains(input_text, use_exclusions=True)
        if result_with != expected_with_exclusions:
            print(f"❌ FAIL (with exclusions): {input_text}")
            print(f"   Expected: {expected_with_exclusions}")
            print(f"   Got:      {result_with}")
        else:
            print(f"✅ PASS (with exclusions): {input_text}")
        
        # Test without exclusions
        result_without = mask_domains(input_text, use_exclusions=False)
        if result_without != expected_without_exclusions:
            print(f"❌ FAIL (without exclusions): {input_text}")
            print(f"   Expected: {expected_without_exclusions}")
            print(f"   Got:      {result_without}")
        else:
            print(f"✅ PASS (without exclusions): {input_text}")


def test_domain_unmasking():
    """Test domain unmasking functionality."""
    print("\nTesting domain unmasking...")
    
    test_cases = [
        ("Visit google__DOT__com today", "Visit google.com today"),
        ("Check bbc__DOT__co_DOT_uk news", "Check bbc.co.uk news"),
        ("Mixed: google__DOT__com and amazon__DOT__co_DOT_uk", "Mixed: google.com and amazon.co.uk"),
        ("No domains here", "No domains here"),
    ]
    
    for masked, expected in test_cases:
        result = unmask_domains(masked)
        if result == expected:
            print(f"✅ PASS: {masked} -> {result}")
        else:
            print(f"❌ FAIL: {masked}")
            print(f"   Expected: {expected}")
            print(f"   Got:      {result}")


def test_spaced_domain_fixing():
    """Test fixing domains that have spaces."""
    print("\nTesting spaced domain fixing...")
    
    test_cases = [
        # (input, expected_with_exclusions, expected_without_exclusions)
        ("Visit google. com today", "Visit google.com today", "Visit google.com today"),
        ("Go to github. de now", "Go to github.de now", "Go to github.de now"),
        ("Y que es uno. de los", "Y que es uno. de los", "Y que es uno.de los"),  # Spanish word preserved with exclusions
        ("Check bbc. co. uk news", "Check bbc.co.uk news", "Check bbc.co.uk news"),
        ("Mixed: google. com and uno. de", "Mixed: google.com and uno. de", "Mixed: google.com and uno.de"),
    ]
    
    for input_text, expected_with, expected_without in test_cases:
        # Test with exclusions
        result_with = fix_spaced_domains(input_text, use_exclusions=True)
        if result_with == expected_with:
            print(f"✅ PASS (with exclusions): {input_text} -> {result_with}")
        else:
            print(f"❌ FAIL (with exclusions): {input_text}")
            print(f"   Expected: {expected_with}")
            print(f"   Got:      {result_with}")
        
        # Test without exclusions
        result_without = fix_spaced_domains(input_text, use_exclusions=False)
        if result_without == expected_without:
            print(f"✅ PASS (without exclusions): {input_text} -> {result_without}")
        else:
            print(f"❌ FAIL (without exclusions): {input_text}")
            print(f"   Expected: {expected_without}")
            print(f"   Got:      {result_without}")


def test_safe_text_processing():
    """Test safe text processing that protects domains."""
    print("\nTesting safe text processing...")
    
    def add_spaces_after_periods(text):
        return re.sub(r'\.([A-Z])', r'. \1', text)
    
    test_cases = [
        ("Visit google.com.Then go home", "Visit google.com. Then go home"),
        ("Go to uno.de.Después come", "Go to uno.de. Después come"),  # Spanish word preserved
        ("Check github.de.Es muy bueno", "Check github.de. Es muy bueno"),
    ]
    
    for input_text, expected in test_cases:
        result = apply_safe_text_processing(input_text, add_spaces_after_periods)
        if result == expected:
            print(f"✅ PASS: {input_text} -> {result}")
        else:
            print(f"❌ FAIL: {input_text}")
            print(f"   Expected: {expected}")
            print(f"   Got:      {result}")


def test_domain_aware_regex():
    """Test domain-aware regex creation."""
    print("\nTesting domain-aware regex...")
    
    # Create a function that adds spaces after periods before capital letters
    space_after_period = create_domain_aware_regex(r'\.([A-Z])', r'. \1')
    
    test_cases = [
        ("Visit google.com.Then go", "Visit google.com. Then go"),
        ("Go to uno.de.Después ven", "Go to uno.de. Después ven"),  # Spanish preserved
        ("Check sites: github.de.Es bueno", "Check sites: github.de. Es bueno"),
    ]
    
    for input_text, expected in test_cases:
        result = space_after_period(input_text)
        if result == expected:
            print(f"✅ PASS: {input_text} -> {result}")
        else:
            print(f"❌ FAIL: {input_text}")
            print(f"   Expected: {expected}")
            print(f"   Got:      {result}")


def test_roundtrip_processing():
    """Test that masking and unmasking is reversible for real domains."""
    print("\nTesting roundtrip processing...")
    
    test_cases = [
        "Visit google.com and github.de for code",
        "Check bbc.co.uk and amazon.com.ar for shopping",
        "Real domains: marca.es, github.de, google.com",
    ]
    
    for original in test_cases:
        # Mask and then unmask - should get back original for real domains
        masked = mask_domains(original, use_exclusions=True)
        unmasked = unmask_domains(masked)
        
        if unmasked == original:
            print(f"✅ PASS: {original}")
        else:
            print(f"❌ FAIL: {original}")
            print(f"   After roundtrip: {unmasked}")


def test_integration_with_user_case():
    """Test the specific case reported by the user."""
    print("\nTesting integration with user's reported case...")
    
    user_input = "Y que es uno.de los lugares más caros para visitar."
    
    # Test that uno.de is NOT masked (because it's a Spanish word)
    masked = mask_domains(user_input, use_exclusions=True)
    if "uno__DOT__de" not in masked:
        print(f"✅ PASS: uno.de correctly NOT masked: {masked}")
    else:
        print(f"❌ FAIL: uno.de incorrectly masked: {masked}")
    
    # Test that after text processing, uno.de can be properly split
    def add_space_after_period(text):
        return re.sub(r'\.([A-Z])', r'. \1', text)
    
    result = apply_safe_text_processing(user_input, add_space_after_period)
    # The processing function won't change this case, but the important thing
    # is that uno.de is not protected as a domain
    print(f"After safe processing: {result}")


if __name__ == "__main__":
    print("Running domain utilities tests...")
    print("=" * 50)
    
    test_spanish_word_detection()
    test_domain_masking()
    test_domain_unmasking()
    test_spaced_domain_fixing()
    test_safe_text_processing()
    test_domain_aware_regex()
    test_roundtrip_processing()
    test_integration_with_user_case()
    
    print("\n" + "=" * 50)
    print("Domain utilities tests completed!")
