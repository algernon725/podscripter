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


def test_escucha_it_false_domain():
    """Test that 'Escucha. It's' is NOT merged as a domain (Escucha.it).
    
    Regression test for the bug where fix_spaced_domains treated
    'Escucha. It' as a spaced domain (Italy's .it TLD), producing
    'Escucha.it's raining cats and dogs.' instead of keeping the
    sentence boundary intact.
    """
    print("\nTesting Escucha.it false domain prevention...")
    
    # fix_spaced_domains should NOT merge 'Escucha. It' since .it is not in SINGLE_TLDS
    input_text = "Escucha. It's raining cats and dogs."
    result = fix_spaced_domains(input_text, use_exclusions=True, language='es')
    if result == input_text:
        print(f"✅ PASS: 'Escucha. It' correctly NOT merged: {result}")
    else:
        print(f"❌ FAIL: 'Escucha. It' was incorrectly merged")
        print(f"   Expected: {input_text}")
        print(f"   Got:      {result}")
    assert result == input_text, f"Expected unchanged text but got: {result}"
    
    # mask_domains should NOT mask Escucha.it since .it is not in SINGLE_TLDS
    input_text2 = "Escucha.it is wrong"
    masked = mask_domains(input_text2, use_exclusions=True, language='es')
    if "Escucha__DOT__it" not in masked:
        print(f"✅ PASS: Escucha.it correctly NOT masked: {masked}")
    else:
        print(f"❌ FAIL: Escucha.it was incorrectly masked: {masked}")
    assert "Escucha__DOT__it" not in masked, f"Escucha.it should not be masked"


def test_removed_tlds_not_matched():
    """Test that removed TLDs (.it, .nl, .jp, .cn, .in, .ru) are not matched."""
    print("\nTesting removed TLDs are not matched...")
    
    removed_tld_cases = [
        ("example.it", ".it"),
        ("example.nl", ".nl"),
        ("example.jp", ".jp"),
        ("example.cn", ".cn"),
        ("example.in", ".in"),
        ("example.ru", ".ru"),
    ]
    
    for input_text, tld in removed_tld_cases:
        masked = mask_domains(input_text, use_exclusions=True)
        if "__DOT__" not in masked:
            print(f"✅ PASS: {tld} correctly NOT masked in '{input_text}'")
        else:
            print(f"❌ FAIL: {tld} was incorrectly masked in '{input_text}': {masked}")
        assert "__DOT__" not in masked, f"{tld} should not be matched as a TLD"
    
    for input_text, tld in removed_tld_cases:
        spaced = input_text.replace(".", ". ")
        result = fix_spaced_domains(spaced, use_exclusions=True)
        if result == spaced:
            print(f"✅ PASS: {tld} correctly NOT fixed in '{spaced}'")
        else:
            print(f"❌ FAIL: {tld} was incorrectly fixed in '{spaced}': {result}")
        assert result == spaced, f"{tld} should not be fixed as a spaced domain"


def test_popular_tlds_still_work():
    """Test that popular TLDs still work correctly after refactor."""
    print("\nTesting popular TLDs still work...")
    
    popular_tld_cases = [
        ("google.com", "google__DOT__com"),
        ("github.io", "github__DOT__io"),
        ("harvard.edu", "harvard__DOT__edu"),
        ("example.net", "example__DOT__net"),
        ("example.org", "example__DOT__org"),
        ("example.fr", "example__DOT__fr"),
        ("example.br", "example__DOT__br"),
        ("example.ca", "example__DOT__ca"),
        ("example.au", "example__DOT__au"),
    ]
    
    for input_text, expected_masked in popular_tld_cases:
        masked = mask_domains(input_text, use_exclusions=False)
        if expected_masked in masked:
            print(f"✅ PASS: {input_text} correctly masked")
        else:
            print(f"❌ FAIL: {input_text} not correctly masked: {masked}")
        assert expected_masked in masked, f"Expected {expected_masked} in {masked}"


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
    test_escucha_it_false_domain()
    test_removed_tlds_not_matched()
    test_popular_tlds_still_work()
    
    print("\n" + "=" * 50)
    print("Domain utilities tests completed!")
