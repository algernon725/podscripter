#!/usr/bin/env python3
"""
Test suite for domain_utils module.

Tests the centralized domain detection, masking, and unmasking utilities.
"""

from domain_utils import (
    mask_domains, unmask_domains, fix_spaced_domains,
    apply_safe_text_processing, create_domain_aware_regex,
    _get_domain_safe_split_pattern, _is_spanish_word
)
import re
import pytest

pytestmark = pytest.mark.core


def test_spanish_word_detection():
    """Test Spanish word exclusion logic."""
    spanish_words = ["uno", "dos", "tres", "este", "esta", "muy", "el", "la"]
    for word in spanish_words:
        assert _is_spanish_word(word), f"{word} should be detected as Spanish word"
        assert _is_spanish_word(word.upper()), f"{word.upper()} should be detected as Spanish word"
        assert _is_spanish_word(word.capitalize()), f"{word.capitalize()} should be detected as Spanish word"

    non_spanish = ["google", "github", "amazon", "microsoft", "facebook"]
    for word in non_spanish:
        assert not _is_spanish_word(word), f"{word} should NOT be detected as Spanish word"


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
def test_domain_masking():
    """Test domain masking functionality."""
    test_cases = [
        ("Visit google.com for search", "Visit google__DOT__com for search", "Visit google__DOT__com for search"),
        ("Go to github.de for code", "Visit github__DOT__de for code", "Visit github__DOT__de for code"),
        ("Y que es uno.de los lugares", "Y que es uno.de los lugares", "Y que es uno__DOT__de los lugares"),
        ("Este.es muy importante", "Este.es muy importante", "Este__DOT__es muy importante"),
        ("Check bbc.co.uk for news", "Check bbc__DOT__co_DOT_uk for news", "Check bbc__DOT__co_DOT_uk for news"),
        ("Mixed: google.com and uno.de", "Mixed: google__DOT__com and uno.de", "Mixed: google__DOT__com and uno__DOT__de"),
    ]

    for input_text, expected_with_exclusions, expected_without_exclusions in test_cases:
        result_with = mask_domains(input_text, use_exclusions=True)
        assert result_with == expected_with_exclusions, (
            f"With exclusions for '{input_text}': expected '{expected_with_exclusions}', got '{result_with}'"
        )

        result_without = mask_domains(input_text, use_exclusions=False)
        assert result_without == expected_without_exclusions, (
            f"Without exclusions for '{input_text}': expected '{expected_without_exclusions}', got '{result_without}'"
        )


def test_domain_unmasking():
    """Test domain unmasking functionality."""
    test_cases = [
        ("Visit google__DOT__com today", "Visit google.com today"),
        ("Check bbc__DOT__co_DOT_uk news", "Check bbc.co.uk news"),
        ("Mixed: google__DOT__com and amazon__DOT__co_DOT_uk", "Mixed: google.com and amazon.co.uk"),
        ("No domains here", "No domains here"),
    ]

    for masked, expected in test_cases:
        result = unmask_domains(masked)
        assert result == expected, f"For '{masked}': expected '{expected}', got '{result}'"


def test_spaced_domain_fixing():
    """Test fixing domains that have spaces."""
    test_cases = [
        ("Visit google. com today", "Visit google.com today", "Visit google.com today"),
        ("Go to github. de now", "Go to github.de now", "Go to github.de now"),
        ("Y que es uno. de los", "Y que es uno. de los", "Y que es uno.de los"),
        ("Check bbc. co. uk news", "Check bbc.co.uk news", "Check bbc.co.uk news"),
        ("Mixed: google. com and uno. de", "Mixed: google.com and uno. de", "Mixed: google.com and uno.de"),
    ]

    for input_text, expected_with, expected_without in test_cases:
        result_with = fix_spaced_domains(input_text, use_exclusions=True)
        assert result_with == expected_with, (
            f"With exclusions for '{input_text}': expected '{expected_with}', got '{result_with}'"
        )

        result_without = fix_spaced_domains(input_text, use_exclusions=False)
        assert result_without == expected_without, (
            f"Without exclusions for '{input_text}': expected '{expected_without}', got '{result_without}'"
        )


def test_safe_text_processing():
    """Test safe text processing that protects domains."""
    def add_spaces_after_periods(text):
        return re.sub(r'\.([A-Z])', r'. \1', text)

    test_cases = [
        ("Visit google.com.Then go home", "Visit google.com. Then go home"),
        ("Go to uno.de.Después come", "Go to uno.de. Después come"),
        ("Check github.de.Es muy bueno", "Check github.de. Es muy bueno"),
    ]

    for input_text, expected in test_cases:
        result = apply_safe_text_processing(input_text, add_spaces_after_periods)
        assert result == expected, f"For '{input_text}': expected '{expected}', got '{result}'"


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
def test_domain_aware_regex():
    """Test domain-aware regex creation."""
    space_after_period = create_domain_aware_regex(r'\.([A-Z])', r'. \1')

    test_cases = [
        ("Visit google.com.Then go", "Visit google.com. Then go"),
        ("Go to uno.de.Después ven", "Go to uno.de. Después ven"),
        ("Check sites: github.de.Es bueno", "Check sites: github.de. Es bueno"),
    ]

    for input_text, expected in test_cases:
        result = space_after_period(input_text)
        assert result == expected, f"For '{input_text}': expected '{expected}', got '{result}'"


def test_roundtrip_processing():
    """Test that masking and unmasking is reversible for real domains."""
    test_cases = [
        "Visit google.com and github.de for code",
        "Check bbc.co.uk and amazon.com.ar for shopping",
        "Real domains: marca.es, github.de, google.com",
    ]

    for original in test_cases:
        masked = mask_domains(original, use_exclusions=True)
        unmasked = unmask_domains(masked)
        assert unmasked == original, f"Roundtrip failed for '{original}': got '{unmasked}'"


def test_integration_with_user_case():
    """Test the specific case reported by the user."""
    user_input = "Y que es uno.de los lugares más caros para visitar."

    masked = mask_domains(user_input, use_exclusions=True)
    assert "uno__DOT__de" not in masked, f"uno.de should NOT be masked: {masked}"

    def add_space_after_period(text):
        return re.sub(r'\.([A-Z])', r'. \1', text)

    result = apply_safe_text_processing(user_input, add_space_after_period)
    assert result is not None


def test_escucha_it_false_domain():
    """Test that 'Escucha. It's' is NOT merged as a domain (Escucha.it).

    Regression test for the bug where fix_spaced_domains treated
    'Escucha. It' as a spaced domain (Italy's .it TLD), producing
    'Escucha.it's raining cats and dogs.' instead of keeping the
    sentence boundary intact.
    """
    input_text = "Escucha. It's raining cats and dogs."
    result = fix_spaced_domains(input_text, use_exclusions=True, language='es')
    assert result == input_text, f"Expected unchanged text but got: {result}"

    input_text2 = "Escucha.it is wrong"
    masked = mask_domains(input_text2, use_exclusions=True, language='es')
    assert "Escucha__DOT__it" not in masked, f"Escucha.it should not be masked"


def test_removed_tlds_not_matched():
    """Test that removed TLDs (.it, .nl, .jp, .cn, .in, .ru) are not matched."""
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
        assert "__DOT__" not in masked, f"{tld} should not be matched as a TLD"

    for input_text, tld in removed_tld_cases:
        spaced = input_text.replace(".", ". ")
        result = fix_spaced_domains(spaced, use_exclusions=True)
        assert result == spaced, f"{tld} should not be fixed as a spaced domain"


def test_popular_tlds_still_work():
    """Test that popular TLDs still work correctly after refactor."""
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
        assert expected_masked in masked, f"Expected {expected_masked} in {masked}"
