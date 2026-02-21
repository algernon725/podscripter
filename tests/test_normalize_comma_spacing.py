#!/usr/bin/env python3
"""
Unit tests for the centralized _normalize_comma_spacing() function.

This test ensures the refactored comma spacing logic works correctly
and consistently across all call sites.
"""

from punctuation_restorer import _normalize_comma_spacing
import pytest

pytestmark = pytest.mark.core


def test_basic_comma_spacing():
    """Test basic comma spacing normalization."""
    assert _normalize_comma_spacing("a,b,c") == "a, b, c"
    assert _normalize_comma_spacing("uno,dos,tres") == "uno, dos, tres"

    assert _normalize_comma_spacing("a, b, c") == "a, b, c"


def test_remove_spaces_before_commas():
    """Test that spaces before commas are removed."""
    result = _normalize_comma_spacing("palabra ,otra")
    assert result == "palabra, otra", f"Expected 'palabra, otra' but got '{result}'"

    result = _normalize_comma_spacing("test  ,  value")
    assert result == "test, value", f"Expected 'test, value' but got '{result}'"

    result = _normalize_comma_spacing("a , b , c")
    assert result == "a, b, c", f"Expected 'a, b, c' but got '{result}'"


def test_deduplicate_commas():
    """Test that multiple commas are deduplicated."""
    assert _normalize_comma_spacing("test,,doble") == "test, doble"
    assert _normalize_comma_spacing("test, ,doble") == "test, doble"
    assert _normalize_comma_spacing("test,,,triple") == "test, triple"
    assert _normalize_comma_spacing("a, , , ,b") == "a, b"


def test_number_lists():
    """Test that number lists get proper spacing."""
    assert _normalize_comma_spacing("episodio 147,151,156") == "episodio 147, 151, 156"
    assert _normalize_comma_spacing("147,151,156,164,170,177") == "147, 151, 156, 164, 170, 177"

    assert _normalize_comma_spacing("1,2,3,4,5") == "1, 2, 3, 4, 5"
    assert _normalize_comma_spacing("10,20,30,40") == "10, 20, 30, 40"
    assert _normalize_comma_spacing("100,200,300") == "100, 200, 300"


def test_thousands_get_spaces():
    """Test that thousands separators also get spaces (documented trade-off)."""
    assert _normalize_comma_spacing("1,000") == "1, 000"
    assert _normalize_comma_spacing("25,000") == "25, 000"
    assert _normalize_comma_spacing("1,000,000") == "1, 000, 000"


def test_mixed_content():
    """Test text with mixed commas (lists, phrases, etc)."""
    text = "episodios 147,151,156 con María,Juan,Pedro y otros"
    result = _normalize_comma_spacing(text)
    assert result == "episodios 147, 151, 156 con María, Juan, Pedro y otros"

    text = "en el año 2,000 había más de 1,000 personas"
    result = _normalize_comma_spacing(text)
    assert result == "en el año 2, 000 había más de 1, 000 personas"


def test_preserves_already_correct():
    """Test that already correctly spaced text is preserved."""
    text = "This is correct, with proper spacing, throughout the sentence."
    assert _normalize_comma_spacing(text) == text

    text = "uno, dos, tres, cuatro"
    assert _normalize_comma_spacing(text) == text


def test_empty_and_edge_cases():
    """Test edge cases like empty strings."""
    assert _normalize_comma_spacing("") == ""
    assert _normalize_comma_spacing(None) == ""
    assert _normalize_comma_spacing(",") == ","
    assert _normalize_comma_spacing(",,") == ", "
    assert _normalize_comma_spacing(" , ") == ", "
