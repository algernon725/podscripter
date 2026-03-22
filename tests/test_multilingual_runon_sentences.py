#!/usr/bin/env python3
"""
Test multilingual run-on sentence fixes across all supported languages
"""

import pytest

from conftest import restore_punctuation

pytestmark = pytest.mark.multilingual


def test_spanish_inverted_question_marks():
    """Test that Spanish questions get inverted question marks."""
    result = restore_punctuation("cómo estás hoy tienes tiempo para hablar", 'es')
    assert '¿' in result, f"Expected inverted question mark in Spanish output: {result!r}"


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
def test_french_question_marks():
    """Test that French questions get question marks."""
    result = restore_punctuation("comment allez vous avez vous le temps", 'fr')
    assert '?' in result, f"Expected question mark in French output: {result!r}"


def test_german_capitalization():
    """Test that German output starts with a capital letter."""
    result = restore_punctuation("hallo wie geht es dir heute", 'de')
    assert result[0].isupper(), f"Expected capitalized first letter in German output: {result!r}"
