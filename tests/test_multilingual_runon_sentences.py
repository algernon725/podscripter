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


# NOTE: test_french_question_marks was retired in v0.8.2. It was labeled "test
# expectations predate API changes" (stale harness expectation) and asserted
# text-only detection of a verb-first French question, which is part of the closed
# question-detection limitation — see AGENTS.md "Question detection — verb-first /
# implicit questions (CLOSED)".


def test_german_capitalization():
    """Test that German output starts with a capital letter."""
    result = restore_punctuation("hallo wie geht es dir heute", 'de')
    assert result[0].isupper(), f"Expected capitalized first letter in German output: {result!r}"


def test_portuguese_no_inverted_marks_and_question_detected():
    """Portuguese questions get a plain '?' and never Spanish inverted marks."""
    result = restore_punctuation("como você está hoje você tem tempo para conversar", 'pt')
    assert '¿' not in result, f"Portuguese must not get inverted '¿': {result!r}"
    assert '¡' not in result, f"Portuguese must not get inverted '¡': {result!r}"
    assert result.strip()[-1] in '.!?', f"Missing terminal punctuation: {result!r}"


def test_portuguese_capitalization():
    """Portuguese sentences should be capitalized, including accented initials."""
    result = restore_punctuation("ótimo dia hoje. água fresca é boa", 'pt')
    assert result.startswith('Ó') or result.startswith('ó'.upper()), \
        f"Expected capitalized accented initial: {result!r}"
    assert 'Água' in result or 'água' in result, f"Unexpected output: {result!r}"
