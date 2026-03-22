#!/usr/bin/env python3
"""
Test normalization of person initials and organizational acronyms.

Tests both the _normalize_initials_and_acronyms function directly (unit tests)
and the end-to-end pipeline behavior through restore_punctuation.

Known limitation: spaCy's _apply_spacy_capitalization() can re-space initials
downstream, so the end-to-end pipeline does not always preserve compact initials.
The normalization function itself works correctly.
"""

from conftest import restore_punctuation
from punctuation_restorer import _normalize_initials_and_acronyms
import pytest

pytestmark = pytest.mark.core


# ── Unit tests for _normalize_initials_and_acronyms ──────────────────────────

def test_two_initial_person_name():
    """Two spaced initials + surname → compact with periods."""
    assert _normalize_initials_and_acronyms("C. S. Lewis") == "C.S. Lewis"
    assert _normalize_initials_and_acronyms("J. K. Rowling") == "J.K. Rowling"


def test_three_initial_person_name():
    """Three spaced initials + surname → compact with periods."""
    assert _normalize_initials_and_acronyms("J. R. R. Tolkien") == "J.R.R. Tolkien"


def test_person_name_in_spanish_context():
    """Initials in a Spanish sentence are compacted."""
    result = _normalize_initials_and_acronyms(
        "es a C. S. Lewis porque él escribió muchos libros"
    )
    assert "C.S. Lewis" in result


def test_person_name_in_french_context():
    result = _normalize_initials_and_acronyms(
        "j'aime lire C. S. Lewis parce qu'il a écrit des livres"
    )
    assert "C.S. Lewis" in result


def test_multiple_names_in_one_sentence():
    result = _normalize_initials_and_acronyms(
        "me gusta C. S. Lewis y también J. K. Rowling"
    )
    assert "C.S. Lewis" in result
    assert "J.K. Rowling" in result


def test_three_letter_acronym_at_end():
    """Three-letter acronym at end of text → periods and spaces removed."""
    assert "USA" in _normalize_initials_and_acronyms("in the U. S. A.")


def test_two_letter_acronym_before_lowercase():
    """Two-letter acronym before lowercase word → compact without periods."""
    result = _normalize_initials_and_acronyms("in the U. S. today")
    assert "US" in result


def test_empty_and_none():
    assert _normalize_initials_and_acronyms("") == ""
    assert _normalize_initials_and_acronyms(None) is None


def test_no_initials_unchanged():
    text = "Hello world, this has no initials at all."
    assert _normalize_initials_and_acronyms(text) == text


# ── End-to-end pipeline tests ────────────────────────────────────────────────

def test_english_organizational_acronyms():
    """English organizational acronyms survive the full pipeline."""
    test_cases = [
        {
            'input': 'the U. S. Capitol is in Washington D. C.',
            'description': 'U.S. and D.C. acronyms',
            'expected_acronyms': ['US', 'DC']
        },
        {
            'input': 'he lives in the U. S. A. and works for the F. B. I.',
            'description': 'USA and FBI acronyms',
            'expected_acronyms': ['USA', 'FBI']
        },
    ]

    for test in test_cases:
        result = restore_punctuation(test['input'], 'en')

        for acronym in test['expected_acronyms']:
            assert acronym in result, \
                f"[{test['description']}] Expected acronym '{acronym}' not found in '{result}'"


@pytest.mark.xfail(reason="WIP: spaCy _apply_spacy_capitalization re-spaces initials downstream")
def test_person_initials_survive_full_pipeline():
    """Person initials should survive the full restore_punctuation pipeline.

    Currently fails because spaCy's tokenizer/detokenizer re-inserts spaces
    during _apply_spacy_capitalization(), undoing the normalization.
    See AGENT.md 'Person Initials Normalization (WIP - Partial)'.
    """
    result = restore_punctuation(
        'es a C. S. Lewis porque él escribió muchos libros', 'es'
    )
    assert 'C.S. Lewis' in result, \
        f"Expected compact 'C.S. Lewis' in pipeline output, got: '{result}'"
