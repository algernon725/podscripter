#!/usr/bin/env python3
"""
Test normalization of person initials and organizational acronyms.

Tests both the _normalize_initials_and_acronyms function directly (unit tests)
and the end-to-end pipeline behavior through restore_punctuation.

Normalization runs in podscripter._assemble_sentences(), before
restore_punctuation(). restore_punctuation() on its own does not normalize, so
end-to-end checks must go through _assemble_sentences().
"""

from conftest import restore_punctuation
from podscripter import _assemble_sentences
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


def test_person_initials_survive_full_pipeline():
    """Person initials survive the full pipeline (_assemble_sentences).

    This was an xfail blamed on spaCy re-spacing initials. The real cause was
    that it called restore_punctuation(), which never normalizes; the pipeline
    normalizes first, in _assemble_sentences().
    """
    sentences, _ = _assemble_sentences(
        'es a C. S. Lewis porque él escribió muchos libros', [], 'es', True
    )
    result = ' '.join(s.text for s in sentences)
    assert 'C.S. Lewis' in result, \
        f"Expected compact 'C.S. Lewis' in pipeline output, got: '{result}'"
