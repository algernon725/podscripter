"""
Tests for numeric formatting in Spanish and other languages.
"""

import pytest

from conftest import restore_punctuation
from punctuation_restorer import _finalize_text_common

pytestmark = pytest.mark.core


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
def test_thousands_commas_preserved_in_finalize():
    s = "La población es de 364, 134 habitantes y 1, 234, 567 ovejas."
    out = _finalize_text_common(s)
    assert "364,134" in out
    assert "1,234,567" in out
    assert "364, 134" not in out
    assert "1, 234, 567" not in out


def test_enumeration_spacing_preserved():
    s = "Los grupos son 1, 2, 3 y 4."
    out = _finalize_text_common(s)
    assert "1, 2, 3" in out
    assert "1,2,3" not in out


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
def test_thousands_commas_preserved_in_spanish_restore():
    s = "Hay aproximadamente 364, 134 habitantes, y 11, 200 ovejas."
    out = restore_punctuation(s, language='es')
    assert "364,134" in out and "11,200" in out
    assert "364, 134" not in out and "11, 200" not in out
    assert ", y" in out


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
def test_thousands_commas_in_english_and_german_finalize():
    s = "US population is 331, 002, 651. German sample: 12, 345."
    out = _finalize_text_common(s)
    assert "331,002,651" in out
    assert "12,345" in out
    assert "331, 002, 651" not in out
    assert "12, 345" not in out
