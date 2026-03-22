"""
Tests for numeric formatting in Spanish and other languages.
"""

import pytest

from punctuation_restorer import _finalize_text_common

pytestmark = pytest.mark.core


def test_enumeration_spacing_preserved():
    s = "Los grupos son 1, 2, 3 y 4."
    out = _finalize_text_common(s)
    assert "1, 2, 3" in out
    assert "1,2,3" not in out
