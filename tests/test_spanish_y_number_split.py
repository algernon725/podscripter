#!/usr/bin/env python3
"""
Test the specific Spanish "y 184" splitting bug.
"""

import pytest

from conftest import restore_punctuation
from punctuation_restorer import _semantic_split_into_sentences, _load_sentence_transformer

pytestmark = pytest.mark.core


def test_spanish_y_number():
    """Test that 'y 184' doesn't get split in Spanish."""

    text = "puedes ir al episodio 147, 151, 156, 164, 170, 177 y 184"

    model = _load_sentence_transformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    text_with_period = text + "."
    sentences = _semantic_split_into_sentences(text_with_period, 'es', model)

    assert not any(s.strip().endswith("y.") or s.strip().endswith("y") for s in sentences), \
        f"'y' was separated from '184': {sentences}"

    assert not any(s.strip() == "184." or s.strip() == "184" for s in sentences), \
        f"'184' is a standalone sentence: {sentences}"

    assert any("y 184" in s for s in sentences), \
        f"'y 184' not kept together: {sentences}"


def test_full_restoration_spanish():
    """Test full restoration pipeline for Spanish."""

    text = "Pero si tú quieres escuchar los episodios anteriores, puedes ir al episodio 147, 151, 156, 164, 170, 177 y 184"

    result = restore_punctuation(text, language='es')

    if "y." in result and "184" in result:
        assert "y. 184" not in result and "y.\n184" not in result, \
            f"'y' and '184' were split: {result}"

    assert "y 184" in result or "177 y 184" in result, \
        f"Number list not preserved: {result}"


def test_simple_spanish_list():
    """Test simple Spanish number list."""
    text = "Los episodios son 1, 2, 3 y 4"

    result = restore_punctuation(text, language='es')

    assert "3 y 4" in result, f"Simple list broken: {result}"
