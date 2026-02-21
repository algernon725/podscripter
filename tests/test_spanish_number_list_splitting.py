#!/usr/bin/env python3
"""
Test for Spanish number list splitting bug.

Bug: Sentences containing lists of numbers with conjunctions (y/o) are being
incorrectly split at the final number's period.

Example:
Input: "puedes ir al episodio 147, 151, 156, 164, 170, 177 y 184."
Expected: Single sentence
Actual: Split into "...177 y." and "184."
"""

import pytest

from conftest import restore_punctuation
from punctuation_restorer import assemble_sentences_from_processed

pytestmark = pytest.mark.core


def test_number_list_with_y_not_split():
    """Test that a list of numbers with 'y' is not split at the final number."""
    text = "Pero si tú quieres escuchar los episodios anteriores, puedes ir al episodio 147, 151, 156, 164, 170, 177 y 184"
    result = restore_punctuation(text, language='es')

    lines = result.split('\n')

    standalone_184 = any(line.strip() == "184." for line in lines)
    assert not standalone_184, \
        f"Number list incorrectly split - 184 is standalone. Result:\n{result}"

    assert not ("y.\n184" in result or "y. 184." == result.split('\n')[-1].strip()), \
        f"Number list incorrectly split. Result:\n{result}"

    assert "177 y 184" in result, \
        f"Number list broken apart. Result:\n{result}"


def test_simple_number_list_with_y():
    """Test a simple number list with 'y' conjunction."""
    text = "Los episodios son 1 2 3 y 4"
    result = restore_punctuation(text, language='es')

    assert "3 y 4" in result, \
        f"Simple number list broken. Result: {result}"
    assert "y." not in result or "y. 4" not in result, \
        f"Incorrectly split at 'y'. Result: {result}"


def test_number_list_with_o():
    """Test that number lists with 'o' (or) are also preserved."""
    text = "Puedes elegir la opción 1 2 o 3"
    result = restore_punctuation(text, language='es')

    assert "2 o 3" in result, \
        f"Number list with 'o' broken. Result: {result}"
    assert "o." not in result or "o. 3" not in result, \
        f"Incorrectly split at 'o'. Result: {result}"


def test_year_list_with_y():
    """Test that lists of years are also preserved."""
    text = "Los años 2015 2016 2017 y 2018 fueron importantes"
    result = restore_punctuation(text, language='es')

    assert "2017 y 2018" in result, \
        f"Year list broken. Result: {result}"


def test_mixed_content_after_number_list():
    """Test that we still split correctly when there's genuinely new content."""
    text = "Ve a los episodios 1 2 y 3 Luego continúa con el 4"
    result = restore_punctuation(text, language='es')

    assert "1, 2 y 3" in result or "1 2 y 3" in result, \
        f"Number list broken. Result: {result}"


def test_sentence_assembly_number_list():
    """Test that sentence assembly doesn't split number lists."""
    restored = "Pero si tú quieres escuchar los episodios anteriores, puedes ir al episodio 147, 151, 156, 164, 170, 177 y 184."

    sentences, trailing = assemble_sentences_from_processed(restored, 'es')

    all_sents = sentences + ([trailing] if trailing else [])

    standalone_184 = any(s.strip() in ["184.", "184"] for s in all_sents)
    assert not standalone_184, \
        f"Number list incorrectly split - 184 is standalone. Sentences: {all_sents}"

    full_list_found = any("177 y 184" in s for s in all_sents)
    assert full_list_found, \
        f"Number list broken apart. Sentences: {all_sents}"
