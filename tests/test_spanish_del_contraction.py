#!/usr/bin/env python3
"""
Test for Spanish contraction sentence splitting bug.

Bug: Sentences were being split incorrectly after 'del' (de + el) and 'al' (a + el).
Example: "en la parte del. Amazonas" should be "en la parte del Amazonas"

This test ensures that Spanish contractions 'del' and 'al' prevent inappropriate
sentence breaks, following the same logic as their component prepositions.
"""

from conftest import restore_punctuation
import pytest

pytestmark = pytest.mark.core


def test_del_contraction_no_split():
    """Test that 'del' (de + el) prevents sentence splitting."""
    text = (
        "O yo hablaba con un estudiante que vive en Brasil de hecho y él me decía que "
        "en la parte del Amazonas en Brasil en la parte del Amazonas estaban talando "
        "muchos árboles para poder crear sabanas"
    )

    result = restore_punctuation(text, language='es')

    assert "del." not in result, f"Sentence incorrectly split after 'del'. Result: {result}"
    assert "del Amazonas" in result, f"'del Amazonas' should stay together. Result: {result}"


def test_al_contraction_no_split():
    """Test that 'al' (a + el) prevents sentence splitting."""
    text = (
        "Cuando llegué al aeropuerto había mucha gente esperando y todos estaban muy "
        "emocionados porque era un día especial para la ciudad"
    )

    result = restore_punctuation(text, language='es')

    assert "al." not in result, f"Sentence incorrectly split after 'al'. Result: {result}"
    assert "al aeropuerto" in result, f"'al aeropuerto' should stay together. Result: {result}"


def test_multiple_del_occurrences():
    """Test text with multiple 'del' occurrences."""
    text = (
        "La parte del norte del país es diferente del sur porque tiene un clima "
        "distinto y la geografía del terreno cambia mucho"
    )

    result = restore_punctuation(text, language='es')

    assert result.count("del.") == 0, f"Found inappropriate splits after 'del'. Result: {result}"
    assert "del norte del país" in result, f"Multiple 'del' should stay connected. Result: {result}"


def test_contractions_with_proper_nouns():
    """Test contractions followed by proper nouns (location names)."""
    text = (
        "Yo soy de Texas Estados Unidos pero viajé al Amazonas en Brasil y también "
        "fui del Amazonas directo a Colombia"
    )

    result = restore_punctuation(text, language='es')

    assert "al." not in result, f"Should not split after 'al'. Result: {result}"
    assert "del." not in result, f"Should not split after 'del'. Result: {result}"

    assert "al amazonas" in result.lower() or "al Amazonas" in result, \
        f"'al' should stay connected to 'Amazonas'. Result: {result}"
    assert "del amazonas" in result.lower() or "del Amazonas" in result, \
        f"'del' should stay connected to 'Amazonas'. Result: {result}"
