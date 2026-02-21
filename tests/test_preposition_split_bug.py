#!/usr/bin/env python3
"""
Test for sentence splitting bug where sentences are incorrectly split after prepositions.

Issue: Sentences end with prepositions like "a" in Spanish (similar to ending with "to" in English),
which is grammatically incorrect. Example:
- "entonces yo conocí a." | "Un amigo que trabajaba con cámaras" (INCORRECT)
- Should be: "entonces yo conocí a un amigo que trabajaba con cámaras" (CORRECT)

This test ensures that common prepositions in all supported languages (ES/EN/FR/DE) 
do not cause sentence splits.
"""

from conftest import restore_punctuation
import pytest

pytestmark = pytest.mark.core


def test_spanish_preposition_a_no_split():
    """Test that Spanish preposition 'a' does not cause sentence splits."""
    text = "entonces yo conocí a un amigo que trabajaba con cámaras pero él estaba muy ocupado"
    result = restore_punctuation(text, 'es')

    assert not result.endswith(' a.'), f"Sentence incorrectly ends with 'a.': {result}"
    assert ' a. ' not in result, f"Sentence incorrectly split after 'a': {result}"


def test_spanish_prepositions_comprehensive():
    """Test various Spanish prepositions that should not end sentences."""
    test_cases = [
        ("buscamos a una persona que supiera grabar videos", "a"),
        ("voy a la tienda mañana", "a"),
        ("estamos ante una situación difícil ahora", "ante"),
    ]

    for text, preposition in test_cases:
        result = restore_punctuation(text, 'es')
        error_pattern = f' {preposition}. '
        assert error_pattern not in result, \
            f"Sentence incorrectly split after '{preposition}': {result}"


def test_english_prepositions_no_split():
    """Test that English prepositions do not cause sentence splits."""
    test_cases = [
        ("I went to the store yesterday", "to"),
        ("we need to find someone who can help", "to"),
        ("she works at a large company in the city", "at"),
        ("he comes from Texas in the United States", "from"),
        ("I talked with my friend about the project", "with"),
    ]

    for text, preposition in test_cases:
        result = restore_punctuation(text, 'en')
        error_pattern = f' {preposition}. '
        assert error_pattern not in result, \
            f"English sentence incorrectly split after '{preposition}': {result}"


def test_french_prepositions_no_split():
    """Test that French prepositions do not cause sentence splits."""
    test_cases = [
        ("je vais à la maison maintenant", "à"),
        ("nous parlons à notre ami demain", "à"),
        ("il travaille avec son équipe aujourd'hui", "avec"),
    ]

    for text, preposition in test_cases:
        result = restore_punctuation(text, 'fr')
        error_pattern = f' {preposition}. '
        assert error_pattern not in result, \
            f"French sentence incorrectly split after '{preposition}': {result}"


def test_german_prepositions_no_split():
    """Test that German prepositions do not cause sentence splits."""
    test_cases = [
        ("ich gehe zu meinem Freund morgen", "zu"),
        ("er arbeitet bei einer großen Firma", "bei"),
        ("sie spricht mit ihrem Lehrer heute", "mit"),
    ]

    for text, preposition in test_cases:
        result = restore_punctuation(text, 'de')
        error_pattern = f' {preposition}. '
        assert error_pattern not in result, \
            f"German sentence incorrectly split after '{preposition}': {result}"


def test_user_reported_bug():
    """Test the exact scenario reported by the user from Episodio190_raw.txt"""
    text = """buscamos a una persona que supiera grabar videos entonces yo conocí a 
    un amigo que trabajaba con cámaras pero él estaba muy ocupado y él me recomendó 
    a un amigo de él y este chico pues nos grababa y todo pero él nunca mencionó 
    como tienen que comprar un micrófono de solapa"""

    result = restore_punctuation(text, 'es')

    assert ' a. U' not in result and ' a. u' not in result, \
        f"Bug reproduced: sentence split after 'a': {result}"

    sentences = result.split('.')
    for sentence in sentences:
        stripped = sentence.strip()
        assert not (stripped and stripped.endswith(' a')), \
            f"Sentence ends with preposition 'a': {stripped}"
