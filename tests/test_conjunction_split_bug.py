"""
Test to reproduce and verify fix for the conjunction splitting bug.

Issue: Sentences are being split after coordinating conjunctions like "y" (and).
Example: "teníamos muchos errores y." | "Eco una vez..." 
This is grammatically incorrect - "y" should never end a sentence.

This test verifies the fix works across all supported languages.
"""

import pytest

from conftest import restore_punctuation

pytestmark = pytest.mark.core


def test_spanish_y_followed_by_regular_word():
    """Test that Spanish 'y' + regular word doesn't split."""
    text = "teníamos muchos errores y eco"
    result = restore_punctuation(text, language='es')

    assert not (result.strip().endswith("y.") or ". Eco" in result or ".\nEco" in result), \
        f"'y' was separated from 'eco': {result}"
    assert "y eco" in result.lower(), f"'y eco' not kept together: {result}"


def test_spanish_pero_followed_by_word():
    """Test that Spanish 'pero' (but) + regular word doesn't split."""
    text = "era muy difícil pero empezamos a trabajar"
    result = restore_punctuation(text, language='es')

    assert not (result.strip().endswith("pero.") or ". empezamos" in result.lower()), \
        f"'pero' was separated from 'empezamos': {result}"
    assert "pero empezamos" in result.lower(), f"'pero empezamos' not kept together: {result}"


def test_english_and_followed_by_word():
    """Test that English 'and' + regular word doesn't split."""
    text = "we had many errors and echo in the garage"
    result = restore_punctuation(text, language='en')

    assert not (result.strip().endswith("and.") or ". echo" in result.lower() or ".\necho" in result.lower()), \
        f"'and' was separated from 'echo': {result}"
    assert "and echo" in result.lower(), f"'and echo' not kept together: {result}"


def test_french_et_followed_by_word():
    """Test that French 'et' (and) + regular word doesn't split."""
    text = "nous avions beaucoup d'erreurs et écho dans le garage"
    result = restore_punctuation(text, language='fr')

    assert not (result.strip().endswith("et.") or ". écho" in result.lower() or ".\nécho" in result.lower()), \
        f"'et' was separated from 'écho': {result}"
    assert "et écho" in result.lower(), f"'et écho' not kept together: {result}"


def test_german_und_followed_by_word():
    """Test that German 'und' (and) + regular word doesn't split."""
    text = "wir hatten viele Fehler und Echo in der Garage"
    result = restore_punctuation(text, language='de')

    assert not (result.strip().endswith("und.") or ". Echo" in result or ".\nEcho" in result), \
        f"'und' was separated from 'Echo': {result}"
    assert "und Echo" in result, f"'und Echo' not kept together: {result}"


def test_longer_spanish_context():
    """Test with longer context similar to the actual bug."""
    text = "Ah sí no eran muy muy buenos teníamos muchos errores y eco una vez que grabábamos el video en el garaje y había tanto eco"
    result = restore_punctuation(text, language='es')

    sentences = result.split('.')
    for sent in sentences:
        sent_stripped = sent.strip()
        assert not (sent_stripped.endswith(' y') or sent_stripped.endswith(',y')), \
            f"Found sentence ending with 'y': '{sent_stripped}'"
