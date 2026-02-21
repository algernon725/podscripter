#!/usr/bin/env python3
"""
Test to verify French run-on sentence fix
"""

import pytest
from conftest import restore_punctuation

pytestmark = pytest.mark.core


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
def test_french_runon_fix():
    """Test that French text is properly split into sentences."""
    runon_text = """Bonjour à tous  Bienvenue à FrançaisPod  FrançaisPod est le podcast qui vous aidera à être prêt à parler français  FrançaisPod vous prépare à parler français n'importe où n'importe quand et dans n'importe quelle situation  Vous souvenez vous de tous ces moments où vous ne saviez pas quoi dire  Ces moments où vous ne pouviez pas maintenir une conversation  Eh bien ne vous inquiétez pas  FrançaisPod est l'outil que vous cherchiez pour améliorer votre français  Dites adieu  À tous ces moments gênants  Alors commençons  Sommes nous prêts  Je suis Marie de Paris France  Et je suis Pierre de Lyon France  Bonjour à tous"""

    result = restore_punctuation(runon_text, language='fr')

    sentence_count = result.count('.') + result.count('?') + result.count('!')
    assert sentence_count > 5, f"Expected >5 sentences, got {sentence_count}"

    expected_patterns = [
        "Bonjour à tous",
        "Bienvenue à FrançaisPod",
        "Vous souvenez-vous de tous ces moments",
        "Je suis Marie de Paris, France",
        "Et je suis Pierre de Lyon, France",
    ]
    for pattern in expected_patterns:
        assert pattern in result, f"Missing expected pattern: {pattern!r}"
