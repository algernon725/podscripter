#!/usr/bin/env python3
"""
Test for Spanish false domain detection fix.

This test ensures that common Spanish words followed by TLD-like suffixes 
are not incorrectly treated as domain names.

Example bug: "uno.de los lugares" was incorrectly preserved as "uno.de" domain
instead of being split into "uno. de" (number followed by preposition).
"""

import pytest
from podscripter import _assemble_sentences
from conftest import restore_punctuation

pytestmark = pytest.mark.core


def test_real_domains_preserved():
    """Test that real domains are preserved and not split."""

    real_domain_cases = [
        ("Visita github.de para el código", "github.de"),
        ("Ve a google.com para buscar", "google.com"),
        ("Consulta marca.es para noticias", "marca.es"),
        ("Accede a amazon.co.uk para comprar", "amazon.co.uk"),
    ]

    for test_input, domain in real_domain_cases:
        sentences, _meta = _assemble_sentences(test_input, [], 'es', True)
        sentences = [s.text if hasattr(s, 'text') else s for s in sentences]
        output = " ".join(sentences) if sentences else test_input

        assert domain in output, (
            f"Real domain '{domain}' not preserved in '{test_input}': got '{output}'"
        )


def test_punctuation_restoration_false_domains():
    """Test punctuation restoration doesn't create false domains."""

    test_cases = [
        ("Y que es uno.de los lugares más caros para visitar", "uno. de"),
        ("Este.es muy importante para el proyecto", "este. es"),
        ("Son dos.com de los mejores sitios web", "dos. com"),
    ]

    for test_input, expected_split in test_cases:
        result = restore_punctuation(test_input, 'es')

        assert expected_split in result.lower(), (
            f"Spanish words not properly separated in '{test_input}': got '{result}'"
        )
