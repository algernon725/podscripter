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


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
def test_spanish_false_domains():
    """Test that common Spanish words + TLD suffixes are not treated as domains."""

    test_cases = [
        ("Y que es uno.de los lugares más caros", "uno.de should split into 'uno. de'"),
        ("Este.es muy importante para nosotros", "Este.es should split into 'Este. es'"),
        ("Son dos.com de los mejores sitios", "dos.com should split into 'dos. com'"),
        ("Hay tres.es opciones disponibles", "tres.es should split into 'tres. es'"),
        ("El.co es una buena opción", "El.co should split into 'El. co'"),
        ("Muy.de acuerdo con la propuesta", "Muy.de should split into 'Muy. de'"),
        ("Necesita ser tratada.de hecho", "tratada.de should split into 'tratada. de'"),
        ("Era una noche.de verano", "noche.de should split into 'noche. de'"),
        ("La historia.de siempre", "historia.de should split into 'historia. de'"),
        ("Un poco.de todo", "poco.de should split into 'poco. de'"),
        ("estas fuentes naturales.es bien", "naturales.es should split into 'naturales. es'"),
        ("en las cuidades no.es que una maquina", "no.es should split into 'no. es'"),
    ]

    for test_input, description in test_cases:
        result = _assemble_sentences(test_input, 'es', quiet=True)
        output = result[0] if result else test_input

        correctly_split = False
        if ".de" in test_input and ". de" in output.lower():
            correctly_split = True
        elif ".es" in test_input and ". es" in output.lower():
            correctly_split = True
        elif ".com" in test_input and ". com" in output.lower():
            correctly_split = True
        elif ".co" in test_input and ". co" in output.lower():
            correctly_split = True

        assert correctly_split, (
            f"False domain not split for '{test_input}' ({description}): got '{output}'"
        )


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
def test_real_domains_preserved():
    """Test that real domains are preserved and not split."""

    real_domain_cases = [
        ("Visita github.de para el código", "github.de"),
        ("Ve a google.com para buscar", "google.com"),
        ("Consulta marca.es para noticias", "marca.es"),
        ("Accede a amazon.co.uk para comprar", "amazon.co.uk"),
    ]

    for test_input, domain in real_domain_cases:
        result = _assemble_sentences(test_input, 'es', quiet=True)
        output = result[0] if result else test_input

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
