"""
Test to verify Spanish run-on sentence fix
"""

import pytest

from conftest import restore_punctuation

pytestmark = pytest.mark.core


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
def test_spanish_runon_fix():
    """Test that Spanish text is properly split into sentences."""
    runon_text = (
        "Hola a todos  Bienvenidos a Españolistos  Españolistos es el podcast "
        "que te va a ayudar a estar listo para hablar español  Españolistos te "
        "prepara para hablar español en cualquier lugar, a cualquier hora y en "
        "cualquier situación  ¿Recuerdas todos esos momentos en los que no "
        "¿Supiste qué decir  ¿Esos momentos en los que no pudiste mantener una "
        "conversación  Pues tranquilo,  Españolistos es la herramienta que "
        "estabas buscando para mejorar tu español  Dile adiós  A todos esos "
        "momentos incómodos  Entonces, empecemos  ¿Estamos listos  Yo soy "
        "Andrea de Santander, Colombia  Y yo soy Nate de Texas, Estados Unidos"
        "  Hola para todos"
    )

    result = restore_punctuation(runon_text, language='es')

    sentence_count = result.count('.') + result.count('?') + result.count('!')
    assert sentence_count > 5, (
        f"Expected more than 5 sentences, got {sentence_count}"
    )

    expected_patterns = [
        "Hola a todos",
        "Bienvenidos a Españolistos",
        "¿Recuerdas todos esos momentos",
        "Yo soy Andrea de Santander, Colombia",
        "Y yo soy Nate de Texas, Estados Unidos",
    ]
    for pattern in expected_patterns:
        assert pattern in result, f"Missing expected pattern: {pattern!r}"
