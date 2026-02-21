"""
Test to verify that punctuation is preserved correctly in transcription output.
"""

import pytest

from conftest import restore_punctuation

pytestmark = pytest.mark.core


@pytest.mark.parametrize("text", [
    "Hola a todos",
    "Bienvenidos a Españolistos",
    "Españolistos es el podcast que te va a ayudar a estar listo para hablar español",
    "Españolistos te prepara para hablar español en cualquier lugar, a cualquier hora y en cualquier situación",
    "Recuerdas todos esos momentos en los que no supiste qué decir",
    "Esos momentos en los que no pudiste mantener una conversación",
    "Pues tranquilo",
    "Españolistos es la herramienta que estabas buscando para mejorar tu español",
])
def test_punctuation_preservation(text):
    """Test that punctuation is preserved correctly for individual sentences."""
    result = restore_punctuation(text, 'es')
    assert result and result.strip(), f"Empty result for input: {text}"
    assert result.strip()[-1] in '.!?¿¡', \
        f"Result doesn't end with punctuation: '{result}'"


def test_full_text_punctuation():
    """Test punctuation restoration on a full transcription-like text block."""
    full_text = (
        "Hola a todos "
        "Bienvenidos a Españolistos "
        "Españolistos es el podcast que te va a ayudar a estar listo para hablar español "
        "Españolistos te prepara para hablar español en cualquier lugar, a cualquier hora y en cualquier situación "
        "Recuerdas todos esos momentos en los que no supiste qué decir "
        "Esos momentos en los que no pudiste mantener una conversación "
        "Pues tranquilo "
        "Españolistos es la herramienta que estabas buscando para mejorar tu español"
    )

    result = restore_punctuation(full_text, 'es')
    assert result and result.strip(), f"Empty result for full text input"
    assert result.strip()[-1] in '.!?', \
        f"Full text result doesn't end with punctuation: '{result}'"
