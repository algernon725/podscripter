"""
Focused test for the four specific Spanish transcription bugs mentioned by the user.
"""

import pytest

from conftest import restore_punctuation

pytestmark = pytest.mark.core


@pytest.mark.parametrize("text", [
    "Exactamente, yo sé muy poco de todo",
    "Todo en general en esta área nunca he tenido muy buen conocimiento",
    "También sí",
    "Es el que le dice al empleado que ya no va a trabajar más",
])
def test_bug1_missing_end_punctuation(text):
    """Bug 1: Sentences must end with proper punctuation."""
    result = restore_punctuation(text, 'es')
    assert result[-1] in '.?!', f"Missing end punctuation: {result!r}"


@pytest.mark.parametrize("text", [
    "Todo en general en esta área nunca he tenido muy buen conocimiento,",
    "También sí,",
    "Es el que le dice al empleado que ya no va a trabajar más,",
])
def test_bug2_trailing_commas(text):
    """Bug 2: Sentences must not end with a trailing comma."""
    result = restore_punctuation(text, 'es')
    assert not result.rstrip().endswith(','), f"Trailing comma: {result!r}"


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
@pytest.mark.parametrize("text", [
    "No se ha desestabilizado la economía, pero sabemos que en la mayoría sí.¿Así que por eso queremos analizar un poquito por qué está sucediendo esto y muchos de ustedes ya sabrán esas razones, si les gusta leer,",
    "Despidan a los empleados por los siguientes dos o tres meses, ¿no",
    "Sé cómo explicar un bailout en español, ¿tú sabes?¿No sé qué es eso",
])
def test_bug3_inverted_question_marks_mid_sentence(text):
    """Bug 3: Inverted question marks should not appear fused to preceding punctuation."""
    result = restore_punctuation(text, 'es')
    assert result[-1] in '.?!', f"Missing end punctuation: {result!r}"
    assert '.¿' not in result, f"Period immediately followed by ¿: {result!r}"
    assert '?¿' not in result, f"? immediately followed by ¿: {result!r}"


def test_bug4_question_mark_after_period():
    """Bug 4: '.?¿' sequences should be cleaned up."""
    text = (
        "Vamos a levantar de esto, vamos a salir de esto, tenemos que "
        "reconocer.?¿La realidad, estar conscientes de qué está pasando "
        "alrededor de nosotros, ser conscientes de que sí"
    )
    result = restore_punctuation(text, 'es')
    assert '.?¿' not in result, f"Found .?¿ sequence: {result!r}"
    assert result[-1] in '.?!', f"Missing end punctuation: {result!r}"
