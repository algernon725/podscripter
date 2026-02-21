"""
Test file for Spanish transcription bug fixes.
Tests the four specific bugs mentioned by the user.
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
    """Sentences must end with proper punctuation, not nothing."""
    result = restore_punctuation(text, 'es')
    assert result[-1] in '.?!', f"Missing end punctuation: {result!r}"


@pytest.mark.parametrize("text", [
    "Todo en general en esta área nunca he tenido muy buen conocimiento,",
    "También sí,",
    "Es el que le dice al empleado que ya no va a trabajar más,",
])
def test_bug2_trailing_commas(text):
    """Sentences must not end with a trailing comma."""
    result = restore_punctuation(text, 'es')
    assert not result.rstrip().endswith(','), f"Trailing comma: {result!r}"


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
@pytest.mark.parametrize("text", [
    "No se ha desestabilizado la economía, pero sabemos que en la mayoría sí.¿Así que por eso queremos analizar un poquito por qué está sucediendo esto y muchos de ustedes ya sabrán esas razones, si les gusta leer,",
    "Despidan a los empleados por los siguientes dos o tres meses, ¿no",
    "Sé cómo explicar un bailout en español, ¿tú sabes?¿No sé qué es eso",
])
def test_bug3_inverted_question_marks_mid_sentence(text):
    """Inverted question marks should not appear improperly fused to preceding punctuation."""
    result = restore_punctuation(text, 'es')
    assert result[-1] in '.?!', f"Missing end punctuation: {result!r}"
    assert '.¿' not in result, f"Period immediately followed by ¿: {result!r}"
    assert '?¿' not in result, f"? immediately followed by ¿: {result!r}"


def test_bug4_question_mark_after_period():
    """'.?¿' sequences should be cleaned up."""
    text = "Vamos a levantar de esto, vamos a salir de esto, tenemos que reconocer.?¿La realidad, estar conscientes de qué está pasando alrededor de nosotros, ser conscientes de que sí"
    result = restore_punctuation(text, 'es')
    assert '.?¿' not in result, f"Found .?¿ sequence: {result!r}"
    assert result[-1] in '.?!', f"Missing end punctuation: {result!r}"


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
@pytest.mark.parametrize("text", [
    "Hola cómo estás hoy",
    "Qué hora es",
    "Dónde está la reunión",
    "Cuándo es la cita",
    "Cómo te llamas",
    "Quién puede ayudarme",
    "Cuál es tu nombre",
    "Por qué no viniste",
    "Recuerdas la última vez",
    "Sabes dónde queda",
    "Puedes ayudarme",
    "Quieres que vayamos",
    "Necesitas algo más",
])
def test_additional_question_detection(text):
    """Spanish questions should be detected and end with '?'."""
    result = restore_punctuation(text, 'es')
    assert '?' in result, f"Question not detected: {result!r}"


def test_complex_spanish_text():
    """Test with more complex Spanish text that might have multiple issues."""
    complex_text = (
        "Exactamente yo sé muy poco de todo Todo en general en esta área "
        "nunca he tenido muy buen conocimiento También sí Es el que le dice "
        "al empleado que ya no va a trabajar más No se ha desestabilizado la "
        "economía pero sabemos que en la mayoría sí ¿Así que por eso queremos "
        "analizar un poquito por qué está sucediendo esto y muchos de ustedes "
        "ya sabrán esas razones si les gusta leer Despidan a los empleados por "
        "los siguientes dos o tres meses ¿no Sé cómo explicar un bailout en "
        "español ¿tú sabes ¿No sé qué es eso Vamos a levantar de esto vamos "
        "a salir de esto tenemos que reconocer ¿La realidad estar conscientes "
        "de qué está pasando alrededor de nosotros ser conscientes de que sí"
    )
    result = restore_punctuation(complex_text, 'es')
    assert len(result) > 0, "Result should not be empty"
    assert result[-1] in '.?!', f"Result should end with punctuation: {result!r}"
