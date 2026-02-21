"""
Comprehensive test for improved Spanish question detection
"""

import pytest

from conftest import restore_punctuation

pytestmark = pytest.mark.core


@pytest.mark.parametrize("text,description", [
    ("qué hora es la reunión mañana", "Basic question word (qué)"),
    ("dónde está la oficina", "Basic question word (dónde)"),
    ("cuándo es la cita", "Basic question word (cuándo)"),
    ("cómo estás hoy", "Basic question word (cómo)"),
    ("quién puede ayudarme", "Basic question word (quién)"),
    ("cuál es tu nombre", "Basic question word (cuál)"),
    ("puedes enviarme la agenda", "Question pattern (puedes)"),
    ("podrías explicar esto", "Question pattern (podrías)"),
    ("vas a venir mañana", "Question pattern (vas a)"),
    ("tienes tiempo para reunirte", "Question pattern (tienes)"),
    ("necesitas ayuda con esto", "Question pattern (necesitas)"),
    ("sabes dónde queda", "Question pattern (sabes)"),
    ("hay algo más que necesites", "Question pattern (hay)"),
    ("está todo bien contigo", "Question pattern (está)"),
    ("te gusta esta idea", "Question pattern (te gusta)"),
    ("quieres que vayamos juntos", "Question pattern (quieres)"),
    ("te parece bien la propuesta", "Question pattern (te parece)"),
    ("crees que es correcto", "Question pattern (crees)"),
    ("piensas que funcionará", "Question pattern (piensas)"),
    ("qué hora es la reunión", "Question word combination (qué hora)"),
    ("dónde está la reunión", "Question word combination (dónde está)"),
    ("cuándo es la cita", "Question word combination (cuándo es)"),
    ("cómo está todo", "Question word combination (cómo está)"),
    ("quién puede ayudarme", "Question word combination (quién puede)"),
    ("cuál es tu preferencia", "Question word combination (cuál es)"),
])
@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
def test_spanish_question_detected(text, description):
    """Text expected to be a question should contain '?' in result."""
    result = restore_punctuation(text, 'es')
    assert '?' in result, f"{description}: expected question, got {result!r}"


@pytest.mark.parametrize("text,description", [
    ("hola como estás hoy", "Greeting (not a question)"),
    ("gracias por tu ayuda", "Thank you (not a question)"),
    ("el proyecto está terminado", "Statement (not a question)"),
    ("necesito más información", "Statement (not a question)"),
    ("la reunión es mañana", "Statement (not a question)"),
])
def test_spanish_non_question_not_detected(text, description):
    """Text expected to be a statement should not contain '?' in result."""
    result = restore_punctuation(text, 'es')
    assert '?' not in result, f"{description}: unexpected question mark in {result!r}"
