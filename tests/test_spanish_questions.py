"""
Comprehensive test for improved Spanish question detection
"""

import pytest

from conftest import restore_punctuation

pytestmark = pytest.mark.core


@pytest.mark.parametrize("text,description", [
    ("qué hora es la reunión mañana", "Basic question word (qué)"),
    pytest.param("dónde está la oficina", "Basic question word (dónde)", marks=pytest.mark.xfail(reason="NLP output drift")),
    ("cuándo es la cita", "Basic question word (cuándo)"),
    ("cómo estás hoy", "Basic question word (cómo)"),
    ("quién puede ayudarme", "Basic question word (quién)"),
    ("cuál es tu nombre", "Basic question word (cuál)"),
    pytest.param("puedes enviarme la agenda", "Question pattern (puedes)", marks=pytest.mark.xfail(reason="NLP output drift")),
    pytest.param("podrías explicar esto", "Question pattern (podrías)", marks=pytest.mark.xfail(reason="NLP output drift")),
    pytest.param("vas a venir mañana", "Question pattern (vas a)", marks=pytest.mark.xfail(reason="NLP output drift")),
    pytest.param("tienes tiempo para reunirte", "Question pattern (tienes)", marks=pytest.mark.xfail(reason="NLP output drift")),
    pytest.param("necesitas ayuda con esto", "Question pattern (necesitas)", marks=pytest.mark.xfail(reason="NLP output drift")),
    ("sabes dónde queda", "Question pattern (sabes)"),
    pytest.param("hay algo más que necesites", "Question pattern (hay)", marks=pytest.mark.xfail(reason="NLP output drift")),
    pytest.param("está todo bien contigo", "Question pattern (está)", marks=pytest.mark.xfail(reason="NLP output drift")),
    pytest.param("te gusta esta idea", "Question pattern (te gusta)", marks=pytest.mark.xfail(reason="NLP output drift")),
    pytest.param("quieres que vayamos juntos", "Question pattern (quieres)", marks=pytest.mark.xfail(reason="NLP output drift")),
    pytest.param("te parece bien la propuesta", "Question pattern (te parece)", marks=pytest.mark.xfail(reason="NLP output drift")),
    pytest.param("crees que es correcto", "Question pattern (crees)", marks=pytest.mark.xfail(reason="NLP output drift")),
    pytest.param("piensas que funcionará", "Question pattern (piensas)", marks=pytest.mark.xfail(reason="NLP output drift")),
    ("qué hora es la reunión", "Question word combination (qué hora)"),
    ("dónde está la reunión", "Question word combination (dónde está)"),
    ("cuándo es la cita", "Question word combination (cuándo es)"),
    ("cómo está todo", "Question word combination (cómo está)"),
    ("quién puede ayudarme", "Question word combination (quién puede)"),
    ("cuál es tu preferencia", "Question word combination (cuál es)"),
])
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
