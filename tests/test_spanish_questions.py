"""
Comprehensive test for improved Spanish question detection
"""

import pytest

from conftest import restore_punctuation

pytestmark = pytest.mark.core


# NOTE: Verb-first / implicit question cases were retired in v0.8.2 (accepted
# limitation — see AGENT.md "Question detection — verb-first / implicit questions
# (CLOSED)"). Production relies on Whisper's native punctuation; text-only
# detection of verb-first questions is inherently ambiguous. Only explicit
# question-word cases (which work reliably) remain here.
@pytest.mark.parametrize("text,description", [
    ("qué hora es la reunión mañana", "Basic question word (qué)"),
    ("cuándo es la cita", "Basic question word (cuándo)"),
    ("cómo estás hoy", "Basic question word (cómo)"),
    ("quién puede ayudarme", "Basic question word (quién)"),
    ("cuál es tu nombre", "Basic question word (cuál)"),
    ("sabes dónde queda", "Question pattern (sabes)"),
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
