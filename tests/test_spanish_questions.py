"""
Comprehensive test for improved Spanish question detection
"""

import pytest

from conftest import restore_punctuation

pytestmark = pytest.mark.core


# NOTE: Verb-first / implicit question cases were retired in v0.8.2 (accepted
# limitation — see AGENTS.md "Question detection — verb-first / implicit questions
# (CLOSED)"). Production relies on Whisper's native punctuation; text-only
# detection of verb-first questions is inherently ambiguous. Only explicit
# question-word cases (which work reliably) remain here.
# NOTE: test_spanish_question_detected (12 unpunctuated wh-word cases) was retired
# in v0.18.0 with the sentence-similarity model (#34): text-only question detection
# joins the accepted limitation for verb-first questions (#12). Production keeps
# Whisper's own '?'. Most cases were copies of the model's seed sentences.
# The negative cases below guard against reintroducing a guess.


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
