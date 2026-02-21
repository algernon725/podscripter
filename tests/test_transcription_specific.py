"""
Specific regression tests for Spanish intro formatting and exclamation handling
using a snippet from Episodio174 human-verified transcript.
"""

import re

import pytest

from conftest import restore_punctuation

pytestmark = pytest.mark.transcription


def normalize(text: str) -> str:
    t = re.sub(r"\s+", " ", text.strip())
    t = re.sub(r"\s+([,.!?])", r"\1", t)
    return t


def test_no_exclamation_plus_period():
    out = restore_punctuation("Hola a todos bienvenidos a Españolistos", "es")
    norm = normalize(out)
    assert "!." not in norm, f"Found forbidden sequence '!.': {norm}"


def test_greeting_and_question_formatting():
    out = normalize(restore_punctuation("hola para todos como estan", "es"))
    assert out.startswith("Hola para todos, ¿"), out
