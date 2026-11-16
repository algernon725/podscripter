#!/usr/bin/env python3
"""
Test multilingual question patterns across all supported languages
"""

import pytest

from conftest import restore_punctuation

pytestmark = pytest.mark.multilingual

# NOTE: Text-only question detection is an accepted limitation (#12): verb-first
# cases were retired in v0.8.2, the rest with the sentence-similarity model in
# v0.18.0 (#34). Production relies on Whisper's native punctuation, so the inputs
# carry Whisper's '?'; the cases check it survives alongside capitalization,
# Spanish '¿' and French clitic hyphenation.
_question_cases = [
    pytest.param("en", "can you help me with this?", "Can you help me with this?", "English request question", id="English request question"),
    pytest.param("es", "cómo estás hoy?", "¿Cómo estás hoy?", "Spanish basic question", id="Spanish basic question"),
    pytest.param("es", "qué hora es la reunión?", "¿Qué hora es la reunión?", "Spanish wh-question", id="Spanish wh-question"),
    pytest.param("fr", "pouvez vous m'aider avec ceci?", "Pouvez-vous m'aider avec ceci?", "French request question", id="French request question"),
    pytest.param("de", "kannst du mir dabei helfen?", "Kannst du mir dabei helfen?", "German request question", id="German request question"),
    pytest.param("pt", "você pode me ajudar com isso?", "Você pode me ajudar com isso?", "Portuguese request question", id="Portuguese request question"),
]


@pytest.mark.parametrize(
    "lang,input_text,expected,description",
    _question_cases,
)
def test_multilingual_question(lang, input_text, expected, description):
    """Test question patterns across multiple languages."""
    result = restore_punctuation(input_text, lang)
    assert result.strip() == expected.strip(), \
        f"{description}: expected {expected!r}, got {result!r}"
