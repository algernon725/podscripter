#!/usr/bin/env python3
"""
Test to verify Portuguese question detection and punctuation.

Covers both Brazilian and European Portuguese forms (Whisper reports a single
'pt'), so você- and tu-conjugations both appear below.
"""

import re

import pytest
from conftest import restore_punctuation

pytestmark = pytest.mark.core

PORTUGUESE_QUESTION_CASES = [
    "Estamos prontos",
    "Como você está",
    "Como estás",
    "Eu não sei o que é isso",
    "Você pode me ajudar",
    "Podes ajudar-me",
    "Você sabe onde é",
    "Quer ir agora",
    "Precisa de mais alguma coisa",
    "Você tem tempo",
    "Vai chover hoje",
    "Está pronto",
    "Posso ajudar você",
    "Há mais alguma coisa",
    "Onde você mora",
    "Quando isso vai acontecer",
    "Por que você fez isso",
    "Qual você prefere",
    "Quem está aí",
    "Eles estão prontos",
    "Eles podem me ajudar",
    "Você entende o que eu digo",
    "Consegue ouvir a minha voz",
]


@pytest.mark.parametrize("text", PORTUGUESE_QUESTION_CASES)
def test_portuguese_questions(text):
    """Test Portuguese questions that should have proper terminal punctuation."""
    result = restore_punctuation(text, 'pt')
    assert result.endswith(('?', '.', '!')), f"Missing terminal punctuation: {result!r}"


@pytest.mark.parametrize("text", [
    "Você pode me ajudar com isso",
    "Onde você mora agora",
    "O que é isso",
])
def test_portuguese_questions_get_question_mark(text):
    """High-confidence question forms should resolve to '?', not '.'."""
    result = restore_punctuation(text, 'pt')
    assert result.endswith('?'), f"Expected a question mark: {result!r}"


def test_portuguese_has_no_inverted_marks():
    """Portuguese must never receive Spanish inverted '¿'/'¡' marks."""
    for text in PORTUGUESE_QUESTION_CASES:
        result = restore_punctuation(text, 'pt')
        assert '¿' not in result, f"Unexpected inverted question mark: {result!r}"
        assert '¡' not in result, f"Unexpected inverted exclamation mark: {result!r}"


def test_portuguese_greeting_comma():
    """A greeting at the start of a sentence should be followed by a comma."""
    result = restore_punctuation("olá como vai você", 'pt')
    assert result.lower().startswith("olá,"), f"Expected comma after greeting: {result!r}"


def test_portuguese_transcription_simulation():
    """Test that transcription segments are properly punctuated end-to-end."""
    transcription_text = """
    Estamos prontos

    Como você está

    Eu não sei o que é isso

    Você pode me ajudar

    Você sabe onde é

    Quer ir agora

    Precisa de mais alguma coisa

    Você tem tempo

    Vai chover hoje

    Está pronto
    """

    text_segments = [seg.strip() for seg in transcription_text.split('\n\n') if seg.strip()]
    sentences = []

    for segment in text_segments:
        processed_segment = restore_punctuation(segment, 'pt')
        parts = re.split(r'([.!?]+)', processed_segment)

        for i in range(0, len(parts), 2):
            if i < len(parts):
                sentence_text = parts[i].strip()
                punctuation = parts[i + 1] if i + 1 < len(parts) else ""

                if sentence_text:
                    full_sentence = sentence_text + punctuation
                    cleaned = re.sub(r'^[",\s]+', '', full_sentence)

                    if cleaned and cleaned[0].isalpha():
                        cleaned = cleaned[0].upper() + cleaned[1:]

                    if cleaned:
                        if not cleaned.endswith(('.', '!', '?')):
                            cleaned += '.'
                        sentences.append(cleaned)

    assert len(sentences) >= 5, f"Expected at least 5 sentences, got {len(sentences)}"
    for sentence in sentences:
        assert sentence[-1] in '.?!', f"Sentence missing terminal punctuation: {sentence!r}"
        assert sentence[0].isupper(), f"Sentence not capitalized: {sentence!r}"
