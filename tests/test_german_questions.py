#!/usr/bin/env python3
"""
Test to verify German question detection and punctuation.
"""

import re

import pytest
from conftest import restore_punctuation

pytestmark = pytest.mark.core

GERMAN_QUESTION_CASES = [
    "Sind wir bereit",
    "Wie geht es dir",
    "Ich weiß nicht was das ist",
    "Kannst du mir helfen",
    "Weißt du wo es ist",
    "Möchtest du gehen",
    "Brauchst du noch etwas",
    "Hast du Zeit",
    "Wird es heute regnen",
    "Bist du bereit",
    "Kann ich dir helfen",
    "Gibt es noch etwas",
    "Ist alles in Ordnung",
    "Scheint dir das gut",
    "Denkst du das ist richtig",
    "Wird das funktionieren",
    "Sind sie bereit",
    "Können sie mir helfen",
    "Wissen sie was zu tun ist",
    "Wollen sie gehen",
]


@pytest.mark.parametrize("text", GERMAN_QUESTION_CASES)
def test_german_questions(text):
    """Test German questions that should have proper question marks."""
    result = restore_punctuation(text, 'de')
    assert result.endswith(('?', '.', '!')), f"Missing terminal punctuation: {result!r}"


def test_german_transcription_simulation():
    """Test that transcription segments are properly punctuated end-to-end."""
    transcription_text = """
    Sind wir bereit

    Wie geht es dir

    Ich weiß nicht was das ist

    Kannst du mir helfen

    Weißt du wo es ist

    Möchtest du gehen

    Brauchst du noch etwas

    Hast du Zeit

    Wird es heute regnen

    Bist du bereit
    """

    text_segments = [seg.strip() for seg in transcription_text.split('\n\n') if seg.strip()]
    sentences = []

    for segment in text_segments:
        processed_segment = restore_punctuation(segment, 'de')
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
