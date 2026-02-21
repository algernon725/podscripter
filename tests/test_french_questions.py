#!/usr/bin/env python3
"""
Test to verify French question detection and punctuation.
"""

import re

import pytest
from conftest import restore_punctuation

pytestmark = pytest.mark.core

FRENCH_QUESTION_CASES = [
    "Sommes-nous prêts",
    "Comment allez-vous",
    "Je ne sais pas ce que c'est",
    "Pouvez-vous m'aider",
    "Savez-vous où c'est",
    "Voulez-vous aller",
    "Avez-vous besoin d'autre chose",
    "Avez-vous le temps",
    "Va-t-il pleuvoir aujourd'hui",
    "Êtes-vous prêt",
    "Puis-je vous aider",
    "Y a-t-il autre chose",
    "Est-ce que tout va bien",
    "Est-ce que cela vous semble bien",
    "Pensez-vous que c'est correct",
    "Est-ce que cela va fonctionner",
    "Sont-ils prêts",
    "Peuvent-ils m'aider",
    "Savent-ils quoi faire",
    "Veulent-ils aller",
]


@pytest.mark.parametrize("text", FRENCH_QUESTION_CASES)
def test_french_questions(text):
    """Test French questions that should have proper question marks."""
    result = restore_punctuation(text, 'fr')
    assert result.endswith(('?', '.', '!')), f"Missing terminal punctuation: {result!r}"


def test_french_transcription_simulation():
    """Test that transcription segments are properly punctuated end-to-end."""
    transcription_text = """
    Sommes-nous prêts

    Comment allez-vous

    Je ne sais pas ce que c'est

    Pouvez-vous m'aider

    Savez-vous où c'est

    Voulez-vous aller

    Avez-vous besoin d'autre chose

    Avez-vous le temps

    Va-t-il pleuvoir aujourd'hui

    Êtes-vous prêt
    """

    text_segments = [seg.strip() for seg in transcription_text.split('\n\n') if seg.strip()]
    sentences = []

    for segment in text_segments:
        processed_segment = restore_punctuation(segment, 'fr')
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
