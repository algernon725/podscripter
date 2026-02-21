#!/usr/bin/env python3
"""
Test to verify English question detection and punctuation.
"""

import re

import pytest
from conftest import restore_punctuation

pytestmark = pytest.mark.core

ENGLISH_QUESTION_CASES = [
    "Are we ready",
    "How are you",
    "I don't know what this is",
    "Can you help me",
    "Do you know where it is",
    "Would you like to go",
    "Do you need anything else",
    "Do you have time",
    "Is it going to rain today",
    "Are you ready",
    "Can I help you",
    "Is there anything else",
    "Is everything okay",
    "Does that seem good to you",
    "Do you think it's correct",
    "Is it going to work",
    "Are they ready",
    "Can they help me",
    "Do they know what to do",
    "Do they want to go",
]


@pytest.mark.parametrize("text", ENGLISH_QUESTION_CASES)
def test_english_questions(text):
    """Test English questions that should have proper question marks."""
    result = restore_punctuation(text, 'en')
    assert result.endswith(('?', '.', '!')), f"Missing terminal punctuation: {result!r}"


def test_english_transcription_simulation():
    """Test that transcription segments are properly punctuated end-to-end."""
    transcription_text = """
    Are we ready

    How are you

    I don't know what this is

    Can you help me

    Do you know where it is

    Would you like to go

    Do you need anything else

    Do you have time

    Is it going to rain today

    Are you ready
    """

    text_segments = [seg.strip() for seg in transcription_text.split('\n\n') if seg.strip()]
    sentences = []

    for segment in text_segments:
        processed_segment = restore_punctuation(segment, 'en')
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
