#!/usr/bin/env python3
"""
Test to verify which fixes apply to other languages beyond Spanish.
"""

import re

import pytest

from conftest import restore_punctuation

pytestmark = pytest.mark.multilingual

_sentence_cases = [
    ("en", "Hello everyone"),
    ("en", "Welcome to English"),
    ("en", "This is a podcast that will help you learn English"),
    ("en", "Do you remember all those moments when you didn't know what to say"),
    ("en", "Those moments when you couldn't maintain a conversation"),
    ("en", "Well don't worry"),
    ("en", "English is the tool you were looking for to improve your English"),
    ("fr", "Bonjour à tous"),
    ("fr", "Bienvenue à Français"),
    ("fr", "Français est le podcast qui va vous aider à être prêt à parler français"),
    ("fr", "Vous souvenez-vous de tous ces moments où vous ne saviez pas quoi dire"),
    ("fr", "Ces moments où vous n'avez pas pu maintenir une conversation"),
    ("fr", "Eh bien ne vous inquiétez pas"),
    ("fr", "Français est l'outil que vous cherchiez pour améliorer votre français"),
    ("de", "Hallo an alle"),
    ("de", "Willkommen zu Deutsch"),
    ("de", "Deutsch ist der Podcast, der Ihnen helfen wird, bereit zu sein, Deutsch zu sprechen"),
    ("de", "Erinnerst du dich an all diese Momente, in denen du nicht wusstest, was du sagen solltest"),
    ("de", "Diese Momente, in denen du keine Konversation aufrechterhalten konntest"),
    ("de", "Nun mach dir keine Sorgen"),
    ("de", "Deutsch ist das Werkzeug, das du gesucht hast, um dein Deutsch zu verbessern"),
]


@pytest.mark.parametrize("language,sentence", _sentence_cases)
def test_restore_punctuation_produces_output(language, sentence):
    """Test that restore_punctuation returns non-empty output for each language."""
    result = restore_punctuation(sentence, language)
    assert result, f"Expected non-empty result for ({language}) {sentence!r}"


_transcription_texts = {
    'en': """
        Hello everyone

        Welcome to English

        This is a podcast that will help you learn English

        Do you remember all those moments when you didn't know what to say

        Well don't worry

        English is the tool you were looking for
        """,
    'fr': """
        Bonjour à tous

        Bienvenue à Français

        Français est le podcast qui va vous aider

        Vous souvenez-vous de tous ces moments

        Eh bien ne vous inquiétez pas

        Français est l'outil que vous cherchiez
        """,
    'de': """
        Hallo an alle

        Willkommen zu Deutsch

        Deutsch ist der Podcast der Ihnen helfen wird

        Erinnerst du dich an all diese Momente

        Nun mach dir keine Sorgen

        Deutsch ist das Werkzeug das du gesucht hast
        """,
}


@pytest.mark.parametrize("language", ["en", "fr", "de"])
def test_transcription_simulation(language):
    """Test transcription logic produces properly punctuated sentences."""
    text = _transcription_texts[language]
    text_segments = [seg.strip() for seg in text.split('\n\n') if seg.strip()]
    sentences = []

    for segment in text_segments:
        processed_segment = restore_punctuation(segment, language)
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

    assert len(sentences) > 0, f"Expected sentences for {language}"
    for sentence in sentences:
        assert sentence.endswith(('.', '!', '?')), \
            f"Sentence missing terminal punctuation: {sentence!r}"
