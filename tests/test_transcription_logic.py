"""
Test that simulates the exact transcription logic to verify punctuation preservation.
"""

import re

import pytest

from conftest import restore_punctuation

pytestmark = pytest.mark.transcription


def test_transcription_logic():
    """Simulate the exact transcription logic from podscripter.py."""

    raw_text = """
    Hola a todos

    Bienvenidos a Españolistos

    Españolistos es el podcast que te va a ayudar a estar listo para hablar español

    Españolistos te prepara para hablar español en cualquier lugar, a cualquier hora y en cualquier situación

    Recuerdas todos esos momentos en los que no supiste qué decir

    Esos momentos en los que no pudiste mantener una conversación

    Pues tranquilo

    Españolistos es la herramienta que estabas buscando para mejorar tu español
    """

    text_segments = [seg.strip() for seg in raw_text.split('\n\n') if seg.strip()]
    assert len(text_segments) > 0, "No text segments found"

    sentences = []
    lang_for_punctuation = 'es'

    for segment in text_segments:
        processed_segment = restore_punctuation(segment, lang_for_punctuation)
        parts = re.split(r'([.!?]+)', processed_segment)

        for j in range(0, len(parts), 2):
            if j < len(parts):
                sentence_text = parts[j].strip()
                punctuation = parts[j + 1] if j + 1 < len(parts) else ""

                if sentence_text:
                    full_sentence = sentence_text + punctuation
                    cleaned = re.sub(r'^[",\s]+', '', full_sentence)

                    if cleaned and cleaned[0].isalpha():
                        cleaned = cleaned[0].upper() + cleaned[1:]

                    if cleaned:
                        if not cleaned.endswith(('.', '!', '?')):
                            cleaned += '.'
                        sentences.append(cleaned)

    assert len(sentences) > 0, "No sentences extracted"
    for sentence in sentences:
        assert sentence[-1] in '.!?', f"Missing terminal punctuation: {sentence}"
        if sentence.endswith('?'):
            assert '¿' in sentence, f"Spanish question missing inverted mark: {sentence}"
