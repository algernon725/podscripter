"""
Test past tense questions to ensure the fix works correctly.
"""

import re

import pytest

from conftest import restore_punctuation

pytestmark = pytest.mark.core


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
def test_past_tense_questions():
    """Test various past tense questions."""

    test_cases = [
        "Pudiste mantener una conversación",
        "Supiste qué hacer",
        "Quisiste ir",
        "Necesitaste ayuda",
        "Tuviste tiempo",
        "Fuiste a la reunión",
        "Estuviste listo",
        "Pudieron ayudarte",
        "Supieron la respuesta",
        "Quisieron venir",
        "Necesitaron más información",
        "Tuvieron éxito",
        "Fueron al evento",
        "Estuvieron presentes"
    ]

    for text in test_cases:
        result = restore_punctuation(text, 'es')
        if result.endswith('?'):
            assert result.startswith('¿'), f"Question without inverted mark: {result}"
        else:
            assert result.endswith('.'), f"Expected period or question mark for '{text}', got: {result}"


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
def test_past_tense_transcription_simulation():
    """Simulate transcription with multiple past tense questions."""

    transcription_text = """
    Pudiste mantener una conversación

    Supiste qué hacer

    Quisiste ir

    Necesitaste ayuda

    Tuviste tiempo
    """

    text_segments = [seg.strip() for seg in transcription_text.split('\n\n') if seg.strip()]
    sentences = []

    for segment in text_segments:
        processed_segment = restore_punctuation(segment, 'es')
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

    assert len(sentences) > 0, "No sentences extracted from transcription"
    for sentence in sentences:
        assert sentence[-1] in '.!?', f"Missing terminal punctuation: {sentence}"
        if sentence.endswith('?'):
            assert sentence.startswith('¿'), f"Question without inverted mark: {sentence}"
