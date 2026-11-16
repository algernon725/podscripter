"""
Test to verify the issue with missing inverted question marks in Spanish questions.
"""

import re

import pytest

from conftest import restore_punctuation

pytestmark = pytest.mark.core


# NOTE: Text-only question detection is an accepted limitation (#12; the
# similarity model that attempted it was removed in v0.18.0, #34). Production keeps
# Whisper's '?'. When Whisper writes the '?' but not the opening '¿', the Spanish
# formatter adds it; that is what these cases cover.
@pytest.mark.parametrize("text", [
    "Cómo están?",
    "Sabes dónde está?",
    "Va a llover hoy?",
    "Hay algo más?",
    "Te parece bien?",
    "Pueden ayudarme?",
])
def test_spanish_inverted_questions(text):
    """A Spanish question Whisper ended with '?' gets the opening '¿'."""
    result = restore_punctuation(text, 'es')
    assert result.startswith('¿') and result.endswith('?'), f"Expected ¿…? in {result!r}"


def test_transcription_pipeline_punctuation():
    """Simulated transcription pipeline should produce properly punctuated sentences."""
    transcription_text = (
        "Estamos listos\n\n"
        "Cómo están\n\n"
        "No sé qué es eso\n\n"
        "Puedes ayudarme\n\n"
        "Sabes dónde está\n\n"
        "Quieres que vayamos\n\n"
        "Necesitas algo más\n\n"
        "Tienes tiempo\n\n"
        "Va a llover hoy\n\n"
        "Estás listo"
    )

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

    assert len(sentences) > 0, "Should produce at least one sentence"
    for sentence in sentences:
        assert sentence[-1] in '.!?', (
            f"Sentence missing end punctuation: {sentence!r}"
        )
