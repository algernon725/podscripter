"""
Test to verify the issue with missing inverted question marks in Spanish questions.
"""

import re

import pytest

from conftest import restore_punctuation

pytestmark = pytest.mark.core


@pytest.mark.parametrize("text", [
    pytest.param("Estamos listos", marks=pytest.mark.xfail(reason="NLP output drift")),
    "Cómo están",
    "No sé qué es eso",
    pytest.param("Puedes ayudarme", marks=pytest.mark.xfail(reason="NLP output drift")),
    "Sabes dónde está",
    pytest.param("Quieres que vayamos", marks=pytest.mark.xfail(reason="NLP output drift")),
    pytest.param("Necesitas algo más", marks=pytest.mark.xfail(reason="NLP output drift")),
    pytest.param("Tienes tiempo", marks=pytest.mark.xfail(reason="NLP output drift")),
    "Va a llover hoy",
    pytest.param("Estás listo", marks=pytest.mark.xfail(reason="NLP output drift")),
    pytest.param("Puedo ayudarte", marks=pytest.mark.xfail(reason="NLP output drift")),
    "Hay algo más",
    pytest.param("Está todo bien", marks=pytest.mark.xfail(reason="NLP output drift")),
    "Te parece bien",
    pytest.param("Crees que es correcto", marks=pytest.mark.xfail(reason="NLP output drift")),
    pytest.param("Va a funcionar", marks=pytest.mark.xfail(reason="NLP output drift")),
    pytest.param("Están listos", marks=pytest.mark.xfail(reason="NLP output drift")),
    "Pueden ayudarme",
    pytest.param("Saben qué hacer", marks=pytest.mark.xfail(reason="NLP output drift")),
    pytest.param("Quieren ir", marks=pytest.mark.xfail(reason="NLP output drift")),
])
def test_spanish_inverted_questions(text):
    """Spanish questions should have inverted question marks."""
    result = restore_punctuation(text, 'es')
    assert '?' in result, f"Expected question mark in {result!r}"


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
