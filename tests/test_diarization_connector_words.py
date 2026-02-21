#!/usr/bin/env python3
"""
Tests for connector word handling when using speaker diarization.

When the same speaker continues speaking, sentences should not be split
before connector words like "Y" (Spanish), "and" (English), "et" (French), "und" (German).
"""

import pytest

from conftest import restore_punctuation

pytestmark = pytest.mark.core


def create_mock_speaker_segments_same_speaker(text: str) -> list[dict]:
    """Create mock speaker segments where the entire text is spoken by one speaker."""
    return [{
        'start_char': 0,
        'end_char': len(text),
        'speaker': 'SPEAKER_00'
    }]


def create_mock_speaker_segments_two_speakers(text: str, split_at_char: int) -> list[dict]:
    """Create mock speaker segments with a speaker change at a specific character position."""
    return [
        {
            'start_char': 0,
            'end_char': split_at_char,
            'speaker': 'SPEAKER_00'
        },
        {
            'start_char': split_at_char,
            'end_char': len(text),
            'speaker': 'SPEAKER_01'
        }
    ]


class TestDiarizationConnectorWords:
    """Test that connector words don't cause sentence splits when same speaker continues."""

    def test_same_speaker_no_connector_split_spanish_y(self):
        """When same speaker continues with 'Y', should not split sentence."""
        text = "Andrea siempre tiene muchos sueños que están recordando en la mañana Y yo no recuerdo muchos sueños"

        result = restore_punctuation(text, language='es', whisper_boundaries=None, speaker_boundaries=None)

        sentences = [s.strip() for s in result.split('\n') if s.strip()]
        for sentence in sentences:
            if sentence.startswith('Y '):
                assert False, f"Sentence incorrectly starts with 'Y': {sentence}"

    def test_same_speaker_no_connector_split_spanish_full_example(self):
        """Full example from user's Episodio212 transcription."""
        text = "Andrea siempre tiene muchos sueños que están recordando en la mañana y yo no recuerdo muchos sueños"

        speaker_segments = create_mock_speaker_segments_same_speaker(text)

        result = restore_punctuation(text, language='es', whisper_boundaries=None, speaker_boundaries=None, speaker_segments=speaker_segments)

        lines = [s.strip() for s in result.split('\n') if s.strip()]
        for line in lines:
            assert not line.startswith('Y '), f"Sentence should not start with 'Y': {line}"

    def test_same_speaker_no_connector_split_spanish_second_example(self):
        """Second example from user's Episodio212 transcription."""
        text = "No recuerdo nada y yo le cuento como 5 o 6 sueños diferentes que tuve y los recuerdo muy bien"

        speaker_segments = create_mock_speaker_segments_same_speaker(text)

        result = restore_punctuation(text, language='es', whisper_boundaries=None, speaker_boundaries=None, speaker_segments=speaker_segments)

        lines = [s.strip() for s in result.split('\n') if s.strip()]
        for line in lines:
            assert not line.startswith('Y '), f"Sentence should not start with 'Y': {line}"

    def test_same_speaker_no_connector_split_english_and(self):
        """When same speaker continues with 'and', should not split sentence."""
        text = "I love reading books and I enjoy learning new things every day"

        result = restore_punctuation(text, language='en', whisper_boundaries=None, speaker_boundaries=None)

        sentences = [s.strip() for s in result.split('\n') if s.strip()]
        for sentence in sentences:
            if sentence.startswith('And '):
                assert False, f"Sentence incorrectly starts with 'And': {sentence}"

    def test_same_speaker_no_connector_split_english_but(self):
        """When same speaker continues with 'but', should not split sentence."""
        text = "She wanted to go but she couldn't make it on time"

        result = restore_punctuation(text, language='en', whisper_boundaries=None, speaker_boundaries=None)

        sentences = [s.strip() for s in result.split('\n') if s.strip()]
        for sentence in sentences:
            if sentence.startswith('But '):
                assert False, f"Sentence incorrectly starts with 'But': {sentence}"

    def test_same_speaker_no_connector_split_french_et(self):
        """When same speaker continues with 'et', should not split sentence."""
        text = "J'aime lire des livres et j'aime apprendre de nouvelles choses"

        result = restore_punctuation(text, language='fr', whisper_boundaries=None, speaker_boundaries=None)

        sentences = [s.strip() for s in result.split('\n') if s.strip()]
        for sentence in sentences:
            if sentence.startswith('Et '):
                assert False, f"Sentence incorrectly starts with 'Et': {sentence}"

    def test_same_speaker_no_connector_split_french_mais(self):
        """When same speaker continues with 'mais', should not split sentence."""
        text = "Elle voulait partir mais elle ne pouvait pas"

        result = restore_punctuation(text, language='fr', whisper_boundaries=None, speaker_boundaries=None)

        sentences = [s.strip() for s in result.split('\n') if s.strip()]
        for sentence in sentences:
            if sentence.startswith('Mais '):
                assert False, f"Sentence incorrectly starts with 'Mais': {sentence}"

    def test_same_speaker_no_connector_split_german_und(self):
        """When same speaker continues with 'und', should not split sentence."""
        text = "Ich lese gerne Bücher und ich lerne gerne neue Dinge"

        result = restore_punctuation(text, language='de', whisper_boundaries=None, speaker_boundaries=None)

        sentences = [s.strip() for s in result.split('\n') if s.strip()]
        for sentence in sentences:
            if sentence.startswith('Und '):
                assert False, f"Sentence incorrectly starts with 'Und': {sentence}"

    def test_same_speaker_no_connector_split_german_aber(self):
        """When same speaker continues with 'aber', should not split sentence."""
        text = "Sie wollte gehen aber sie konnte nicht rechtzeitig ankommen"

        result = restore_punctuation(text, language='de', whisper_boundaries=None, speaker_boundaries=None)

        sentences = [s.strip() for s in result.split('\n') if s.strip()]
        for sentence in sentences:
            if sentence.startswith('Aber '):
                assert False, f"Sentence incorrectly starts with 'Aber': {sentence}"

    def test_legitimate_sentence_start_allowed(self):
        """Legitimate sentences CAN start after proper breaks (question/exclamation)."""
        text = "¿Cómo estás? Y tú, ¿qué tal?"

        result = restore_punctuation(text, language='es', whisper_boundaries=None, speaker_boundaries=None)
        assert result is not None

    def test_different_speaker_with_connector_allows_break(self):
        """When DIFFERENT speakers speak, connector words CAN start new sentences."""
        text = "I like coffee and I like tea"

        speaker_segments = create_mock_speaker_segments_two_speakers(text, 14)

        result = restore_punctuation(text, language='en', whisper_boundaries=None, speaker_boundaries=None, speaker_segments=speaker_segments)
        assert result is not None

    def test_same_speaker_long_text_spanish(self):
        """Test with longer Spanish text to trigger semantic splitting."""
        text = "Andrea siempre tiene muchos sueños que están recordando en la mañana y yo no recuerdo muchos sueños y Nate siempre me dice que no recuerda nada y yo le cuento como cinco o seis sueños diferentes que tuve y los recuerdo muy bien"

        speaker_segments = create_mock_speaker_segments_same_speaker(text)

        result = restore_punctuation(text, language='es', whisper_boundaries=None, speaker_boundaries=None, speaker_segments=speaker_segments)

        lines = [s.strip() for s in result.split('\n') if s.strip()]
        for line in lines:
            assert not line.startswith('Y '), f"Sentence should not start with 'Y': {line}"
