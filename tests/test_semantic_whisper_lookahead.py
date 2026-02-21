#!/usr/bin/env python3
"""
Tests for the semantic-whisper lookahead fix.

When the semantic coherence check (PRIORITY 5) would fire, but a Whisper
boundary exists within the next N words, the semantic split should be
deferred so the higher-priority Whisper boundary can be evaluated at its
natural position.

Bug example fixed (Episodio243):
- "...necesitan saber para de verdad tomar." | "Su español al siguiente nivel."
  → should be: "...necesitan saber para de verdad tomar su español al siguiente nivel."

Root cause: at 42 words the chunk hit min_chunk_semantic_break and the
semantic model's 10-word lookahead window crossed a real sentence boundary,
producing a false-positive split 5 words before the Whisper boundary at "nivel."
"""

import unittest
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_splitter import SentenceSplitter


class MockConfig:
    """Mock config matching Spanish thresholds."""
    def __init__(self):
        self.thresholds = {
            'min_total_words_no_split': 30,
            'min_chunk_before_split': 20,
            'min_chunk_semantic_break': 42,
            'min_words_whisper_break': 10,
            'semantic_whisper_lookahead': 8,
        }


class _SentinelModel:
    """Non-None object so the splitter enters the semantic check branch."""
    pass


class TestSemanticWhisperLookahead(unittest.TestCase):
    """Verify that semantic splits are deferred when a Whisper boundary is nearby."""

    def setUp(self):
        self.config = MockConfig()

    def _build_long_spanish_text(self):
        """
        Reproduce the Episodio243 pattern: a 47-word sentence where the
        semantic threshold (42) is reached 5 words before the Whisper boundary.
        """
        return (
            "Ellos son Alberto y Judith. "
            "Ellos son dos de los profesores de Spanish 55 "
            "y los hemos invitado para que ellos nos cuenten sobre su experiencia "
            "y nos compartan cuáles son esos tips que ustedes estudiantes necesitan saber "
            "para de verdad tomar su español al siguiente nivel. "
            "Así que muchísimas gracias Judith por estar aquí hoy."
        )

    def _find_word_index(self, words, target):
        for i, w in enumerate(words):
            if w.rstrip('.,;:!?') == target.rstrip('.,;:!?'):
                return i
        return None

    def test_defers_semantic_split_when_whisper_boundary_nearby(self):
        """
        Even if the semantic model wants to split, the lookahead should
        prevent the split at 'tomar' because a Whisper boundary exists
        at 'nivel.' (5 words ahead, within the 8-word window).
        """
        splitter = SentenceSplitter('es', _SentinelModel(), self.config)
        text = self._build_long_spanish_text()
        words = text.split()

        tomar_idx = self._find_word_index(words, 'tomar')
        nivel_idx = self._find_word_index(words, 'nivel')
        self.assertIsNotNone(tomar_idx)
        self.assertIsNotNone(nivel_idx)

        whisper_word_boundaries = {nivel_idx}
        current_chunk = words[:tomar_idx + 1]

        self.assertGreaterEqual(
            len(current_chunk),
            self.config.thresholds['min_chunk_semantic_break'],
        )

        # _check_semantic_break should never even be called
        with patch.object(splitter, '_check_semantic_break', return_value=True) as mock_sem:
            result = splitter._should_end_sentence_here(
                words, tomar_idx, current_chunk,
                whisper_word_boundaries=whisper_word_boundaries,
                speaker_word_boundaries=None,
                speaker_word_segments=None,
            )
            mock_sem.assert_not_called()

        self.assertFalse(
            result,
            "Should NOT split at 'tomar' — Whisper boundary at 'nivel.' is within lookahead"
        )

    def test_allows_semantic_split_when_no_whisper_boundary_nearby(self):
        """
        When no Whisper boundary is within the lookahead window, the
        semantic model decision should be honoured.
        """
        splitter = SentenceSplitter('es', _SentinelModel(), self.config)
        text = self._build_long_spanish_text()
        words = text.split()

        tomar_idx = self._find_word_index(words, 'tomar')
        self.assertIsNotNone(tomar_idx)

        # Boundary far away (well beyond 8-word window)
        whisper_word_boundaries = {tomar_idx + 20}
        current_chunk = words[:tomar_idx + 1]

        with patch.object(splitter, '_check_semantic_break', return_value=True):
            result = splitter._should_end_sentence_here(
                words, tomar_idx, current_chunk,
                whisper_word_boundaries=whisper_word_boundaries,
                speaker_word_boundaries=None,
                speaker_word_segments=None,
            )

        self.assertTrue(
            result,
            "Should allow semantic split when no Whisper boundary is within lookahead"
        )

    def test_allows_semantic_split_when_no_whisper_boundaries_at_all(self):
        """
        When no Whisper boundaries are provided, semantic splitting works normally.
        """
        splitter = SentenceSplitter('es', _SentinelModel(), self.config)
        text = self._build_long_spanish_text()
        words = text.split()

        tomar_idx = self._find_word_index(words, 'tomar')
        self.assertIsNotNone(tomar_idx)
        current_chunk = words[:tomar_idx + 1]

        with patch.object(splitter, '_check_semantic_break', return_value=True):
            result = splitter._should_end_sentence_here(
                words, tomar_idx, current_chunk,
                whisper_word_boundaries=None,
                speaker_word_boundaries=None,
                speaker_word_segments=None,
            )

        self.assertTrue(
            result,
            "Should allow semantic split when Whisper boundaries are not available"
        )

    def test_lookahead_boundary_exactly_at_edge(self):
        """
        When the Whisper boundary is exactly at the edge of the lookahead
        window (8 words ahead), it should still defer.
        """
        splitter = SentenceSplitter('es', _SentinelModel(), self.config)
        text = self._build_long_spanish_text()
        words = text.split()

        tomar_idx = self._find_word_index(words, 'tomar')
        self.assertIsNotNone(tomar_idx)

        whisper_word_boundaries = {tomar_idx + 8}
        current_chunk = words[:tomar_idx + 1]

        with patch.object(splitter, '_check_semantic_break', return_value=True) as mock_sem:
            result = splitter._should_end_sentence_here(
                words, tomar_idx, current_chunk,
                whisper_word_boundaries=whisper_word_boundaries,
                speaker_word_boundaries=None,
                speaker_word_segments=None,
            )
            mock_sem.assert_not_called()

        self.assertFalse(
            result,
            "Should defer when Whisper boundary is exactly at lookahead edge"
        )

    def test_no_defer_when_boundary_just_beyond_window(self):
        """
        When the Whisper boundary is 9 words ahead (beyond the 8-word window),
        the semantic split should proceed.
        """
        splitter = SentenceSplitter('es', _SentinelModel(), self.config)
        text = self._build_long_spanish_text()
        words = text.split()

        tomar_idx = self._find_word_index(words, 'tomar')
        self.assertIsNotNone(tomar_idx)

        whisper_word_boundaries = {tomar_idx + 9}
        current_chunk = words[:tomar_idx + 1]

        with patch.object(splitter, '_check_semantic_break', return_value=True) as mock_sem:
            result = splitter._should_end_sentence_here(
                words, tomar_idx, current_chunk,
                whisper_word_boundaries=whisper_word_boundaries,
                speaker_word_boundaries=None,
                speaker_word_segments=None,
            )
            mock_sem.assert_called_once()

        self.assertTrue(
            result,
            "Should allow semantic split when Whisper boundary is beyond lookahead window"
        )

    def test_english_lookahead_also_works(self):
        """Cross-language: the lookahead should work for English too."""
        config = MockConfig()
        config.thresholds['min_chunk_semantic_break'] = 25
        splitter = SentenceSplitter('en', _SentinelModel(), config)

        words = (
            "They are two of the professors of Spanish 55 and we have invited "
            "them so they can tell us about their experience and share with us "
            "what are those tips that you students need to know to truly take "
            "your Spanish to the next level so thank you very much Judith"
        ).split()

        take_idx = self._find_word_index(words, 'take')
        level_idx = self._find_word_index(words, 'level')
        self.assertIsNotNone(take_idx)
        self.assertIsNotNone(level_idx)

        whisper_word_boundaries = {level_idx}
        current_chunk = words[:take_idx + 1]

        with patch.object(splitter, '_check_semantic_break', return_value=True) as mock_sem:
            result = splitter._should_end_sentence_here(
                words, take_idx, current_chunk,
                whisper_word_boundaries=whisper_word_boundaries,
                speaker_word_boundaries=None,
                speaker_word_segments=None,
            )
            mock_sem.assert_not_called()

        self.assertFalse(
            result,
            "English: should defer semantic split when Whisper boundary is nearby"
        )

    def test_semantic_model_says_no_split_still_no_split(self):
        """
        When the semantic model decides NOT to split (even with no lookahead
        deferral), the result should still be False.
        """
        splitter = SentenceSplitter('es', _SentinelModel(), self.config)
        text = self._build_long_spanish_text()
        words = text.split()

        tomar_idx = self._find_word_index(words, 'tomar')
        self.assertIsNotNone(tomar_idx)

        # Boundary far away
        whisper_word_boundaries = {tomar_idx + 20}
        current_chunk = words[:tomar_idx + 1]

        with patch.object(splitter, '_check_semantic_break', return_value=False):
            result = splitter._should_end_sentence_here(
                words, tomar_idx, current_chunk,
                whisper_word_boundaries=whisper_word_boundaries,
                speaker_word_boundaries=None,
                speaker_word_segments=None,
            )

        self.assertFalse(
            result,
            "Should not split when semantic model says no"
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
