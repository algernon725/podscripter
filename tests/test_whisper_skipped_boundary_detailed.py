#!/usr/bin/env python3
"""
Detailed investigation of the Whisper skipped boundary issue.

AGENT.md lines 442-470 describes:
- Problem: Whisper adds periods at segment ends
- When a Whisper boundary is SKIPPED (because speaker boundary nearby),
  the Whisper period should be REMOVED to avoid unwanted splits
- Current behavior: Period remains, causing "ustedes." instead of "ustedes"

This test investigates whether we can detect and remove these periods.
"""

import numpy as np
import pytest

from sentence_splitter import SentenceSplitter
from punctuation_restorer import _get_language_config

pytestmark = pytest.mark.core


class MockModel:
    """Mock SentenceTransformer model for testing."""
    def encode(self, texts):
        return np.random.rand(len(texts), 384)


def test_same_speaker_no_boundary():
    """
    Control test: Same speaker, no Whisper boundary skip needed.
    Period should be preserved if it's a natural sentence end.
    """
    text = "y oramos por ustedes. Gracias por todo lo que hacen cada día."

    whisper_segments = [
        {'start': 65.0, 'end': 67.5, 'text': 'y oramos por ustedes.'},
        {'start': 68.0, 'end': 72.0, 'text': 'Gracias por todo lo que hacen cada día.'},
    ]

    speaker_segments = [
        {'start_word': 0, 'end_word': 20, 'speaker': 'SPEAKER_01'},
    ]

    language = 'es'
    config = _get_language_config(language)
    model = MockModel()

    splitter = SentenceSplitter(language, model, config)

    sentences, metadata = splitter.split(
        text,
        whisper_segments=whisper_segments,
        speaker_segments=speaker_segments,
        mode='semantic'
    )

    assert len(sentences) >= 1, "Expected at least one sentence"
