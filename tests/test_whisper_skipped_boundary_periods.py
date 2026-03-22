#!/usr/bin/env python3
"""
Test for "Whisper-Added Periods at Skipped Boundaries" known issue.

According to AGENT.md (lines 442-470), this issue occurs when:
1. A Whisper segment ends with a period (e.g., "ustedes.")
2. A speaker boundary is nearby (within 15 words), so the Whisper boundary is skipped
3. But the Whisper period remains, causing an unwanted split
4. Result: "...ustedes." | "Mateo 712." instead of "...ustedes Mateo 712."

With the v0.4.0 unified SentenceSplitter, this should potentially be resolved
since we now track punctuation provenance and can remove Whisper periods
at skipped boundaries.
"""

import pytest

from sentence_splitter import SentenceSplitter
from punctuation_restorer import _get_language_config

pytestmark = pytest.mark.core


class MockModel:
    """Mock SentenceTransformer model for testing."""
    def encode(self, texts):
        import numpy as np
        return np.random.rand(len(texts), 384)


def test_whisper_period_with_connector_removed():
    """
    Test the related case: Whisper period before same-speaker connector.
    This was the "trabajo. Y este meta" bug that v0.4.0 was supposed to fix.
    """
    text = "es importante tener una estructura como un trabajo. Y este meta es tu trabajo cada día."

    whisper_segments = [
        {'start': 10.0, 'end': 15.0, 'text': 'es importante tener una estructura como un trabajo.'},
        {'start': 15.5, 'end': 20.0, 'text': 'Y este meta es tu trabajo cada día.'},
    ]

    speaker_segments = [
        {'start_word': 0, 'end_word': 50, 'speaker': 'SPEAKER_01'},
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

    assert len(sentences) == 1, (
        f"Expected single sentence with connector merged, got {len(sentences)}: {sentences}"
    )
    first_text = sentences[0].text if hasattr(sentences[0], 'text') else sentences[0]
    assert ' y este meta' in first_text.lower(), (
        f"Expected lowercased 'y este' in merged sentence: {first_text}"
    )
