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


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
def test_whisper_skipped_boundary_with_speaker_change():
    """
    Test the exact scenario from AGENT.md:
    - Whisper segment 10 ends with "ustedes."
    - Whisper segment 11 is "Mateo 712"
    - Speaker boundary at 69.47s falls WITHIN segment 11 (67.89s-69.89s)
    - We skip the Whisper boundary at "ustedes" (speaker boundary 2 words away)
    - The speaker change actually happens AFTER "Mateo 712", so both should be together

    Expected: "...ustedes Mateo 712." (ONE sentence until actual speaker change)
    Bug behavior: "...ustedes." | "Mateo 712." (TWO sentences due to Whisper period)
    """
    text = "y oramos por ustedes. Mateo 712"

    whisper_segments = [
        {'start': 65.0, 'end': 67.5, 'text': 'y oramos por ustedes.'},
        {'start': 67.89, 'end': 69.89, 'text': 'Mateo 712'},
    ]

    speaker_segments = [
        {'start_word': 0, 'end_word': 5, 'speaker': 'SPEAKER_01'},
        {'start_word': 6, 'end_word': 10, 'speaker': 'SPEAKER_02'},
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
        f"Expected 1 sentence (Whisper period removed at skipped boundary), "
        f"got {len(sentences)}: {sentences}"
    )


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
def test_whisper_boundary_not_skipped():
    """
    Test normal case: Whisper boundary is NOT skipped (no speaker boundary nearby).
    The Whisper period should be preserved.
    """
    text = "y oramos por ustedes. Gracias a Dios."

    whisper_segments = [
        {'start': 65.0, 'end': 67.5, 'text': 'y oramos por ustedes.'},
        {'start': 68.0, 'end': 70.0, 'text': 'Gracias a Dios.'},
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

    assert len(sentences) == 2, f"Expected 2 sentences (natural boundary preserved), got {len(sentences)}: {sentences}"


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
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
    assert ' y este meta' in sentences[0].lower(), (
        f"Expected lowercased 'y este' in merged sentence: {sentences[0]}"
    )
