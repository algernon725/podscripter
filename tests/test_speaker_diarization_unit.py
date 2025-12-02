#!/usr/bin/env python3
"""
Unit tests for speaker diarization utilities.

Tests the core functions of speaker_diarization.py without requiring
actual audio files or models.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from speaker_diarization import (
    _extract_speaker_boundaries,
    _merge_boundaries,
    SpeakerSegment,
    MIN_SPEAKER_SEGMENT_SEC,
    SPEAKER_BOUNDARY_EPSILON_SEC,
)


class TestExtractSpeakerBoundaries(unittest.TestCase):
    """Test extraction of speaker boundaries from segments."""
    
    def test_empty_segments(self):
        """Empty segments should return empty boundaries."""
        result = _extract_speaker_boundaries([])
        self.assertEqual(result, [])
    
    def test_single_segment(self):
        """Single segment should have no boundaries."""
        segments: list[SpeakerSegment] = [
            {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"}
        ]
        result = _extract_speaker_boundaries(segments)
        self.assertEqual(result, [])
    
    def test_same_speaker_no_boundary(self):
        """Segments with same speaker should not create boundaries."""
        segments: list[SpeakerSegment] = [
            {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"},
            {"start": 10.0, "end": 20.0, "speaker": "SPEAKER_00"},
            {"start": 20.0, "end": 30.0, "speaker": "SPEAKER_00"}
        ]
        result = _extract_speaker_boundaries(segments)
        self.assertEqual(result, [])
    
    def test_speaker_change_creates_boundary(self):
        """Speaker change should create boundary at end of first segment."""
        segments: list[SpeakerSegment] = [
            {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"},
            {"start": 10.0, "end": 20.0, "speaker": "SPEAKER_01"}
        ]
        result = _extract_speaker_boundaries(segments)
        self.assertEqual(result, [10.0])
    
    def test_multiple_speaker_changes(self):
        """Multiple speaker changes should create multiple boundaries."""
        segments: list[SpeakerSegment] = [
            {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"},
            {"start": 10.0, "end": 15.0, "speaker": "SPEAKER_01"},
            {"start": 15.0, "end": 25.0, "speaker": "SPEAKER_00"},
            {"start": 25.0, "end": 35.0, "speaker": "SPEAKER_02"}
        ]
        result = _extract_speaker_boundaries(segments)
        self.assertEqual(result, [10.0, 15.0, 25.0])
    
    def test_short_segments_filtered(self):
        """Very short segments (< MIN_SPEAKER_SEGMENT_SEC) should be filtered."""
        segments: list[SpeakerSegment] = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},  # Too short (1s)
            {"start": 1.0, "end": 4.0, "speaker": "SPEAKER_01"},  # Long enough (3s)
            {"start": 4.0, "end": 5.5, "speaker": "SPEAKER_00"}   # Too short (1.5s)
        ]
        result = _extract_speaker_boundaries(segments)
        # Only the boundary after the 3s segment should be included
        self.assertEqual(result, [4.0])
    
    def test_unsorted_segments(self):
        """Unsorted segments should be sorted before processing."""
        segments: list[SpeakerSegment] = [
            {"start": 20.0, "end": 30.0, "speaker": "SPEAKER_01"},
            {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"},
            {"start": 10.0, "end": 20.0, "speaker": "SPEAKER_01"}
        ]
        result = _extract_speaker_boundaries(segments)
        # Should detect boundary at 10.0 where speaker changes from 00 to 01
        self.assertEqual(result, [10.0])


class TestMergeBoundaries(unittest.TestCase):
    """Test merging of Whisper and speaker boundaries."""
    
    def test_both_none(self):
        """Both None should return empty list."""
        result = _merge_boundaries(None, None)
        self.assertEqual(result, [])
    
    def test_whisper_only(self):
        """Only Whisper boundaries should be returned sorted."""
        whisper = [15.0, 5.0, 10.0]
        result = _merge_boundaries(whisper, None)
        self.assertEqual(result, [5.0, 10.0, 15.0])
    
    def test_speaker_only(self):
        """Only speaker boundaries should be returned sorted."""
        speaker = [20.0, 10.0, 30.0]
        result = _merge_boundaries(None, speaker)
        self.assertEqual(result, [10.0, 20.0, 30.0])
    
    def test_no_overlap_merge(self):
        """Non-overlapping boundaries should all be included."""
        whisper = [5.0, 15.0, 25.0]
        speaker = [10.0, 20.0, 30.0]
        result = _merge_boundaries(whisper, speaker)
        self.assertEqual(result, [5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    
    def test_deduplication_within_epsilon(self):
        """Boundaries within epsilon should be deduplicated."""
        # Default epsilon is 1.0 second
        whisper = [10.0, 20.0]
        speaker = [10.5, 20.8]  # Within 1s of whisper boundaries
        result = _merge_boundaries(whisper, speaker, epsilon=1.0)
        # Should keep only the first in each cluster
        self.assertEqual(result, [10.0, 20.0])
    
    def test_deduplication_keeps_first(self):
        """Deduplication should keep first boundary in cluster."""
        whisper = [10.5]
        speaker = [10.0]  # Speaker comes first when sorted
        result = _merge_boundaries(whisper, speaker, epsilon=1.0)
        # Should keep 10.0 (speaker) as it comes first
        self.assertEqual(result, [10.0])
    
    def test_custom_epsilon(self):
        """Custom epsilon should be respected."""
        whisper = [10.0]
        speaker = [10.3]  # 0.3s apart
        # With epsilon=0.5, should deduplicate
        result1 = _merge_boundaries(whisper, speaker, epsilon=0.5)
        self.assertEqual(result1, [10.0])
        # With epsilon=0.2, should keep both
        result2 = _merge_boundaries(whisper, speaker, epsilon=0.2)
        self.assertEqual(result2, [10.0, 10.3])
    
    def test_multiple_clusters(self):
        """Multiple boundary clusters should each be deduplicated."""
        whisper = [5.0, 5.3, 15.0, 15.5]
        speaker = [5.2, 15.4]
        result = _merge_boundaries(whisper, speaker, epsilon=1.0)
        # Each cluster should reduce to one boundary
        self.assertEqual(len(result), 2)
        self.assertIn(5.0, result)  # First of first cluster
        self.assertIn(15.0, result)  # First of second cluster
    
    def test_empty_boundaries(self):
        """Empty boundary lists should be handled."""
        result = _merge_boundaries([], [])
        self.assertEqual(result, [])


class TestBoundaryPriority(unittest.TestCase):
    """Test that speaker boundaries have higher priority in merging."""
    
    def test_speaker_priority_when_close(self):
        """When boundaries are close, first one (sorted) is kept."""
        # This tests the implicit priority through deduplication
        whisper = [10.0]
        speaker = [10.2]  # Very close to whisper
        result = _merge_boundaries(whisper, speaker, epsilon=0.5)
        # Should keep only one, and it should be the first when sorted
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], 10.0)
    
    def test_both_boundaries_far_apart(self):
        """When boundaries are far apart, both should be kept."""
        whisper = [10.0]
        speaker = [15.0]  # Far from whisper
        result = _merge_boundaries(whisper, speaker, epsilon=1.0)
        self.assertEqual(result, [10.0, 15.0])


if __name__ == '__main__':
    unittest.main()

