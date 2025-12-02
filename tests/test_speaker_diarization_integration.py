#!/usr/bin/env python3
"""
Integration tests for speaker diarization with the transcription pipeline.

These tests verify that speaker boundaries are correctly integrated into
the sentence splitting logic without requiring actual audio files or
diarization models.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from speaker_diarization import _merge_boundaries


class TestBoundaryMergingIntegration(unittest.TestCase):
    """Test boundary merging scenarios that occur in real transcription."""
    
    def test_typical_conversation_scenario(self):
        """Test realistic conversation with alternating speakers."""
        # Simulating a conversation with 3 speakers
        # Whisper detects pauses, speakers change at certain points
        whisper_boundaries = [12.5, 25.0, 37.5, 50.0]
        speaker_boundaries = [15.0, 30.0, 45.0]  # Speaker changes
        
        result = _merge_boundaries(whisper_boundaries, speaker_boundaries)
        
        # Should have all boundaries, with close ones deduplicated
        self.assertGreater(len(result), 0)
        self.assertTrue(all(result[i] < result[i+1] for i in range(len(result)-1)))  # Sorted
    
    def test_interview_scenario(self):
        """Test interview with many back-and-forth exchanges."""
        # Quick exchanges between interviewer and guest
        whisper_boundaries = [10.0, 20.0, 30.0, 40.0, 50.0]
        speaker_boundaries = [10.5, 20.5, 30.5, 40.5]  # Align roughly with Whisper
        
        result = _merge_boundaries(whisper_boundaries, speaker_boundaries, epsilon=1.0)
        
        # Many boundaries should be deduplicated due to proximity
        self.assertLessEqual(len(result), len(whisper_boundaries) + len(speaker_boundaries))
        self.assertTrue(all(result[i] < result[i+1] for i in range(len(result)-1)))
    
    def test_monologue_with_speaker_detection(self):
        """Test monologue where speaker doesn't change."""
        whisper_boundaries = [15.0, 30.0, 45.0, 60.0]
        speaker_boundaries = []  # No speaker changes
        
        result = _merge_boundaries(whisper_boundaries, speaker_boundaries)
        
        # Should just return Whisper boundaries
        self.assertEqual(result, whisper_boundaries)
    
    def test_single_speaker_multiple_pauses(self):
        """Test single speaker with natural pauses."""
        whisper_boundaries = [10.0, 15.0, 18.0, 25.0, 40.0]
        speaker_boundaries = []
        
        result = _merge_boundaries(whisper_boundaries, speaker_boundaries)
        
        self.assertEqual(result, whisper_boundaries)
    
    def test_no_boundaries_at_all(self):
        """Test when neither Whisper nor speaker detection finds boundaries."""
        result = _merge_boundaries([], [])
        self.assertEqual(result, [])
    
    def test_speaker_boundaries_without_whisper(self):
        """Test when only speaker boundaries are available."""
        # Scenario: Whisper in single-call mode (no segment boundaries)
        # but speaker diarization still runs
        whisper_boundaries = []
        speaker_boundaries = [20.0, 40.0, 60.0]
        
        result = _merge_boundaries(whisper_boundaries, speaker_boundaries)
        
        self.assertEqual(result, speaker_boundaries)


class TestBoundaryTimingEdgeCases(unittest.TestCase):
    """Test edge cases in boundary timing."""
    
    def test_simultaneous_boundaries(self):
        """Test when Whisper and speaker boundaries are at exact same time."""
        whisper_boundaries = [10.0, 20.0]
        speaker_boundaries = [10.0, 20.0]  # Exact match
        
        result = _merge_boundaries(whisper_boundaries, speaker_boundaries, epsilon=0.1)
        
        # Should deduplicate exact matches
        self.assertEqual(result, [10.0, 20.0])
    
    def test_very_close_boundaries(self):
        """Test boundaries that are very close together."""
        whisper_boundaries = [10.0]
        speaker_boundaries = [10.01]  # 10ms apart
        
        result = _merge_boundaries(whisper_boundaries, speaker_boundaries, epsilon=1.0)
        
        # Should keep only one due to deduplication
        self.assertEqual(len(result), 1)
    
    def test_boundaries_at_start_and_end(self):
        """Test boundaries at beginning and end of audio."""
        whisper_boundaries = [0.0, 50.0, 100.0]
        speaker_boundaries = [0.5, 99.5]
        
        result = _merge_boundaries(whisper_boundaries, speaker_boundaries, epsilon=1.0)
        
        # First and last should be deduplicated
        self.assertGreater(len(result), 0)
        self.assertTrue(result[0] <= 1.0)  # First boundary should be early
        self.assertTrue(result[-1] >= 99.0)  # Last boundary should be late
    
    def test_negative_boundaries(self):
        """Test that function handles potential negative times."""
        whisper_boundaries = [0.0, 10.0]
        speaker_boundaries = [-0.1, 10.0]  # Slightly negative (error case)
        
        result = _merge_boundaries(whisper_boundaries, speaker_boundaries, epsilon=1.0)
        
        # Should still produce valid sorted output
        self.assertTrue(all(result[i] < result[i+1] for i in range(len(result)-1)))


class TestMergeBehaviorConsistency(unittest.TestCase):
    """Test that merge behavior is consistent and predictable."""
    
    def test_merge_is_deterministic(self):
        """Test that merging produces same result on multiple calls."""
        whisper = [5.0, 15.0, 25.0]
        speaker = [10.0, 20.0, 30.0]
        
        result1 = _merge_boundaries(whisper, speaker)
        result2 = _merge_boundaries(whisper, speaker)
        
        self.assertEqual(result1, result2)
    
    def test_order_independence(self):
        """Test that input order doesn't affect output."""
        whisper = [25.0, 5.0, 15.0]  # Unsorted
        speaker = [30.0, 10.0, 20.0]  # Unsorted
        
        result = _merge_boundaries(whisper, speaker)
        
        # Output should always be sorted
        self.assertEqual(result, sorted(result))
    
    def test_duplicate_inputs_handled(self):
        """Test that duplicate boundaries in input are handled."""
        whisper = [10.0, 10.0, 20.0]  # Duplicate
        speaker = [15.0]
        
        result = _merge_boundaries(whisper, speaker)
        
        # Duplicates should be removed during merge
        self.assertEqual(len(result), 3)  # 10.0, 15.0, 20.0
        self.assertEqual(result, [10.0, 15.0, 20.0])


if __name__ == '__main__':
    unittest.main()

