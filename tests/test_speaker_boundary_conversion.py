#!/usr/bin/env python3
"""
Tests for speaker boundary timestamp to character position conversion.

This tests the fix for the unit mismatch bug where speaker boundaries (timestamps in seconds)
were incorrectly merged with Whisper boundaries (character positions).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from podscripter import _convert_speaker_timestamps_to_char_positions


class TestSpeakerBoundaryConversion:
    """Tests for _convert_speaker_timestamps_to_char_positions."""
    
    def test_basic_conversion(self):
        """Test basic timestamp to character position conversion."""
        # Simulate Whisper segments with timing info
        whisper_segments = [
            {'start': 0.0, 'end': 5.0, 'text': 'Hello world'},       # 11 chars
            {'start': 5.0, 'end': 10.0, 'text': 'How are you'},      # 11 chars
            {'start': 10.0, 'end': 15.0, 'text': 'I am fine'},       # 9 chars
        ]
        text = "Hello world How are you I am fine"  # Concatenated with spaces
        
        # Speaker boundary at 5.0s should map to end of first segment
        speaker_boundaries = [5.0]
        result = _convert_speaker_timestamps_to_char_positions(
            speaker_boundaries, whisper_segments, text
        )
        
        # First segment "Hello world" = 11 chars, so char position should be 11
        assert len(result) == 1
        assert result[0] == 11
    
    def test_boundary_within_segment(self):
        """Test boundary that falls within a segment."""
        whisper_segments = [
            {'start': 0.0, 'end': 10.0, 'text': 'First segment text'},
            {'start': 10.0, 'end': 20.0, 'text': 'Second segment text'},
        ]
        text = "First segment text Second segment text"
        
        # Boundary at 5.0s falls within first segment (0-10s)
        speaker_boundaries = [5.0]
        result = _convert_speaker_timestamps_to_char_positions(
            speaker_boundaries, whisper_segments, text
        )
        
        # Should map to end of first segment (18 chars)
        assert len(result) == 1
        assert result[0] == 18
    
    def test_boundary_in_gap(self):
        """Test boundary that falls in a gap between segments."""
        whisper_segments = [
            {'start': 0.0, 'end': 5.0, 'text': 'First'},
            {'start': 7.0, 'end': 12.0, 'text': 'Second'},  # Gap from 5-7s
        ]
        text = "First Second"
        
        # Boundary at 6.0s is in the gap, should map to end of first segment
        speaker_boundaries = [6.0]
        result = _convert_speaker_timestamps_to_char_positions(
            speaker_boundaries, whisper_segments, text
        )
        
        assert len(result) == 1
        assert result[0] == 5  # "First" = 5 chars
    
    def test_multiple_boundaries(self):
        """Test conversion of multiple speaker boundaries."""
        whisper_segments = [
            {'start': 0.0, 'end': 10.0, 'text': 'Segment one'},
            {'start': 10.0, 'end': 20.0, 'text': 'Segment two'},
            {'start': 20.0, 'end': 30.0, 'text': 'Segment three'},
        ]
        text = "Segment one Segment two Segment three"
        
        speaker_boundaries = [10.0, 20.0]
        result = _convert_speaker_timestamps_to_char_positions(
            speaker_boundaries, whisper_segments, text
        )
        
        assert len(result) == 2
        # "Segment one" = 11 chars
        assert result[0] == 11
        # "Segment one" (11) + separator (1) + "Segment two" (11) = 23
        assert result[1] == 23
    
    def test_empty_inputs(self):
        """Test with empty inputs."""
        assert _convert_speaker_timestamps_to_char_positions([], [], "") == []
        assert _convert_speaker_timestamps_to_char_positions([1.0], [], "") == []
        assert _convert_speaker_timestamps_to_char_positions([], [{'start': 0, 'end': 1, 'text': 'test'}], "test") == []
    
    def test_removes_duplicates(self):
        """Test that duplicate char positions are removed."""
        whisper_segments = [
            {'start': 0.0, 'end': 10.0, 'text': 'Only segment'},
        ]
        text = "Only segment"
        
        # Multiple boundaries that map to the same segment
        speaker_boundaries = [5.0, 7.0, 9.0]  # All within 0-10s
        result = _convert_speaker_timestamps_to_char_positions(
            speaker_boundaries, whisper_segments, text
        )
        
        # Should all map to same position and be deduplicated
        assert len(result) == 1
    
    def test_realistic_scenario(self):
        """Test a realistic scenario similar to the reported bug."""
        # Simulating the Mateo 712 scenario
        whisper_segments = [
            {'start': 1279.18, 'end': 1284.18, 'text': 'Ama a tu prójimo como a ti mismo.'},
            {'start': 1284.18, 'end': 1289.18, 'text': 'Y también hay otro versículo que me gusta muchísimo que dice'},
            {'start': 1289.18, 'end': 1294.18, 'text': 'Así que en todo, traten ustedes a los demás'},
            {'start': 1294.18, 'end': 1299.18, 'text': 'tal y como quieren que ellos los traten a ustedes.'},
            {'start': 1299.18, 'end': 1301.18, 'text': 'Mateo 712'},
            {'start': 1301.18, 'end': 1309.18, 'text': 'Bueno, Andrea, creo que tú puedes explicar este tema muchísimo mejor que yo'},
        ]
        
        # Build concatenated text
        text = " ".join(seg['text'] for seg in whisper_segments)
        
        # Speaker changes at 1300.60s (after "Mateo 712", before "Bueno, Andrea")
        speaker_boundaries = [1300.60]
        result = _convert_speaker_timestamps_to_char_positions(
            speaker_boundaries, whisper_segments, text
        )
        
        # Should map to end of "Mateo 712" segment (segment index 4)
        assert len(result) == 1
        
        # Calculate expected position:
        # Sum of first 5 segments' text lengths plus separators
        expected_pos = 0
        for i, seg in enumerate(whisper_segments[:5]):  # First 5 segments
            expected_pos += len(seg['text'])
            if i < 4:  # Add separator between segments
                expected_pos += 1
        
        assert result[0] == expected_pos, f"Expected {expected_pos}, got {result[0]}"
        
        # Verify this position is in the right place in the text
        # The text at this position should be right after "Mateo 712"
        text_before = text[:result[0]]
        assert text_before.endswith("Mateo 712"), f"Expected text before boundary to end with 'Mateo 712', got: ...{text_before[-20:]}"


def test_conversion_basic():
    """Simple test that can be run standalone."""
    t = TestSpeakerBoundaryConversion()
    t.test_basic_conversion()
    t.test_boundary_within_segment()
    t.test_boundary_in_gap()
    t.test_multiple_boundaries()
    t.test_empty_inputs()
    t.test_removes_duplicates()
    t.test_realistic_scenario()
    print("All speaker boundary conversion tests passed!")


if __name__ == "__main__":
    test_conversion_basic()
