#!/usr/bin/env python3
"""
Debug script to understand Whisper boundary word index calculation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_splitter import SentenceSplitter
from punctuation_restorer import _get_language_config
import numpy as np


class MockModel:
    """Mock SentenceTransformer model for testing."""
    def encode(self, texts):
        return np.random.rand(len(texts), 384)


# Test data
text = "y oramos por ustedes. Mateo 712"
whisper_segments = [
    {'start': 65.0, 'end': 67.5, 'text': 'y oramos por ustedes.'},
    {'start': 67.89, 'end': 69.89, 'text': 'Mateo 712'},
]
speaker_segments = [
    {'start_word': 0, 'end_word': 5, 'speaker': 'SPEAKER_01'},
    {'start_word': 6, 'end_word': 10, 'speaker': 'SPEAKER_02'},
]

print("="*80)
print("DEBUG: Whisper Boundary Word Index Calculation")
print("="*80)
print(f"Text: '{text}'")
print(f"Words: {text.split()}")
print()
print("Whisper segments:")
for i, seg in enumerate(whisper_segments):
    print(f"  Seg {i}: '{seg['text']}'")
print()

# Initialize splitter
language = 'es'
config = _get_language_config(language)
model = MockModel()
splitter = SentenceSplitter(language, model, config)

# Convert segments to word boundaries
whisper_word_boundaries = splitter._convert_segments_to_word_boundaries(whisper_segments, text)
speaker_word_boundaries = splitter._convert_segments_to_word_boundaries(speaker_segments, text, is_speaker=True)

print(f"Whisper word boundaries: {whisper_word_boundaries}")
print(f"Speaker word boundaries: {speaker_word_boundaries}")
print()

# Build char to word map to understand the conversion
char_to_word = splitter._build_char_to_word_map(text)
print("Character to word mapping (sample):")
for char_pos in range(min(40, len(text))):
    if char_pos in char_to_word:
        print(f"  char {char_pos:2d} ('{text[char_pos]}') → word {char_to_word[char_pos]}")
print()

# Calculate expected boundary positions
print("Expected Whisper boundary calculation:")
seg0_text = whisper_segments[0]['text']
print(f"  Segment 0 text: '{seg0_text}'")
print(f"  Segment 0 length: {len(seg0_text)} chars")
print(f"  Segment 0 ends at char position: {len(seg0_text) - 1} (0-indexed)")
print(f"  That char position maps to word: {char_to_word.get(len(seg0_text) - 1, 'NOT FOUND')}")
print()

# Now run the actual split to see what happens
sentences, metadata = splitter.split(
    text,
    whisper_segments=whisper_segments,
    speaker_segments=speaker_segments,
    mode='semantic'
)

print("Results:")
print(f"  Sentences: {sentences}")
print(f"  Skipped boundaries: {splitter.skipped_whisper_boundaries}")
print(f"  Removed periods: {metadata.get('removed_periods', [])}")

