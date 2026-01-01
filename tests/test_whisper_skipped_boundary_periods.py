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

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_splitter import SentenceSplitter
from punctuation_restorer import _get_language_config


class MockModel:
    """Mock SentenceTransformer model for testing."""
    def encode(self, texts):
        # Return dummy embeddings
        import numpy as np
        return np.random.rand(len(texts), 384)


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
    # Simulate the scenario
    # Text with Whisper-added periods at segment boundaries
    text = "y oramos por ustedes. Mateo 712"
    
    # Whisper segments (with periods added by Whisper)
    whisper_segments = [
        {'start': 65.0, 'end': 67.5, 'text': 'y oramos por ustedes.'},
        {'start': 67.89, 'end': 69.89, 'text': 'Mateo 712'},
    ]
    
    # Speaker segments: actual speaker change happens AFTER "Mateo 712"
    # Speaker 1 speaks through "ustedes Mateo 712", then Speaker 2 starts
    speaker_segments = [
        {'start_word': 0, 'end_word': 5, 'speaker': 'SPEAKER_01'},  # "y oramos por ustedes Mateo 712"
        {'start_word': 6, 'end_word': 10, 'speaker': 'SPEAKER_02'}, # (whatever comes next)
    ]
    
    # Initialize SentenceSplitter
    language = 'es'
    config = _get_language_config(language)
    model = MockModel()
    
    splitter = SentenceSplitter(language, model, config)
    
    # Run split
    sentences, metadata = splitter.split(
        text,
        whisper_segments=whisper_segments,
        speaker_segments=speaker_segments,
        mode='semantic'
    )
    
    print("="*70)
    print("TEST: Whisper Skipped Boundary with Speaker Change")
    print("="*70)
    print(f"Input text: '{text}'")
    print(f"Whisper segments: {len(whisper_segments)}")
    print(f"  Seg 0: {whisper_segments[0]}")
    print(f"  Seg 1: {whisper_segments[1]}")
    print(f"Speaker segments: {len(speaker_segments)}")
    print(f"  Spk 0: {speaker_segments[0]}")
    print(f"  Spk 1: {speaker_segments[1]}")
    print()
    print(f"Result: {len(sentences)} sentence(s)")
    for i, s in enumerate(sentences):
        print(f"  [{i}]: '{s}'")
    print()
    print(f"Removed periods: {metadata.get('removed_periods', [])}")
    print()
    
    # Check: Should be ONE sentence since speaker doesn't change until after "712"
    if len(sentences) == 1:
        print("✅ PASS: Single sentence as expected (Whisper period removed at skipped boundary)")
        print(f"   Result: '{sentences[0]}'")
        return True
    else:
        print("❌ ISSUE PERSISTS: Split into multiple sentences")
        print(f"   Expected: 'y oramos por ustedes Mateo 712' (1 sentence)")
        print(f"   Got: {sentences}")
        return False


def test_whisper_boundary_not_skipped():
    """
    Test normal case: Whisper boundary is NOT skipped (no speaker boundary nearby).
    The Whisper period should be preserved.
    """
    text = "y oramos por ustedes. Gracias a Dios."
    
    # Whisper segments with periods
    whisper_segments = [
        {'start': 65.0, 'end': 67.5, 'text': 'y oramos por ustedes.'},
        {'start': 68.0, 'end': 70.0, 'text': 'Gracias a Dios.'},
    ]
    
    # No speaker segments (or same speaker throughout)
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
    
    print("="*70)
    print("TEST: Whisper Boundary NOT Skipped (Normal Case)")
    print("="*70)
    print(f"Input text: '{text}'")
    print(f"Result: {len(sentences)} sentence(s)")
    for i, s in enumerate(sentences):
        print(f"  [{i}]: '{s}'")
    print()
    
    # Should be TWO sentences (natural sentence break)
    if len(sentences) == 2:
        print("✅ PASS: Two sentences as expected (Whisper periods preserved)")
        return True
    else:
        print("❌ UNEXPECTED: Expected 2 sentences")
        return False


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
    
    # Same speaker throughout
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
    
    print("="*70)
    print("TEST: Whisper Period Before Same-Speaker Connector")
    print("="*70)
    print(f"Input text: '{text}'")
    print(f"Result: {len(sentences)} sentence(s)")
    for i, s in enumerate(sentences):
        print(f"  [{i}]: '{s}'")
    print()
    print(f"Removed periods: {metadata.get('removed_periods', [])}")
    print()
    
    # Should be ONE sentence (connector "Y" should be merged and lowercased)
    # Expected: "es importante...trabajo y este meta es tu trabajo cada día."
    if len(sentences) == 1 and ' y este meta' in sentences[0].lower():
        print("✅ PASS: Period removed before same-speaker connector")
        print(f"   Connector lowercased: {' y este meta' in sentences[0]}")
        return True
    else:
        print("❌ ISSUE: Expected single sentence with lowercased 'y este'")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("INVESTIGATING: Whisper-Added Periods at Skipped Boundaries")
    print("="*70)
    print()
    print("This tests the known issue documented in AGENT.md lines 442-470.")
    print("With the v0.4.0 unified SentenceSplitter, we expect this to be resolved.")
    print()
    
    results = []
    
    # Test 1: The specific known issue
    results.append(("Skipped boundary with speaker change", test_whisper_skipped_boundary_with_speaker_change()))
    print()
    
    # Test 2: Normal case (boundary not skipped)
    results.append(("Normal boundary (not skipped)", test_whisper_boundary_not_skipped()))
    print()
    
    # Test 3: Related case (same-speaker connector)
    results.append(("Same-speaker connector", test_whisper_period_with_connector_removed()))
    print()
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    print()
    print(f"Passed: {passed}/{total}")
    print()
    
    if passed == total:
        print("🎉 All tests passed! The known issue appears to be RESOLVED.")
    else:
        print("⚠️  Some tests failed. The issue may still exist or need refinement.")
    
    sys.exit(0 if passed == total else 1)

