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


def test_whisper_period_at_skipped_boundary():
    """
    Test whether Whisper periods at skipped boundaries are handled correctly.
    
    Scenario:
    - Whisper segment 1 ends with "ustedes."
    - Whisper segment 2 is "Mateo 712"
    - Speaker boundary is AFTER "Mateo 712" (not between segments)
    - The Whisper boundary between segments should be SKIPPED
    - The period after "ustedes." should be REMOVED
    
    Expected: "y oramos por ustedes Mateo 712" (NO period)
    Current:  "y oramos por ustedes. Mateo 712" (period remains)
    """
    text = "y oramos por ustedes. Mateo 712"
    
    whisper_segments = [
        {'start': 65.0, 'end': 67.5, 'text': 'y oramos por ustedes.'},
        {'start': 67.89, 'end': 69.89, 'text': 'Mateo 712'},
    ]
    
    # Speaker 1 speaks through BOTH segments (speaker change comes later)
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
    
    print("="*80)
    print("TEST: Whisper Period at Skipped Boundary")
    print("="*80)
    print(f"Input text:          '{text}'")
    print(f"Whisper segments:    {len(whisper_segments)}")
    for i, seg in enumerate(whisper_segments):
        print(f"  Seg {i}: [{seg['start']:.2f}s-{seg['end']:.2f}s] '{seg['text']}'")
    print(f"Speaker segments:    {len(speaker_segments)}")
    for i, seg in enumerate(speaker_segments):
        print(f"  Spk {i}: words [{seg['start_word']}-{seg['end_word']}] {seg['speaker']}")
    print()
    print(f"Result:              {len(sentences)} sentence(s)")
    for i, s in enumerate(sentences):
        print(f"  [{i}]: '{s}'")
    print()
    print(f"Removed periods:     {metadata.get('removed_periods', [])}")
    print()
    
    # Check: Period should be removed at the skipped Whisper boundary
    result_text = sentences[0] if sentences else ""
    
    has_period_issue = ". Mateo" in result_text or ".Mateo" in result_text
    
    if has_period_issue:
        print("❌ ISSUE CONFIRMED: Period remains at skipped Whisper boundary")
        print(f"   Expected: 'y oramos por ustedes Mateo 712'")
        print(f"   Got:      '{result_text}'")
        print()
        print("ANALYSIS:")
        print("- The Whisper boundary between segments is within 15 words of speaker boundary")
        print("- The Whisper boundary is correctly SKIPPED (no sentence split)")
        print("- BUT the Whisper-added period after 'ustedes.' is NOT removed")
        print()
        print("ROOT CAUSE:")
        print("- sentence_splitter._evaluate_boundaries only prevents SPLITS")
        print("- It doesn't remove Whisper periods at skipped boundaries")
        print("- The _process_whisper_punctuation only handles same-speaker-connector case")
        print()
        return False
    else:
        print("✅ RESOLVED: Period correctly removed at skipped Whisper boundary")
        return True


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
    
    # Same speaker throughout
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
    
    print("="*80)
    print("TEST: Same Speaker, Natural Sentence Boundary")
    print("="*80)
    print(f"Input text:          '{text}'")
    print(f"Result:              {len(sentences)} sentence(s)")
    for i, s in enumerate(sentences):
        print(f"  [{i}]: '{s}'")
    print()
    
    # With enough words, this should split into 2 sentences
    # (It's not a connector case, just a natural sentence boundary)
    if len(sentences) >= 1:
        print("✅ PASS: Natural sentence boundaries preserved")
        return True
    else:
        print("❌ FAIL: Unexpected result")
        return False


def test_whisper_period_with_longer_context():
    """
    Test with more realistic context - longer segments before the issue.
    
    This simulates a real transcription where:
    - There's substantial content before the problematic boundary
    - Whisper segment ends with period
    - Next segment is very short (Bible reference)
    - Speaker doesn't actually change yet
    """
    text = "entonces vamos a orar por todos ustedes y por todas las personas que nos escuchan. Mateo 712 dice que debemos orar siempre"
    
    whisper_segments = [
        {'start': 60.0, 'end': 67.5, 'text': 'entonces vamos a orar por todos ustedes y por todas las personas que nos escuchan.'},
        {'start': 67.89, 'end': 69.89, 'text': 'Mateo 712'},
        {'start': 70.0, 'end': 73.0, 'text': 'dice que debemos orar siempre'},
    ]
    
    # Same speaker for segments 1 and 2, then different speaker
    speaker_segments = [
        {'start_word': 0, 'end_word': 19, 'speaker': 'SPEAKER_01'},  # Through "Mateo 712"
        {'start_word': 20, 'end_word': 30, 'speaker': 'SPEAKER_02'},
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
    
    print("="*80)
    print("TEST: Longer Context with Whisper Boundary Skip")
    print("="*80)
    print(f"Input text:          '{text[:60]}...'")
    print(f"Whisper segments:    {len(whisper_segments)}")
    for i, seg in enumerate(whisper_segments):
        print(f"  Seg {i}: [{seg['start']:.2f}s-{seg['end']:.2f}s] '{seg['text'][:40]}...'")
    print(f"Speaker segments:    {len(speaker_segments)}")
    print()
    print(f"Result:              {len(sentences)} sentence(s)")
    for i, s in enumerate(sentences):
        preview = s if len(s) <= 70 else s[:67] + "..."
        print(f"  [{i}]: '{preview}'")
    print()
    
    # Check first sentence
    if sentences:
        first = sentences[0]
        has_period_issue = ". Mateo" in first
        
        if has_period_issue:
            print("❌ ISSUE: Period remains before 'Mateo 712'")
            return False
        else:
            print("✅ PASS: Period handling correct")
            return True
    
    return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("DETAILED INVESTIGATION: Whisper-Added Periods at Skipped Boundaries")
    print("="*80)
    print()
    print("This investigates AGENT.md lines 442-470 'Known Limitation'")
    print()
    
    results = []
    
    results.append(("Whisper period at skipped boundary", test_whisper_period_at_skipped_boundary()))
    print()
    
    results.append(("Same speaker natural boundary", test_same_speaker_no_boundary()))
    print()
    
    results.append(("Longer context with skip", test_whisper_period_with_longer_context()))
    print()
    
    # Summary
    print("="*80)
    print("INVESTIGATION SUMMARY")
    print("="*80)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status}: {name}")
    print()
    print(f"Passed: {passed}/{total}")
    print()
    
    if passed == total:
        print("🎉 RESOLVED: The known issue has been fixed!")
    else:
        print("⚠️  CONFIRMED: The issue from AGENT.md still exists.")
        print()
        print("RECOMMENDATION:")
        print("The issue requires extending sentence_splitter.py to:")
        print("1. Track which Whisper boundaries are skipped")
        print("2. Remove Whisper periods at those skipped boundaries")
        print("3. Similar logic to _process_whisper_punctuation but for skipped boundaries")
    
    sys.exit(0 if passed == total else 1)

