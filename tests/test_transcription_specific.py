#!/usr/bin/env python3
"""
Test the specific case in transcription simulation.
"""

import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punctuation_restorer import restore_punctuation


def test_transcription_specific():
    """Test the specific case in transcription simulation."""
    
    print("Testing specific case in transcription simulation...")
    print("=" * 60)
    
    # Simulate the exact text from transcription
    transcription_text = """
    Pudiste mantener una conversación
    """
    
    print("Step 1: Split into segments")
    print("-" * 30)
    text_segments = [seg.strip() for seg in transcription_text.split('\n\n') if seg.strip()]
    for i, seg in enumerate(text_segments, 1):
        print(f"{i}. '{seg}'")
    
    print("\nStep 2: Process each segment")
    print("-" * 40)
    
    sentences = []
    for i, segment in enumerate(text_segments, 1):
        print(f"\nSegment {i}: '{segment}'")
        
        # Step 2a: Process with punctuation restoration
        processed_segment = restore_punctuation(segment, 'es')
        print(f"  After punctuation restoration: '{processed_segment}'")
        
        # Step 2b: Split by punctuation
        parts = re.split(r'([.!?]+)', processed_segment)
        print(f"  Split parts: {parts}")
        
        # Step 2c: Process each part
        for j in range(0, len(parts), 2):
            if j < len(parts):
                sentence_text = parts[j].strip()
                punctuation = parts[j + 1] if j + 1 < len(parts) else ""
                
                print(f"    Part {j//2 + 1}: text='{sentence_text}', punctuation='{punctuation}'")
                
                if sentence_text:
                    # Combine sentence text with its punctuation
                    full_sentence = sentence_text + punctuation
                    print(f"      Combined: '{full_sentence}'")
                    
                    # Remove leading punctuation and whitespace
                    cleaned = re.sub(r'^[",\s]+', '', full_sentence)
                    print(f"      After cleaning: '{cleaned}'")
                    
                    # Capitalize first letter if it's a letter
                    if cleaned and cleaned[0].isalpha():
                        cleaned = cleaned[0].upper() + cleaned[1:]
                        print(f"      After capitalization: '{cleaned}'")
                    
                    if cleaned:
                        # Check if it needs punctuation
                        if not cleaned.endswith(('.', '!', '?')):
                            print(f"      → No punctuation, adding '.'")
                            cleaned += '.'
                        else:
                            print(f"      → Already has punctuation")
                        
                        sentences.append(cleaned)
                        print(f"      → Final sentence: '{cleaned}'")
    
    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    print("=" * 60)
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. {sentence}")
        
        # Check for the specific issue
        if sentence.endswith('?') and not sentence.startswith('¿'):
            print(f"   ❌ PROBLEM: Question without inverted question mark!")
        elif sentence.endswith('?') and sentence.startswith('¿'):
            print(f"   ✅ Correct: Question with inverted question mark")
        elif sentence.endswith('.'):
            print(f"   ✅ Correct: Statement with period")


if __name__ == "__main__":
    test_transcription_specific()
