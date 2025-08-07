#!/usr/bin/env python3
"""
Detailed test to debug the transcription logic step by step.
"""

import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punctuation_restorer import restore_punctuation


def test_transcription_debug():
    """Debug the transcription logic step by step."""
    
    print("Debugging transcription logic step by step...")
    print("=" * 60)
    
    # Test the exact text from the transcription simulation
    transcription_text = """
    Estamos listos

    Cómo están

    No sé qué es eso

    Puedes ayudarme

    Sabes dónde está

    Quieres que vayamos

    Necesitas algo más

    Tienes tiempo

    Va a llover hoy

    Estás listo
    """
    
    print("Step 1: Split into segments")
    print("-" * 30)
    text_segments = [seg.strip() for seg in transcription_text.split('\n\n') if seg.strip()]
    for i, seg in enumerate(text_segments, 1):
        print(f"{i}. '{seg}'")
    
    print("\nStep 2: Process each segment with punctuation restoration")
    print("-" * 60)
    
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
    print("FINAL RESULTS:")
    print("=" * 60)
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. {sentence}")


if __name__ == "__main__":
    test_transcription_debug()
