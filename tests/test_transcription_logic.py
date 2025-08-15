#!/usr/bin/env python3
"""
Test that simulates the exact transcription logic to verify punctuation preservation.
"""

import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punctuation_restorer import restore_punctuation


def simulate_transcription_logic():
    """Simulate the exact transcription logic from podscripter.py."""
    
    print("Simulating transcription logic...")
    print("=" * 50)
    
    # Simulate the raw transcription text (as it would come from Whisper)
    raw_text = """
    Hola a todos

    Bienvenidos a Españolistos

    Españolistos es el podcast que te va a ayudar a estar listo para hablar español

    Españolistos te prepara para hablar español en cualquier lugar, a cualquier hora y en cualquier situación

    Recuerdas todos esos momentos en los que no supiste qué decir

    Esos momentos en los que no pudiste mantener una conversación

    Pues tranquilo

    Españolistos es la herramienta que estabas buscando para mejorar tu español
    """
    
    print("Raw transcription text:")
    print(raw_text.strip())
    print("\n" + "-" * 50)
    
    # Step 1: Split into individual segments (as in the new logic)
    print("Step 1: Split into individual segments")
    text_segments = [seg.strip() for seg in raw_text.split('\n\n') if seg.strip()]
    print(f"Found {len(text_segments)} segments:")
    for i, seg in enumerate(text_segments, 1):
        print(f"{i}. {seg}")
    print("\n" + "-" * 50)
    
    # Step 2: Process each segment individually
    print("Step 2: Process each segment individually")
    sentences = []
    lang_for_punctuation = 'es'
    
    for i, segment in enumerate(text_segments, 1):
        print(f"\nProcessing segment {i}: {segment}")
        
        # Process each segment individually for better punctuation
        processed_segment = restore_punctuation(segment, lang_for_punctuation)
        print(f"After processing: {processed_segment}")
        
        # Split the processed segment into sentences while preserving punctuation
        parts = re.split(r'([.!?]+)', processed_segment)
        
        for j in range(0, len(parts), 2):
            if j < len(parts):
                sentence_text = parts[j].strip()
                punctuation = parts[j + 1] if j + 1 < len(parts) else ""
                
                if sentence_text:
                    # Combine sentence text with its punctuation
                    full_sentence = sentence_text + punctuation
                    
                    # Remove leading punctuation and whitespace
                    cleaned = re.sub(r'^[",\s]+', '', full_sentence)
                    
                    # Capitalize first letter if it's a letter
                    if cleaned and cleaned[0].isalpha():
                        cleaned = cleaned[0].upper() + cleaned[1:]
                    
                    if cleaned:
                        # Ensure the sentence ends with punctuation (as in podscripter.py)
                        if not cleaned.endswith(('.', '!', '?')):
                            cleaned += '.'
                        sentences.append(cleaned)
                        print(f"  -> Extracted sentence: {cleaned}")
    
    print("\n" + "-" * 50)
    print("Final sentences with punctuation:")
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. {sentence}")


if __name__ == "__main__":
    simulate_transcription_logic()
