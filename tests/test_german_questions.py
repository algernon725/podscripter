#!/usr/bin/env python3
"""
Test to verify German question detection and punctuation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from punctuation_restorer import restore_punctuation

def test_german_questions():
    """Test German questions that should have proper question marks."""
    
    print("Testing German question detection...")
    print("=" * 50)
    
    # Test cases that should have question marks
    test_cases = [
        "Sind wir bereit",
        "Wie geht es dir",
        "Ich weiß nicht was das ist",
        "Kannst du mir helfen",
        "Weißt du wo es ist",
        "Möchtest du gehen",
        "Brauchst du noch etwas",
        "Hast du Zeit",
        "Wird es heute regnen",
        "Bist du bereit",
        "Kann ich dir helfen",
        "Gibt es noch etwas",
        "Ist alles in Ordnung",
        "Scheint dir das gut",
        "Denkst du das ist richtig",
        "Wird das funktionieren",
        "Sind sie bereit",
        "Können sie mir helfen",
        "Wissen sie was zu tun ist",
        "Wollen sie gehen"
    ]
    
    print("Testing individual questions:")
    print("-" * 40)
    
    for text in test_cases:
        result = restore_punctuation(text, 'de')
        print(f"Input:  {text}")
        print(f"Output: {result}")
        print()
    
    # Test the transcription logic simulation
    print("\n" + "=" * 50)
    print("TESTING TRANSCRIPTION LOGIC SIMULATION")
    print("=" * 50)
    
    transcription_text = """
    Sind wir bereit

    Wie geht es dir

    Ich weiß nicht was das ist

    Kannst du mir helfen

    Weißt du wo es ist

    Möchtest du gehen

    Brauchst du noch etwas

    Hast du Zeit

    Wird es heute regnen

    Bist du bereit
    """
    
    # Simulate the transcription logic
    text_segments = [seg.strip() for seg in transcription_text.split('\n\n') if seg.strip()]
    sentences = []
    
    for segment in text_segments:
        processed_segment = restore_punctuation(segment, 'de')
        parts = re.split(r'([.!?]+)', processed_segment)
        
        for i in range(0, len(parts), 2):
            if i < len(parts):
                sentence_text = parts[i].strip()
                punctuation = parts[i + 1] if i + 1 < len(parts) else ""
                
                if sentence_text:
                    full_sentence = sentence_text + punctuation
                    cleaned = re.sub(r'^[",\s]+', '', full_sentence)
                    
                    if cleaned and cleaned[0].isalpha():
                        cleaned = cleaned[0].upper() + cleaned[1:]
                    
                    if cleaned:
                        if not cleaned.endswith(('.', '!', '?')):
                            cleaned += '.'
                        sentences.append(cleaned)
    
    print("Final sentences with punctuation:")
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. {sentence}")

if __name__ == "__main__":
    import re
    test_german_questions()
