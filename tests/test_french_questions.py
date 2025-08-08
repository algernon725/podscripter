#!/usr/bin/env python3
"""
Test to verify French question detection and punctuation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from punctuation_restorer import restore_punctuation

def test_french_questions():
    """Test French questions that should have proper question marks."""
    
    print("Testing French question detection...")
    print("=" * 50)
    
    # Test cases that should have question marks
    test_cases = [
        "Sommes-nous prêts",
        "Comment allez-vous",
        "Je ne sais pas ce que c'est",
        "Pouvez-vous m'aider",
        "Savez-vous où c'est",
        "Voulez-vous aller",
        "Avez-vous besoin d'autre chose",
        "Avez-vous le temps",
        "Va-t-il pleuvoir aujourd'hui",
        "Êtes-vous prêt",
        "Puis-je vous aider",
        "Y a-t-il autre chose",
        "Est-ce que tout va bien",
        "Est-ce que cela vous semble bien",
        "Pensez-vous que c'est correct",
        "Est-ce que cela va fonctionner",
        "Sont-ils prêts",
        "Peuvent-ils m'aider",
        "Savent-ils quoi faire",
        "Veulent-ils aller"
    ]
    
    print("Testing individual questions:")
    print("-" * 40)
    
    for text in test_cases:
        result = restore_punctuation(text, 'fr')
        print(f"Input:  {text}")
        print(f"Output: {result}")
        print()
    
    # Test the transcription logic simulation
    print("\n" + "=" * 50)
    print("TESTING TRANSCRIPTION LOGIC SIMULATION")
    print("=" * 50)
    
    transcription_text = """
    Sommes-nous prêts

    Comment allez-vous

    Je ne sais pas ce que c'est

    Pouvez-vous m'aider

    Savez-vous où c'est

    Voulez-vous aller

    Avez-vous besoin d'autre chose

    Avez-vous le temps

    Va-t-il pleuvoir aujourd'hui

    Êtes-vous prêt
    """
    
    # Simulate the transcription logic
    text_segments = [seg.strip() for seg in transcription_text.split('\n\n') if seg.strip()]
    sentences = []
    
    for segment in text_segments:
        processed_segment = restore_punctuation(segment, 'fr')
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
    test_french_questions()
