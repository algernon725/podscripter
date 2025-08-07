#!/usr/bin/env python3
"""
Test past tense questions to ensure the fix works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punctuation_restorer import restore_punctuation


def test_past_tense_questions():
    """Test various past tense questions."""
    
    print("Testing past tense questions...")
    print("=" * 50)
    
    # Test cases with past tense verbs
    test_cases = [
        "Pudiste mantener una conversación",
        "Supiste qué hacer",
        "Quisiste ir",
        "Necesitaste ayuda",
        "Tuviste tiempo",
        "Fuiste a la reunión",
        "Estuviste listo",
        "Pudieron ayudarte",
        "Supieron la respuesta",
        "Quisieron venir",
        "Necesitaron más información",
        "Tuvieron éxito",
        "Fueron al evento",
        "Estuvieron presentes"
    ]
    
    print("Testing individual questions:")
    print("-" * 40)
    
    for text in test_cases:
        result = restore_punctuation(text, 'es')
        print(f"Input:  {text}")
        print(f"Output: {result}")
        
        # Check if it's correctly formatted
        if result.endswith('?') and result.startswith('¿'):
            print(f"  ✅ Correct: Question with inverted question mark")
        elif result.endswith('?') and not result.startswith('¿'):
            print(f"  ❌ PROBLEM: Question without inverted question mark!")
        elif result.endswith('.'):
            print(f"  ✅ Correct: Statement with period")
        else:
            print(f"  ⚠️  No punctuation added")
        print()
    
    print("=" * 50)
    print("TRANSCRIPTION SIMULATION:")
    print("=" * 50)
    
    # Simulate transcription with multiple past tense questions
    transcription_text = """
    Pudiste mantener una conversación

    Supiste qué hacer

    Quisiste ir

    Necesitaste ayuda

    Tuviste tiempo
    """
    
    import re
    
    # Process as in transcription
    text_segments = [seg.strip() for seg in transcription_text.split('\n\n') if seg.strip()]
    sentences = []
    
    for segment in text_segments:
        processed_segment = restore_punctuation(segment, 'es')
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
        
        # Check for the specific issue
        if sentence.endswith('?') and not sentence.startswith('¿'):
            print(f"   ❌ PROBLEM: Question without inverted question mark!")
        elif sentence.endswith('?') and sentence.startswith('¿'):
            print(f"   ✅ Correct: Question with inverted question mark")
        elif sentence.endswith('.'):
            print(f"   ✅ Correct: Statement with period")


if __name__ == "__main__":
    test_past_tense_questions()
