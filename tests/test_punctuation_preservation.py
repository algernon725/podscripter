#!/usr/bin/env python3
"""
Test to verify that punctuation is preserved correctly in transcription output.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punctuation_restorer import restore_punctuation


def test_punctuation_preservation():
    """Test that punctuation is preserved correctly."""
    
    print("Testing punctuation preservation in transcription...")
    print("=" * 60)
    
    # Test cases that should have proper punctuation
    test_cases = [
        "Hola a todos",
        "Bienvenidos a Españolistos",
        "Españolistos es el podcast que te va a ayudar a estar listo para hablar español",
        "Españolistos te prepara para hablar español en cualquier lugar, a cualquier hora y en cualquier situación",
        "Recuerdas todos esos momentos en los que no supiste qué decir",
        "Esos momentos en los que no pudiste mantener una conversación",
        "Pues tranquilo",
        "Españolistos es la herramienta que estabas buscando para mejorar tu español"
    ]
    
    print("Testing individual sentences:")
    print("-" * 40)
    
    for text in test_cases:
        result = restore_punctuation(text, 'es')
        print(f"Input:  {text}")
        print(f"Output: {result}")
        print()
    
    # Test the full text as it would appear in transcription
    print("Testing full transcription text:")
    print("-" * 40)
    
    full_text = """
    Hola a todos

    Bienvenidos a Españolistos

    Españolistos es el podcast que te va a ayudar a estar listo para hablar español

    Españolistos te prepara para hablar español en cualquier lugar, a cualquier hora y en cualquier situación

    Recuerdas todos esos momentos en los que no supiste qué decir

    Esos momentos en los que no pudiste mantener una conversación

    Pues tranquilo

    Españolistos es la herramienta que estabas buscando para mejorar tu español
    """
    
    result = restore_punctuation(full_text, 'es')
    print(f"Input:  {full_text.strip()}")
    print(f"Output: {result}")
    
    # Test sentence splitting logic
    print("\nTesting sentence splitting logic:")
    print("-" * 40)
    
    # Simulate the sentence splitting logic from podscripter.py
    parts = result.split('\n\n')
    sentences = []
    
    for part in parts:
        part = part.strip()
        if part:
            # Split by sentence-ending punctuation but capture the punctuation
            sentence_parts = re.split(r'([.!?]+)', part)
            
            for i in range(0, len(sentence_parts), 2):
                if i < len(sentence_parts):
                    sentence_text = sentence_parts[i].strip()
                    punctuation = sentence_parts[i + 1] if i + 1 < len(sentence_parts) else ""
                    
                    if sentence_text:
                        # Combine sentence text with its punctuation
                        full_sentence = sentence_text + punctuation
                        
                        # Remove leading punctuation and whitespace
                        cleaned = re.sub(r'^[",\s]+', '', full_sentence)
                        
                        # Capitalize first letter if it's a letter
                        if cleaned and cleaned[0].isalpha():
                            cleaned = cleaned[0].upper() + cleaned[1:]
                        
                        if cleaned:
                            sentences.append(cleaned)
    
    print("Final sentences with punctuation:")
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. {sentence}")


if __name__ == "__main__":
    import re
    test_punctuation_preservation()
