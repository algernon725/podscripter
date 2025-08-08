#!/usr/bin/env python3
"""
Test to verify English question detection and punctuation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from punctuation_restorer import restore_punctuation

def test_english_questions():
    """Test English questions that should have proper question marks."""
    
    print("Testing English question detection...")
    print("=" * 50)
    
    # Test cases that should have question marks
    test_cases = [
        "Are we ready",
        "How are you",
        "I don't know what this is",
        "Can you help me",
        "Do you know where it is",
        "Would you like to go",
        "Do you need anything else",
        "Do you have time",
        "Is it going to rain today",
        "Are you ready",
        "Can I help you",
        "Is there anything else",
        "Is everything okay",
        "Does that seem good to you",
        "Do you think it's correct",
        "Is it going to work",
        "Are they ready",
        "Can they help me",
        "Do they know what to do",
        "Do they want to go"
    ]
    
    print("Testing individual questions:")
    print("-" * 40)
    
    for text in test_cases:
        result = restore_punctuation(text, 'en')
        print(f"Input:  {text}")
        print(f"Output: {result}")
        print()
    
    # Test the transcription logic simulation
    print("\n" + "=" * 50)
    print("TESTING TRANSCRIPTION LOGIC SIMULATION")
    print("=" * 50)
    
    transcription_text = """
    Are we ready

    How are you

    I don't know what this is

    Can you help me

    Do you know where it is

    Would you like to go

    Do you need anything else

    Do you have time

    Is it going to rain today

    Are you ready
    """
    
    # Simulate the transcription logic
    text_segments = [seg.strip() for seg in transcription_text.split('\n\n') if seg.strip()]
    sentences = []
    
    for segment in text_segments:
        processed_segment = restore_punctuation(segment, 'en')
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
    test_english_questions()
