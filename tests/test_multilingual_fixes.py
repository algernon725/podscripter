#!/usr/bin/env python3
"""
Test to verify which fixes apply to other languages beyond Spanish.
"""

import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punctuation_restorer import restore_punctuation


def test_multilingual_fixes():
    """Test which fixes apply to different languages."""
    
    print("Testing multilingual fixes...")
    print("=" * 50)
    
    # Test cases for different languages
    test_cases = {
        'en': [
            "Hello everyone",
            "Welcome to English",
            "This is a podcast that will help you learn English",
            "Do you remember all those moments when you didn't know what to say",
            "Those moments when you couldn't maintain a conversation",
            "Well don't worry",
            "English is the tool you were looking for to improve your English"
        ],
        'fr': [
            "Bonjour à tous",
            "Bienvenue à Français",
            "Français est le podcast qui va vous aider à être prêt à parler français",
            "Vous souvenez-vous de tous ces moments où vous ne saviez pas quoi dire",
            "Ces moments où vous n'avez pas pu maintenir une conversation",
            "Eh bien ne vous inquiétez pas",
            "Français est l'outil que vous cherchiez pour améliorer votre français"
        ],
        'de': [
            "Hallo an alle",
            "Willkommen zu Deutsch",
            "Deutsch ist der Podcast, der Ihnen helfen wird, bereit zu sein, Deutsch zu sprechen",
            "Erinnerst du dich an all diese Momente, in denen du nicht wusstest, was du sagen solltest",
            "Diese Momente, in denen du keine Konversation aufrechterhalten konntest",
            "Nun mach dir keine Sorgen",
            "Deutsch ist das Werkzeug, das du gesucht hast, um dein Deutsch zu verbessern"
        ]
    }
    
    for language, sentences in test_cases.items():
        print(f"\n{language.upper()} LANGUAGE:")
        print("-" * 30)
        
        for sentence in sentences:
            result = restore_punctuation(sentence, language)
            print(f"Input:  {sentence}")
            print(f"Output: {result}")
            print()
    
    # Test the transcription logic for different languages
    print("\n" + "=" * 50)
    print("TESTING TRANSCRIPTION LOGIC FOR DIFFERENT LANGUAGES")
    print("=" * 50)
    
    transcription_texts = {
        'en': """
        Hello everyone

        Welcome to English

        This is a podcast that will help you learn English

        Do you remember all those moments when you didn't know what to say

        Well don't worry

        English is the tool you were looking for
        """,
        'fr': """
        Bonjour à tous

        Bienvenue à Français

        Français est le podcast qui va vous aider

        Vous souvenez-vous de tous ces moments

        Eh bien ne vous inquiétez pas

        Français est l'outil que vous cherchiez
        """,
        'de': """
        Hallo an alle

        Willkommen zu Deutsch

        Deutsch ist der Podcast der Ihnen helfen wird

        Erinnerst du dich an all diese Momente

        Nun mach dir keine Sorgen

        Deutsch ist das Werkzeug das du gesucht hast
        """
    }
    
    for language, text in transcription_texts.items():
        print(f"\n{language.upper()} TRANSCRIPTION SIMULATION:")
        print("-" * 40)
        
        # Simulate the transcription logic
        text_segments = [seg.strip() for seg in text.split('\n\n') if seg.strip()]
        sentences = []
        
        for segment in text_segments:
            processed_segment = restore_punctuation(segment, language)
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
    test_multilingual_fixes()
