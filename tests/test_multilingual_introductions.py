#!/usr/bin/env python3
"""
Test multilingual introduction patterns across all supported languages
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from punctuation_restorer import restore_punctuation

def test_multilingual_introductions():
    """Test introduction patterns across multiple languages."""
    
    print("Testing Multilingual Introduction Patterns")
    print("=" * 60)
    
    # Test cases for different languages
    test_cases = [
        # English introductions
        {
            'lang': 'en',
            'input': "hello my name is john smith and i'm from new york",
            'expected': "Hello, my name is John Smith and I'm from New York.",
            'description': 'English formal introduction'
        },
        {
            'lang': 'en',
            'input': "hi i'm sarah from london england",
            'expected': "Hi, I'm Sarah from London, England.",
            'description': 'English casual introduction'
        },
        
        # Spanish introductions
        {
            'lang': 'es',
            'input': "hola me llamo carlos rodríguez y soy de madrid españa",
            'expected': "Hola, me llamo Carlos Rodríguez y soy de Madrid, España.",
            'description': 'Spanish formal introduction'
        },
        {
            'lang': 'es',
            'input': "buenos días soy maría de colombia",
            'expected': "Buenos días, soy María de Colombia.",
            'description': 'Spanish morning introduction'
        },
        
        # French introductions
        {
            'lang': 'fr',
            'input': "bonjour je m'appelle pierre dubois et je viens de paris france",
            'expected': "Bonjour, je m'appelle Pierre Dubois et je viens de Paris, France.",
            'description': 'French formal introduction'
        },
        {
            'lang': 'fr',
            'input': "salut je suis marie de lyon",
            'expected': "Salut, je suis Marie de Lyon.",
            'description': 'French casual introduction'
        },
        
        # German introductions
        {
            'lang': 'de',
            'input': "guten tag ich heiße hans mueller und ich komme aus berlin deutschland",
            'expected': "Guten Tag, ich heiße Hans Müller und ich komme aus Berlin, Deutschland.",
            'description': 'German formal introduction'
        },
        {
            'lang': 'de',
            'input': "hallo ich bin anna aus münchen",
            'expected': "Hallo, ich bin Anna aus München.",
            'description': 'German casual introduction'
        },
        
        # Italian introductions
        {
            'lang': 'it',
            'input': "ciao mi chiamo marco rossi e vengo da roma italia",
            'expected': "Ciao, mi chiamo Marco Rossi e vengo da Roma, Italia.",
            'description': 'Italian introduction'
        },
        
        # Portuguese introductions
        {
            'lang': 'pt',
            'input': "olá meu nome é joão silva e sou de são paulo brasil",
            'expected': "Olá, meu nome é João Silva e sou de São Paulo, Brasil.",
            'description': 'Portuguese introduction'
        },
        
        # Dutch introductions
        {
            'lang': 'nl',
            'input': "hallo ik heet jan jansen en ik kom uit amsterdam nederland",
            'expected': "Hallo, ik heet Jan Jansen en ik kom uit Amsterdam, Nederland.",
            'description': 'Dutch introduction'
        },
        
        # Japanese introductions (basic)
        {
            'lang': 'ja',
            'input': "konnichiwa watashi wa yamada desu tokyo kara kimashita",
            'expected': "Konnichiwa, watashi wa Yamada desu. Tokyo kara kimashita.",
            'description': 'Japanese introduction'
        },
        
        # Russian introductions
        {
            'lang': 'ru',
            'input': "privet menya zovut ivan petrov i ya iz moskvy rossiya",
            'expected': "Privet, menya zovut Ivan Petrov i ya iz Moskvy, Rossiya.",
            'description': 'Russian introduction'
        }
    ]
    
    correct_results = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        input_text = test_case['input']
        expected = test_case['expected']
        description = test_case['description']
        language = test_case['lang']
        
        print(f"\nTest {i}: {description} ({language})")
        print(f"Input:  {input_text}")
        
        try:
            result = restore_punctuation(input_text, language)
            print(f"Output: {result}")
            print(f"Expected: {expected}")
            
            # Check if result matches expected (allowing for minor variations)
            if result.strip() == expected.strip():
                print("✓ CORRECT")
                correct_results += 1
            else:
                print("✗ INCORRECT")
                
        except Exception as e:
            print(f"✗ Error: {e}")
        
        print("-" * 40)
    
    # Summary
    accuracy = correct_results / total_tests
    print(f"\n{'='*60}")
    print("MULTILINGUAL INTRODUCTION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {total_tests}")
    print(f"Correct results: {correct_results}")
    print(f"Accuracy: {accuracy:.1%}")
    
    if accuracy >= 0.8:
        print("🎉 Excellent! Multilingual introductions are working well!")
    elif accuracy >= 0.6:
        print("✅ Good! Most multilingual introductions are correctly handled")
    else:
        print("⚠️ Multilingual introductions need improvement")

if __name__ == "__main__":
    test_multilingual_introductions()
