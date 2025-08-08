#!/usr/bin/env python3
"""
Test multilingual run-on sentence fixes across all supported languages
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from punctuation_restorer import restore_punctuation

def test_multilingual_runon_sentences():
    """Test run-on sentence fixes across multiple languages."""
    
    print("Testing Multilingual Run-on Sentence Fixes")
    print("=" * 60)
    
    # Test cases for different languages
    test_cases = [
        # English run-on
        {
            'lang': 'en',
            'input': "hello everyone welcome to our podcast today we are going to talk about language learning this is very important for everyone who wants to improve their skills",
            'description': 'English run-on sentence'
        },
        
        # Spanish run-on
        {
            'lang': 'es',
            'input': "hola a todos bienvenidos a nuestro podcast hoy vamos a hablar sobre el aprendizaje de idiomas esto es muy importante para todos los que quieren mejorar sus habilidades",
            'description': 'Spanish run-on sentence'
        },
        
        # French run-on
        {
            'lang': 'fr',
            'input': "bonjour à tous bienvenue à notre podcast aujourd'hui nous allons parler de l'apprentissage des langues c'est très important pour tous ceux qui veulent améliorer leurs compétences",
            'description': 'French run-on sentence'
        },
        
        # German run-on
        {
            'lang': 'de',
            'input': "hallo an alle willkommen zu unserem podcast heute werden wir über sprachlernen sprechen das ist sehr wichtig für alle die ihre fähigkeiten verbessern möchten",
            'description': 'German run-on sentence'
        },
        
        # Italian run-on
        {
            'lang': 'it',
            'input': "ciao a tutti benvenuti al nostro podcast oggi parleremo dell'apprendimento delle lingue questo è molto importante per tutti coloro che vogliono migliorare le loro abilità",
            'description': 'Italian run-on sentence'
        },
        
        # Portuguese run-on
        {
            'lang': 'pt',
            'input': "olá a todos bem vindos ao nosso podcast hoje vamos falar sobre aprendizado de idiomas isso é muito importante para todos que querem melhorar suas habilidades",
            'description': 'Portuguese run-on sentence'
        },
        
        # Dutch run-on
        {
            'lang': 'nl',
            'input': "hallo iedereen welkom bij onze podcast vandaag gaan we praten over taal leren dit is heel belangrijk voor iedereen die hun vaardigheden wil verbeteren",
            'description': 'Dutch run-on sentence'
        },
        
        # Russian run-on
        {
            'lang': 'ru',
            'input': "privet vsem dobro pozhalovat v nash podcast segodnya my budem govorit ob izuchenii yazykov eto ochen vazhno dlya vsekh kto khochet uluchshit svoi navyki",
            'description': 'Russian run-on sentence'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        input_text = test_case['input']
        description = test_case['description']
        language = test_case['lang']
        
        print(f"\nTest {i}: {description} ({language})")
        print(f"Input:  {input_text}")
        
        try:
            result = restore_punctuation(input_text, language)
            print(f"Output: {result}")
            
            # Count sentences
            sentence_count = result.count('.') + result.count('?') + result.count('!')
            print(f"Sentences detected: {sentence_count}")
            
            if sentence_count > 2:
                print("✅ SUCCESS: Text properly split into multiple sentences")
            else:
                print("❌ FAILURE: Text still appears to be a run-on sentence")
                
        except Exception as e:
            print(f"✗ Error: {e}")
        
        print("-" * 40)
    
    # Test specific language patterns
    print("\n" + "=" * 60)
    print("TESTING LANGUAGE-SPECIFIC PATTERNS")
    print("=" * 60)
    
    # Test Spanish inverted question marks
    spanish_text = "cómo estás hoy tienes tiempo para hablar"
    print(f"\nSpanish test: {spanish_text}")
    result = restore_punctuation(spanish_text, 'es')
    print(f"Result: {result}")
    if '¿' in result:
        print("✅ Spanish inverted question marks working")
    else:
        print("❌ Spanish inverted question marks missing")
    
    # Test French question patterns
    french_text = "comment allez vous avez vous le temps"
    print(f"\nFrench test: {french_text}")
    result = restore_punctuation(french_text, 'fr')
    print(f"Result: {result}")
    if '?' in result:
        print("✅ French question marks working")
    else:
        print("❌ French question marks missing")
    
    # Test German capitalization
    german_text = "hallo wie geht es dir heute"
    print(f"\nGerman test: {german_text}")
    result = restore_punctuation(german_text, 'de')
    print(f"Result: {result}")
    if result[0].isupper():
        print("✅ German capitalization working")
    else:
        print("❌ German capitalization missing")

if __name__ == "__main__":
    test_multilingual_runon_sentences()
