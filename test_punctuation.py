#!/usr/bin/env python3
"""
Test script for sentence transformer punctuation restoration functionality
"""

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("SentenceTransformers not available. Install with: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

def test_advanced_punctuation():
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return
    
    test_cases = [
        # English
        {
            'lang': 'en',
            'input': "hello how are you today I hope you are doing well thank you",
            'description': 'English basic conversation'
        },
        # Spanish
        {
            'lang': 'es', 
            'input': "hola como estas hoy espero que estes bien gracias",
            'description': 'Spanish basic conversation'
        },
        # German
        {
            'lang': 'de',
            'input': "hallo wie geht es dir heute ich hoffe es geht dir gut danke",
            'description': 'German basic conversation'
        },
        # French
        {
            'lang': 'fr',
            'input': "bonjour comment allez vous aujourd'hui j'espere que vous allez bien merci",
            'description': 'French basic conversation'
        }
    ]
    
    print("Testing advanced punctuation restoration...")
    
    for i, test_case in enumerate(test_cases, 1):
        lang = test_case['lang']
        input_text = test_case['input']
        description = test_case['description']
        
        print(f"\nTest {i}: {description} ({lang})")
        print(f"Input:  {input_text}")
        
        try:
            from transcribe_sentences import restore_punctuation
            result = restore_punctuation(input_text, lang)
            print(f"Output: {result}")
            print("✓ Success")
        except Exception as e:
            print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_advanced_punctuation() 