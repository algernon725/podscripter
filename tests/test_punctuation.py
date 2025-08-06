#!/usr/bin/env python3
"""
MIT License

Copyright (c) 2025 Algernon Greenidge

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

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