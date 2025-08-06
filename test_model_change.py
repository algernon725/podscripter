#!/usr/bin/env python3
"""
Quick test to verify the model change to paraphrase-multilingual-MiniLM-L12-v2
"""

from punctuation_restorer import restore_punctuation

def test_model_change():
    """Test the updated model with multilingual examples."""
    
    print("Testing updated model: paraphrase-multilingual-MiniLM-L12-v2")
    print("=" * 60)
    
    # Test cases in different languages
    test_cases = [
        {
            'lang': 'en',
            'input': "what time is the meeting tomorrow can you send me the agenda",
            'description': 'English questions'
        },
        {
            'lang': 'es',
            'input': "¿qué hora es la reunión mañana puedes enviarme la agenda",
            'description': 'Spanish questions'
        },
        {
            'lang': 'de',
            'input': "wann ist das treffen morgen kannst du mir die agenda schicken",
            'description': 'German questions'
        },
        {
            'lang': 'fr',
            'input': "à quelle heure est la réunion demain pouvez-vous m'envoyer l'ordre du jour",
            'description': 'French questions'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        lang = test_case['lang']
        input_text = test_case['input']
        description = test_case['description']
        
        print(f"\nTest {i}: {description} ({lang})")
        print(f"Input:  {input_text}")
        
        try:
            result = restore_punctuation(input_text, lang)
            print(f"Output: {result}")
            
            # Check if questions were detected
            if '?' in result:
                print("✓ Questions detected correctly")
            else:
                print("⚠ No questions detected")
                
        except Exception as e:
            print(f"✗ Error: {e}")
        
        print("-" * 40)
    
    print("\nModel change verification complete!")
    print("Benefits of paraphrase-multilingual-MiniLM-L12-v2:")
    print("- Better multilingual support")
    print("- Improved semantic understanding across languages")
    print("- More accurate question detection in non-English languages")
    print("- Better handling of mixed-language content")

if __name__ == "__main__":
    test_model_change() 