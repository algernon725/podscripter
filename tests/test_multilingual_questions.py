#!/usr/bin/env python3
"""
Test multilingual question patterns across all supported languages
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from punctuation_restorer import restore_punctuation

def test_multilingual_questions():
    """Test question patterns across multiple languages."""
    
    print("Testing Multilingual Question Patterns")
    print("=" * 60)
    
    # Test cases for different languages
    test_cases = [
        # English questions
        {
            'lang': 'en',
            'input': "how are you today",
            'expected': "How are you today?",
            'description': 'English basic question'
        },
        {
            'lang': 'en',
            'input': "can you help me with this",
            'expected': "Can you help me with this?",
            'description': 'English request question'
        },
        {
            'lang': 'en',
            'input': "what time is the meeting",
            'expected': "What time is the meeting?",
            'description': 'English wh-question'
        },
        
        # Spanish questions (with inverted question marks)
        {
            'lang': 'es',
            'input': "cómo estás hoy",
            'expected': "¿Cómo estás hoy?",
            'description': 'Spanish basic question'
        },
        {
            'lang': 'es',
            'input': "puedes ayudarme con esto",
            'expected': "¿Puedes ayudarme con esto?",
            'description': 'Spanish request question'
        },
        {
            'lang': 'es',
            'input': "qué hora es la reunión",
            'expected': "¿Qué hora es la reunión?",
            'description': 'Spanish wh-question'
        },
        
        # French questions
        {
            'lang': 'fr',
            'input': "comment allez vous aujourd'hui",
            'expected': "Comment allez-vous aujourd'hui?",
            'description': 'French basic question'
        },
        {
            'lang': 'fr',
            'input': "pouvez vous m'aider avec ceci",
            'expected': "Pouvez-vous m'aider avec ceci?",
            'description': 'French request question'
        },
        {
            'lang': 'fr',
            'input': "à quelle heure est la réunion",
            'expected': "À quelle heure est la réunion?",
            'description': 'French wh-question'
        },
        
        # German questions
        {
            'lang': 'de',
            'input': "wie geht es dir heute",
            'expected': "Wie geht es dir heute?",
            'description': 'German basic question'
        },
        {
            'lang': 'de',
            'input': "kannst du mir dabei helfen",
            'expected': "Kannst du mir dabei helfen?",
            'description': 'German request question'
        },
        {
            'lang': 'de',
            'input': "um wie viel uhr ist das treffen",
            'expected': "Um wie viel Uhr ist das Treffen?",
            'description': 'German wh-question'
        },
        
        # Italian questions
        {
            'lang': 'it',
            'input': "come stai oggi",
            'expected': "Come stai oggi?",
            'description': 'Italian basic question'
        },
        {
            'lang': 'it',
            'input': "puoi aiutarmi con questo",
            'expected': "Puoi aiutarmi con questo?",
            'description': 'Italian request question'
        },
        
        # Portuguese questions
        {
            'lang': 'pt',
            'input': "como você está hoje",
            'expected': "Como você está hoje?",
            'description': 'Portuguese basic question'
        },
        {
            'lang': 'pt',
            'input': "você pode me ajudar com isso",
            'expected': "Você pode me ajudar com isso?",
            'description': 'Portuguese request question'
        },
        
        # Dutch questions
        {
            'lang': 'nl',
            'input': "hoe gaat het vandaag",
            'expected': "Hoe gaat het vandaag?",
            'description': 'Dutch basic question'
        },
        {
            'lang': 'nl',
            'input': "kun je me hierbij helpen",
            'expected': "Kun je me hierbij helpen?",
            'description': 'Dutch request question'
        },
        
        # Japanese questions (basic)
        {
            'lang': 'ja',
            'input': "kyou wa dou desu ka",
            'expected': "Kyou wa dou desu ka?",
            'description': 'Japanese basic question'
        },
        
        # Russian questions
        {
            'lang': 'ru',
            'input': "kak dela segodnya",
            'expected': "Kak dela segodnya?",
            'description': 'Russian basic question'
        },
        {
            'lang': 'ru',
            'input': "mozhesh li ty mne pomoch",
            'expected': "Mozhesh li ty mne pomoch?",
            'description': 'Russian request question'
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
    print("MULTILINGUAL QUESTION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {total_tests}")
    print(f"Correct results: {correct_results}")
    print(f"Accuracy: {accuracy:.1%}")
    
    if accuracy >= 0.8:
        print("🎉 Excellent! Multilingual questions are working well!")
    elif accuracy >= 0.6:
        print("✅ Good! Most multilingual questions are correctly handled")
    else:
        print("⚠️ Multilingual questions need improvement")

if __name__ == "__main__":
    test_multilingual_questions()
