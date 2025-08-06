#!/usr/bin/env python3
"""
Test to reproduce and fix Spanish sentence splitting issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from punctuation_restorer import restore_punctuation

def test_spanish_sentence_splitting():
    """Test Spanish sentence splitting issues."""
    
    print("Testing Spanish Sentence Splitting Issues")
    print("=" * 60)
    
    # Test cases that demonstrate the problems
    test_cases = [
        {
            'input': "yo soy andrea de santander colombia",
            'expected': "Yo soy Andrea de Santander Colombia.",
            'description': 'Introduction should be one sentence'
        },
        {
            'input': "recuerdas todos esos momentos en los que no supiste qué decir",
            'expected': "¿Recuerdas todos esos momentos en los que no supiste qué decir?",
            'description': 'Question should be one sentence with question mark'
        },
        {
            'input': "hola cómo estás hoy",
            'expected': "¿Hola, cómo estás hoy?",
            'description': 'Greeting with question'
        },
        {
            'input': "me llamo carlos y vivo en madrid",
            'expected': "Me llamo Carlos y vivo en Madrid.",
            'description': 'Introduction with conjunction'
        },
        {
            'input': "qué hora es la reunión mañana",
            'expected': "¿Qué hora es la reunión mañana?",
            'description': 'Question about time'
        },
        {
            'input': "es importante que todos estén presentes",
            'expected': "Es importante que todos estén presentes.",
            'description': 'Statement about importance'
        }
    ]
    
    correct_results = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        input_text = test_case['input']
        expected = test_case['expected']
        description = test_case['description']
        
        print(f"\nTest {i}: {description}")
        print(f"Input:  {input_text}")
        
        try:
            result = restore_punctuation(input_text, 'es')
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
    print("SPANISH SENTENCE SPLITTING TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {total_tests}")
    print(f"Correct results: {correct_results}")
    print(f"Accuracy: {accuracy:.1%}")
    
    if accuracy >= 0.8:
        print("🎉 Excellent! Spanish sentence splitting is working well!")
    elif accuracy >= 0.6:
        print("✅ Good! Most Spanish sentences are correctly handled")
    else:
        print("⚠️ Spanish sentence splitting needs improvement")

if __name__ == "__main__":
    test_spanish_sentence_splitting() 