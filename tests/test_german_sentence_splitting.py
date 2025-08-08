#!/usr/bin/env python3
"""
Test to reproduce and fix German sentence splitting issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from punctuation_restorer import restore_punctuation

def test_german_sentence_splitting():
    """Test German sentence splitting issues."""
    
    print("Testing German Sentence Splitting Issues")
    print("=" * 60)
    
    # Test cases that demonstrate the problems
    test_cases = [
        {
            'input': "ich bin hans aus berlin deutschland",
            'expected': "Ich bin Hans aus Berlin, Deutschland.",
            'description': 'Introduction should be one sentence'
        },
        {
            'input': "erinnerst du dich an all diese momente als du nicht wusstest was du sagen solltest",
            'expected': "Erinnerst du dich an all diese Momente, als du nicht wusstest, was du sagen solltest?",
            'description': 'Question should be one sentence with question mark'
        },
        {
            'input': "hallo wie geht es dir heute",
            'expected': "Hallo, wie geht es dir heute?",
            'description': 'Greeting with question'
        },
        {
            'input': "ich heiße anna und ich wohne in münchen",
            'expected': "Ich heiße Anna und ich wohne in München.",
            'description': 'Introduction with conjunction'
        },
        {
            'input': "um wie viel uhr ist das treffen morgen",
            'expected': "Um wie viel Uhr ist das Treffen morgen?",
            'description': 'Question about time'
        },
        {
            'input': "es ist wichtig dass alle anwesend sind",
            'expected': "Es ist wichtig, dass alle anwesend sind.",
            'description': 'Statement about importance'
        },
        {
            'input': "kannst du mir bei diesem projekt helfen",
            'expected': "Kannst du mir bei diesem Projekt helfen?",
            'description': 'Request for help'
        },
        {
            'input': "ich möchte mich vorstellen",
            'expected': "Ich möchte mich vorstellen.",
            'description': 'Formal introduction'
        },
        {
            'input': "wo hast du deutsch gelernt",
            'expected': "Wo hast du Deutsch gelernt?",
            'description': 'Question about learning'
        },
        {
            'input': "danke für deine zeit heute",
            'expected': "Danke für deine Zeit heute.",
            'description': 'Polite closing'
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
            result = restore_punctuation(input_text, 'de')
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
    print("GERMAN SENTENCE SPLITTING TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {total_tests}")
    print(f"Correct results: {correct_results}")
    print(f"Accuracy: {accuracy:.1%}")
    
    if accuracy >= 0.8:
        print("🎉 Excellent! German sentence splitting is working well!")
    elif accuracy >= 0.6:
        print("✅ Good! Most German sentences are correctly handled")
    else:
        print("⚠️ German sentence splitting needs improvement")

if __name__ == "__main__":
    test_german_sentence_splitting()
