#!/usr/bin/env python3
"""
Test to reproduce and fix French sentence splitting issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from punctuation_restorer import restore_punctuation

def test_french_sentence_splitting():
    """Test French sentence splitting issues."""
    
    print("Testing French Sentence Splitting Issues")
    print("=" * 60)
    
    # Test cases that demonstrate the problems
    test_cases = [
        {
            'input': "je suis marie de paris france",
            'expected': "Je suis Marie de Paris, France.",
            'description': 'Introduction should be one sentence'
        },
        {
            'input': "vous souvenez vous de tous ces moments où vous ne saviez pas quoi dire",
            'expected': "Vous souvenez-vous de tous ces moments où vous ne saviez pas quoi dire?",
            'description': 'Question should be one sentence with question mark'
        },
        {
            'input': "bonjour comment allez vous aujourd'hui",
            'expected': "Bonjour, comment allez-vous aujourd'hui?",
            'description': 'Greeting with question'
        },
        {
            'input': "je m'appelle pierre et j'habite à lyon",
            'expected': "Je m'appelle Pierre et j'habite à Lyon.",
            'description': 'Introduction with conjunction'
        },
        {
            'input': "à quelle heure est la réunion demain",
            'expected': "À quelle heure est la réunion demain?",
            'description': 'Question about time'
        },
        {
            'input': "il est important que tout le monde soit présent",
            'expected': "Il est important que tout le monde soit présent.",
            'description': 'Statement about importance'
        },
        {
            'input': "pouvez vous m'aider avec ce projet",
            'expected': "Pouvez-vous m'aider avec ce projet?",
            'description': 'Request for help'
        },
        {
            'input': "je voudrais me présenter",
            'expected': "Je voudrais me présenter.",
            'description': 'Formal introduction'
        },
        {
            'input': "où avez vous appris à parler français",
            'expected': "Où avez-vous appris à parler français?",
            'description': 'Question about learning'
        },
        {
            'input': "merci pour votre temps aujourd'hui",
            'expected': "Merci pour votre temps aujourd'hui.",
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
            result = restore_punctuation(input_text, 'fr')
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
    print("FRENCH SENTENCE SPLITTING TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {total_tests}")
    print(f"Correct results: {correct_results}")
    print(f"Accuracy: {accuracy:.1%}")
    
    if accuracy >= 0.8:
        print("🎉 Excellent! French sentence splitting is working well!")
    elif accuracy >= 0.6:
        print("✅ Good! Most French sentences are correctly handled")
    else:
        print("⚠️ French sentence splitting needs improvement")

if __name__ == "__main__":
    test_french_sentence_splitting()
