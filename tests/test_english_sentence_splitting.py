#!/usr/bin/env python3
"""
Test to reproduce and fix English sentence splitting issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from punctuation_restorer import restore_punctuation

def test_english_sentence_splitting():
    """Test English sentence splitting issues."""
    
    print("Testing English Sentence Splitting Issues")
    print("=" * 60)
    
    # Test cases that demonstrate the problems
    test_cases = [
        {
            'input': "i am john from new york city",
            'expected': "I am John from New York City.",
            'description': 'Introduction should be one sentence'
        },
        {
            'input': "do you remember all those moments when you didn't know what to say",
            'expected': "Do you remember all those moments when you didn't know what to say?",
            'description': 'Question should be one sentence with question mark'
        },
        {
            'input': "hello how are you today",
            'expected': "Hello, how are you today?",
            'description': 'Greeting with question'
        },
        {
            'input': "my name is sarah and i live in london",
            'expected': "My name is Sarah and I live in London.",
            'description': 'Introduction with conjunction'
        },
        {
            'input': "what time is the meeting tomorrow",
            'expected': "What time is the meeting tomorrow?",
            'description': 'Question about time'
        },
        {
            'input': "it is important that everyone is present",
            'expected': "It is important that everyone is present.",
            'description': 'Statement about importance'
        },
        {
            'input': "can you help me with this project",
            'expected': "Can you help me with this project?",
            'description': 'Request for help'
        },
        {
            'input': "i would like to introduce myself",
            'expected': "I would like to introduce myself.",
            'description': 'Formal introduction'
        },
        {
            'input': "where did you learn to speak english",
            'expected': "Where did you learn to speak English?",
            'description': 'Question about learning'
        },
        {
            'input': "thank you for your time today",
            'expected': "Thank you for your time today.",
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
            result = restore_punctuation(input_text, 'en')
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
    print("ENGLISH SENTENCE SPLITTING TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {total_tests}")
    print(f"Correct results: {correct_results}")
    print(f"Accuracy: {accuracy:.1%}")
    
    if accuracy >= 0.8:
        print("🎉 Excellent! English sentence splitting is working well!")
    elif accuracy >= 0.6:
        print("✅ Good! Most English sentences are correctly handled")
    else:
        print("⚠️ English sentence splitting needs improvement")

if __name__ == "__main__":
    test_english_sentence_splitting()
