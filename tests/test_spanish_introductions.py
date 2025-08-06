#!/usr/bin/env python3
"""
Test to verify Spanish introductions and statements are not incorrectly detected as questions
"""

from punctuation_restorer import restore_punctuation

def test_spanish_introductions():
    """Test Spanish introductions and statements that should NOT be questions."""
    
    print("Testing Spanish Introductions and Statements")
    print("=" * 60)
    
    # Test cases that should NOT be detected as questions
    test_cases = [
        {
            'input': "yo soy andrea de santander colombia",
            'description': 'Introduction with "yo soy"',
            'expected': False
        },
        {
            'input': "soy andrea de santander colombia",
            'description': 'Introduction with "soy"',
            'expected': False
        },
        {
            'input': "mi nombre es andrea",
            'description': 'Introduction with "mi nombre es"',
            'expected': False
        },
        {
            'input': "me llamo andrea",
            'description': 'Introduction with "me llamo"',
            'expected': False
        },
        {
            'input': "vivo en colombia",
            'description': 'Statement with "vivo en"',
            'expected': False
        },
        {
            'input': "trabajo en santander",
            'description': 'Statement with "trabajo en"',
            'expected': False
        },
        {
            'input': "estoy en la oficina",
            'description': 'Statement with "estoy en"',
            'expected': False
        },
        {
            'input': "es importante el proyecto",
            'description': 'Statement with "es importante"',
            'expected': False
        },
        {
            'input': "está bien la reunión",
            'description': 'Statement with "está bien"',
            'expected': False
        },
        {
            'input': "soy de colombia",
            'description': 'Statement with "soy de"',
            'expected': False
        },
        {
            'input': "es de santander",
            'description': 'Statement with "es de"',
            'expected': False
        },
        {
            'input': "estoy de acuerdo",
            'description': 'Statement with "estoy de"',
            'expected': False
        }
    ]
    
    correct_detections = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        input_text = test_case['input']
        description = test_case['description']
        expected = test_case['expected']
        
        print(f"\nTest {i}: {description}")
        print(f"Input:  {input_text}")
        
        try:
            result = restore_punctuation(input_text, 'es')
            print(f"Output: {result}")
            
            # Check if question was detected
            is_question = '?' in result
            print(f"Question detected: {is_question}")
            print(f"Expected: {expected}")
            
            if is_question == expected:
                print("✓ CORRECT")
                correct_detections += 1
            else:
                print("✗ INCORRECT")
                
        except Exception as e:
            print(f"✗ Error: {e}")
        
        print("-" * 40)
    
    # Summary
    accuracy = correct_detections / total_tests
    print(f"\n{'='*60}")
    print("SPANISH INTRODUCTIONS TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {total_tests}")
    print(f"Correct detections: {correct_detections}")
    print(f"Accuracy: {accuracy:.1%}")
    
    if accuracy >= 0.9:
        print("🎉 Excellent! Spanish introductions are correctly handled!")
    elif accuracy >= 0.7:
        print("✅ Good! Most Spanish introductions are correctly handled")
    else:
        print("⚠️ Spanish introductions still need improvement")

if __name__ == "__main__":
    test_spanish_introductions() 