#!/usr/bin/env python3
"""
Comprehensive test for improved Spanish question detection
"""

from punctuation_restorer import restore_punctuation

def test_spanish_questions():
    """Test various Spanish question patterns."""
    
    print("Testing Improved Spanish Question Detection")
    print("=" * 60)
    
    # Test cases covering different Spanish question patterns
    test_cases = [
        # Basic question words
        {
            'input': "qué hora es la reunión mañana",
            'description': 'Basic question word (qué)',
            'expected': True
        },
        {
            'input': "dónde está la oficina",
            'description': 'Basic question word (dónde)',
            'expected': True
        },
        {
            'input': "cuándo es la cita",
            'description': 'Basic question word (cuándo)',
            'expected': True
        },
        {
            'input': "cómo estás hoy",
            'description': 'Basic question word (cómo)',
            'expected': True
        },
        {
            'input': "quién puede ayudarme",
            'description': 'Basic question word (quién)',
            'expected': True
        },
        {
            'input': "cuál es tu nombre",
            'description': 'Basic question word (cuál)',
            'expected': True
        },
        
        # Question patterns with verbs
        {
            'input': "puedes enviarme la agenda",
            'description': 'Question pattern (puedes)',
            'expected': True
        },
        {
            'input': "podrías explicar esto",
            'description': 'Question pattern (podrías)',
            'expected': True
        },
        {
            'input': "vas a venir mañana",
            'description': 'Question pattern (vas a)',
            'expected': True
        },
        {
            'input': "tienes tiempo para reunirte",
            'description': 'Question pattern (tienes)',
            'expected': True
        },
        {
            'input': "necesitas ayuda con esto",
            'description': 'Question pattern (necesitas)',
            'expected': True
        },
        {
            'input': "sabes dónde queda",
            'description': 'Question pattern (sabes)',
            'expected': True
        },
        {
            'input': "hay algo más que necesites",
            'description': 'Question pattern (hay)',
            'expected': True
        },
        {
            'input': "está todo bien contigo",
            'description': 'Question pattern (está)',
            'expected': True
        },
        {
            'input': "te gusta esta idea",
            'description': 'Question pattern (te gusta)',
            'expected': True
        },
        {
            'input': "quieres que vayamos juntos",
            'description': 'Question pattern (quieres)',
            'expected': True
        },
        {
            'input': "te parece bien la propuesta",
            'description': 'Question pattern (te parece)',
            'expected': True
        },
        {
            'input': "crees que es correcto",
            'description': 'Question pattern (crees)',
            'expected': True
        },
        {
            'input': "piensas que funcionará",
            'description': 'Question pattern (piensas)',
            'expected': True
        },
        
        # Question word combinations
        {
            'input': "qué hora es la reunión",
            'description': 'Question word combination (qué hora)',
            'expected': True
        },
        {
            'input': "dónde está la reunión",
            'description': 'Question word combination (dónde está)',
            'expected': True
        },
        {
            'input': "cuándo es la cita",
            'description': 'Question word combination (cuándo es)',
            'expected': True
        },
        {
            'input': "cómo está todo",
            'description': 'Question word combination (cómo está)',
            'expected': True
        },
        {
            'input': "quién puede ayudarme",
            'description': 'Question word combination (quién puede)',
            'expected': True
        },
        {
            'input': "cuál es tu preferencia",
            'description': 'Question word combination (cuál es)',
            'expected': True
        },
        
        # Non-questions (should not be detected as questions)
        {
            'input': "hola como estás hoy",
            'description': 'Greeting (not a question)',
            'expected': False
        },
        {
            'input': "gracias por tu ayuda",
            'description': 'Thank you (not a question)',
            'expected': False
        },
        {
            'input': "el proyecto está terminado",
            'description': 'Statement (not a question)',
            'expected': False
        },
        {
            'input': "necesito más información",
            'description': 'Statement (not a question)',
            'expected': False
        },
        {
            'input': "la reunión es mañana",
            'description': 'Statement (not a question)',
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
    print("SPANISH QUESTION DETECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {total_tests}")
    print(f"Correct detections: {correct_detections}")
    print(f"Accuracy: {accuracy:.1%}")
    
    if accuracy >= 0.8:
        print("🎉 Excellent Spanish question detection!")
    elif accuracy >= 0.6:
        print("✅ Good Spanish question detection")
    else:
        print("⚠️ Spanish question detection needs improvement")

if __name__ == "__main__":
    test_spanish_questions() 