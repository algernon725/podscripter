#!/usr/bin/env python3
"""
Test script for improved punctuation restoration functionality
"""

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("SentenceTransformers not available. Install with: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

def test_improved_punctuation():
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return
    
    # Test cases that demonstrate the improvements
    test_cases = [
        # English - Long run-on sentences that should be properly segmented
        {
            'lang': 'en',
            'input': "hello how are you today I hope you are doing well thank you for asking about my day it was quite busy but productive I managed to finish all my tasks and even had time for a coffee break",
            'description': 'English long conversation - should create meaningful sentences'
        },
        # English - Questions mixed with statements
        {
            'lang': 'en',
            'input': "what time is the meeting tomorrow I need to prepare my presentation and also check if everyone received the agenda can you confirm the location",
            'description': 'English mixed questions and statements'
        },
        # English - Sentences with conjunctions that should continue
        {
            'lang': 'en',
            'input': "I went to the store and bought some groceries then I came home and started cooking dinner because I was hungry and wanted to eat something healthy",
            'description': 'English sentences with conjunctions - should keep related clauses together'
        },
        # Spanish - Similar patterns
        {
            'lang': 'es', 
            'input': "hola como estas hoy espero que estes bien gracias por preguntar sobre mi dia fue bastante ocupado pero productivo logre terminar todas mis tareas",
            'description': 'Spanish conversation - should create meaningful sentences'
        },
        # German - Long sentences
        {
            'lang': 'de',
            'input': "hallo wie geht es dir heute ich hoffe es geht dir gut danke fur das fragen uber meinen tag es war ziemlich beschaftigt aber produktiv",
            'description': 'German conversation - should create meaningful sentences'
        },
        # French - Mixed content
        {
            'lang': 'fr',
            'input': "bonjour comment allez vous aujourd'hui j'espere que vous allez bien merci de demander de ma journee elle etait assez occupee mais productive",
            'description': 'French conversation - should create meaningful sentences'
        }
    ]
    
    print("Testing improved punctuation restoration...")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        lang = test_case['lang']
        input_text = test_case['input']
        description = test_case['description']
        
        print(f"\nTest {i}: {description} ({lang})")
        print(f"Input:  {input_text}")
        
        try:
            from punctuation_restorer import restore_punctuation
            result = restore_punctuation(input_text, lang)
            print(f"Output: {result}")
            
            # Count sentences in output
            sentences = [s.strip() for s in result.split('.') if s.strip()]
            print(f"Sentences created: {len(sentences)}")
            for j, sent in enumerate(sentences, 1):
                print(f"  {j}. {sent}")
            
            print("✓ Success")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        print("-" * 40)

def test_specific_improvements():
    """Test specific improvements in sentence boundary detection"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return
    
    print("\n" + "=" * 60)
    print("TESTING SPECIFIC IMPROVEMENTS")
    print("=" * 60)
    
    # Test cases that specifically show the improvements
    specific_tests = [
        {
            'lang': 'en',
            'input': "I went to the store and bought milk bread and eggs then I came home and started cooking dinner because I was hungry",
            'description': 'Should keep related clauses together, not split at every "and"'
        },
        {
            'lang': 'en',
            'input': "what time is the meeting can you send me the agenda I need to prepare my presentation",
            'description': 'Should detect questions and separate them properly'
        },
        {
            'lang': 'en',
            'input': "thank you for your help that was amazing I really appreciate it",
            'description': 'Should detect exclamations and gratitude expressions'
        },
        {
            'lang': 'en',
            'input': "the weather is nice today however it might rain later so we should bring umbrellas just in case",
            'description': 'Should handle transitional words like "however" properly'
        }
    ]
    
    for i, test_case in enumerate(specific_tests, 1):
        lang = test_case['lang']
        input_text = test_case['input']
        description = test_case['description']
        
        print(f"\nSpecific Test {i}: {description}")
        print(f"Input:  {input_text}")
        
        try:
            from punctuation_restorer import restore_punctuation
            result = restore_punctuation(input_text, lang)
            print(f"Output: {result}")
            
            # Show sentence breakdown
            sentences = [s.strip() for s in result.split('.') if s.strip()]
            print(f"Sentences: {len(sentences)}")
            for j, sent in enumerate(sentences, 1):
                print(f"  {j}. {sent}")
            
        except Exception as e:
            print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_improved_punctuation()
    test_specific_improvements() 