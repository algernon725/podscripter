#!/usr/bin/env python3
"""
Test to debug the specific case "Pudiste mantener una conversación?"
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punctuation_restorer import restore_punctuation, has_question_indicators, is_question_semantic
from sentence_transformers import SentenceTransformer


def test_specific_question():
    """Test the specific case that's still failing."""
    
    print("Testing specific question case...")
    print("=" * 50)
    
    test_text = "Pudiste mantener una conversación"
    
    print(f"Input: '{test_text}'")
    print("-" * 30)
    
    # Test pattern-based detection
    pattern_result = has_question_indicators(test_text, 'es')
    print(f"Pattern detection: {pattern_result}")
    
    # Test semantic detection
    try:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        semantic_result = is_question_semantic(test_text, model, 'es')
        print(f"Semantic detection: {semantic_result}")
    except Exception as e:
        print(f"Semantic detection failed: {e}")
    
    # Test full punctuation restoration
    full_result = restore_punctuation(test_text, 'es')
    print(f"Full result: '{full_result}'")
    
    # Check if it ends with ? or .
    if full_result.endswith('?'):
        print(f"→ Ends with ? (detected as question)")
        if not full_result.startswith('¿'):
            print(f"❌ PROBLEM: Question detected but missing inverted question mark!")
        else:
            print(f"✅ Correct: Question with inverted question mark")
    elif full_result.endswith('.'):
        print(f"→ Ends with . (detected as statement)")
    else:
        print(f"→ No punctuation added")
    
    print("\n" + "=" * 50)
    print("ANALYSIS:")
    print("=" * 50)
    
    print("'Pudiste mantener una conversación' should be detected as a question because:")
    print("- It starts with 'Pudiste' (past tense of 'poder' - can/could)")
    print("- It's asking about ability in the past")
    print("- It has question intonation")
    print("- It should be: '¿Pudiste mantener una conversación?'")


if __name__ == "__main__":
    test_specific_question()
