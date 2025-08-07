#!/usr/bin/env python3
"""
Test to debug the question detection inconsistency.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punctuation_restorer import restore_punctuation, has_question_indicators, is_question_semantic
from sentence_transformers import SentenceTransformer


def test_question_detection_debug():
    """Debug the question detection inconsistency."""
    
    print("Debugging question detection inconsistency...")
    print("=" * 60)
    
    # Test cases that are getting question marks but shouldn't
    test_cases = [
        "Estamos listos",
        "Están listos", 
        "Cómo están",
        "Estás listo"
    ]
    
    # Initialize the model for semantic detection
    try:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    print("\nTesting question detection methods:")
    print("-" * 50)
    
    for text in test_cases:
        print(f"\nText: '{text}'")
        
        # Test pattern-based detection
        pattern_result = has_question_indicators(text, 'es')
        print(f"  Pattern detection: {pattern_result}")
        
        # Test semantic detection
        try:
            semantic_result = is_question_semantic(text, model, 'es')
            print(f"  Semantic detection: {semantic_result}")
        except Exception as e:
            print(f"  Semantic detection failed: {e}")
        
        # Test full punctuation restoration
        full_result = restore_punctuation(text, 'es')
        print(f"  Full result: '{full_result}'")
        
        # Check if it ends with ? or .
        if full_result.endswith('?'):
            print(f"  → Ends with ? (detected as question)")
        elif full_result.endswith('.'):
            print(f"  → Ends with . (detected as statement)")
        else:
            print(f"  → No punctuation added")
    
    print("\n" + "=" * 60)
    print("ANALYSIS:")
    print("=" * 60)
    
    print("The issue is that there are two question detection systems:")
    print("1. Pattern-based detection (has_question_indicators) - looks for specific words")
    print("2. Semantic detection (is_question_semantic) - uses AI to understand meaning")
    print("\nIf semantic detection says it's a question, it gets a ? even if pattern detection doesn't.")


if __name__ == "__main__":
    test_question_detection_debug()
