#!/usr/bin/env python3
"""
Test to verify French run-on sentence fix
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from punctuation_restorer import restore_punctuation

def test_french_runon_fix():
    """Test that French text is properly split into sentences."""
    
    print("Testing French Run-on Sentence Fix")
    print("=" * 60)
    
    # The problematic run-on text in French
    runon_text = """Bonjour à tous  Bienvenue à FrançaisPod  FrançaisPod est le podcast qui vous aidera à être prêt à parler français  FrançaisPod vous prépare à parler français n'importe où n'importe quand et dans n'importe quelle situation  Vous souvenez vous de tous ces moments où vous ne saviez pas quoi dire  Ces moments où vous ne pouviez pas maintenir une conversation  Eh bien ne vous inquiétez pas  FrançaisPod est l'outil que vous cherchiez pour améliorer votre français  Dites adieu  À tous ces moments gênants  Alors commençons  Sommes nous prêts  Je suis Marie de Paris France  Et je suis Pierre de Lyon France  Bonjour à tous"""
    
    print("Input (run-on text):")
    print(repr(runon_text))
    print()
    
    # Process the text
    result = restore_punctuation(runon_text, language='fr')
    
    print("Output (processed text):")
    print(result)
    print()
    
    # Count sentences (rough estimate by counting periods, question marks, exclamation marks)
    sentence_count = result.count('.') + result.count('?') + result.count('!')
    
    print(f"Number of sentences detected: {sentence_count}")
    print()
    
    # Check if the text is properly split
    if sentence_count > 5:  # Should have multiple sentences
        print("✅ SUCCESS: Text is properly split into multiple sentences")
    else:
        print("❌ FAILURE: Text is still a run-on sentence")
    
    # Check for specific expected patterns
    expected_patterns = [
        "Bonjour à tous",
        "Bienvenue à FrançaisPod",
        "Vous souvenez-vous de tous ces moments",
        "Je suis Marie de Paris, France",
        "Et je suis Pierre de Lyon, France"
    ]
    
    print("\nChecking for expected sentence patterns:")
    for pattern in expected_patterns:
        if pattern in result:
            print(f"✅ Found: {pattern}")
        else:
            print(f"❌ Missing: {pattern}")

if __name__ == "__main__":
    test_french_runon_fix()
