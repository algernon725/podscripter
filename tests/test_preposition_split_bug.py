#!/usr/bin/env python3
"""
Test for sentence splitting bug where sentences are incorrectly split after prepositions.

Issue: Sentences end with prepositions like "a" in Spanish (similar to ending with "to" in English),
which is grammatically incorrect. Example:
- "entonces yo conocí a." | "Un amigo que trabajaba con cámaras" (INCORRECT)
- Should be: "entonces yo conocí a un amigo que trabajaba con cámaras" (CORRECT)

This test ensures that common prepositions in all supported languages (ES/EN/FR/DE) 
do not cause sentence splits.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from punctuation_restorer import restore_punctuation


def test_spanish_preposition_a_no_split():
    """Test that Spanish preposition 'a' does not cause sentence splits."""
    # From user's example: "entonces yo conocí a un amigo que trabajaba con cámaras"
    text = "entonces yo conocí a un amigo que trabajaba con cámaras pero él estaba muy ocupado"
    result = restore_punctuation(text, 'es')
    
    # Should NOT split after "a"
    # Should be one sentence or sensibly split, but NOT ending with "a."
    assert not result.endswith(' a.'), f"Sentence incorrectly ends with 'a.': {result}"
    assert ' a. ' not in result, f"Sentence incorrectly split after 'a': {result}"
    print(f"✓ Spanish 'a' test passed: {result}")


def test_spanish_prepositions_comprehensive():
    """Test various Spanish prepositions that should not end sentences."""
    test_cases = [
        # preposition: a (to)
        ("buscamos a una persona que supiera grabar videos", "a"),
        ("voy a la tienda mañana", "a"),
        # preposition: ante (before)
        ("estamos ante una situación difícil ahora", "ante"),
    ]
    
    for text, preposition in test_cases:
        result = restore_punctuation(text, 'es')
        error_pattern = f' {preposition}. '
        assert error_pattern not in result, \
            f"Sentence incorrectly split after '{preposition}': {result}"
        print(f"✓ Spanish '{preposition}' test passed: {result}")


def test_english_prepositions_no_split():
    """Test that English prepositions do not cause sentence splits."""
    test_cases = [
        # preposition: to
        ("I went to the store yesterday", "to"),
        ("we need to find someone who can help", "to"),
        # preposition: at
        ("she works at a large company in the city", "at"),
        # preposition: from
        ("he comes from Texas in the United States", "from"),
        # preposition: with
        ("I talked with my friend about the project", "with"),
    ]
    
    for text, preposition in test_cases:
        result = restore_punctuation(text, 'en')
        error_pattern = f' {preposition}. '
        assert error_pattern not in result, \
            f"English sentence incorrectly split after '{preposition}': {result}"
        print(f"✓ English '{preposition}' test passed: {result}")


def test_french_prepositions_no_split():
    """Test that French prepositions do not cause sentence splits."""
    test_cases = [
        # preposition: à (to/at)
        ("je vais à la maison maintenant", "à"),
        ("nous parlons à notre ami demain", "à"),
        # preposition: avec (with)
        ("il travaille avec son équipe aujourd'hui", "avec"),
    ]
    
    for text, preposition in test_cases:
        result = restore_punctuation(text, 'fr')
        error_pattern = f' {preposition}. '
        assert error_pattern not in result, \
            f"French sentence incorrectly split after '{preposition}': {result}"
        print(f"✓ French '{preposition}' test passed: {result}")


def test_german_prepositions_no_split():
    """Test that German prepositions do not cause sentence splits."""
    test_cases = [
        # preposition: zu (to)
        ("ich gehe zu meinem Freund morgen", "zu"),
        # preposition: bei (at/with)
        ("er arbeitet bei einer großen Firma", "bei"),
        # preposition: mit (with)
        ("sie spricht mit ihrem Lehrer heute", "mit"),
    ]
    
    for text, preposition in test_cases:
        result = restore_punctuation(text, 'de')
        error_pattern = f' {preposition}. '
        assert error_pattern not in result, \
            f"German sentence incorrectly split after '{preposition}': {result}"
        print(f"✓ German '{preposition}' test passed: {result}")


def test_user_reported_bug():
    """Test the exact scenario reported by the user from Episodio190_raw.txt"""
    # From segments showing: "entonces yo conocí a un amigo que trabajaba con cámaras"
    # This was being split as:
    # "Buscamos a una persona que supiera grabar videos entonces yo conocí a."
    # "Un amigo que trabajaba con cámaras pero él estaba muy ocupado..."
    
    text = """buscamos a una persona que supiera grabar videos entonces yo conocí a 
    un amigo que trabajaba con cámaras pero él estaba muy ocupado y él me recomendó 
    a un amigo de él y este chico pues nos grababa y todo pero él nunca mencionó 
    como tienen que comprar un micrófono de solapa"""
    
    result = restore_punctuation(text, 'es')
    
    # Check that we don't split after "a"
    assert ' a. U' not in result and ' a. u' not in result, \
        f"Bug reproduced: sentence split after 'a': {result}"
    
    # The preposition "a" should be followed by content, not a sentence break
    sentences = result.split('.')
    for sentence in sentences:
        stripped = sentence.strip()
        if stripped and stripped.endswith(' a'):
            raise AssertionError(f"Sentence ends with preposition 'a': {stripped}")
    
    print(f"✓ User-reported bug test passed")
    print(f"Result: {result[:200]}...")


if __name__ == '__main__':
    print("Testing preposition split bug...")
    print("="*60)
    
    try:
        test_spanish_preposition_a_no_split()
        test_spanish_prepositions_comprehensive()
        test_user_reported_bug()
        test_english_prepositions_no_split()
        test_french_prepositions_no_split()
        test_german_prepositions_no_split()
        
        print("="*60)
        print("✅ All tests passed!")
    except AssertionError as e:
        print("="*60)
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print("="*60)
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

