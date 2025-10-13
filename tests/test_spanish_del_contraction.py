#!/usr/bin/env python3
"""
Test for Spanish contraction sentence splitting bug.

Bug: Sentences were being split incorrectly after 'del' (de + el) and 'al' (a + el).
Example: "en la parte del. Amazonas" should be "en la parte del Amazonas"

This test ensures that Spanish contractions 'del' and 'al' prevent inappropriate
sentence breaks, following the same logic as their component prepositions.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punctuation_restorer import restore_punctuation


def test_del_contraction_no_split():
    """Test that 'del' (de + el) prevents sentence splitting."""
    # Raw text from Episodio192 where splitting was happening after "del"
    text = (
        "O yo hablaba con un estudiante que vive en Brasil de hecho y él me decía que "
        "en la parte del Amazonas en Brasil en la parte del Amazonas estaban talando "
        "muchos árboles para poder crear sabanas"
    )
    
    result = restore_punctuation(text, language='es')
    
    # The sentence should NOT be split after "del"
    # It should remain as one continuous sentence about the Amazon
    assert "del." not in result, f"Sentence incorrectly split after 'del'. Result: {result}"
    assert "del Amazonas" in result, f"'del Amazonas' should stay together. Result: {result}"
    
    print("✓ 'del' contraction test passed")
    print(f"  Result: {result[:150]}...")


def test_al_contraction_no_split():
    """Test that 'al' (a + el) prevents sentence splitting."""
    # Test case with 'al' contraction
    text = (
        "Cuando llegué al aeropuerto había mucha gente esperando y todos estaban muy "
        "emocionados porque era un día especial para la ciudad"
    )
    
    result = restore_punctuation(text, language='es')
    
    # The sentence should NOT be split after "al"
    assert "al." not in result, f"Sentence incorrectly split after 'al'. Result: {result}"
    assert "al aeropuerto" in result, f"'al aeropuerto' should stay together. Result: {result}"
    
    print("✓ 'al' contraction test passed")
    print(f"  Result: {result}")


def test_multiple_del_occurrences():
    """Test text with multiple 'del' occurrences."""
    text = (
        "La parte del norte del país es diferente del sur porque tiene un clima "
        "distinto y la geografía del terreno cambia mucho"
    )
    
    result = restore_punctuation(text, language='es')
    
    # None of the 'del' occurrences should have splits after them
    assert result.count("del.") == 0, f"Found inappropriate splits after 'del'. Result: {result}"
    assert "del norte del país" in result, f"Multiple 'del' should stay connected. Result: {result}"
    
    print("✓ Multiple 'del' occurrences test passed")
    print(f"  Result: {result}")


def test_contractions_with_proper_nouns():
    """Test contractions followed by proper nouns (location names)."""
    text = (
        "Yo soy de Texas Estados Unidos pero viajé al Amazonas en Brasil y también "
        "fui del Amazonas directo a Colombia"
    )
    
    result = restore_punctuation(text, language='es')
    
    # Primary test: Contractions should not cause sentence splits
    # (Capitalization is handled separately and not part of this fix)
    assert "al." not in result, f"Should not split after 'al'. Result: {result}"
    assert "del." not in result, f"Should not split after 'del'. Result: {result}"
    
    # Verify the contractions stay connected to following words
    assert "al amazonas" in result.lower() or "al Amazonas" in result, \
        f"'al' should stay connected to 'Amazonas'. Result: {result}"
    assert "del amazonas" in result.lower() or "del Amazonas" in result, \
        f"'del' should stay connected to 'Amazonas'. Result: {result}"
    
    print("✓ Contractions with proper nouns test passed")
    print(f"  Result: {result}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Testing Spanish Contraction Sentence Splitting")
    print("="*70 + "\n")
    
    try:
        test_del_contraction_no_split()
        test_al_contraction_no_split()
        test_multiple_del_occurrences()
        test_contractions_with_proper_nouns()
        
        print("\n" + "="*70)
        print("✅ All Spanish contraction tests passed!")
        print("="*70 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

