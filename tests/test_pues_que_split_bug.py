#!/usr/bin/env python3
"""
Test for Spanish question splitting bug: "Pues, ¿qué pasó, Nate?" incorrectly
splits into "Pues, ¿Qué?" and "¿Pasó, Nate?" during semantic sentence splitting.

The issue occurs when:
1. A question starts with "¿" mid-sentence
2. The first word after "¿" gets capitalized
3. The semantic splitter sees the capitalized word and incorrectly splits there
4. The guard that should prevent this split is missing

Root cause: No guard prevents splitting after/before "¿" in _should_end_sentence_here()
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punctuation_restorer import restore_punctuation


def test_pues_que_split():
    """
    Test that "Pues, ¿qué pasó, Nate?" is NOT split into two sentences.
    
    This is a real-world case from Episodio 190, Segment 42.
    Raw Whisper output: "Pues, ¿qué pasó, Nate?"
    Expected: Single sentence preserved
    Bug: Incorrectly splits to "Pues, ¿Qué?" | "¿Pasó, Nate?"
    """
    # Simulating the raw Whisper output (single segment)
    raw_input = "Pues, ¿qué pasó, Nate?"
    
    # Process with Spanish punctuation restoration
    result = restore_punctuation(raw_input, language='es')
    
    print(f"Input:  '{raw_input}'")
    print(f"Output: '{result}'")
    print()
    
    # The result should be a single sentence, not split
    # Acceptable variations:
    # - "Pues, ¿qué pasó, Nate?" (preserved as-is)
    # - "Pues ¿qué pasó, Nate?" (comma removed before ¿)
    
    # Check that it's NOT split into two sentences
    lines = result.strip().split('\n')
    
    if len(lines) > 1:
        print("❌ FAILED: Sentence was incorrectly split!")
        print(f"   Got {len(lines)} lines:")
        for i, line in enumerate(lines, 1):
            print(f"   Line {i}: '{line}'")
        return False
    
    # Check that "pasó" and "Nate" are in the same sentence
    if 'pasó' in result and 'Nate' in result:
        # They should be close together (in the same sentence)
        pasa_idx = result.lower().find('pasó')
        nate_idx = result.lower().find('nate')
        
        if pasa_idx >= 0 and nate_idx >= 0:
            # They should be within reasonable distance (same sentence)
            distance = abs(nate_idx - pasa_idx)
            if distance < 50:  # Conservative check
                print("✅ PASSED: Sentence preserved as single unit")
                print(f"   'pasó' and 'Nate' are {distance} chars apart (same sentence)")
                return True
    
    print("❌ FAILED: 'pasó' and 'Nate' not found together")
    return False


def test_pues_que_split_with_context():
    """
    Test with realistic context from Episodio 190.
    
    The bug occurs when there's sufficient preceding text to trigger semantic splitting.
    The semantic splitter sees "¿Qué" (capitalized) and incorrectly splits there.
    """
    # Real context from Episodio 190, segments leading up to segment 42
    raw_input = """Bueno, en este episodio vamos a seguir hablando de nuestra historia pero ya más sobre la historia de Spanish Land School para aquellos que han estado escuchando nuestro podcast por un tiempo saben que nosotros hemos estado haciendo una serie de episodios sobre toda nuestra historia desde que éramos niños hasta ahora así que hoy vamos a hablar de cómo inició Spanish Land School básicamente hablaremos de los dos primeros años de Spanish Land School y les contaremos los detalles pero si tú quieres escuchar los episodios anteriores puedes ir al episodio 147, 151, 156, 164, 170, 177 y 184 el episodio más reciente sobre esta serie fue el 184 donde hicimos el capítulo 7 y hoy vamos a hacer el capítulo 8 sí, nuestra historia es un poco más largo que pensaste, ¿cierto, Andrea? Pues es que dimos muchos detalles, quizás no deberíamos haber dado tantos sí, bueno, es que queremos que ustedes aprendan con nuestras historias y algunos tienen preguntas de quién somos o quiénes son las personas que están haciendo este podcast exacto, entonces si tienes curiosidad y quieres escuchar conversaciones naturales, pues puedes ir a escuchar esos episodios que acabé de nombrar bueno, entonces, en el episodio pasado, ¿en dónde quedamos? Nosotros nos casamos en Colombia y tuvimos una ceremonia religiosa allá luego fuimos una semana a México de luna de miel y luego fuimos a California donde los papás de Nate y pasamos allá la Navidad de ese año 2016 y en Estados Unidos fue donde nos casamos por lo legal así que hablemos de lo que pasó después en el año 2017 Pues, ¿qué pasó, Nate?"""
    
    # Process with Spanish punctuation restoration
    result = restore_punctuation(raw_input, language='es')
    
    print(f"Testing with realistic context (long text)")
    print(f"Looking for: 'Pues, ¿qué pasó, Nate?' (should be one sentence)")
    print()
    
    # Split into sentences to check
    sentences = result.strip().split('\n')
    
    # Find sentences containing the target phrase parts
    pues_sentence = None
    paso_sentence = None
    
    for i, s in enumerate(sentences):
        s_lower = s.lower()
        if 'pues' in s_lower and '¿' in s:
            pues_sentence = (i, s)
        if 'pasó' in s_lower and 'nate' in s_lower:
            paso_sentence = (i, s)
    
    print(f"Total sentences: {len(sentences)}")
    
    if pues_sentence:
        print(f"Sentence with 'Pues': [{pues_sentence[0]}] '{pues_sentence[1]}'")
    if paso_sentence:
        print(f"Sentence with 'pasó, Nate': [{paso_sentence[0]}] '{paso_sentence[1]}'")
    print()
    
    # Check if they're in the same sentence
    if pues_sentence and paso_sentence:
        if pues_sentence[0] == paso_sentence[0]:
            print("✅ PASSED: 'Pues, ¿qué pasó, Nate?' preserved in one sentence")
            return True
        else:
            print("❌ FAILED: Incorrectly split into separate sentences!")
            print(f"   'Pues' in sentence {pues_sentence[0]}: '{pues_sentence[1]}'")
            print(f"   'pasó, Nate' in sentence {paso_sentence[0]}: '{paso_sentence[1]}'")
            return False
    
    print("❌ FAILED: Could not find target phrases")
    return False


def test_similar_patterns():
    """Test similar patterns that should NOT be split."""
    test_cases = [
        # (input, description)
        ("Entonces, ¿qué hiciste ayer?", "Question after 'Entonces'"),
        ("Bueno, ¿cómo estuvo tu día?", "Question after 'Bueno'"),
        ("Pues, ¿dónde vives ahora?", "Question after 'Pues'"),
        ("Y, ¿cuándo vas a venir?", "Question after 'Y'"),
        ("Así que, ¿qué pasó después?", "Question after 'Así que'"),
    ]
    
    all_passed = True
    
    for input_text, desc in test_cases:
        result = restore_punctuation(input_text, language='es')
        lines = result.strip().split('\n')
        
        if len(lines) > 1:
            print(f"❌ FAILED: {desc}")
            print(f"   Input:  '{input_text}'")
            print(f"   Output: '{result}'")
            print(f"   Split into {len(lines)} lines")
            all_passed = False
        else:
            print(f"✅ PASSED: {desc}")
    
    return all_passed


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Spanish question split bug: 'Pues, ¿qué pasó, Nate?'")
    print("=" * 70)
    print()
    
    test1 = test_pues_que_split()
    print()
    print("-" * 70)
    print()
    
    test2 = test_pues_que_split_with_context()
    print()
    print("-" * 70)
    print()
    
    print("Testing similar patterns:")
    print()
    test3 = test_similar_patterns()
    print()
    
    if test1 and test2 and test3:
        print("=" * 70)
        print("ALL TESTS PASSED ✅")
        print("=" * 70)
        sys.exit(0)
    else:
        print("=" * 70)
        print("SOME TESTS FAILED ❌")
        print("=" * 70)
        sys.exit(1)

