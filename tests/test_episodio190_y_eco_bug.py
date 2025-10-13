#!/usr/bin/env python3
"""
Test the specific 'y eco' bug from Episodio 190 using segments from actual transcription.

The bug: When processing with --single --language es, sentences split incorrectly like:
"No eran Ah sí, no eran muy, muy buenos teníamos muchos errores y." | "Eco una vez..."

This test simulates the actual segment processing that happens during transcription.
"""

from punctuation_restorer import restore_punctuation, assemble_sentences_from_processed


def test_y_eco_from_segments():
    """Test the exact segments that cause the 'y eco' split bug."""
    
    # These are consecutive segments from Episodio 190
    segments = [
        "Ah sí, no eran muy muy buenos",  # Segment 112
        "teníamos muchos errores y eco",  # Segment 113
        "una vez que grabábamos el video en el garaje",  # Segment 114
        "y había tanto eco",  # Segment 115
    ]
    
    # Concatenate segments as they would be in single mode
    combined_text = " ".join(segments)
    
    print("="*70)
    print("Testing 'y eco' bug from Episodio 190 segments")
    print("="*70)
    print(f"Combined input:\n{combined_text}\n")
    
    # Process through punctuation restoration
    restored = restore_punctuation(combined_text, language='es')
    print(f"After punctuation restoration:\n{restored}\n")
    
    # Assemble into sentences (this is where the split might occur)
    sentences, trailing = assemble_sentences_from_processed(restored, 'es')
    
    print(f"Split into {len(sentences)} sentences:")
    for i, sent in enumerate(sentences, 1):
        print(f"  {i}. {sent}")
    if trailing:
        print(f"  Trailing: {trailing}")
    print()
    
    # Check for the bug: a sentence ending with "y"
    bug_found = False
    for i, sent in enumerate(sentences):
        sent_stripped = sent.strip().rstrip('.!?')
        if sent_stripped.endswith(' y') or sent_stripped.endswith(',y'):
            print(f"❌ BUG FOUND: Sentence {i+1} ends with 'y':")
            print(f"   '{sent}'")
            bug_found = True
        
        # Also check if "eco" starts a new sentence
        if sent.strip().lower().startswith('eco '):
            print(f"❌ BUG FOUND: Sentence {i+1} starts with 'Eco' (should be after 'y'):")
            print(f"   '{sent}'")
            bug_found = True
    
    if not bug_found:
        # Verify 'y eco' is together in some sentence
        full_text = ' '.join(sentences)
        if 'y eco' in full_text.lower():
            print("✅ SUCCESS: 'y eco' kept together")
            return True
        else:
            print("⚠️  'y eco' not found in output")
            return False
    else:
        return False


def test_simpler_y_eco_case():
    """Test a simpler version to isolate the issue."""
    
    # Simplified version of the problem
    text = "No eran muy buenos teníamos muchos errores y eco"
    
    print("="*70)
    print("Testing simplified 'y eco' case")
    print("="*70)
    print(f"Input: {text}")
    
    restored = restore_punctuation(text, language='es')
    print(f"Restored: {restored}")
    
    sentences, trailing = assemble_sentences_from_processed(restored, 'es')
    
    print(f"Sentences ({len(sentences)}):")
    for i, sent in enumerate(sentences, 1):
        print(f"  {i}. {sent}")
    print()
    
    # Check for bug
    for sent in sentences:
        if sent.strip().rstrip('.!?').endswith(' y'):
            print("❌ FAILED: Found 'y' at end of sentence")
            return False
    
    full_text = ' '.join(sentences)
    if 'y eco' in full_text.lower():
        print("✅ SUCCESS: 'y eco' kept together")
        return True
    else:
        print("⚠️  'y eco' not found")
        return False


def test_very_long_text_with_y_eco():
    """Test with longer text to trigger min_chunk_before_split thresholds."""
    
    # Create text long enough to potentially trigger semantic splitting
    # The threshold is min_chunk_before_split=18 words
    prefix = "Bueno en este episodio vamos a seguir hablando de nuestra historia pero ya más sobre la historia de Spanish Land School para aquellos que han estado escuchando nuestro podcast por un tiempo"
    problem_part = "saben que nosotros hemos estado haciendo una serie de episodios sobre toda nuestra historia desde que éramos niños hasta ahora así que hoy vamos a hablar de cómo inició Spanish Land School y básicamente hablaremos de los dos primeros años de Spanish Land School y les contaremos todos los detalles pero si tú quieres escuchar los episodios anteriores recuerdo que fue como el veintidós o veintitrés de marzo que empezamos a grabar los videos así que estaba en Colombia pero tenía su trabajo remoto y trabajaba de nueve de la mañana a cinco de la tarde y yo tenía mi universidad todavía tenía algunas clases y ahora estaba empezando a grabar estos videos así que fue una locura sí todo el mes de marzo estuvimos grabando por un mes por un mes grabamos seis videos y los editamos en el lapso de un mes porque el canal de YouTube empezó el veintitrés de abril a finales de abril sí y para los que han visto los primeros videos no eran o no fueron no eran ah sí no eran muy muy buenos teníamos muchos errores y eco"
    
    text = prefix + " " + problem_part
    
    print("="*70)
    print("Testing very long text with 'y eco' (triggers semantic split)")
    print("="*70)
    print(f"Input length: {len(text.split())} words")
    print(f"Last 50 chars: ...{text[-50:]}")
    print()
    
    restored = restore_punctuation(text, language='es')
    sentences, trailing = assemble_sentences_from_processed(restored, 'es')
    
    print(f"Split into {len(sentences)} sentences")
    print()
    print("Last 3 sentences:")
    for sent in sentences[-3:]:
        print(f"  - {sent[:80]}..." if len(sent) > 80 else f"  - {sent}")
    print()
    
    # Check for bug
    bug_found = False
    for i, sent in enumerate(sentences):
        sent_stripped = sent.strip().rstrip('.!?')
        if sent_stripped.endswith(' y'):
            print(f"❌ BUG FOUND: Sentence ends with 'y':")
            print(f"   ...{sent[-80:]}")
            bug_found = True
            break
    
    if not bug_found:
        print("✅ SUCCESS: No sentences end with 'y'")
        return True
    else:
        return False


if __name__ == "__main__":
    results = []
    
    results.append(test_y_eco_from_segments())
    print()
    results.append(test_simpler_y_eco_case())
    print()
    results.append(test_very_long_text_with_y_eco())
    print()
    
    print("="*70)
    if all(results):
        print("🎉 All tests passed!")
        print("="*70)
        exit(0)
    else:
        print(f"❌ {results.count(False)}/{len(results)} tests failed")
        print("="*70)
        exit(1)

