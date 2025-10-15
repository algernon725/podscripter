#!/usr/bin/env python3
"""
Test for the "estaba" sentence break bug with LONG TEXT.

Bug: With long text (>219 words), sentences incorrectly break at continuative/
auxiliary verbs like "estaba" when semantic splitting thresholds are triggered.

This is similar to the coordinating conjunction bug - it only manifests when
the text is long enough to trigger semantic sentence splitting.

Example from Episodio190 (segments 43-70):
The text should keep "y yo estaba en Colombia y estaba continuando" together,
but instead breaks at "estaba."
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from punctuation_restorer import restore_punctuation


def test_estaba_break_with_long_text():
    """
    Test that 'estaba' doesn't break with long text that triggers semantic splitting.
    
    This reproduces the actual bug from Episodio190 where the sentence incorrectly
    broke at "estaba" due to semantic splitting in long text.
    """
    
    # From Episodio190 segments 43-70 - build up enough context to trigger semantic split
    long_text = """
    siempre estamos pensando en nuestro primer mes de matrimonio porque después de la boda 
    estamos muy tranquilos y disfrutando la vida y fue muy divertido cierto sí por supuesto 
    el primer mes cuando uno se casa es la parte más fácil donde más se disfruta entonces 
    pasábamos mucho tiempo juntos estábamos en Austin Texas enfrentamos un apartamento y 
    quedaba cerca de un lago y pues íbamos a caminar salíamos a comer yo tenía tiempo de 
    cocinar pues porque ahí yo estaba de vacaciones en la universidad y realmente yo no 
    tenía un trabajo yo estaba dictando clases en una plataforma online entonces Nate iba 
    a su trabajo y regresaba y ya después de las cinco de la tarde teníamos como todo el 
    tiempo libre y los fines de semana también pero sí fue un tiempo muy muy bonito y muy 
    muy relajado y después qué pasó yo tenía una loca idea bueno creo que tú pensaste en 
    hacer algo no o tú no estabas pensando en nada no yo es que lo que pasó es que en el 
    dos mil diecisiete en el mes de enero el primer mes estuvimos juntos allá en Austin 
    Texas pero después yo regresé a Colombia a finales de enero y Nate se quedó en Austin 
    fue muy loco porque nos casamos estuvimos juntos por un mes y después un mes separados 
    y yo estaba en Colombia y estaba continuando con la universidad porque yo no me había 
    graduado ese era mi último semestre
    """.replace('\n', ' ')
    
    result = restore_punctuation(long_text, language='es')
    
    print("=" * 80)
    print("TEST: Long text - 'estaba' should not break sentence")
    print("=" * 80)
    print(f"Input length: {len(long_text.split())} words")
    print()
    
    # Split result into sentences to analyze
    sentences = [s.strip() for s in result.replace('?', '.').replace('!', '.').split('.') if s.strip()]
    
    print("Output sentences:")
    for i, sent in enumerate(sentences, 1):
        print(f"{i}. {sent}")
    print()
    
    # Check for the bug: sentence incorrectly ending with "estaba"
    bug_found = False
    for sent in sentences:
        sent_lower = sent.lower().strip()
        # Check if sentence ends with just "estaba" (or similar continuative verbs)
        if sent_lower.endswith(' estaba') or sent_lower.endswith(' estaban'):
            print(f"✗ BUG FOUND: Sentence ends with continuative verb:")
            print(f"  '{sent}'")
            print()
            bug_found = True
        if sent_lower.endswith(' era') or sent_lower.endswith(' eran'):
            print(f"✗ BUG FOUND: Sentence ends with continuative verb:")
            print(f"  '{sent}'")
            print()
            bug_found = True
        if sent_lower.endswith(' tenía') or sent_lower.endswith(' tenían'):
            print(f"✗ BUG FOUND: Sentence ends with continuative verb:")
            print(f"  '{sent}'")
            print()
            bug_found = True
        if sent_lower.endswith(' había') or sent_lower.endswith(' habían'):
            print(f"✗ BUG FOUND: Sentence ends with continuative verb:")
            print(f"  '{sent}'")
            print()
            bug_found = True
    
    # The key phrase should stay together
    result_lower = result.lower()
    if "estaba continuando" not in result_lower and "estaba en colombia y estaba" in result_lower:
        print("✗ BUG: 'estaba continuando' was incorrectly split")
        bug_found = True
    
    assert not bug_found, "Sentence incorrectly breaks at continuative verb in long text"
    
    print("✓ Long text test passed: no breaks at continuative verbs")
    print()
    return result


def test_multiple_long_text_patterns():
    """Test various patterns with long text to ensure robustness"""
    
    test_cases = [
        {
            'name': 'estaba + gerund',
            'text': """
            nosotros empezamos una nueva vida cuando nos mudamos a la ciudad y todo era 
            muy diferente al principio pero nos acostumbramos rápido y empezamos a conocer 
            gente nueva y a explorar nuevos lugares cada fin de semana salíamos a descubrir 
            algo diferente y fue una experiencia increíble que nunca vamos a olvidar porque 
            aprendimos muchísimo y crecimos como personas y como pareja y yo estaba trabajando 
            en ese proyecto importante cuando me di cuenta de que necesitaba más tiempo
            """.replace('\n', ' '),
            'must_contain': 'estaba trabajando'
        },
        {
            'name': 'era + description',
            'text': """
            cuando empecé mi carrera en la universidad no sabía exactamente qué quería hacer 
            con mi vida pero tenía muchas ganas de aprender y de descubrir cosas nuevas cada 
            día era una aventura diferente y conocí a personas increíbles que me enseñaron 
            mucho sobre la vida y sobre mí mismo y poco a poco fui encontrando mi camino y 
            mi pasión y todo era muy emocionante porque estaba descubriendo quién era yo 
            realmente y qué quería hacer con mi futuro
            """.replace('\n', ' '),
            'must_contain': 'era muy emocionante'
        },
        {
            'name': 'tenía + object',
            'text': """
            en aquellos días cuando vivíamos en el campo la vida era muy tranquila y simple 
            nos levantábamos temprano cada mañana y trabajábamos en el jardín cultivando 
            vegetales y flores y cuidando de los animales y por las tardes nos sentábamos 
            en el porche a ver el atardecer y a hablar de nuestros sueños y planes para el 
            futuro y yo siempre tenía una sensación de paz y tranquilidad que nunca había 
            experimentado antes en la ciudad
            """.replace('\n', ' '),
            'must_contain': 'tenía una sensación'
        }
    ]
    
    print("=" * 80)
    print("TEST: Multiple long text patterns with continuative verbs")
    print("=" * 80)
    
    all_passed = True
    for test_case in test_cases:
        name = test_case['name']
        text = test_case['text']
        must_contain = test_case['must_contain']
        
        result = restore_punctuation(text, language='es')
        result_lower = result.lower()
        
        # Check that the phrase stays together
        if must_contain not in result_lower:
            print(f"✗ FAILED: {name}")
            print(f"  Expected phrase '{must_contain}' was split")
            print(f"  Result: {result}")
            print()
            all_passed = False
        else:
            print(f"✓ PASSED: {name}")
    
    print()
    assert all_passed, "Some long text pattern tests failed"
    print("✓ All long text pattern tests passed")
    print()


if __name__ == '__main__':
    test_estaba_break_with_long_text()
    test_multiple_long_text_patterns()
    
    print("=" * 80)
    print("ALL LONG TEXT TESTS PASSED!")
    print("=" * 80)

