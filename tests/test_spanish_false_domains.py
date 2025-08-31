#!/usr/bin/env python3
"""
Test for Spanish false domain detection fix.

This test ensures that common Spanish words followed by TLD-like suffixes 
are not incorrectly treated as domain names.

Example bug: "uno.de los lugares" was incorrectly preserved as "uno.de" domain
instead of being split into "uno. de" (number followed by preposition).
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from podscripter import _assemble_sentences
from punctuation_restorer import restore_punctuation


def test_spanish_false_domains():
    """Test that common Spanish words + TLD suffixes are not treated as domains."""
    
    # Test cases: [input, expected_behavior_description]
    test_cases = [
        ("Y que es uno.de los lugares más caros", "uno.de should split into 'uno. de'"),
        ("Este.es muy importante para nosotros", "Este.es should split into 'Este. es'"),
        ("Son dos.com de los mejores sitios", "dos.com should split into 'dos. com'"),
        ("Hay tres.es opciones disponibles", "tres.es should split into 'tres. es'"),
        ("El.co es una buena opción", "El.co should split into 'El. co'"),
        ("Muy.de acuerdo con la propuesta", "Muy.de should split into 'Muy. de'"),
        # New cases for common Spanish preposition "de"
        ("Necesita ser tratada.de hecho", "tratada.de should split into 'tratada. de'"),
        ("Era una noche.de verano", "noche.de should split into 'noche. de'"),
        ("La historia.de siempre", "historia.de should split into 'historia. de'"),
        ("Un poco.de todo", "poco.de should split into 'poco. de'"),
    ]
    
    # Real domains that should be preserved
    real_domain_cases = [
        ("Visita github.de para el código", "github.de should be preserved as domain"),
        ("Ve a google.com para buscar", "google.com should be preserved as domain"),
        ("Consulta marca.es para noticias", "marca.es should be preserved as domain"),
        ("Accede a amazon.co.uk para comprar", "amazon.co.uk should be preserved as domain"),
    ]
    
    print("Testing Spanish false domain detection...")
    
    # Test false domains (should be split)
    for test_input, description in test_cases:
        print(f"\nTesting: {test_input}")
        print(f"Expected: {description}")
        
        # Test with sentence assembly
        result = _assemble_sentences(test_input, 'es', quiet=True)
        output = result[0] if result else test_input
        
        # Check if it was correctly split
        correctly_split = False
        if ".de" in test_input and ". de" in output.lower():
            correctly_split = True
        elif ".es" in test_input and ". es" in output.lower():
            correctly_split = True
        elif ".com" in test_input and ". com" in output.lower():
            correctly_split = True
        elif ".co" in test_input and ". co" in output.lower():
            correctly_split = True
            
        if correctly_split:
            print(f"✅ PASS: {output}")
        else:
            print(f"❌ FAIL: {output}")
            print(f"   Expected word splitting, but got potential domain preservation")
    
    # Test real domains (should be preserved)
    for test_input, description in real_domain_cases:
        print(f"\nTesting: {test_input}")
        print(f"Expected: {description}")
        
        result = _assemble_sentences(test_input, 'es', quiet=True)
        output = result[0] if result else test_input
        
        # Check if domain was preserved
        if ("github.de" in output or "google.com" in output or "marca.es" in output or "amazon.co.uk" in output):
            print(f"✅ PASS: {output}")
        else:
            print(f"❌ FAIL: {output}")
            print(f"   Expected domain preservation, but got word splitting")
    
    print("\nTesting complete.")


def test_punctuation_restoration_false_domains():
    """Test punctuation restoration doesn't create false domains."""
    
    test_cases = [
        "Y que es uno.de los lugares más caros para visitar",
        "Este.es muy importante para el proyecto",
        "Son dos.com de los mejores sitios web",
    ]
    
    print("\nTesting punctuation restoration with false domains...")
    
    for test_input in test_cases:
        print(f"\nInput: {test_input}")
        
        result = restore_punctuation(test_input, 'es')
        print(f"Output: {result}")
        
        # Check that Spanish words are properly separated
        has_proper_separation = (
            ("uno. de" in result.lower() or "uno. De" in result) or
            ("este. es" in result.lower() or "Este. Es" in result) or  
            ("dos. com" in result.lower() or "dos. Com" in result)
        )
        
        if has_proper_separation:
            print("✅ PASS: Spanish words properly separated")
        else:
            print("❌ FAIL: Spanish words may be incorrectly preserved as domains")


if __name__ == "__main__":
    test_spanish_false_domains()
    test_punctuation_restoration_false_domains()
    print("\nAll Spanish false domain tests completed.")
