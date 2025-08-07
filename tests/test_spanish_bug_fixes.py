#!/usr/bin/env python3
"""
Test file for Spanish transcription bug fixes.
Tests the four specific bugs mentioned by the user.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punctuation_restorer import restore_punctuation


def test_spanish_bug_fixes():
    """Test the four Spanish transcription bugs mentioned by the user."""
    
    print("Testing Spanish transcription bug fixes...")
    print("=" * 60)
    
    # Bug 1: Missing punctuation at the end of sentences
    print("\nBUG 1: Missing punctuation at the end of sentences")
    print("-" * 50)
    
    test_cases_bug1 = [
        "Exactamente, yo sé muy poco de todo",
        "Todo en general en esta área nunca he tenido muy buen conocimiento",
        "También sí",
        "Es el que le dice al empleado que ya no va a trabajar más"
    ]
    
    for text in test_cases_bug1:
        result = restore_punctuation(text, 'es')
        print(f"Input:  {text}")
        print(f"Output: {result}")
        print()
    
    # Bug 2: Sentences ending in commas
    print("\nBUG 2: Sentences ending in commas")
    print("-" * 40)
    
    test_cases_bug2 = [
        "Todo en general en esta área nunca he tenido muy buen conocimiento,",
        "También sí,",
        "Es el que le dice al empleado que ya no va a trabajar más,"
    ]
    
    for text in test_cases_bug2:
        result = restore_punctuation(text, 'es')
        print(f"Input:  {text}")
        print(f"Output: {result}")
        print()
    
    # Bug 3: Inverted question marks in the middle of sentences
    print("\nBUG 3: Inverted question marks in the middle of sentences")
    print("-" * 55)
    
    test_cases_bug3 = [
        "No se ha desestabilizado la economía, pero sabemos que en la mayoría sí.¿Así que por eso queremos analizar un poquito por qué está sucediendo esto y muchos de ustedes ya sabrán esas razones, si les gusta leer,",
        "Despidan a los empleados por los siguientes dos o tres meses, ¿no",
        "Sé cómo explicar un bailout en español, ¿tú sabes?¿No sé qué es eso"
    ]
    
    for text in test_cases_bug3:
        result = restore_punctuation(text, 'es')
        print(f"Input:  {text}")
        print(f"Output: {result}")
        print()
    
    # Bug 4: Question marks followed by inverted question marks in the middle
    print("\nBUG 4: Question marks followed by inverted question marks in the middle")
    print("-" * 65)
    
    test_cases_bug4 = [
        "Vamos a levantar de esto, vamos a salir de esto, tenemos que reconocer.?¿La realidad, estar conscientes de qué está pasando alrededor de nosotros, ser conscientes de que sí"
    ]
    
    for text in test_cases_bug4:
        result = restore_punctuation(text, 'es')
        print(f"Input:  {text}")
        print(f"Output: {result}")
        print()
    
    # Additional test cases for comprehensive coverage
    print("\nADDITIONAL TEST CASES")
    print("-" * 25)
    
    additional_cases = [
        "Hola cómo estás hoy",
        "Qué hora es",
        "Dónde está la reunión",
        "Cuándo es la cita",
        "Cómo te llamas",
        "Quién puede ayudarme",
        "Cuál es tu nombre",
        "Por qué no viniste",
        "Recuerdas la última vez",
        "Sabes dónde queda",
        "Puedes ayudarme",
        "Quieres que vayamos",
        "Necesitas algo más"
    ]
    
    for text in additional_cases:
        result = restore_punctuation(text, 'es')
        print(f"Input:  {text}")
        print(f"Output: {result}")
        print()


def test_complex_spanish_text():
    """Test with more complex Spanish text that might have multiple issues."""
    
    print("\nCOMPLEX SPANISH TEXT TEST")
    print("=" * 30)
    
    complex_text = """
    Exactamente yo sé muy poco de todo Todo en general en esta área nunca he tenido muy buen conocimiento También sí Es el que le dice al empleado que ya no va a trabajar más No se ha desestabilizado la economía pero sabemos que en la mayoría sí ¿Así que por eso queremos analizar un poquito por qué está sucediendo esto y muchos de ustedes ya sabrán esas razones si les gusta leer Despidan a los empleados por los siguientes dos o tres meses ¿no Sé cómo explicar un bailout en español ¿tú sabes ¿No sé qué es eso Vamos a levantar de esto vamos a salir de esto tenemos que reconocer ¿La realidad estar conscientes de qué está pasando alrededor de nosotros ser conscientes de que sí
    """
    
    result = restore_punctuation(complex_text, 'es')
    print(f"Input:  {complex_text.strip()}")
    print(f"Output: {result}")


if __name__ == "__main__":
    test_spanish_bug_fixes()
    test_complex_spanish_text()
