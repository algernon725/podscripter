#!/usr/bin/env python3
"""
Focused test for the four specific Spanish transcription bugs mentioned by the user.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punctuation_restorer import restore_punctuation


def test_specific_bugs():
    """Test the four specific bugs mentioned by the user."""
    
    print("Testing the four specific Spanish transcription bugs...")
    print("=" * 70)
    
    # Bug 1: Missing punctuation at the end of sentences
    print("\nBUG 1: Missing punctuation at the end of sentences")
    print("-" * 55)
    
    bug1_cases = [
        "Exactamente, yo sé muy poco de todo",
        "Todo en general en esta área nunca he tenido muy buen conocimiento",
        "También sí",
        "Es el que le dice al empleado que ya no va a trabajar más"
    ]
    
    for text in bug1_cases:
        result = restore_punctuation(text, 'es')
        print(f"Input:  {text}")
        print(f"Output: {result}")
        print()
    
    # Bug 2: Sentences ending in commas
    print("\nBUG 2: Sentences ending in commas")
    print("-" * 35)
    
    bug2_cases = [
        "Todo en general en esta área nunca he tenido muy buen conocimiento,",
        "También sí,",
        "Es el que le dice al empleado que ya no va a trabajar más,"
    ]
    
    for text in bug2_cases:
        result = restore_punctuation(text, 'es')
        print(f"Input:  {text}")
        print(f"Output: {result}")
        print()
    
    # Bug 3: Inverted question marks in the middle of sentences
    print("\nBUG 3: Inverted question marks in the middle of sentences")
    print("-" * 55)
    
    bug3_cases = [
        "No se ha desestabilizado la economía, pero sabemos que en la mayoría sí.¿Así que por eso queremos analizar un poquito por qué está sucediendo esto y muchos de ustedes ya sabrán esas razones, si les gusta leer,",
        "Despidan a los empleados por los siguientes dos o tres meses, ¿no",
        "Sé cómo explicar un bailout en español, ¿tú sabes?¿No sé qué es eso"
    ]
    
    for text in bug3_cases:
        result = restore_punctuation(text, 'es')
        print(f"Input:  {text}")
        print(f"Output: {result}")
        print()
    
    # Bug 4: Question marks followed by inverted question marks in the middle
    print("\nBUG 4: Question marks followed by inverted question marks in the middle")
    print("-" * 70)
    
    bug4_cases = [
        "Vamos a levantar de esto, vamos a salir de esto, tenemos que reconocer.?¿La realidad, estar conscientes de qué está pasando alrededor de nosotros, ser conscientes de que sí"
    ]
    
    for text in bug4_cases:
        result = restore_punctuation(text, 'es')
        print(f"Input:  {text}")
        print(f"Output: {result}")
        print()


if __name__ == "__main__":
    test_specific_bugs()
