#!/usr/bin/env python3
"""
Test to verify Spanish run-on sentence fix
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from punctuation_restorer import restore_punctuation

def test_spanish_runon_fix():
    """Test that Spanish text is properly split into sentences."""
    
    print("Testing Spanish Run-on Sentence Fix")
    print("=" * 60)
    
    # The problematic run-on text from the user
    runon_text = """Hola a todos  Bienvenidos a Españolistos  Españolistos es el podcast que te va a ayudar a estar listo para hablar español  Españolistos te prepara para hablar español en cualquier lugar, a cualquier hora y en cualquier situación  ¿Recuerdas todos esos momentos en los que no ¿Supiste qué decir  ¿Esos momentos en los que no pudiste mantener una conversación  Pues tranquilo,  Españolistos es la herramienta que estabas buscando para mejorar tu español  Dile adiós  A todos esos momentos incómodos  Entonces, empecemos  ¿Estamos listos  Yo soy Andrea de Santander, Colombia  Y yo soy Nate de Texas, Estados Unidos  Hola para todos"""
    
    print("Input (run-on text):")
    print(repr(runon_text))
    print()
    
    # Process the text
    result = restore_punctuation(runon_text, language='es')
    
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
        "Hola a todos",
        "Bienvenidos a Españolistos",
        "¿Recuerdas todos esos momentos",
        "Yo soy Andrea de Santander, Colombia",
        "Y yo soy Nate de Texas, Estados Unidos"
    ]
    
    print("\nChecking for expected sentence patterns:")
    for pattern in expected_patterns:
        if pattern in result:
            print(f"✅ Found: {pattern}")
        else:
            print(f"❌ Missing: {pattern}")

if __name__ == "__main__":
    test_spanish_runon_fix() 