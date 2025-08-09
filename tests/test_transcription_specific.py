#!/usr/bin/env python3
"""
Specific regression tests for Spanish intro formatting and exclamation handling
using a snippet from Episodio174 human-verified transcript.
"""

import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punctuation_restorer import restore_punctuation


def normalize(text: str) -> str:
    t = re.sub(r"\s+", " ", text.strip())
    t = re.sub(r"\s+([,.!?])", r"\1", t)
    return t

def test_no_exclamation_plus_period():
    out = restore_punctuation("Hola a todos bienvenidos a Españolistos", "es")
    norm = normalize(out)
    assert "!." not in norm, f"Found forbidden sequence '!.': {norm}"

def test_greeting_and_question_formatting():
    out = normalize(restore_punctuation("hola para todos como estan", "es"))
    # Be robust to lack of diacritics in input; require comma after greeting and inverted question start
    assert out.startswith("Hola para todos, ¿"), out


if __name__ == "__main__":
    # Simple manual run of the assertions
    test_no_exclamation_plus_period()
    test_greeting_and_question_formatting()
    print("OK")
