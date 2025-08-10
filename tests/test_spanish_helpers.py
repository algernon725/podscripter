#!/usr/bin/env python3
"""
Focused unit tests for Spanish helper utilities in punctuation_restorer.py

Covers:
- Tag question normalization
- Collocation repairs
- Possessive/aux+gerund/capitalized-split merges
- Inverted question pairing
- Greeting/lead-in commas
- Imperative exclamation wrapping
"""

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import punctuation_restorer as pr  # noqa: E402


def test_es_normalize_tag_questions():
    # Unchanged when ending with dot at end-of-string
    assert pr._es_normalize_tag_questions("Está bien, ¿no.") == "Está bien, ¿no."
    # Normalizes missing comma before tag
    assert pr._es_normalize_tag_questions("Está bien ¿verdad?") == "Está bien, ¿verdad?"


def test_es_fix_collocations():
    # Repair "por? supuesto" and add comma when sentence-initial
    s = "por? supuesto vamos a empezar."
    out = pr._es_fix_collocations(s)
    assert out.startswith("Por supuesto,")


def test_es_merge_possessive_splits():
    assert pr._es_merge_possessive_splits("tu. Español") == "tu español"
    assert pr._es_merge_possessive_splits("Mi. Amigo") == "Mi amigo"


def test_es_merge_aux_gerund():
    # Helper preserves inner capitalization; full pipeline may lowercase later
    assert pr._es_merge_aux_gerund("Estamos. Hablando") == "Estamos Hablando"


def test_es_merge_capitalized_one_word_sentences():
    assert pr._es_merge_capitalized_one_word_sentences("Estados. Unidos.") == "Estados Unidos."


def test_es_pair_inverted_questions():
    assert pr._es_pair_inverted_questions("¿Dónde está.") == "¿Dónde está?"
    assert pr._es_pair_inverted_questions("Cómo estás?") == "¿Cómo estás?"


def test_es_greeting_and_leadin_commas():
    # Moves period to comma before a following question; does not rewrite content
    assert pr._es_greeting_and_leadin_commas("Hola como estan. ¿Listos?") == "Hola como estan, ¿Listos?"
    assert pr._es_greeting_and_leadin_commas("Hola, a todos") == "Hola a todos"
    assert pr._es_greeting_and_leadin_commas("Como siempre vamos a revisar") == "Como siempre, vamos a revisar"


def test_es_wrap_imperative_exclamations():
    assert pr._es_wrap_imperative_exclamations("Vamos a empezar.") == "¡Vamos a empezar!"
    assert pr._es_wrap_imperative_exclamations("Bienvenidos a Españolistos.") == "¡Bienvenidos a Españolistos!"


if __name__ == "__main__":
    # Run tests directly
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All Spanish helper tests passed")


