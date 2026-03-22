#!/usr/bin/env python3

import pytest

import punctuation_restorer as pr  # noqa: E402

pytestmark = pytest.mark.core


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


def test_es_normalize_tag_questions():
    assert pr._es_normalize_tag_questions("Está bien, ¿no.") == "Está bien, ¿no."
    assert pr._es_normalize_tag_questions("Está bien ¿verdad?") == "Está bien, ¿verdad?"


def test_es_fix_collocations():
    s = "por? supuesto vamos a empezar."
    out = pr._es_fix_collocations(s)
    assert out.startswith("Por supuesto,")


def test_es_merge_possessive_splits():
    assert pr._es_merge_possessive_splits("tu. Español") == "tu español"
    assert pr._es_merge_possessive_splits("Mi. Amigo") == "Mi amigo"


def test_es_merge_aux_gerund():
    assert pr._es_merge_aux_gerund("Estamos. Hablando") == "Estamos Hablando"


def test_es_merge_capitalized_one_word_sentences():
    assert pr._es_merge_capitalized_one_word_sentences("Estados. Unidos.") == "Estados Unidos."


def test_es_pair_inverted_questions():
    assert pr._es_pair_inverted_questions("¿Dónde está.") == "¿Dónde está?"
    assert pr._es_pair_inverted_questions("Cómo estás?") == "¿Cómo estás?"


def test_es_greeting_and_leadin_commas():
    assert pr._es_greeting_and_leadin_commas("Hola como estan. ¿Listos?") == "Hola como estan, ¿Listos?"
    assert pr._es_greeting_and_leadin_commas("Hola, a todos") == "Hola a todos"
    assert pr._es_greeting_and_leadin_commas("Como siempre vamos a revisar") == "Como siempre, vamos a revisar"
    assert pr._es_greeting_and_leadin_commas("Hola para todos, ¿Cómo están?") == "Hola para todos, ¿Cómo están?"
    assert pr._es_greeting_and_leadin_commas("Hola para todos ¿Cómo están?") == "Hola para todos, ¿Cómo están?"


def test_es_wrap_imperative_exclamations():
    assert pr._es_wrap_imperative_exclamations("Vamos a empezar.") == "¡Vamos a empezar!"
    assert pr._es_wrap_imperative_exclamations("Bienvenidos a Españolistos.") == "¡Bienvenidos a Españolistos!"


def test_es_mid_sentence_exclamation_closure():
    s = "Entonces, ¡empecemos."
    out = pr._spanish_cleanup_postprocess(s)
    assert out.endswith("¡empecemos!") or out == "Entonces, ¡empecemos!"
    s2 = "Bueno, ¡vamos."
    out2 = pr._spanish_cleanup_postprocess(s2)
    assert out2.endswith("¡vamos!") or out2 == "Bueno, ¡vamos!"


def test_duplicate_commas_are_deduped_in_finalize():
    s = "Hola, , descubrí este podcast hace tres años."
    out = pr._finalize_text_common(s)
    assert out == "Hola, descubrí este podcast hace tres años.", out
