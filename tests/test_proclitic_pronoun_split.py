#!/usr/bin/env python3
"""
Tests for the Spanish proclitic-pronoun sentence-break guard.

Root cause (Episodio300): the PRIORITY-5 semantic heuristic split a long
single-speaker chunk between a proclitic object/reflexive pronoun and its verb
("… no estás usando AirBnB, me | imagino que …"). Spanish proclitic pronouns
(me/te/se/nos/os/le/les/lo) attach forward to the following verb, so a sentence
can never end on one. The fix adds them to the Spanish branch of
``_violates_grammatical_rules`` (PRIORITY 2, "never break here").
"""

import numpy as np
import pytest

from sentence_splitter import SentenceSplitter
from punctuation_restorer import _get_language_config

pytestmark = pytest.mark.core


class LowSimilarityModel:
    """Mock model whose encodings are orthogonal -> cosine ~0, so the semantic
    path would split at every eligible position unless a guard prevents it."""

    def encode(self, texts):
        n = len(texts)
        return np.eye(n, max(n, 2), dtype=float)


def _make_splitter(language: str) -> SentenceSplitter:
    return SentenceSplitter(language, LowSimilarityModel(), _get_language_config(language))


@pytest.mark.parametrize("pronoun", ['me', 'te', 'se', 'nos', 'os', 'le', 'les', 'lo'])
def test_proclitic_pronoun_forbidden_as_sentence_end(pronoun):
    splitter = _make_splitter('es')
    assert splitter._violates_grammatical_rules(pronoun, 'imagino') is True
    # Case/punctuation-insensitive (guard lowercases + strips).
    assert splitter._violates_grammatical_rules(pronoun.capitalize(), 'verbo') is True


@pytest.mark.parametrize("word", ['imagino', 'casa', 'importante', 'mesa'])
def test_content_words_still_allowed_as_sentence_end(word):
    splitter = _make_splitter('es')
    assert splitter._violates_grammatical_rules(word, 'que') is False


@pytest.mark.parametrize("apocope", ['buen', 'gran', 'primer', 'tercer', 'algún', 'san'])
def test_apocopated_adjective_forbidden_as_sentence_end(apocope):
    splitter = _make_splitter('es')
    assert splitter._violates_grammatical_rules(apocope, 'apartamento') is True
    assert splitter._violates_grammatical_rules(apocope.capitalize(), 'hombre') is True


@pytest.mark.parametrize("word", ['bueno', 'malo', 'mal', 'grande', 'primero'])
def test_full_forms_and_adverb_mal_still_allowed(word):
    # Non-apocopated forms and the adverb "mal" can legitimately end a sentence.
    splitter = _make_splitter('es')
    assert splitter._violates_grammatical_rules(word, 'y') is False


# Isolation: reconstructed long Episodio300 run (raw segment 66-70 region),
# single speaker, so the semantic path is eligible (chunk >> 42 words).
_ES_WORDS = (
    "Vivienda cierto pues bueno Pues nosotros usamos AirBnB y este AirBnB es más "
    "o menos como ochocientos dólares al mes pero esto está todo amoblado con "
    "todas las cosas pero si tú vas a vivir allá y tú pues no estás usando AirBnB "
    "me imagino que puedes encontrar un buen apartamento entre trescientos hasta "
    "setecientos dólares al mes"
).split()


def _should_end(splitter, words, index):
    return splitter._should_end_sentence_here(
        words, index, words[:index + 1],
        whisper_word_boundaries=None,
        speaker_word_boundaries=None,
        speaker_word_segments=None,
    )


def test_semantic_split_suppressed_at_proclitic_pronoun():
    splitter = _make_splitter('es')
    me_idx = _ES_WORDS.index('me')
    assert _ES_WORDS[me_idx + 1] == 'imagino'
    # Guarded: must NOT end the sentence between "me" and "imagino".
    assert _should_end(splitter, _ES_WORDS, me_idx) is False


def test_semantic_split_still_fires_at_content_word():
    splitter = _make_splitter('es')
    # "apartamento" -> "entre": an unguarded content-word junction past the es
    # semantic threshold (chunk >= 42); the low-similarity model triggers a
    # break, proving the guards (not text length) are what suppress the splits
    # at the pronoun/hinge/modal positions.
    apartamento_idx = _ES_WORDS.index('apartamento')
    assert apartamento_idx + 1 >= 42
    assert bool(_should_end(splitter, _ES_WORDS, apartamento_idx)) is True


def test_semantic_split_suppressed_before_subordinating_que():
    splitter = _make_splitter('es')
    # "imagino" -> "que": subordinating "que" attaches backward and must not
    # start a new sentence. Even though "imagino" is a valid sentence-final
    # verb, the following "que" blocks the semantic split.
    imagino_idx = _ES_WORDS.index('imagino')
    assert _ES_WORDS[imagino_idx + 1] == 'que'
    assert _should_end(splitter, _ES_WORDS, imagino_idx) is False


def test_semantic_split_suppressed_after_subordinating_que():
    splitter = _make_splitter('es')
    # "que" -> "puedes": the hinge "que" binds forward to the clause it
    # introduces, so a break on its trailing edge is ungrammatical too.
    que_idx = _ES_WORDS.index('que')
    assert _ES_WORDS[que_idx + 1] == 'puedes'
    assert _should_end(splitter, _ES_WORDS, que_idx) is False


def test_semantic_split_suppressed_between_modal_and_infinitive():
    splitter = _make_splitter('es')
    # "puedes" -> "encontrar": modal + governed infinitive (periphrasis).
    puedes_idx = _ES_WORDS.index('puedes')
    assert _ES_WORDS[puedes_idx + 1] == 'encontrar'
    assert _should_end(splitter, _ES_WORDS, puedes_idx) is False


@pytest.mark.parametrize("current,nxt", [
    ('puedes', 'encontrar'),
    ('quiero', 'comer'),
    ('debo', 'ir'),            # short/irregular infinitive
    ('puede', 'ser'),          # short/irregular infinitive
    ('suele', 'hacerlo'),      # enclitic-carrying infinitive
    ('sabes', 'nadar'),
])
def test_modal_infinitive_predicate_positive(current, nxt):
    splitter = _make_splitter('es')
    assert splitter._is_modal_infinitive_break(current, nxt) is True


@pytest.mark.parametrize("current,nxt", [
    ('puedes', 'Hazlo'),       # capitalized -> modal was sentence-final
    ('puedes', 'mucho'),       # not an infinitive
    ('quiero', 'eso'),         # object pronoun, not an infinitive
    ('casa', 'encontrar'),     # current not a modal governor
])
def test_modal_infinitive_predicate_negative(current, nxt):
    splitter = _make_splitter('es')
    assert splitter._is_modal_infinitive_break(current, nxt) is False


def test_semantic_split_suppressed_between_infinitive_and_complement():
    splitter = _make_splitter('es')
    # "encontrar" -> "un": infinitive severed from its object.
    encontrar_idx = _ES_WORDS.index('encontrar')
    assert _ES_WORDS[encontrar_idx + 1] == 'un'
    assert _should_end(splitter, _ES_WORDS, encontrar_idx) is False


@pytest.mark.parametrize("current,nxt", [
    ('encontrar', 'un'),       # infinitive + object
    ('comer', 'algo'),
    ('hablar', 'español'),
])
def test_infinitive_complement_predicate_positive(current, nxt):
    splitter = _make_splitter('es')
    assert splitter._is_infinitive_complement_break(current, nxt) is True


@pytest.mark.parametrize("current,nxt", [
    ('encontrar', 'Un'),       # capitalized -> infinitive was sentence-final
    ('apartamento', 'entre'),  # current not an infinitive (noun)
    ('bien', 'hecho'),         # current not an infinitive
])
def test_infinitive_complement_predicate_negative(current, nxt):
    splitter = _make_splitter('es')
    assert splitter._is_infinitive_complement_break(current, nxt) is False
