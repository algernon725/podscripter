#!/usr/bin/env python3
"""
Tests for the bound-comparative sentence-break guard.

Root cause (Episodio300): the PRIORITY-5 semantic heuristic in
``SentenceSplitter`` could split a long single-speaker chunk immediately before
a comparative particle that binds backward to a degree/quantity word, severing
fixed phrases like "un poco | más baratos". These tests cover the guard
(``_is_bound_comparative_break``) directly and verify it suppresses the split in
the semantic path, across all supported languages (es/en/fr/de).
"""

import numpy as np
import pytest

from sentence_splitter import SentenceSplitter
from punctuation_restorer import _get_language_config

pytestmark = pytest.mark.core


class LowSimilarityModel:
    """Mock SentenceTransformer whose encodings are mutually orthogonal, so
    ``_check_semantic_break`` always sees cosine similarity ~0 (< threshold)
    and would split at every eligible position unless a guard prevents it."""

    def encode(self, texts):
        # Return distinct one-hot rows -> cosine similarity 0 between them.
        n = len(texts)
        return np.eye(n, max(n, 2), dtype=float)


def _make_splitter(language: str) -> SentenceSplitter:
    return SentenceSplitter(language, LowSimilarityModel(), _get_language_config(language))


# ---------------------------------------------------------------------------
# Unit tests for the guard predicate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("language,current,nxt", [
    # (a) break BEFORE the particle, bound to a preceding degree head
    ('es', 'poco', 'más'),
    ('es', 'poco', 'menos'),
    ('es', 'mucho', 'más'),
    ('es', 'vez', 'más'),      # "cada vez más"
    ('es', 'nada', 'más'),
    ('en', 'little', 'more'),
    ('en', 'lot', 'more'),     # "a lot more"
    ('en', 'far', 'fewer'),
    ('fr', 'peu', 'plus'),     # "un peu plus"
    ('fr', 'beaucoup', 'moins'),
    ('fr', 'encore', 'plus'),
    ('de', 'etwas', 'mehr'),   # "etwas mehr"
    ('de', 'viel', 'weniger'),
    ('de', 'noch', 'mehr'),
    # (b) break AFTER the particle, before the (lowercase) modified adjective
    ('es', 'más', 'baratos'),   # "más baratos"
    ('es', 'menos', 'gente'),
    ('en', 'more', 'expensive'),
    ('fr', 'plus', 'chers'),
    ('de', 'mehr', 'teuer'),
])
def test_guard_positive(language, current, nxt):
    splitter = _make_splitter(language)
    assert splitter._is_bound_comparative_break(current, nxt) is True


@pytest.mark.parametrize("language,current,nxt", [
    # A comparative particle followed by a CAPITALIZED word was sentence-final
    # ("un poco más. Ahora…") and must stay splittable.
    ('es', 'más', 'Ahora'),
    ('en', 'more', 'Later'),
    ('fr', 'plus', 'Maintenant'),
    ('de', 'mehr', 'Jetzt'),
])
def test_guard_allows_split_after_sentence_final_particle(language, current, nxt):
    splitter = _make_splitter(language)
    assert splitter._is_bound_comparative_break(current, nxt) is False


@pytest.mark.parametrize("language,current,nxt", [
    # Degree head but next word is NOT a comparative particle -> allowed
    ('es', 'poco', 'baratos'),
    ('en', 'little', 'expensive'),
    ('fr', 'peu', 'chers'),
    ('de', 'etwas', 'teurer'),
    # Comparative particle but preceding word is NOT a degree head ->
    # sentence-initial "Más/More/Plus/Mehr …" must stay splittable
    ('es', 'terminado', 'más'),
    ('en', 'done', 'more'),
    ('fr', 'fini', 'plus'),
    ('de', 'fertig', 'mehr'),
    # Neither side matches
    ('es', 'costos', 'baratos'),
])
def test_guard_negative(language, current, nxt):
    splitter = _make_splitter(language)
    assert splitter._is_bound_comparative_break(current, nxt) is False


def test_guard_handles_punctuation_and_case():
    splitter = _make_splitter('es')
    # Trailing punctuation and capitalization should not defeat the guard.
    assert splitter._is_bound_comparative_break('poco', 'Más') is True
    assert splitter._is_bound_comparative_break('Poco,', 'más') is True


def test_guard_unknown_language_returns_false():
    # Languages with no entry in the dicts must never raise or match.
    splitter = _make_splitter('it')
    assert splitter._is_bound_comparative_break('poco', 'più') is False


# ---------------------------------------------------------------------------
# Isolation tests: the guard suppresses the split in the semantic path, while
# a non-bound position at the same length still splits (proving the semantic
# path is live and the False at the bound position is due to the guard).
# ---------------------------------------------------------------------------

# Authentic Episodio300 run (raw Whisper segments 56-58, single speaker).
_ES_S56 = "Sí, una pregunta muy importante porque si vas a vivir allá por muchos años, debes saber los costos."
_ES_S57 = "Y, pues, el razón por que hay muchos expats en Colombia es que es un muy bonito lugar para vivir."
_ES_S58 = "Y los costos son un poco más baratos para los de los países de Europa o de Estados Unidos, Australia."
_ES_WORDS = (_ES_S56 + " " + _ES_S57 + " " + _ES_S58).split()


def _should_end(splitter, words, index, chunk_len=None):
    chunk = words[:(chunk_len if chunk_len is not None else index + 1)]
    return splitter._should_end_sentence_here(
        words, index, chunk,
        whisper_word_boundaries=None,
        speaker_word_boundaries=None,
        speaker_word_segments=None,
    )


def test_es_semantic_split_suppressed_at_bound_comparative():
    splitter = _make_splitter('es')
    poco_idx = _ES_WORDS.index('poco')
    assert _ES_WORDS[poco_idx + 1].strip('.,') == 'más'
    # Guarded: must NOT end the sentence between "poco" and "más".
    assert _should_end(splitter, _ES_WORDS, poco_idx) is False


def test_es_semantic_split_suppressed_after_particle():
    splitter = _make_splitter('es')
    mas_idx = _ES_WORDS.index('más')
    assert _ES_WORDS[mas_idx + 1].strip('.,') == 'baratos'
    # Guarded: must NOT end the sentence between "más" and "baratos".
    assert _should_end(splitter, _ES_WORDS, mas_idx) is False


def test_es_semantic_split_still_fires_at_unbound_position():
    splitter = _make_splitter('es')
    # "baratos" -> "para": not a bound comparative; semantic path is eligible
    # (chunk >= 42 words) and the low-similarity model triggers a break.
    baratos_idx = _ES_WORDS.index('baratos')
    assert bool(_should_end(splitter, _ES_WORDS, baratos_idx)) is True


def _synthetic_run(head: str, particle: str, tail: str, filler_len: int = 40):
    filler = ['cosa'] * filler_len
    words = filler + [head, particle, tail] + ['cosa'] * 5
    return words, len(filler)  # head is at index == filler_len


@pytest.mark.parametrize("language,head,particle,tail", [
    ('en', 'little', 'more', 'expensive'),
    ('fr', 'peu', 'plus', 'chers'),
    ('de', 'etwas', 'mehr', 'teuer'),
])
def test_other_languages_semantic_split_suppressed(language, head, particle, tail):
    splitter = _make_splitter(language)
    words, head_idx = _synthetic_run(head, particle, tail)
    assert words[head_idx] == head and words[head_idx + 1] == particle
    # Guarded at the head->particle boundary...
    assert _should_end(splitter, words, head_idx) is False
    # ...but a later non-bound position still splits (semantic path is live).
    tail_idx = head_idx + 2
    assert bool(_should_end(splitter, words, tail_idx)) is True
