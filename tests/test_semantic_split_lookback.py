#!/usr/bin/env python3
"""
Semantic splits move back to a Whisper sentence end (#32).

The PRIORITY-5 split fires wherever a long chunk reaches ``min_chunk_semantic_break``
and used to insert a period mid-phrase ("…las comunidades alrededor pero al
principio." / "No era fácil…"). When the chunk already holds a sentence end that
Whisper wrote (here one removed before a same-speaker connector), the split now
lands there. With none, it stays where it was, so unpunctuated input stays bounded.

(Until v0.18.0 a similarity model also had to agree; it said yes almost
everywhere and was removed, #34.)
"""

import pytest

from sentence_splitter import SentenceSplitter
from punctuation_restorer import _get_language_config

pytestmark = pytest.mark.core


HEAD = ("hoy tenemos una lista de espera de comunidades que quieren nuestras soluciones "
        "y se ha generado un voz a voz entre las familias de la región y también en "
        "las comunidades alrededor")
TAIL = ("al principio no era fácil generar esa confianza con las personas porque nadie "
        "nos conocía todavía en esa zona del país y teníamos que visitar cada casa "
        "para explicar el proyecto.")


def _split(text, with_speakers):
    words = text.split()
    splitter = SentenceSplitter('es', _get_language_config('es'))
    speakers = [{'start_word': 0, 'end_word': len(words), 'speaker': 'SPEAKER_00'}] if with_speakers else None
    sentences, _ = splitter.split(
        text, whisper_segments=[{'start': 0.0, 'end': 30.0, 'text': text}], speaker_segments=speakers
    )
    return [s.text for s in sentences]


@pytest.mark.parametrize('with_speakers', [True, False], ids=['diarized', 'no-diarization'])
def test_semantic_split_moves_back_to_whisper_sentence_end(with_speakers):
    # With diarization the period before same-speaker "Pero" is removed inline;
    # the moved break must restore it and not be merged back afterwards.
    sentences = _split(f"{HEAD}. Pero {TAIL}", with_speakers)
    assert sentences == [f"{HEAD}.", f"Pero {TAIL}"]


def test_semantic_split_without_whisper_sentence_end_stays_put():
    # Nothing to move back to: the split still fires once the chunk is long
    # enough, which is what bounds sentence length on unpunctuated input.
    sentences = _split(f"{HEAD} pero {TAIL}", with_speakers=True)
    assert len(sentences) == 2
    assert len(sentences[0].split()) >= _get_language_config('es').thresholds['min_chunk_semantic_break']
