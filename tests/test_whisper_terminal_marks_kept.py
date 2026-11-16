#!/usr/bin/env python3
"""
A sentence keeps the terminal mark Whisper gave it (#34).

Until v0.18.0 every sentence went through a sentence-similarity model that
could replace Whisper's final mark: a sentence that opened with a short
question ("¿Listo? Andrea tiene algo que quiere decir.") got '?' at the end
because it started with '¿', and statements that resembled the model's seed
sentences got '?' or '!' ("Listo." -> "¿Listo?", "Bueno." -> "Bueno!").
Across six Spanish episodes this changed 43 paragraph endings, all of them
on statements.

Synthetic text modeled on those cases; it goes through _assemble_sentences(),
the path the TXT file is written from.
"""

import pytest

from podscripter import _assemble_sentences

pytestmark = pytest.mark.core


@pytest.mark.parametrize("text", [
    "¿Listo? Andrea tiene algo que quiere decir.",
    "¿Estás bien? Ya vas a mejorar tu español.",
    "Listo.",
    "Bueno.",
    "Que estaba sorprendida, algo así.",
])
def test_spanish_whisper_period_is_kept(text):
    segments = [{'start': 0.0, 'end': 5.0, 'text': text}]
    sentences, _ = _assemble_sentences(text, segments, 'es', True)
    assert [s.text for s in sentences] == [text]
