#!/usr/bin/env python3
"""
Short speaker segments are kept only when Whisper corroborates them (#7).

Speaker segments under 1.3 s used to be dropped outright, so a short genuine
reply ("Guinea.") merged into the other speaker's paragraph. Now one is kept when
it covers at least SHORT_SEGMENT_MIN_WHISPER_COVERAGE of a Whisper segment that is
not a continuation of the previous one. Misattributed slivers inside one person's
speech must still change nothing (the v0.6.0.1 "Errores en español" class).
"""

import pytest

from conftest import transcript

pytestmark = pytest.mark.core


def _run(whisper, speakers, tmp_path):
    text = ' '.join(seg['text'] for seg in whisper)
    return transcript(text, 'es', tmp_path, segments=whisper, speaker_segments=speakers, paragraphs=True)


def test_corroborated_short_reply_gets_its_own_paragraph(tmp_path):
    whisper = [
        {'start': 0.0, 'end': 4.0, 'text': 'Hoy vamos a hablar de un país de África, ¿sabes cuál es?'},
        {'start': 4.0, 'end': 5.0, 'text': 'Guinea.'},
        {'start': 5.0, 'end': 9.0, 'text': 'Exacto, es un país muy pequeño y tiene muchas islas.'},
    ]
    speakers = [
        {'start': 0.0, 'end': 4.1, 'speaker': 'SPEAKER_00'},
        {'start': 3.9, 'end': 5.1, 'speaker': 'SPEAKER_01'},  # 1.2 s, covers all of "Guinea."
        {'start': 5.2, 'end': 9.0, 'speaker': 'SPEAKER_00'},
    ]
    paragraphs = _run(whisper, speakers, tmp_path)
    assert 'Guinea.' in paragraphs, f"Short reply should be its own paragraph: {paragraphs}"


_LONG_UTTERANCE = [
    {'start': 0.0, 'end': 4.0, 'text': 'Hoy quiero hablar de los errores más comunes.'},
    {'start': 4.0, 'end': 12.0,
     'text': 'Muchas personas cometen errores en español. Entonces, yo siempre les digo que practiquen.'},
    {'start': 12.0, 'end': 16.0, 'text': 'Practicar todos los días es la clave.'},
]


@pytest.mark.parametrize('whisper,stray', [
    pytest.param(
        _LONG_UTTERANCE,
        {'start': 7.5, 'end': 8.78, 'speaker': 'SPEAKER_01'},  # 1.28 s, 16% of the 8 s Whisper segment
        id='mid-utterance-1.28s',
    ),
    pytest.param(
        _LONG_UTTERANCE,
        {'start': 7.5, 'end': 7.58, 'speaker': 'SPEAKER_01'},  # 0.08 s sliver
        id='sliver-0.08s',
    ),
    pytest.param(
        [{'start': 0.0, 'end': 3.0, 'text': 'Para comprar la canasta familiar, una familia necesita 365'},
         {'start': 3.0, 'end': 4.0, 'text': 'dólares al mes.'},
         {'start': 4.0, 'end': 8.0, 'text': 'Y eso es mucho más que el sueldo mínimo.'}],
        {'start': 3.0, 'end': 3.9, 'speaker': 'SPEAKER_01'},  # covers 90% of a continuation
        id='whisper-continuation',
    ),
])
def test_uncorroborated_short_segment_changes_nothing(whisper, stray, tmp_path):
    end = whisper[-1]['end']
    speaker_00 = [{'start': 0.0, 'end': stray['start'], 'speaker': 'SPEAKER_00'},
                  {'start': stray['end'], 'end': end, 'speaker': 'SPEAKER_00'}]
    with_stray = _run(whisper, [speaker_00[0], stray, speaker_00[1]], tmp_path)
    without_stray = _run(whisper, speaker_00, tmp_path)
    assert with_stray == without_stray, (
        f"A short segment Whisper does not corroborate must not split the text.\n"
        f"with: {with_stray}\nwithout: {without_stray}"
    )
