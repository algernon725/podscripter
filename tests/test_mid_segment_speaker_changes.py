#!/usr/bin/env python3
"""
A speaker change never splits a Whisper segment (#6).

Each Whisper segment goes wholly to the speaker with the most overlap. Splitting a
segment at a time-proportional point cut one person's phrase or word into
fragments ("las tareas de la. / Escuela rápido antes de."); Whisper starts a new
segment at a real turn change, so the turn belongs at a segment edge.
"""

import pytest

from conftest import transcript

pytestmark = pytest.mark.core


def _run(whisper, speakers, tmp_path):
    text = ' '.join(seg['text'] for seg in whisper)
    return transcript(text, 'es', tmp_path, segments=whisper, speaker_segments=speakers, paragraphs=True)


def test_misattributed_segment_inside_a_phrase_changes_nothing(tmp_path):
    whisper = [
        {'start': 0.0, 'end': 4.0, 'text': 'Vi un video donde había una mujer que hablaba de que'},
        {'start': 4.0, 'end': 8.2, 'text': 'su hijo tenía que hacer las tareas de la escuela rápido'},
        {'start': 8.2, 'end': 10.2, 'text': 'antes de las seis de la tarde'},
        {'start': 10.2, 'end': 15.0, 'text': 'porque cuando se pone el sol ya es una oscuridad completa.'},
    ]
    # 1.6 s of SPEAKER_01 across "escuela rápido / antes de", inside SPEAKER_00's sentence
    with_stray = _run(whisper, [
        {'start': 0.0, 'end': 7.2, 'speaker': 'SPEAKER_00'},
        {'start': 7.1, 'end': 8.7, 'speaker': 'SPEAKER_01'},
        {'start': 8.7, 'end': 15.0, 'speaker': 'SPEAKER_00'},
    ], tmp_path)
    without_stray = _run(whisper, [{'start': 0.0, 'end': 15.0, 'speaker': 'SPEAKER_00'}], tmp_path)
    assert with_stray == without_stray, (
        f"A speaker segment inside a phrase must not fragment it.\n"
        f"with: {with_stray}\nwithout: {without_stray}"
    )


def test_turn_placed_late_moves_to_the_segment_start(tmp_path):
    whisper = [
        {'start': 0.0, 'end': 5.0, 'text': 'Es quien determina todo lo concerniente al idioma español.'},
        {'start': 5.0, 'end': 8.0, 'text': 'Listo, número 13.'},
        {'start': 8.0, 'end': 13.0, 'text': 'México es el país hispanohablante más grande del planeta.'},
    ]
    # The diarizer ends SPEAKER_00's turn 0.9 s into "Listo, número 13.", which SPEAKER_01 says
    speakers = [
        {'start': 0.0, 'end': 5.9, 'speaker': 'SPEAKER_00'},
        {'start': 6.7, 'end': 13.0, 'speaker': 'SPEAKER_01'},
    ]
    paragraphs = _run(whisper, speakers, tmp_path)
    assert paragraphs == [
        'Es quien determina todo lo concerniente al idioma español.',
        'Listo, número 13. México es el país hispanohablante más grande del planeta.',
    ], f"The turn should fall at the Whisper segment edge: {paragraphs}"
