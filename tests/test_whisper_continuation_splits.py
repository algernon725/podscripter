#!/usr/bin/env python3
"""
A Whisper segment edge inside a phrase is not a sentence end (#29).

When a segment ends with no punctuation and the next starts lowercase, Whisper cut
one sentence in two. Honoring that edge added a period and a capital mid-phrase
("…mucho en ese." / "Tiempo era…").
"""

import pytest

from conftest import transcript

pytestmark = pytest.mark.core


def _run(whisper, language, tmp_path, speakers=None):
    text = ' '.join(seg['text'] for seg in whisper)
    return transcript(text, language, tmp_path, segments=whisper, speaker_segments=speakers, paragraphs=True)


def _segments(first, second):
    return [
        {'start': 0.0, 'end': 4.0, 'text': first},
        {'start': 4.0, 'end': 9.0, 'text': second},
    ]


def test_continuation_edge_stays_one_sentence(tmp_path):
    whisper = _segments(
        'Entonces lo que hace la gente aún hoy en día y lo que hacían mucho en ese',
        'tiempo era, se iban a caminar con unos bolsos, con algo de ropa y ya.',
    )
    assert _run(whisper, 'es', tmp_path) == [
        'Entonces lo que hace la gente aún hoy en día y lo que hacían mucho en ese '
        'tiempo era, se iban a caminar con unos bolsos, con algo de ropa y ya.'
    ]


@pytest.mark.parametrize('language,first,second', [
    ('en', 'So what people still do today and what they used to do a lot back in the old',
     'days was going for a walk with some bags, with some clothes, and after that they just came back home again.'),
    ('fr', "Alors ce que font les gens encore aujourd'hui et ce qu'ils faisaient beaucoup à la belle",
     'époque était de partir marcher avec des sacs, avec quelques vêtements et ensuite ils rentraient à la maison.'),
    ('de', 'Also was die Leute heute noch machen und was sie damals sehr oft in jener',
     'zeit machten war mit ein paar Taschen und etwas Kleidung spazieren zu gehen und dann wieder nach Hause zu kommen.'),
    ('pt', 'Então o que as pessoas fazem ainda hoje e o que faziam muito naquele',
     'tempo era sair para caminhar com umas bolsas, com alguma roupa e depois voltar para casa com toda a família.'),
    ('it', 'Allora quello che fa la gente ancora oggi e quello che facevano molto in quel',
     'tempo era andare a camminare con delle borse, con qualche vestito e poi tornare a casa con la famiglia.'),
])
def test_continuation_edge_not_split_in_other_languages(language, first, second, tmp_path):
    edge = f'{first.split()[-1]} {second.split()[0]}'
    output = ' '.join(_run(_segments(first, second), language, tmp_path))
    assert edge in output, f'Split inside "{edge}": {output}'


def test_punctuated_edge_still_splits(tmp_path):
    whisper = _segments(
        'Entonces lo que hace la gente aún hoy en día es caminar mucho por la ciudad.',
        'tiempo era lo que más les faltaba en aquellos años tan difíciles de la historia de su país.',
    )
    paragraphs = _run(whisper, 'es', tmp_path)
    assert len(paragraphs) == 2 and paragraphs[0].endswith('por la ciudad.')


def test_speaker_change_at_continuation_edge_still_splits(tmp_path):
    whisper = _segments(
        'Entonces lo que hace la gente aún hoy en día y lo que hacían mucho en ese',
        'tiempo era, se iban a caminar con unos bolsos, con algo de ropa y ya.',
    )
    speakers = [
        {'start': 0.0, 'end': 4.0, 'speaker': 'SPEAKER_00'},
        {'start': 4.0, 'end': 9.0, 'speaker': 'SPEAKER_01'},
    ]
    assert len(_run(whisper, 'es', tmp_path, speakers)) == 2
