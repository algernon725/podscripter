#!/usr/bin/env python3
"""
Test speaker-aware infrastructure (v0.6.0).

Tests that speaker information is tracked through the pipeline and that
output formatting remains uniform (single blank line between all paragraphs).
"""

import tempfile
import os
import pytest

from sentence_splitter import Sentence, Utterance
from podscripter import _write_txt

pytestmark = pytest.mark.core


def test_speaker_change_normal_paragraph_break():
    """Verify normal paragraph break between different speakers (no extra break)."""
    sentences = [
        Sentence(
            text="Podcast tú dijiste 15 expresiones de.",
            utterances=[Utterance("Podcast tú dijiste 15 expresiones de.", "SPEAKER_01", 0, 6)],
            speaker="SPEAKER_01"
        ),
        Sentence(
            text="Pablo.",
            utterances=[Utterance("Pablo.", "SPEAKER_02", 6, 7)],
            speaker="SPEAKER_02"
        ),
    ]

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        _write_txt(sentences, f.name, language='es')
        output_path = f.name

    with open(output_path, 'r') as f:
        content = f.read()

    assert '\n\n\n' not in content, f"Unexpected extra paragraph break in: {content}"
    assert '\n\n' in content, f"Expected normal paragraph breaks in: {content}"

    os.unlink(output_path)


def test_same_speaker_no_extra_break():
    """Verify no extra paragraph break when same speaker continues."""
    sentences = [
        Sentence(
            text="En espanolistos.com slash best.",
            utterances=[Utterance("En espanolistos.com slash best.", "SPEAKER_01", 0, 4)],
            speaker="SPEAKER_01"
        ),
        Sentence(
            text="Esto fue todo por el episodio de hoy.",
            utterances=[Utterance("Esto fue todo por el episodio de hoy.", "SPEAKER_01", 4, 11)],
            speaker="SPEAKER_01"
        ),
    ]

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        _write_txt(sentences, f.name, language='es')
        output_path = f.name

    with open(output_path, 'r') as f:
        content = f.read()

    assert '\n\n\n' not in content, f"Unexpected extra paragraph break in: {content}"
    assert '\n\n' in content, f"Expected normal paragraph breaks in: {content}"

    os.unlink(output_path)


def test_backward_compat_with_strings():
    """Verify backward compatibility when sentences are strings instead of Sentence objects."""
    sentences = [
        "Primera oración.",
        "Segunda oración.",
    ]

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        _write_txt(sentences, f.name, language='es')
        output_path = f.name

    with open(output_path, 'r') as f:
        content = f.read()

    assert "Primera oración." in content
    assert "Segunda oración." in content
    assert '\n\n' in content
    assert '\n\n\n' not in content

    os.unlink(output_path)


def test_multiple_speaker_changes_uniform_formatting():
    """Verify multiple speaker changes use uniform paragraph formatting."""
    sentences = [
        Sentence(
            text="Hola, soy Andrea.",
            utterances=[Utterance("Hola, soy Andrea.", "SPEAKER_00", 0, 3)],
            speaker="SPEAKER_00"
        ),
        Sentence(
            text="Y yo soy Nate.",
            utterances=[Utterance("Y yo soy Nate.", "SPEAKER_01", 3, 7)],
            speaker="SPEAKER_01"
        ),
        Sentence(
            text="Bienvenidos al podcast.",
            utterances=[Utterance("Bienvenidos al podcast.", "SPEAKER_00", 7, 10)],
            speaker="SPEAKER_00"
        ),
        Sentence(
            text="Vamos a empezar.",
            utterances=[Utterance("Vamos a empezar.", "SPEAKER_01", 10, 13)],
            speaker="SPEAKER_01"
        ),
    ]

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        _write_txt(sentences, f.name, language='es')
        output_path = f.name

    with open(output_path, 'r') as f:
        content = f.read()

    triple_newline_count = content.count('\n\n\n')
    assert triple_newline_count == 0, f"Expected no extra paragraph breaks, got {triple_newline_count} in: {content}"
    assert '\n\n' in content, f"Expected normal paragraph breaks in: {content}"

    os.unlink(output_path)


def test_no_speaker_info_no_extra_breaks():
    """Verify no extra breaks when Sentence objects have no speaker info."""
    sentences = [
        Sentence(
            text="Primera oración.",
            utterances=[],
            speaker=None
        ),
        Sentence(
            text="Segunda oración.",
            utterances=[],
            speaker=None
        ),
    ]

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        _write_txt(sentences, f.name, language='es')
        output_path = f.name

    with open(output_path, 'r') as f:
        content = f.read()

    assert '\n\n\n' not in content, f"Unexpected extra paragraph break in: {content}"
    assert '\n\n' in content, f"Expected normal paragraph breaks in: {content}"

    os.unlink(output_path)
