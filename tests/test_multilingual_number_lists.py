#!/usr/bin/env python3
"""
Test if the number list splitting bug affects English and French.
"""

import os
import tempfile

import pytest

from punctuation_restorer import assemble_sentences_from_processed
from conftest import restore_punctuation

pytestmark = pytest.mark.multilingual


def _read_txt_output(filepath):
    """Read a TXT file and return list of non-empty paragraphs."""
    with open(filepath, 'r') as f:
        content = f.read()
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    return paragraphs


def test_english_number_list():
    """Test if English number lists are also affected."""
    text = ("But if you want to listen to the previous episodes, "
            "you can go to episode 147, 151, 156, 164, 170, 177 and 184. "
            "The most recent episode in this series was episode 184.")

    restored = restore_punctuation(text, language='en')
    sentences, trailing = assemble_sentences_from_processed(restored, 'en')
    if trailing:
        sentences.append(trailing)

    from podscripter import _write_txt
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_file = f.name

    try:
        _write_txt(sentences, temp_file, language='en')
        paragraphs = _read_txt_output(temp_file)

        standalone_184 = any(p.strip() in ["184.", "184"] for p in paragraphs)
        and_ending = any(p.strip().endswith("and.") for p in paragraphs)
        number_list_intact = any("177 and 184" in p for p in paragraphs)

        assert not standalone_184, "Bug: standalone '184.' paragraph found"
        assert not and_ending, "Bug: paragraph ending with 'and.' found"
        assert number_list_intact, "Number list '177 and 184' not preserved"
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_french_number_list():
    """Test if French number lists are also affected."""
    text = ("Mais si tu veux écouter les épisodes précédents, "
            "tu peux aller à l'épisode 147, 151, 156, 164, 170, 177 et 184. "
            "L'épisode le plus récent de cette série était l'épisode 184.")

    restored = restore_punctuation(text, language='fr')
    sentences, trailing = assemble_sentences_from_processed(restored, 'fr')
    if trailing:
        sentences.append(trailing)

    from podscripter import _write_txt
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_file = f.name

    try:
        _write_txt(sentences, temp_file, language='fr')
        paragraphs = _read_txt_output(temp_file)

        standalone_184 = any(p.strip() in ["184.", "184"] for p in paragraphs)
        et_ending = any(p.strip().endswith("et.") for p in paragraphs)
        number_list_intact = any("177 et 184" in p for p in paragraphs)

        assert not standalone_184, "Bug: standalone '184.' paragraph found"
        assert not et_ending, "Bug: paragraph ending with 'et.' found"
        assert number_list_intact, "Number list '177 et 184' not preserved"
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_german_number_list():
    """Test if German number lists are also affected."""
    text = ("Aber wenn du die vorherigen Episoden hören möchtest, "
            "kannst du zu Episode 147, 151, 156, 164, 170, 177 und 184 gehen. "
            "Die neueste Episode in dieser Serie war Episode 184.")

    restored = restore_punctuation(text, language='de')
    sentences, trailing = assemble_sentences_from_processed(restored, 'de')
    if trailing:
        sentences.append(trailing)

    from podscripter import _write_txt
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_file = f.name

    try:
        _write_txt(sentences, temp_file, language='de')
        paragraphs = _read_txt_output(temp_file)

        standalone_184 = any(p.strip() in ["184.", "184"] for p in paragraphs)
        und_ending = any(p.strip().endswith("und.") for p in paragraphs)
        number_list_intact = any("177 und 184" in p for p in paragraphs)

        assert not standalone_184, "Bug: standalone '184.' paragraph found"
        assert not und_ending, "Bug: paragraph ending with 'und.' found"
        assert number_list_intact, "Number list '177 und 184' not preserved"
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
