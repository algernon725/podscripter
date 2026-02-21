"""
Comprehensive test for the Episodio 190 number list splitting bug.
Tests the complete pipeline from punctuation restoration to TXT writing.
"""

import os
import tempfile

import pytest

from conftest import restore_punctuation
from punctuation_restorer import assemble_sentences_from_processed

pytestmark = pytest.mark.core


def read_txt_output(filepath):
    """Read a TXT file and return list of non-empty paragraphs."""
    with open(filepath, 'r') as f:
        content = f.read()
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    return paragraphs


def test_full_pipeline_episodio190():
    """Test the complete pipeline with the exact text from Episodio 190."""
    from podscripter import _write_txt

    text = ("Pero si tú quieres escuchar los episodios anteriores, "
            "puedes ir al episodio 147, 151, 156, 164, 170, 177 y 184. "
            "El episodio más reciente sobre esta serie fue el 184 "
            "donde hicimos el capítulo 7 y hoy vamos a hacer el capítulo 8.")

    restored = restore_punctuation(text, language='es')
    sentences, trailing = assemble_sentences_from_processed(restored, 'es')
    if trailing:
        sentences.append(trailing)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_file = f.name

    try:
        _write_txt(sentences, temp_file, language='es')
        paragraphs = read_txt_output(temp_file)

        standalone_184 = any(p.strip() in ["184.", "184"] for p in paragraphs)
        y_ending = any(p.strip().endswith("y.") for p in paragraphs)
        number_list_intact = any("177 y 184" in p for p in paragraphs)

        assert not standalone_184, f"'184' is a standalone paragraph: {paragraphs}"
        assert not y_ending, f"Found paragraph ending with 'y.': {paragraphs}"
        assert number_list_intact, f"Number list '177 y 184' was broken apart: {paragraphs}"
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


@pytest.mark.parametrize("name,text,should_have,should_not_have_separate", [
    (
        'Number list with spaces in numbers',
        'Ve a los episodios 1, 2, 3 y 4. Luego continúa.',
        '3 y 4',
        '4.',
    ),
    (
        'Number list with "o" instead of "y"',
        'Elige opción 1, 2 o 3. Después decide.',
        '2 o 3',
        '3.',
    ),
    (
        'Year list',
        'Los años 2015, 2016, 2017 y 2018 fueron importantes. Ahora estamos en 2025.',
        '2017 y 2018',
        '2018.',
    ),
])
def test_edge_cases(name, text, should_have, should_not_have_separate):
    """Test edge cases for number list handling."""
    from podscripter import _write_txt

    restored = restore_punctuation(text, language='es')
    sentences, trailing = assemble_sentences_from_processed(restored, 'es')
    if trailing:
        sentences.append(trailing)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_file = f.name

    try:
        _write_txt(sentences, temp_file, language='es')
        paragraphs = read_txt_output(temp_file)

        full_text = ' '.join(paragraphs)
        assert should_have in full_text, \
            f"[{name}] Expected '{should_have}' in output: {paragraphs}"
        assert not any(p.strip() == should_not_have_separate for p in paragraphs), \
            f"[{name}] '{should_not_have_separate}' is a standalone paragraph: {paragraphs}"
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
