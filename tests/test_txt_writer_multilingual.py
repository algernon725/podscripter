"""
Test that the TXT writer fix works for all languages
when the sentences are already correctly punctuated.
"""

import tempfile
import os

import pytest

pytestmark = pytest.mark.core


def read_txt_output(filepath):
    """Read a TXT file and return list of non-empty paragraphs."""
    with open(filepath, 'r') as f:
        content = f.read()
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    return paragraphs


def test_txt_writer_with_correct_punctuation():
    """Test TXT writer doesn't split correctly-punctuated number lists."""
    from podscripter import _write_txt

    test_cases = [
        {
            'language': 'es',
            'sentence': "Pero si tú quieres escuchar los episodios anteriores, puedes ir al episodio 147,151,156,164,170,177 y 184. El episodio más reciente fue el 184.",
            'should_have': '177 y 184',
            'name': 'Spanish'
        },
        {
            'language': 'en',
            'sentence': "But if you want to listen to the previous episodes, you can go to episode 147,151,156,164,170,177 and 184. The most recent episode was episode 184.",
            'should_have': '177 and 184',
            'name': 'English'
        },
        {
            'language': 'fr',
            'sentence': "Mais si tu veux écouter les épisodes précédents, tu peux aller à l'épisode 147,151,156,164,170,177 et 184. L'épisode le plus récent était l'épisode 184.",
            'should_have': '177 et 184',
            'name': 'French'
        },
        {
            'language': 'de',
            'sentence': "Aber wenn du die vorherigen Episoden hören möchtest, kannst du zu Episode 147,151,156,164,170,177 und 184 gehen. Die neueste Episode war Episode 184.",
            'should_have': '177 und 184',
            'name': 'German'
        },
    ]

    for case in test_cases:
        sentences = [case['sentence']]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_file = f.name

        try:
            _write_txt(sentences, temp_file, language=case['language'])
            paragraphs = read_txt_output(temp_file)

            full_text = ' '.join(paragraphs)
            has_list = case['should_have'] in full_text
            standalone_184 = any(p.strip() in ["184.", "184"] for p in paragraphs)

            assert has_list, (
                f"{case['name']}: number list '{case['should_have']}' not preserved in output: {paragraphs}"
            )
            assert not standalone_184, (
                f"{case['name']}: '184.' was split into standalone paragraph: {paragraphs}"
            )
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


def test_simple_number_lists_all_languages():
    """Test simple number lists with 'and/y/et/und'."""
    from podscripter import _write_txt

    test_cases = [
        ('es', 'Los episodios son 1, 2, 3 y 4. Luego continúa.', '3 y 4'),
        ('en', 'The episodes are 1, 2, 3 and 4. Then continue.', '3 and 4'),
        ('fr', 'Les épisodes sont 1, 2, 3 et 4. Ensuite continue.', '3 et 4'),
        ('de', 'Die Episoden sind 1, 2, 3 und 4. Dann fortfahren.', '3 und 4'),
    ]

    for lang, sentence, pattern in test_cases:
        sentences = [sentence]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_file = f.name

        try:
            _write_txt(sentences, temp_file, language=lang)
            paragraphs = read_txt_output(temp_file)

            full_text = ' '.join(paragraphs)
            has_pattern = pattern in full_text
            not_split = len(paragraphs) == 1 or (len(paragraphs) == 2 and '4.' not in paragraphs[1])

            assert has_pattern, (
                f"{lang.upper()}: pattern '{pattern}' not found in output: {paragraphs}"
            )
            assert not_split, f"{lang.upper()}: number list was split: {paragraphs}"
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
