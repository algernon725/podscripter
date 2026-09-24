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
        {
            'language': 'pt',
            'sentence': "Mas se você quiser ouvir os episódios anteriores, pode ir ao episódio 147,151,156,164,170,177 e 184. O episódio mais recente foi o 184.",
            'should_have': '177 e 184',
            'name': 'Portuguese'
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
        ('pt', 'Os episódios são 1, 2, 3 e 4. Depois continue.', '3 e 4'),
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


def write_one(sentence, language):
    """Run a single sentence through _write_txt() and return the file contents."""
    from podscripter import _write_txt

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_file = f.name
    try:
        _write_txt([sentence], temp_file, language=language)
        with open(temp_file) as fh:
            return fh.read().strip()
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


@pytest.mark.parametrize("language,sentence,expected", [
    # Positive: a coordinating conjunction capitalized mid-sentence is lowered.
    ('en', "I went home And then I slept.", "and then"),
    ('fr', "Je suis rentré Et puis j'ai dormi.", "et puis"),
    ('de', "Ich ging nach Hause Und dann schlief ich.", "und dann"),
    ('es', "Fui a casa Y luego dormí.", "y luego"),
    ('pt', "Fui para casa E depois dormi.", "e depois"),
])
def test_mid_sentence_conjunctions_are_lowercased(language, sentence, expected):
    """Each language lowercases its own mid-sentence coordinating conjunctions."""
    out = write_one(sentence, language)
    assert expected in out, f"{language}: expected {expected!r} in {out!r}"


@pytest.mark.parametrize("language,sentence,must_keep", [
    # Regression (v0.13.0): en/fr/de used to fall back to the SPANISH word list,
    # so Spanish "U" (= "or" before o-) lowercased the English acronym in
    # "the U.S. Capitol" -> "the u.S.", which then defeated the initials guard in
    # _finalize_text_common() and produced "the u. S. Capitol".
    ('en', "He calls the U.S. Capitol a fine building.", "U"),
    # Single letters are never lowercased: these are legitimate capitals.
    ('en', "Take Vitamin A daily.", "Vitamin A"),
    ('en', "Read Section B first.", "Section B"),
    ('en', "We watched Robert De Niro.", "De Niro"),
    ('fr', "Il prend la Vitamine A.", "Vitamine A"),
    # German capitalizes every noun; the list must not touch them.
    ('de', "Das Haus und der Garten sind groß.", "Haus"),
    ('de', "Der Vertrag Und die Bedingungen.", "Vertrag"),
])
def test_legitimate_capitals_survive(language, sentence, must_keep):
    """The mid-sentence pass must not lowercase words that belong capitalized."""
    out = write_one(sentence, language)
    assert must_keep in out, f"{language}: {must_keep!r} was corrupted in {out!r}"


def test_unknown_language_is_a_no_op():
    """A language with no word list leaves capitalization alone (no Spanish fallback)."""
    sentence = "Ciao A tutti E buona giornata."
    assert write_one(sentence, 'it') == sentence
