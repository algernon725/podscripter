#!/usr/bin/env python3
"""
Generic (non-tailored) languages must degrade to Whisper's own output.

Only the languages in `language_support.TAILORED_LANGUAGES` get language-specific
processing. Until v0.14.0 every other language silently received the *English*
rules, which corrupted text Whisper had already produced correctly:

  * English spaCy capitalized every out-of-vocabulary foreign token
    ("Ciao a Tutti, Benvenuti Al Podcast").
  * The domain rejoin turned Italian "Non lo so. Io non ci credo." into
    "Non lo so.io non ci credo." (`io` is a TLD and the word "I").
  * The ".!?"-only terminal check appended "." after Japanese "？" and the English
    question seeds rewrote Greek ";" and doubled Arabic "؟".

The contract pinned here is round-trip preservation: Whisper-quality text in,
identical text out.
"""

import os
import subprocess
import sys
import tempfile
import time
import types

import pytest

import podscripter
from language_support import (
    is_tailored,
    tailored_languages,
    whisper_languages,
)
from podscripter import _assemble_sentences, _write_txt, validate_language_code, InvalidInputError
from punctuation_restorer import (
    _get_question_patterns,
    _get_exclamation_patterns,
    has_question_indicators,
    restore_punctuation,
)
from domain_utils import fix_spaced_domains

pytestmark = pytest.mark.core

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Whisper-quality transcripts: already punctuated and capitalized correctly.
PRESERVED = [
    ('it', "Ciao a tutti, benvenuti al podcast. Oggi parliamo di Roma e della sua storia antica."),
    ('it', "Mi chiamo Marco e vivo a Milano. Che cosa fai stasera?"),
    ('it', "Non lo so. Io non ci credo."),
    ('it', "È tardi. Io vado a casa."),
    ('ru', "Привет всем, добро пожаловать в подкаст. Сегодня мы говорим о Москве и её истории."),
    ('ja', "こんにちは、皆さん。 今日はいい天気ですね。 東京に行きましょうか？"),
    ('zh', "大家好。 今天我们讨论历史。 你觉得怎么样？"),
    ('el', "Γεια σας. Τι κάνετε;"),
    ('ar', "مرحبا بكم. كيف حالك؟"),
]


def _segments(text):
    """One Whisper segment per sentence, as Whisper would emit them."""
    import re
    parts = [p for p in re.split(r'(?<=[.!?。？;؟])\s+', text) if p]
    return [{'start': 2.5 * i, 'end': 2.5 * i + 2, 'text': ' ' + p} for i, p in enumerate(parts)]


def _write(sentences, language):
    fd, path = tempfile.mkstemp(suffix='.txt')
    os.close(fd)
    try:
        _write_txt(sentences, path, language=language)
        with open(path, encoding='utf-8') as f:
            return f.read()
    finally:
        os.unlink(path)


def _paragraphs(txt):
    return ' '.join(p.strip() for p in txt.split('\n\n') if p.strip())


# ---------------------------------------------------------------- the predicate

def test_tailored_set_is_pinned():
    """Adding or dropping a tailored language must be a deliberate, reviewed change."""
    assert tailored_languages() == {'en', 'es', 'fr', 'de', 'pt'}


@pytest.mark.parametrize("language", ['en', 'es', 'fr', 'de', 'pt'])
def test_image_languages_are_tailored(language):
    assert is_tailored(language), f"{language!r} lost its language-specific processing"
    assert is_tailored(language.upper())


@pytest.mark.parametrize("language", ['it', 'ru', 'ja', 'zh', 'el', 'ar', None, ''])
def test_other_languages_are_generic(language):
    assert not is_tailored(language)


def test_spacy_is_not_imported():
    """spaCy was removed in v0.15.0 (its output never reached the transcript).

    The Docker image no longer installs it, so an import that slips back in would
    only fail at `docker build`. Catch it here, in a fresh interpreter.
    """
    code = ("import sys, podscripter, punctuation_restorer, sentence_splitter, "
            "sentence_formatter, domain_utils, language_support; "
            "sys.exit('spacy' in sys.modules)")
    proc = subprocess.run([sys.executable, '-c', code], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"spaCy was imported\n{proc.stderr[-2000:]}"


def test_whisper_language_codes_still_resolve():
    """whisper_languages() reads a private faster-whisper symbol; pin that it exists."""
    codes = whisper_languages()
    assert codes is not None, "faster_whisper.tokenizer._LANGUAGE_CODES disappeared"
    assert {'en', 'es', 'it', 'ru', 'ja'} <= codes


# --------------------------------------------------------- no English stand-ins

@pytest.mark.parametrize("language", ['it', 'ru'])
def test_generic_language_gets_no_english_question_rules(language):
    assert _get_question_patterns(language) == []
    assert _get_exclamation_patterns(language) == []
    assert has_question_indicators("what is this", language) is False


# ------------------------------------------------------ round-trip preservation

@pytest.mark.parametrize("language,text", PRESERVED)
def test_restore_punctuation_preserves_whisper_text(language, text):
    out, sentences = restore_punctuation(text, language)
    assert out == text, f"{language}: text changed:\n  in : {text!r}\n  out: {out!r}"
    assert ' '.join(s.text for s in sentences) == text


@pytest.mark.parametrize("language,text", PRESERVED)
def test_txt_output_preserves_whisper_text(language, text):
    """Through the real assembly + writer path, with and without Whisper segments."""
    for segments in ([], _segments(text)):
        sentences, _ = _assemble_sentences(text, segments, language, True)
        out = _paragraphs(_write(sentences, language))
        assert out == text, f"{language}: TXT changed:\n  in : {text!r}\n  out: {out!r}"


def test_io_tld_does_not_rejoin_italian_sentences():
    assert fix_spaced_domains("Non lo so. Io non ci credo.", language='it') == \
        "Non lo so. Io non ci credo."


@pytest.mark.parametrize("text", [
    "Visitate il sito google.com per sapere di più.",
    "Scrivete a podcast.io oggi.",
])
def test_generic_language_keeps_contiguous_domains(text):
    """No rejoin for generic languages, so the TXT writer must never split domains."""
    assert _paragraphs(_write([text], 'it')) == text


@pytest.mark.parametrize("language,text,expected", [
    ('it', "Mi chiamo Marco, e vivo a Milano", "Mi chiamo Marco, e vivo a Milano."),
    ('ru', "как дела", "как дела."),
])
def test_missing_terminal_is_added_without_inventing_anything_else(language, text, expected):
    out, _ = restore_punctuation(text, language)
    assert out == expected


def test_cyrillic_is_neither_capitalized_nor_dropped():
    text = "как дела? хорошо. спасибо."
    out, _ = restore_punctuation(text, 'ru')
    assert out == text


# ------------------------------------------------------------------ validation

@pytest.mark.parametrize("code", ['klingon', 'xx', 'english'])
def test_invalid_language_code_is_rejected(code):
    with pytest.raises(InvalidInputError):
        validate_language_code(code)


@pytest.mark.parametrize("code", ['it', 'ru', 'ja', 'es', None])
def test_valid_language_code_is_accepted(code):
    assert validate_language_code(code) == code


def test_cli_rejects_invalid_language_with_exit_2_before_loading_a_model(tmp_path):
    media = tmp_path / "clip.mp3"
    media.write_bytes(b"")
    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "podscripter.py"), str(media),
         "--output_dir", str(tmp_path), "--language", "klingon"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 2, proc.stderr
    assert "Unknown language code 'klingon'" in proc.stderr
    assert "Loading transcription model" not in proc.stderr
    assert time.monotonic() - start < 60


# ----------------------------------------------------------------- diarization

def test_diarization_runs_for_a_generic_language(tmp_path, monkeypatch):
    """Diarization is acoustic-only; nothing may gate it on language."""
    calls = []

    def fake_diarize(media_file, **kwargs):
        calls.append(media_file)
        return {'speaker_boundaries': [], 'segments': []}

    class _Stop(Exception):
        pass

    def stop_model_load(*args, **kwargs):
        raise _Stop()

    fake_module = types.ModuleType("speaker_diarization")
    fake_module.diarize_audio = fake_diarize
    monkeypatch.setitem(sys.modules, "speaker_diarization", fake_module)
    monkeypatch.setattr(podscripter, "_load_model", stop_model_load)

    media = tmp_path / "clip.mp3"
    media.write_bytes(b"")
    with pytest.raises(podscripter.ModelLoadError):
        podscripter.transcribe(str(media), language='it', write_output=False,
                               enable_diarization=True, quiet=True)
    assert calls == [str(media)], "diarization was not attempted for a generic language"
