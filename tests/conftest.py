import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from sentence_splitter import SentenceSplitter
from punctuation_restorer import restore_punctuation as _restore_punctuation


def restore_punctuation(text, language='en', **kwargs):
    """Wrapper around restore_punctuation that returns only the text string.

    The real restore_punctuation returns (text, sentences). Tests almost always
    only need the text, so this helper unpacks it.
    """
    result = _restore_punctuation(text, language, **kwargs)
    if isinstance(result, tuple):
        return result[0]
    return result


def transcript(text, language, tmp_path, *, segments=None, speaker_segments=None, paragraphs=False):
    """Run text through the real TXT path and return the transcript, whitespace-collapsed.

    Unlike the restore_punctuation wrapper above, this is what users get:
    _assemble_sentences() -> _write_txt(). Prefer it for output assertions (#22).

    segments: Whisper segments ('start', 'end', 'text'); default one 0-5 s segment
    holding all of text. speaker_segments: diarization segments ('start', 'end',
    'speaker'). paragraphs=True returns the list of paragraphs instead.
    """
    from podscripter import _assemble_sentences, _write_txt

    if segments is None:
        segments = [{'start': 0.0, 'end': 5.0, 'text': text}]
    sentences, _ = _assemble_sentences(
        text, segments, language, True, speaker_segments=speaker_segments
    )
    out = tmp_path / 'transcript.txt'
    _write_txt(sentences, str(out), language=language)
    content = out.read_text(encoding='utf-8')
    if paragraphs:
        return [' '.join(p.split()) for p in content.split('\n\n') if p.strip()]
    return ' '.join(content.split())


class MockConfig:
    """Shared mock config for tests that need SentenceSplitter without full language config."""
    def __init__(self, **overrides):
        self.thresholds = {
            'min_total_words_no_split': 25,
            'min_chunk_before_split': 20,
            'min_chunk_semantic_break': 42,
            'min_words_whisper_break': 10,
            'semantic_whisper_lookahead': 8,
        }
        self.thresholds.update(overrides)


@pytest.fixture
def mock_config():
    """Fixture providing a MockConfig instance. Pass overrides via request.param."""
    return MockConfig()


@pytest.fixture
def es_splitter(mock_config):
    """Spanish SentenceSplitter with mock config."""
    return SentenceSplitter('es', None, mock_config)


@pytest.fixture
def en_splitter(mock_config):
    """English SentenceSplitter with mock config."""
    return SentenceSplitter('en', None, mock_config)


@pytest.fixture
def de_splitter(mock_config):
    """German SentenceSplitter with mock config."""
    return SentenceSplitter('de', None, mock_config)


@pytest.fixture
def fr_splitter(mock_config):
    """French SentenceSplitter with mock config."""
    return SentenceSplitter('fr', None, mock_config)


@pytest.fixture
def pt_splitter(mock_config):
    """Portuguese SentenceSplitter with mock config."""
    return SentenceSplitter('pt', None, mock_config)
