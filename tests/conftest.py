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
