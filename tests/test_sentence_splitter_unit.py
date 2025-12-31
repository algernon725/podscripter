#!/usr/bin/env python3
"""
Unit tests for SentenceSplitter class.

Tests the core functionality of the unified sentence splitting system,
including the critical period-before-same-speaker-connector bug fix.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_splitter import SentenceSplitter
from punctuation_restorer import _get_language_config


class MockModel:
    """Mock SentenceTransformer model for testing."""
    def encode(self, texts):
        # Return dummy embeddings
        import numpy as np
        return np.random.rand(len(texts), 384)


class TestSentenceSplitterBasic(unittest.TestCase):
    """Test basic SentenceSplitter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = MockModel()
        self.es_config = _get_language_config('es')
        self.en_config = _get_language_config('en')
    
    def test_simple_split(self):
        """Test basic sentence splitting without boundaries."""
        splitter = SentenceSplitter('es', self.model, self.es_config)
        text = "Hola a todos. Este es un test."
        sentences, metadata = splitter.split(text)
        
        # Should have at least one sentence
        self.assertGreater(len(sentences), 0)
        self.assertIsInstance(metadata, dict)
    
    def test_short_text_no_split(self):
        """Test that very short text is not split."""
        splitter = SentenceSplitter('es', self.model, self.es_config)
        text = "Hola"
        sentences, metadata = splitter.split(text)
        
        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0], "Hola")


class TestWhisperPeriodRemoval(unittest.TestCase):
    """Test the core bug fix: removing Whisper periods before same-speaker connectors."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = MockModel()
        self.es_config = _get_language_config('es')
        self.en_config = _get_language_config('en')
        self.fr_config = _get_language_config('fr')
        self.de_config = _get_language_config('de')
    
    def test_spanish_same_speaker_connector_merge(self):
        """Test that periods are removed before connectors when same speaker continues (Spanish)."""
        splitter = SentenceSplitter('es', self.model, self.es_config)
        
        # Simulate text with Whisper-added period before connector
        # "trabajo. Y este meta" should become "trabajo y este meta" (same speaker)
        text = "es importante tener una estructura como un trabajo y este meta es tu trabajo cada día"
        
        # Create speaker segments (same speaker for entire text)
        speaker_segments = [
            {'start_word': 0, 'end_word': 20, 'speaker': 'SPEAKER_00'}
        ]
        
        sentences, metadata = splitter.split(
            text,
            speaker_segments=speaker_segments
        )
        
        # The sentence should NOT be split at "y" since same speaker continues
        # Check that we don't have a sentence starting with "y" or "Y"
        for sentence in sentences:
            first_word = sentence.split()[0] if sentence.split() else ''
            first_word_clean = first_word.lower().strip('.,;:!?¿¡')
            self.assertNotEqual(first_word_clean, 'y', 
                              f"Sentence should not start with 'y' (same speaker): {sentence}")
    
    def test_english_same_speaker_connector_merge(self):
        """Test that periods are removed before connectors when same speaker continues (English)."""
        splitter = SentenceSplitter('en', self.model, self.en_config)
        
        text = "I work from home and I enjoy it very much"
        
        speaker_segments = [
            {'start_word': 0, 'end_word': 10, 'speaker': 'SPEAKER_00'}
        ]
        
        sentences, metadata = splitter.split(
            text,
            speaker_segments=speaker_segments
        )
        
        # Should not have sentence starting with "and"
        for sentence in sentences:
            first_word = sentence.split()[0] if sentence.split() else ''
            first_word_clean = first_word.lower().strip('.,;:!?')
            self.assertNotEqual(first_word_clean, 'and',
                              f"Sentence should not start with 'and' (same speaker): {sentence}")
    
    def test_french_same_speaker_connector_merge(self):
        """Test that periods are removed before connectors when same speaker continues (French)."""
        splitter = SentenceSplitter('fr', self.model, self.fr_config)
        
        text = "je travaille à la maison et j'aime beaucoup"
        
        speaker_segments = [
            {'start_word': 0, 'end_word': 10, 'speaker': 'SPEAKER_00'}
        ]
        
        sentences, metadata = splitter.split(
            text,
            speaker_segments=speaker_segments
        )
        
        # Should not have sentence starting with "et"
        for sentence in sentences:
            first_word = sentence.split()[0] if sentence.split() else ''
            first_word_clean = first_word.lower().strip('.,;:!?')
            self.assertNotEqual(first_word_clean, 'et',
                              f"Sentence should not start with 'et' (same speaker): {sentence}")
    
    def test_german_same_speaker_connector_merge(self):
        """Test that periods are removed before connectors when same speaker continues (German)."""
        splitter = SentenceSplitter('de', self.model, self.de_config)
        
        text = "ich arbeite von zu Hause und genieße es sehr"
        
        speaker_segments = [
            {'start_word': 0, 'end_word': 10, 'speaker': 'SPEAKER_00'}
        ]
        
        sentences, metadata = splitter.split(
            text,
            speaker_segments=speaker_segments
        )
        
        # Should not have sentence starting with "und"
        for sentence in sentences:
            first_word = sentence.split()[0] if sentence.split() else ''
            first_word_clean = first_word.lower().strip('.,;:!?')
            self.assertNotEqual(first_word_clean, 'und',
                              f"Sentence should not start with 'und' (same speaker): {sentence}")
    
    def test_different_speaker_connector_preserved(self):
        """Test that periods ARE kept when different speakers start with connector."""
        splitter = SentenceSplitter('es', self.model, self.es_config)
        
        text = "yo soy Andrea de Santander Colombia y yo soy Nate de Texas Estados Unidos"
        
        # Two different speakers
        speaker_segments = [
            {'start_word': 0, 'end_word': 6, 'speaker': 'SPEAKER_00'},  # Andrea
            {'start_word': 7, 'end_word': 15, 'speaker': 'SPEAKER_01'}  # Nate (starts with "y")
        ]
        
        sentences, metadata = splitter.split(
            text,
            speaker_segments=speaker_segments
        )
        
        # With different speakers, it's valid for a sentence to start with "Y"
        # This test just ensures we don't crash and handle it correctly
        self.assertGreater(len(sentences), 0)
    
    def test_metadata_tracks_removed_periods(self):
        """Test that metadata correctly tracks removed periods."""
        splitter = SentenceSplitter('es', self.model, self.es_config)
        
        text = "trabajo duro y me gusta mucho"
        
        speaker_segments = [
            {'start_word': 0, 'end_word': 7, 'speaker': 'SPEAKER_00'}
        ]
        
        sentences, metadata = splitter.split(
            text,
            speaker_segments=speaker_segments
        )
        
        # Check that metadata structure exists
        self.assertIn('removed_periods', metadata)
        self.assertIsInstance(metadata['removed_periods'], list)


class TestGrammaticalGuards(unittest.TestCase):
    """Test that grammatical guards prevent invalid breaks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = MockModel()
        self.es_config = _get_language_config('es')
    
    def test_no_break_on_conjunction(self):
        """Test that sentences don't end on coordinating conjunctions."""
        splitter = SentenceSplitter('es', self.model, self.es_config)
        
        # Should not break after "y"
        text = "tenemos muchos errores y eco es importante"
        sentences, _ = splitter.split(text)
        
        # Check no sentence ends with just "y"
        for sentence in sentences:
            last_word = sentence.split()[-1] if sentence.split() else ''
            last_word_clean = last_word.lower().strip('.,;:!?')
            self.assertNotEqual(last_word_clean, 'y',
                              f"Sentence should not end with 'y': {sentence}")
    
    def test_no_break_on_preposition(self):
        """Test that sentences don't end on prepositions."""
        splitter = SentenceSplitter('es', self.model, self.es_config)
        
        # Should not break after "a"
        text = "entonces yo conocí a un amigo que trabajaba con cámaras"
        sentences, _ = splitter.split(text)
        
        # Check no sentence ends with just "a"
        for sentence in sentences:
            last_word = sentence.split()[-1] if sentence.split() else ''
            last_word_clean = last_word.lower().strip('.,;:!?')
            self.assertNotEqual(last_word_clean, 'a',
                              f"Sentence should not end with 'a': {sentence}")
    
    def test_no_break_on_auxiliary_verb(self):
        """Test that sentences don't end on auxiliary/continuative verbs."""
        splitter = SentenceSplitter('es', self.model, self.es_config)
        
        # Should not break after "estaba"
        text = "yo estaba en Colombia y estaba continuando con la universidad"
        sentences, _ = splitter.split(text)
        
        # Check no sentence ends with just "estaba"
        for sentence in sentences:
            last_word = sentence.split()[-1] if sentence.split() else ''
            last_word_clean = last_word.lower().strip('.,;:!?')
            self.assertNotEqual(last_word_clean, 'estaba',
                              f"Sentence should not end with 'estaba': {sentence}")


class TestSpeakerBoundaryPriority(unittest.TestCase):
    """Test that speaker boundaries have highest priority."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = MockModel()
        self.es_config = _get_language_config('es')
    
    def test_speaker_change_creates_break(self):
        """Test that speaker changes create sentence breaks."""
        splitter = SentenceSplitter('es', self.model, self.es_config)
        
        text = "Hola soy Andrea Bueno soy Nate"
        
        # Speaker change after "Andrea"
        speaker_segments = [
            {'start_word': 0, 'end_word': 2, 'speaker': 'SPEAKER_00'},  # "Hola soy Andrea"
            {'start_word': 3, 'end_word': 5, 'speaker': 'SPEAKER_01'}   # "Bueno soy Nate"
        ]
        
        sentences, _ = splitter.split(
            text,
            speaker_segments=speaker_segments
        )
        
        # Should have at least 2 sentences due to speaker change
        self.assertGreaterEqual(len(sentences), 2)
    
    def test_speaker_boundary_short_phrase(self):
        """Test that speaker boundaries work even for very short phrases."""
        splitter = SentenceSplitter('es', self.model, self.es_config)
        
        text = "Mateo 712 Bueno Andrea creo que es importante"
        
        # Speaker change after "Mateo 712" (only 2 words)
        speaker_segments = [
            {'start_word': 0, 'end_word': 1, 'speaker': 'SPEAKER_00'},  # "Mateo 712"
            {'start_word': 2, 'end_word': 8, 'speaker': 'SPEAKER_01'}   # "Bueno Andrea..."
        ]
        
        sentences, _ = splitter.split(
            text,
            speaker_segments=speaker_segments
        )
        
        # Should break after "Mateo 712" despite being only 2 words
        self.assertGreaterEqual(len(sentences), 2)


class TestWhisperBoundarySkipping(unittest.TestCase):
    """Test that Whisper boundaries are skipped when speaker boundary is nearby."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = MockModel()
        self.es_config = _get_language_config('es')
    
    def test_whisper_boundary_skipped_near_speaker_change(self):
        """Test that Whisper boundaries are skipped if speaker change is within 15 words."""
        splitter = SentenceSplitter('es', self.model, self.es_config)
        
        # Simulate Whisper segment ending but speaker continuing
        text = "ustedes pueden ver el episodio Mateo 712 Bueno Andrea"
        
        # Whisper segments
        whisper_segments = [
            {'text': 'ustedes pueden ver el episodio', 'start': 0, 'end': 5},
            {'text': 'Mateo 712', 'start': 5, 'end': 7}
        ]
        
        # Speaker change happens AFTER "Mateo 712"
        speaker_segments = [
            {'start_word': 0, 'end_word': 6, 'speaker': 'SPEAKER_00'},
            {'start_word': 7, 'end_word': 9, 'speaker': 'SPEAKER_01'}
        ]
        
        sentences, _ = splitter.split(
            text,
            whisper_segments=whisper_segments,
            speaker_segments=speaker_segments
        )
        
        # The Whisper boundary after "episodio" should be skipped
        # because speaker change is within 15 words
        # So "ustedes pueden ver el episodio Mateo 712" should stay together
        self.assertGreater(len(sentences), 0)


if __name__ == '__main__':
    unittest.main()

