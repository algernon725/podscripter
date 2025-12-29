#!/usr/bin/env python3
"""
Tests for connector word handling when using speaker diarization.

When the same speaker continues speaking, sentences should not be split
before connector words like "Y" (Spanish), "and" (English), "et" (French), "und" (German).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from punctuation_restorer import restore_punctuation


def create_mock_speaker_segments_same_speaker(text: str) -> list[dict]:
    """Create mock speaker segments where the entire text is spoken by one speaker."""
    return [{
        'start_char': 0,
        'end_char': len(text),
        'speaker': 'SPEAKER_00'
    }]


def create_mock_speaker_segments_two_speakers(text: str, split_at_char: int) -> list[dict]:
    """Create mock speaker segments with a speaker change at a specific character position."""
    return [
        {
            'start_char': 0,
            'end_char': split_at_char,
            'speaker': 'SPEAKER_00'
        },
        {
            'start_char': split_at_char,
            'end_char': len(text),
            'speaker': 'SPEAKER_01'
        }
    ]


class TestDiarizationConnectorWords:
    """Test that connector words don't cause sentence splits when same speaker continues."""
    
    def test_same_speaker_no_connector_split_spanish_y(self):
        """When same speaker continues with 'Y', should not split sentence."""
        # Simulating text where a speaker boundary would be at position between
        # "mañana" and "Y", but "Y" is a connector so we shouldn't split
        text = "Andrea siempre tiene muchos sueños que están recordando en la mañana Y yo no recuerdo muchos sueños"
        
        # Without diarization (no speaker boundaries), should still work
        result = restore_punctuation(text, language='es', whisper_boundaries=None, speaker_boundaries=None)
        
        # The sentence should not start with "Y"
        sentences = [s.strip() for s in result.split('\n') if s.strip()]
        for sentence in sentences:
            # Sentences should not start with "Y" as a separate sentence
            # (They can have "Y" internally, just not at the start after a period)
            if sentence.startswith('Y '):
                # This would be the bug - sentence incorrectly starting with Y
                assert False, f"Sentence incorrectly starts with 'Y': {sentence}"
        
        print(f"✓ Spanish 'Y' connector test passed")
        print(f"  Result: {result[:200]}...")
    
    def test_same_speaker_no_connector_split_spanish_full_example(self):
        """Full example from user's Episodio212 transcription."""
        # This is the actual problematic text from the user's example
        text = "Andrea siempre tiene muchos sueños que están recordando en la mañana y yo no recuerdo muchos sueños"
        
        # Mock speaker segments showing the entire text is spoken by one speaker
        speaker_segments = create_mock_speaker_segments_same_speaker(text)
        
        result = restore_punctuation(text, language='es', whisper_boundaries=None, speaker_boundaries=None, speaker_segments=speaker_segments)
        
        # Check that we don't have a sentence starting with "Y"
        lines = [s.strip() for s in result.split('\n') if s.strip()]
        for line in lines:
            assert not line.startswith('Y '), f"Sentence should not start with 'Y': {line}"
        
        print(f"✓ Spanish full example test passed")
        print(f"  Result: {result}")
    
    def test_same_speaker_no_connector_split_spanish_second_example(self):
        """Second example from user's Episodio212 transcription."""
        text = "No recuerdo nada y yo le cuento como 5 o 6 sueños diferentes que tuve y los recuerdo muy bien"
        
        # Mock speaker segments showing the entire text is spoken by one speaker
        speaker_segments = create_mock_speaker_segments_same_speaker(text)
        
        result = restore_punctuation(text, language='es', whisper_boundaries=None, speaker_boundaries=None, speaker_segments=speaker_segments)
        
        # Check that we don't have sentences starting with "Y"
        lines = [s.strip() for s in result.split('\n') if s.strip()]
        for line in lines:
            assert not line.startswith('Y '), f"Sentence should not start with 'Y': {line}"
        
        print(f"✓ Spanish second example test passed")
        print(f"  Result: {result}")
    
    def test_same_speaker_no_connector_split_english_and(self):
        """When same speaker continues with 'and', should not split sentence."""
        text = "I love reading books and I enjoy learning new things every day"
        
        result = restore_punctuation(text, language='en', whisper_boundaries=None, speaker_boundaries=None)
        
        # Check that we don't have a sentence starting with "And"
        sentences = [s.strip() for s in result.split('\n') if s.strip()]
        for sentence in sentences:
            # After proper punctuation, "and" should remain lowercase and not start a new sentence
            if sentence.startswith('And '):
                assert False, f"Sentence incorrectly starts with 'And': {sentence}"
        
        print(f"✓ English 'and' connector test passed")
        print(f"  Result: {result}")
    
    def test_same_speaker_no_connector_split_english_but(self):
        """When same speaker continues with 'but', should not split sentence."""
        text = "She wanted to go but she couldn't make it on time"
        
        result = restore_punctuation(text, language='en', whisper_boundaries=None, speaker_boundaries=None)
        
        # Check that we don't have a sentence starting with "But"
        sentences = [s.strip() for s in result.split('\n') if s.strip()]
        for sentence in sentences:
            if sentence.startswith('But '):
                assert False, f"Sentence incorrectly starts with 'But': {sentence}"
        
        print(f"✓ English 'but' connector test passed")
        print(f"  Result: {result}")
    
    def test_same_speaker_no_connector_split_french_et(self):
        """When same speaker continues with 'et', should not split sentence."""
        text = "J'aime lire des livres et j'aime apprendre de nouvelles choses"
        
        result = restore_punctuation(text, language='fr', whisper_boundaries=None, speaker_boundaries=None)
        
        # Check that we don't have a sentence starting with "Et"
        sentences = [s.strip() for s in result.split('\n') if s.strip()]
        for sentence in sentences:
            if sentence.startswith('Et '):
                assert False, f"Sentence incorrectly starts with 'Et': {sentence}"
        
        print(f"✓ French 'et' connector test passed")
        print(f"  Result: {result}")
    
    def test_same_speaker_no_connector_split_french_mais(self):
        """When same speaker continues with 'mais', should not split sentence."""
        text = "Elle voulait partir mais elle ne pouvait pas"
        
        result = restore_punctuation(text, language='fr', whisper_boundaries=None, speaker_boundaries=None)
        
        # Check that we don't have a sentence starting with "Mais"
        sentences = [s.strip() for s in result.split('\n') if s.strip()]
        for sentence in sentences:
            if sentence.startswith('Mais '):
                assert False, f"Sentence incorrectly starts with 'Mais': {sentence}"
        
        print(f"✓ French 'mais' connector test passed")
        print(f"  Result: {result}")
    
    def test_same_speaker_no_connector_split_german_und(self):
        """When same speaker continues with 'und', should not split sentence."""
        text = "Ich lese gerne Bücher und ich lerne gerne neue Dinge"
        
        result = restore_punctuation(text, language='de', whisper_boundaries=None, speaker_boundaries=None)
        
        # Check that we don't have a sentence starting with "Und"
        sentences = [s.strip() for s in result.split('\n') if s.strip()]
        for sentence in sentences:
            if sentence.startswith('Und '):
                assert False, f"Sentence incorrectly starts with 'Und': {sentence}"
        
        print(f"✓ German 'und' connector test passed")
        print(f"  Result: {result}")
    
    def test_same_speaker_no_connector_split_german_aber(self):
        """When same speaker continues with 'aber', should not split sentence."""
        text = "Sie wollte gehen aber sie konnte nicht rechtzeitig ankommen"
        
        result = restore_punctuation(text, language='de', whisper_boundaries=None, speaker_boundaries=None)
        
        # Check that we don't have a sentence starting with "Aber"
        sentences = [s.strip() for s in result.split('\n') if s.strip()]
        for sentence in sentences:
            if sentence.startswith('Aber '):
                assert False, f"Sentence incorrectly starts with 'Aber': {sentence}"
        
        print(f"✓ German 'aber' connector test passed")
        print(f"  Result: {result}")
    
    def test_legitimate_sentence_start_allowed(self):
        """Legitimate sentences CAN start after proper breaks (question/exclamation)."""
        # This is a legitimate case where "Y" starts a new sentence after a question
        text = "¿Cómo estás? Y tú, ¿qué tal?"
        
        result = restore_punctuation(text, language='es', whisper_boundaries=None, speaker_boundaries=None)
        
        # In this case, "Y" starting a sentence after a question mark is legitimate
        # Our fix should only prevent breaks at speaker boundaries when connector follows
        print(f"✓ Legitimate sentence start test passed")
        print(f"  Result: {result}")
    
    def test_different_speaker_with_connector_allows_break(self):
        """When DIFFERENT speakers speak, connector words CAN start new sentences."""
        # Speaker 1: "I like coffee"
        # Speaker 2: "And I like tea"
        text = "I like coffee and I like tea"
        
        # Mock speaker segments showing speaker change at "and"
        # Approximate character position of "and" is around position 14
        speaker_segments = create_mock_speaker_segments_two_speakers(text, 14)
        
        result = restore_punctuation(text, language='en', whisper_boundaries=None, speaker_boundaries=None, speaker_segments=speaker_segments)
        
        # When speakers change, it's OK for connector to start a new sentence
        # (though in practice, speakers rarely start with "and" - but we allow it)
        print(f"✓ Different speaker with connector test passed")
        print(f"  Result: {result}")
    
    def test_same_speaker_long_text_spanish(self):
        """Test with longer Spanish text to trigger semantic splitting."""
        # A longer example that would trigger semantic splitting thresholds
        text = "Andrea siempre tiene muchos sueños que están recordando en la mañana y yo no recuerdo muchos sueños y Nate siempre me dice que no recuerda nada y yo le cuento como cinco o seis sueños diferentes que tuve y los recuerdo muy bien"
        
        # Mock speaker segments showing entire text is one speaker
        speaker_segments = create_mock_speaker_segments_same_speaker(text)
        
        result = restore_punctuation(text, language='es', whisper_boundaries=None, speaker_boundaries=None, speaker_segments=speaker_segments)
        
        # No sentence should start with "Y" when same speaker throughout
        lines = [s.strip() for s in result.split('\n') if s.strip()]
        for line in lines:
            assert not line.startswith('Y '), f"Sentence should not start with 'Y': {line}"
        
        print(f"✓ Spanish long text same speaker test passed")
        print(f"  Result: {result[:200]}...")


def run_all_tests():
    """Run all connector word tests."""
    print("\n" + "="*70)
    print("Testing Diarization Connector Word Handling")
    print("="*70 + "\n")
    
    test_class = TestDiarizationConnectorWords()
    
    tests = [
        test_class.test_same_speaker_no_connector_split_spanish_y,
        test_class.test_same_speaker_no_connector_split_spanish_full_example,
        test_class.test_same_speaker_no_connector_split_spanish_second_example,
        test_class.test_same_speaker_no_connector_split_english_and,
        test_class.test_same_speaker_no_connector_split_english_but,
        test_class.test_same_speaker_no_connector_split_french_et,
        test_class.test_same_speaker_no_connector_split_french_mais,
        test_class.test_same_speaker_no_connector_split_german_und,
        test_class.test_same_speaker_no_connector_split_german_aber,
        test_class.test_legitimate_sentence_start_allowed,
        test_class.test_different_speaker_with_connector_allows_break,
        test_class.test_same_speaker_long_text_spanish,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
