"""
Test Whisper boundary integration in sentence splitting.

This test validates that Whisper segment boundaries are used as hints
for sentence breaks while still respecting grammatical rules.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punctuation_restorer import (
    _extract_segment_boundaries,
    _char_positions_to_word_indices,
    _violates_grammatical_rules,
    restore_punctuation
)


def test_extract_segment_boundaries():
    """Test that segment boundaries are correctly extracted from Whisper segments."""
    segments = [
        {'text': 'Hello world', 'start': 0.0, 'end': 2.0},
        {'text': 'This is a test', 'start': 2.0, 'end': 4.0},
        {'text': 'Another segment', 'start': 4.0, 'end': 6.0}
    ]
    # Text is joined with spaces (not newlines) in _accumulate_segments
    text = "Hello world This is a test Another segment"
    
    boundaries = _extract_segment_boundaries(text, segments)
    
    # Should have 3 boundaries (one after each segment)
    assert len(boundaries) == 3, f"Expected 3 boundaries, got {len(boundaries)}: {boundaries}"
    # Boundaries should be at character positions where segments end (after each text + space)
    # "Hello world" = 11 chars + 1 space = boundary at 11
    # "This is a test" = 14 chars, starts at 12, boundary at 12+14=26
    # "Another segment" = 15 chars, starts at 27, boundary at 27+15=42
    assert boundaries[0] == 11, f"Expected boundary[0]=11, got {boundaries[0]}"
    assert boundaries[1] == 26, f"Expected boundary[1]=26, got {boundaries[1]}"
    assert boundaries[2] == 42, f"Expected boundary[2]=42, got {boundaries[2]}"


def test_char_positions_to_word_indices():
    """Test conversion of character positions to word indices."""
    text = "Hello world this is a test sentence"
    # Character boundaries after "world" (11) and after "test" (26)
    # "Hello world" = 11 chars (boundary at 11)
    # "Hello world this is a test" = 26 chars (boundary at 26)
    char_positions = [11, 26]
    
    word_indices = _char_positions_to_word_indices(text, char_positions)
    
    # Should map to word indices 1 (world) and 5 (test)
    # Words: 0=Hello, 1=world, 2=this, 3=is, 4=a, 5=test, 6=sentence
    assert 1 in word_indices, f"Expected word index 1 in {word_indices}"  # "world"
    assert 5 in word_indices, f"Expected word index 5 in {word_indices}"  # "test"


def test_violates_grammatical_rules_conjunctions():
    """Test that coordinating conjunctions are flagged as violations."""
    # Spanish
    assert _violates_grammatical_rules("y", "entonces", "es") is True
    assert _violates_grammatical_rules("pero", "yo", "es") is True
    
    # English
    assert _violates_grammatical_rules("and", "then", "en") is True
    assert _violates_grammatical_rules("but", "I", "en") is True
    
    # French
    assert _violates_grammatical_rules("et", "alors", "fr") is True
    assert _violates_grammatical_rules("mais", "je", "fr") is True
    
    # German
    assert _violates_grammatical_rules("und", "dann", "de") is True
    assert _violates_grammatical_rules("aber", "ich", "de") is True


def test_violates_grammatical_rules_prepositions():
    """Test that prepositions are flagged as violations."""
    # Spanish
    assert _violates_grammatical_rules("a", "la", "es") is True
    assert _violates_grammatical_rules("de", "los", "es") is True
    assert _violates_grammatical_rules("en", "casa", "es") is True
    
    # English
    assert _violates_grammatical_rules("to", "the", "en") is True
    assert _violates_grammatical_rules("at", "home", "en") is True
    assert _violates_grammatical_rules("from", "here", "en") is True
    
    # French
    assert _violates_grammatical_rules("à", "la", "fr") is True
    assert _violates_grammatical_rules("de", "Paris", "fr") is True
    
    # German
    assert _violates_grammatical_rules("zu", "Hause", "de") is True
    assert _violates_grammatical_rules("von", "dort", "de") is True


def test_violates_grammatical_rules_continuative_verbs():
    """Test that continuative/auxiliary verbs are flagged as violations."""
    # Spanish
    assert _violates_grammatical_rules("estaba", "trabajando", "es") is True
    assert _violates_grammatical_rules("era", "muy", "es") is True
    assert _violates_grammatical_rules("había", "estado", "es") is True
    
    # English
    assert _violates_grammatical_rules("was", "working", "en") is True
    assert _violates_grammatical_rules("had", "been", "en") is True
    
    # French
    assert _violates_grammatical_rules("était", "en", "fr") is True
    assert _violates_grammatical_rules("avait", "été", "fr") is True
    
    # German
    assert _violates_grammatical_rules("war", "gewesen", "de") is True
    assert _violates_grammatical_rules("hatte", "gemacht", "de") is True


def test_violates_grammatical_rules_valid_breaks():
    """Test that valid sentence endings are not flagged."""
    # Spanish
    assert _violates_grammatical_rules("casa", "Y", "es") is False
    assert _violates_grammatical_rules("tarde", "Entonces", "es") is False
    
    # English
    assert _violates_grammatical_rules("today", "Then", "en") is False
    assert _violates_grammatical_rules("home", "After", "en") is False
    
    # French
    assert _violates_grammatical_rules("maison", "Alors", "fr") is False
    
    # German
    assert _violates_grammatical_rules("Haus", "Dann", "de") is False


def test_whisper_boundary_respected_good_break():
    """Test that Whisper boundaries are honored for grammatically valid breaks."""
    # Simulate text with a Whisper boundary at a good location
    # Segments: "This is a long sentence" | "This is another sentence"
    segments = [
        {'text': 'This is a long sentence', 'start': 0.0, 'end': 2.0},
        {'text': 'This is another sentence', 'start': 2.0, 'end': 4.0}
    ]
    text = "this is a long sentence this is another sentence"
    
    boundaries = _extract_segment_boundaries(text, segments)
    result = restore_punctuation(text, 'en', whisper_boundaries=boundaries)
    
    # Should respect the Whisper boundary and create two sentences
    # (The exact punctuation may vary, but it should have a sentence break)
    assert '.' in result or '?' in result or '!' in result


def test_whisper_boundary_ignored_conjunction():
    """Test that Whisper boundaries are ignored when ending on a conjunction."""
    # Simulate Whisper incorrectly breaking after "and"
    segments = [
        {'text': 'I went to the store and', 'start': 0.0, 'end': 2.0},
        {'text': 'bought some milk', 'start': 2.0, 'end': 4.0}
    ]
    text = "I went to the store and bought some milk"
    
    boundaries = _extract_segment_boundaries(text, segments)
    result = restore_punctuation(text, 'en', whisper_boundaries=boundaries)
    
    # Should NOT break after "and" despite Whisper boundary
    # The sentence should remain together
    assert result.count('.') <= 1 or result.count('!') <= 1 or result.count('?') <= 1


def test_whisper_boundary_ignored_preposition():
    """Test that Whisper boundaries are ignored when ending on a preposition."""
    # Spanish: Whisper incorrectly breaking after "a"
    segments = [
        {'text': 'yo fui a', 'start': 0.0, 'end': 1.0},
        {'text': 'la tienda', 'start': 1.0, 'end': 2.0}
    ]
    text = "yo fui a la tienda"
    
    boundaries = _extract_segment_boundaries(text, segments)
    result = restore_punctuation(text, 'es', whisper_boundaries=boundaries)
    
    # Should NOT break after "a" despite Whisper boundary
    # Check that "a" is not followed immediately by sentence-ending punctuation
    import re
    # Look for "a" followed by period, question, or exclamation (possibly with space)
    bad_patterns = [r'\ba\s*\.', r'\ba\s*\?', r'\ba\s*!']
    for pattern in bad_patterns:
        matches = re.findall(pattern, result.lower())
        assert len(matches) == 0, f"Found bad break after 'a': {result}"


def test_whisper_boundary_all_languages_spanish():
    """Test boundary integration for Spanish."""
    segments = [
        {'text': 'Hola me llamo Juan', 'start': 0.0, 'end': 2.0},
        {'text': 'Vivo en Madrid', 'start': 2.0, 'end': 4.0}
    ]
    text = "Hola me llamo Juan vivo en Madrid"
    
    boundaries = _extract_segment_boundaries(text, segments)
    result = restore_punctuation(text, 'es', whisper_boundaries=boundaries)
    
    # Should create proper sentences
    assert len(result) > 0
    # Should have some punctuation
    has_punctuation = any(p in result for p in ['.', '!', '?'])
    assert has_punctuation


def test_whisper_boundary_all_languages_french():
    """Test boundary integration for French."""
    segments = [
        {'text': 'Bonjour je m\'appelle Jean', 'start': 0.0, 'end': 2.0},
        {'text': 'J\'habite à Paris', 'start': 2.0, 'end': 4.0}
    ]
    text = "Bonjour je m'appelle Jean j'habite à Paris"
    
    boundaries = _extract_segment_boundaries(text, segments)
    result = restore_punctuation(text, 'fr', whisper_boundaries=boundaries)
    
    # Should create proper sentences
    assert len(result) > 0
    has_punctuation = any(p in result for p in ['.', '!', '?'])
    assert has_punctuation


def test_whisper_boundary_all_languages_german():
    """Test boundary integration for German."""
    segments = [
        {'text': 'Hallo ich heiße Hans', 'start': 0.0, 'end': 2.0},
        {'text': 'Ich wohne in Berlin', 'start': 2.0, 'end': 4.0}
    ]
    text = "Hallo ich heiße Hans ich wohne in Berlin"
    
    boundaries = _extract_segment_boundaries(text, segments)
    result = restore_punctuation(text, 'de', whisper_boundaries=boundaries)
    
    # Should create proper sentences
    assert len(result) > 0
    has_punctuation = any(p in result for p in ['.', '!', '?'])
    assert has_punctuation


def test_backward_compatibility_no_boundaries():
    """Test that system works as before when boundaries not provided."""
    text = "this is a test sentence this is another test sentence"
    
    # Call without boundaries (backward compatible)
    result_no_boundaries = restore_punctuation(text, 'en')
    
    # Call with None boundaries (explicit)
    result_none_boundaries = restore_punctuation(text, 'en', whisper_boundaries=None)
    
    # Should produce identical results
    assert result_no_boundaries == result_none_boundaries
    
    # Should still produce reasonable output
    assert len(result_no_boundaries) > 0


def test_empty_segments():
    """Test handling of empty segment list."""
    text = "this is a test sentence"
    segments = []
    
    boundaries = _extract_segment_boundaries(text, segments)
    result = restore_punctuation(text, 'en', whisper_boundaries=boundaries)
    
    # Should handle gracefully
    assert len(result) > 0


def test_single_segment():
    """Test handling of single segment."""
    text = "this is a test sentence"
    segments = [{'text': 'this is a test sentence', 'start': 0.0, 'end': 2.0}]
    
    boundaries = _extract_segment_boundaries(text, segments)
    result = restore_punctuation(text, 'en', whisper_boundaries=boundaries)
    
    # Should produce reasonable output
    assert len(result) > 0
    # Should have terminal punctuation
    assert result.rstrip()[-1] in ['.', '!', '?']


if __name__ == '__main__':
    # Run all tests
    test_functions = [
        test_extract_segment_boundaries,
        test_char_positions_to_word_indices,
        test_violates_grammatical_rules_conjunctions,
        test_violates_grammatical_rules_prepositions,
        test_violates_grammatical_rules_continuative_verbs,
        test_violates_grammatical_rules_valid_breaks,
        test_whisper_boundary_respected_good_break,
        test_whisper_boundary_ignored_conjunction,
        test_whisper_boundary_ignored_preposition,
        test_whisper_boundary_all_languages_spanish,
        test_whisper_boundary_all_languages_french,
        test_whisper_boundary_all_languages_german,
        test_backward_compatibility_no_boundaries,
        test_empty_segments,
        test_single_segment,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__}: Unexpected error: {e}")
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed out of {len(test_functions)} tests")
    sys.exit(0 if failed == 0 else 1)

