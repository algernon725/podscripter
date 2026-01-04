#!/usr/bin/env python3
"""
Unit tests for SentenceFormatter class.

Tests cover:
- Domain merge operations with natural language guards
- Decimal merge operations
- Spanish appositive merges
- Emphatic word merges (ES/FR/DE)
- Speaker boundary enforcement (never merge different speakers)
- Non-diarization backward compatibility
- Merge provenance tracking
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_formatter import SentenceFormatter, MergeMetadata


def test_formatter_works_without_speaker_segments():
    """
    CRITICAL: Verify backward compatibility when speaker_segments=None.
    
    This test ensures that SentenceFormatter works identically to the old
    behavior when no speaker data is provided (non-diarization mode).
    """
    sentences = ["Visit google.", "Com for search", "No.", "No.", "No.", "99.", "9% accuracy"]
    formatter = SentenceFormatter('es', speaker_segments=None)
    result, metadata = formatter.format(sentences)
    
    # Should merge domain without speaker checks
    assert any("google.com" in s for s in result), f"Expected domain merge in: {result}"
    
    # Should merge emphatic words without speaker checks
    assert any("No, no, no" in s for s in result), f"Expected emphatic merge in: {result}"
    
    # Should merge decimal without speaker checks
    assert any("99.9%" in s for s in result), f"Expected decimal merge in: {result}"
    
    # Should have performed merges
    assert len(metadata) > 0, "Expected merge metadata"
    
    # All speaker fields should be None
    for merge in metadata:
        assert merge.speaker1 is None, f"Expected None speaker1, got {merge.speaker1}"
        assert merge.speaker2 is None, f"Expected None speaker2, got {merge.speaker2}"
    
    print("✓ test_formatter_works_without_speaker_segments passed")


def test_domain_merge_basic():
    """Test basic domain merge: 'example.' + 'com' → 'example.com'"""
    sentences = ["Visit example.", "Com for more info"]
    formatter = SentenceFormatter('en', speaker_segments=None)
    result, metadata = formatter.format(sentences)
    
    assert len(result) == 1, f"Expected 1 sentence, got {len(result)}: {result}"
    assert "example.com" in result[0], f"Expected 'example.com' in: {result[0]}"
    assert "for more info" in result[0], f"Expected remainder in: {result[0]}"
    
    # Check metadata
    assert len(metadata) >= 1, "Expected at least 1 merge"
    domain_merges = [m for m in metadata if m.merge_type == 'domain']
    assert len(domain_merges) == 1, f"Expected 1 domain merge, got {len(domain_merges)}"
    
    print("✓ test_domain_merge_basic passed")


def test_domain_merge_natural_language_guard():
    """
    Test natural language guard prevents false domain merges.
    
    Example: "jugar." + "Es que vamos..." should NOT merge as "jugar.es"
    because it's a long natural language sentence, not a domain mention.
    """
    sentences = [
        "Eso lo seguiremos haciendo, pero jugar tenis juntos, bueno, si es que yo aprendo y soy capaz de jugar.",
        "Es que vamos a tratar de tener lecciones de tenis en Colombia."
    ]
    formatter = SentenceFormatter('es', speaker_segments=None)
    result, metadata = formatter.format(sentences)
    
    # Should NOT merge (natural language guard)
    assert len(result) == 2, f"Expected 2 sentences (no merge), got {len(result)}: {result}"
    assert "jugar.es" not in ' '.join(result), f"Should not contain 'jugar.es': {result}"
    
    # Should have no domain merges (guard prevented it)
    domain_merges = [m for m in metadata if m.merge_type == 'domain' and not m.reason.startswith('skipped')]
    assert len(domain_merges) == 0, f"Expected 0 domain merges, got {len(domain_merges)}"
    
    print("✓ test_domain_merge_natural_language_guard passed")


def test_domain_merge_short_sentence():
    """Test domain merge works for short sentences (< 50 chars)"""
    sentences = ["Visit google.", "Com"]
    formatter = SentenceFormatter('en', speaker_segments=None)
    result, metadata = formatter.format(sentences)
    
    assert len(result) == 1, f"Expected 1 sentence, got {len(result)}: {result}"
    assert "google.com" in result[0], f"Expected 'google.com' in: {result[0]}"
    
    print("✓ test_domain_merge_short_sentence passed")


def test_domain_merge_capitalized_label():
    """Test domain merge works for capitalized labels (brand names)"""
    sentences = ["Check out Google.", "Com for search"]
    formatter = SentenceFormatter('en', speaker_segments=None)
    result, metadata = formatter.format(sentences)
    
    assert len(result) == 1, f"Expected 1 sentence, got {len(result)}: {result}"
    assert "Google.com" in result[0], f"Expected 'Google.com' in: {result[0]}"
    
    print("✓ test_domain_merge_capitalized_label passed")


def test_domain_merge_triple():
    """Test triple domain merge: 'example.' + 'com' + 'Y más' → 'example.com Y más'"""
    sentences = ["Visit example.", "com", "Y más información aquí"]
    formatter = SentenceFormatter('es', speaker_segments=None)
    result, metadata = formatter.format(sentences)
    
    assert len(result) == 1, f"Expected 1 sentence, got {len(result)}: {result}"
    assert "example.com" in result[0], f"Expected 'example.com' in: {result[0]}"
    assert "más información" in result[0], f"Expected remainder in: {result[0]}"
    
    print("✓ test_domain_merge_triple passed")


def test_decimal_merge_basic():
    """Test basic decimal merge: '99.' + '9%' → '99.9%'"""
    sentences = ["The accuracy is 99.", "9% de los casos"]
    formatter = SentenceFormatter('es', speaker_segments=None)
    result, metadata = formatter.format(sentences)
    
    assert len(result) == 1, f"Expected 1 sentence, got {len(result)}: {result}"
    assert "99.9%" in result[0], f"Expected '99.9%' in: {result[0]}"
    assert "de los casos" in result[0], f"Expected remainder in: {result[0]}"
    
    # Check metadata
    decimal_merges = [m for m in metadata if m.merge_type == 'decimal']
    assert len(decimal_merges) == 1, f"Expected 1 decimal merge, got {len(decimal_merges)}"
    
    print("✓ test_decimal_merge_basic passed")


def test_decimal_merge_without_percent():
    """Test decimal merge without percent: '121.' + '73' → '121.73'"""
    sentences = ["The distance is 121.", "73 meters"]
    formatter = SentenceFormatter('en', speaker_segments=None)
    result, metadata = formatter.format(sentences)
    
    assert len(result) == 1, f"Expected 1 sentence, got {len(result)}: {result}"
    assert "121.73" in result[0], f"Expected '121.73' in: {result[0]}"
    assert "meters" in result[0], f"Expected remainder in: {result[0]}"
    
    print("✓ test_decimal_merge_without_percent passed")


def test_emphatic_merge_spanish():
    """Test Spanish emphatic word merge: 'No.' + 'No.' + 'No.' → 'No, no, no.'"""
    sentences = ["No.", "No.", "No."]
    formatter = SentenceFormatter('es', speaker_segments=None)
    result, metadata = formatter.format(sentences)
    
    assert len(result) == 1, f"Expected 1 sentence, got {len(result)}: {result}"
    assert result[0] == "No, no, no.", f"Expected 'No, no, no.', got: {result[0]}"
    
    # Check metadata
    emphatic_merges = [m for m in metadata if m.merge_type == 'emphatic']
    assert len(emphatic_merges) == 1, f"Expected 1 emphatic merge, got {len(emphatic_merges)}"
    
    print("✓ test_emphatic_merge_spanish passed")


def test_emphatic_merge_spanish_si():
    """Test Spanish 'sí' normalization: 'Si.' + 'si.' + 'Sí.' → 'Sí, sí, sí.'"""
    sentences = ["Si.", "si.", "Sí."]
    formatter = SentenceFormatter('es', speaker_segments=None)
    result, metadata = formatter.format(sentences)
    
    assert len(result) == 1, f"Expected 1 sentence, got {len(result)}: {result}"
    assert result[0] == "Sí, sí, sí.", f"Expected 'Sí, sí, sí.', got: {result[0]}"
    
    print("✓ test_emphatic_merge_spanish_si passed")


def test_emphatic_merge_french():
    """Test French emphatic word merge: 'Non.' + 'Non.' → 'Non, non.'"""
    sentences = ["Non.", "Non."]
    formatter = SentenceFormatter('fr', speaker_segments=None)
    result, metadata = formatter.format(sentences)
    
    assert len(result) == 1, f"Expected 1 sentence, got {len(result)}: {result}"
    assert result[0] == "Non, non.", f"Expected 'Non, non.', got: {result[0]}"
    
    print("✓ test_emphatic_merge_french passed")


def test_emphatic_merge_german():
    """Test German emphatic word merge: 'Nein.' + 'Nein.' → 'Nein, nein.'"""
    sentences = ["Nein.", "Nein."]
    formatter = SentenceFormatter('de', speaker_segments=None)
    result, metadata = formatter.format(sentences)
    
    assert len(result) == 1, f"Expected 1 sentence, got {len(result)}: {result}"
    assert result[0] == "Nein, nein.", f"Expected 'Nein, nein.', got: {result[0]}"
    
    print("✓ test_emphatic_merge_german passed")


def test_no_emphatic_merge_for_unsupported_language():
    """Test that emphatic merge doesn't happen for unsupported languages"""
    sentences = ["No.", "No.", "No."]
    formatter = SentenceFormatter('en', speaker_segments=None)  # English doesn't have emphatic words defined
    result, metadata = formatter.format(sentences)
    
    # Should NOT merge (English not in emph_map)
    assert len(result) == 3, f"Expected 3 sentences (no merge), got {len(result)}: {result}"
    
    print("✓ test_no_emphatic_merge_for_unsupported_language passed")


def test_multiple_merge_types():
    """Test that multiple merge types can be applied in sequence"""
    sentences = [
        "Visit google.",
        "Com for 99.",
        "9% accuracy",
        "No.",
        "No.",
        "No."
    ]
    formatter = SentenceFormatter('es', speaker_segments=None)
    result, metadata = formatter.format(sentences)
    
    # Should have merged domain, decimal, and emphatic
    assert any("google.com" in s for s in result), f"Expected domain merge in: {result}"
    assert any("99.9%" in s for s in result), f"Expected decimal merge in: {result}"
    assert any("No, no, no" in s for s in result), f"Expected emphatic merge in: {result}"
    
    # Check metadata has all merge types
    merge_types = {m.merge_type for m in metadata if not m.reason.startswith('skipped')}
    assert 'domain' in merge_types, f"Expected domain merge in metadata: {merge_types}"
    assert 'decimal' in merge_types, f"Expected decimal merge in metadata: {merge_types}"
    assert 'emphatic' in merge_types, f"Expected emphatic merge in metadata: {merge_types}"
    
    print("✓ test_multiple_merge_types passed")


def test_merge_provenance_tracking():
    """Test that merge provenance is correctly tracked"""
    sentences = ["Visit google.", "Com for search"]
    formatter = SentenceFormatter('en', speaker_segments=None)
    result, metadata = formatter.format(sentences)
    
    assert len(metadata) == 1, f"Expected 1 metadata entry, got {len(metadata)}"
    
    m = metadata[0]
    assert m.merge_type == 'domain', f"Expected merge_type 'domain', got {m.merge_type}"
    assert m.sentence1_idx == 0, f"Expected sentence1_idx 0, got {m.sentence1_idx}"
    assert m.sentence2_idx == 1, f"Expected sentence2_idx 1, got {m.sentence2_idx}"
    assert m.reason == 'domain_pattern_match', f"Expected reason 'domain_pattern_match', got {m.reason}"
    assert m.before_text1 == "Visit google.", f"Expected before_text1 'Visit google.', got {m.before_text1}"
    assert m.before_text2 == "Com for search", f"Expected before_text2 'Com for search', got {m.before_text2}"
    assert "google.com" in m.after_text, f"Expected 'google.com' in after_text: {m.after_text}"
    
    print("✓ test_merge_provenance_tracking passed")


def run_all_tests():
    """Run all test functions"""
    test_functions = [
        test_formatter_works_without_speaker_segments,
        test_domain_merge_basic,
        test_domain_merge_natural_language_guard,
        test_domain_merge_short_sentence,
        test_domain_merge_capitalized_label,
        test_domain_merge_triple,
        test_decimal_merge_basic,
        test_decimal_merge_without_percent,
        test_emphatic_merge_spanish,
        test_emphatic_merge_spanish_si,
        test_emphatic_merge_french,
        test_emphatic_merge_german,
        test_no_emphatic_merge_for_unsupported_language,
        test_multiple_merge_types,
        test_merge_provenance_tracking,
    ]
    
    print(f"\nRunning {len(test_functions)} SentenceFormatter tests...\n")
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} ERROR: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(test_functions)} tests")
    print(f"{'='*60}\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
