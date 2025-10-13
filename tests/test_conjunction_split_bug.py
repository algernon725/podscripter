#!/usr/bin/env python3
"""
Test to reproduce and verify fix for the conjunction splitting bug.

Issue: Sentences are being split after coordinating conjunctions like "y" (and).
Example: "teníamos muchos errores y." | "Eco una vez..." 
This is grammatically incorrect - "y" should never end a sentence.

This test verifies the fix works across all supported languages.
"""

from punctuation_restorer import restore_punctuation


def test_spanish_y_followed_by_regular_word():
    """Test that Spanish 'y' + regular word doesn't split."""
    
    # This is the exact pattern from Episodio 190 that caused the bug
    text = "teníamos muchos errores y eco"
    
    print("="*70)
    print("Testing Spanish 'y' + regular word")
    print("="*70)
    print(f"Input: {text}")
    
    result = restore_punctuation(text, language='es')
    print(f"Output: {result}")
    print()
    
    # Check that 'y' is not at the end of a sentence
    if "y." in result or "y!" in result or "y?" in result:
        # Make sure it's actually ending a sentence, not just appearing mid-text
        if result.strip().endswith("y.") or ". Eco" in result or ".\nEco" in result:
            print("❌ FAILED: 'y' was separated from 'eco'")
            return False
    
    if "y eco" in result or "y Eco" in result:
        print("✅ SUCCESS: 'y eco' kept together")
        return True
    else:
        print("⚠️  Unexpected output format")
        return False


def test_spanish_pero_followed_by_word():
    """Test that Spanish 'pero' (but) + regular word doesn't split."""
    
    text = "era muy difícil pero empezamos a trabajar"
    
    print("="*70)
    print("Testing Spanish 'pero' + regular word")
    print("="*70)
    print(f"Input: {text}")
    
    result = restore_punctuation(text, language='es')
    print(f"Output: {result}")
    print()
    
    # Check that 'pero' is not at the end of a sentence
    if result.strip().endswith("pero.") or ". empezamos" in result.lower():
        print("❌ FAILED: 'pero' was separated from 'empezamos'")
        return False
    
    if "pero empezamos" in result.lower():
        print("✅ SUCCESS: 'pero empezamos' kept together")
        return True
    else:
        print("⚠️  Unexpected output format")
        return False


def test_english_and_followed_by_word():
    """Test that English 'and' + regular word doesn't split."""
    
    text = "we had many errors and echo in the garage"
    
    print("="*70)
    print("Testing English 'and' + regular word")
    print("="*70)
    print(f"Input: {text}")
    
    result = restore_punctuation(text, language='en')
    print(f"Output: {result}")
    print()
    
    # Check that 'and' is not at the end of a sentence
    if result.strip().endswith("and.") or ". echo" in result.lower() or ".\necho" in result.lower():
        print("❌ FAILED: 'and' was separated from 'echo'")
        return False
    
    if "and echo" in result.lower():
        print("✅ SUCCESS: 'and echo' kept together")
        return True
    else:
        print("⚠️  Unexpected output format")
        return False


def test_french_et_followed_by_word():
    """Test that French 'et' (and) + regular word doesn't split."""
    
    text = "nous avions beaucoup d'erreurs et écho dans le garage"
    
    print("="*70)
    print("Testing French 'et' + regular word")
    print("="*70)
    print(f"Input: {text}")
    
    result = restore_punctuation(text, language='fr')
    print(f"Output: {result}")
    print()
    
    # Check that 'et' is not at the end of a sentence
    if result.strip().endswith("et.") or ". écho" in result.lower() or ".\nécho" in result.lower():
        print("❌ FAILED: 'et' was separated from 'écho'")
        return False
    
    if "et écho" in result.lower():
        print("✅ SUCCESS: 'et écho' kept together")
        return True
    else:
        print("⚠️  Unexpected output format")
        return False


def test_german_und_followed_by_word():
    """Test that German 'und' (and) + regular word doesn't split."""
    
    text = "wir hatten viele Fehler und Echo in der Garage"
    
    print("="*70)
    print("Testing German 'und' + regular word")
    print("="*70)
    print(f"Input: {text}")
    
    result = restore_punctuation(text, language='de')
    print(f"Output: {result}")
    print()
    
    # Check that 'und' is not at the end of a sentence
    if result.strip().endswith("und.") or ". Echo" in result or ".\nEcho" in result:
        print("❌ FAILED: 'und' was separated from 'Echo'")
        return False
    
    if "und Echo" in result:
        print("✅ SUCCESS: 'und Echo' kept together")
        return True
    else:
        print("⚠️  Unexpected output format")
        return False


def test_longer_spanish_context():
    """Test with longer context similar to the actual bug."""
    
    text = "Ah sí no eran muy muy buenos teníamos muchos errores y eco una vez que grabábamos el video en el garaje y había tanto eco"
    
    print("="*70)
    print("Testing longer Spanish context (actual bug scenario)")
    print("="*70)
    print(f"Input: {text[:80]}...")
    
    result = restore_punctuation(text, language='es')
    print(f"Output: {result[:100]}...")
    print()
    
    # Check that neither 'y' instance ends a sentence
    sentences = result.split('.')
    for i, sent in enumerate(sentences):
        sent_stripped = sent.strip()
        if sent_stripped.endswith(' y') or sent_stripped.endswith(',y'):
            print(f"❌ FAILED: Found sentence ending with 'y': '{sent_stripped}'")
            return False
    
    print("✅ SUCCESS: No sentences end with 'y'")
    return True


if __name__ == "__main__":
    results = []
    
    results.append(test_spanish_y_followed_by_regular_word())
    print()
    results.append(test_spanish_pero_followed_by_word())
    print()
    results.append(test_english_and_followed_by_word())
    print()
    results.append(test_french_et_followed_by_word())
    print()
    results.append(test_german_und_followed_by_word())
    print()
    results.append(test_longer_spanish_context())
    print()
    
    print("="*70)
    if all(results):
        print("🎉 All tests passed! Conjunction split bug fixed")
        print("="*70)
        exit(0)
    else:
        print(f"❌ {results.count(False)}/{len(results)} tests failed")
        print("="*70)
        exit(1)

