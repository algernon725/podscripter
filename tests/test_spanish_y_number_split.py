#!/usr/bin/env python3
"""
Test the specific Spanish "y 184" splitting bug.
"""

from punctuation_restorer import restore_punctuation, _semantic_split_into_sentences, _load_sentence_transformer


def test_spanish_y_number():
    """Test that 'y 184' doesn't get split in Spanish."""
    
    # This is the exact pattern from Episodio 190
    text = "puedes ir al episodio 147, 151, 156, 164, 170, 177 y 184"
    
    print("="*70)
    print("Testing Spanish 'y NUMBER' pattern")
    print("="*70)
    print(f"Input: {text}")
    print()
    
    # Test semantic splitting directly
    model = _load_sentence_transformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # Test with period at the end (as Whisper would produce)
    text_with_period = text + "."
    sentences = _semantic_split_into_sentences(text_with_period, 'es', model)
    
    print(f"Semantic split result ({len(sentences)} sentences):")
    for i, s in enumerate(sentences, 1):
        print(f"  {i}. '{s}'")
    print()
    
    # Check if split incorrectly
    if any(s.strip().endswith("y.") or s.strip().endswith("y") for s in sentences):
        print("❌ FAILED: 'y' was separated from '184'")
        return False
    
    if any(s.strip() == "184." or s.strip() == "184" for s in sentences):
        print("❌ FAILED: '184' is a standalone sentence")
        return False
    
    # Check if kept together
    if any("y 184" in s for s in sentences):
        print("✅ SUCCESS: 'y 184' kept together")
        return True
    else:
        print("❌ FAILED: Pattern not found")
        return False


def test_full_restoration_spanish():
    """Test full restoration pipeline for Spanish."""
    
    text = "Pero si tú quieres escuchar los episodios anteriores, puedes ir al episodio 147, 151, 156, 164, 170, 177 y 184"
    
    print("="*70)
    print("Testing full Spanish restoration")
    print("="*70)
    print(f"Input:\n  {text}")
    print()
    
    result = restore_punctuation(text, language='es')
    print(f"Output:\n  {result}")
    print()
    
    # Check for the bug
    if "y." in result and "184" in result:
        # Check if they're separated
        if "y. 184" in result or "y.\n184" in result:
            print("❌ FAILED: 'y. 184' found (split)")
            return False
    
    if "y 184" in result or "177 y 184" in result:
        print("✅ SUCCESS: Number list preserved")
        return True
    else:
        print("⚠️  Unexpected output format")
        return False


def test_simple_spanish_list():
    """Test simple Spanish number list."""
    text = "Los episodios son 1, 2, 3 y 4"
    
    print("="*70)
    print("Testing simple Spanish list")
    print("="*70)
    print(f"Input: {text}")
    
    result = restore_punctuation(text, language='es')
    print(f"Output: {result}")
    print()
    
    if "3 y 4" in result:
        print("✅ Simple list preserved")
        return True
    else:
        print("❌ Simple list broken")
        return False


if __name__ == "__main__":
    results = []
    
    results.append(test_spanish_y_number())
    print()
    results.append(test_full_restoration_spanish())
    print()
    results.append(test_simple_spanish_list())
    print()
    
    print("="*70)
    if all(results):
        print("🎉 All tests passed! Spanish 'y NUMBER' fix working")
    else:
        print(f"❌ {results.count(False)}/{len(results)} tests failed")
    print("="*70)

