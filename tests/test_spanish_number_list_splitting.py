#!/usr/bin/env python3
"""
Test for Spanish number list splitting bug.

Bug: Sentences containing lists of numbers with conjunctions (y/o) are being
incorrectly split at the final number's period.

Example:
Input: "puedes ir al episodio 147, 151, 156, 164, 170, 177 y 184."
Expected: Single sentence
Actual: Split into "...177 y." and "184."
"""

from punctuation_restorer import restore_punctuation, assemble_sentences_from_processed


def test_number_list_with_y_not_split():
    """Test that a list of numbers with 'y' is not split at the final number."""
    # This matches the actual Whisper output which already has commas
    text = "Pero si tú quieres escuchar los episodios anteriores, puedes ir al episodio 147, 151, 156, 164, 170, 177 y 184"
    result = restore_punctuation(text, language='es')
    
    # The sentence should remain as one complete sentence
    # It should NOT be split into "...y." and "184."
    # Check if incorrectly split by looking for a sentence break before 184
    lines = result.split('\n')
    
    # If split incorrectly, we'd see "184" as its own sentence/line
    standalone_184 = any(line.strip() == "184." for line in lines)
    assert not standalone_184, \
        f"Number list incorrectly split - 184 is standalone. Result:\n{result}"
    
    # Also check the text doesn't have "y." followed by newline or sentence break
    assert not ("y.\n184" in result or "y. 184." == result.split('\n')[-1].strip()), \
        f"Number list incorrectly split. Result:\n{result}"
    
    # The full number sequence should be preserved together
    assert "177 y 184" in result, \
        f"Number list broken apart. Result:\n{result}"
    
    print(f"✓ Number list preserved correctly:\n{result}")


def test_simple_number_list_with_y():
    """Test a simple number list with 'y' conjunction."""
    text = "Los episodios son 1 2 3 y 4"
    result = restore_punctuation(text, language='es')
    
    # Should keep the list together
    assert "3 y 4" in result, \
        f"Simple number list broken. Result: {result}"
    assert "y." not in result or "y. 4" not in result, \
        f"Incorrectly split at 'y'. Result: {result}"
    
    print(f"✓ Simple number list preserved: {result}")


def test_number_list_with_o():
    """Test that number lists with 'o' (or) are also preserved."""
    text = "Puedes elegir la opción 1 2 o 3"
    result = restore_punctuation(text, language='es')
    
    # Should keep the list together
    assert "2 o 3" in result, \
        f"Number list with 'o' broken. Result: {result}"
    assert "o." not in result or "o. 3" not in result, \
        f"Incorrectly split at 'o'. Result: {result}"
    
    print(f"✓ Number list with 'o' preserved: {result}")


def test_year_list_with_y():
    """Test that lists of years are also preserved."""
    text = "Los años 2015 2016 2017 y 2018 fueron importantes"
    result = restore_punctuation(text, language='es')
    
    # Should keep the year list together
    assert "2017 y 2018" in result, \
        f"Year list broken. Result: {result}"
    
    print(f"✓ Year list preserved: {result}")


def test_mixed_content_after_number_list():
    """Test that we still split correctly when there's genuinely new content."""
    text = "Ve a los episodios 1 2 y 3 Luego continúa con el 4"
    result = restore_punctuation(text, language='es')
    
    # This SHOULD split because "Luego" starts a new sentence
    # But the number list "1 2 y 3" should stay together
    assert "1, 2 y 3" in result or "1 2 y 3" in result, \
        f"Number list broken. Result: {result}"
    
    print(f"✓ Mixed content handled: {result}")


def test_sentence_assembly_number_list():
    """Test that sentence assembly doesn't split number lists."""
    # Start with restored text that has proper punctuation
    restored = "Pero si tú quieres escuchar los episodios anteriores, puedes ir al episodio 147, 151, 156, 164, 170, 177 y 184."
    
    # Use the sentence assembly function that splits into sentences
    sentences, trailing = assemble_sentences_from_processed(restored, 'es')
    
    # All sentences combined (plus trailing if any)
    all_sents = sentences + ([trailing] if trailing else [])
    
    # Check if 184 ended up as a standalone sentence
    standalone_184 = any(s.strip() in ["184.", "184"] for s in all_sents)
    
    if standalone_184:
        print(f"❌ BUG REPRODUCED! Sentences split incorrectly:")
        for i, s in enumerate(all_sents, 1):
            print(f"  {i}. {s}")
        assert False, "Number list incorrectly split - 184 is standalone"
    
    # The number list should be in a single sentence
    full_list_found = any("177 y 184" in s for s in all_sents)
    assert full_list_found, \
        f"Number list broken apart. Sentences: {all_sents}"
    
    print(f"✓ Sentence assembly preserved number list correctly")
    print(f"  Sentences: {sentences}")
    if trailing:
        print(f"  Trailing: {trailing}")


if __name__ == "__main__":
    print("Testing Spanish number list splitting...\n")
    
    test_number_list_with_y_not_split()
    test_simple_number_list_with_y()
    test_number_list_with_o()
    test_year_list_with_y()
    test_mixed_content_after_number_list()
    test_sentence_assembly_number_list()
    
    print("\n✅ All tests passed!")

