#!/usr/bin/env python3
"""
Comprehensive test for the Episodio 190 number list splitting bug.
Tests the complete pipeline from punctuation restoration to TXT writing.
"""

import tempfile
import os
from punctuation_restorer import restore_punctuation, assemble_sentences_from_processed


def read_txt_output(filepath):
    """Read a TXT file and return list of non-empty paragraphs."""
    with open(filepath, 'r') as f:
        content = f.read()
    # Split on double newlines (paragraph breaks)
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    return paragraphs


def test_full_pipeline_episodio190():
    """Test the complete pipeline with the exact text from Episodio 190."""
    # Exact text from segments 25-27 of Episodio190_raw.txt
    text = ("Pero si tú quieres escuchar los episodios anteriores, "
            "puedes ir al episodio 147, 151, 156, 164, 170, 177 y 184. "
            "El episodio más reciente sobre esta serie fue el 184 "
            "donde hicimos el capítulo 7 y hoy vamos a hacer el capítulo 8.")
    
    print("="*70)
    print("Testing Episodio 190 Number List - Full Pipeline")
    print("="*70)
    print()
    print(f"Input text:")
    print(f"  {text}")
    print()
    
    # Step 1: Restore punctuation
    restored = restore_punctuation(text, language='es')
    print(f"After punctuation restoration:")
    print(f"  {restored}")
    print()
    
    # Step 2: Assemble sentences
    sentences, trailing = assemble_sentences_from_processed(restored, 'es')
    if trailing:
        sentences.append(trailing)
    
    print(f"After sentence assembly ({len(sentences)} sentences):")
    for i, s in enumerate(sentences, 1):
        print(f"  {i}. {s}")
    print()
    
    # Step 3: Simulate TXT writing (import the actual function)
    from podscripter import _write_txt
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_file = f.name
    
    try:
        _write_txt(sentences, temp_file, language='es')
        
        # Read back the output
        paragraphs = read_txt_output(temp_file)
        
        print(f"After TXT writing ({len(paragraphs)} paragraphs):")
        for i, p in enumerate(paragraphs, 1):
            print(f"  {i}. {p}")
        print()
        
        # Verify the bug is fixed
        # 1. Check that "184." is not standalone
        standalone_184 = any(p.strip() in ["184.", "184"] for p in paragraphs)
        
        # 2. Check that we don't have a paragraph ending with "y."
        y_ending = any(p.strip().endswith("y.") for p in paragraphs)
        
        # 3. Check that "177 y 184" appears together in one paragraph
        number_list_intact = any("177 y 184" in p or "177 y 184" in p for p in paragraphs)
        
        print("Validation:")
        print(f"  - Standalone '184.': {standalone_184} (should be False)")
        print(f"  - Paragraph ending 'y.': {y_ending} (should be False)")
        print(f"  - Number list intact '177 y 184': {number_list_intact} (should be True)")
        print()
        
        if standalone_184:
            print("❌ FAILED: '184' is a standalone paragraph")
            return False
        
        if y_ending:
            print("❌ FAILED: Found paragraph ending with 'y.'")
            return False
        
        if not number_list_intact:
            print("❌ FAILED: Number list '177 y 184' was broken apart")
            return False
        
        print("✅ SUCCESS: Number list preserved correctly through full pipeline!")
        return True
        
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_edge_cases():
    """Test edge cases for number list handling."""
    test_cases = [
        {
            'name': 'Number list with spaces in numbers',
            'text': 'Ve a los episodios 1, 2, 3 y 4. Luego continúa.',
            'should_have': '3 y 4',
            'should_not_have_separate': '4.'
        },
        {
            'name': 'Number list with "o" instead of "y"',
            'text': 'Elige opción 1, 2 o 3. Después decide.',
            'should_have': '2 o 3',
            'should_not_have_separate': '3.'
        },
        {
            'name': 'Year list',
            'text': 'Los años 2015, 2016, 2017 y 2018 fueron importantes. Ahora estamos en 2025.',
            'should_have': '2017 y 2018',
            'should_not_have_separate': '2018.'
        },
    ]
    
    print("="*70)
    print("Testing Edge Cases")
    print("="*70)
    print()
    
    from podscripter import _write_txt
    all_passed = True
    
    for case in test_cases:
        print(f"Test: {case['name']}")
        print(f"  Input: {case['text']}")
        
        # Process through pipeline
        restored = restore_punctuation(case['text'], language='es')
        sentences, trailing = assemble_sentences_from_processed(restored, 'es')
        if trailing:
            sentences.append(trailing)
        
        # Write to TXT
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_file = f.name
        
        try:
            _write_txt(sentences, temp_file, language='es')
            paragraphs = read_txt_output(temp_file)
            
            # Check expectations
            full_text = ' '.join(paragraphs)
            has_expected = case['should_have'] in full_text
            not_separate = not any(p.strip() == case['should_not_have_separate'] for p in paragraphs)
            
            if has_expected and not_separate:
                print(f"  ✅ Passed")
            else:
                print(f"  ❌ Failed")
                print(f"     Has '{case['should_have']}': {has_expected}")
                print(f"     Not separate '{case['should_not_have_separate']}': {not_separate}")
                print(f"     Paragraphs: {paragraphs}")
                all_passed = False
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        print()
    
    return all_passed


if __name__ == "__main__":
    result1 = test_full_pipeline_episodio190()
    print()
    result2 = test_edge_cases()
    
    print("="*70)
    if result1 and result2:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")
    print("="*70)

