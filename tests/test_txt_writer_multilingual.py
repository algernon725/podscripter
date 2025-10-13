#!/usr/bin/env python3
"""
Test that the TXT writer fix works for all languages
when the sentences are already correctly punctuated.
"""

import tempfile
import os


def read_txt_output(filepath):
    """Read a TXT file and return list of non-empty paragraphs."""
    with open(filepath, 'r') as f:
        content = f.read()
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    return paragraphs


def test_txt_writer_with_correct_punctuation():
    """Test TXT writer doesn't split correctly-punctuated number lists."""
    from podscripter import _write_txt
    
    test_cases = [
        {
            'language': 'es',
            'sentence': "Pero si tú quieres escuchar los episodios anteriores, puedes ir al episodio 147,151,156,164,170,177 y 184. El episodio más reciente fue el 184.",
            'should_have': '177 y 184',
            'name': 'Spanish'
        },
        {
            'language': 'en',
            'sentence': "But if you want to listen to the previous episodes, you can go to episode 147,151,156,164,170,177 and 184. The most recent episode was episode 184.",
            'should_have': '177 and 184',
            'name': 'English'
        },
        {
            'language': 'fr',
            'sentence': "Mais si tu veux écouter les épisodes précédents, tu peux aller à l'épisode 147,151,156,164,170,177 et 184. L'épisode le plus récent était l'épisode 184.",
            'should_have': '177 et 184',
            'name': 'French'
        },
        {
            'language': 'de',
            'sentence': "Aber wenn du die vorherigen Episoden hören möchtest, kannst du zu Episode 147,151,156,164,170,177 und 184 gehen. Die neueste Episode war Episode 184.",
            'should_have': '177 und 184',
            'name': 'German'
        },
    ]
    
    print("="*70)
    print("Testing TXT Writer with Correctly Punctuated Number Lists")
    print("="*70)
    print()
    
    all_passed = True
    
    for case in test_cases:
        print(f"{case['name']}:")
        print(f"  Input: {case['sentence'][:80]}...")
        
        # Write using TXT writer (as a single sentence)
        sentences = [case['sentence']]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_file = f.name
        
        try:
            _write_txt(sentences, temp_file, language=case['language'])
            paragraphs = read_txt_output(temp_file)
            
            # Check if the number list stayed together
            full_text = ' '.join(paragraphs)
            has_list = case['should_have'] in full_text
            
            # Check if "184." is standalone
            standalone_184 = any(p.strip() in ["184.", "184"] for p in paragraphs)
            
            if has_list and not standalone_184:
                print(f"  ✅ TXT writer preserved number list correctly")
            else:
                print(f"  ❌ TXT writer split the list")
                print(f"     Has '{case['should_have']}': {has_list}")
                print(f"     Standalone '184.': {standalone_184}")
                print(f"     Paragraphs: {paragraphs}")
                all_passed = False
                
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        print()
    
    return all_passed


def test_simple_number_lists_all_languages():
    """Test simple number lists with 'and/y/et/und'."""
    from podscripter import _write_txt
    
    test_cases = [
        ('es', 'Los episodios son 1, 2, 3 y 4. Luego continúa.', '3 y 4'),
        ('en', 'The episodes are 1, 2, 3 and 4. Then continue.', '3 and 4'),
        ('fr', 'Les épisodes sont 1, 2, 3 et 4. Ensuite continue.', '3 et 4'),
        ('de', 'Die Episoden sind 1, 2, 3 und 4. Dann fortfahren.', '3 und 4'),
    ]
    
    print("="*70)
    print("Testing Simple Number Lists - All Languages")
    print("="*70)
    print()
    
    all_passed = True
    
    for lang, sentence, pattern in test_cases:
        sentences = [sentence]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_file = f.name
        
        try:
            _write_txt(sentences, temp_file, language=lang)
            paragraphs = read_txt_output(temp_file)
            
            full_text = ' '.join(paragraphs)
            has_pattern = pattern in full_text
            not_split = len(paragraphs) == 1 or (len(paragraphs) == 2 and '4.' not in paragraphs[1])
            
            if has_pattern and not_split:
                print(f"  ✅ {lang.upper()}: '{pattern}' preserved")
            else:
                print(f"  ❌ {lang.upper()}: Failed")
                print(f"     Paragraphs: {paragraphs}")
                all_passed = False
                
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    print()
    return all_passed


if __name__ == "__main__":
    result1 = test_txt_writer_with_correct_punctuation()
    result2 = test_simple_number_lists_all_languages()
    
    print("="*70)
    if result1 and result2:
        print("🎉 TXT writer handles number lists correctly in all languages!")
        print()
        print("Note: English and French still have a separate bug in punctuation")
        print("restoration that inserts periods before standalone numbers like '184'.")
        print("That's a different issue from the TXT writer splitting bug.")
    else:
        print("❌ Some tests failed")
    print("="*70)

