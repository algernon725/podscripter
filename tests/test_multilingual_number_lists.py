#!/usr/bin/env python3
"""
Test if the number list splitting bug affects English and French.
"""

import tempfile
import os
from punctuation_restorer import restore_punctuation, assemble_sentences_from_processed


def read_txt_output(filepath):
    """Read a TXT file and return list of non-empty paragraphs."""
    with open(filepath, 'r') as f:
        content = f.read()
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    return paragraphs


def test_english_number_list():
    """Test if English number lists are also affected."""
    text = ("But if you want to listen to the previous episodes, "
            "you can go to episode 147, 151, 156, 164, 170, 177 and 184. "
            "The most recent episode in this series was episode 184.")
    
    print("="*70)
    print("Testing English Number List")
    print("="*70)
    print(f"Input: {text}")
    print()
    
    # Process through pipeline
    restored = restore_punctuation(text, language='en')
    print(f"After restoration: {restored}")
    print()
    
    sentences, trailing = assemble_sentences_from_processed(restored, 'en')
    if trailing:
        sentences.append(trailing)
    
    print(f"Sentences ({len(sentences)}):")
    for i, s in enumerate(sentences, 1):
        print(f"  {i}. {s}")
    print()
    
    # Write to TXT
    from podscripter import _write_txt
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_file = f.name
    
    try:
        _write_txt(sentences, temp_file, language='en')
        paragraphs = read_txt_output(temp_file)
        
        print(f"TXT output ({len(paragraphs)} paragraphs):")
        for i, p in enumerate(paragraphs, 1):
            print(f"  {i}. {p}")
        print()
        
        # Check for the bug
        standalone_184 = any(p.strip() in ["184.", "184"] for p in paragraphs)
        and_ending = any(p.strip().endswith("and.") for p in paragraphs)
        number_list_intact = any("177 and 184" in p for p in paragraphs)
        
        print("Validation:")
        print(f"  - Standalone '184.': {standalone_184}")
        print(f"  - Paragraph ending 'and.': {and_ending}")
        print(f"  - Number list intact '177 and 184': {number_list_intact}")
        print()
        
        if standalone_184 or and_ending:
            print("❌ BUG FOUND in English!")
            return False
        elif number_list_intact:
            print("✅ English number list preserved correctly")
            return True
        else:
            print("⚠️  Unexpected behavior")
            return False
            
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_french_number_list():
    """Test if French number lists are also affected."""
    text = ("Mais si tu veux écouter les épisodes précédents, "
            "tu peux aller à l'épisode 147, 151, 156, 164, 170, 177 et 184. "
            "L'épisode le plus récent de cette série était l'épisode 184.")
    
    print("="*70)
    print("Testing French Number List")
    print("="*70)
    print(f"Input: {text}")
    print()
    
    # Process through pipeline
    restored = restore_punctuation(text, language='fr')
    print(f"After restoration: {restored}")
    print()
    
    sentences, trailing = assemble_sentences_from_processed(restored, 'fr')
    if trailing:
        sentences.append(trailing)
    
    print(f"Sentences ({len(sentences)}):")
    for i, s in enumerate(sentences, 1):
        print(f"  {i}. {s}")
    print()
    
    # Write to TXT
    from podscripter import _write_txt
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_file = f.name
    
    try:
        _write_txt(sentences, temp_file, language='fr')
        paragraphs = read_txt_output(temp_file)
        
        print(f"TXT output ({len(paragraphs)} paragraphs):")
        for i, p in enumerate(paragraphs, 1):
            print(f"  {i}. {p}")
        print()
        
        # Check for the bug
        standalone_184 = any(p.strip() in ["184.", "184"] for p in paragraphs)
        et_ending = any(p.strip().endswith("et.") for p in paragraphs)
        number_list_intact = any("177 et 184" in p for p in paragraphs)
        
        print("Validation:")
        print(f"  - Standalone '184.': {standalone_184}")
        print(f"  - Paragraph ending 'et.': {et_ending}")
        print(f"  - Number list intact '177 et 184': {number_list_intact}")
        print()
        
        if standalone_184 or et_ending:
            print("❌ BUG FOUND in French!")
            return False
        elif number_list_intact:
            print("✅ French number list preserved correctly")
            return True
        else:
            print("⚠️  Unexpected behavior")
            return False
            
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_german_number_list():
    """Test if German number lists are also affected."""
    text = ("Aber wenn du die vorherigen Episoden hören möchtest, "
            "kannst du zu Episode 147, 151, 156, 164, 170, 177 und 184 gehen. "
            "Die neueste Episode in dieser Serie war Episode 184.")
    
    print("="*70)
    print("Testing German Number List")
    print("="*70)
    print(f"Input: {text}")
    print()
    
    # Process through pipeline
    restored = restore_punctuation(text, language='de')
    print(f"After restoration: {restored}")
    print()
    
    sentences, trailing = assemble_sentences_from_processed(restored, 'de')
    if trailing:
        sentences.append(trailing)
    
    print(f"Sentences ({len(sentences)}):")
    for i, s in enumerate(sentences, 1):
        print(f"  {i}. {s}")
    print()
    
    # Write to TXT
    from podscripter import _write_txt
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_file = f.name
    
    try:
        _write_txt(sentences, temp_file, language='de')
        paragraphs = read_txt_output(temp_file)
        
        print(f"TXT output ({len(paragraphs)} paragraphs):")
        for i, p in enumerate(paragraphs, 1):
            print(f"  {i}. {p}")
        print()
        
        # Check for the bug
        standalone_184 = any(p.strip() in ["184.", "184"] for p in paragraphs)
        und_ending = any(p.strip().endswith("und.") for p in paragraphs)
        number_list_intact = any("177 und 184" in p for p in paragraphs)
        
        print("Validation:")
        print(f"  - Standalone '184.': {standalone_184}")
        print(f"  - Paragraph ending 'und.': {und_ending}")
        print(f"  - Number list intact '177 und 184': {number_list_intact}")
        print()
        
        if standalone_184 or und_ending:
            print("❌ BUG FOUND in German!")
            return False
        elif number_list_intact:
            print("✅ German number list preserved correctly")
            return True
        else:
            print("⚠️  Unexpected behavior")
            return False
            
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


if __name__ == "__main__":
    results = []
    
    results.append(test_english_number_list())
    print()
    results.append(test_french_number_list())
    print()
    results.append(test_german_number_list())
    print()
    
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"English: {'✅ Fixed' if results[0] else '❌ Bug exists'}")
    print(f"French:  {'✅ Fixed' if results[1] else '❌ Bug exists'}")
    print(f"German:  {'✅ Fixed' if results[2] else '❌ Bug exists'}")
    print()
    
    if all(results):
        print("🎉 All languages handled correctly!")
    else:
        print("⚠️  Bug exists in some languages - need to extend the fix")

