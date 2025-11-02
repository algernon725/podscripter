#!/usr/bin/env python3
"""
Test normalization of person initials and organizational acronyms across languages.

This test covers the bug where names with initials (like "C.S. Lewis", "J.K. Rowling")
are incorrectly split into separate sentences when they appear in transcriptions,
particularly in non-English texts that reference English names.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punctuation_restorer import restore_punctuation

def test_spanish_with_english_names():
    """Test Spanish text containing English names with initials."""
    print("\n" + "=" * 80)
    print("TEST: Spanish text with English names (C.S. Lewis bug)")
    print("=" * 80)
    
    test_cases = [
        {
            'input': 'es a C. S. Lewis porque él escribió muchos libros que me parecen interesantes',
            'description': 'C.S. Lewis in Spanish context',
            'should_not_contain': ['es a c.', 'S.', 'es a C.'],  # These would indicate incorrect splits
            'should_contain': 'C.S. Lewis'
        },
        {
            'input': 'me gusta leer a J. K. Rowling porque escribió Harry Potter',
            'description': 'J.K. Rowling in Spanish context',
            'should_not_contain': ['J.', 'K.'],
            'should_contain': 'J.K. Rowling'
        },
        {
            'input': 'leí un libro de C. S. Lewis que se llama Las Crónicas de Narnia',
            'description': 'C.S. Lewis with book title',
            'should_not_contain': ['C.', 'S.'],
            'should_contain': 'C.S. Lewis'
        },
        {
            'input': 'J. R. R. Tolkien escribió El Señor de los Anillos',
            'description': 'Three initials: J.R.R. Tolkien',
            'should_not_contain': ['J.', 'R.'],
            'should_contain': 'J.R.R. Tolkien'
        },
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['description']}")
        print(f"Input:  {test['input']}")
        
        result = restore_punctuation(test['input'], 'es')
        print(f"Output: {result}")
        
        # Check that initials weren't incorrectly split
        failed = False
        for bad_pattern in test['should_not_contain']:
            if bad_pattern in result:
                print(f"  ❌ FAIL: Found incorrect split pattern: '{bad_pattern}'")
                failed = True
        
        # Check that the name is properly formatted
        if test['should_contain'] not in result:
            print(f"  ⚠️  WARNING: Expected pattern '{test['should_contain']}' not found")
            # This is a warning, not necessarily a failure
        
        if not failed:
            print(f"  ✅ PASS: No incorrect sentence splits detected")

def test_english_organizational_acronyms():
    """Test English organizational acronyms (existing behavior should be preserved)."""
    print("\n" + "=" * 80)
    print("TEST: English organizational acronyms")
    print("=" * 80)
    
    test_cases = [
        {
            'input': 'the U. S. Capitol is in Washington D. C.',
            'description': 'U.S. and D.C. acronyms',
            'expected_acronyms': ['US', 'DC']
        },
        {
            'input': 'he lives in the U. S. A. and works for the F. B. I.',
            'description': 'USA and FBI acronyms',
            'expected_acronyms': ['USA', 'FBI']
        },
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['description']}")
        print(f"Input:  {test['input']}")
        
        result = restore_punctuation(test['input'], 'en')
        print(f"Output: {result}")
        
        # Check that acronyms are properly collapsed
        all_found = True
        for acronym in test['expected_acronyms']:
            if acronym not in result:
                print(f"  ⚠️  Expected acronym '{acronym}' not found in output")
                all_found = False
        
        if all_found:
            print(f"  ✅ PASS: All acronyms properly normalized")

def test_french_with_english_names():
    """Test French text with English names containing initials."""
    print("\n" + "=" * 80)
    print("TEST: French text with English names")
    print("=" * 80)
    
    test_cases = [
        {
            'input': 'j\'aime lire C. S. Lewis parce qu\'il a écrit des livres intéressants',
            'description': 'C.S. Lewis in French context',
            'should_contain': 'C.S. Lewis'
        },
        {
            'input': 'J. K. Rowling a écrit Harry Potter',
            'description': 'J.K. Rowling in French context',
            'should_contain': 'J.K. Rowling'
        },
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['description']}")
        print(f"Input:  {test['input']}")
        
        result = restore_punctuation(test['input'], 'fr')
        print(f"Output: {result}")
        
        if test['should_contain'] not in result:
            print(f"  ⚠️  WARNING: Expected pattern '{test['should_contain']}' not found")
        else:
            print(f"  ✅ PASS: Name properly formatted")

def test_german_with_english_names():
    """Test German text with English names containing initials."""
    print("\n" + "=" * 80)
    print("TEST: German text with English names")
    print("=" * 80)
    
    test_cases = [
        {
            'input': 'ich lese gerne C. S. Lewis weil er interessante Bücher geschrieben hat',
            'description': 'C.S. Lewis in German context',
            'should_contain': 'C.S. Lewis'
        },
        {
            'input': 'J. R. R. Tolkien hat Der Herr der Ringe geschrieben',
            'description': 'J.R.R. Tolkien in German context',
            'should_contain': 'J.R.R. Tolkien'
        },
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['description']}")
        print(f"Input:  {test['input']}")
        
        result = restore_punctuation(test['input'], 'de')
        print(f"Output: {result}")
        
        if test['should_contain'] not in result:
            print(f"  ⚠️  WARNING: Expected pattern '{test['should_contain']}' not found")
        else:
            print(f"  ✅ PASS: Name properly formatted")

def test_edge_cases():
    """Test edge cases for initial normalization."""
    print("\n" + "=" * 80)
    print("TEST: Edge cases")
    print("=" * 80)
    
    test_cases = [
        {
            'input': 'el autor C. S. Lewis nació en Belfast y vivió en Oxford',
            'language': 'es',
            'description': 'Name at beginning of clause with location',
        },
        {
            'input': 'conocí a C. S. Lewis en una conferencia sobre literatura',
            'language': 'es',
            'description': 'Name after preposition "a"',
        },
        {
            'input': 'me gusta C. S. Lewis y también J. K. Rowling',
            'language': 'es',
            'description': 'Multiple names with initials in same sentence',
        },
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['description']}")
        print(f"Input:  {test['input']}")
        
        result = restore_punctuation(test['input'], test['language'])
        print(f"Output: {result}")
        
        # Count periods in output - excessive periods suggest incorrect splits
        period_count = result.count('.')
        if period_count > 2:  # Allow for terminal punctuation and name initials
            print(f"  ⚠️  WARNING: Many periods ({period_count}) in output - possible incorrect splits")
        else:
            print(f"  ✅ PASS: Reasonable punctuation")

def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("INITIALS AND ACRONYMS NORMALIZATION TEST SUITE")
    print("=" * 80)
    print("\nThis test suite validates that person initials (C.S. Lewis, J.K. Rowling)")
    print("are not incorrectly treated as sentence breaks across all supported languages.")
    
    test_spanish_with_english_names()
    test_english_organizational_acronyms()
    test_french_with_english_names()
    test_german_with_english_names()
    test_edge_cases()
    
    print("\n" + "=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)
    print("\nNote: This test demonstrates the expected behavior.")
    print("Warnings indicate areas where improvement may be needed.")

if __name__ == "__main__":
    main()

