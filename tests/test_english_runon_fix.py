#!/usr/bin/env python3
"""
Test to verify English run-on sentence fix
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from punctuation_restorer import restore_punctuation

def test_english_runon_fix():
    """Test that English text is properly split into sentences."""
    
    print("Testing English Run-on Sentence Fix")
    print("=" * 60)
    
    # The problematic run-on text in English
    runon_text = """Hello everyone  Welcome to EnglishPod  EnglishPod is the podcast that will help you get ready to speak English  EnglishPod prepares you to speak English anywhere anytime and in any situation  Do you remember all those moments when you didn't know what to say  Those moments when you couldn't maintain a conversation  Well don't worry  EnglishPod is the tool you were looking for to improve your English  Say goodbye  To all those awkward moments  So let's get started  Are we ready  I am John from New York City  And I am Sarah from London England  Hello everyone"""
    
    print("Input (run-on text):")
    print(repr(runon_text))
    print()
    
    # Process the text
    result = restore_punctuation(runon_text, language='en')
    
    print("Output (processed text):")
    print(result)
    print()
    
    # Count sentences (rough estimate by counting periods, question marks, exclamation marks)
    sentence_count = result.count('.') + result.count('?') + result.count('!')
    
    print(f"Number of sentences detected: {sentence_count}")
    print()
    
    # Check if the text is properly split
    if sentence_count > 5:  # Should have multiple sentences
        print("✅ SUCCESS: Text is properly split into multiple sentences")
    else:
        print("❌ FAILURE: Text is still a run-on sentence")
    
    # Check for specific expected patterns
    expected_patterns = [
        "Hello everyone",
        "Welcome to EnglishPod",
        "Do you remember all those moments",
        "I am John from New York City",
        "And I am Sarah from London England"
    ]
    
    print("\nChecking for expected sentence patterns:")
    for pattern in expected_patterns:
        if pattern in result:
            print(f"✅ Found: {pattern}")
        else:
            print(f"❌ Missing: {pattern}")

if __name__ == "__main__":
    test_english_runon_fix()
