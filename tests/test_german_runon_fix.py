#!/usr/bin/env python3
"""
Test to verify German run-on sentence fix
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from punctuation_restorer import restore_punctuation

def test_german_runon_fix():
    """Test that German text is properly split into sentences."""
    
    print("Testing German Run-on Sentence Fix")
    print("=" * 60)
    
    # The problematic run-on text in German
    runon_text = """Hallo an alle  Willkommen bei DeutschPod  DeutschPod ist der Podcast der Ihnen helfen wird bereit zu sein deutsch zu sprechen  DeutschPod bereitet Sie darauf vor deutsch zu sprechen überall jederzeit und in jeder situation  Erinnern Sie sich an all diese momente als Sie nicht wussten was Sie sagen sollten  Diese momente als Sie keine konversation aufrechterhalten konnten  Nun machen Sie sich keine sorgen  DeutschPod ist das tool das Sie gesucht haben um Ihr deutsch zu verbessern  Verabschieden Sie sich  Von all diesen peinlichen momenten  Also fangen wir an  Sind wir bereit  Ich bin Hans aus Berlin Deutschland  Und ich bin Anna aus München Deutschland  Hallo an alle"""
    
    print("Input (run-on text):")
    print(repr(runon_text))
    print()
    
    # Process the text
    result = restore_punctuation(runon_text, language='de')
    
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
        "Hallo an alle",
        "Willkommen bei DeutschPod",
        "Erinnern Sie sich an all diese Momente",
        "Ich bin Hans aus Berlin, Deutschland",
        "Und ich bin Anna aus München, Deutschland"
    ]
    
    print("\nChecking for expected sentence patterns:")
    for pattern in expected_patterns:
        if pattern in result:
            print(f"✅ Found: {pattern}")
        else:
            print(f"❌ Missing: {pattern}")

if __name__ == "__main__":
    test_german_runon_fix()
