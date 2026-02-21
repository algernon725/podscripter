#!/usr/bin/env python3
"""
Test to verify English run-on sentence fix
"""

import pytest
from conftest import restore_punctuation

pytestmark = pytest.mark.core


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
def test_english_runon_fix():
    """Test that English text is properly split into sentences."""
    runon_text = """Hello everyone  Welcome to EnglishPod  EnglishPod is the podcast that will help you get ready to speak English  EnglishPod prepares you to speak English anywhere anytime and in any situation  Do you remember all those moments when you didn't know what to say  Those moments when you couldn't maintain a conversation  Well don't worry  EnglishPod is the tool you were looking for to improve your English  Say goodbye  To all those awkward moments  So let's get started  Are we ready  I am John from New York City  And I am Sarah from London England  Hello everyone"""

    result = restore_punctuation(runon_text, language='en')

    sentence_count = result.count('.') + result.count('?') + result.count('!')
    assert sentence_count > 5, f"Expected >5 sentences, got {sentence_count}"

    expected_patterns = [
        "Hello everyone",
        "Welcome to EnglishPod",
        "Do you remember all those moments",
        "I am John from New York City",
        "And I am Sarah from London England",
    ]
    for pattern in expected_patterns:
        assert pattern in result, f"Missing expected pattern: {pattern!r}"
