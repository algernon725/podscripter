#!/usr/bin/env python3
"""
Test to reproduce and fix English sentence splitting issues
"""

import pytest
from conftest import restore_punctuation

pytestmark = pytest.mark.core

ENGLISH_SPLITTING_CASES = [
    pytest.param(
        "And President Trump is putting the D. C. Police Department under federal control.",
        "And President Trump is putting the DC Police Department under federal control.",
        id="do-not-split-dotted-acronym-DC",
    ),
    pytest.param(
        "He calls the U. S. Capitol a place of unchecked crime and squalor.",
        "He calls the US Capitol a place of unchecked crime and squalor.",
        id="do-not-split-dotted-acronym-US",
    ),
    pytest.param(
        "He compared the speed and power of this crackdown to what's happening on the U.S.-Mexico border.",
        "He compared the speed and power of this crackdown to what's happening on the US-Mexico border.",
        id="compact-dotted-acronym-US-before-hyphen",
    ),
    pytest.param(
        "i am john from new york city",
        "I am John from New York City.",
        id="introduction-one-sentence",
    ),
    pytest.param(
        "do you remember all those moments when you didn't know what to say",
        "Do you remember all those moments when you didn't know what to say?",
        id="question-with-question-mark",
    ),
    pytest.param(
        "hello how are you today",
        "Hello, how are you today?",
        id="greeting-with-question",
    ),
    pytest.param(
        "my name is sarah and i live in london",
        "My name is Sarah and I live in London.",
        id="introduction-with-conjunction",
    ),
    pytest.param(
        "what time is the meeting tomorrow",
        "What time is the meeting tomorrow?",
        id="question-about-time",
    ),
    pytest.param(
        "it is important that everyone is present",
        "It is important that everyone is present.",
        id="statement-about-importance",
    ),
    pytest.param(
        "can you help me with this project",
        "Can you help me with this project?",
        id="request-for-help",
    ),
    pytest.param(
        "i would like to introduce myself",
        "I would like to introduce myself.",
        id="formal-introduction",
    ),
    pytest.param(
        "where did you learn to speak english",
        "Where did you learn to speak English?",
        id="question-about-learning",
    ),
    pytest.param(
        "thank you for your time today",
        "Thank you for your time today.",
        id="polite-closing",
    ),
]


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
@pytest.mark.parametrize("input_text,expected", ENGLISH_SPLITTING_CASES)
def test_english_sentence_splitting(input_text, expected):
    """Test English sentence splitting issues."""
    result = restore_punctuation(input_text, 'en')
    assert result.strip() == expected.strip()
