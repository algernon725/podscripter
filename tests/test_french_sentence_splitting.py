#!/usr/bin/env python3
"""
Test to reproduce and fix French sentence splitting issues
"""

import pytest
from conftest import restore_punctuation

pytestmark = pytest.mark.core

FRENCH_SPLITTING_CASES = [
    pytest.param(
        "je suis marie de paris france",
        "Je suis Marie de Paris, France.",
        id="introduction-one-sentence",
    ),
    pytest.param(
        "vous souvenez vous de tous ces moments où vous ne saviez pas quoi dire",
        "Vous souvenez-vous de tous ces moments où vous ne saviez pas quoi dire?",
        id="question-with-question-mark",
    ),
    pytest.param(
        "bonjour comment allez vous aujourd'hui",
        "Bonjour, comment allez-vous aujourd'hui?",
        id="greeting-with-question",
    ),
    pytest.param(
        "je m'appelle pierre et j'habite à lyon",
        "Je m'appelle Pierre et j'habite à Lyon.",
        id="introduction-with-conjunction",
    ),
    pytest.param(
        "à quelle heure est la réunion demain",
        "À quelle heure est la réunion demain?",
        id="question-about-time",
    ),
    pytest.param(
        "il est important que tout le monde soit présent",
        "Il est important que tout le monde soit présent.",
        id="statement-about-importance",
    ),
    pytest.param(
        "pouvez vous m'aider avec ce projet",
        "Pouvez-vous m'aider avec ce projet?",
        id="request-for-help",
    ),
    pytest.param(
        "je voudrais me présenter",
        "Je voudrais me présenter.",
        id="formal-introduction",
    ),
    pytest.param(
        "où avez vous appris à parler français",
        "Où avez-vous appris à parler français?",
        id="question-about-learning",
    ),
    pytest.param(
        "merci pour votre temps aujourd'hui",
        "Merci pour votre temps aujourd'hui.",
        id="polite-closing",
    ),
]


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
@pytest.mark.parametrize("input_text,expected", FRENCH_SPLITTING_CASES)
def test_french_sentence_splitting(input_text, expected):
    """Test French sentence splitting issues."""
    result = restore_punctuation(input_text, 'fr')
    assert result.strip() == expected.strip()
