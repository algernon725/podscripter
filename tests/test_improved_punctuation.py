"""
Test script for improved punctuation restoration functionality
"""

import pytest

from sentence_transformers import SentenceTransformer
from conftest import restore_punctuation

pytestmark = pytest.mark.core


@pytest.mark.parametrize("lang,input_text,description", [
    (
        'en',
        "hello how are you today I hope you are doing well thank you for asking about my day it was quite busy but productive I managed to finish all my tasks and even had time for a coffee break",
        'English long conversation - should create meaningful sentences',
    ),
    (
        'en',
        "what time is the meeting tomorrow I need to prepare my presentation and also check if everyone received the agenda can you confirm the location",
        'English mixed questions and statements',
    ),
    (
        'en',
        "I went to the store and bought some groceries then I came home and started cooking dinner because I was hungry and wanted to eat something healthy",
        'English sentences with conjunctions - should keep related clauses together',
    ),
    (
        'es',
        "hola como estas hoy espero que estes bien gracias por preguntar sobre mi dia fue bastante ocupado pero productivo logre terminar todas mis tareas",
        'Spanish conversation - should create meaningful sentences',
    ),
    (
        'de',
        "hallo wie geht es dir heute ich hoffe es geht dir gut danke fur das fragen uber meinen tag es war ziemlich beschaftigt aber produktiv",
        'German conversation - should create meaningful sentences',
    ),
    (
        'fr',
        "bonjour comment allez vous aujourd'hui j'espere que vous allez bien merci de demander de ma journee elle etait assez occupee mais productive",
        'French conversation - should create meaningful sentences',
    ),
])
def test_improved_punctuation(lang, input_text, description):
    """Test that improved punctuation restoration creates properly punctuated output."""
    result = restore_punctuation(input_text, lang)
    assert result and result.strip(), f"[{description}] Empty result for input: {input_text}"
    assert result.strip()[-1] in '.!?', \
        f"[{description}] Result doesn't end with punctuation: {result}"
    sentences = [s.strip() for s in result.split('.') if s.strip()]
    assert len(sentences) >= 1, f"[{description}] No sentences produced: {result}"


@pytest.mark.parametrize("input_text,description", [
    (
        "I went to the store and bought milk bread and eggs then I came home and started cooking dinner because I was hungry",
        'Should keep related clauses together, not split at every "and"',
    ),
    (
        "what time is the meeting can you send me the agenda I need to prepare my presentation",
        'Should detect questions and separate them properly',
    ),
    (
        "thank you for your help that was amazing I really appreciate it",
        'Should detect exclamations and gratitude expressions',
    ),
    (
        "the weather is nice today however it might rain later so we should bring umbrellas just in case",
        'Should handle transitional words like "however" properly',
    ),
])
def test_specific_improvements(input_text, description):
    """Test specific improvements in sentence boundary detection."""
    result = restore_punctuation(input_text, 'en')
    assert result and result.strip(), f"[{description}] Empty result for input: {input_text}"
    assert result.strip()[-1] in '.!?', \
        f"[{description}] Result doesn't end with punctuation: {result}"
