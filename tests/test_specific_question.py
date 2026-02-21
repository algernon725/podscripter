"""
Test to debug the specific case "Pudiste mantener una conversación?"
"""

import pytest

from conftest import restore_punctuation
from punctuation_restorer import has_question_indicators, is_question_semantic
from sentence_transformers import SentenceTransformer

pytestmark = pytest.mark.core


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
def test_specific_question():
    """Test the specific case that's still failing."""
    test_text = "Pudiste mantener una conversación"

    pattern_result = has_question_indicators(test_text, 'es')
    assert pattern_result, (
        f"Pattern detection should flag {test_text!r} as a question"
    )

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    semantic_result = is_question_semantic(test_text, model, 'es')
    assert semantic_result, (
        f"Semantic detection should flag {test_text!r} as a question"
    )

    full_result = restore_punctuation(test_text, 'es')
    assert full_result.endswith('?'), (
        f"Expected question mark at end: {full_result!r}"
    )
    assert full_result.startswith('¿'), (
        f"Expected inverted question mark at start: {full_result!r}"
    )
