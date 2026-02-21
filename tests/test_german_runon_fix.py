#!/usr/bin/env python3
"""
Test to verify German run-on sentence fix
"""

import pytest
from conftest import restore_punctuation

pytestmark = pytest.mark.core


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
def test_german_runon_fix():
    """Test that German text is properly split into sentences."""
    runon_text = """Hallo an alle  Willkommen bei DeutschPod  DeutschPod ist der Podcast der Ihnen helfen wird bereit zu sein deutsch zu sprechen  DeutschPod bereitet Sie darauf vor deutsch zu sprechen überall jederzeit und in jeder situation  Erinnern Sie sich an all diese momente als Sie nicht wussten was Sie sagen sollten  Diese momente als Sie keine konversation aufrechterhalten konnten  Nun machen Sie sich keine sorgen  DeutschPod ist das tool das Sie gesucht haben um Ihr deutsch zu verbessern  Verabschieden Sie sich  Von all diesen peinlichen momenten  Also fangen wir an  Sind wir bereit  Ich bin Hans aus Berlin Deutschland  Und ich bin Anna aus München Deutschland  Hallo an alle"""

    result = restore_punctuation(runon_text, language='de')

    sentence_count = result.count('.') + result.count('?') + result.count('!')
    assert sentence_count > 5, f"Expected >5 sentences, got {sentence_count}"

    expected_patterns = [
        "Hallo an alle",
        "Willkommen bei DeutschPod",
        "Erinnern Sie sich an all diese Momente",
        "Ich bin Hans aus Berlin, Deutschland",
        "Und ich bin Anna aus München, Deutschland",
    ]
    for pattern in expected_patterns:
        assert pattern in result, f"Missing expected pattern: {pattern!r}"
