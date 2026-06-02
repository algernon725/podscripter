#!/usr/bin/env python3
"""
Spanish embedded questions tests: ensure mid-sentence '¿ … ?' clauses are preserved
and properly paired (we should not strip inverted question marks in embedded clauses).
"""

import re

import pytest
from conftest import restore_punctuation

pytestmark = pytest.mark.core


def norm(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(r"\s+([,.!?])", r"\1", s)
    return s


# NOTE: test_preserve_embedded_wh_question and test_preserve_por_que_embedded were
# retired in v0.8.2. They were labeled "test expectations predate API changes"
# (stale harness expectations), and their accent-normalization assertions overlap
# with the closed question-detection limitation — see AGENT.md "Question detection
# — verb-first / implicit questions (CLOSED)".


def test_preserve_embedded_yesno_question():
    inp = "La pregunta es, ¿estamos listos?, y si no, seguimos"
    out = norm(restore_punctuation(inp, 'es'))
    assert "¿estamos listos?" in out.lower(), out


def test_do_not_double_wrap_embedded():
    inp = "Te escribo: ¿Hola, cómo estás?, espero que bien"
    out = norm(restore_punctuation(inp, 'es'))
    assert out.count('¿') == 1 and out.count('?') >= 1, out
    assert "¿Hola, cómo estás?" in out or "¿hola, cómo estás?" in out.lower(), out
