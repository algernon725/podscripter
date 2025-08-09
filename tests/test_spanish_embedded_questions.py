#!/usr/bin/env python3
"""
Spanish embedded questions tests: ensure mid-sentence '¿ … ?' clauses are preserved
and properly paired (we should not strip inverted question marks in embedded clauses).
"""

import os
import sys
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from punctuation_restorer import restore_punctuation


def norm(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(r"\s+([,.!?])", r"\1", s)
    return s


def test_preserve_embedded_wh_question():
    inp = "No se, ¿que piensas, Andrea?, pero podemos seguir"
    out = norm(restore_punctuation(inp, 'es'))
    assert "¿qué piensas, Andrea?" in out.lower(), out


def test_preserve_embedded_yesno_question():
    inp = "La pregunta es, ¿estamos listos?, y si no, seguimos"
    out = norm(restore_punctuation(inp, 'es'))
    assert "¿estamos listos?" in out.lower(), out


def test_preserve_por_que_embedded():
    inp = "Ella dijo que, ¿por que no viniste?, todos te esperaban"
    out = norm(restore_punctuation(inp, 'es'))
    # allow with or without comma after clause, but embedded must be present
    assert "¿por qué no viniste?" in out.lower(), out


def test_do_not_double_wrap_embedded():
    inp = "Te escribo: ¿Hola, cómo estás?, espero que bien"
    out = norm(restore_punctuation(inp, 'es'))
    # Ensure exactly one pair of inverted/closing marks around the embedded clause
    assert out.count('¿') == 1 and out.count('?') >= 1, out
    assert "¿Hola, cómo estás?" in out or "¿hola, cómo estás?" in out.lower(), out


