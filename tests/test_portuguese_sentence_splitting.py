#!/usr/bin/env python3
"""
Test Portuguese sentence splitting and formatting.

Mirrors the per-language splitting suites for en/fr/de. Cases marked xfail are
limitations shared with the other light-path languages (en/fr/de), not
Portuguese-specific regressions -- see the individual reasons.
"""

import pytest
from conftest import restore_punctuation

pytestmark = pytest.mark.core

PORTUGUESE_SPLITTING_CASES = [
    pytest.param(
        "olá como vai você hoje",
        "Olá, como vai você hoje?",
        id="greeting-with-question",
    ),
    pytest.param(
        "é importante que todos estejam presentes",
        "É importante que todos estejam presentes.",
        id="subordinate-clause-stays-one-sentence",
    ),
    pytest.param(
        "eu acho que ele vem porque ele prometeu",
        "Eu acho que ele vem porque ele prometeu.",
        id="two-subordinate-clauses",
    ),
    pytest.param(
        "eu gostaria de um pouco mais de café",
        "Eu gostaria de um pouco mais de café.",
        id="comparative-particle-not-split",
    ),
    pytest.param(
        "você pode encontrar um bom apartamento aqui",
        "Você pode encontrar um bom apartamento aqui.",
        id="modal-plus-infinitive-not-split",
    ),
    pytest.param(
        "nós fomos dirigidos para a sala errada",
        "Nós fomos dirigidos para a sala errada.",
        id="auxiliary-plus-participle-not-split",
    ),
    pytest.param(
        "ele tem vinte e cinco anos de idade",
        "Ele tem vinte e cinco anos de idade.",
        id="number-plus-unit-not-split",
    ),
    pytest.param(
        "não sei o que dizer sobre isso",
        "Não sei o que dizer sobre isso.",
        id="declarative-gets-period",
    ),
    pytest.param(
        "eu sou o joão de lisboa portugal",
        "Eu sou o João de Lisboa, Portugal.",
        id="location-appositive-comma",
        marks=pytest.mark.xfail(
            reason="Shared light-path gap: the location-comma heuristic needs "
                   "capitalized place names, and nothing capitalizes lowercase input "
                   "(spaCy, removed in v0.15.0, only ever capitalized a discarded "
                   "string). en ('from London England') and fr ('de Paris France') "
                   "behave identically."
        ),
    ),
    pytest.param(
        "a que horas é a reunião amanhã",
        "A que horas é a reunião amanhã?",
        id="question-about-time",
        marks=pytest.mark.xfail(
            reason="Shared light-path gap: question detection matches on the first "
                   "token, so 'a que horas ...' is missed. de and en are xfail/fail "
                   "on the same construction."
        ),
    ),
]


@pytest.mark.parametrize("input_text,expected", PORTUGUESE_SPLITTING_CASES)
def test_portuguese_sentence_splitting(input_text, expected):
    """Test Portuguese sentence splitting and formatting."""
    result = restore_punctuation(input_text, 'pt')
    assert result.strip() == expected.strip()


@pytest.mark.parametrize("text", [
    "eu moro no porto e trabalho com tecnologia",
    "ela foi para a praia com os amigos dela",
    "nós temos que falar sobre o projeto novo",
])
def test_portuguese_no_split_after_function_words(text):
    """Contractions and prepositions must never end a sentence."""
    result = restore_punctuation(text, 'pt')
    forbidden_endings = (
        ' no.', ' na.', ' do.', ' da.', ' dos.', ' das.', ' ao.', ' com.',
        ' para.', ' de.', ' em.', ' os.', ' as.', ' que.',
    )
    for bad in forbidden_endings:
        assert bad not in result, f"Split after function word in {result!r} ({bad!r})"
