#!/usr/bin/env python3
"""
Dotted abbreviations and sentence-start domains in the TXT transcript (#20).

The TXT writer's _space_after_terminals() safety net spaces every "." followed
by a letter, which re-spaced English abbreviations ("a.m." -> "a. m.") and undid
the compact "C.S. Lewis" that _normalize_initials_and_acronyms() produces.
English now keeps them compact. Spanish and German keep the spaced forms, which
are their standard styles ("EE. UU.", "z. B."). Separately, a sentence that
starts with a domain no longer has it capitalized ("Www.tierragrata.org").
"""

import pytest
from conftest import transcript

pytestmark = pytest.mark.core


@pytest.mark.parametrize("language,text,expected", [
    # English keeps dotted abbreviations compact.
    ('en', "The meeting is at 10 a.m. tomorrow, e.g. before lunch.", ["a.m.", "e.g."]),
    ('en', "She has a Ph.D. in linguistics.", ["Ph.D."]),
    ('en', "He compared it to what's happening on the U.S.-Mexico border.", ["U.S.-Mexico"]),
    ('en', "I love reading C. S. Lewis because he wrote many books.", ["C.S. Lewis"]),
    # Spanish and German keep their standard spaced forms (decision on #20).
    ('es', "Vivo en EE.UU. desde hace diez años.", ["EE. UU."]),
    ('de', "Das kostet z.B. drei Euro.", ["z. B."]),
    # The safety net still separates sentences that were joined without a space.
    ('en', "He left.She stayed.", ["He left. She stayed."]),
])
def test_abbreviation_spacing_in_txt_output(language, text, expected, tmp_path):
    """Abbreviations are compact in English and spaced in es/de."""
    result = transcript(text, language, tmp_path)
    for fragment in expected:
        assert fragment in result, f"[{language}] expected '{fragment}' in transcript: '{result}'"


@pytest.mark.parametrize("language,text,start", [
    # Real Whisper segment (Episodio314) that became "Www.tierragrata.org ...".
    ('es', "www.tierragrata.org tiene toda la información.", "www.tierragrata.org tiene"),
    ('en', "blog.example.com has the full notes.", "blog.example.com has"),
])
def test_sentence_starting_with_domain_is_not_capitalized(language, text, start, tmp_path):
    """A domain at the start of a sentence keeps its lowercase form."""
    result = transcript(text, language, tmp_path)
    assert result.startswith(start), f"[{language}] expected to start with '{start}': '{result}'"
