#!/usr/bin/env python3
"""
_finalize_text_common() must resolve domain exclusions from the transcript's
own language.

Until v0.13.0 it hardcoded `language='es'` for every language except `pt`, so
en/fr/de inherited Spanish's `.de`/`.es` TLD suppression. Spanish suppresses
those two TLDs deliberately — "de" is a preposition and "es" is a verb, so
"tratada.de" must become "tratada. De" — but applying that rule to German broke
"spiegel.de" into "spiegel. De".
"""

import pytest

from punctuation_restorer import _finalize_text_common

pytestmark = pytest.mark.core


@pytest.mark.parametrize("language,text", [
    ('de', "Besuchen Sie spiegel.de für mehr Informationen."),
    ('en', "Visit example.de for more information."),
    ('fr', "Visitez example.es pour plus d'informations."),
    ('en', "Visit example.es for more information."),
])
def test_de_and_es_domains_survive_in_non_spanish(language, text):
    """.de / .es are real TLDs everywhere except Spanish, where they collide."""
    out = _finalize_text_common(text, language)
    assert out == text, f"{language}: domain was broken: {out!r}"


@pytest.mark.parametrize("text,expected", [
    # history.md: Spanish must NOT treat these as domains.
    ("La página tratada.de forma correcta.", "La página tratada. De forma correcta."),
    ("Visita naturales.es ahora.", "Visita naturales. Es ahora."),
])
def test_spanish_still_splits_its_colliding_tlds(text, expected):
    """The long-standing Spanish guarantee is unchanged."""
    assert _finalize_text_common(text, 'es') == expected


@pytest.mark.parametrize("language", ['es', 'en', 'fr', 'de', 'pt'])
def test_real_domains_survive_in_every_language(language):
    """A TLD that collides with no common word is preserved everywhere."""
    text = f"Go to example.com and example.org now."
    assert _finalize_text_common(text, language) == text


def test_portuguese_com_is_still_handled():
    """'com' is Portuguese for 'with'; the contiguous form is still a domain."""
    text = "Acesse exemplo.com agora."
    assert _finalize_text_common(text, 'pt') == text


def test_no_language_falls_back_to_spanish_exclusions():
    """Called without a language, behavior is unchanged from before v0.13.0.

    _is_excluded_label() applies SPANISH_EXCLUSIONS for every language, so the
    common-word guard does not depend on this argument; only the TLD table does.
    """
    text = "Visit example.com today."
    assert _finalize_text_common(text) == text
