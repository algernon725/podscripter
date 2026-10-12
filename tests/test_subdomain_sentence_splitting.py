#!/usr/bin/env python3
"""
Regression test for subdomain splitting in the TXT transcript.

Ensures that domains with subdomain prefixes (www., blog., ftp., etc.) survive
into the written transcript intact, in every tailored language.

Bug was: "Visit www.google.com" became "Visit www. google.com" (#21). The TXT
writer's _space_after_terminals() spaced every dot and only fix_spaced_domains()
repaired "label. tld", never the "www. " prefix. This test used to check
restore_punctuation()[0], which the TXT writer never uses, so it passed while
the transcript was broken; it now runs _assemble_sentences() -> _write_txt().
"""

import pytest
from conftest import transcript
from domain_utils import mask_domains, unmask_domains

pytestmark = pytest.mark.core


@pytest.mark.parametrize("language,text,domains", [
    ('es', "Visit www.google.com for search", ["www.google.com"]),
    ('es', "Go to blog.example.org please", ["blog.example.org"]),
    ('es', "Check ftp.downloads.net today", ["ftp.downloads.net"]),
    ('es', "Ve a espanolistos.com ahora", ["espanolistos.com"]),
    ('es', "Visita www.espanolistos.com por favor", ["www.espanolistos.com"]),
    ('es', "Visit mail.company.co.uk", ["mail.company.co.uk"]),
    ('es', "Check api.service.com", ["api.service.com"]),
    ('es', "Go to cdn.assets.net", ["cdn.assets.net"]),
    ('es', "Visit help.support.org", ["help.support.org"]),
    ('es', "Visit www.google.com and then espanolistos.com", ["www.google.com", "espanolistos.com"]),
    # Real Whisper output (Episodio314): the transcript had "www. tierragrata.org".
    ('es', "Bueno, en nuestra página web www.tierragrata.org hay una parte que se llama Contribuye.",
     ["www.tierragrata.org"]),
    ('en', "Go to blog.example.com and subscribe.", ["blog.example.com"]),
    ('fr', "Allez sur www.exemple.fr pour plus d'informations.", ["www.exemple.fr"]),
    ('de', "Besuchen Sie www.spiegel.de für mehr.", ["www.spiegel.de"]),
    ('pt', "Acesse www.exemplo.com.br agora.", ["www.exemplo.com.br"]),
    # Portuguese suppresses the ".com" rejoin ("acabou. Com ele"), so a plain .com
    # domain was split to "exemplo. com" with nothing to repair it.
    ('pt', "Acesse exemplo.com agora.", ["exemplo.com"]),
])
def test_subdomain_survives_txt_output(language, text, domains, tmp_path):
    """Domains (with or without a subdomain prefix) reach the transcript intact."""
    result = transcript(text, language, tmp_path)
    for domain in domains:
        assert domain in result, (
            f"[{language}] domain '{domain}' split in transcript: '{result}'"
        )


def test_subdomain_masking():
    """Test that subdomain masking works correctly."""

    test_cases = [
        "Visit www.google.com",
        "Check blog.example.org",
        "Go to ftp.downloads.net",
        "Visit mail.company.co.uk",
        "Check sinónimosonline.com",
    ]

    for test_case in test_cases:
        masked = mask_domains(test_case, use_exclusions=True, language='es')
        unmasked = unmask_domains(masked)

        assert test_case == unmasked, (
            f"Masking round-trip failed for '{test_case}': "
            f"masked='{masked}', unmasked='{unmasked}'"
        )

        has_subdomain = any(prefix in test_case for prefix in ['www.', 'blog.', 'ftp.', 'mail.'])
        subdomain_masked = '__DOT__' in masked and any(prefix in masked for prefix in ['www', 'blog', 'ftp', 'mail'])

        if has_subdomain:
            assert subdomain_masked, f"Subdomain not masked for '{test_case}': masked='{masked}'"
