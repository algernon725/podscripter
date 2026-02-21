#!/usr/bin/env python3
"""
Regression test for subdomain sentence splitting bug.

Ensures that domains with subdomain prefixes (www., blog., ftp., etc.) 
are not incorrectly split during Spanish transcription processing.

Bug was: "Visit www.google.com" became "Visit www. Google.com" and split into two sentences.
Fix: Enhanced domain masking to handle subdomain patterns before space insertion.
"""

import pytest
from conftest import restore_punctuation
from domain_utils import mask_domains, unmask_domains

pytestmark = pytest.mark.core


def test_subdomain_sentence_splitting():
    """Test that subdomain patterns are not split during sentence processing."""

    test_cases = [
        ("Visit www.google.com for search", "Visit www.google.com for search."),
        ("Go to blog.example.org please", "Go to blog.example.org please."),
        ("Check ftp.downloads.net today", "Check ftp.downloads.net today."),
        ("Ve a espanolistos.com ahora", "Ve a espanolistos.com ahora."),
        ("Visita www.espanolistos.com por favor", "Visita www.espanolistos.com por favor."),
        ("Visit mail.company.co.uk", "Visit mail.company.co.uk."),
        ("Check api.service.com", "Check api.service.com."),
        ("Go to cdn.assets.net", "Go to cdn.assets.net."),
        ("Visit help.support.org", "Visit help.support.org."),
        ("Visit www.google.com and then espanolistos.com", "Visit www.google.com and then espanolistos.com."),
    ]

    for original, expected in test_cases:
        result = restore_punctuation(original, language='es')

        has_split_domain = any(pattern in result for pattern in [
            ". com", ". org", ". net", ". co.uk", ". com.br",
            "www. ", "blog. ", "ftp. ", "api. ", "cdn. "
        ])

        assert not has_split_domain, (
            f"Domain split detected for '{original}': got '{result}'"
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
