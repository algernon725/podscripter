#!/usr/bin/env python3

import pytest

from punctuation_restorer import assemble_sentences_from_processed

pytestmark = pytest.mark.core


def assert_eq(a, b):
    if a != b:
        raise AssertionError(f"Expected {b!r} but got {a!r}")


def test_es_ellipsis_continuation_three_dots():
    processed = "De tener bancaroca... rota"
    sentences, trailing = assemble_sentences_from_processed(processed, 'es')
    combined = " ".join(sentences) + " " + trailing if sentences else trailing
    assert "bancaroca..." in combined, f"Ellipsis should not split text: {sentences}, trailing={trailing!r}"
    assert "rota" in combined, f"Text after ellipsis should stay together: {sentences}, trailing={trailing!r}"


def test_es_ellipsis_continuation_unicode():
    processed = "De tener bancaroca\u2026 rota"
    sentences, trailing = assemble_sentences_from_processed(processed, 'es')
    combined = " ".join(sentences) + " " + trailing if sentences else trailing
    assert "bancaroca\u2026" in combined, f"Unicode ellipsis should not split text: {sentences}, trailing={trailing!r}"
    assert "rota" in combined, f"Text after ellipsis should stay together: {sentences}, trailing={trailing!r}"


def test_es_domain_preservation_common_tlds():
    for domain in ["espanolistos.com", "ejemplo.NET", "sitio.Org"]:
        processed = f"Puedes ir a {domain} de nuevo"
        sentences, trailing = assemble_sentences_from_processed(processed, 'es')
        combined = " ".join(sentences) + " " + trailing if sentences else trailing
        assert domain.lower() in combined.lower(), (
            f"Domain {domain} should not be split: sentences={sentences}, trailing={trailing!r}"
        )


def test_fr_merge_short_connector_breaks():
    sentences = [
        "On a tout un tas de ressources qui sont gratuites, accessibles à toutes et tous, au.",
        "Moins en consultation."
    ]
    out, _ = assemble_sentences_from_processed(" ".join(sentences), 'fr')
    assert_eq(out, [
        "On a tout un tas de ressources qui sont gratuites, accessibles à toutes et tous, au moins en consultation."
    ])


def test_fr_merge_short_connector_breaks_negative():
    sentences = ["Ceci est une phrase.", "Bonjour."]
    out, _ = assemble_sentences_from_processed(" ".join(sentences), 'fr')
    assert_eq(out, sentences)


def test_es_decimal_preservation_percent_and_plain():
    processed = "Qué significa eso de que hay una alfabetización del 99.9% de la población. También hablaremos de 121.73 kilómetros."
    out, trailing = assemble_sentences_from_processed(processed, 'es')
    assert_eq(trailing, "")
    assert any("99.9" in s for s in out), f"Missing 99.9 in {out}"
    assert any("121.73" in s for s in out), f"Missing 121.73 in {out}"

