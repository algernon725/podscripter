#!/usr/bin/env python3

from punctuation_restorer import assemble_sentences_from_processed


def assert_eq(a, b):
    if a != b:
        raise AssertionError(f"Expected {b!r} but got {a!r}")


def test_es_ellipsis_continuation_three_dots():
    processed = "De tener bancaroca... rota"
    sentences, trailing = assemble_sentences_from_processed(processed, 'es')
    assert_eq(sentences, ["De tener bancaroca... rota."])
    assert_eq(trailing, "")


def test_es_ellipsis_continuation_unicode():
    processed = "De tener bancaroca… rota"
    sentences, trailing = assemble_sentences_from_processed(processed, 'es')
    assert_eq(sentences, ["De tener bancaroca… rota."])
    assert_eq(trailing, "")


def test_es_domain_preservation_common_tlds():
    for domain in ["espanolistos.com", "ejemplo.NET", "sitio.Org"]:
        processed = f"Puedes ir a {domain} de nuevo"
        sentences, trailing = assemble_sentences_from_processed(processed, 'es')
        expected = f"Puedes ir a {domain} de nuevo."
        assert_eq(sentences, [expected])
        assert_eq(trailing, "")


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


