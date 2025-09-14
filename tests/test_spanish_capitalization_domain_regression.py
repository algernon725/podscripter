#!/usr/bin/env python3
"""
Regression tests for Spanish capitalization and domain handling interaction.

This test specifically covers the regression where:
1. Domain merging works correctly to create "www.espanolistos.com"
2. Final capitalization correction doesn't break the merged domains
3. Common Spanish words like "episodio" are correctly lowercased in mid-sentence contexts
4. Both fixes work together without interfering with each other

This addresses the issue where spacing rules in _sanitize_sentence_output were
breaking domains that had been correctly merged by the domain merging logic.
"""

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from podscripter import _assemble_sentences  # noqa: E402


def test_domain_capitalization_regression():
    """Test the specific regression where domain merging and capitalization correction interfered."""
    
    # This simulates the exact segments that caused the regression:
    # Segment 62: "Así que asegúrate de ir a www.espanolistos.com"
    # Segment 63: "Ve ya mismo o apenas puedas a www.espanolistos.com"  
    # The key is that domains should be intact in each sentence
    
    test_text = """Así que asegúrate de ir a www.espanolistos.com

Ve ya mismo o apenas puedas a www.espanolistos.com y Ahí Vas a ver este"""
    
    sentences = _assemble_sentences(test_text, 'es', quiet=True)
    
    # Should have 2 sentences, both with intact domains
    assert len(sentences) == 2, f"Expected 2 sentences, got {len(sentences)}: {sentences}"
    
    # Check both sentences have intact domains (case-insensitive)
    for i, sentence in enumerate(sentences):
        if "espanolistos.com" in sentence.lower():
            # Domain should be intact (no spaces)
            assert "www. " not in sentence, f"Domain broken with space after www. in sentence {i}: {sentence}"
            assert ". com" not in sentence, f"Domain broken with space before .com in sentence {i}: {sentence}"
    
    # Verify we found domains in the sentences
    domain_count = sum(1 for s in sentences if "espanolistos.com" in s.lower())
    assert domain_count >= 1, f"No intact domains found in sentences: {sentences}"


def test_episodio_capitalization_fix():
    """Test that 'episodio' is correctly lowercased in mid-sentence contexts."""
    
    # This simulates the segments that caused the episodio capitalization issue:
    # Segment 49: "Hoy en este podcast no solo tenemos el transcript,"
    # Segment 50: "también vamos a darte el cheat sheet de este episodio"
    # Segment 51: "con los siete formas de mejorar su español."
    
    test_text = """Hoy en este podcast no solo tenemos el transcript,

también vamos a darte el cheat sheet de este episodio

con los siete formas de mejorar su español."""
    
    sentences = _assemble_sentences(test_text, 'es', quiet=True)
    
    # Find the sentence containing episodio
    episodio_sentence = None
    for sentence in sentences:
        if 'episodio' in sentence.lower():
            episodio_sentence = sentence
            break
    
    assert episodio_sentence is not None, f"No sentence containing 'episodio' found in: {sentences}"
    
    # episodio should be lowercase in mid-sentence context
    assert "episodio" in episodio_sentence, f"'episodio' not found in: {episodio_sentence}"
    assert "Episodio" not in episodio_sentence, f"'Episodio' incorrectly capitalized in: {episodio_sentence}"


def test_combined_domain_and_episodio_fix():
    """Test that both domain handling and episodio capitalization work together."""
    
    # This combines both issues in one test case
    test_text = """Hoy en este podcast tenemos el cheat sheet de este, episodio con información importante.

Así que asegúrate de ir a www.espanolistos.com para descargar el episodio completo."""
    
    sentences = _assemble_sentences(test_text, 'es', quiet=True)
    
    # Should have the right number of sentences
    assert len(sentences) >= 1, f"Expected at least 1 sentence, got {len(sentences)}: {sentences}"
    
    # Find sentences with our test content
    episodio_sentence = None
    domain_sentence = None
    
    for sentence in sentences:
        if 'cheat sheet' in sentence and 'episodio' in sentence.lower():
            episodio_sentence = sentence
        if 'espanolistos.com' in sentence.lower():
            domain_sentence = sentence
    
    # Test episodio capitalization
    assert episodio_sentence is not None, f"Episodio sentence not found in: {sentences}"
    assert ", episodio " in episodio_sentence, f"'episodio' not lowercase after comma in: {episodio_sentence}"
    assert ", Episodio " not in episodio_sentence, f"'Episodio' incorrectly capitalized after comma in: {episodio_sentence}"
    
    # Test domain integrity  
    assert domain_sentence is not None, f"Domain sentence not found in: {sentences}"
    assert "espanolistos.com" in domain_sentence.lower(), f"Domain not intact in: {domain_sentence}"
    assert "www. " not in domain_sentence, f"Domain broken with space after www. in: {domain_sentence}"
    assert ". com" not in domain_sentence, f"Domain broken with space before .com in: {domain_sentence}"


def test_subdomain_patterns():
    """Test various subdomain patterns that should not be broken by spacing rules."""
    
    subdomains = [
        "www.example.com",
        "ftp.example.com", 
        "mail.example.com",
        "blog.example.com",
        "shop.example.com",
        "api.example.com",
        "cdn.example.com",
        "static.example.com",
        "news.example.com",
        "support.example.com",
        "help.example.com",
        "docs.example.com",
        "admin.example.com",
        "secure.example.com",
        "login.example.com",
        "mobile.example.com",
        "store.example.com",
        "sub.example.com",
        "dev.example.com",
        "test.example.com",
        "staging.example.com",
        "prod.example.com",
        "beta.example.com",
        "alpha.example.com"
    ]
    
    for subdomain in subdomains:
        test_text = f"Visita {subdomain} para más información sobre este episodio."
        sentences = _assemble_sentences(test_text, 'es', quiet=True)
        
        assert len(sentences) == 1, f"Subdomain {subdomain} caused sentence split: {sentences}"
        sentence = sentences[0]
        
        # Domain should be intact (case-insensitive check)
        assert subdomain.lower() in sentence.lower(), f"Subdomain {subdomain} not found intact in: {sentence}"
        
        # Should not have spaces breaking the domain
        assert "www. " not in sentence, f"Subdomain {subdomain} broken with space after www. in: {sentence}"
        assert ". com" not in sentence, f"Subdomain {subdomain} broken with space before .com in: {sentence}"
        
        # episodio should be lowercase
        assert "episodio" in sentence, f"'episodio' not lowercase with {subdomain}: {sentence}"
        assert "Episodio" not in sentence, f"'Episodio' incorrectly capitalized with {subdomain}: {sentence}"


def test_spanish_common_words_capitalization():
    """Test that common Spanish words are correctly lowercased in mid-sentence contexts."""
    
    common_words = [
        'episodio', 'capítulo', 'temporada', 'parte', 'sección', 'tema', 'momento', 'tiempo',
        'ojalá', 'entonces', 'pero', 'también', 'además', 'ahora', 'después', 'antes', 
        'luego', 'finalmente', 'mientras', 'cuando', 'donde', 'aunque', 'porque'
    ]
    
    for word in common_words:
        # Test with comma before the word (mid-sentence context)
        test_text = f"Hoy en este podcast, {word} va a ser muy importante para todos."
        sentences = _assemble_sentences(test_text, 'es', quiet=True)
        
        assert len(sentences) == 1, f"Word {word} caused sentence split: {sentences}"
        sentence = sentences[0]
        
        # Word should be lowercase after comma
        assert f", {word} " in sentence, f"Word '{word}' not lowercase after comma in: {sentence}"
        assert f", {word.capitalize()} " not in sentence, f"Word '{word}' incorrectly capitalized after comma in: {sentence}"


if __name__ == "__main__":
    # Run tests directly
    test_domain_capitalization_regression()
    test_episodio_capitalization_fix()
    test_combined_domain_and_episodio_fix()
    test_subdomain_patterns()
    test_spanish_common_words_capitalization()
    print("All Spanish capitalization and domain regression tests passed")
