#!/usr/bin/env python3
"""
Test for trailing comma bug fix.

Bug: When Whisper segment ends with comma and contains question words,
Spanish formatting was appending '?' without removing the comma, resulting in ", ?"

Example: "Sin importar si crees en Dios o no, sin importar si vas a la iglesia o no,"
Was producing: "...o no, ?"
Should produce: "...o no?"  or "...o no."
"""

import unittest

import pytest

from punctuation_restorer import restore_punctuation

pytestmark = pytest.mark.core


class TestTrailingCommaBug(unittest.TestCase):
    """Test that trailing commas are stripped before adding terminal punctuation."""
    
    def test_spanish_trailing_comma_with_question_words(self):
        """Test the specific bug case: sentence with trailing comma containing 'no'."""
        # This is the exact text from the bug report
        text = "Sin importar si crees en Dios o no, sin importar si vas a la iglesia o no,"
        
        processed, sentences = restore_punctuation(text, 'es')
        
        # Should not have ", ?" anywhere
        self.assertNotIn(', ?', processed, 
                        "Should not have comma followed by question mark")
        
        # Should not have ", ?" in any sentence
        for sentence in sentences:
            s = sentence.text if hasattr(sentence, 'text') else sentence
            self.assertNotIn(', ?', s,
                           f"Sentence should not contain ', ?': {s}")
            self.assertTrue(s.rstrip().endswith(('.', '!', '?')),
                          f"Sentence should end with terminal punctuation: {s}")
    
    def test_spanish_trailing_comma_no_question_words(self):
        """Test trailing comma is removed even without question words."""
        # Use a sentence without any question words (avoid "es", "no", etc.)
        text = "Hola a todos,"
        
        processed, sentences = restore_punctuation(text, 'es')
        
        # Should not have trailing comma
        self.assertFalse(processed.rstrip().endswith(','),
                        "Should not end with comma")
        
        # Should end with period
        self.assertTrue(processed.rstrip().endswith('.'),
                       "Should end with period")
    
    def test_spanish_trailing_semicolon(self):
        """Test that other trailing punctuation is also removed."""
        text = "Este es otro ejemplo;"
        
        processed, sentences = restore_punctuation(text, 'es')
        
        # Should not have trailing semicolon
        self.assertFalse(processed.rstrip().endswith(';'),
                        "Should not end with semicolon")
    
    def test_spanish_already_has_period(self):
        """Test that sentences with proper punctuation are not modified."""
        text = "Este es un ejemplo correcto."
        
        processed, sentences = restore_punctuation(text, 'es')
        
        # Should still end with period
        self.assertTrue(processed.rstrip().endswith('.'),
                       "Should still end with period")
        
        # Should not have duplicates
        self.assertNotIn('..', processed,
                        "Should not have double periods")


class TestApplySemanticPunctuationTrailingComma(unittest.TestCase):
    """
    Test that _apply_semantic_punctuation strips trailing commas before
    appending terminal punctuation.

    Bug: When a Whisper segment was split mid-clause at a speaker boundary
    (e.g. "Bueno, más o menos." → "Bueno," + "Más o menos."), the leading
    fragment "Bueno," was classified as an exclamation by semantic similarity
    to "¡Qué bueno!" and the code appended '!' without stripping the trailing
    comma, producing "Bueno,!".

    Found in Episodio270.txt (line 103) when running with --enable-diarization.
    """

    @classmethod
    def setUpClass(cls):
        from punctuation_restorer import _load_sentence_transformer
        cls.model = _load_sentence_transformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )

    def _apply(self, text, language='es'):
        from punctuation_restorer import _apply_semantic_punctuation
        return _apply_semantic_punctuation(text, self.model, language, 0, 1)

    def test_spanish_bueno_with_trailing_comma(self):
        """The exact regression case: "Bueno," must not become "Bueno,!"."""
        out = self._apply("Bueno,", 'es')
        self.assertNotIn(',!', out, f"Should not contain ',!' artifact: {out!r}")
        self.assertNotIn(',?', out, f"Should not contain ',?' artifact: {out!r}")
        self.assertNotIn(',.', out, f"Should not contain ',.' artifact: {out!r}")
        self.assertTrue(
            out.endswith(('.', '!', '?')),
            f"Should end with terminal punctuation: {out!r}",
        )
        # Stripped comma + terminal punctuation: "Bueno!" / "Bueno." / "Bueno?"
        self.assertEqual(out.rstrip('.!?'), 'Bueno', f"Unexpected body: {out!r}")

    def test_no_dangling_punct_before_terminal_es(self):
        """Multiple Spanish fragments with trailing commas/semicolons/colons."""
        cases = ["Bueno,", "Hola,", "Sin embargo,", "O sea,", "Pues bien;", "Listo:"]
        for text in cases:
            with self.subTest(text=text):
                out = self._apply(text, 'es')
                self.assertNotIn(',!', out, f"{text!r} -> {out!r}")
                self.assertNotIn(',?', out, f"{text!r} -> {out!r}")
                self.assertNotIn(';!', out, f"{text!r} -> {out!r}")
                self.assertNotIn(';?', out, f"{text!r} -> {out!r}")
                self.assertNotIn(':!', out, f"{text!r} -> {out!r}")
                self.assertNotIn(':?', out, f"{text!r} -> {out!r}")
                self.assertTrue(
                    out.endswith(('.', '!', '?')),
                    f"Should end with terminal punctuation: {out!r}",
                )

    def test_no_dangling_punct_before_terminal_multilingual(self):
        """Same guarantee should hold for the other primary languages."""
        cases = [
            ('en', "Well,"),
            ('en', "Hello,"),
            ('fr', "Bon,"),
            ('fr', "Bonjour,"),
            ('de', "Gut,"),
            ('de', "Hallo,"),
        ]
        for lang, text in cases:
            with self.subTest(lang=lang, text=text):
                out = self._apply(text, lang)
                self.assertNotIn(',!', out, f"[{lang}] {text!r} -> {out!r}")
                self.assertNotIn(',?', out, f"[{lang}] {text!r} -> {out!r}")
                self.assertTrue(
                    out.endswith(('.', '!', '?')),
                    f"[{lang}] Should end with terminal punctuation: {out!r}",
                )


