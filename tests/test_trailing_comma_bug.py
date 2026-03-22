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


