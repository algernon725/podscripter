#!/usr/bin/env python3
"""
Tests for the long sentence break fixes.

These tests verify that:
1. Numbers are not split from following time/measurement units
2. Infinitive verbs are not split from their complements
3. Past tense auxiliary verbs are not split from following past participles

Bug examples fixed:
- "a los 18. Años" → "a los 18 años"
- "sin ser. Parte" → "sin ser parte"
- "fueron. Dirigidos" → "fueron dirigidos"
"""

import unittest

import pytest

from conftest import MockConfig
from sentence_splitter import SentenceSplitter

pytestmark = pytest.mark.core


class TestNumberTimeUnitGuard(unittest.TestCase):
    """Test that numbers are not split from following time/measurement units."""
    
    def setUp(self):
        self.config = MockConfig()
        self.splitter = SentenceSplitter('es', None, self.config)
    
    def test_number_followed_by_anos(self):
        """Numbers followed by 'años' should not be split."""
        words = "a los 18 años él tomó el poder".split()
        current_index = 2  # "18"
        current_chunk = words[:current_index + 1]  # ["a", "los", "18"]
        next_word = "años"
        
        result = self.splitter._passes_language_specific_checks(
            words, current_index, current_chunk, next_word
        )
        # Should return False (don't allow split after number before 'años')
        self.assertFalse(result, "Should NOT split after number before 'años'")
    
    def test_number_followed_by_personas(self):
        """Numbers followed by 'personas' should not be split."""
        words = "fueron 60 personas muertas".split()
        current_index = 1  # "60"
        current_chunk = words[:current_index + 1]
        next_word = "personas"
        
        result = self.splitter._passes_language_specific_checks(
            words, current_index, current_chunk, next_word
        )
        self.assertFalse(result, "Should NOT split after number before 'personas'")
    
    def test_number_followed_by_years_english(self):
        """Numbers followed by 'years' should not be split."""
        splitter = SentenceSplitter('en', None, self.config)
        words = "for 18 years he managed the company".split()
        current_index = 1  # "18"
        current_chunk = words[:current_index + 1]
        next_word = "years"
        
        result = splitter._passes_language_specific_checks(
            words, current_index, current_chunk, next_word
        )
        self.assertFalse(result, "Should NOT split after number before 'years'")
    
    def test_number_followed_by_regular_word(self):
        """Numbers followed by regular words CAN be split (if other conditions met)."""
        words = "capítulo 18 entonces empezamos".split()
        current_index = 1  # "18"
        current_chunk = words[:current_index + 1]
        next_word = "entonces"
        
        result = self.splitter._passes_language_specific_checks(
            words, current_index, current_chunk, next_word
        )
        # This should pass (True) as "entonces" is not a time unit
        self.assertTrue(result, "Should allow split after number before regular word")


class TestInfinitiveVerbGuard(unittest.TestCase):
    """Test that infinitive verbs are not split from their complements."""
    
    def setUp(self):
        self.config = MockConfig()
        self.splitter = SentenceSplitter('es', None, self.config)
    
    def test_ser_in_continuative_verbs(self):
        """'ser' should be in the CONTINUATIVE_AUXILIARY_VERBS set."""
        self.assertIn('ser', SentenceSplitter.CONTINUATIVE_AUXILIARY_VERBS)
    
    def test_estar_in_continuative_verbs(self):
        """'estar' should be in the CONTINUATIVE_AUXILIARY_VERBS set."""
        self.assertIn('estar', SentenceSplitter.CONTINUATIVE_AUXILIARY_VERBS)
    
    def test_haber_in_continuative_verbs(self):
        """'haber' should be in the CONTINUATIVE_AUXILIARY_VERBS set."""
        self.assertIn('haber', SentenceSplitter.CONTINUATIVE_AUXILIARY_VERBS)
    
    def test_grammatical_guard_for_ser(self):
        """'ser' should trigger grammatical guard (never end on auxiliary verbs)."""
        current_word = "ser"
        next_word = "parte"
        
        result = self.splitter._violates_grammatical_rules(current_word, next_word)
        self.assertTrue(result, "'ser' should violate grammatical rules (cannot end sentence)")


class TestPastTenseAuxiliaryGuard(unittest.TestCase):
    """Test that past tense auxiliary verbs are not split from past participles."""
    
    def setUp(self):
        self.config = MockConfig()
        self.splitter = SentenceSplitter('es', None, self.config)
    
    def test_fueron_in_continuative_verbs(self):
        """'fueron' should be in the CONTINUATIVE_AUXILIARY_VERBS set."""
        self.assertIn('fueron', SentenceSplitter.CONTINUATIVE_AUXILIARY_VERBS)
    
    def test_fue_in_continuative_verbs(self):
        """'fue' should be in the CONTINUATIVE_AUXILIARY_VERBS set."""
        self.assertIn('fue', SentenceSplitter.CONTINUATIVE_AUXILIARY_VERBS)
    
    def test_past_participle_detection_spanish(self):
        """Spanish past participles should be detected."""
        self.assertTrue(self.splitter._is_past_participle('dirigidos'))
        self.assertTrue(self.splitter._is_past_participle('dirigido'))
        self.assertTrue(self.splitter._is_past_participle('hablado'))
        self.assertTrue(self.splitter._is_past_participle('comido'))
        # Irregular
        self.assertTrue(self.splitter._is_past_participle('hecho'))
        self.assertTrue(self.splitter._is_past_participle('escrito'))
        self.assertTrue(self.splitter._is_past_participle('visto'))
    
    def test_past_participle_detection_english(self):
        """English past participles should be detected."""
        splitter = SentenceSplitter('en', None, self.config)
        self.assertTrue(splitter._is_past_participle('directed'))
        self.assertTrue(splitter._is_past_participle('spoken'))
        self.assertTrue(splitter._is_past_participle('written'))
        # Irregular
        self.assertTrue(splitter._is_past_participle('done'))
        self.assertTrue(splitter._is_past_participle('gone'))
        self.assertTrue(splitter._is_past_participle('been'))
    
    def test_auxiliary_verb_before_participle_guard(self):
        """Auxiliary verbs should not be split from following past participles."""
        words = "estos homicidios fueron dirigidos por pandilleros".split()
        current_index = 2  # "fueron"
        current_chunk = words[:current_index + 1]
        next_word = "dirigidos"
        
        result = self.splitter._passes_language_specific_checks(
            words, current_index, current_chunk, next_word
        )
        # Should return False (don't split auxiliary from participle)
        self.assertFalse(result, "Should NOT split auxiliary verb from past participle")
    
    def test_auxiliary_verb_before_regular_word_ok(self):
        """Auxiliary verbs followed by regular words CAN be split (if conditions met)."""
        words = "ellos fueron y luego regresaron".split()
        current_index = 1  # "fueron"
        current_chunk = words[:current_index + 1]
        next_word = "y"
        
        # Note: This would still be blocked by the grammatical rules check
        # on the auxiliary verb itself, but the participle-specific check
        # should pass since "y" is not a participle
        result = self.splitter._is_past_participle("y")
        self.assertFalse(result, "'y' should not be detected as past participle")


class TestGermanAuxiliaryGuard(unittest.TestCase):
    """Test German auxiliary verbs."""
    
    def setUp(self):
        self.config = MockConfig()
        self.splitter = SentenceSplitter('de', None, self.config)
    
    def test_german_auxiliaries_in_set(self):
        """German auxiliary verbs should be in the CONTINUATIVE_AUXILIARY_VERBS set."""
        german_auxiliaries = ['sein', 'haben', 'werden', 'ist', 'sind', 'hat', 'wurde', 'wurden']
        for aux in german_auxiliaries:
            self.assertIn(aux, SentenceSplitter.CONTINUATIVE_AUXILIARY_VERBS,
                         f"German auxiliary '{aux}' should be in CONTINUATIVE_AUXILIARY_VERBS")
    
    def test_german_past_participles(self):
        """German past participles should be detected."""
        self.assertTrue(self.splitter._is_past_participle('gemacht'))
        self.assertTrue(self.splitter._is_past_participle('gesehen'))
        self.assertTrue(self.splitter._is_past_participle('gewesen'))
        self.assertTrue(self.splitter._is_past_participle('gehabt'))


class TestFrenchAuxiliaryGuard(unittest.TestCase):
    """Test French auxiliary verbs."""
    
    def setUp(self):
        self.config = MockConfig()
        self.splitter = SentenceSplitter('fr', None, self.config)
    
    def test_french_auxiliaries_in_set(self):
        """French auxiliary verbs should be in the CONTINUATIVE_AUXILIARY_VERBS set."""
        french_auxiliaries = ['être', 'avoir', 'est', 'sont', 'a', 'ont', 'fut', 'furent']
        for aux in french_auxiliaries:
            self.assertIn(aux, SentenceSplitter.CONTINUATIVE_AUXILIARY_VERBS,
                         f"French auxiliary '{aux}' should be in CONTINUATIVE_AUXILIARY_VERBS")
    
    def test_french_past_participles(self):
        """French past participles should be detected."""
        self.assertTrue(self.splitter._is_past_participle('parlé'))
        self.assertTrue(self.splitter._is_past_participle('fini'))
        self.assertTrue(self.splitter._is_past_participle('vendu'))
        self.assertTrue(self.splitter._is_past_participle('fait'))
        self.assertTrue(self.splitter._is_past_participle('été'))
