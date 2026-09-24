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
        """'ser' should be in the Spanish CONTINUATIVE_AUXILIARY_VERBS pool."""
        self.assertIn('ser', self.splitter.CONTINUATIVE_AUXILIARY_VERBS)
    
    def test_estar_in_continuative_verbs(self):
        """'estar' should be in the Spanish CONTINUATIVE_AUXILIARY_VERBS pool."""
        self.assertIn('estar', self.splitter.CONTINUATIVE_AUXILIARY_VERBS)
    
    def test_haber_in_continuative_verbs(self):
        """'haber' should be in the Spanish CONTINUATIVE_AUXILIARY_VERBS pool."""
        self.assertIn('haber', self.splitter.CONTINUATIVE_AUXILIARY_VERBS)
    
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
        """'fueron' should be in the Spanish CONTINUATIVE_AUXILIARY_VERBS pool."""
        self.assertIn('fueron', self.splitter.CONTINUATIVE_AUXILIARY_VERBS)
    
    def test_fue_in_continuative_verbs(self):
        """'fue' should be in the Spanish CONTINUATIVE_AUXILIARY_VERBS pool."""
        self.assertIn('fue', self.splitter.CONTINUATIVE_AUXILIARY_VERBS)
    
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
        """German auxiliary verbs should be in the German pool."""
        german_auxiliaries = ['sein', 'haben', 'werden', 'ist', 'sind', 'hat', 'wurde', 'wurden']
        for aux in german_auxiliaries:
            self.assertIn(aux, self.splitter.CONTINUATIVE_AUXILIARY_VERBS,
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
        """French auxiliary verbs should be in the French pool."""
        french_auxiliaries = ['être', 'avoir', 'est', 'sont', 'a', 'ont', 'fut', 'furent']
        for aux in french_auxiliaries:
            self.assertIn(aux, self.splitter.CONTINUATIVE_AUXILIARY_VERBS,
                         f"French auxiliary '{aux}' should be in CONTINUATIVE_AUXILIARY_VERBS")
    
    def test_french_past_participles(self):
        """French past participles should be detected."""
        self.assertTrue(self.splitter._is_past_participle('parlé'))
        self.assertTrue(self.splitter._is_past_participle('fini'))
        self.assertTrue(self.splitter._is_past_participle('vendu'))
        self.assertTrue(self.splitter._is_past_participle('fait'))
        self.assertTrue(self.splitter._is_past_participle('été'))


class TestWordPoolLanguageIsolation(unittest.TestCase):
    """No language may see another language's function words.

    CONNECTOR_WORDS / COORDINATING_CONJUNCTIONS / CONTINUATIVE_AUXILIARY_VERBS are
    language-keyed dicts (*_BY_LANG) resolved onto the instance in
    SentenceSplitter.__init__. Before v0.13.0 the es/en/fr/de words sat in one
    shared pool, so a Spanish sentence had German auxiliaries and French 'a'/'or'
    forbidden as endings; Portuguese could not join that pool at all and was
    unioned onto the instance as a special case.

    These tests pin the per-language separation so a future change cannot
    re-merge the pools.
    """

    # (word, language it would break if pooled, why)
    COLLIDING_WORDS = [
        ('logo', 'en', 'English noun: "Check out the new logo."'),
        ('vamos', 'es', 'Spanish: "¡Vamos!"'),
        ('vais', 'fr', 'French: "J\'y vais."'),
        ('ora', 'es', 'Spanish orar imperative'),
    ]

    # (word, owning language, language that must NOT see it, why)
    CROSS_LANGUAGE_WORDS = [
        ('sein', 'de', 'es', 'German infinitive; Spanish has no such word'),
        ('hatte', 'de', 'fr', 'German preterite'),
        ('a', 'fr', 'es', 'French "a" (avoir); Spanish "a" is a preposition '
                          'handled by its own forbidden set'),
        ('or', 'fr', 'de', 'French "or" (= now/yet)'),
        ('yet', 'en', 'es', 'English coordinating conjunction'),
        ('sondern', 'de', 'en', 'German coordinating conjunction'),
    ]

    @staticmethod
    def _all_pools(splitter):
        return (splitter.CONNECTOR_WORDS
                | splitter.COORDINATING_CONJUNCTIONS
                | splitter.CONTINUATIVE_AUXILIARY_VERBS)

    def _splitter(self, language):
        return SentenceSplitter(language, None, MockConfig())

    def test_portuguese_instance_has_the_words(self):
        pt_pool = self._all_pools(self._splitter('pt'))
        for word, _lang, _why in self.COLLIDING_WORDS:
            self.assertIn(word, pt_pool,
                          f"Portuguese splitter should recognize {word!r}")

    def test_other_languages_do_not_see_portuguese_words(self):
        for word, lang, why in self.COLLIDING_WORDS:
            pool = self._all_pools(self._splitter(lang))
            self.assertNotIn(word, pool,
                             f"{word!r} leaked into {lang!r} pool — {why}")

    def test_languages_do_not_see_each_others_words(self):
        """The v0.13.0 split: es/en/fr/de no longer share one pool either."""
        for word, owner, other, why in self.CROSS_LANGUAGE_WORDS:
            owner_pool = self._all_pools(self._splitter(owner))
            other_pool = self._all_pools(self._splitter(other))
            self.assertIn(word, owner_pool,
                          f"{word!r} should be in the {owner!r} pool")
            self.assertNotIn(word, other_pool,
                             f"{word!r} leaked into {other!r} pool — {why}")

    def test_pools_are_keyed_by_language(self):
        """Each supported language resolves its own set, not a shared one."""
        pools = {lang: self._all_pools(self._splitter(lang))
                 for lang in ('es', 'en', 'fr', 'de', 'pt')}
        for lang, pool in pools.items():
            for other, other_pool in pools.items():
                if lang != other:
                    self.assertNotEqual(pool, other_pool,
                                        f"{lang!r} and {other!r} share a pool")

    def test_unknown_language_falls_back_to_the_union(self):
        """A language with no entry keeps the pre-v0.13.0 pooled behavior."""
        splitter = self._splitter('it')
        self.assertEqual(splitter.CONNECTOR_WORDS,
                         SentenceSplitter.ALL_CONNECTOR_WORDS)
        self.assertEqual(splitter.COORDINATING_CONJUNCTIONS,
                         SentenceSplitter.ALL_COORDINATING_CONJUNCTIONS)
        self.assertEqual(splitter.CONTINUATIVE_AUXILIARY_VERBS,
                         SentenceSplitter.ALL_CONTINUATIVE_AUXILIARY_VERBS)

    def test_class_level_dicts_are_not_mutated(self):
        """Resolution must bind instance attributes, never edit the class dicts."""
        before = {
            lang: set(words)
            for lang, words in SentenceSplitter.CONNECTOR_WORDS_BY_LANG.items()
        }
        for lang in ('pt', 'es', 'en', 'fr', 'de', 'it'):
            self._splitter(lang)
        after = {
            lang: set(words)
            for lang, words in SentenceSplitter.CONNECTOR_WORDS_BY_LANG.items()
        }
        self.assertEqual(before, after)

    def test_instantiating_portuguese_does_not_affect_later_splitters(self):
        """Building a pt splitter must not bleed into subsequently built ones."""
        self._splitter('pt')  # build a pt splitter first
        es_pool = self._all_pools(self._splitter('es'))
        self.assertNotIn('vamos', es_pool)
        self.assertNotIn('logo', es_pool)

    def test_portuguese_forbidden_endings_are_language_scoped(self):
        """Portuguese 'no' (= "in the") is forbidden; Spanish 'no' (= "not") is not."""
        pt = self._splitter('pt')
        es = self._splitter('es')
        self.assertTrue(pt._violates_grammatical_rules('no', 'Porto'))
        self.assertFalse(es._violates_grammatical_rules('no', 'Claro'))


class TestQuotedTeachingLanguageGuard(unittest.TestCase):
    """A language must not end a sentence on a quoted word of the language it teaches.

    podscripter targets language-learning podcasts, where the host speaks one
    language and quotes another as vocabulary. Episodio311 is a Spanish episode
    about conjunctions in which "so" appears twelve times as an English word under
    discussion; breaking after it strands the following verb with no subject
    ("Esa palabra, so." / "Tiene como más de diez traducciones").

    Before v0.13.0 this was protected by accident, because every language shared
    one word pool. Splitting the pools per language removed that protection, so
    the requirement is stated explicitly by QUOTED_TEACHING_LANGUAGE_WORDS.
    """

    def _splitter(self, language):
        return SentenceSplitter(language, None, MockConfig())

    def test_spanish_does_not_end_on_quoted_english_conjunctions(self):
        es = self._splitter('es')
        for word in ('so', 'yet', 'nor', 'and', 'but'):
            self.assertTrue(
                es._violates_grammatical_rules(word, 'Tiene'),
                f"Spanish should not end a sentence on quoted English {word!r}",
            )

    def test_english_does_not_end_on_quoted_spanish_conjunctions(self):
        en = self._splitter('en')
        for word in ('pero', 'sino', 'mas'):
            self.assertTrue(
                en._violates_grammatical_rules(word, 'Means'),
                f"English should not end a sentence on quoted Spanish {word!r}",
            )

    def test_guard_is_additive_not_a_pool_merge(self):
        """The quoted words stay out of the language's own pools."""
        es = self._splitter('es')
        for word in ('so', 'yet', 'nor'):
            self.assertNotIn(word, es.COORDINATING_CONJUNCTIONS)
            self.assertNotIn(word, es.CONNECTOR_WORDS)
            self.assertNotIn(word, es.CONTINUATIVE_AUXILIARY_VERBS)
            self.assertIn(word, es.QUOTED_TEACHING_WORDS)

    def test_guard_does_not_block_other_foreign_words(self):
        """Only the listed function words are guarded, not any foreign word."""
        es = self._splitter('es')
        self.assertFalse(es._violates_grammatical_rules('trampolines', 'A'))
        self.assertFalse(es._violates_grammatical_rules('podcast', 'Obviamente'))

    def test_unknown_language_has_no_quoted_guard(self):
        """An unrecognised language already gets the full union; no extra guard."""
        self.assertEqual(self._splitter('it').QUOTED_TEACHING_WORDS, frozenset())
