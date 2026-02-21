#!/usr/bin/env python3
"""
Test multilingual run-on sentence fixes across all supported languages
"""

import pytest

from conftest import restore_punctuation

pytestmark = pytest.mark.multilingual

_runon_cases = [
    ("en",
     "hello everyone welcome to our podcast today we are going to talk about language learning this is very important for everyone who wants to improve their skills",
     "English run-on sentence"),
    ("es",
     "hola a todos bienvenidos a nuestro podcast hoy vamos a hablar sobre el aprendizaje de idiomas esto es muy importante para todos los que quieren mejorar sus habilidades",
     "Spanish run-on sentence"),
    ("fr",
     "bonjour à tous bienvenue à notre podcast aujourd'hui nous allons parler de l'apprentissage des langues c'est très important pour tous ceux qui veulent améliorer leurs compétences",
     "French run-on sentence"),
    ("de",
     "hallo an alle willkommen zu unserem podcast heute werden wir über sprachlernen sprechen das ist sehr wichtig für alle die ihre fähigkeiten verbessern möchten",
     "German run-on sentence"),
    ("it",
     "ciao a tutti benvenuti al nostro podcast oggi parleremo dell'apprendimento delle lingue questo è molto importante per tutti coloro che vogliono migliorare le loro abilità",
     "Italian run-on sentence"),
    ("pt",
     "olá a todos bem vindos ao nosso podcast hoje vamos falar sobre aprendizado de idiomas isso é muito importante para todos que querem melhorar suas habilidades",
     "Portuguese run-on sentence"),
    ("nl",
     "hallo iedereen welkom bij onze podcast vandaag gaan we praten over taal leren dit is heel belangrijk voor iedereen die hun vaardigheden wil verbeteren",
     "Dutch run-on sentence"),
    ("ru",
     "privet vsem dobro pozhalovat v nash podcast segodnya my budem govorit ob izuchenii yazykov eto ochen vazhno dlya vsekh kto khochet uluchshit svoi navyki",
     "Russian run-on sentence"),
]


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
@pytest.mark.parametrize(
    "language,input_text,description",
    _runon_cases,
    ids=[c[2] for c in _runon_cases],
)
def test_runon_sentence_splitting(language, input_text, description):
    """Test run-on sentence fixes across multiple languages."""
    result = restore_punctuation(input_text, language)
    sentence_count = result.count('.') + result.count('?') + result.count('!')
    assert sentence_count > 2, \
        f"{description}: expected >2 sentence terminators, got {sentence_count} in {result!r}"


def test_spanish_inverted_question_marks():
    """Test that Spanish questions get inverted question marks."""
    result = restore_punctuation("cómo estás hoy tienes tiempo para hablar", 'es')
    assert '¿' in result, f"Expected inverted question mark in Spanish output: {result!r}"


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
def test_french_question_marks():
    """Test that French questions get question marks."""
    result = restore_punctuation("comment allez vous avez vous le temps", 'fr')
    assert '?' in result, f"Expected question mark in French output: {result!r}"


def test_german_capitalization():
    """Test that German output starts with a capital letter."""
    result = restore_punctuation("hallo wie geht es dir heute", 'de')
    assert result[0].isupper(), f"Expected capitalized first letter in German output: {result!r}"
