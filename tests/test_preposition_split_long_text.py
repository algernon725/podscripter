#!/usr/bin/env python3
"""
Test for preposition split bug in LONG texts where semantic splitting is triggered.

The bug occurs when semantic splitting thresholds are exceeded (min_total_words_no_split).
This test uses realistic long Spanish text to trigger the semantic splitter.
"""

from conftest import restore_punctuation
import pytest

pytestmark = pytest.mark.core


def test_spanish_long_text_no_preposition_split():
    """Test that Spanish preposition 'a' doesn't cause splits in long texts."""
    long_text = """Claramente no sabíamos muy bien lo que estábamos haciendo 
    porque qué hicimos buscamos a una persona que supiera grabar videos 
    entonces yo conocí a un amigo que trabajaba con cámaras pero él estaba 
    muy ocupado y él me recomendó a un amigo de él y este chico pues nos 
    grababa y todo pero él nunca mencionó como tienen que comprar un 
    micrófono de solapa o sea un micrófono de los que se pone en la camisa 
    él nunca mencionó eso y es lo más básico y nosotros no lo pensamos 
    nosotros no teníamos la noción del eco y todo eso hicimos videos con 
    pantalla verde en el fondo y cambiamos el fondo pero sí ustedes pueden 
    ir a ver como los primeros veinte videos yo creo que la calidad pues 
    no es muy buena pero ya después conocimos a otra persona cambiamos de 
    videógrafo y así fuimos mejorando"""

    result = restore_punctuation(long_text, 'es')

    prepositions = ['a', 'de', 'en', 'con', 'por', 'para', 'ante']
    for prep in prepositions:
        error_pattern = f' {prep}. '
        assert error_pattern not in result, \
            f"Long text: sentence incorrectly split after '{prep}': ...{result[max(0, result.find(error_pattern)-50):result.find(error_pattern)+100]}..."

    sentences = [s.strip() for s in result.split('.') if s.strip()]
    for i, sentence in enumerate(sentences):
        for prep in prepositions:
            assert not sentence.endswith(f' {prep}'), \
                f"Sentence {i+1} ends with preposition '{prep}': '{sentence}'"


def test_all_spanish_prepositions_comprehensive():
    """Test comprehensive list of Spanish prepositions in realistic contexts."""
    test_cases = [
        ("vamos a la ciudad mañana por la tarde", "a"),
        ("vengo de Colombia y vivo en Estados Unidos", "de"),
        ("trabajo en una oficina grande", "en"),
        ("hablo con mi amigo sobre el proyecto", "con"),
        ("es un regalo para mi hermana", "para"),
        ("salgo sin mi chaqueta hoy", "sin"),
        ("el libro está sobre la mesa", "sobre"),
        ("caminamos entre los árboles del parque", "entre"),
        ("llegó tras una larga espera", "tras"),
        ("estudié durante toda la noche", "durante"),
        ("lo hizo mediante un proceso complejo", "mediante"),
        ("según las noticias va a llover", "según"),
        ("voy hacia el norte de la ciudad", "hacia"),
        ("trabajamos hasta las cinco de la tarde", "hasta"),
        ("vengo desde muy lejos", "desde"),
        ("luchamos contra la injusticia", "contra"),
        ("estamos ante un gran desafío", "ante"),
        ("vive bajo el puente", "bajo"),
    ]

    for text, preposition in test_cases:
        extended_text = f"Hola para todos cómo están ojalá que muy bien les cuento que {text} y eso es todo lo que quería decir"
        result = restore_punctuation(extended_text, 'es')

        error_pattern = f' {preposition}. '
        if error_pattern in result:
            context_start = max(0, result.find(error_pattern) - 60)
            context_end = min(len(result), result.find(error_pattern) + 60)
            context = result[context_start:context_end]
            raise AssertionError(f"Preposition '{preposition}' caused split. Context: ...{context}...")


def test_contracted_prepositions():
    """Test Spanish contracted prepositions (al, del)."""
    test_cases = [
        ("voy al mercado mañana temprano", "al"),
        ("vengo del trabajo muy cansado", "del"),
    ]

    for text, contraction in test_cases:
        extended_text = f"Les cuento una historia interesante {text} y fue una experiencia increíble"
        result = restore_punctuation(extended_text, 'es')

        error_pattern = f' {contraction}. '
        assert error_pattern not in result, \
            f"Contracted preposition '{contraction}' caused split: {result}"
