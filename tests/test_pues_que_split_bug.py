"""
Test for Spanish question splitting bug: "Pues, ¿qué pasó, Nate?" incorrectly
splits into "Pues, ¿Qué?" and "¿Pasó, Nate?" during semantic sentence splitting.

The issue occurs when:
1. A question starts with "¿" mid-sentence
2. The first word after "¿" gets capitalized
3. The semantic splitter sees the capitalized word and incorrectly splits there
4. The guard that should prevent this split is missing

Root cause: No guard prevents splitting after/before "¿" in _should_end_sentence_here()
"""

import pytest

from conftest import restore_punctuation

pytestmark = pytest.mark.core


def test_pues_que_split():
    """
    Test that "Pues, ¿qué pasó, Nate?" is NOT split into two sentences.

    This is a real-world case from Episodio 190, Segment 42.
    Raw Whisper output: "Pues, ¿qué pasó, Nate?"
    Expected: Single sentence preserved
    Bug: Incorrectly splits to "Pues, ¿Qué?" | "¿Pasó, Nate?"
    """
    raw_input = "Pues, ¿qué pasó, Nate?"
    result = restore_punctuation(raw_input, language='es')

    lines = result.strip().split('\n')
    assert len(lines) == 1, \
        f"Sentence was incorrectly split into {len(lines)} lines: {lines}"

    assert 'pasó' in result and 'Nate' in result, \
        f"'pasó' and 'Nate' not found in result: {result}"

    pasa_idx = result.lower().find('pasó')
    nate_idx = result.lower().find('nate')
    distance = abs(nate_idx - pasa_idx)
    assert distance < 50, \
        f"'pasó' and 'Nate' are {distance} chars apart (expected same sentence): {result}"


def test_pues_que_split_with_context():
    """
    Test with realistic context from Episodio 190.

    The bug occurs when there's sufficient preceding text to trigger semantic splitting.
    The semantic splitter sees "¿Qué" (capitalized) and incorrectly splits there.
    """
    raw_input = """Bueno, en este episodio vamos a seguir hablando de nuestra historia pero ya más sobre la historia de Spanish Land School para aquellos que han estado escuchando nuestro podcast por un tiempo saben que nosotros hemos estado haciendo una serie de episodios sobre toda nuestra historia desde que éramos niños hasta ahora así que hoy vamos a hablar de cómo inició Spanish Land School básicamente hablaremos de los dos primeros años de Spanish Land School y les contaremos los detalles pero si tú quieres escuchar los episodios anteriores puedes ir al episodio 147, 151, 156, 164, 170, 177 y 184 el episodio más reciente sobre esta serie fue el 184 donde hicimos el capítulo 7 y hoy vamos a hacer el capítulo 8 sí, nuestra historia es un poco más largo que pensaste, ¿cierto, Andrea? Pues es que dimos muchos detalles, quizás no deberíamos haber dado tantos sí, bueno, es que queremos que ustedes aprendan con nuestras historias y algunos tienen preguntas de quién somos o quiénes son las personas que están haciendo este podcast exacto, entonces si tienes curiosidad y quieres escuchar conversaciones naturales, pues puedes ir a escuchar esos episodios que acabé de nombrar bueno, entonces, en el episodio pasado, ¿en dónde quedamos? Nosotros nos casamos en Colombia y tuvimos una ceremonia religiosa allá luego fuimos una semana a México de luna de miel y luego fuimos a California donde los papás de Nate y pasamos allá la Navidad de ese año 2016 y en Estados Unidos fue donde nos casamos por lo legal así que hablemos de lo que pasó después en el año 2017 Pues, ¿qué pasó, Nate?"""

    result = restore_punctuation(raw_input, language='es')
    sentences = result.strip().split('\n')

    pues_sentence = None
    paso_sentence = None

    for i, s in enumerate(sentences):
        s_lower = s.lower()
        if 'pues' in s_lower and '¿' in s:
            pues_sentence = (i, s)
        if 'pasó' in s_lower and 'nate' in s_lower:
            paso_sentence = (i, s)

    assert pues_sentence is not None, f"Could not find 'Pues' + '¿' in any sentence: {sentences}"
    assert paso_sentence is not None, f"Could not find 'pasó' + 'Nate' in any sentence: {sentences}"
    assert pues_sentence[0] == paso_sentence[0], (
        f"'Pues, ¿qué pasó, Nate?' was split across sentences: "
        f"'Pues' in sentence {pues_sentence[0]}: '{pues_sentence[1]}', "
        f"'pasó, Nate' in sentence {paso_sentence[0]}: '{paso_sentence[1]}'"
    )


@pytest.mark.parametrize("input_text,desc", [
    ("Entonces, ¿qué hiciste ayer?", "Question after 'Entonces'"),
    ("Bueno, ¿cómo estuvo tu día?", "Question after 'Bueno'"),
    ("Pues, ¿dónde vives ahora?", "Question after 'Pues'"),
    ("Y, ¿cuándo vas a venir?", "Question after 'Y'"),
    ("Así que, ¿qué pasó después?", "Question after 'Así que'"),
])
def test_similar_patterns(input_text, desc):
    """Test similar patterns that should NOT be split."""
    result = restore_punctuation(input_text, language='es')
    lines = result.strip().split('\n')
    assert len(lines) == 1, \
        f"[{desc}] Input '{input_text}' was split into {len(lines)} lines: {lines}"
