"""
Test the specific 'y eco' bug from Episodio 190 using segments from actual transcription.

The bug: When processing with --single --language es, sentences split incorrectly like:
"No eran Ah sí, no eran muy, muy buenos teníamos muchos errores y." | "Eco una vez..."

This test simulates the actual segment processing that happens during transcription.
"""

import pytest

from conftest import restore_punctuation
from punctuation_restorer import assemble_sentences_from_processed

pytestmark = pytest.mark.core


def test_y_eco_from_segments():
    """Test the exact segments that cause the 'y eco' split bug."""
    segments = [
        "Ah sí, no eran muy muy buenos",
        "teníamos muchos errores y eco",
        "una vez que grabábamos el video en el garaje",
        "y había tanto eco",
    ]

    combined_text = " ".join(segments)
    restored = restore_punctuation(combined_text, language='es')
    sentences, trailing = assemble_sentences_from_processed(restored, 'es')

    for i, sent in enumerate(sentences):
        sent_stripped = sent.strip().rstrip('.!?')
        assert not (sent_stripped.endswith(' y') or sent_stripped.endswith(',y')), \
            f"Sentence {i+1} ends with 'y': '{sent}'"
        assert not sent.strip().lower().startswith('eco '), \
            f"Sentence {i+1} starts with 'Eco' (should be after 'y'): '{sent}'"

    full_text = ' '.join(sentences)
    if trailing:
        full_text += ' ' + trailing
    assert 'y eco' in full_text.lower(), f"'y eco' not found together in output: {full_text}"


def test_simpler_y_eco_case():
    """Test a simpler version to isolate the issue."""
    text = "No eran muy buenos teníamos muchos errores y eco"

    restored = restore_punctuation(text, language='es')
    sentences, trailing = assemble_sentences_from_processed(restored, 'es')

    for sent in sentences:
        assert not sent.strip().rstrip('.!?').endswith(' y'), \
            f"Found 'y' at end of sentence: '{sent}'"

    full_text = ' '.join(sentences)
    if trailing:
        full_text += ' ' + trailing
    assert 'y eco' in full_text.lower(), f"'y eco' not found together in output: {full_text}"


def test_very_long_text_with_y_eco():
    """Test with longer text to trigger min_chunk_before_split thresholds."""
    prefix = "Bueno en este episodio vamos a seguir hablando de nuestra historia pero ya más sobre la historia de Spanish Land School para aquellos que han estado escuchando nuestro podcast por un tiempo"
    problem_part = "saben que nosotros hemos estado haciendo una serie de episodios sobre toda nuestra historia desde que éramos niños hasta ahora así que hoy vamos a hablar de cómo inició Spanish Land School y básicamente hablaremos de los dos primeros años de Spanish Land School y les contaremos todos los detalles pero si tú quieres escuchar los episodios anteriores recuerdo que fue como el veintidós o veintitrés de marzo que empezamos a grabar los videos así que estaba en Colombia pero tenía su trabajo remoto y trabajaba de nueve de la mañana a cinco de la tarde y yo tenía mi universidad todavía tenía algunas clases y ahora estaba empezando a grabar estos videos así que fue una locura sí todo el mes de marzo estuvimos grabando por un mes por un mes grabamos seis videos y los editamos en el lapso de un mes porque el canal de YouTube empezó el veintitrés de abril a finales de abril sí y para los que han visto los primeros videos no eran o no fueron no eran ah sí no eran muy muy buenos teníamos muchos errores y eco"

    text = prefix + " " + problem_part
    restored = restore_punctuation(text, language='es')
    sentences, trailing = assemble_sentences_from_processed(restored, 'es')

    for i, sent in enumerate(sentences):
        sent_stripped = sent.strip().rstrip('.!?')
        assert not sent_stripped.endswith(' y'), \
            f"Sentence {i+1} ends with 'y': '...{sent[-80:]}'"
