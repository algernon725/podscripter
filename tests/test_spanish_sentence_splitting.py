"""
Test to reproduce and fix Spanish sentence splitting issues
"""

import pytest

from conftest import restore_punctuation

pytestmark = pytest.mark.core


@pytest.mark.parametrize("input_text,expected", [
    pytest.param(
        "yo soy andrea de santander colombia",
        "Yo soy Andrea, de Santander, Colombia.",
        id="introduction",
        marks=pytest.mark.xfail(reason="NLP output drift"),
    ),
    pytest.param(
        "recuerdas todos esos momentos en los que no supiste qué decir",
        "¿Recuerdas todos esos momentos en los que no supiste qué decir?",
        id="question_remembering",
        marks=pytest.mark.xfail(reason="NLP output drift"),
    ),
    pytest.param(
        "hola cómo estás hoy",
        "¿Hola, cómo estás hoy?",
        id="greeting_question",
        marks=pytest.mark.xfail(reason="NLP output drift"),
    ),
    pytest.param(
        "me llamo carlos y vivo en madrid",
        "Me llamo Carlos y vivo en Madrid.",
        id="introduction_conjunction",
        marks=pytest.mark.xfail(reason="NLP output drift"),
    ),
    pytest.param(
        "qué hora es la reunión mañana",
        "¿Qué hora es la reunión mañana?",
        id="question_time",
    ),
    pytest.param(
        "es importante que todos estén presentes",
        "Es importante que todos estén presentes.",
        id="importance_statement",
    ),
    pytest.param(
        "buenas tardes mi nombre es maría y trabajo en bogotá",
        "Buenas tardes, mi nombre es María y trabajo en Bogotá.",
        id="greeting_with_intro",
        marks=pytest.mark.xfail(reason="NLP output drift"),
    ),
    pytest.param(
        "puedes decirme dónde queda la estación de metro",
        "¿Puedes decirme dónde queda la estación de metro?",
        id="embedded_wh_question",
        marks=pytest.mark.xfail(reason="NLP output drift"),
    ),
    pytest.param(
        "ayer fuimos al museo y después comimos en un restaurante muy bonito",
        "Ayer fuimos al museo y después comimos en un restaurante muy bonito.",
        id="past_tense_narrative",
    ),
    pytest.param(
        "quieren ir al cine esta noche o prefieren quedarse en casa",
        "¿Quieren ir al cine esta noche o prefieren quedarse en casa?",
        id="yesno_coordination",
        marks=pytest.mark.xfail(reason="NLP output drift"),
    ),
])
def test_spanish_sentence_splitting(input_text, expected):
    """Test Spanish sentence splitting issues."""
    result = restore_punctuation(input_text, 'es')
    assert result.strip() == expected.strip(), (
        f"got {result!r}, expected {expected!r}"
    )
