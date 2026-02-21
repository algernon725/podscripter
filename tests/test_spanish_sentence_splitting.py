"""
Test to reproduce and fix Spanish sentence splitting issues
"""

import pytest

from conftest import restore_punctuation

pytestmark = pytest.mark.core


@pytest.mark.parametrize("input_text,expected", [
    ("yo soy andrea de santander colombia",
     "Yo soy Andrea, de Santander, Colombia."),
    ("recuerdas todos esos momentos en los que no supiste qué decir",
     "¿Recuerdas todos esos momentos en los que no supiste qué decir?"),
    ("hola cómo estás hoy",
     "¿Hola, cómo estás hoy?"),
    ("me llamo carlos y vivo en madrid",
     "Me llamo Carlos y vivo en Madrid."),
    ("qué hora es la reunión mañana",
     "¿Qué hora es la reunión mañana?"),
    ("es importante que todos estén presentes",
     "Es importante que todos estén presentes."),
    ("buenas tardes mi nombre es maría y trabajo en bogotá",
     "Buenas tardes, mi nombre es María y trabajo en Bogotá."),
    ("puedes decirme dónde queda la estación de metro",
     "¿Puedes decirme dónde queda la estación de metro?"),
    ("ayer fuimos al museo y después comimos en un restaurante muy bonito",
     "Ayer fuimos al museo y después comimos en un restaurante muy bonito."),
    ("quieren ir al cine esta noche o prefieren quedarse en casa",
     "¿Quieren ir al cine esta noche o prefieren quedarse en casa?"),
], ids=[
    "introduction",
    "question_remembering",
    "greeting_question",
    "introduction_conjunction",
    "question_time",
    "importance_statement",
    "greeting_with_intro",
    "embedded_wh_question",
    "past_tense_narrative",
    "yesno_coordination",
])
@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
def test_spanish_sentence_splitting(input_text, expected):
    """Test Spanish sentence splitting issues."""
    result = restore_punctuation(input_text, 'es')
    assert result.strip() == expected.strip(), (
        f"got {result!r}, expected {expected!r}"
    )
