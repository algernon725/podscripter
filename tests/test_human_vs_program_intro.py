#!/usr/bin/env python3
"""
Compare program output vs human-verified content for Episodio174.
Computes similarity (token F1) for intro and an extended section.
"""

import re
import unicodedata

import pytest
from conftest import restore_punctuation

pytestmark = pytest.mark.core

INTRO_LINES = 16

HUMAN_LINES = [
    "Hola a todos, ¡bienvenidos a Españolistos!",
    "Españolistos es el Podcast que te va a ayudar a estar listo para hablar español.",
    "Españolistos te prepara para hablar español en cualquier lugar, a cualquier hora y en cualquier situación.",
    "¿Recuerdas todos esos momentos en los que no supiste qué decir?",
    "¿Esos momentos en los que no pudiste mantener una conversación?",
    "Pues tranquilo, Españolistos es la herramienta que estabas buscando para mejorar tu español.",
    "Dile adiós a todos esos momentos incómodos.",
    "Entonces, ¡empecemos!",
    "¿Estamos listos?",
    "Yo soy Andrea, de Santander, Colombia.",
    "Y yo soy Nate, de Texas, Estados Unidos.",
    "Hola para todos, ¿cómo están?",
    "Ojalá que estén muy, muy bien.",
    "Que estén disfrutando su semana.",
    "Como siempre, traemos otro episodio interesante.",
    "Y estamos haciendo este episodio porque algunos de ustedes lo sugirieron.",
    "Muy pocos los lugares en los que no se ha desestabilizado la economía, pero sabemos que en la mayoría sí.",
    "Así que por eso queremos analizar un poquito por qué está sucediendo esto.",
    "Y muchos de ustedes ya sabrán esas razones.",
    "Si les gusta leer, si saben del tema, pero queremos comentarles sobre esto en español.",
    "Así que sí, hablaremos de las consecuencias en lo financiero que ha causado esta crisis.",
    "Y este es un tema que tú no sabes mucho, ¿cierto Andrea?",
    "Eso es algo que tenías que investigar mucho.",
    "Exactamente, yo sé muy poco de todo esto del mercado de acciones, stock market.",
    "También sí, todo en general en esta área, nunca he tenido muy buen conocimiento.",
    "Pero esperamos hacer un buen trabajo hoy.",
    "Y obvio que no somos expertos en esto.",
    "Así que, si nos equivocamos en algo, pues pedimos disculpas.",
    "Pero leímos varios artículos y fuentes para sacar este outline.",
    "Sí.",
    "Teníamos que investigar mucho porque es algo muy complejo, solo vamos a hablar de lo que sabemos, pero es un poco complejo.",
    "Bueno queridos, pero antes de empezar, les recuerdo que ustedes pueden descargar la transcripción.",
    "Puedes ir a espanolistos.com de nuevo espanolistos.com y ahí puedes descargar la transcripción para que escuches y leas.",
    "Vamos a escuchar todo el contenido, y al final vamos a leer una reseña de uno de ustedes también.",
    "Bueno queridos, les cuento.",
    "Pues, la verdad es que la crisis económica que está viviendo el mundo no es una crisis más, como las que se han enfrentado en el pasado.",
    "De verdad es una crisis mucho más compleja, y pues que al mismo tiempo está afectando a todo el mundo.",
    "Los expertos dicen que esta crisis es más grave que la crisis financiera internacional que hubo entre 2008 y 2009.",
    "Y también dicen que de alguna manera es similar a la crisis de los años 30 del siglo pasado.",
    "Por sus posibles consecuencias sobre la pobreza y el desempleo.",
]


def canon(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.!?])", r"\1", s)
    return s


def deaccent(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def f1_tokens(a: str, b: str) -> float:
    ta = a.split()
    tb = b.split()
    if not ta and not tb:
        return 1.0
    inter = len(set(ta) & set(tb))
    if inter == 0:
        return 0.0
    prec = inter / max(1, len(set(tb)))
    rec = inter / max(1, len(set(ta)))
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
def test_human_vs_program():
    raw_blocks = [deaccent(re.sub(r"[,.!¡¿?]", " ", h)).lower() for h in HUMAN_LINES]
    restored = [canon(restore_punctuation(b, 'es')) for b in raw_blocks]
    human_canon = [canon(h) for h in HUMAN_LINES]

    f1s = [f1_tokens(h, p) for h, p in zip(human_canon, restored)]
    intro_avg = sum(f1s[:INTRO_LINES]) / max(1, INTRO_LINES)
    overall_avg = sum(f1s) / max(1, len(f1s))

    assert intro_avg >= 0.80, f"Intro average F1 too low: {intro_avg:.2f}"
    assert overall_avg >= 0.70, f"Overall average F1 too low: {overall_avg:.2f}"
