#!/usr/bin/env python3
"""
Compare program output vs human-verified intro snippet for Episodio174.
"""

import os
import sys
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from punctuation_restorer import restore_punctuation

HUMAN_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'audio-files', 'Episodio174-human-verified.txt')

INTRO_LINES = 16  # compare first 16 non-empty lines (intro + greetings)

def canon(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.!?])", r"\1", s)
    return s

def test_human_vs_program_intro():
    assert os.path.exists(HUMAN_FILE), f"Missing {HUMAN_FILE}"
    with open(HUMAN_FILE, 'r', encoding='utf-8') as f:
        human_lines = [l.strip() for l in f.readlines() if l.strip()]

    human_intro = human_lines[:INTRO_LINES]

    # Build a raw program-like text by removing punctuation and accents to simulate ASR roughness
    raw_program = (
        "hola a todos bienvenidos a espanolistos\n\n"
        "espanolistos es el podcast que te va a ayudar a estar listo para hablar espanol\n\n"
        "espanolistos te prepara para hablar espanol en cualquier lugar a cualquier hora y en cualquier situacion\n\n"
        "recuerdas todos esos momentos en los que no supiste que decir\n\n"
        "esos momentos en los que no pudiste mantener una conversacion\n\n"
        "pues tranquilo espanolistos es la herramienta que estabas buscando para mejorar tu espanol\n\n"
        "dile adios a todos esos momentos incomodos\n\n"
        "entonces empecemos\n\n"
        "estamos listos\n\n"
        "yo soy andrea de santander colombia\n\n"
        "y yo soy nate de texas estados unidos\n\n"
        "hola para todos como estan\n\n"
        "ojala que esten muy muy bien\n\n"
        "que esten disfrutando su semana\n\n"
        "como siempre traemos otro episodio interesante\n\n"
    )

    program_sentences = []
    for block in [seg for seg in raw_program.split("\n\n") if seg.strip()]:
        program_sentences.append(canon(restore_punctuation(block, 'es')))

    # Canonicalize human lines too
    human_sentences = [canon(x) for x in human_intro]

    # Compare prefix-wise allowing minor stylistic differences
    mismatches = []
    for i, (h, p) in enumerate(zip(human_sentences, program_sentences), 1):
        if not p or not h:
            continue
        # require that program starts with human up to first punctuation token
        h_prefix = re.split(r"(?=[,.!?])", h)[0]
        if not p.startswith(h_prefix):
            mismatches.append((i, h, p))

    assert not mismatches, "\n" + "\n".join([
        f"Line {i}:\n  HUMAN:   {h}\n  PROGRAM: {p}" for i, h, p in mismatches
    ])


