#!/usr/bin/env python3
# Minimal Spanish samples for unit testing (no external media files).

# Simulated raw ASR segments (lowercased, unpunctuated, minimal diacritics)
SPANISH_ASR_SEGMENTS = [
    (
        "hola a todos bienvenidos a espanolistos espanolistos es el podcast que te va a ayudar "
        "a estar listo para hablar espanol espanolistos te prepara para hablar espanol en cualquier lugar "
        "a cualquier hora y en cualquier situacion"
    ),
    (
        "recuerdas todos esos momentos en los que no supiste que decir esos momentos en los que no pudiste "
        "mantener una conversacion pues tranquilo espanolistos es la herramienta que estabas buscando para mejorar tu espanol "
        "dile adios a todos esos momentos incomodos entonces empecemos estamos listos yo soy andrea de santander colombia "
        "y yo soy nate de texas estados unidos"
    ),
]


# Minimal human-verified reference excerpt for scoring
HUMAN_REFERENCE_TEXT = (
    "Hola a todos, ¡bienvenidos a Españolistos!\n\n"
    "Españolistos es el Podcast que te va a ayudar a estar listo para hablar español. "
    "Españolistos te prepara para hablar español en cualquier lugar, a cualquier hora y en cualquier situación.\n\n"
    "¿Recuerdas todos esos momentos en los que no supiste qué decir?\n"
    "¿Esos momentos en los que no pudiste mantener una conversación?\n"
    "Pues tranquilo, Españolistos es la herramienta que estabas buscando para mejorar tu español.\n"
    "Dile adiós a todos esos momentos incómodos. Entonces, ¡empecemos!\n"
    "¿Estamos listos?\n\n"
    "Yo soy Andrea, de Santander, Colombia. Y yo soy Nate, de Texas, Estados Unidos."
)


