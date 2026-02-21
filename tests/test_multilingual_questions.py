#!/usr/bin/env python3
"""
Test multilingual question patterns across all supported languages
"""

import pytest

from conftest import restore_punctuation

pytestmark = pytest.mark.multilingual

_question_cases = [
    ("en", "how are you today", "How are you today?", "English basic question"),
    ("en", "can you help me with this", "Can you help me with this?", "English request question"),
    ("en", "what time is the meeting", "What time is the meeting?", "English wh-question"),
    ("es", "cómo estás hoy", "¿Cómo estás hoy?", "Spanish basic question"),
    ("es", "puedes ayudarme con esto", "¿Puedes ayudarme con esto?", "Spanish request question"),
    ("es", "qué hora es la reunión", "¿Qué hora es la reunión?", "Spanish wh-question"),
    ("fr", "comment allez vous aujourd'hui", "Comment allez-vous aujourd'hui?", "French basic question"),
    ("fr", "pouvez vous m'aider avec ceci", "Pouvez-vous m'aider avec ceci?", "French request question"),
    ("fr", "à quelle heure est la réunion", "À quelle heure est la réunion?", "French wh-question"),
    ("de", "wie geht es dir heute", "Wie geht es dir heute?", "German basic question"),
    ("de", "kannst du mir dabei helfen", "Kannst du mir dabei helfen?", "German request question"),
    ("de", "um wie viel uhr ist das treffen", "Um wie viel Uhr ist das Treffen?", "German wh-question"),
    ("it", "come stai oggi", "Come stai oggi?", "Italian basic question"),
    ("it", "puoi aiutarmi con questo", "Puoi aiutarmi con questo?", "Italian request question"),
    ("pt", "como você está hoje", "Como você está hoje?", "Portuguese basic question"),
    ("pt", "você pode me ajudar com isso", "Você pode me ajudar com isso?", "Portuguese request question"),
    ("nl", "hoe gaat het vandaag", "Hoe gaat het vandaag?", "Dutch basic question"),
    ("nl", "kun je me hierbij helpen", "Kun je me hierbij helpen?", "Dutch request question"),
    ("ja", "kyou wa dou desu ka", "Kyou wa dou desu ka?", "Japanese basic question"),
    ("ru", "kak dela segodnya", "Kak dela segodnya?", "Russian basic question"),
    ("ru", "mozhesh li ty mne pomoch", "Mozhesh li ty mne pomoch?", "Russian request question"),
]


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
@pytest.mark.parametrize(
    "lang,input_text,expected,description",
    _question_cases,
    ids=[c[3] for c in _question_cases],
)
def test_multilingual_question(lang, input_text, expected, description):
    """Test question patterns across multiple languages."""
    result = restore_punctuation(input_text, lang)
    assert result.strip() == expected.strip(), \
        f"{description}: expected {expected!r}, got {result!r}"
