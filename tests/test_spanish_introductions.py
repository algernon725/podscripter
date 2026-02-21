"""
Test to verify Spanish introductions and statements are not incorrectly detected as questions
"""

import pytest

from conftest import restore_punctuation

pytestmark = pytest.mark.core


@pytest.mark.parametrize("text,description", [
    ("yo soy andrea de santander colombia", 'Introduction with "yo soy"'),
    ("soy andrea de santander colombia", 'Introduction with "soy"'),
    ("mi nombre es andrea", 'Introduction with "mi nombre es"'),
    ("me llamo andrea", 'Introduction with "me llamo"'),
    ("vivo en colombia", 'Statement with "vivo en"'),
    ("trabajo en santander", 'Statement with "trabajo en"'),
    ("estoy en la oficina", 'Statement with "estoy en"'),
    ("es importante el proyecto", 'Statement with "es importante"'),
    ("está bien la reunión", 'Statement with "está bien"'),
    ("soy de colombia", 'Statement with "soy de"'),
    ("es de santander", 'Statement with "es de"'),
    ("estoy de acuerdo", 'Statement with "estoy de"'),
])
def test_spanish_introductions(text, description):
    """Spanish introductions and statements should NOT be detected as questions."""
    result = restore_punctuation(text, 'es')
    assert '?' not in result, f"{description}: unexpected question mark in {result!r}"
