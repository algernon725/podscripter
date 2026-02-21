"""
MIT License

Copyright (c) 2025 Algernon Greenidge

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
Test script for sentence transformer punctuation restoration functionality
"""

import pytest

from sentence_transformers import SentenceTransformer
from conftest import restore_punctuation

pytestmark = pytest.mark.core


@pytest.mark.parametrize("lang,input_text,description", [
    (
        'en',
        "hello how are you today I hope you are doing well thank you",
        'English basic conversation',
    ),
    (
        'es',
        "hola como estas hoy espero que estes bien gracias",
        'Spanish basic conversation',
    ),
    (
        'de',
        "hallo wie geht es dir heute ich hoffe es geht dir gut danke",
        'German basic conversation',
    ),
    (
        'fr',
        "bonjour comment allez vous aujourd'hui j'espere que vous allez bien merci",
        'French basic conversation',
    ),
])
def test_advanced_punctuation(lang, input_text, description):
    """Test that punctuation restoration produces output with terminal punctuation."""
    result = restore_punctuation(input_text, lang)
    assert result and result.strip(), f"[{description}] Empty result for input: {input_text}"
    assert result.strip()[-1] in '.!?', \
        f"[{description}] Result doesn't end with punctuation: {result}"
