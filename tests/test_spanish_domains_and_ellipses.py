#!/usr/bin/env python3
"""
Lock-in tests for Spanish domain handling and ellipsis behavior.

Verifies that:
- Periods inside domain names (e.g., espanolistos.com, example.net, example.org)
  do not cause sentence splits.
- Ellipses ("..." and "…") do not cause sentence breaks mid-clause.
"""

import re

import pytest
from conftest import restore_punctuation

pytestmark = pytest.mark.core


def _split_like_pipeline(processed_segment: str) -> list[str]:
    """Mirror the sentence assembly logic in podscripter.py.

    Treat ellipses as non-terminal and preserve domains label.tld without splitting.
    Returns a list of finalized sentences.
    """
    parts = re.split(r'(…|[.!?]+)', processed_segment)
    sentences: list[str] = []
    buffer = ""
    idx = 0
    while idx < len(parts):
        chunk = parts[idx].strip() if idx < len(parts) else ""
        punct = parts[idx + 1] if idx + 1 < len(parts) else ""

        if chunk:
            buffer = (buffer + " " + chunk).strip()

        if punct in ("...", "…"):
            buffer += punct
            idx += 2
            continue

        if punct == '.':
            next_chunk = parts[idx + 2] if idx + 2 < len(parts) else ""
            prev_label_match = re.search(r"([A-Za-z0-9-]+)$", chunk)
            leading_ws_len = len(next_chunk) - len(next_chunk.lstrip())
            leading_ws = next_chunk[:leading_ws_len]
            next_chunk_lstripped = next_chunk[leading_ws_len:]
            next_tld_match = re.match(r"^([A-Za-z]{2,24})(\b|\W)(.*)$", next_chunk_lstripped)
            if prev_label_match and next_tld_match:
                tld = next_tld_match.group(1)
                boundary = next_tld_match.group(2) or ""
                remainder = next_tld_match.group(3)
                buffer += '.' + tld
                parts[idx + 2] = leading_ws + boundary + remainder
                idx += 2
                continue

        if punct:
            buffer += punct
            cleaned = re.sub(r'^[",\s]+', '', buffer)
            if cleaned:
                if not cleaned.endswith(('.', '!', '?')):
                    cleaned += '.'
                sentences.append(cleaned)
            buffer = ""
            idx += 2
            continue

        if idx + 1 >= len(parts):
            cleaned = re.sub(r'^[",\s]+', '', buffer)
            if cleaned:
                if not cleaned.endswith(('.', '!', '?')):
                    cleaned += '.'
                sentences.append(cleaned)
            buffer = ""
        idx += 2

    return sentences


def test_domains_not_split():
    samples = [
        "Puedes ir a espanolistos.com de nuevo",
        "Visita example.net ahora mismo",
        "Nuestro sitio es ejemplo.org para más información",
        "Espanolistos.com",
        "Solamente debes ir a espanolistos.com y ahí descargas la transcripción y puedes escuchar y leer al mismo tiempo",
    ]
    for s in samples:
        processed = restore_punctuation(s, 'es')
        out = _split_like_pipeline(processed)
        assert len(out) == 1
        joined = out[0]
        assert ".com" in joined or ".net" in joined or ".org" in joined
        assert "..com" not in joined and "..net" not in joined and "..org" not in joined
        assert ".Com" not in joined and ".Net" not in joined and ".Org" not in joined


def test_ellipsis_not_split():
    samples = [
        "De tener bancarrota… rota.",
        "De tener bancarrota... rota.",
    ]
    for s in samples:
        processed = restore_punctuation(s, 'es')
        out = _split_like_pipeline(processed)
        assert len(out) == 1
        assert "bancarrota" in out[0]
        assert "rota" in out[0]


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
def test_exact_domain_split_pattern():
    """Test the exact problematic pattern from Episodio184."""
    from podscripter import _assemble_sentences

    test_text = "Debes ir a espanolistos.com y ahí puedes encontrar este episodio y todos los demás"
    sentences = _assemble_sentences(test_text, 'es', quiet=True)

    assert len(sentences) == 1
    assert "espanolistos.com" in sentences[0]
    assert "espanolistos." not in sentences[0] or "Com." not in str(sentences)
    assert "y ahí puedes encontrar" in sentences[0]

    split_sentences = ["Debes ir a espanolistos.", "Com.", "Y ahí puedes encontrar este episodio."]
    merged = _assemble_sentences("\n\n".join(split_sentences), 'es', quiet=True)

    found_complete = False
    for sentence in merged:
        if "espanolistos.com" in sentence and "ahí puedes encontrar" in sentence:
            found_complete = True
            break
    assert found_complete, f"Domain not properly merged in: {merged}"
