#!/usr/bin/env python3
"""
Lock-in tests for Spanish domain handling and ellipsis behavior.

Verifies that:
- Periods inside domain names (e.g., espanolistos.com, example.net, example.org)
  do not cause sentence splits.
- Ellipses ("..." and "…") do not cause sentence breaks mid-clause.
"""

import os
import sys
import re

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from punctuation_restorer import restore_punctuation  # noqa: E402


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

        # Ellipses are not sentence boundaries; keep accumulating
        if punct in ("...", "…"):
            buffer += punct
            idx += 2
            continue

        # Domain glue: label + '.' + TLD (2-24 letters)
        if punct == '.':
            next_chunk = parts[idx + 2] if idx + 2 < len(parts) else ""
            prev_label_match = re.search(r"([A-Za-z0-9-]+)$", chunk)
            # Allow leading whitespace before TLD and preserve it
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

        # Default: flush on terminal punctuation
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

        # End without explicit punctuation
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
        # Real-world sample reported in bug
        "Solamente debes ir a espanolistos.com y ahí descargas la transcripción y puedes escuchar y leer al mismo tiempo",
    ]
    for s in samples:
        processed = restore_punctuation(s, 'es')
        out = _split_like_pipeline(processed)
        # Expect exactly one sentence and the domain intact
        assert len(out) == 1
        joined = out[0]
        assert ".com" in joined or ".net" in joined or ".org" in joined
        # No duplicate dot before TLD
        assert "..com" not in joined and "..net" not in joined and "..org" not in joined
        # TLD should be lowercase
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


if __name__ == "__main__":
    # Run tests directly
    test_domains_not_split()
    test_ellipsis_not_split()
    print("All Spanish domain/ellipsis tests passed")


