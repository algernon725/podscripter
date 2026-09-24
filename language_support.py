#!/usr/bin/env python3
"""
Which languages get language-specific processing.

podscripter has two processing modes:

  * **Tailored** languages have a spaCy model installed (baked into the Docker
    image) and the per-language rules that go with it: question/exclamation
    detection, sentence-splitting guards, domain rejoining, capitalization.
  * **Generic** languages get Whisper's own text laid out into sentences and
    paragraphs, with every language-specific heuristic switched off. Before
    v0.14.0 they silently received the *English* rules instead, which corrupted
    output Whisper had already produced correctly.

The tailored set is derived at runtime from the installed spaCy models, so the
Dockerfile is the single source of truth: installing `it_core_news_sm` would
promote Italian to tailored without a code change here.

Leaf module: no project imports, and spaCy / faster-whisper are imported lazily
inside functions, so `domain_utils` can depend on this module without gaining a
spaCy import at module load.
"""

import re
from functools import lru_cache

# spaCy's trained pipelines are named "<lang>_core_<genre>_<size>".
_SPACY_CORE_MODEL_RE = re.compile(r"^([a-z]{2,3})_core_")


@lru_cache(maxsize=1)
def _installed_spacy_models() -> dict[str, str]:
    """Map language code -> installed spaCy core model name.

    When several models exist for one language, the alphabetically first name
    wins, so the choice is deterministic.
    """
    try:
        import spacy.util
        names = spacy.util.get_installed_models()
    except Exception:
        return {}
    models: dict[str, str] = {}
    for name in sorted(names):
        m = _SPACY_CORE_MODEL_RE.match(name)
        if m:
            models.setdefault(m.group(1), name)
    return models


def tailored_languages() -> frozenset[str]:
    """Language codes that get language-specific processing."""
    return frozenset(_installed_spacy_models())


def is_tailored(language: str | None) -> bool:
    """True if `language` gets language-specific processing.

    The single predicate for "is this language supported". Replaces the three
    notions that existed before v0.14.0 (FOCUS_LANGS, get_supported_languages()
    and scattered `== 'es'/'en'/...` literals).
    """
    return (language or "").lower() in _installed_spacy_models()


def spacy_model_name(language: str | None) -> str | None:
    """The installed spaCy model for `language`, or None for a generic language."""
    return _installed_spacy_models().get((language or "").lower())


@lru_cache(maxsize=1)
def whisper_languages() -> frozenset[str] | None:
    """Language codes Whisper accepts, or None if they cannot be determined.

    Reads faster-whisper's `_LANGUAGE_CODES`, which is a *private* symbol; if a
    future release removes it we accept any code rather than reject valid ones
    (Whisper itself will still reject a bad code, just later). A test pins that
    the import still resolves.
    """
    try:
        from faster_whisper.tokenizer import _LANGUAGE_CODES
    except Exception:
        return None
    return frozenset(_LANGUAGE_CODES)


def is_whisper_language(language: str) -> bool:
    """True if Whisper accepts `language` (permissive when that is unknowable)."""
    codes = whisper_languages()
    return codes is None or language in codes
