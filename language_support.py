#!/usr/bin/env python3
"""
Which languages get language-specific processing.

podscripter has two processing modes:

  * **Tailored** languages have per-language rules in this codebase:
    question/exclamation detection, sentence-splitting guards, domain
    rejoining, formatting.
  * **Generic** languages get Whisper's own text laid out into sentences and
    paragraphs, with every language-specific heuristic switched off. Before
    v0.14.0 they silently received the *English* rules instead, which corrupted
    output Whisper had already produced correctly.

The tailored set is a static list. Adding a language means writing its rules
and adding its code to TAILORED_LANGUAGES; nothing is inferred from installed
packages. (Before v0.15.0 the set was derived from the installed spaCy models.)

Leaf module: no project imports, and faster-whisper is imported lazily inside
a function, so `domain_utils` can depend on this module cheaply.
"""

from functools import lru_cache

TAILORED_LANGUAGES: frozenset[str] = frozenset({"en", "es", "fr", "de", "pt"})


def tailored_languages() -> frozenset[str]:
    """Language codes that get language-specific processing."""
    return TAILORED_LANGUAGES


def is_tailored(language: str | None) -> bool:
    """True if `language` gets language-specific processing.

    The single predicate for "is this language supported". Replaces the three
    notions that existed before v0.14.0 (FOCUS_LANGS, get_supported_languages()
    and scattered `== 'es'/'en'/...` literals).
    """
    return (language or "").lower() in TAILORED_LANGUAGES


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
