#!/usr/bin/env python3
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
Punctuation restoration module for multilingual text processing.
Supports English, Spanish, French, and German with advanced NLP techniques.

NOTE: Sentence splitting logic has been consolidated into sentence_splitter.py (v0.4.0).
This module now focuses on punctuation restoration and language-specific formatting.
"""

# Per-language tuning guide (constants and thresholds)
# -----------------------------------------------------
# Where to adjust behavior without touching core logic:
#
# - Thresholds (splitting and semantic gating):
#   * Function: _get_language_thresholds(language)
#   * Wrapper: LanguageConfig (get_language_config)
#   * Used by: should_end_sentence_here, is_question_semantic
#   * Keys (es):
#       - semantic_question_threshold_with_indicator
#       - semantic_question_threshold_default
#       - min_total_words_no_split
#       - min_chunk_before_split
#       - min_chunk_inside_question
#       - min_chunk_capital_break
#       - min_chunk_semantic_break
#       - semantic_whisper_lookahead
#
# - Spanish keyword/constants:
#   * ES_QUESTION_WORDS_CORE, ES_QUESTION_STARTERS_EXTRA
#   * ES_GREETINGS
#   * Spanish helpers live in functions prefixed with _es_ (e.g., _es_greeting_and_leadin_commas)
#
# - French/German keyword/constants:
#   * FR_GREETINGS, DE_GREETINGS
#   * FR_QUESTION_STARTERS, DE_QUESTION_STARTERS
#
# - English question starters:
#   * EN_QUESTION_STARTERS
#
# - Shared utilities:
#   * split_sentences_preserving_delims(): consistent splitting
#   * normalize_mixed_terminal_punctuation(): final punctuation cleanup
#
# After changes, run the test suite (tests/run_all_tests.py). All constants are
# centralized here to keep tuning safe and maintainable across languages.

import re
import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import os
import numpy as np
import warnings
from domain_utils import (
    mask_domains,
    unmask_domains,
    SINGLE_TLDS,
    SINGLE_TLDS_CONSERVATIVE,
    SINGLE_MASK,
    UPPER_ACCENTED,
    LOWER_ACCENTED,
)
from sentence_splitter import SentenceSplitter, Sentence
from language_support import is_tailored

# Suppress PyTorch FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Module logger for library-friendly messaging
logger = logging.getLogger("podscripter.punctuation")

__all__ = [
    "restore_punctuation",
    "assemble_sentences_from_processed",
]

# Try to import sentence transformers for better punctuation restoration
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    cosine_similarity = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("SentenceTransformers not available. Advanced punctuation restoration may be limited.")


_SENTENCE_TRANSFORMER_SINGLETON = None

"""
Lightweight utilities and caches
"""

# Precompiled, shared regexes
PUNCT_SPLIT_RE = re.compile(r'([.!?]+)')

# Spanish keywords for questions and greetings (centralized)
ES_QUESTION_WORDS_CORE = ['qué', 'dónde', 'cuándo', 'cómo', 'quién', 'cuál', 'por qué']
ES_QUESTION_STARTERS_EXTRA = ['recuerdas', 'sabes', 'puedes', 'puede', 'podrías', 'podría',
                              'quieres', 'quiere', 'quieren', 'necesitas', 'necesita', 'hay',
                              'estás', 'están', 'es', 'son', 'vas', 'va', 'tienes', 'tiene']
ES_GREETINGS = ['hola', 'buenos días', 'buenas tardes', 'buenas noches']

# French/German greeting starters (for consistency and future tuning)
FR_GREETINGS = ['bonjour']
DE_GREETINGS = ['hallo']

# French/German question starters (heuristic, used in light formatter)
FR_QUESTION_STARTERS = ['comment', 'où', 'quand', 'pourquoi', 'qui', 'quel', 'quelle', 'quels', 'quelles', 'est-ce que']
DE_QUESTION_STARTERS = [
    # wh- and copula
    'wie', 'wo', 'wann', 'warum', 'wer', 'welche', 'welches', 'welcher', 'ist', 'sind', 'seid',
    # modal/auxiliary starts
    'kann', 'kannst', 'können', 'könnt', 'möchte', 'möchtest', 'möchten', 'will', 'willst', 'wollen',
    'soll', 'sollst', 'sollen', 'sollt', 'darf', 'darfst', 'dürfen', 'dürft',
    'hast', 'hat', 'habe', 'haben', 'hatten', 'hatte', 'war', 'waren',
    'wird', 'werden', 'gibt es'
]

# English question starters (for completeness in light formatter)
EN_QUESTION_STARTERS = ['what', 'where', 'when', 'why', 'how', 'who', 'which', 'do', 'does', 'did', 'is', 'are', 'can', 'could', 'would', 'will', 'am']

# Portuguese keywords. Cover both Brazilian and European Portuguese (Whisper reports a single 'pt').
PT_QUESTION_WORDS_CORE = ['que', 'o que', 'quê', 'onde', 'quando', 'como', 'quem',
                          'qual', 'quais', 'por que', 'porquê', 'porque']
PT_QUESTION_STARTERS_EXTRA = ['pode', 'podes', 'podem', 'sabe', 'sabes', 'quer', 'queres',
                              'tem', 'tens', 'têm', 'há', 'está', 'estão', 'é', 'são',
                              'vai', 'vais', 'consegue', 'consegues', 'lembra', 'lembras',
                              'precisa', 'precisas', 'gostaria', 'poderia', 'poderias']
PT_GREETINGS = ['olá', 'oi', 'bom dia', 'boa tarde', 'boa noite']

# Language thresholds (centralized for tuning)

# Baseline used by every language; per-language deltas live in _THRESHOLD_OVERRIDES.
_BASE_THRESHOLDS = {
    'semantic_question_threshold_default_any': 0.60,
    'min_total_words_no_split': 25,
    'min_chunk_before_split': 15,
    'min_chunk_inside_question': 20,
    'min_chunk_capital_break': 20,
    'min_chunk_semantic_break': 25,
    # Whisper boundary integration thresholds
    'min_words_whisper_break': 10,  # Minimum words before honoring Whisper boundary
    'max_words_force_split': 100,   # Force split on very long segments even without boundary
    'semantic_whisper_lookahead': 8,  # Defer semantic split if Whisper boundary is within N words
}

# Romance languages run longer clauses with heavier subordination than the
# Germanic/English baseline, so they need a larger chunk before a split is
# considered. Shared by 'es' and 'pt'.
_ROMANCE_SPLIT_THRESHOLDS = {
    'min_total_words_no_split': 30,
    'min_chunk_before_split': 20,
    'min_chunk_inside_question': 25,
    'min_chunk_capital_break': 38,
    'min_chunk_semantic_break': 42,
}

_THRESHOLD_OVERRIDES = {
    'es': {
        **_ROMANCE_SPLIT_THRESHOLDS,
        # Spanish selects between two question thresholds depending on whether the
        # sentence opens with a question indicator (see is_question_semantic).
        # Slightly lower than the generic threshold to improve recall of genuine questions.
        'semantic_question_threshold_with_indicator': 0.64,
        'semantic_question_threshold_default': 0.74,
    },
    # Portuguese shares the Romance split profile but keeps the generic single
    # question threshold: it has no inverted '¿', so the indicator-sensitive
    # two-threshold path that is gated on language == 'es' never runs for 'pt'.
    'pt': dict(_ROMANCE_SPLIT_THRESHOLDS),
}


def _get_language_thresholds(language: str) -> dict:
    """Return thresholds controlling semantic gating and splitting heuristics."""
    return {**_BASE_THRESHOLDS, **_THRESHOLD_OVERRIDES.get(language, {})}


@dataclass
class LanguageConfig:
    thresholds: dict
    greetings: list
    question_starters: list


def _get_language_config(language: str) -> LanguageConfig:
    if language == 'es':
        return LanguageConfig(
            thresholds=_get_language_thresholds(language),
            greetings=ES_GREETINGS,
            question_starters=ES_QUESTION_WORDS_CORE + ES_QUESTION_STARTERS_EXTRA,
        )
    if language == 'fr':
        return LanguageConfig(
            thresholds=_get_language_thresholds(language),
            greetings=FR_GREETINGS,
            question_starters=FR_QUESTION_STARTERS,
        )
    if language == 'de':
        return LanguageConfig(
            thresholds=_get_language_thresholds(language),
            greetings=DE_GREETINGS,
            question_starters=DE_QUESTION_STARTERS,
        )
    if language == 'en':
        return LanguageConfig(
            thresholds=_get_language_thresholds(language),
            greetings=['hello'],
            question_starters=EN_QUESTION_STARTERS,
        )
    if language == 'pt':
        return LanguageConfig(
            thresholds=_get_language_thresholds(language),
            greetings=PT_GREETINGS,
            question_starters=PT_QUESTION_WORDS_CORE + PT_QUESTION_STARTERS_EXTRA,
        )
    return LanguageConfig(
        thresholds=_get_language_thresholds(language),
        greetings=[],
        question_starters=[],
    )

def _split_sentences_preserving_delims(text: str) -> list:
    """Split text into [chunk, delimiter, chunk, delimiter, ...] using common punctuation.
    Returns a list where even indices are text chunks and odd indices are delimiters.
    """
    return PUNCT_SPLIT_RE.split(text)


def _normalize_mixed_terminal_punctuation(text: str) -> str:
    """Normalize mixed terminal punctuation like '?.', '!.', '!?'.

    Rules:
    - Collapse mixed pairs like .? or ?.
    - For exclamations/questions, collapse runs (e.g., '!!!' -> '!').
    - Preserve ellipses: keep '…' as-is and keep '...' as '...'.
      Only reduce sequences of four or more dots to '...'.
      Reduce exactly two dots to one dot.
    Safe to run multiple times.
    """
    out = text
    # Mixed pairs
    out = re.sub(r"\.\s*\?", "?", out)   # .? -> ?
    out = re.sub(r"\?\s*\.", "?", out)    # ?. -> ?
    out = re.sub(r"!\s*\.", "!", out)      # !. -> !
    out = re.sub(r"!\s*\?", "!", out)      # !? -> !
    out = re.sub(r"\?\s*!", "!", out)      # ?! -> !
    # Preserve ellipses
    out = re.sub(r"\.{4,}", "...", out)     # 4+ dots -> ...
    out = re.sub(r"(?<!\.)\.\.(?!\.)", ".", out)  # exactly two dots -> one
    # Collapse runs of question/exclamation
    out = re.sub(r"([!?]){2,}", r"\1", out)
    return out


def _fix_location_appositive_punctuation(text: str, language: str) -> str:
    """Fix incorrect periods in location appositives across languages.
    
    Handles two patterns:
    1. Preposition-based: ", de Texas. Estados Unidos" -> ", de Texas, Estados Unidos"
    2. Direct comma-separated: "Austin, Texas. Y allá" -> "Austin, Texas y allá"
    
    Args:
        text: Text that may contain incorrect location appositive punctuation
        language: Language code (es, en, fr, de)
        
    Returns:
        Text with corrected location appositive punctuation
        
    Examples:
        "I'm John, from Texas. United States." -> "I'm John, from Texas, United States."
        "Soy Juan, de Texas. Estados Unidos." -> "Soy Juan, de Texas, Estados Unidos."
        "Living in Austin, Texas. And working there." -> "Living in Austin, Texas and working there."
    """
    if not text or not language:
        return text
        
    # Define location prepositions by language (same as in TXT writer)
    location_prepositions = {
        'es': r'de',
        'en': r'from|in',
        'fr': r'de|du|des',
        'de': r'aus|von|in',
        'pt': r'de|do|da|dos|das|em'
    }
    
    lang_code = language.lower()
    prepositions = location_prepositions.get(lang_code, r'de|from|aus|von|in|du|des')
    
    # Pattern 1: comma + preposition + location + period + location
    # Convert the period to a comma for proper appositive punctuation
    # But exclude cases that start new sentences with subjects like "Y yo soy", "And I'm", etc.
    pattern1 = rf'(,\s*(?:{prepositions})\s+[A-Z{UPPER_ACCENTED}][\w{UPPER_ACCENTED}-]*)\.\s+([A-Z{UPPER_ACCENTED}][\w{UPPER_ACCENTED}-]*)'
    
    # Check if the following text starts a new sentence with a subject
    def _safe_location_merge_preposition(match):
        prefix = match.group(1)
        following = match.group(2)
        
        # Don't merge if following text looks like start of new sentence with subject
        # Common patterns: "Y yo", "And I", "Et je", "Und ich", etc.
        new_sentence_patterns = [
            r'^(Y|E|And|Et|Und)\s+(yo|eu|I|je|ich)',  # "Y yo", "E eu", "And I", "Et je", "Und ich"
            r'^(Y|E|And|Et|Und)\s+\w+\s+(soy|sou|é|am|suis|bin)',  # "Y alguien soy", "E alguém é"
        ]
        
        for pattern in new_sentence_patterns:
            if re.match(pattern, following, re.IGNORECASE):
                return match.group(0)  # Return unchanged - don't merge
        
        return f"{prefix}, {following}"
    
    result = re.sub(pattern1, _safe_location_merge_preposition, text, flags=re.IGNORECASE)
    
    # Pattern 2: Direct comma-separated locations (City, State/Country pattern)
    # Handle cases like "Austin, Texas. Y" -> "Austin, Texas y" but avoid merging new sentences
    def _safe_location_merge_direct(match):
        location_part = match.group(1)
        following_word = match.group(2)
        
        # Don't merge if following text starts a new sentence with subject
        new_sentence_patterns = [
            r'^(Y|E|And|Et|Und)\s+(yo|eu|I|je|ich)',  # "Y yo", "E eu", "And I", "Et je", "Und ich"
            r'^(Y|E|And|Et|Und)\s+\w+\s+(soy|sou|é|am|suis|bin)',  # "Y alguien soy", "E alguém é"
        ]
        
        # Check the full following context (might be more than one word)
        rest_of_text = text[match.end():]
        full_following = following_word + " " + rest_of_text.split('.')[0][:50]  # Check first 50 chars
        
        for pattern in new_sentence_patterns:
            if re.match(pattern, full_following, re.IGNORECASE):
                return match.group(0)  # Return unchanged - don't merge
        
        return f"{location_part} {following_word.lower()}"
    
    pattern2 = rf'(\b[A-Z{UPPER_ACCENTED}][\w{UPPER_ACCENTED}-]*,\s+[A-Z{UPPER_ACCENTED}][\w{UPPER_ACCENTED}-]*)\.\s+([A-Z{UPPER_ACCENTED}a-z{LOWER_ACCENTED}][\w{UPPER_ACCENTED}-]*)'
    result = re.sub(pattern2, _safe_location_merge_direct, result, flags=re.IGNORECASE)
    
    return result


def _normalize_comma_spacing(text: str) -> str:
    """Normalize comma spacing in text.
    
    This function:
    1. Removes spaces before commas
    2. Deduplicates multiple commas
    3. Adds space after all commas
    
    Trade-off: Thousands separators like "1,000" become "1, 000".
    This is acceptable because:
    - Number lists (episode numbers, dates) are more common in transcriptions
    - "1, 000" is still understandable
    - The alternative (trying to detect thousands) caused false positives
    
    Examples:
        >>> _normalize_comma_spacing("episodio 147,151,156")
        "episodio 147, 151, 156"
        >>> _normalize_comma_spacing("hay 1,000 personas")
        "hay 1, 000 personas"
        >>> _normalize_comma_spacing("palabra ,otra")
        "palabra, otra"
        >>> _normalize_comma_spacing("test, ,doble")
        "test, doble"
    
    Args:
        text: Input text with potentially inconsistent comma spacing
        
    Returns:
        Text with normalized comma spacing
    """
    if not text:
        return text if text is not None else ""
    
    # 1) Remove spaces before commas everywhere
    text = re.sub(r"\s+,", ",", text)
    
    # 2) Deduplicate accidental double commas (allowing optional spaces between)
    # e.g., ", ," -> ", " or ",,," -> ", "
    text = re.sub(r",\s*,+", ", ", text)
    
    # 3) Normalize space after commas: ensure exactly one space (or none if at end)
    # First, normalize any existing spaces after commas
    text = re.sub(r",\s+", ", ", text)
    # Then add space where missing (when followed by non-whitespace)
    text = re.sub(r",(?=\S)", ", ", text)
    
    return text


# Final universal cleanup applied at the end of the pipeline
def _finalize_text_common(text: str, language: str | None = None) -> str:
    """Apply safe, language-agnostic cleanup at the very end.

    - Normalize mixed terminal punctuation
    - Normalize whitespace
    - Ensure a space after sentence punctuation before capital letters

    Args:
        text: Text to finalize.
        language: Language code, used only for domain-exclusion selection.
    """
    if not text:
        return text
    out = _normalize_mixed_terminal_punctuation(text)
    out = re.sub(r"\s+", " ", out)
    # Domain masking exclusions, resolved from the transcript's own language.
    # Until v0.13.0 this call hardcoded 'es' for everything except 'pt', so en/fr/de
    # inherited Spanish's .de/.es TLD suppression and could not recognise a German
    # or Spanish national domain. The common-word guard is unaffected either way:
    # _is_excluded_label() applies SPANISH_EXCLUSIONS for every language.
    masked = mask_domains(out, use_exclusions=True, language=language)
    # Ensure single space after sentence punctuation when followed by a letter (including lowercase accented)
    # But NOT for person initials like "C.S." where the period is part of the initial
    # Use negative lookbehind to avoid: periods in ellipses, periods after single capital letters (initials)
    _letter = rf"A-Za-z{UPPER_ACCENTED}{LOWER_ACCENTED}"
    masked = re.sub(rf"(?<!\.)(?<![A-Z])\.\s*([{_letter}])", r". \1", masked)
    masked = re.sub(rf"\?\s*([{_letter}])", r"? \1", masked)
    masked = re.sub(rf"!\s*([{_letter}])", r"! \1", masked)
    # Capitalize after terminators when appropriate
    masked = re.sub(rf"([.!?])\s+([a-z{LOWER_ACCENTED}])", lambda m: f"{m.group(1)} {m.group(2).upper()}", masked)
    # Unmask domains using centralized function
    out = unmask_domains(masked)
    # Normalize comma spacing using centralized function
    out = _normalize_comma_spacing(out)
    return out.strip()


# --- Cross-language helpers exposed for orchestration ---
def _normalize_initials_and_acronyms(text: str) -> str:
    """Normalize person initials and organizational acronyms to avoid false sentence splits.
    
    This function is language-agnostic and handles:
    1. Person initials: Remove spaces but keep periods to prevent sentence breaks
       - "C. S. Lewis" → "C.S. Lewis"
       - "J. K. Rowling" → "J.K. Rowling"  
       - "J. R. R. Tolkien" → "J.R.R. Tolkien"
    2. Organizational acronyms at sentence/phrase boundaries: Remove both spaces and periods
       - "U. S. A." → "USA" (at end or before punctuation)
       - "in the U. S. today" → "in the US today" (before lowercase word)
    
    Examples:
      - "es a C. S. Lewis porque" → "es a C.S. Lewis porque" (person initials)
      - "the U. S. Capitol" → "the U.S. Capitol" (acronym before proper noun - keep periods)
      - "in the U. S. A." → "in the USA" (acronym at end - remove periods)
    """
    if not text:
        return text
    
    # Strategy: ALWAYS remove spaces from initials/acronyms, but only remove periods
    # when it's clearly an organizational acronym (not a person name).
    
    # Pattern 1: Three spaced initials followed by a word starting with capital + lowercase
    # This is almost certainly a person name: "J. R. R. Tolkien"
    # Convert to compact form with periods: "J.R.R. Tolkien"
    text = re.sub(
        r"\b([A-Z])\.\s+([A-Z])\.\s+([A-Z])\.\s+([A-Z][a-z]+)",
        r"\1.\2.\3. \4",
        text
    )
    
    # Pattern 2: Two spaced initials followed by a word starting with capital + lowercase
    # This is likely a person name: "C. S. Lewis", "J. K. Rowling"
    # Convert to compact form with periods: "C.S. Lewis"
    text = re.sub(
        r"\b([A-Z])\.\s+([A-Z])\.\s+([A-Z][a-z]+)",
        r"\1.\2. \3",
        text
    )
    
    # Pattern 3: Three-letter acronyms at end or before punctuation/lowercase
    # Remove both spaces AND periods: "U. S. A." → "USA"
    text = re.sub(
        r"\b([A-Z])\.\s*([A-Z])\.\s*([A-Z])\.(?=\s*[,;:!?\.\)\]\"']|\s+[a-z]|\s*$)",
        lambda m: ''.join(m.groups()),
        text
    )
    
    # Pattern 4: Two-letter acronyms before lowercase words or at boundaries
    # Remove both spaces AND periods: "U. S. today" → "US today"
    text = re.sub(
        r"\b([A-Z])\.\s+([A-Z])\.(?=\s+[a-z]|\s*[,;:!?\.\)\]\"']|\s*$)",
        lambda m: ''.join(m.groups()),
        text
    )
    
    # Pattern 5: Compact two-letter forms before lowercase or at boundaries
    # "U.S. today" → "US today", but "C.S. Lewis" stays as is
    text = re.sub(
        r"\b([A-Z])\.([A-Z])\.(?=\s+[a-z]|\s*[,;:!?\.\)\]\"']|\s*$)",
        r"\1\2",
        text
    )
    
    # Pattern 6: Any remaining spaced initials (e.g., "U. S. Capitol")
    # Just remove spaces, keep periods: "U. S." → "U.S."
    text = re.sub(
        r"\b([A-Z])\.\s+([A-Z])\.",
        r"\1.\2.",
        text
    )
    
    return text


# (Removed) emphatic repeat merging to simplify maintenance

def _split_processed_segment(processed: str, language: str) -> tuple[list[str], str]:
    """Split a single, punctuation-restored segment into sentences.

    Preserves ellipses (… and ...), and avoids breaking inside domains (label.tld).
    Returns (sentences, trailing_fragment_without_terminal_punct).
    The trailing fragment should be carried into the next segment by the caller if desired.
    """
    # CRITICAL: Mask domains before splitting to prevent breaking them
    # Use centralized domain masking that handles both simple and subdomain patterns
    processed_masked = mask_domains(processed, use_exclusions=True, language=language)
    
    parts = re.split(r'(…|[.!?]+)', processed_masked)
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

        # Decimal number glue: prev ends with digits and next starts with digits (e.g., 99.9, 121.73)
        # Heuristic: restrict to short numeric groups to avoid gluing years like 2019. 9 meses
        if punct == '.':
            next_chunk = parts[idx + 2] if idx + 2 < len(parts) else ""
            prev_num_match = re.search(r"(\d{1,3})$", chunk)
            next_frac_match = re.match(r"^(\d{1,3})(.*)$", next_chunk)
            if prev_num_match and next_frac_match:
                frac_digits = next_frac_match.group(1)
                remainder_after_frac = next_frac_match.group(2)
                # Glue: append '.' + fraction digits
                buffer += '.' + frac_digits
                # Leave remainder (e.g., '% de la población') for subsequent processing
                parts[idx + 2] = remainder_after_frac
                idx += 2
                continue

        # Note: Domain protection is now handled by masking at the start of this function

        # Default: flush on terminal punctuation
        if punct:
            buffer += punct
            cleaned = re.sub(r'^[",\s]+', '', buffer)
            if cleaned:
                # Use centralized punctuation logic
                cleaned = _should_add_terminal_punctuation(cleaned, language, PunctuationContext.FRAGMENT)
                sentences.append(cleaned)
            buffer = ""
            idx += 2
            continue

        # End without explicit punctuation → return trailing buffer to caller
        if idx + 1 >= len(parts):
            break
        idx += 2

    trailing = buffer.strip()
    
    # Unmask domains in sentences and trailing fragment
    sentences = [unmask_domains(s) for s in sentences]
    trailing = unmask_domains(trailing) if trailing else trailing
    
    return sentences, trailing


def _fr_merge_short_connector_breaks(sentences: list[str]) -> list[str]:
    """Merge French sentences that were split after short function words (e.g., 'au.', 'de.', 'et.').

    Also normalizes stray sequences like ",." to "," before merging.
    """
    if not sentences:
        return sentences
    short_connectors = {
        'a', 'à', 'au', 'aux', 'de', 'du', 'des', 'la', 'le', 'les', 'un', 'une',
        'en', 'et', 'ou', 'mais', 'pour', 'sur', 'sous', 'chez', 'dans', 'par',
        'avec', 'sans', 'vers', 'selon', 'contre', 'entre', 'après', 'avant',
        'depuis', 'pendant', "jusqu'", 'jusque'
    }
    merged: list[str] = []
    for s in sentences:
        if merged:
            prev = merged[-1]
            # Normalize stray ",."
            prev_norm = re.sub(r',\s*\.', ',', prev)
            if prev_norm != prev:
                prev = prev_norm
            m = re.search(r'(\b[\w\u00C0-\u017F]+)\.$', prev)
            if m:
                last_word = m.group(1).lower()
                cur_trim = s.lstrip()
                if last_word in short_connectors and cur_trim:
                    cur_cont = cur_trim[0].lower() + cur_trim[1:]
                    merged[-1] = prev[:m.start(1)] + m.group(1) + ' ' + cur_cont
                    continue
        merged.append(s)
    return merged


def assemble_sentences_from_processed(processed: str, language: str) -> tuple[list[str], str]:
    """
    Public helper to split a single processed segment into sentences with
    language-specific post-processing.

    Returns (sentences, trailing_fragment).
    """
    sentences, trailing = _split_processed_segment(processed, language)
    lang = (language or '').lower()
    if lang == 'fr' and sentences:
        sentences = _fr_merge_short_connector_breaks(sentences)
    if lang == 'es' and sentences:
        sentences = _es_merge_appositive_location_breaks(sentences)
    return sentences, trailing

# --- Spanish helper utilities (pure refactors of existing logic) ---
def _es_merge_appositive_location_breaks(sentences: list[str]) -> list[str]:
    """Merge splits across appositive location chains: ", de <Proper>. <Proper> …" -> ", de <Proper>, <Proper> …".

    This operates at assembly time to fix boundary artifacts that slipped past the splitter.
    """
    import logging
    logger = logging.getLogger("podscripter")
    
    merged: list[str] = []
    i = 0
    while i < len(sentences):
        if i + 1 < len(sentences):
            prev = (sentences[i] or '').strip()
            curr = (sentences[i + 1] or '').strip()
            
            # prev ends with ", de Proper[,.]?" optionally with a trailing period
            m_prev = re.search(r"^(.*?,\s*de\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑ-]+)\.?$", prev)
            m_curr = re.match(r"^([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑ-]+)([\s\S]*)$", curr)
            
            if m_prev and m_curr:
                # CRITICAL: Only merge if the second sentence is just a location name with minimal trailing content
                # Don't merge if it contains additional sentences (indicated by sentence terminators like ., !, ?)
                trailing_content = (m_curr.group(2) or '').strip()
                
                # Allow merging only if trailing content is empty or just a period
                if not trailing_content or trailing_content == '.':
                    base = m_prev.group(1)
                    # Ensure a comma before appositive continuation
                    if not base.endswith(','):
                        base = base + ','
                    cont = m_curr.group(1) + (m_curr.group(2) or '')
                    merged_text = f"{base} {cont}".strip()
                    merged.append(merged_text)
                    i += 2
                    continue
        merged.append(sentences[i])
        i += 1
    return merged

# Embeddings caches for static pattern sets per language
_QUESTION_PATTERN_EMBEDDINGS = {}
_EXCL_PATTERN_EMBEDDINGS = {}

def _get_question_pattern_embeddings(language: str, model):
    if language in _QUESTION_PATTERN_EMBEDDINGS:
        return _QUESTION_PATTERN_EMBEDDINGS[language]
    patterns = _get_question_patterns(language)
    if not patterns:
        _QUESTION_PATTERN_EMBEDDINGS[language] = None
        return None
    try:
        embs = model.encode(patterns)
        _QUESTION_PATTERN_EMBEDDINGS[language] = embs
        return embs
    except Exception:
        _QUESTION_PATTERN_EMBEDDINGS[language] = None
        return None


def _get_exclamation_pattern_embeddings(language: str, model):
    if language in _EXCL_PATTERN_EMBEDDINGS:
        return _EXCL_PATTERN_EMBEDDINGS[language]
    patterns = _get_exclamation_patterns(language)
    if not patterns:
        _EXCL_PATTERN_EMBEDDINGS[language] = None
        return None
    try:
        embs = model.encode(patterns)
        _EXCL_PATTERN_EMBEDDINGS[language] = embs
        return embs
    except Exception:
        _EXCL_PATTERN_EMBEDDINGS[language] = None
        return None

def _get_cache_paths():
    """Return preferred cache paths inside the repo (mounted at /app) or fallbacks.

    We prefer using the repo-mounted caches so tests can reuse downloads across runs:
    - /app/models/sentence-transformers
    - /app/models/huggingface
    Fallbacks to environment defaults used in Docker image.
    """
    # Preferred in-repo caches (persist via -v $(pwd):/app)
    repo_root = "/app"
    models_dir = os.path.join(repo_root, "models")
    st_repo_cache = os.path.join(models_dir, "sentence-transformers")
    hf_repo_cache = os.path.join(models_dir, "huggingface")

    # Docker defaults (already set in Dockerfile/run script)
    st_default_cache = os.getenv("SENTENCE_TRANSFORMERS_HOME", "/root/.cache/torch/sentence_transformers")
    hf_default_cache = os.getenv("HF_HOME", "/root/.cache/huggingface")

    st_cache = st_repo_cache if os.path.isdir(st_repo_cache) else st_default_cache
    hf_cache = hf_repo_cache if os.path.isdir(hf_repo_cache) else hf_default_cache
    return st_cache, hf_cache


def _find_local_model_path(preferred_st_cache: str, model_name: str) -> str | None:
    """Try to find a locally cached folder for the given model under the sentence-transformers cache.

    Returns a directory path if found, else None.
    """
    if not os.path.isdir(preferred_st_cache):
        return None
    try:
        for entry in os.listdir(preferred_st_cache):
            entry_path = os.path.join(preferred_st_cache, entry)
            if os.path.isdir(entry_path) and model_name.replace('/', '-') in entry:
                # Heuristic match on directory name
                return entry_path
    except Exception:
        pass
    return None


def _load_sentence_transformer(model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
    """Load SentenceTransformer once, preferring local cache and enabling offline when possible."""
    global _SENTENCE_TRANSFORMER_SINGLETON
    if _SENTENCE_TRANSFORMER_SINGLETON is not None:
        return _SENTENCE_TRANSFORMER_SINGLETON

    if SentenceTransformer is None:
        return None

    st_cache, hf_cache = _get_cache_paths()

    # Ensure HF_HOME points to our preferred cache to consolidate downloads
    os.environ.setdefault("HF_HOME", hf_cache)

    short_name = 'paraphrase-multilingual-MiniLM-L12-v2'

    logger.info(f"Loading punctuation model ({short_name})...")

    # If a local model directory already exists, the cache is warm and we can
    # load fully offline. NOTE: setting os.environ["HF_HUB_OFFLINE"] here is a
    # no-op — huggingface_hub freezes that value into a module constant at import
    # time, and faster_whisper (imported at startup) pulls in huggingface_hub
    # before we get here, so the constant is already locked to False. The per-call
    # `local_files_only=True` argument below is honored regardless and is what
    # actually prevents the network HEAD request (and its "unauthenticated
    # requests to the HF Hub" warning) on every warm-cache run.
    cache_is_warm = bool(local_model_dir := _find_local_model_path(st_cache, model_name.split('/')[-1])) and os.path.isdir(local_model_dir)

    # First try: fully offline load from the warm cache (no network requests).
    # Load by name with cache_folder so Sentence-Transformers resolves the proper
    # snapshot layout (avoids the "Creating a new one with mean pooling" warning).
    if cache_is_warm:
        for name in (model_name, short_name):
            try:
                _SENTENCE_TRANSFORMER_SINGLETON = SentenceTransformer(
                    name, cache_folder=st_cache, local_files_only=True
                )
                return _SENTENCE_TRANSFORMER_SINGLETON
            except Exception:
                # Cache may be incomplete for this name; try the next, then fall
                # back to an online load below.
                continue

    # Fallback: online load (first run / cold cache) — allowed to download.
    try:
        _SENTENCE_TRANSFORMER_SINGLETON = SentenceTransformer(model_name, cache_folder=st_cache)
        return _SENTENCE_TRANSFORMER_SINGLETON
    except Exception:
        # Last resort: try short name without org (older sbert versions)
        _SENTENCE_TRANSFORMER_SINGLETON = SentenceTransformer(short_name, cache_folder=st_cache)
        return _SENTENCE_TRANSFORMER_SINGLETON


def _extract_segment_boundaries(text: str, segments: list[dict]) -> list[int]:
    """
    Convert Whisper segments to character positions in the concatenated text.
    
    Args:
        text: Concatenated text from all segments
        segments: List of dicts with 'text' field from Whisper
    
    Returns:
        List of character positions where Whisper segments end
    """
    boundaries = []
    position = 0
    for seg in segments:
        seg_text = seg['text'].strip()
        if not seg_text:
            continue
        # Account for joining with newlines/spaces
        position += len(seg_text)
        boundaries.append(position)
        position += 1  # Account for separator (space or newline)
    return boundaries


def _char_positions_to_word_indices(text: str, char_positions: list[int]) -> set[int]:
    """
    Convert character positions to word indices for fast lookup.

    Args:
        text: The full text string
        char_positions: Character positions of segment boundaries

    Returns:
        Set of word indices that are at or near segment boundaries
    """
    if not char_positions:
        return set()

    words = text.split()
    word_boundaries = set()

    # Build mapping of character positions to word indices
    char_to_word = []
    current_pos = 0
    for word_idx, word in enumerate(words):
        word_start = text.find(word, current_pos)
        if word_start == -1:
            # Word not found, skip
            continue
        word_end = word_start + len(word)
        char_to_word.append((word_start, word_end, word_idx))
        current_pos = word_end

    # Map char positions to nearest word indices
    # Allow a small tolerance (+3) for spaces and punctuation
    for char_pos in char_positions:
        for word_start, word_end, word_idx in char_to_word:
            if word_start <= char_pos <= word_end + 3:
                word_boundaries.add(word_idx)
                break

    return word_boundaries


def _violates_grammatical_rules(current_word: str, next_word: str, language: str) -> bool:
    """
    Check if breaking at current position would violate basic grammatical rules.
    
    This consolidates checks for prepositions, conjunctions, and continuative verbs
    that should never end a sentence.
    
    Args:
        current_word: The word at the current position
        next_word: The following word
        language: Language code ('en', 'es', 'fr', 'de')
    
    Returns:
        True if breaking here would violate grammar rules
    """
    current_clean = current_word.lower().strip('.,;:!?¿¡')
    
    # Check coordinating conjunctions (should never end sentences)
    coordinating_conjunctions = {
        'y', 'e', 'o', 'u', 'pero', 'mas', 'sino',  # Spanish
        'and', 'but', 'or', 'nor', 'for', 'so', 'yet',  # English
        'et', 'ou', 'mais', 'donc', 'or', 'ni', 'car',  # French
        'und', 'oder', 'aber', 'denn', 'sondern',  # German
    }
    if current_clean in coordinating_conjunctions:
        return True
    
    # Language-specific checks
    if language == 'es':
        # Spanish prepositions and articles
        spanish_forbidden = {
            'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
            'a', 'ante', 'bajo', 'de', 'del', 'al', 'en', 'con', 'por', 
            'para', 'sin', 'sobre', 'entre', 'tras', 'durante', 'mediante',
            'según', 'hacia', 'hasta', 'desde', 'contra',
            'todo', 'toda', 'todos', 'todas', 'alguno', 'alguna', 'algunos',
            'algunas', 'cualquier', 'cualquiera', 'ningún', 'ninguna', 'ninguno',
            'otro', 'otra', 'otros', 'otras'
        }
        if current_clean in spanish_forbidden:
            return True
        
        # Spanish continuative/auxiliary verbs
        spanish_continuative = {
            'estaba', 'estabas', 'estábamos', 'estaban',
            'era', 'eras', 'éramos', 'eran',
            'tenía', 'tenías', 'teníamos', 'tenían',
            'había', 'habías', 'habíamos', 'habían',
            'iba', 'ibas', 'íbamos', 'iban',
            'hacía', 'hacías', 'hacíamos', 'hacían',
            'podía', 'podías', 'podíamos', 'podían',
            'debía', 'debías', 'debíamos', 'debían',
            'quería', 'querías', 'queríamos', 'querían',
            'sabía', 'sabías', 'sabíamos', 'sabían',
            'venía', 'venías', 'veníamos', 'venían',
            'decía', 'decías', 'decíamos', 'decían',
            'he', 'has', 'ha', 'hemos', 'habéis', 'han'
        }
        if current_clean in spanish_continuative:
            return True
    
    elif language == 'en':
        # English prepositions
        english_prepositions = {
            'to', 'at', 'from', 'with', 'by', 'of', 'in', 'on', 'for',
            'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'under', 'over'
        }
        if current_clean in english_prepositions:
            return True
        
        # English continuative/auxiliary verbs
        english_continuative = {
            'was', 'were', 'had', 'been', 'have', 'has'
        }
        if current_clean in english_continuative:
            return True
    
    elif language == 'fr':
        # French prepositions
        french_prepositions = {
            'à', 'de', 'en', 'pour', 'avec', 'sans', 'sous', 'sur',
            'dans', 'chez', 'vers', 'par', 'entre', 'parmi', 'du', 'des'
        }
        if current_clean in french_prepositions:
            return True
        
        # French continuative verbs
        french_continuative = {
            'étais', 'était', 'étions', 'étiez', 'étaient',
            'avais', 'avait', 'avions', 'aviez', 'avaient',
            'allais', 'allait', 'allions', 'alliez', 'allaient',
            'faisais', 'faisait', 'faisions', 'faisiez', 'faisaient'
        }
        if current_clean in french_continuative:
            return True
    
    elif language == 'de':
        # German prepositions
        german_prepositions = {
            'zu', 'an', 'auf', 'aus', 'bei', 'mit', 'nach', 'von',
            'vor', 'in', 'für', 'über', 'unter', 'durch', 'gegen',
            'ohne', 'um', 'zwischen'
        }
        if current_clean in german_prepositions:
            return True
        
        # German continuative verbs and modals
        german_continuative = {
            'war', 'hatte', 'ging', 'machte', 'konnte', 'wollte',
            'musste', 'sollte'
        }
        if current_clean in german_continuative:
            return True

    elif language == 'pt':
        # Portuguese articles, preposition+article contractions and prepositions.
        # Contractions ("do", "na", "pelo") are extremely frequent and can never
        # end a sentence. Note 'no'/'na' are safe to list here because this block
        # is language-gated: Portuguese "no" = "in the", whereas Spanish "no" =
        # "not" legitimately ends sentences ("Claro que no.").
        portuguese_prepositions = {
            'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas',
            'do', 'da', 'dos', 'das', 'no', 'na', 'nos', 'nas',
            'ao', 'à', 'aos', 'às', 'pelo', 'pela', 'pelos', 'pelas',
            'num', 'numa', 'dum', 'duma',
            'de', 'em', 'para', 'pra', 'por', 'com', 'sem', 'sobre',
            'entre', 'até', 'desde', 'contra', 'durante', 'mediante',
            'perante', 'sob', 'trás',
            # Proclitic object/reflexive pronouns: attach forward to the verb.
            'me', 'te', 'se', 'lhe', 'lhes', 'vos',
            # Quantifiers/determiners that require a following noun.
            'todo', 'toda', 'todos', 'todas', 'algum', 'alguma', 'alguns', 'algumas',
            'nenhum', 'nenhuma', 'qualquer', 'outro', 'outra', 'outros', 'outras', 'cada',
        }
        if current_clean in portuguese_prepositions:
            return True

        # Portuguese continuative/auxiliary verbs (imperfect + present of ter/haver)
        portuguese_continuative = {
            'era', 'eram', 'estava', 'estavam', 'tinha', 'tinham',
            'havia', 'haviam', 'ia', 'iam', 'fazia', 'faziam',
            'podia', 'podiam', 'devia', 'deviam', 'queria', 'queriam',
            'sabia', 'sabiam', 'vinha', 'vinham', 'dizia', 'diziam',
            'tenho', 'tens', 'tem', 'temos', 'têm',
        }
        if current_clean in portuguese_continuative:
            return True

    return False


def restore_punctuation(text: str, language: str = 'en', whisper_segments: list[dict] | None = None, speaker_segments: list[dict] | None = None, whisper_boundaries: list[int] | None = None, speaker_boundaries: list[int] | None = None) -> tuple[str, list[Sentence] | None]:
    """
    Restore punctuation to transcribed text using advanced NLP techniques.

    Args:
        text (str): The transcribed text without proper punctuation
        language (str): Language code ('en', 'es', 'de', 'fr')
        whisper_segments (list[dict] | None): Optional list of Whisper segments with
            'text', 'start', 'end' fields. Used for sentence boundary hints.
        speaker_segments (list[dict] | None): Optional list of speaker segments with
            'start_word', 'end_word', and 'speaker' fields. Used to detect when the
            same speaker continues speaking.
        whisper_boundaries (list[int] | None): DEPRECATED - Use whisper_segments instead.
            Optional list of character positions where Whisper segments end.
        speaker_boundaries (list[int] | None): DEPRECATED - Use speaker_segments instead.
            Optional list of character positions where speakers change.

    Returns:
        tuple[str, list[str] | None]: A tuple of (processed_text, sentences_list).
            - processed_text: Text with restored punctuation as a single string
            - sentences_list: Pre-split list of sentences (always populated in v0.4.0+)
    """
    if not text.strip():
        return text, []
    
    # NOTE: Text normalization (whitespace cleanup) is now done in podscripter.py
    # BEFORE speaker_word_ranges are calculated (v0.4.3 fix). Do NOT normalize here
    # as it would invalidate the word indices in speaker_segments.
    # The text passed here should already be normalized.
    
    # Use advanced punctuation restoration
    try:
        return _advanced_punctuation_restoration(text, language, True, whisper_segments, speaker_segments, whisper_boundaries, speaker_boundaries)
    except Exception as e:
        import traceback
        logger.warning(f"Advanced punctuation restoration failed: {e}")
        logger.warning(f"Traceback: {traceback.format_exc()}")
        logger.info("Returning original text without punctuation restoration.")
        return text, [Sentence(text=text, utterances=[], speaker=None)]


def _advanced_punctuation_restoration(text: str, language: str = 'en', use_custom_patterns: bool = True, whisper_segments: list[dict] | None = None, speaker_segments: list[dict] | None = None, whisper_boundaries: list[int] | None = None, speaker_boundaries: list[int] | None = None) -> tuple[str, list[Sentence] | None]:
    """
    Advanced punctuation restoration using sentence transformers and NLP techniques.

    Args:
        text (str): The transcribed text without proper punctuation
        language (str): Language code ('en', 'es', 'de', 'fr')
        use_custom_patterns (bool): Whether to use custom sentence endings and question word patterns
        whisper_segments (list[dict] | None): Optional Whisper segments for boundary hints
        speaker_segments (list[dict] | None): Optional speaker segments with word ranges and labels
        whisper_boundaries (list[int] | None): DEPRECATED - Optional character positions
        speaker_boundaries (list[int] | None): DEPRECATED - Optional character positions

    Returns:
        tuple[str, list[Sentence] | None]: (processed_text, sentences_list)
    """

    # Use SentenceTransformers for better sentence boundary detection
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        return _transformer_based_restoration(text, language, use_custom_patterns, whisper_segments, speaker_segments, whisper_boundaries, speaker_boundaries)
    else:
        # Simple fallback: text is already normalized in podscripter.py (v0.4.3)
        return _should_add_terminal_punctuation(text, language, PunctuationContext.SENTENCE_END), [Sentence(text=text, utterances=[], speaker=None)]


def _transformer_based_restoration(text: str, language: str = 'en', use_custom_patterns: bool = True, whisper_segments: list[dict] | None = None, speaker_segments: list[dict] | None = None, whisper_boundaries: list[int] | None = None, speaker_boundaries: list[int] | None = None) -> tuple[str, list[Sentence] | None]:
    """
    Improved punctuation restoration using SentenceTransformers for semantic understanding.
    
    Now uses unified SentenceSplitter class (v0.4.0+).
    
    Args:
        text (str): The transcribed text without proper punctuation
        language (str): Language code ('en', 'es', 'de', 'fr')
        use_custom_patterns (bool): Whether to use custom patterns
        whisper_segments (list[dict] | None): Optional Whisper segments with metadata
        speaker_segments (list[dict] | None): Optional speaker segments with word ranges and labels
        whisper_boundaries (list[int] | None): DEPRECATED - Optional character positions
        speaker_boundaries (list[int] | None): DEPRECATED - Optional character positions
    
    Returns:
        tuple[str, list[Sentence]]: (processed_text, sentences_list)
    """
    # Initialize the model once (use multilingual model for better language support)
    model = _load_sentence_transformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    if model is None:
        # Fallback path if sentence-transformers is unavailable
        # Text is already normalized in podscripter.py (v0.4.3)
        return _should_add_terminal_punctuation(text, language, PunctuationContext.SENTENCE_END), [Sentence(text=text, utterances=[], speaker=None)]

    # Get language config for thresholds
    lang_config = _get_language_config(language)

    # 1) Use unified SentenceSplitter for all sentence boundary decisions
    splitter = SentenceSplitter(language, model, lang_config)
    sentences, metadata = splitter.split(
        text,
        whisper_segments=whisper_segments,
        speaker_segments=speaker_segments,
        mode='semantic'
    )
    
    # Log metadata for debugging
    if metadata.get('removed_periods'):
        logger.debug(f"Removed {len(metadata['removed_periods'])} Whisper periods")
        for removal in metadata['removed_periods']:
            connector_info = f" connector='{removal['connector']}'" if 'connector' in removal else ""
            logger.debug(f"  - Position {removal['position']}: {removal['reason']}{connector_info} (speaker: {removal.get('speaker', 'unknown')})")
    # 2) Punctuate each sentence individually (preserving boundaries from SentenceSplitter)
    # Note: sentences is now a list of Sentence objects (v0.6.0)
    punctuated_sentences = []
    sentence_objects = []  # Keep track of Sentence objects for speaker info
    for i, sent_obj in enumerate(sentences):
        # Extract text from Sentence object
        if isinstance(sent_obj, Sentence):
            sent_text = sent_obj.text
            sentence_objects.append(sent_obj)
            logger.debug(f"Extracted text from Sentence object {i}: '{sent_text[:50]}...'")
        else:
            sent_text = sent_obj
            sentence_objects.append(None)
            logger.debug(f"Sentence {i} is a string: '{sent_text[:50] if isinstance(sent_text, str) else type(sent_text)}...'")
        
        if sent_text.strip():
            punctuated = _apply_semantic_punctuation(sent_text, model, language, i, len(sentences))
            logger.debug(f"Punctuated sentence {i} type: {type(punctuated)}, value: '{punctuated[:50] if isinstance(punctuated, str) else punctuated}...'")
            punctuated_sentences.append(punctuated)
        else:
            sentence_objects.pop()  # Remove if empty
    
    logger.debug(f"Total punctuated_sentences: {len(punctuated_sentences)}, types: {[type(s).__name__ for s in punctuated_sentences[:5]]}")
    
    # Ensure all items in punctuated_sentences are strings (defensive programming)
    punctuated_sentences = [
        s.text if isinstance(s, Sentence) else str(s) 
        for s in punctuated_sentences
    ]
    logger.debug(f"After string conversion, punctuated_sentences types: {[type(s).__name__ for s in punctuated_sentences[:5]]}")
    
    # Apply Spanish-specific formatting
    if language == 'es':
        # CRITICAL: Use the punctuated sentences from SentenceSplitter directly
        # Do NOT re-split by punctuation marks - SentenceSplitter has already handled all boundaries
        sentences_list = punctuated_sentences

        # Format each sentence individually WITHOUT re-splitting by punctuation
        formatted_sentences = []
        formatted_sentence_objects = []
        for idx, s in enumerate(sentences_list):
            # Extract text if s is a Sentence object (should already be string, but be defensive)
            if isinstance(s, Sentence):
                sentence_text = s.text
            else:
                sentence_text = s
            sentence = (sentence_text or '').strip()
            if not sentence:
                continue
            
            # Capitalize first letter (but not for domains)
            if sentence and sentence[0].isalpha():
                # Don't capitalize if this looks like a domain name
                if not re.match(rf'^[a-zA-Z0-9\u00C0-\u017F\-]+\.({SINGLE_TLDS_CONSERVATIVE})\b', sentence.lower()):
                    sentence = sentence[0].upper() + sentence[1:]

            # Ensure sentence ends with single terminal punctuation
            if not sentence.endswith(('.', '!', '?')):
                # Strip trailing commas before adding terminal punctuation
                sentence = sentence.rstrip(',;: ')
                
                question_words = ES_QUESTION_WORDS_CORE + ES_QUESTION_STARTERS_EXTRA
                sentence_lower = sentence.lower()
                if any(word in sentence_lower for word in question_words):
                    sentence += '?'
                else:
                    # Use centralized punctuation logic for Spanish
                    sentence = _should_add_terminal_punctuation(sentence, language, PunctuationContext.SPANISH_SPECIFIC)
            else:
                sentence = sentence.rstrip('.!?') + sentence[-1]

            # Add inverted question mark if needed (without re-splitting)
            # Only add if sentence doesn't already have an embedded ¿ (to avoid double inverted marks)
            if sentence.endswith('?') and not sentence.startswith('¿') and '¿' not in sentence:
                sentence_lower = sentence.lower()
                question_patterns = [
                    r'^(qué|dónde|cuándo|cómo|quién|cuál|por qué|recuerdas|sabes|puedes|quieres|necesitas|tienes|vas|estás|están|pueden|saben|quieren|hay|va|es|son|está|están)',
                    r'^(puedes|puede|podrías|podría|sabes|sabe|quieres|quiere|necesitas|necesita|tienes|tiene|vas|va|estás|están|pueden|quieren)',
                    r'^(hay|va|es|son|está|están|te parece|le parece|crees|cree|piensas|piensa)',
                    r'^(estamos|están|listos|listas|listo|lista|bien|mal|correcto|incorrecto|verdad|cierto)'
                ]
                if any(re.search(pattern, sentence_lower) for pattern in question_patterns):
                    if sentence.startswith('¡'):
                        sentence = sentence[1:].lstrip()
                    sentence = '¿' + sentence
            
            # Clean up duplicates
            sentence = re.sub(r'[.!?]{2,}', lambda m: m.group(0)[0], sentence)
            sentence = re.sub(r'¿{2,}', '¿', sentence)
            formatted_sentences.append(sentence)
            
            # Reconstruct Sentence object with formatted text (v0.6.0)
            if idx < len(sentence_objects) and sentence_objects[idx] is not None:
                orig_sent = sentence_objects[idx]
                formatted_sent_obj = Sentence(
                    text=sentence,
                    utterances=orig_sent.utterances,
                    speaker=orig_sent.speaker
                )
                formatted_sentence_objects.append(formatted_sent_obj)
            else:
                # Backward compat: create Sentence with no speaker info
                formatted_sentence_objects.append(Sentence(text=sentence, utterances=[], speaker=None))

        # Join sentences with proper spacing (no re-splitting!)
        result = ' '.join(formatted_sentences)
        
        # Apply cleanup and formatting (same for all paths now)
        # Clean up double/mixed punctuation
        result = _normalize_mixed_terminal_punctuation(result)
        # Fix location appositive punctuation
        result = _fix_location_appositive_punctuation(result, language)
        # Final universal cleanup
        result = _finalize_text_common(result, language)
        # ALWAYS return both the text AND the sentences list (v0.4.0+)
        # v0.6.0: Return Sentence objects instead of strings
        return result, formatted_sentence_objects
    elif not is_tailored(language):
        # Generic language: keep the splitter's sentences as Whisper wrote them.
        # _apply_semantic_punctuation() above already reduced to "add '.' only if
        # no terminal of any script is present" for these languages. No greeting
        # commas or location-appositive commas: each is a per-language rule, and
        # English stand-ins corrupt the text.
        result = ' '.join(s.strip() for s in punctuated_sentences if s.strip())
    else:
        # Apply light, language-aware formatting for non-Spanish languages
        # Format each sentence individually
        formatted_sentences = []
        for s in punctuated_sentences:
            # Extract text if s is a Sentence object (should already be string, but be defensive)
            if isinstance(s, Sentence):
                sentence_text = s.text
            else:
                sentence_text = s
            sentence = (sentence_text or '').strip()
            if not sentence:
                continue
            
            sentence = _format_non_spanish_text(sentence, language)
            formatted_sentences.append(sentence)
        
        result = ' '.join(formatted_sentences)

        # Fix location appositive punctuation across languages
        result = _fix_location_appositive_punctuation(result, language)

        # Final universal cleanup
        result = _finalize_text_common(result, language)
    
    # Return tuple: (processed_text, sentences_list)
    # v0.4.0+: ALWAYS return sentences_list (never None)
    # v0.6.0: Return Sentence objects instead of strings
    # Only non-Spanish (light or generic path) reaches here; the Spanish branch
    # returned its own formatted_sentence_objects above.
    final_sentence_objects = []
    for idx, sent_text in enumerate(punctuated_sentences):
        if idx < len(sentence_objects) and sentence_objects[idx] is not None:
            orig_sent = sentence_objects[idx]
            final_sentence_objects.append(Sentence(
                text=sent_text,
                utterances=orig_sent.utterances,
                speaker=orig_sent.speaker
            ))
        else:
            # Backward compat: create Sentence with no speaker info
            final_sentence_objects.append(Sentence(text=sent_text, utterances=[], speaker=None))
    return result, final_sentence_objects


from typing import List


def _convert_char_ranges_to_word_ranges(text: str, char_ranges: list[dict]) -> list[dict]:
    """
    Convert character-based speaker segments to word-based segments.
    
    Args:
        text: The full text
        char_ranges: List of dicts with 'start_char', 'end_char', 'speaker' fields
    
    Returns:
        List of dicts with 'start_word', 'end_word', 'speaker' fields
    """
    if not char_ranges:
        return []
    
    words = text.split()
    word_ranges = []
    
    # Build a mapping of character positions to word indices by scanning through the text
    # This correctly handles any whitespace (spaces, newlines, tabs) between words
    char_to_word = {}
    word_idx = 0
    in_word = False
    current_word_start = 0
    accumulated_word = ""
    
    for char_pos, char in enumerate(text):
        if char.isspace():
            if in_word:
                # Just finished a word - map all its characters
                for i in range(current_word_start, char_pos):
                    char_to_word[i] = word_idx
                # Also map this whitespace character to the word we just finished
                char_to_word[char_pos] = word_idx
                word_idx += 1
                in_word = False
                accumulated_word = ""
        else:
            if not in_word:
                # Starting a new word
                current_word_start = char_pos
                in_word = True
            accumulated_word += char
    
    # Handle the last word if text doesn't end with whitespace
    if in_word:
        for i in range(current_word_start, len(text)):
            char_to_word[i] = word_idx
    
    logger.debug(f"Built char_to_word mapping: text_len={len(text)}, num_words={len(words)}, char_map_size={len(char_to_word)}")
    
    # Convert each character range to word range
    for idx, char_range in enumerate(char_ranges):
        start_char = char_range['start_char']
        end_char = char_range['end_char']
        speaker = char_range['speaker']
        
        # Find the word index for start and end positions
        start_word = char_to_word.get(start_char)
        end_word = char_to_word.get(end_char - 1) if end_char > 0 else None
        
        # If we couldn't find exact mappings, find the closest words
        if start_word is None:
            # Find the first word at or after this position
            start_word = 0
            for char_pos in range(start_char, len(text)):
                if char_pos in char_to_word:
                    start_word = char_to_word[char_pos]
                    break
        
        if end_word is None:
            # Find the last word at or before this position
            end_word = len(words) - 1
            for char_pos in range(min(end_char - 1, len(text) - 1), -1, -1):
                if char_pos in char_to_word:
                    end_word = char_to_word[char_pos]
                    break
        
        logger.debug(f"Char range {idx}: chars [{start_char}:{end_char}] → words [{start_word}:{end_word}] speaker={speaker}")
        
        if start_word is not None and end_word is not None:
            word_ranges.append({
                'start_word': start_word,
                'end_word': end_word + 1,  # Make end_word exclusive for easier range logic
                'speaker': speaker
            })
    
    # Merge consecutive ranges from the same speaker
    if word_ranges:
        merged_ranges = []
        current_range = word_ranges[0].copy()
        
        for next_range in word_ranges[1:]:
            if (next_range['speaker'] == current_range['speaker'] and 
                next_range['start_word'] <= current_range['end_word']):
                # Same speaker and adjacent/overlapping - merge
                current_range['end_word'] = max(current_range['end_word'], next_range['end_word'])
            else:
                # Different speaker or gap - save current and start new
                merged_ranges.append(current_range)
                current_range = next_range.copy()
        
        # Don't forget the last range
        merged_ranges.append(current_range)
        
        return merged_ranges
    
    return word_ranges


def _is_continuation_word(word: str, language: str) -> bool:
    """
    Check if a word suggests the sentence should continue.
    
    Args:
        word (str): The word to check
        language (str): Language code
    
    Returns:
        bool: True if word suggests continuation
    """
    continuation_words = {
        'en': ['and', 'or', 'but', 'so', 'because', 'if', 'when', 'while', 'since', 'although', 'however', 'therefore', 'thus', 'hence', 'then', 'next', 'also', 'as', 'well', 'as', 'in', 'addition', 'furthermore', 'moreover', 'besides', 'additionally'],
        'es': ['y', 'o', 'pero', 'así', 'porque', 'si', 'cuando', 'mientras', 'desde', 'aunque', 'sin', 'embargo', 'por', 'tanto', 'entonces', 'también', 'además', 'furthermore', 'más', 'aún', 'a', 'al', 'hacia', 'hasta', 'de', 'del', 'en', 'con'],
        'de': ['und', 'oder', 'aber', 'also', 'weil', 'wenn', 'während', 'seit', 'obwohl', 'jedoch', 'daher', 'deshalb', 'dann', 'auch', 'außerdem', 'ferner', 'zudem'],
        'fr': ['et', 'ou', 'mais', 'donc', 'parce', 'si', 'quand', 'pendant', 'depuis', 'bien', 'que', 'cependant', 'donc', 'alors', 'aussi', 'de', 'plus', 'en', 'outre', 'par', 'ailleurs'],
        'pt': ['e', 'ou', 'mas', 'nem', 'então', 'porque', 'se', 'quando', 'enquanto', 'desde', 'embora', 'porém', 'contudo', 'portanto', 'também', 'além', 'ainda', 'mais', 'a', 'ao', 'à', 'para', 'até', 'de', 'do', 'da', 'em', 'no', 'na', 'com', 'por']
    }

    words = continuation_words.get(language, continuation_words['en'])
    return word.lower() in words


class PunctuationContext:
    """Context types for different punctuation scenarios."""
    STANDALONE_SEGMENT = "standalone_segment"      # Single segment from Whisper  
    SENTENCE_END = "sentence_end"                  # End of a complete sentence
    FRAGMENT = "fragment"                          # Partial sentence fragment
    TRAILING = "trailing"                          # Trailing fragment to carry forward
    SPANISH_SPECIFIC = "spanish_specific"          # Spanish-specific formatting context


# Sentence terminals across scripts, for generic languages. Whisper writes these
# natively (Japanese "。", Chinese "？", Arabic "؟", Hindi "।", Armenian "՞",
# Amharic "።", Burmese "။"), and the tailored rule below only knows ".!?" -- it
# turned "行きましょうか？" into "行きましょうか？." and "كيف حالك؟" into "كيف حالك؟?".
# ';' is included because it is the Greek question mark (Whisper emits U+003B, the
# NFC form of U+037E); in other scripts a trailing ';' is left alone, not rewritten.
_GENERIC_TERMINALS = frozenset(".!?…‽;\u037e。！？｡؟۔।॥။።፧՜՞")
# Closing quotes/brackets that may follow the terminal: «Ciao.» / 「はい。」
_GENERIC_CLOSERS = "\"'”’»›)]}）」』】〉》"
# Trailing clause punctuation replaced by the added period.
_GENERIC_TRAILING_CLAUSE = " ,:，、：،"


def _ensure_generic_terminal(text: str) -> str:
    """Append '.' to a generic-language sentence only if it has no terminal at all."""
    core = text.rstrip().rstrip(_GENERIC_CLOSERS)
    if core and core[-1] in _GENERIC_TERMINALS:
        return text
    return text.rstrip(_GENERIC_TRAILING_CLAUSE) + '.'


def _should_add_terminal_punctuation(text: str, language: str, context: str | None = None, model=None) -> str:
    """
    Centralized logic for determining what terminal punctuation to add.
    
    This function consolidates all the scattered period insertion logic throughout
    the codebase into a single, maintainable location.
    
    Args:
        text: The text segment to analyze
        language: Language code ('es', 'en', 'fr', 'de')
        context: Context type from PunctuationContext class
        model: Optional sentence transformer model for semantic analysis
    
    Returns:
        The text with appropriate terminal punctuation added
    """
    if not text:
        return text
    if not is_tailored(language):
        # No continuation words, question detection or short-phrase rules: those
        # are all language-specific. Keep Whisper's terminal whatever its script.
        return _ensure_generic_terminal(text)
    if text.endswith(('.', '!', '?')):
        return text
    
    context = context or PunctuationContext.STANDALONE_SEGMENT
    
    # Check for continuation words (like "Ve a", "Voy a") 
    words = text.split()
    last_word = words[-1] if words else ''
    
    # For standalone segments that end with continuation words, don't add punctuation
    # This fixes the "Ve a" -> "Ve a." bug
    if _is_continuation_word(last_word, language):
        if context == PunctuationContext.STANDALONE_SEGMENT:
            # Don't add punctuation to incomplete segments like "Ve a"
            return text
        elif context == PunctuationContext.TRAILING:
            # Trailing fragments should also not get punctuation
            return text
        # For other contexts, continue with normal punctuation logic
    
    # Check for questions using semantic analysis if model is available
    if model and context != PunctuationContext.FRAGMENT:
        if is_question_semantic(text, model, language):
            # Strip trailing commas/semicolons before adding question mark
            text = text.rstrip(',;: ')
            return text + '?'
        if is_exclamation_semantic(text, model, language):
            # Strip trailing commas/semicolons before adding exclamation mark
            text = text.rstrip(',;: ')
            return text + '!'
        # If semantic analysis is available, trust it and skip word-based fallback
        # Semantic analysis correctly identified this as NOT a question/exclamation
        # Don't override with simplistic word matching
    elif language == 'es':
        # Language-specific question detection fallback (only when semantic analysis unavailable)
        # This is less accurate and prone to false positives
        question_words = ES_QUESTION_WORDS_CORE + ES_QUESTION_STARTERS_EXTRA
        text_lower = text.lower()
        if any(word in text_lower for word in question_words):
            # Strip trailing commas/semicolons before adding question mark
            text = text.rstrip(',;: ')
            return text + '?'
    
    # Special handling for short Spanish phrases
    if language == 'es' and text.lower() in ['también sí', 'sí', 'no', 'claro', 'exacto', 'perfecto', 'vale', 'bien', 'pues tranquilo']:
        return text + '.'
    
    # Default to period for complete sentences
    # Strip trailing commas/semicolons before adding period
    text = text.rstrip(',;: ')
    return text + '.'


def _apply_semantic_punctuation(sentence: str, model, language: str, sentence_index: int, total_sentences: int) -> str:
    """
    Apply appropriate punctuation to a sentence using semantic analysis.
    
    Args:
        sentence (str): The sentence to punctuate
        model: SentenceTransformer model
        language (str): Language code
        sentence_index (int): Index of current sentence
        total_sentences (int): Total number of sentences
    
    Returns:
        str: Sentence with appropriate punctuation
    """
    # Defensive: ensure sentence is a string
    if isinstance(sentence, Sentence):
        logger.warning(f"_apply_semantic_punctuation received Sentence object, extracting text")
        sentence = sentence.text
    elif not isinstance(sentence, str):
        logger.warning(f"_apply_semantic_punctuation received non-string type: {type(sentence)}")
        sentence = str(sentence)
    
    # Check if it's a question using semantic similarity
    if is_question_semantic(sentence, model, language):
        if not sentence.endswith('?'):
            # Strip trailing punctuation (including , ; :) before appending
            # to avoid artifacts like "Bueno,?" when the segment was split mid-clause
            # at a speaker/Whisper boundary leaving a dangling comma.
            sentence = sentence.rstrip('.!,;: ') + '?'
        return sentence
    
    # Check for exclamation patterns
    if is_exclamation_semantic(sentence, model, language):
        if not sentence.endswith('!'):
            # Strip trailing punctuation (including , ; :) before appending
            # to avoid artifacts like "Bueno,!" when the segment was split mid-clause
            # at a speaker/Whisper boundary leaving a dangling comma.
            sentence = sentence.rstrip('.?,;: ') + '!'
        return sentence
    
    # Use centralized punctuation logic
    return _should_add_terminal_punctuation(sentence, language, PunctuationContext.SENTENCE_END, model)


def is_question_semantic(sentence: str, model, language: str) -> bool:
    """
    Determine if a sentence is a question using semantic similarity.
    
    Args:
        sentence (str): The sentence to analyze
        model: SentenceTransformer model
        language (str): Language code
    
    Returns:
        bool: True if sentence is a question
    """
    if cosine_similarity is None:
        return False
    # Defensive: ensure sentence is a string
    if isinstance(sentence, Sentence):
        logger.warning(f"is_question_semantic received Sentence object, extracting text")
        sentence = sentence.text
    elif not isinstance(sentence, str):
        logger.warning(f"is_question_semantic received non-string type: {type(sentence)}")
        sentence = str(sentence)
    
    # Early-accept only for explicit full-sentence cues
    # Accept if sentence starts with '¿' (proper inverted question), otherwise do not
    # blanket-accept just because '?' appears (may be embedded)
    s_trim = sentence.lstrip()
    if s_trim.startswith('¿'):
        return True

    # First check for obvious question indicators (do not auto-accept)
    starts_with_indicator = False
    if language == 'es':
        s = sentence.strip().lower()
        starts_with_indicator = (
            bool(re.match(r"^(qué|dónde|cuándo|cómo|quién|cuál|cuáles|por qué)\b", s)) or
            bool(re.match(r"^(puedes|puede|podrías|podría|quieres|quiere|tienes|tiene|hay|es|está|están|vas|va)\b", s))
        )
        # Broaden indicator signal using generic indicator detector
        try:
            if has_question_indicators(sentence, language):
                starts_with_indicator = True
        except Exception:
            pass
    
    # For Spanish, be extra careful with introductions and statements
    if language == 'es':
        sentence_lower = sentence.lower()
        
        # Check for strong question indicators first
        strong_question_words = ['qué', 'dónde', 'cuándo', 'cómo', 'como', 'quién', 'cuál', 'por qué']
        has_strong_question = any(word in sentence_lower for word in strong_question_words)
        starts_with_question = any(sentence_lower.startswith(word + ' ') for word in strong_question_words)
        
        # If it doesn't have strong question indicators, check for introduction patterns
        if not has_strong_question and not starts_with_question:
            # If it's clearly an introduction or statement, don't use semantic similarity
            introduction_patterns = [
                'soy', 'es', 'estoy', 'está', 'están', 'somos', 'son',
                'mi nombre', 'me llamo', 'vivo en', 'trabajo en', 'estudio en',
                'soy de', 'es de', 'estoy de', 'está de', 'están de',
                'de acuerdo', 'de colombia', 'de españa', 'de méxico', 'de argentina'
            ]
            
            # If it contains introduction patterns, it's likely a statement
            if any(pattern in sentence_lower for pattern in introduction_patterns):
                return False
    
    question_patterns = _get_question_patterns(language)
    if not question_patterns:
        return False
    
    try:
        # Encode sentence and re-use cached pattern embeddings
        sentence_embedding = model.encode([sentence])[0]
        cached = _get_question_pattern_embeddings(language, model)
        if cached is None:
            return False
        question_embeddings = cached
        
        similarities = []
        for q_emb in question_embeddings:
            similarity = cosine_similarity(np.asarray([sentence_embedding]), np.asarray([q_emb]))[0][0]
            similarities.append(similarity)
        
        # Lower threshold for better question detection, but be more conservative for Spanish
        max_similarity = max(similarities)
        cfg = _get_language_config(language)
        if language == 'es':
            thr = cfg.thresholds
            return max_similarity > (thr['semantic_question_threshold_with_indicator'] if starts_with_indicator else thr['semantic_question_threshold_default'])
        else:
            thr = cfg.thresholds
            return max_similarity > thr.get('semantic_question_threshold_default_any', 0.6)
        
    except Exception:
        return False


def has_question_indicators(sentence, language):
    """
    Check for obvious question indicators in the sentence.
    
    Args:
        sentence (str): The sentence to check
        language (str): Language code
    
    Returns:
        bool: True if sentence has question indicators
    """
    # A generic language has no question words; English ones would be guesses.
    if not is_tailored(language):
        return False

    sentence_lower = sentence.lower()
    
    # Question words (Spanish excludes standalone 'por' and 'de'; handled as phrases like 'por qué', 'de quién')
    question_words = {
        'en': ['what', 'where', 'when', 'why', 'how', 'who', 'which', 'whose', 'whom'],
        'es': ['qué', 'dónde', 'cuándo', 'cómo', 'quién', 'cuál', 'cuáles'],
        'de': ['was', 'wo', 'wann', 'warum', 'wie', 'wer', 'welche', 'welches', 'wessen'],
        'fr': ['quoi', 'où', 'quand', 'pourquoi', 'comment', 'qui', 'quel', 'quelle', 'quels', 'quelles'],
        'pt': ['que', 'o que', 'quê', 'onde', 'quando', 'como', 'quem', 'qual', 'quais', 'por que', 'porquê']
    }

    words = question_words.get(language, question_words['en'])
    
    # Check if sentence starts with question words
    for word in words:
        if sentence_lower.startswith(word + ' '):
            return True
    
    # Special case for Spanish: check for question words at the beginning even without ¿
    if language == 'es':
        spanish_question_starters = ES_QUESTION_WORDS_CORE + ['como', 'cuáles']
        for starter in spanish_question_starters:
            if sentence_lower.startswith(starter + ' '):
                return True
        
        # Check for verb-based question starters (present and past tense)
        spanish_verb_starters = [
            'puedes', 'puede', 'pudiste', 'pudo', 'pudieron', 'pudimos',
            'sabes', 'sabe', 'supiste', 'supo', 'supieron',
            'quieres', 'quiere', 'quisiste', 'quiso', 'quisieron',
            'necesitas', 'necesita', 'necesitaste', 'necesitó', 'necesitaron',
            'tienes', 'tiene', 'tuviste', 'tuvo', 'tuvieron',
            'vas', 'va', 'fuiste', 'fue', 'fueron',
            'estás', 'están', 'estuviste', 'estuvo', 'estuvieron'
        ]
        for starter in spanish_verb_starters:
            if sentence_lower.startswith(starter + ' '):
                return True
    
    # Check for question words anywhere in the sentence (for embedded questions)
    # But be more conservative to avoid false positives
    for word in words:
        if ' ' + word + ' ' in sentence_lower:
            # Only consider it a question if it's not a common greeting or statement
            if language == 'es':
                # Avoid false positives for common greetings and statements
                if any(greeting in sentence_lower for greeting in _get_language_config('es').greetings):
                    continue
                if any(statement in sentence_lower for statement in ['gracias', 'por favor', 'de nada', 'no hay problema']):
                    continue
                
                # Avoid false positives for "ser" and "estar" verbs in introductions
                if word in ['soy', 'es', 'estoy', 'está', 'están', 'somos', 'son']:
                    # Check if it's likely an introduction or statement
                    if any(intro_pattern in sentence_lower for intro_pattern in [
                        'yo soy', 'mi nombre es', 'me llamo', 'vivo en', 'trabajo en',
                        'soy de', 'es de', 'estoy de', 'está de', 'están de'
                    ]):
                        continue
                
                # Avoid false positives for "de" when used in locations/descriptions
                if word == 'de':
                    # Check if "de" is used in location patterns (not questions)
                    if any(location_pattern in sentence_lower for location_pattern in [
                        'de colombia', 'de españa', 'de méxico', 'de argentina', 'de santander',
                        'de acuerdo', 'de nada', 'de verdad', 'de hecho'
                    ]):
                        continue
            return True

    # Spanish phrase checks (only as phrases)
    if language == 'es':
        if any(phrase in sentence_lower for phrase in ['por qué', 'de quién', 'a quién']):
            return True
    
    # Check for question marks already present
    if '?' in sentence:
        return True
    
    # Additional Spanish-specific checks to avoid false positives
    if language == 'es':
        # Check if sentence starts with common non-question patterns
        if sentence_lower.startswith(('hola ', 'buenos días ', 'buenas tardes ', 'buenas noches ', 'gracias ', 'por favor ')):
            return False
        
        # Check if sentence contains common statement patterns
        if any(pattern in sentence_lower for pattern in [
            'el proyecto', 'la reunión', 'necesito', 'quiero', 'voy a', 'tengo que',
            'es importante', 'es necesario', 'es correcto', 'está bien'
        ]):
            # Only consider it a question if it has strong question indicators
            has_strong_question = any(word in sentence_lower for word in ES_QUESTION_WORDS_CORE[:-1])
            if not has_strong_question:
                return False
        
        # Prevent false positives with "ser" and "estar" verbs in statements
        # These are common in introductions and descriptions
        if any(pattern in sentence_lower for pattern in [
            'yo soy', 'yo es', 'yo estoy', 'yo está', 'yo están',
            'soy de', 'es de', 'estoy de', 'está de', 'están de',
            'mi nombre es', 'me llamo', 'vivo en', 'trabajo en'
        ]):
            # Only consider it a question if it has strong question indicators
            has_strong_question = any(word in sentence_lower for word in ES_QUESTION_WORDS_CORE[:-1])
            if not has_strong_question:
                return False
        
        # Additional comprehensive check for introduction and statement patterns
        # These patterns are almost never questions in Spanish
        introduction_patterns = [
            'soy', 'es', 'estoy', 'está', 'están', 'somos', 'son',
            'mi nombre', 'me llamo', 'vivo en', 'trabajo en', 'estudio en',
            'soy de', 'es de', 'estoy de', 'está de', 'están de',
            'de acuerdo', 'de colombia', 'de españa', 'de méxico', 'de argentina'
        ]
        
        # If the sentence contains these patterns and doesn't have strong question words, it's likely a statement
        if any(pattern in sentence_lower for pattern in introduction_patterns):
            # Check for strong question indicators
            strong_question_words = ES_QUESTION_WORDS_CORE
            has_strong_question = any(word in sentence_lower for word in strong_question_words)
            
            # Also check if it starts with a question word
            starts_with_question = any(sentence_lower.startswith(word + ' ') for word in strong_question_words)
            
            if not has_strong_question and not starts_with_question:
                return False
    
    # Check for question intonation patterns (common in speech)
    if language == 'en':
        # Common question patterns in English
        if any(pattern in sentence_lower for pattern in [
            'can you', 'could you', 'would you', 'will you', 'do you', 'does', 'did you',
            'are you', 'is this', 'is that', 'are they', 'is it', 'am i'
        ]):
            return True
    elif language == 'es':
        # Common question patterns in Spanish
        # Be more specific to avoid false positives with "ser" and "estar" verbs
        if any(pattern in sentence_lower for pattern in [
            'puedes', 'puede', 'podrías', 'podría', 'vas a', 'va a', 'vas', 'va',
            'tienes', 'tiene', 'tienes que', 'tiene que', 'necesitas', 'necesita',
            'sabes', 'sabe', 'conoces', 'conoce', 'hay',
            'te gusta', 'le gusta', 'te gustaría', 'le gustaría', 'quieres', 'quiere',
            'te parece', 'le parece', 'crees', 'cree', 'piensas', 'piensa'
        ]):
            return True
        # Check for Spanish question word combinations
        if any(pattern in sentence_lower for pattern in [
            'qué hora', 'qué día', 'qué fecha', 'qué tiempo', 'qué tal', 'qué pasa',
            'dónde está', 'dónde vas', 'dónde queda', 'dónde puedo',
            'cuándo es', 'cuándo va', 'cuándo viene', 'cuándo sale',
            'cómo está', 'cómo va', 'cómo te', 'cómo se', 'cómo puedo',
            'quién es', 'quién está', 'quién va', 'quién puede',
            'cuál es', 'cuáles son', 'cuál prefieres', 'cuál te gusta'
        ]):
            return True
    
    return False


def is_exclamation_semantic(sentence: str, model, language: str) -> bool:
    """Determine if a sentence is an exclamation using semantic similarity."""
    if cosine_similarity is None:
        return False
    exclamation_patterns = _get_exclamation_patterns(language)
    if not exclamation_patterns:
        return False
    
    try:
        # Encode sentence and re-use cached pattern embeddings
        sentence_embedding = model.encode([sentence])[0]
        cached = _get_exclamation_pattern_embeddings(language, model)
        if cached is None:
            return False
        exclamation_embeddings = cached
        
        similarities = []
        for e_emb in exclamation_embeddings:
            similarity = cosine_similarity(np.asarray([sentence_embedding]), np.asarray([e_emb]))[0][0]
            similarities.append(similarity)
        
        return max(similarities) > 0.7
        
    except Exception:
        return False


def _get_exclamation_patterns(language):
    """
    Get exclamation patterns for semantic similarity comparison.
    
    Args:
        language (str): Language code
    
    Returns:
        list: List of exclamation patterns
    """
    exclamation_patterns = {
        'en': [
            "That's amazing!",
            "How wonderful!",
            "What a surprise!",
            "I can't believe it!",
            "That's incredible!",
            "How exciting!",
            "What a great idea!",
            "That's fantastic!",
            "How beautiful!",
            "What a relief!"
        ],
        'es': [
            "¡Qué increíble!",
            "¡Qué maravilloso!",
            "¡Qué sorpresa!",
            "¡No puedo creerlo!",
            "¡Qué fantástico!",
            "¡Qué emocionante!",
            "¡Qué gran idea!",
            "¡Qué alivio!",
            "¡Qué hermoso!",
            "¡Qué bueno!"
        ],
        'de': [
            "Das ist unglaublich!",
            "Wie wunderbar!",
            "Was für eine Überraschung!",
            "Ich kann es nicht glauben!",
            "Das ist fantastisch!",
            "Wie aufregend!",
            "Was für eine tolle Idee!",
            "Was für eine Erleichterung!",
            "Wie schön!",
            "Das ist großartig!"
        ],
        'fr': [
            "C'est incroyable!",
            "Comme c'est merveilleux!",
            "Quelle surprise!",
            "Je n'en reviens pas!",
            "C'est fantastique!",
            "Comme c'est excitant!",
            "Quelle excellente idée!",
            "Quel soulagement!",
            "Comme c'est beau!",
            "C'est génial!"
        ],
        'pt': [
            "Que incrível!",
            "Que maravilha!",
            "Que surpresa!",
            "Não acredito!",
            "Que fantástico!",
            "Que emocionante!",
            "Que grande ideia!",
            "Que alívio!",
            "Que lindo!",
            "Isso é ótimo!"
        ]
    }

    # A generic language has no seeds: scoring it against English exclamations
    # would be guessing, so is_exclamation_semantic() returns False instead.
    if not is_tailored(language):
        return []
    return exclamation_patterns.get(language, exclamation_patterns['en'])

def _format_non_spanish_text(text: str, language: str) -> str:
    """Basic capitalization and comma heuristics for non-Spanish languages.

    - Capitalize first letter of each sentence
    - Ensure closing punctuation at end
    - Insert a comma in greeting questions like "Hello how are you" -> "Hello, how are you?"
    - Add common location comma: "from London England" -> "from London, England" (en/de/fr heuristics)
    """
    if not text.strip():
        return text

    # English: normalize dotted acronyms like "U. S.", "D. C." → "US", "DC" to avoid false sentence splits
    if language == 'en':
        def _collapse_acronyms(s: str) -> str:
            # Three-letter sequences: U. S. A. -> USA (allow space or end after final period)
            s = re.sub(r"\b([A-Z])\.\s*([A-Z])\.\s*([A-Z])\.(?=\s|$)", lambda m: ''.join(m.groups()), s)
            # Two-letter sequences: U. S. -> US, D. C. -> DC
            s = re.sub(r"\b([A-Z])\.\s*([A-Z])\.(?=\s|$)", lambda m: ''.join(m.groups()), s)
            # Common compact forms with no spaces: U.S. -> US, D.C. -> DC
            s = re.sub(r"\b([A-Z])\.([A-Z])\.(?=\s|$)", r"\1\2", s)
            return s
        text = _collapse_acronyms(text)

    # Portuguese: protect domains before the punctuation split below.
    #
    # This light path splits on raw '.' without masking, so "exemplo.com" becomes
    # "exemplo. Com". en/fr/de get away with it because the TXT writer's
    # fix_spaced_domains() rejoins "exemplo. com" afterwards -- but Portuguese
    # deliberately suppresses the ".com" rejoin (the word "com" means "with", so
    # rejoining would corrupt ordinary prose like "acabou. Com ele"). Masking here
    # means the domain is never broken in the first place, so pt keeps real
    # domains intact without relying on that round-trip.
    #
    # Scoped to 'pt' on purpose: masking would also fix en/fr/de, but that is a
    # behavior change for them and belongs in its own commit.
    pt_masked = language == 'pt'
    if pt_masked:
        text = mask_domains(text, use_exclusions=True, language='pt')

    # Split keeping punctuation
    parts = _split_sentences_preserving_delims(text)
    sentences = []
    for i in range(0, len(parts), 2):
        if i >= len(parts):
            break
        s = parts[i].strip()
        p = parts[i + 1] if i + 1 < len(parts) else ''
        if not s:
            continue

        # Greeting comma for common patterns, via LanguageConfig
        lower = s.lower()
        cfg_local = _get_language_config(language)
        if cfg_local.greetings:
            if any(lower.startswith(g + ' ') for g in cfg_local.greetings):
                # Insert comma after the greeting token (first word)
                s = re.sub(r'^(\w+)\s+', r"\1, ", s)

        # French clitic hyphenation for inversion/question forms
        if language == 'fr':
            s = _apply_french_hyphenation(s)

        # German-specific enhancements
        if language == 'de':
            # Capitalize "Ich" when preceded by punctuation internally
            s = re.sub(r'([,;:])\s*ich\b', r'\1 Ich', s)
            # Capitalize Herr/Frau + Name
            s = _capitalize_german_titles(s)
            # High-confidence proper nouns (cities/countries)
            s = _capitalize_german_proper_nouns(s)
            # Insert commas before common subordinating conjunctions if preceded by ≥3 words
            s = _apply_german_commas(s)
            # Capitalize nouns after determiners (high-confidence heuristic)
            s = _capitalize_german_nouns_after_determiners(s)
            # Capitalize Deutsch in the common phrase "Deutsch gelernt"
            s = re.sub(r'\bdeutsch\b(?=\s+gelernt\b)', 'Deutsch', s, flags=re.IGNORECASE)

        # Capitalize first alpha, unless the sentence opens with a (masked) domain
        # such as "exemplo__DOT__com" -- "Exemplo.com" is wrong. Mirrors the guard
        # the Spanish path applies before its own sentence capitalization.
        if s and s[0].isalpha():
            first_token = s.split(' ', 1)[0]
            if SINGLE_MASK not in first_token:
                s = s[0].upper() + s[1:]

        # Light location comma heuristic (English/French/German)
        s = re.sub(r'\bfrom\s+([A-Z][a-zA-Zäöüßéèàç]+)\s+([A-Z][a-zA-Zäöüßéèàç]+)\b', r'from \1, \2', s)
        s = re.sub(r'\bde\s+([A-Z][\wäöüßéèàç]+)\s+([A-Z][\wäöüßéèàç]+)\b', r'de \1, \2', s)
        s = re.sub(r'\baus\s+([A-Z][\wäöüßéèàç]+)\s+([A-Z][\wäöüßéèàç]+)\b', r'aus \1, \2', s)
        # Portuguese uses de/do/da/dos/das + location ("de Lisboa, Portugal").
        # Bare "de" is already handled above; add the article contractions.
        if language == 'pt':
            s = re.sub(
                rf'\b(do|da|dos|das|em)\s+([A-Z][\w{UPPER_ACCENTED}{LOWER_ACCENTED}]+)'
                rf'\s+([A-Z][\w{UPPER_ACCENTED}{LOWER_ACCENTED}]+)\b',
                r'\1 \2, \3', s)

        # Ensure punctuation
        if not p:
            # Use question mark if starts with typical question words
            starts_question = False
            starters = _get_language_config(language).question_starters
            lower_s = s.lower()
            for w in starters:
                if lower_s.startswith(w + ' '):
                    starts_question = True
                    break
            p = '?' if starts_question else '.'

        sentences.append(s + p)

    # Cleanup spacing
    out = ' '.join(sentences)
    out = re.sub(r'\s+([,!.?])', r'\1', out)
    # Ensure a space after commas only when not followed by a digit (to preserve thousands groups)
    out = re.sub(r',(?=\S)(?!\d)', ', ', out)
    out = re.sub(r'\s+', ' ', out).strip()
    if pt_masked:
        out = unmask_domains(out)
    return out


def _apply_french_hyphenation(sentence: str) -> str:
    """Apply common French hyphenation rules for clitic inversion in questions.

    Examples:
    - "Comment allez vous" -> "Comment allez-vous"
    - "Pouvez vous m'aider" -> "Pouvez-vous m'aider"
    - "Sommes nous prêts" -> "Sommes-nous prêts"
    - "Y a il" -> "Y a-t-il"
    - "Va il" -> "Va-t-il"
    - "est ce que" -> "est-ce que"
    - "qu est ce que" -> "qu'est-ce que"
    """
    s = sentence
    # Normalize multiple spaces
    s = re.sub(r"\s+", " ", s)

    # est-ce que
    s = re.sub(r"\b([Ee])st\s*ce\s*que\b", r"\1st-ce que", s)
    # qu'est-ce que variants
    s = re.sub(r"\b([Qq])u[' ]?\s*est\s*ce\s*que\b", lambda m: ("Qu" if m.group(1).isupper() else "qu") + "'est-ce que", s)

    # General verb-pronoun inversion hyphenation
    pron = r"(vous|tu|il|elle|on|ils|elles|je|nous)"
    verbs = (
        r"êtes|sommes|sont|suis|est|allons|allez|vont|va|ai|as|avons|avez|ont|"
        r"peux|peut|pouvez|pouvons|peuvent|pourriez|voudriez|voulez|voulais|"
        r"savez|sais|savons|saurez|pensez|pense|faut|faites|fais|faisons|"
        r"souvenez|parlez|parlons|parlez|voulez|êtes"
    )
    s = re.sub(rf"\b({verbs})\s+{pron}\b", r"\1-\2", s, flags=re.IGNORECASE)

    # Euphonic -t- insertion between vowel-ending verb and il/elle/on
    s = re.sub(r"\b(va|vont|a|ont|fera|feront|ira|iront|est)-(il|elle|on)\b", r"\1-t-\2", s, flags=re.IGNORECASE)

    # "y a-t-il" pattern
    s = re.sub(r"\b([Yy])\s*a\s*-?\s*(il|elle|on)\b", lambda m: ("Y" if m.group(1).isupper() else "y") + " a-t-" + m.group(2), s)

    # Clean any doubled hyphens
    s = re.sub(r"-{2,}", "-", s)
    return s


def _apply_german_commas(sentence: str) -> str:
    """Insert a comma before common subordinating conjunctions when reasonably safe.

    Heuristic: add comma before dass|weil|ob|wenn when there are at least 3 words before
    the conjunction in the sentence and no comma directly precedes it.
    """
    targets = {"dass", "weil", "ob", "wenn"}
    tokens = sentence.split()
    if len(tokens) < 5:
        return sentence
    result_tokens = []
    for i, tok in enumerate(tokens):
        lower_tok = tok.lower()
        if lower_tok in targets and i >= 3:
            # If previous token already ends with a comma, don't add another
            if result_tokens and not result_tokens[-1].endswith(','):
                # Insert comma before the conjunction
                result_tokens[-1] = result_tokens[-1] + ','
        result_tokens.append(tok)
    return ' '.join(result_tokens)


def _capitalize_german_titles(sentence: str) -> str:
    """Capitalize names after Herr/Frau titles."""
    def repl(m):
        title = m.group(1)
        name = m.group(2)
        return f"{title} {name[:1].upper()}{name[1:].lower()}"
    return re.sub(r"\b(Herr|Frau)\s+([a-zäöüß][a-zäöüß\-]*)\b", repl, sentence)


def _capitalize_german_proper_nouns(sentence: str) -> str:
    """Capitalize a small whitelist of high-confidence proper nouns (cities/countries)."""
    proper_map = {
        # Countries
        'deutschland': 'Deutschland', 'österreich': 'Österreich', 'schweiz': 'Schweiz',
        # Major cities
        'berlin': 'Berlin', 'münchen': 'München', 'hamburg': 'Hamburg', 'köln': 'Köln', 'frankfurt': 'Frankfurt',
        'stuttgart': 'Stuttgart', 'hannover': 'Hannover', 'bremen': 'Bremen', 'leipzig': 'Leipzig', 'dresden': 'Dresden',
        'zürich': 'Zürich', 'bern': 'Bern', 'basel': 'Basel', 'wien': 'Wien', 'salzburg': 'Salzburg'
    }
    def replace_word(m: re.Match) -> str:
        w: str = m.group(0)
        lw = w.lower()
        return proper_map.get(lw, w)
    # Replace whole-word occurrences only
    pattern = r"\b(" + '|'.join(map(re.escape, proper_map.keys())) + r")\b"
    return re.sub(pattern, replace_word, sentence, flags=re.IGNORECASE)


def _capitalize_german_nouns_after_determiners(sentence: str) -> str:
    """Capitalize likely nouns following German determiners/possessives.

    This is a conservative heuristic: only capitalize the immediate next token
    if it is lowercase, length ≥ 4, and not already capitalized.
    """
    det_pattern = (
        r"der|die|das|den|dem|des|"
        r"ein|eine|einer|eines|einem|einen|"
        r"dies(?:er|e|es|em|en)?|jen(?:er|e|es|em|en)?|"
        r"welch(?:er|e|es|em|en)?|jed(?:er|e|es|em|en)?|manch(?:er|e|es|em|en)?|solch(?:er|e|es|em|en)?|"
        r"kein(?:er|e|es|em|en)?|"
        r"mein(?:e|em|en|es)?|dein(?:e|em|en|es)?|sein(?:e|em|en|es)?|"
        r"ihr(?:e|em|en|es)?|unser(?:e|em|en|es)?|euer(?:e|em|en|es)?|Ihr(?:e|em|en|es)?"
    )
    regex = re.compile(rf"\b((?:{det_pattern})\s+)([a-zäöüß][a-zäöüß\-]{{3,}})\b")

    def repl(m: re.Match) -> str:
        prefix = m.group(1)
        word = m.group(2)
        # Avoid capitalizing if token looks like an adjective with common endings and followed by another lowercase token (likely noun)
        # Keep it simple: still capitalize; heuristics beyond this tend to degrade.
        cap = word[:1].upper() + word[1:].lower()
        return prefix + cap

    return regex.sub(repl, sentence)

def _get_question_patterns(language):
    """
    Get question patterns for semantic similarity comparison.
    
    Args:
        language (str): Language code
    
    Returns:
        list: List of question patterns
    """
    question_patterns = {
        'en': [
            "What is this?",
            "Where are you?",
            "When will it happen?",
            "Why did you do that?",
            "How does it work?",
            "Who is there?",
            "Which one do you prefer?",
            "Can you help me?",
            "Could you explain?",
            "Would you like to go?",
            "Will you come?",
            "Do you understand?",
            "Are you ready?"
        ],
        'es': [
            "¿Qué es esto?",
            "¿Dónde estás?",
            "¿Cuándo pasará?",
            "¿Por qué lo hiciste?",
            "¿Cómo funciona?",
            "¿Quién está ahí?",
            "¿Cuál prefieres?",
            "¿Puedes ayudarme?",
            "¿Podrías explicar?",
            "¿Te gustaría ir?",
            "¿Vas a venir?",
            "¿Haces esto?",
            "¿Eres listo?",
            "¿Qué hora es?",
            "¿Qué día es hoy?",
            "¿Dónde está la reunión?",
            "¿Cuándo es la cita?",
            "¿Cómo estás?",
            "¿Quién puede ayudarme?",
            "¿Cuál es tu nombre?",
            "¿Puedes enviarme la agenda?",
            "¿Tienes tiempo?",
            "¿Sabes dónde queda?",
            "¿Hay algo más?",
            "¿Está todo bien?",
            "¿Te parece bien?",
            "¿Quieres que vayamos?",
            "¿Crees que es correcto?",
            "¿Necesitas ayuda?",
            "¿Va a llover hoy?",
            "¿Estás listo?",
            "¿Puedo ayudarte?"
        ],
        'de': [
            "Was ist das?",
            "Wo bist du?",
            "Wann passiert es?",
            "Warum hast du das gemacht?",
            "Wie funktioniert es?",
            "Wer ist da?",
            "Welches bevorzugst du?",
            "Kannst du mir helfen?",
            "Könntest du erklären?",
            "Würdest du gerne gehen?",
            "Wirst du kommen?",
            "Machst du das?",
            "Bist du bereit?"
        ],
        'fr': [
            "Qu'est-ce que c'est?",
            "Où es-tu?",
            "Quand cela arrivera-t-il?",
            "Pourquoi as-tu fait cela?",
            "Comment ça marche?",
            "Qui est là?",
            "Lequel préfères-tu?",
            "Peux-tu m'aider?",
            "Pourrais-tu expliquer?",
            "Voudrais-tu aller?",
            "Vas-tu venir?",
            "Fais-tu cela?",
            "Es-tu prêt?"
        ],
        'pt': [
            "O que é isso?",
            "Onde você está?",
            "Quando isso vai acontecer?",
            "Por que você fez isso?",
            "Como funciona?",
            "Quem está aí?",
            "Qual você prefere?",
            "Você pode me ajudar?",
            "Poderia explicar?",
            "Você gostaria de ir?",
            "Você vem?",
            "Você entende?",
            "Está pronto?"
        ]
    }

    # A generic language has no seeds (see _get_exclamation_patterns).
    if not is_tailored(language):
        return []
    return question_patterns.get(language, question_patterns['en'])


# (Module has no __main__ block; import-only.)