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
#
# - Spanish keyword/constants:
#   * ES_QUESTION_WORDS_CORE, ES_QUESTION_STARTERS_EXTRA
#   * ES_GREETINGS, ES_CONNECTORS, ES_POSSESSIVES
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
from typing import Optional
from dataclasses import dataclass
import os
import numpy as np
import warnings
from domain_utils import mask_domains, unmask_domains, apply_safe_text_processing, create_domain_aware_regex

# Suppress PyTorch FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Module logger for library-friendly messaging
logger = logging.getLogger("podscripter.punctuation")


# Helper functions for domain-safe text processing
def _domain_safe_split_preserving_delims(text: str, pattern: str = r'([.!?]+)') -> list[str]:
    """Split text while preserving domains and delimiters."""
    def _split_func(text_to_split):
        return re.split(pattern, text_to_split)
    result = apply_safe_text_processing(text, _split_func, use_exclusions=True)
    if isinstance(result, list):
        return result
    return [result]


def _domain_safe_regex_replace(text: str, pattern: str, replacement: str) -> str:
    """Apply regex replacement while preserving domains."""
    replace_func = create_domain_aware_regex(pattern, replacement, use_exclusions=True)
    return replace_func(text)

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
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("SentenceTransformers not available. Advanced punctuation restoration may be limited.")


# Optional spaCy import for NLP-based capitalization
try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False


_SENTENCE_TRANSFORMER_SINGLETON = None

"""
Lightweight utilities and caches
"""

# Precompiled, shared regexes
PUNCT_SPLIT_RE = re.compile(r'([.!?]+)')

# Spanish language config: connectors and possessives
ES_POSSESSIVES = {
    "tu", "tus", "su", "sus", "mi", "mis", "nuestro", "nuestra", "nuestros", "nuestras"
}
ES_CONNECTORS = {
    # Core function words that should not be capitalized mid-sentence
    "de", "del", "la", "las", "los", "el", "lo",
    "y", "e", "o", "u",
    "en", "a", "con", "por", "para", "sin", "sobre", "entre",
    # Foreign particles seen in names
    "da", "di", "do", "du", "van", "von", "des", "le",
    # Possessives/determiners
    *ES_POSSESSIVES,
}

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

# Language thresholds (centralized for tuning)
def _get_language_thresholds(language: str) -> dict:
    """Return thresholds controlling semantic gating and splitting heuristics.

    Values are chosen to preserve current behavior.
    """
    if language == 'es':
        return {
            # Slightly lower thresholds to improve recall of genuine questions in Spanish
            'semantic_question_threshold_with_indicator': 0.64,
            'semantic_question_threshold_default': 0.74,
            'min_total_words_no_split': 30,
            'min_chunk_before_split': 20,
            'min_chunk_inside_question': 25,
            'min_chunk_capital_break': 38,
            'min_chunk_semantic_break': 42,
        }
    # Defaults for other languages (align with existing logic)
    return {
        'semantic_question_threshold_default_any': 0.60,
        'min_total_words_no_split': 25,
        'min_chunk_before_split': 15,
        'min_chunk_inside_question': 20,
        'min_chunk_capital_break': 20,
        'min_chunk_semantic_break': 25,
    }


@dataclass
class LanguageConfig:
    connectors: set
    possessives: set
    thresholds: dict
    greetings: list
    question_starters: list


def _get_language_config(language: str) -> LanguageConfig:
    # Generic defaults
    default_connectors = {
        "de", "del", "la", "las", "los", "el", "lo",
        "y", "e", "o", "u",
        "en", "a", "con", "por", "para", "sin", "sobre", "entre",
        "da", "di", "do", "du", "van", "von", "des", "le",
    }
    if language == 'es':
        return LanguageConfig(
            connectors=ES_CONNECTORS,
            possessives=ES_POSSESSIVES,
            thresholds=_get_language_thresholds(language),
            greetings=ES_GREETINGS,
            question_starters=ES_QUESTION_WORDS_CORE + ES_QUESTION_STARTERS_EXTRA,
        )
    if language == 'fr':
        return LanguageConfig(
            connectors=default_connectors,
            possessives=set(),
            thresholds=_get_language_thresholds(language),
            greetings=FR_GREETINGS,
            question_starters=FR_QUESTION_STARTERS,
        )
    if language == 'de':
        return LanguageConfig(
            connectors=default_connectors,
            possessives=set(),
            thresholds=_get_language_thresholds(language),
            greetings=DE_GREETINGS,
            question_starters=DE_QUESTION_STARTERS,
        )
    if language == 'en':
        return LanguageConfig(
            connectors=default_connectors,
            possessives=set(),
            thresholds=_get_language_thresholds(language),
            greetings=['hello'],
            question_starters=EN_QUESTION_STARTERS,
        )
    return LanguageConfig(
        connectors=default_connectors,
        possessives=set(),
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


# Final universal cleanup applied at the end of the pipeline
def _finalize_text_common(text: str) -> str:
    """Apply safe, language-agnostic cleanup at the very end.

    - Normalize mixed terminal punctuation
    - Normalize whitespace
    - Ensure a space after sentence punctuation before capital letters
    """
    if not text:
        return text
    out = _normalize_mixed_terminal_punctuation(text)
    out = re.sub(r"\s+", " ", out)
    # Use centralized domain masking with Spanish exclusions
    masked = mask_domains(out, use_exclusions=True, language='es')
    # Ensure single space after sentence punctuation when followed by a letter (including lowercase accented)
    masked = re.sub(r"(?<!\.)\.\s*([A-Za-zÁÉÍÓÚÑáéíóúñ])", r". \1", masked)
    masked = re.sub(r"\?\s*([A-Za-zÁÉÍÓÚÑáéíóúñ])", r"? \1", masked)
    masked = re.sub(r"!\s*([A-Za-zÁÉÍÓÚÑáéíóúñ])", r"! \1", masked)
    # Capitalize after terminators when appropriate
    masked = re.sub(r"([.!?])\s+([a-záéíóúñ])", lambda m: f"{m.group(1)} {m.group(2).upper()}", masked)
    # Unmask domains using centralized function
    out = unmask_domains(masked)
    # Normalize comma spacing globally with thousands-aware behavior
    # 1) Remove spaces before commas everywhere
    out = re.sub(r"\s+,", ",", out)
    # 2) Collapse spaces only inside thousands-grouped numbers
    out = re.sub(r"(?<!\d)(\d{1,3})(?:,\s?\d{3})+(?!\d)", lambda m: re.sub(r',\s+', ',', m.group(0)), out)
    # 3) Ensure a single space after commas except when followed by a digit (to preserve thousands and enumerations)
    out = re.sub(r",(?!\d)\s*", ", ", out)
    return out.strip()


# --- Cross-language helpers exposed for orchestration ---
def _normalize_dotted_acronyms_en(text: str) -> str:
    """Collapse dotted uppercase acronyms to avoid false sentence splits in English.

    Examples:
      - "U. S. A." → "USA", "U. S." → "US", "U.S." → "US"
    """
    if not text:
        return text
    # Three-letter sequences: U. S. A. → USA
    text = re.sub(r"\b([A-Z])\.\s*([A-Z])\.\s*([A-Z])\.(?=[\s\-\)\]\}\,\"'`:;]|$)", lambda m: ''.join(m.groups()), text)
    # Two-letter sequences (spaced): U. S. → US, D. C. → DC
    text = re.sub(r"\b([A-Z])\.\s*([A-Z])\.(?=[\s\-\)\]\}\,\"'`:;]|$)", lambda m: ''.join(m.groups()), text)
    # Two-letter sequences (compact): U.S. → US, D.C. → DC
    text = re.sub(r"\b([A-Z])\.([A-Z])\.(?=[\s\-\)\]\}\,\"'`:;]|$)", r"\1\2", text)
    return text


# (Removed) emphatic repeat merging to simplify maintenance

def _split_processed_segment(processed: str, language: str) -> tuple[list[str], str]:
    """Split a single, punctuation-restored segment into sentences.

    Preserves ellipses (… and ...), and avoids breaking inside domains (label.tld).
    Returns (sentences, trailing_fragment_without_terminal_punct).
    The trailing fragment should be carried into the next segment by the caller if desired.
    """
    parts = re.split(r'(…|[.!?]+)', processed)
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
                tld = next_tld_match.group(1).lower()
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

        # End without explicit punctuation → return trailing buffer to caller
        if idx + 1 >= len(parts):
            break
        idx += 2

    trailing = buffer.strip()
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
    if (language or '').lower() == 'fr' and sentences:
        sentences = _fr_merge_short_connector_breaks(sentences)
    return sentences, trailing

# --- Spanish helper utilities (pure refactors of existing logic) ---
def _es_greeting_and_leadin_commas(text: str) -> str:
    """Normalize greeting comma usage and common lead-ins for Spanish.

    Examples:
    - "Hola como estan. ¿Listos?" -> "Hola, ¿como estan?"
    - "Hola, a todos" -> "Hola a todos" (remove comma before prepositional phrase)
    - "Como siempre vamos a..." -> "Como siempre, vamos a..."
    - "¡Hola a todos! ¡Bienvenidos a …!" -> "Hola a todos, ¡bienvenidos a …!"
    """
    # 0) Combine adjacent greeting + welcome exclamations into a single sentence with a comma
    #    Gate: only at paragraph start (start-of-text or after blank line) and when content is short
    def _combine_greeting_welcome(m: re.Match) -> str:
        prefix = m.group(1) or ''
        greeting_tail = (m.group(2) or '').strip()
        welcome = (m.group(3) or '').strip()
        if len(greeting_tail.split()) > 6 or len(welcome.split()) > 10:
            return m.group(0)
        if welcome:
            welcome = welcome[:1].lower() + welcome[1:]
        return f"{prefix}Hola{(' ' + greeting_tail) if greeting_tail else ''}, ¡{welcome}!"

    text = re.sub(
        r"(?is)(^|\n{2,})¡\s*hola\s*([^!\n]*)!\s*¡\s*(bienvenid[oa]s[^!]*)!",
        _combine_greeting_welcome,
        text,
    )

    # Greeting punctuation: if a greeting sentence ends with a period and followed by a question, turn period into comma
    text = re.sub(r"(\bHola[^.!?]*?)\.\s+(¿)", r"\1, \2", text, flags=re.IGNORECASE)
    # Also handle 'como' without accent
    text = re.sub(r"(\bHola[^.!?]*?)\.\s+(como\s+estan)\b", r"\1, ¿\2?", text, flags=re.IGNORECASE)
    # Remove comma right after "Hola" when followed by a prepositional phrase (more natural Spanish)
    text = re.sub(r"(^|[\n\.\?¡!]\s*)Hola,\s+(a|para)\b", r"\1Hola \2", text, flags=re.IGNORECASE)
    # Ensure comma after "Como siempre," lead-in
    text = re.sub(r"(?i)\b(Como siempre)(?!,)\b", r"\1,", text)

    # 1) Add comma after other common lead-ins when followed by a clause (accent-insensitive)
    #    Gate: only at sentence-initial positions (start, newline, or after terminator+space)
    leadins = [
        r"como\s+siempre",
        r"entonces",
        r"bueno",
        r"pues",
        r"adem[aá]s",
        r"as[ií]\s+que",
    ]
    text = re.sub(
        rf"(?i)(^|(?<=\n)|(?<=[\.!?]\s))((?:{'|'.join(leadins)}))(?!,)\b",
        lambda m: f"{m.group(1)}{m.group(2)},",
        text,
    )

    # 2) Normalize greeting without comma before following question/exclamation (sentence-initial only)
    text = re.sub(r"((^|(?<=\n)|(?<=[\.!?]\s))Hola[^.!?]*?)(\s+)([¿¡])", r"\1, \4", text, flags=re.IGNORECASE)
    return text


def _es_wrap_imperative_exclamations(text: str) -> str:
    """Wrap common imperative/greeting starters with exclamation marks when safe.

    Examples:
    - "Vamos a empezar." -> "¡Vamos a empezar!"
    - "Bienvenidos a Españolistos." -> "¡Bienvenidos a Españolistos!"
    (No change if already a question/exclamation.)
    """
    def _wrap_exclamations(line: str) -> str:
        s = line.strip()
        if not s:
            return line
        low = s.lower()
        if s.startswith('¿') or s.endswith('?') or s.startswith('¡') or s.endswith('!'):
            return line
        if re.match(r"^(bienvenidos|empecemos|vamos|dile|atención)\b", low):
            return '¡' + s + '!'
        return line

    parts = _split_sentences_preserving_delims(text)
    for i in range(0, len(parts), 2):
        if i >= len(parts):
            break
        s = parts[i]
        if s:
            wrapped = _wrap_exclamations(s)
            if wrapped != s:
                parts[i] = wrapped
                # Drop following punctuation token if present to avoid duplicates like "!." or "!?"
                if i + 1 < len(parts):
                    parts[i + 1] = ''
    out = ''.join(parts)

    # Ensure that sentences starting with '¡' end with a single '!'
    parts2 = _split_sentences_preserving_delims(out)
    for i in range(0, len(parts2), 2):
        if i >= len(parts2):
            break
        s = parts2[i] or ''
        p = parts2[i + 1] if i + 1 < len(parts2) else ''
        s_stripped = s.strip()
        if not s_stripped:
            continue
        if s_stripped.startswith('¡'):
            if s_stripped.endswith('!'):
                if i + 1 < len(parts2) and p:
                    parts2[i + 1] = ''
            else:
                if i + 1 < len(parts2) and p:
                    parts2[i + 1] = '!'
                else:
                    parts2[i] = s.rstrip(' .?') + '!'
    out = ''.join(parts2)
    # Final mixed-punctuation cleanup after exclamation pairing
    out = re.sub(r'!\s*\.', '!', out)
    out = re.sub(r'!\s*\?', '!', out)
    return out

def _es_capitalize_sentence_starts(text: str) -> str:
    """Capitalize the first alphabetical character of each sentence for Spanish.

    Preserves leading punctuation/quotes and capitalizes after '¿' or '¡'.
    """
    parts = _split_sentences_preserving_delims(text)
    for i in range(0, len(parts), 2):
        if i >= len(parts):
            break
        s = parts[i] or ''
        if not s:
            continue
        idx = 0
        n = len(s)
        # Skip leading whitespace
        while idx < n and s[idx].isspace():
            idx += 1
        # Skip opening quotes/brackets/dashes
        while idx < n and s[idx] in '"“"«»([({—-':
            idx += 1
        # Skip inverted punctuation then spaces
        if idx < n and s[idx] in '¡¿':
            idx += 1
            while idx < n and s[idx].isspace():
                idx += 1
        if idx < n and s[idx].islower():
            parts[i] = s[:idx] + s[idx].upper() + s[idx+1:]
    return ''.join(parts)

def _es_normalize_tag_questions(text: str) -> str:
    """Normalize common Spanish tag questions and ensure comma before the tag.

    Examples:
    - "Está bien, ¿no." -> "Está bien, ¿no?"
    - "Está bien ¿verdad?" -> "Está bien, ¿verdad?"
    """
    out = text
    out = re.sub(r",\s*¿\s*(no|cierto|verdad)\s*\.\b", r", ¿\1?", out, flags=re.IGNORECASE)
    out = re.sub(r"([^?,\.\s])\s+¿\s*(no|cierto|verdad)\s*\?", r"\1, ¿\2?", out, flags=re.IGNORECASE)
    return out

def _es_fix_collocations(text: str) -> str:
    """Repair frequent collocations.

    Examples:
    - "por? supuesto" -> "Por supuesto"
    - Sentence-initial "Por supuesto" -> "Por supuesto,"
    """
    out = text
    out = re.sub(r"\b[Pp]or\s*\?\s*[Ss]upuesto\b", "Por supuesto", out)
    out = re.sub(r"(^|[\n\.!?]\s*)(Por supuesto)(\b)", r"\1\2,", out)
    return out

def _es_pair_inverted_questions(text: str) -> str:
    """Ensure Spanish inverted question pairing and consistency.

    Rules:
    - Convert "¿ ... ." to "¿ ... ?"
    - If a segment ends with '?' but lacks opening '¿', add it
    - If a segment contains '¿' but lacks a closing '?', append one before boundary

    Examples:
    - "¿Dónde está." -> "¿Dónde está?"
    - "Cómo estás?" -> "¿Cómo estás?"
    - "Dijo: ¿qué hacemos. Mañana" -> "Dijo: ¿qué hacemos? Mañana"
    """
    # Ensure paired punctuation consistency: "¿ ... ." -> "¿ ... ?"
    out = re.sub(r"¿\s*([^?\n]+)\.", r"¿\1?", text)

    # Ensure opening inverted question mark for any Spanish question lacking it (including embedded)
    parts = _split_sentences_preserving_delims(out)
    for i in range(0, len(parts), 2):
        if i >= len(parts):
            break
        s = (parts[i] or '').strip()
        p = parts[i + 1] if i + 1 < len(parts) else ''
        if not s:
            continue
        # Full-sentence question: add opening '¿' if missing and no existing embedded '¿'
        if p == '?' and '¿' not in s and not s.startswith('¿'):
            # If it erroneously starts with '¡', convert to '¿'
            if s.startswith('¡'):
                s = s[1:].lstrip()
            parts[i] = '¿' + s
        # Embedded question: contains '¿' but no '?', add before boundary
        if '¿' in s and p != '?':
            if '?' not in s:
                parts[i] = s + '?'
    return ''.join(parts)

def _es_merge_possessive_splits(text: str) -> str:
    """Merge possessive/function-word splits and lowercase the following noun.

    Examples:
    - "tu. Español" -> "tu español"
    - "mi. Amigo" -> "mi amigo"
    """
    return re.sub(
        r"\b(tu|su|mi)\s*\.\s+([A-Za-zÁÉÍÓÚÑáéíóúñ][\wÁÉÍÓÚÑáéíóúñ-]*)",
        lambda m: m.group(1) + " " + m.group(2).lower(),
        text,
        flags=re.IGNORECASE,
    )

def _es_merge_aux_gerund(text: str) -> str:
    """Merge auxiliary + gerund splits.

    Example:
    - "Estamos. Hablando" -> "Estamos hablando"
    """
    return re.sub(
        r"(?i)\b(Estoy|Estás|Está|Estamos|Están)\.\s+([a-záéíóúñ]+(?:ando|iendo|yendo))\b",
        r"\1 \2",
        text,
    )

def _es_merge_capitalized_one_word_sentences(text: str) -> str:
    """Merge consecutive one-word capitalized sentences.

    Example:
    - "Estados. Unidos." -> "Estados Unidos."
    """ 
    return re.sub(
        r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)\.(\s+)([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)\.",
        r"\1 \3.",
        text,
    )

def _es_intro_location_appositive_commas(text: str) -> str:
    """Add appositive commas for introductions and locations.

    Example:
    - "Yo soy Andrea de Santander, Colombia" -> "Yo soy Andrea, de Santander, Colombia"
    """
    def _intro_loc_commas(m: re.Match) -> str:
        cue = m.group(1)
        name = m.group(2).strip()
        place1 = m.group(3).strip()
        place2 = m.group(4).strip() if m.group(4) else None
        out = f"{cue} {name}, de {place1}"
        if place2:
            out += f", {place2}"
        return out
    # Allow an optional comma after the name (e.g., "Yo soy Andrea, de …")
    return re.sub(
        r"(?i)\b(Y yo soy|Yo soy)\s+([^,\n]+?)\s*,?\s+de\s+([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑ-]+)\s*,?\s*([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑ-]+)?",
        _intro_loc_commas,
        text,
    )

# Embeddings caches for static pattern sets per language
_QUESTION_PATTERN_EMBEDDINGS = {}
_EXCL_PATTERN_EMBEDDINGS = {}

def _get_question_patterns(language: str) -> list[str]:
    """Return curated question pattern prompts per language for semantic gating.

    These are short exemplars used to build an embedding bank that helps
    classify whether a sentence is interrogative. Keep lists small and
    language-appropriate. For languages without curation, return a minimal set.
    """
    lang = (language or 'en').lower()
    if lang == 'es':
        # Core interrogatives and common verb-first yes/no starters
        core = [
            'qué', 'dónde', 'cuándo', 'cómo', 'quién', 'cuál', 'por qué',
        ]
        starters = [
            'puedes', 'puede', 'podrías', 'podría', 'quieres', 'quiere',
            'tienes', 'tiene', 'hay', 'es', 'está', 'están', 'vas', 'va',
            'te parece', 'le parece', 'crees', 'cree', 'piensas', 'piensa',
        ]
        # Include accented/non-accented variants for robustness
        variants = list(core) + [s for s in starters]
        return [f'{w} … ?' for w in variants] + [
            '¿…?', '¿Podemos …?', '¿Te gustaría …?', '¿Hay …?',
        ]
    if lang == 'fr':
        return [
            'qui … ?', 'quoi … ?', 'où … ?', 'quand … ?', 'pourquoi … ?', 'comment … ?',
            'est-ce que … ?',
        ]
    if lang == 'de':
        return [
            'wer … ?', 'was … ?', 'wo … ?', 'wann … ?', 'warum … ?', 'wie … ?',
            'ist … ?', 'sind … ?', 'gibt es … ?',
        ]
    # English/minimal default
    return [
        'what … ?', 'where … ?', 'when … ?', 'why … ?', 'how … ?', 'who … ?', 'is … ?', 'are … ?',
    ]

def _get_exclamation_patterns(language: str) -> list[str]:
    """Return curated exclamation exemplars per language for semantic gating."""
    lang = (language or 'en').lower()
    if lang == 'es':
        return [
            '¡qué increíble!', '¡qué maravilloso!', '¡qué sorpresa!', '¡no puedo creerlo!',
            '¡qué fantástico!', '¡qué emocionante!', '¡qué gran idea!', '¡qué alivio!',
            '¡qué hermoso!', '¡qué bueno!', '¡vamos!', '¡bienvenidos!', '¡empecemos!',
        ]
    if lang == 'fr':
        return [
            "c'est incroyable !", 'quelle surprise !', 'formidable !', 'incroyable !',
        ]
    if lang == 'de':
        return [
            'das ist unglaublich!', 'wunderbar!', 'fantastisch!', 'toll!',
        ]
    # English/minimal default
    return [
        'amazing!', 'incredible!', 'unbelievable!', 'what a relief!', "let's go!", 'attention!',
    ]

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

    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None

    st_cache, hf_cache = _get_cache_paths()

    # Ensure HF_HOME points to our preferred cache to consolidate downloads
    os.environ.setdefault("HF_HOME", hf_cache)

    # First try: if a local model directory exists, prefer offline mode.
    # Only load by direct path if Sentence-Transformers metadata is present (modules.json),
    # otherwise skip to name+cache load to avoid the "Creating a new one with mean pooling" warning.
    local_model_dir = _find_local_model_path(st_cache, model_name.split('/')[-1])
    if local_model_dir and os.path.isdir(local_model_dir):
        # Enable offline to avoid any network HEAD calls when cache is warm
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        modules_json = os.path.join(local_model_dir, "modules.json")
        config_st = os.path.join(local_model_dir, "config_sentence_transformers.json")
        if os.path.isfile(modules_json) or os.path.isfile(config_st):
            try:
                _SENTENCE_TRANSFORMER_SINGLETON = SentenceTransformer(local_model_dir)
                return _SENTENCE_TRANSFORMER_SINGLETON
            except Exception:
                # Fallback to normal loading below
                pass

    # Second try: load by name but direct cache_folder to sentence-transformers cache
    try:
        _SENTENCE_TRANSFORMER_SINGLETON = SentenceTransformer(model_name, cache_folder=st_cache)
        return _SENTENCE_TRANSFORMER_SINGLETON
    except Exception:
        # Last resort: try short name without org (older sbert versions)
        short_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        _SENTENCE_TRANSFORMER_SINGLETON = SentenceTransformer(short_name, cache_folder=st_cache)
        return _SENTENCE_TRANSFORMER_SINGLETON


def restore_punctuation(text: str, language: str = 'en') -> str:
    """
    Restore punctuation to transcribed text using advanced NLP techniques.
    
    Args:
        text (str): The transcribed text without proper punctuation
        language (str): Language code ('en', 'es', 'de', 'fr')
    
    Returns:
        str: Text with restored punctuation
    """
    if not text.strip():
        return text
    
    # Clean up the text first
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Use advanced punctuation restoration
    try:
        return _advanced_punctuation_restoration(text, language, True)  # Enable custom patterns by default
    except Exception as e:
        logger.warning(f"Advanced punctuation restoration failed: {e}")
        logger.info("Returning original text without punctuation restoration.")
        return text


def _advanced_punctuation_restoration(text: str, language: str = 'en', use_custom_patterns: bool = True) -> str:
    """
    Advanced punctuation restoration using sentence transformers and NLP techniques.
    
    Args:
        text (str): The transcribed text without proper punctuation
        language (str): Language code ('en', 'es', 'de', 'fr')
        use_custom_patterns (bool): Whether to use custom sentence endings and question word patterns
    
    Returns:
        str: Text with restored punctuation
    """
    
    # Use SentenceTransformers for better sentence boundary detection
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        return _transformer_based_restoration(text, language, use_custom_patterns)
    else:
        # Simple fallback: just clean up whitespace and add basic punctuation
        text = re.sub(r'\s+', ' ', text.strip())
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        return text


def _transformer_based_restoration(text: str, language: str = 'en', use_custom_patterns: bool = True) -> str:
    """
    Improved punctuation restoration using SentenceTransformers for semantic understanding.
    
    Args:
        text (str): The transcribed text without proper punctuation
        language (str): Language code ('en', 'es', 'de', 'fr')
        use_custom_patterns (bool): Whether to use custom patterns
    
    Returns:
        str: Text with restored punctuation
    """
    # Initialize the model once (use multilingual model for better language support)
    model = _load_sentence_transformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    if model is None:
        # Fallback path if sentence-transformers is unavailable
        text = re.sub(r'\s+', ' ', text.strip())
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        return text

    # 1) Semantic split into sentences (token-based loop with _should_end_sentence_here)
    sentences = _semantic_split_into_sentences(text, language, model)
    # 2) Punctuate sentences and join
    result = _punctuate_semantic_sentences(sentences, model, language)
    
    # Apply Spanish-specific formatting
    if language == 'es':
        # Domain-preserving sentence split and formatting
        # CRITICAL: Mask domains before calling assemble_sentences_from_processed to prevent re-splitting
        tld_alt = r"com|net|org|co|es|io|edu|gov|uk|us|ar|mx"
        result_masked = re.sub(rf"\b([a-z0-9\-]{{3,}})\.({tld_alt})\b", r"\1__DOT__\2", result, flags=re.IGNORECASE)
        sentences_list, trailing_fragment = assemble_sentences_from_processed(result_masked, 'es')
        # Unmask domains in the resulting sentences
        sentences_list = [re.sub(r"__DOT__", ".", s) for s in sentences_list]
        if trailing_fragment:
            trailing_fragment = re.sub(r"__DOT__", ".", trailing_fragment)
        if trailing_fragment:
            cleaned = re.sub(r'^[",\s]+', '', trailing_fragment)
            if cleaned and not cleaned.endswith(('.', '!', '?')):
                cleaned += '.'
            if cleaned:
                sentences_list.append(cleaned)

        formatted_sentences = []
        for s in sentences_list:
            sentence = (s or '').strip()
            if not sentence:
                continue
            # Capitalize first letter
            if sentence and sentence[0].isalpha():
                sentence = sentence[0].upper() + sentence[1:]

            # Ensure sentence ends with single terminal punctuation
            if not sentence.endswith(('.', '!', '?')):
                question_words = ES_QUESTION_WORDS_CORE + ES_QUESTION_STARTERS_EXTRA
                sentence_lower = sentence.lower()
                if any(word in sentence_lower for word in question_words):
                    sentence += '?'
                else:
                    sentence += '.'
            else:
                sentence = sentence.rstrip('.!?') + sentence[-1]

            # Clean up duplicates
            sentence = re.sub(r'[.!?]{2,}', lambda m: m.group(0)[0], sentence)
            sentence = re.sub(r'¿{2,}', '¿', sentence)
            formatted_sentences.append(sentence)

        result = ' '.join(formatted_sentences)
        
        # Add inverted question marks for questions (comprehensive approach)
        # First, identify all sentences that end with question marks
        # CRITICAL: Mask domains before splitting to prevent breaking them
        result_masked_for_split = re.sub(rf"\b([a-z0-9\-]{{3,}})\.({tld_alt})\b", r"\1__DOT__\2", result, flags=re.IGNORECASE)
        sentences = _split_sentences_preserving_delims(result_masked_for_split)
        # Unmask domains in the split sentences
        sentences = [re.sub(r"__DOT__", ".", s) for s in sentences]
        for i in range(0, len(sentences), 2):
            if i >= len(sentences):
                break
            sentence_text = sentences[i].strip()
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
            # If the sentence ends with a question mark, it should start with ¿
            if punctuation == '?' and sentence_text and not sentence_text.startswith('¿'):
                sentence_lower = sentence_text.lower()
                # Question patterns that should have inverted question marks
                question_patterns = [
                    r'^(qué|dónde|cuándo|cómo|quién|cuál|por qué|recuerdas|sabes|puedes|quieres|necesitas|tienes|vas|estás|están|pueden|saben|quieren|hay|va|es|son|está|están)',
                    r'^(puedes|puede|podrías|podría|sabes|sabe|quieres|quiere|necesitas|necesita|tienes|tiene|vas|va|estás|están|pueden|quieren)',
                    r'^(hay|va|es|son|está|están|te parece|le parece|crees|cree|piensas|piensa)',
                    r'^(estamos|están|listos|listas|listo|lista|bien|mal|correcto|incorrecto|verdad|cierto)'
                ]
                if any(re.search(pattern, sentence_lower) for pattern in question_patterns):
                    # If leading is '¡', switch it to '¿'
                    if sentence_text.startswith('¡'):
                        sentence_text = sentence_text[1:].lstrip()
                    sentences[i] = '¿' + sentence_text
        result = ''.join(sentences)
        
        # Clean up double/mixed punctuation in one place
        result = _normalize_mixed_terminal_punctuation(result)
        
        # Special handling for greetings and lead-ins
        result = _es_greeting_and_leadin_commas(result)
        
        # Remove the comma addition for location patterns - Spanish grammar doesn't require it
        # result = re.sub(r'\b(andrea|carlos|maría|juan|ana)\s+(de)\s+(colombia|santander|madrid|españa|méxico|argentina)\b', r'\1 \2, \3', result, flags=re.IGNORECASE)
        
        # Ensure proper sentence separation
        # CRITICAL: Mask domains before space insertion to prevent breaking them
        result_masked_for_separation = re.sub(rf"\b([a-z0-9\-]{{3,}})\.({tld_alt})\b", r"\1__DOT__\2", result, flags=re.IGNORECASE)
        result_masked_for_separation = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', result_masked_for_separation)
        result = re.sub(r"__DOT__", ".", result_masked_for_separation)
        
        # Insert comma after common Spanish greeting starters
        result = re.sub(r'(^|[\n\.\!?]\s*)(Hola)\s+', r"\1\2, ", result, flags=re.IGNORECASE)
        result = re.sub(r'(^|[\n\.\!?]\s*)(Buenos días)\s+', r"\1\2, ", result, flags=re.IGNORECASE)
        result = re.sub(r'(^|[\n\.\!?]\s*)(Buenas tardes)\s+', r"\1\2, ", result, flags=re.IGNORECASE)
        result = re.sub(r'(^|[\n\.\!?]\s*)(Buenas noches)\s+', r"\1\2, ", result, flags=re.IGNORECASE)
        
        # Final cleanup of any remaining double punctuation
        result = _normalize_mixed_terminal_punctuation(result)
        
        # General fix: merge sentences that were incorrectly split at question words
        # This handles cases where a question was split in the middle
        result = re.sub(r'(\w+)\?\.¿(\w+)', r'¿\1 \2', result)
    
        # BUG FIXES FOR SPANISH TRANSCRIPTION ISSUES
        
        # Bug 1: Fix missing punctuation at the end of sentences
        # Ensure all sentences end with proper punctuation
        # CRITICAL: Mask domains before splitting to prevent breaking them
        result_masked_for_punct = re.sub(rf"\b([a-z0-9\-]{{3,}})\.({tld_alt})\b", r"\1__DOT__\2", result, flags=re.IGNORECASE)
        sentences = _split_sentences_preserving_delims(result_masked_for_punct)
        # Unmask domains in the split sentences
        sentences = [re.sub(r"__DOT__", ".", s) for s in sentences]
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i].strip()
                if sentence and not sentence.endswith(('.', '!', '?')):
                    # Check if it's a question - be more specific about question patterns
                    sentence_lower = sentence.lower()
                    
                    # Only treat as question if it clearly starts with a question word
                    question_starters = ES_QUESTION_WORDS_CORE
                    starts_with_question = any(sentence_lower.startswith(word + ' ') for word in question_starters)
                    
                    # For embedded questions, only treat as question if it starts with ¿
                    if starts_with_question or sentence.startswith('¿'):
                        sentence += '?'
                    else:
                        sentence += '.'
                    sentences[i] = sentence
        result = ''.join(sentences)
        
        # Additional fix: ensure short phrases get proper punctuation
        # Handle cases like "También sí" that don't get punctuation
        result = re.sub(r'\b(también sí|sí|no|claro|exacto|perfecto|vale|bien)\s*$', r'\1.', result, flags=re.IGNORECASE)
        
        # Fix any remaining sentences without punctuation at the end
        # This catches any sentences that might have been missed
        # CRITICAL: Mask domains before splitting to prevent breaking them  
        result_masked_for_final = re.sub(rf"\b([a-z0-9\-]{{3,}})\.({tld_alt})\b", r"\1__DOT__\2", result, flags=re.IGNORECASE)
        sentences = _split_sentences_preserving_delims(result_masked_for_final)
        # Unmask domains in the split sentences
        sentences = [re.sub(r"__DOT__", ".", s) for s in sentences]
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i].strip()
                if sentence and not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                    sentences[i] = sentence
        result = ''.join(sentences)
        
        # Final fix for specific short phrases that might be missed
        result = re.sub(r'\b(también sí)\s*$', r'\1.', result, flags=re.IGNORECASE)
        result = re.sub(r'\b(sí)\s*$', r'\1.', result, flags=re.IGNORECASE)
        result = re.sub(r'\b(no)\s*$', r'\1.', result, flags=re.IGNORECASE)
        result = re.sub(r'\b(pues tranquilo)\s*$', r'\1.', result, flags=re.IGNORECASE)
        
        # Additional comprehensive fix for any remaining sentences without punctuation
        # Split by sentences and ensure each one ends with punctuation
        # CRITICAL: Mask domains before splitting to prevent breaking them
        result_masked_for_comprehensive = re.sub(rf"\b([a-z0-9\-]{{3,}})\.({tld_alt})\b", r"\1__DOT__\2", result, flags=re.IGNORECASE)
        sentences = re.split(r'([.!?]+)', result_masked_for_comprehensive)
        # Unmask domains in the split sentences
        sentences = [re.sub(r"__DOT__", ".", s) for s in sentences]
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i].strip()
                if sentence and not sentence.endswith(('.', '!', '?')):
                    # Special handling for short phrases
                    if sentence.lower() in ['también sí', 'sí', 'no', 'claro', 'exacto', 'perfecto', 'vale', 'bien', 'pues tranquilo']:
                        sentence += '.'
                    else:
                        sentence += '.'
                    sentences[i] = sentence
        result = ''.join(sentences)
        
        # Bug 2: Fix sentences ending in commas
        # Replace trailing commas with proper punctuation
        result = re.sub(r',\s*$', '.', result)  # End of text
        result = re.sub(r',\s*([.!?])', r'\1', result)  # Before other punctuation
        # CRITICAL: Mask domains before comma-to-period conversion to prevent breaking them
        result_masked_for_comma = re.sub(rf"\b([a-z0-9\-]{{3,}})\.({tld_alt})\b", r"\1__DOT__\2", result, flags=re.IGNORECASE)
        result_masked_for_comma = re.sub(r',\s*([A-Z])', r'. \1', result_masked_for_comma)  # Before capital letters (new sentence)
        result = re.sub(r"__DOT__", ".", result_masked_for_comma)
        
        # Specific fix for short phrases ending in comma
        result = re.sub(r'\b(también sí|sí|no|claro|exacto|perfecto|vale|bien)\s*,', r'\1.', result, flags=re.IGNORECASE)
        
        # Bug 4: Fix question marks followed by inverted question marks in the middle
        # Remove the problematic pattern "?¿" in the middle of sentences
        result = re.sub(r'\?\s*¿(\w+)', r' \1', result)  # ?¿ followed by word
        result = re.sub(r'(\w+)\?\s*¿(\w+)', r'\1? \2', result)  # word?¿word
        
        # Additional cleanup for Spanish-specific patterns
        # Fix sentences that start with ¿ but don't end with ?
        # CRITICAL: Mask domains before splitting to prevent breaking them
        result_masked_for_patterns = re.sub(rf"\b([a-z0-9\-]{{3,}})\.({tld_alt})\b", r"\1__DOT__\2", result, flags=re.IGNORECASE)
        sentences = re.split(r'([.!?]+)', result_masked_for_patterns)
        # Unmask domains in the split sentences
        sentences = [re.sub(r"__DOT__", ".", s) for s in sentences]
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i].strip()
                if sentence.startswith('¿') and not sentence.endswith('?'):
                    # Remove the ¿ and add proper punctuation
                    sentence = sentence[1:]  # Remove ¿
                    if not sentence.endswith(('.', '!', '?')):
                        sentence += '.'
                    sentences[i] = sentence
        result = ''.join(sentences)
        
        # Final cleanup: ensure proper spacing and remove any remaining artifacts
        result = re.sub(r'\s+', ' ', result)  # Normalize whitespace
        result = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', result)  # Proper sentence separation
        result = re.sub(r'¿{2,}', '¿', result)  # Remove duplicate ¿
        result = re.sub(r'\?{2,}', '?', result)  # Remove duplicate ?
        result = re.sub(r'\.{2,}', '.', result)  # Remove duplicate .
        result = re.sub(r'!{2,}', '!', result)  # Remove duplicate !
        
        # Fix double punctuation patterns like .? or ?., and !.
        result = re.sub(r'\.\s*\?', '?', result)  # .? -> ?
        result = re.sub(r'\?\s*\.', '?', result)  # ?. -> ?
        result = re.sub(r'!\s*\.', '!', result)    # !. -> !
        result = re.sub(r'!\s*\?', '!', result)   # !? -> !
        result = re.sub(r'\?\s*!', '!', result)   # ?! -> !
        
        # Clean up any remaining mixed punctuation
        result = re.sub(r'[.!?]{2,}', lambda m: m.group(0)[0], result)
        
        # Ensure all sentences end with proper punctuation
        if result and not result.endswith(('.', '!', '?')):
            result += '.'
        
        # Final comprehensive fix for any remaining issues
        # Handle the specific case of "También sí" and similar short phrases
        result = re.sub(r'\b(también sí)\s*$', r'\1.', result, flags=re.IGNORECASE)
        result = re.sub(r'\b(sí)\s*$', r'\1.', result, flags=re.IGNORECASE)
        result = re.sub(r'\b(no)\s*$', r'\1.', result, flags=re.IGNORECASE)
        
        # Also handle these phrases when they appear at the end of a sentence
        result = re.sub(r'\b(también sí)\s*([.!?])', r'\1.', result, flags=re.IGNORECASE)
        result = re.sub(r'\b(sí)\s*([.!?])', r'\1.', result, flags=re.IGNORECASE)
        result = re.sub(r'\b(no)\s*([.!?])', r'\1.', result, flags=re.IGNORECASE)
        
        # FINAL STEP: Add inverted question marks for all questions
        # This runs after all punctuation has been added
        # Use semantic gating to decide whether to keep/add inverted question marks
        # CRITICAL: Mask domains before splitting to prevent breaking them
        result_masked_for_semantic = re.sub(rf"\b([a-z0-9\-]{{3,}})\.({tld_alt})\b", r"\1__DOT__\2", result, flags=re.IGNORECASE)
        sentences = re.split(r'([.!?]+)', result_masked_for_semantic)
        # Unmask domains in the split sentences
        sentences = [re.sub(r"__DOT__", ".", s) for s in sentences]
        model_for_gate = _load_sentence_transformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence_text = sentences[i].strip()
                punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
                if not sentence_text:
                    continue
                # Normalize whitespace inside sentence
                sentence_text = re.sub(r"\s+", " ", sentence_text)
                # If it looks like a question (either starts with ¿ or ends with ?), use semantic gate
                looks_question = punctuation == '?' or sentence_text.startswith('¿')
                if looks_question and model_for_gate is not None:
                    core = sentence_text[1:] if sentence_text.startswith('¿') else sentence_text
                    if is_question_semantic(core, model_for_gate, 'es'):
                        # Ensure starts with ¿ and ends with ?
                        if not core.startswith('¿'):
                            sentences[i] = '¿' + core
                        else:
                            sentences[i] = sentence_text
                        sentences[i + 1] = '?' if i + 1 < len(sentences) else '?'
                    else:
                        # Not a question: strip leading ¿ and ensure period
                        core = core.rstrip(' ?!').strip()
                        sentences[i] = core
                        sentences[i + 1] = '.'
        result = ''.join(sentences)

        # Apply targeted Spanish cleanup after semantic gating
        result = _spanish_cleanup_postprocess(result)

        # Optional spaCy capitalization pass (env NLP_CAPITALIZATION=1)
        if os.environ.get('NLP_CAPITALIZATION', '0') == '1':
            result = _apply_spacy_capitalization(result, language)
        # (Removed) final collapse of emphatic repeats
        # Normalize comma spacing with thousands-aware behavior
        # 1) Remove spaces before commas
        result = re.sub(r'\s+,', ',', result)
        # 2) Collapse spaces only inside thousands-grouped numbers
        result = re.sub(r'(?<!\d)(\d{1,3})(?:,\s?\d{3})+(?!\d)', lambda m: re.sub(r',\s+', ',', m.group(0)), result)
        # 3) Ensure a single space after commas except when followed by a digit
        result = re.sub(r',(?!\d)\s*', ', ', result)
        # Ensure a single space after terminal punctuation when followed by a non-space and not part of an ellipsis
        # CRITICAL: Mask domains before space insertion to prevent breaking them
        result_masked_for_spacing = re.sub(rf"\b([a-z0-9\-]{{3,}})\.({tld_alt})\b", r"\1__DOT__\2", result, flags=re.IGNORECASE)
        result_masked_for_spacing = re.sub(r'(?<!\.)\.([^\s.])', r'. \1', result_masked_for_spacing)
        result = re.sub(r"__DOT__", ".", result_masked_for_spacing)
        result = re.sub(r'\?\s*(?=\S)', r'? ', result)
        result = re.sub(r'!\s*(?=\S)', r'! ', result)
        # Capitalize the first letter after sentence terminators when appropriate (Spanish sentences start capitalized)
        def _cap_after_terminator(m):
            punct = m.group(1)
            ch = m.group(2)
            return f"{punct} {ch.upper()}"
        result = re.sub(r'([.!?])\s+([a-záéíóúñ])', _cap_after_terminator, result)
        # Final domain TLD lowercasing safeguard (after all formatting/capitalization)
        result = re.sub(r"\b([a-z0-9\-]+)\.([A-Za-z]{2,24})\b", lambda m: f"{m.group(1)}.{m.group(2).lower()}", result, flags=re.IGNORECASE)
    else:
        # Apply light, language-aware formatting for non-Spanish languages
        result = _format_non_spanish_text(result, language)

        # Optional spaCy capitalization pass (env NLP_CAPITALIZATION=1)
        if os.environ.get('NLP_CAPITALIZATION', '0') == '1':
            result = _apply_spacy_capitalization(result, language)

    # Final universal cleanup
    return _finalize_text_common(result)


from typing import List


def _semantic_split_into_sentences(text: str, language: str, model) -> List[str]:
    """Split text into sentences using semantic boundary checks.

    Returns a list of raw sentences (without per-language post-formatting).
    """
    # CRITICAL: Mask domains before splitting to prevent semantic splitter from breaking them
    # Support both single and compound TLDs, exclude common Spanish words to avoid false positives
    text_masked = mask_domains(text, use_exclusions=True, language=language)
    
    words = text_masked.split()
    if len(words) < 3:
        # Unmask before returning using centralized function
        unmasked = unmask_domains(text_masked)
        return [unmasked]

    sentences: List[str] = []
    current_chunk: List[str] = []
    for i, word in enumerate(words):
        current_chunk.append(word)
        if _should_end_sentence_here(words, i, current_chunk, model, language):
            sentence_text = ' '.join(current_chunk).strip()
            if sentence_text:
                # Unmask domains before adding to sentences using centralized function
                sentence_text = unmask_domains(sentence_text)
                sentences.append(sentence_text)
            current_chunk = []
    if current_chunk:
        sentence_text = ' '.join(current_chunk).strip()
        if sentence_text:
            # Unmask domains before adding to sentences using centralized function
            sentence_text = unmask_domains(sentence_text)
            sentences.append(sentence_text)
    return sentences


def _punctuate_semantic_sentences(sentences: List[str], model, language: str) -> str:
    """Apply semantic punctuation to each sentence and join into a string.

    Mirrors the previous in-function logic to preserve behavior.
    """
    processed: List[str] = []
    total = len(sentences)
    for i, sentence in enumerate(sentences):
        s = (sentence or '').strip()
        if not s:
            continue
        punctuated = _apply_semantic_punctuation(s, model, language, i, total)
        processed.append(punctuated)

    out = ' '.join(processed)
    # Final cleanup (preserve previous behavior)
    out = re.sub(r'\s+', ' ', out)
    out = re.sub(r'\s+\.', '.', out)
    out = re.sub(r'\s+\?', '?', out)
    out = re.sub(r'\s+\!', '!', out)
    return out


def _should_end_sentence_here(words: List[str], current_index: int, current_chunk: List[str], model, language: str) -> bool:
    """
    Determine if a sentence should end at the current position using semantic coherence.
    
    Args:
        words (list): List of all words in the text
        current_index (int): Current word index
        current_chunk (list): Current sentence chunk
        model: SentenceTransformer model
        language (str): Language code
    
    Returns:
        bool: True if sentence should end here
    """
    # Don't end sentence if we're at the beginning
    if current_index < 2:
        return False
    
    # Don't end sentence if we're at the very end
    if current_index >= len(words) - 1:
        return False
    
    # Minimum sentence length (avoid very short sentences)
    # For very short inputs, don't split at all
    cfg = _get_language_config(language)
    thresholds = cfg.thresholds
    if len(words) <= thresholds.get('min_total_words_no_split', 25):  # If the entire input is short, don't split
        return False
    
    if len(current_chunk) < thresholds.get('min_chunk_before_split', 18):  # Slightly less aggressive
        return False
    
    # Check for natural sentence endings
    current_word = words[current_index]
    next_word = words[current_index + 1] if current_index + 1 < len(words) else ""
    
    # Strong indicators as soft hints (never hard breaks)
    strong_end_indicators = _get_strong_end_indicators(language)
    if current_word.lower() in strong_end_indicators:
        # For Spanish, never end on clause-internal "no"/"sí"
        if language == 'es' and current_word.lower() in {'no', 'sí'}:
            return False
        # Require capital-break gate and semantic corroboration
        if not (next_word and next_word[:1].isupper() and
                len(current_chunk) >= thresholds.get('min_chunk_capital_break', 28)):
            return False
        # Don't split if next token is a connector/determiner
        next_low = next_word.lower() if next_word else ''
        connectors = _get_language_config(language).connectors
        if next_low in connectors or next_low in {'los', 'las', 'el', 'la', 'de', 'del'}:
            return False
        # Only split if semantic break agrees (and we have enough context length)
        if len(current_chunk) >= thresholds.get('min_chunk_semantic_break', 30):
            return _check_semantic_break(words, current_index, model)
        return False
    
    # Spanish-specific sentence breaking patterns
    if language == 'es':
        # Don't break questions in the middle - include strong starters as well
        question_words_core = ES_QUESTION_WORDS_CORE
        question_words_ext = ES_QUESTION_STARTERS_EXTRA
        question_words_all = set(w.lower() for w in (question_words_core + question_words_ext))
        current_text = ' '.join(current_chunk).lower()
        if any(word in current_text for word in question_words_all):
            # If we're in the middle of a question, don't break unless it's very long
            if len(current_chunk) < thresholds.get('min_chunk_inside_question', 25):
                return False
        
        # Don't break in the middle of common Spanish phrases
        # Check if we're in the middle of a phrase that should stay together
        current_text = ' '.join(current_chunk).lower()
        
        # General rule: don't break after articles, prepositions, determiners, or common quantifiers
        # These words typically continue the sentence
        if current_word.lower() in ['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'de', 'en', 'con', 'por', 'para', 'sin', 'sobre', 'entre', 'tras', 'durante', 'mediante', 'según', 'hacia', 'hasta', 'desde', 'contra',
                                    'todo', 'toda', 'todos', 'todas', 'alguno', 'alguna', 'algunos', 'algunas', 'cualquier', 'cualquiera', 'ningún', 'ninguna', 'ninguno', 'otro', 'otra', 'otros', 'otras']:
            return False
        
        # General rule: don't break after "que" followed by common verbs
        if current_word.lower() == 'que' and next_word and next_word.lower() in ['ha', 'han', 'he', 'hemos', 'has', 'habéis', 'ha', 'han', 'está', 'están', 'es', 'son', 'era', 'eran', 'fue', 'fueron']:
            return False
        
        # General rule: don't break after "no" when it's part of a negative construction
        # Include a broader set of likely verb forms (heuristic by endings)
        if current_word.lower() == 'no' and next_word:
            if next_word.lower() in ['es', 'son', 'está', 'están', 'ha', 'han', 'puede', 'pueden', 'debe', 'deben']:
                return False
            # Heuristic: common Spanish verb endings (past/present/imperfect/future/gerund)
            if re.search(r"(?i)(?:o|as|a|amos|an|es|e|emos|en|imos|\b|aba|abas|aban|ía|ías|ían|é|aste|ó|amos|aron|í|iste|ió|imos|ieron|ará|erá|irá|rán|remos|yendo|ando|iendo|ado|ido)$", next_word):
                return False

        # Don't break after possessive determiners like "tu", "su", "mi" etc.
        if current_word.lower() in ['tu', 'su', 'mi', 'tus', 'sus', 'mis', 'nuestro', 'nuestra', 'nuestros', 'nuestras']:
            return False
        
        # More general check: don't break if the next word is a common Spanish conjunction or preposition
        # that typically continues the sentence
        if next_word and next_word.lower() in ['y', 'o', 'pero', 'mas', 'sino', 'aunque', 'como', 'que', 'de', 'en', 'con', 'por', 'para', 'sin', 'sobre', 'entre', 'tras', 'durante', 'mediante', 'según', 'hacia', 'hasta', 'desde', 'contra']:
            return False
        
        # Don't break if the current word ends with a preposition that should continue
        if current_word.lower() in ['de', 'en', 'con', 'por', 'para', 'sin', 'sobre', 'entre', 'tras', 'durante', 'mediante', 'según', 'hacia', 'hasta', 'desde', 'contra']:
            return False
        
    # Check for capital letter after reasonable length (be much more conservative for Spanish ASR capitalization)
    next_next_word = words[current_index + 2] if current_index + 2 < len(words) else ""
    if (next_word and next_word[0].isupper() and
        len(current_chunk) >= thresholds.get('min_chunk_capital_break', 28) and  # conservative
        not _is_continuation_word(current_word, language) and
        not _is_transitional_word(current_word, language)):
        # Avoid breaking when followed by two capitalized tokens (likely multi-word proper noun: "Estados Unidos")
        if next_next_word and next_next_word[0].isupper():
            return False
        # Avoid breaking after comma or after preposition like "de"
        if current_word.endswith(',') or current_word.lower() in ['de', 'en']:
            return False
        # Avoid breaking when the next word is a determiner/preposition starting a continuation
        if next_word.lower() in ['los', 'las', 'el', 'la', 'de', 'del']:
            return False
        return True
    
    # Use semantic coherence to determine if chunks should be separated
    if len(current_chunk) >= thresholds.get('min_chunk_semantic_break', 30):  # conservative semantic split
        # If the next token is a lowercase continuation or preposition, do not break
        if next_word and next_word and next_word[0].islower():
            if next_word.lower() in ['y', 'o', 'pero', 'que', 'de', 'en', 'para', 'con', 'por', 'sin', 'sobre']:
                return False
        return _check_semantic_break(words, current_index, model)
    
    return False


def _get_strong_end_indicators(language):
    """
    Get strong indicators that suggest a sentence should end.
    
    Args:
        language (str): Language code
    
    Returns:
        list: List of strong end indicators
    """
    indicators = {
        'en': ['thank', 'thanks', 'goodbye', 'bye', 'okay', 'ok', 'right', 'sure', 'yes', 'no'],
        'es': ['gracias', 'adiós', 'hasta', 'vale', 'bien', 'sí', 'no', 'claro'],
        'de': ['danke', 'tschüss', 'auf', 'wiedersehen', 'okay', 'ja', 'nein', 'klar'],
        'fr': ['merci', 'au', 'revoir', 'salut', 'okay', 'oui', 'non', 'd\'accord']
    }
    return indicators.get(language, indicators['en'])


def _is_transitional_word(word: str, language: str) -> bool:
    """
    Check if a word is a transitional word that suggests the sentence should continue.
    
    Args:
        word (str): The word to check
        language (str): Language code
    
    Returns:
        bool: True if word suggests continuation
    """
    transitional_words = {
        'en': ['then', 'next', 'after', 'before', 'while', 'during', 'since', 'until', 'when', 'where', 'if', 'unless', 'although', 'though', 'even', 'though', 'despite', 'in', 'spite', 'of'],
        'es': ['entonces', 'después', 'antes', 'mientras', 'durante', 'desde', 'hasta', 'cuando', 'donde', 'si', 'aunque', 'a', 'pesar', 'de', 'que', 'los', 'las', 'en', 'de', 'con', 'por', 'para', 'sin', 'sobre', 'entre', 'detrás', 'delante', 'cerca', 'lejos'],
        'de': ['dann', 'nächste', 'nach', 'vor', 'während', 'seit', 'bis', 'wenn', 'wo', 'falls', 'obwohl', 'trotz'],
        'fr': ['alors', 'après', 'avant', 'pendant', 'depuis', 'jusqu\'à', 'quand', 'où', 'si', 'bien', 'que', 'malgré']
    }
    
    words = transitional_words.get(language, transitional_words['en'])
    return word.lower() in words


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
        'es': ['y', 'o', 'pero', 'así', 'porque', 'si', 'cuando', 'mientras', 'desde', 'aunque', 'sin', 'embargo', 'por', 'tanto', 'entonces', 'también', 'además', 'furthermore', 'más', 'aún'],
        'de': ['und', 'oder', 'aber', 'also', 'weil', 'wenn', 'während', 'seit', 'obwohl', 'jedoch', 'daher', 'deshalb', 'dann', 'auch', 'außerdem', 'ferner', 'zudem'],
        'fr': ['et', 'ou', 'mais', 'donc', 'parce', 'si', 'quand', 'pendant', 'depuis', 'bien', 'que', 'cependant', 'donc', 'alors', 'aussi', 'de', 'plus', 'en', 'outre', 'par', 'ailleurs']
    }
    
    words = continuation_words.get(language, continuation_words['en'])
    return word.lower() in words


def _check_semantic_break(words: List[str], current_index: int, model) -> bool:
    """
    Check if there's a semantic break at the current position.
    
    Args:
        words (list): List of all words
        current_index (int): Current position
        model: SentenceTransformer model
    
    Returns:
        bool: True if there's a semantic break
    """
    try:
        # Create text chunks before and after current position
        before_text = ' '.join(words[:current_index + 1])
        after_text = ' '.join(words[current_index + 1:current_index + 10])  # Look ahead 10 words
        
        if not before_text.strip() or not after_text.strip():
            return False
        
        # Calculate semantic similarity
        embeddings = model.encode([before_text, after_text])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # Lower similarity suggests a semantic break
        return similarity < 0.6
        
    except Exception:
        # If semantic analysis fails, be conservative
        return False


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
    # Check if it's a question using semantic similarity
    if is_question_semantic(sentence, model, language):
        if not sentence.endswith('?'):
            sentence = sentence.rstrip('.!') + '?'
        return sentence
    
    # Check for exclamation patterns
    if is_exclamation_semantic(sentence, model, language):
        if not sentence.endswith('!'):
            sentence = sentence.rstrip('.?') + '!'
        return sentence
    
    # Default to period if no other punctuation
    if sentence and not sentence.endswith(('.', '!', '?')):
        # Don't add period if it ends with a continuation word
        if not _is_continuation_word(sentence.split()[-1] if sentence.split() else '', language):
            sentence += '.'
    
    return sentence


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
            similarity = cosine_similarity([sentence_embedding], [q_emb])[0][0]
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
    sentence_lower = sentence.lower()
    
    # Question words (Spanish excludes standalone 'por' and 'de'; handled as phrases like 'por qué', 'de quién')
    question_words = {
        'en': ['what', 'where', 'when', 'why', 'how', 'who', 'which', 'whose', 'whom'],
        'es': ['qué', 'dónde', 'cuándo', 'cómo', 'quién', 'cuál', 'cuáles'],
        'de': ['was', 'wo', 'wann', 'warum', 'wie', 'wer', 'welche', 'welches', 'wessen'],
        'fr': ['quoi', 'où', 'quand', 'pourquoi', 'comment', 'qui', 'quel', 'quelle', 'quels', 'quelles']
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
            similarity = cosine_similarity([sentence_embedding], [e_emb])[0][0]
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
        ]
    }
    
    return exclamation_patterns.get(language, exclamation_patterns['en'])


def _apply_basic_punctuation_rules(sentence, language, use_custom_patterns):
    """
    Apply basic punctuation rules to a sentence.
    
    Args:
        sentence (str): The sentence to process
        language (str): Language code
        use_custom_patterns (bool): Whether to use custom patterns (now deprecated with SentenceTransformers)
    
    Returns:
        str: Sentence with basic punctuation applied
    """
    # Handle repeated words for emphasis (still useful for some languages)
    if language == 'es':
        # Add commas between repeated "sí" or "no" for emphasis
        sentence = re.sub(r'\b(sí)\s+\1\b', r'\1, \1', sentence, flags=re.IGNORECASE)
        sentence = re.sub(r'\b(no)\s+\1\b', r'\1, \1', sentence, flags=re.IGNORECASE)
    elif language == 'fr':
        # Add commas between repeated "oui" or "non" for emphasis
        sentence = re.sub(r'\b(oui)\s+\1\b', r'\1, \1', sentence, flags=re.IGNORECASE)
        sentence = re.sub(r'\b(non)\s+\1\b', r'\1, \1', sentence, flags=re.IGNORECASE)
    elif language == 'de':
        # Add commas between repeated "ja" or "nein" for emphasis
        sentence = re.sub(r'\b(ja)\s+\1\b', r'\1, \1', sentence, flags=re.IGNORECASE)
        sentence = re.sub(r'\b(nein)\s+\1\b', r'\1, \1', sentence, flags=re.IGNORECASE)
    
    # Add period if sentence doesn't end with punctuation
    if sentence and not sentence.endswith(('.', '!', '?')):
        # Don't add period if it ends with a conjunction
        if not sentence.lower().endswith(('and', 'or', 'but', 'so', 'because', 'if', 'when', 'while', 'since', 'although')):
            sentence += '.'
    
    return sentence

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

        # Capitalize first alpha
        if s and s[0].isalpha():
            s = s[0].upper() + s[1:]

        # Light location comma heuristic (English/French/German)
        s = re.sub(r'\bfrom\s+([A-Z][a-zA-Zäöüßéèàç]+)\s+([A-Z][a-zA-Zäöüßéèàç]+)\b', r'from \1, \2', s)
        s = re.sub(r'\bde\s+([A-Z][\wäöüßéèàç]+)\s+([A-Z][\wäöüßéèàç]+)\b', r'de \1, \2', s)
        s = re.sub(r'\baus\s+([A-Z][\wäöüßéèàç]+)\s+([A-Z][\wäöüßéèàç]+)\b', r'aus \1, \2', s)

        # Ensure punctuation
        if not p:
            # Use question mark if starts with typical question words
            starts_question = False
            starters = _get_language_config(language).question_starters or EN_QUESTION_STARTERS
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
    return out


# ---------------- NLP Capitalization (spaCy) ----------------
_SPACY_PIPELINES = {}

def _get_spacy_pipeline(language: str):
    if not SPACY_AVAILABLE:
        return None
    if language in _SPACY_PIPELINES:
        return _SPACY_PIPELINES[language]
    model_map = {
        'en': 'en_core_web_sm',
        'es': 'es_core_news_sm',
        'fr': 'fr_core_news_sm',
        'de': 'de_core_news_sm',
    }
    name = model_map.get(language)
    if not name:
        return None
    try:
        nlp = spacy.load(name, disable=["lemmatizer"])  # speed
        _SPACY_PIPELINES[language] = nlp
        return nlp
    except Exception:
        return None


def _apply_spacy_capitalization(text: str, language: str) -> str:
    """Capitalize named entities and proper nouns using spaCy, conservatively.

    - Capitalize tokens in entities PERSON/ORG/GPE/LOC
    - Capitalize tokens with POS=PROPN
    - Preserve particles: de, del, la, las, los, y, o, da, di, do, du, van, von, du, des, le
    - Skip URLs/emails/handles
    """
    nlp = _get_spacy_pipeline(language)
    if nlp is None or not text.strip():
        return text

    try:
        doc = nlp(text)
    except Exception:
        return text

    ent_token_idxs = set()
    ent_types = {"PERSON", "ORG", "GPE", "LOC"}
    for ent in doc.ents:
        if ent.label_ in ent_types:
            for t in ent:
                ent_token_idxs.add(t.i)

    # Never capitalize these connectors (common Spanish function words)
    cfg = _get_language_config(language)
    connectors = cfg.connectors

    def should_capitalize(tok) -> bool:
        txt = tok.text
        # Skip URLs/email/handles or tokens that themselves look like domain fragments
        if any(ch in txt for ch in ['@', '/', '://']) or re.search(r"\w+\.\w+", txt):
            return False
        # Skip TLD token in domain pattern split across tokens: label '.' TLD
        try:
            if tok.i >= 2:
                prev_dot = doc[tok.i - 1].text
                prev_label = doc[tok.i - 2].text
                if prev_dot == '.' and re.fullmatch(r"[A-Za-z0-9-]+", prev_label) and re.fullmatch(r"[A-Za-z]{2,24}", txt):
                    return False
        except Exception:
            pass
        low = txt.lower()
        if low in connectors:
            return False
        if tok.i in ent_token_idxs:
            if low in connectors and tok.ent_iob_ != 'B':
                return False
            return True
        if tok.pos_ == 'PROPN':
            if low in connectors:
                return False
            return True
        return False

    out = []
    for tok in doc:
        t = tok.text
        if should_capitalize(tok):
            if t:
                t = t[0].upper() + t[1:]
            # Spanish-specific de-capitalization for possessive + noun artifacts: "tu Español" -> "tu español"
            if language == 'es' and tok.i > 0:
                prev = doc[tok.i - 1].text.lower()
                if prev in _get_language_config(language).possessives and re.match(r"^[A-ZÁÉÍÓÚÑ]", t):
                    t = t[0].lower() + t[1:]
            # Additionally, avoid capitalizing possessive itself mid-sentence: "Tu" -> "tu"
        if language == 'es':
            if tok.text in {w.title() for w in _get_language_config(language).possessives}:
                # Lowercase unless at sentence start
                at_sent_start = getattr(tok, 'is_sent_start', False)
                prev_text = doc[tok.i - 1].text if tok.i > 0 else ''
                if not at_sent_start and prev_text not in {'.', '!', '?', '¿', '¡'}:
                    t = t.lower()
        out.append(t + tok.whitespace_)
    result = ''.join(out)

    if language == 'es':
        # Entity-driven appositive commas: PERSON (,)? de GPE (,) GPE
        # Insert missing comma after PERSON when followed by "de <GPE/...>"
        # and missing comma between two consecutive GPE/LOC entities.
        try:
            edits: list[tuple[int, str]] = []
            for sent in doc.sents:
                # Work within sentence text slice
                for ent in doc.ents:
                    if ent.label_ != 'PERSON':
                        continue
                    if not (sent.start_char <= ent.start_char and ent.end_char <= sent.end_char):
                        continue
                    # Find next non-space char after PERSON
                    tail = text[ent.end_char:sent.end_char]
                    tail_lstrip = tail.lstrip()
                    if not tail_lstrip:
                        continue
                    # If already has a comma immediately after spaces, skip first comma insertion
                    has_comma = tail_lstrip.startswith(',')
                    # After optional comma and spaces, check for 'de'
                    after = tail_lstrip[1:].lstrip() if has_comma else tail_lstrip
                    if not after.lower().startswith('de'):
                        continue
                    # Insert comma after PERSON if missing
                    if not has_comma:
                        edits.append((ent.end_char, ','))
                    # Now check for two consecutive location entities: GPE/LOC (,)? GPE/LOC
                    # Find first entity that starts after this PERSON within the sentence
                    following_ents = [e for e in doc.ents if e.start_char >= ent.end_char and e.start_char < sent.end_char and e.label_ in {"GPE","LOC"}]
                    if len(following_ents) >= 2:
                        first_loc = following_ents[0]
                        between = text[first_loc.end_char:following_ents[1].start_char]
                        # If there is no comma between two locations, insert one
                        if ',' not in between:
                            edits.append((first_loc.end_char, ','))
            # Apply edits in reverse order so positions remain valid
            if edits:
                for pos, ins in sorted(edits, key=lambda x: x[0], reverse=True):
                    # Insert before any spaces at position (keep comma tight to previous token)
                    result = result[:pos] + ins + result[pos:]
        except Exception:
            pass

        # Capitalize names after introduction cues (general, not whitelists)
        def cap_word(w: str) -> str:
            return w[:1].upper() + w[1:] if w else w
        stop_after_intro = {"de", "del", "la", "el", "los", "las", "y", "e", "o", "u"}
        # mi nombre es / me llamo / yo soy + Name
        def intro_repl(m: re.Match) -> str:
            cue = m.group(1)
            nxt = m.group(2)
            return f"{cue} {nxt if nxt.lower() in stop_after_intro else cap_word(nxt)}"
        result = re.sub(r'(?i)\b(mi nombre es|me llamo|yo soy)\s+([\wáéíóúñÁÉÍÓÚÑ-]+)', intro_repl, result)
        # Locations after vivo en / trabajo en + Place (skip stopwords)
        def loc_repl(m: re.Match) -> str:
            cue = m.group(1)
            nxt = m.group(2)
            return f"{cue} {nxt if nxt.lower() in stop_after_intro else cap_word(nxt)}"
        result = re.sub(r'(?i)\b(vivo en|trabajo en)\s+([\wáéíóúñÁÉÍÓÚÑ-]+)', loc_repl, result)
    
    return result


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
    def replace_word(m):
        w = m.group(0)
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
        ]
    }
    
    return question_patterns.get(language, question_patterns['en'])


# --- Spanish post-processing helpers (hybrid regex + semantic gating) ---

def _spanish_cleanup_postprocess(text: str) -> str:
    """Apply robust regex fixes for common ASR artifacts in Spanish.

    - Protect/normalize domains like espanolistos.com
    - Remove spurious leading '¿' before declarative discourse markers
    - Normalize tag questions: ", ¿no." -> ", ¿no?"
    - Fix collocations like "por? supuesto" -> "Por supuesto,"
    - Capitalize after opening "¿"
    - Ensure opening "¿" implies closing "?" and vice versa
    """
    if not text:
        return text

    # Normalize domains: join tokens like "espanolistos . com" -> "espanolistos.com"
    text = re.sub(r"\b([a-z0-9\-]+)\s*[.\-]\s*(com|net|org|co|es|io|edu|gov|uk|us|ar|mx)\b", lambda m: f"{m.group(1)}.{m.group(2).lower()}", text, flags=re.IGNORECASE)
    # Also handle 'www . domain . tld'
    text = re.sub(r"\b(www)\s*[.\-]\s*([a-z0-9\-]+)\s*[.\-]\s*(com|net|org|co|es|io|edu|gov|uk|us|ar|mx)\b", lambda m: f"{m.group(1)}.{m.group(2)}.{m.group(3).lower()}", text, flags=re.IGNORECASE)

    # Ensure TLDs are lowercase within domains: label.TLD -> label.tld
    def _lowercase_tld(m):
        return f"{m.group(1)}.{m.group(2).lower()}"
    text = re.sub(r"\b([a-z0-9\-]+)\.([A-Za-z]{2,24})\b", _lowercase_tld, text, flags=re.IGNORECASE)

    # Don't touch inside domains thereafter (best-effort by skipping tokens with ".tld")

    # Strip leading '¿' for common declarative starters
    starters = (
        r"así que|obvio|pues|entonces|bueno|además|también|porque|pero|y|o sea|"
        r"por supuesto|al mismo tiempo|a nivel|esto|esa|ese|estos|esas|esos|"
        r"los|las|el|la|lo|un|una|unos|unas"
    )
    text = re.sub(rf"(^|[\n\.!?]\s*)¿\s*(?=(?:{starters})\b)", r"\1", text, flags=re.IGNORECASE)

    # If sentence starts with '¿' but ends with '.', convert to '?'
    parts = _split_sentences_preserving_delims(text)
    for i in range(0, len(parts), 2):
        if i >= len(parts):
            break
        s = parts[i].strip()
        p = parts[i + 1] if i + 1 < len(parts) else ''
        if not s:
            continue
        if s.startswith('¿') and p == '.':
            parts[i + 1] = '?'
    text = ''.join(parts)

    # Tag questions normalization
    text = _es_normalize_tag_questions(text)

    # Collocation repairs
    text = _es_fix_collocations(text)

    # Capitalize first letter after opening '¿'
    def _cap_after_inverted_q(m: re.Match) -> str:
        prefix = m.group(1)
        mark = m.group(2)
        letter = m.group(3)
        return f"{prefix}{mark}{letter.upper()}"
    text = re.sub(r"(^|[\n\s])([¿])\s*([a-záéíóúñ])", _cap_after_inverted_q, text)

    # Ensure paired punctuation consistency
    text = _es_pair_inverted_questions(text)

    # Preserve embedded questions introduced mid-sentence by '¿'
    # If we see mid-sentence '¿', keep it when followed by interrogative cues; otherwise leave untouched
    # Later we will pair it with a closing '?' if missing before the next boundary
    # (Do not blanket-remove mid-sentence '¿')

    # Convert declarative sentences that start with common yes/no verb starters into questions
    starters = (
        r"puedes|puede|podrías|podría|pudiste|pudo|pudieron|pudimos|"
        r"quieres|quiere|quieren|quisiste|quiso|quisieron|"
        r"tienes|tiene|tienen|tuviste|tuvo|tuvieron|"
        r"vas|va|vamos|fuiste|fue|fueron|"
        r"estás|está|están|"
        r"hay|"
        r"te parece|le parece|crees|cree|piensas|piensa"
    )
    model_gate = _load_sentence_transformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    parts2 = _split_sentences_preserving_delims(text)
    for i in range(0, len(parts2), 2):
        if i >= len(parts2):
            break
        s = (parts2[i] or '').strip()
        if not s:
            continue
        p = parts2[i + 1] if i + 1 < len(parts2) else ''
        # only adjust those ending with '.' or missing punctuation
        if p in ('', '.'):
            if re.search(rf'^\s*(?:{starters})\b', s, flags=re.IGNORECASE):
                # semantic gate to avoid false positives like "Es importante..."
                if model_gate is not None and is_question_semantic(s, model_gate, 'es'):
                    parts2[i] = s
                    parts2[i + 1] = '?'
    text = ''.join(parts2)

    # Ensure opening and closing pairing
    text = _es_pair_inverted_questions(text)

    # Declarative starters should not be questions
    # If a sentence starts with these starters and is marked as question, convert to statement
    starters_block = (
        r"a nivel|a nivel de|obvio que|como siempre|así que|los |las |el |la |un |una |en |de |haciendo |por supuesto"
    )
    def _block_decl_questions(m: re.Match) -> str:
        s = m.group(1)
        p = m.group(2)
        if re.match(rf"^\s*(?:{starters_block})", s, flags=re.IGNORECASE):
            s = re.sub(r"^\s*¿\s*", "", s)
            return s.rstrip(" ?!") + "."
        return m.group(0)
    text = re.sub(r"(¿[^\n\.?]+)(\?)", _block_decl_questions, text)

    # Merge possessive/function-word splits
    text = _es_merge_possessive_splits(text)

    # Merge consecutive one-word capitalized sentences
    text = _es_merge_capitalized_one_word_sentences(text)

    # Appositive commas for introductions and locations
    text = _es_intro_location_appositive_commas(text)

    # Greetings and lead-ins
    text = _es_greeting_and_leadin_commas(text)

    # Ensure comma after common discourse markers at sentence start
    # Applies only when followed by a likely clause (lowercase start)
    markers = [
        "Bueno", "Entonces", "Pues", "Además", "Ademas", "Así que", "Asi que",
        "Obvio", "O sea", "A ver", "Miren", "Mira", "Veamos", "En fin"
    ]
    for m in markers:
        pattern = rf"(?i)(^|[\n\.!?]\s*)({m})(\s+)(?!,)(?=[a-záéíóúñ])"
        repl = r"\1\2,\3"
        try:
            text = re.sub(pattern, repl, text)
        except re.error:
            pass

    # Style: repeated adverb comma (muy muy -> muy, muy)
    text = re.sub(r"(?i)\bmuy\s+muy\b", "muy, muy", text)

    # Convert "Entonces, empecemos." to exclamative form (keeps adverbial lead-in)
    text = re.sub(r"(?i)(^|[\n\.\?!]\s*)(Entonces,\s*)(empecemos)\.", r"\1\2¡\3!", text)

    # Merge auxiliary + gerund splits
    text = _es_merge_aux_gerund(text)

    # Upgrade common short yes/no patterns to questions when mistakenly left as statements
    text = re.sub(r"(?i)\b(Estamos|Están)\s+list[oa]s\.", lambda m: '¿' + m.group(0)[:-1] + '?', text)

    # Coordinated yes/no question pattern: verb-initial start and later " o <verb> ..."
    coord_starters = (
        r"puedes|puede|podrías|podría|pudiste|pudo|pudieron|pudimos|"
        r"quieres|quiere|quieren|"
        r"tienes|tiene|tienen|"
        r"vas|va|vamos|"
        r"estás|está|están|"
        r"hay|es|son"
    )
    # CRITICAL: Mask domains before splitting to prevent breaking them
    tld_alt = r"com|net|org|co|es|io|edu|gov|uk|us|ar|mx"
    text_masked_for_coord = re.sub(rf"\b([a-z0-9\-]{{3,}})\.({tld_alt})\b", r"\1__DOT__\2", text, flags=re.IGNORECASE)
    parts_coord = re.split(r'([.!?]+)', text_masked_for_coord)
    # Unmask domains in the split parts
    parts_coord = [re.sub(r"__DOT__", ".", part) for part in parts_coord]
    for i in range(0, len(parts_coord), 2):
        if i >= len(parts_coord):
            break
        s = (parts_coord[i] or '').strip()
        if not s:
            continue
        p = parts_coord[i + 1] if i + 1 < len(parts_coord) else ''
        if p in ('', '.') and \
           re.search(rf'^\s*(?:{coord_starters})\b', s, flags=re.IGNORECASE) and \
           (
               re.search(rf'\bo\s+(?:{coord_starters})\b', s, flags=re.IGNORECASE)
               or re.search(r'\bo\s+[a-záéíóúñ]+(?:an|en|as|es|a|e|amos|emos|imos)\b', s, flags=re.IGNORECASE)
           ):
            if not s.startswith('¿'):
                parts_coord[i] = '¿' + s
            parts_coord[i + 1] = '?'
    text = ''.join(parts_coord)

    # Merge premature exclamation closure when a lowercase connector continues the clause
    # Example: "¡Dile adiós! a todos esos momentos incómodos." -> "¡Dile adiós a todos esos momentos incómodos!"
    text = re.sub(r"!\s+(?=(?:a|al|de|del|en|con|por|para|y|o|que)\b)", " ", text, flags=re.IGNORECASE)

    # Repair proper-noun country/location pairs split by an erroneous period after a prepositional phrase
    # Example: "... de Texas. Estados Unidos." -> "... de Texas, Estados Unidos."
    text = re.sub(
        r"(\bde\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]*),?\s*\.\s+([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]*)\s+([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]*)\.",
        r"\1, \2 \3.",
        text,
    )

    # Add exclamations for common imperative/greeting starters (not if it's already a question/exclamation)
    def _wrap_exclamations(line: str) -> str:
        s = line.strip()
        if not s:
            return line
        low = s.lower()
        if s.startswith('¿') or s.endswith('?') or s.startswith('¡') or s.endswith('!'):
            return line
        if re.match(r"^(bienvenidos|empecemos|vamos|dile|atención)\b", low):
            return '¡' + s + '!'
        return line
    # Apply per sentence
    parts4 = _split_sentences_preserving_delims(text)
    for i in range(0, len(parts4), 2):
        if i >= len(parts4):
            break
        s = parts4[i]
        p = parts4[i + 1] if i + 1 < len(parts4) else ''
        if s:
            wrapped = _wrap_exclamations(s)
            if wrapped != s:
                parts4[i] = wrapped
                # Drop following punctuation token if present to avoid duplicates like "!." or "!?"
                if i + 1 < len(parts4):
                    parts4[i + 1] = ''
    text = ''.join(parts4)

    # Ensure that sentences starting with '¡' end with a single '!'
    parts5 = _split_sentences_preserving_delims(text)
    for i in range(0, len(parts5), 2):
        if i >= len(parts5):
            break
        s = parts5[i] or ''
        p = parts5[i + 1] if i + 1 < len(parts5) else ''
        s_stripped = s.strip()
        if not s_stripped:
            continue
        if s_stripped.startswith('¡'):
            # If content already ends with '!', drop trailing punctuation token
            if s_stripped.endswith('!'):
                if i + 1 < len(parts5) and p:
                    parts5[i + 1] = ''
            else:
                # Replace trailing punctuation with '!' or append '!'
                if i + 1 < len(parts5) and p:
                    parts5[i + 1] = '!'
                else:
                    parts5[i] = s.rstrip(' .?') + '!'
        else:
            # Mid-sentence exclamation: if the sentence contains '¡' anywhere and lacks a closing '!'
            if '¡' in s_stripped:
                # Consider it needing closure when no '!' in content and trailing punct missing or '.'
                needs_closure = ('!' not in s_stripped)
                if needs_closure and (p in ('', '.')):
                    if i + 1 < len(parts5) and p:
                        parts5[i + 1] = '!'
                    else:
                        parts5[i] = s.rstrip(' .?') + '!'
    text = ''.join(parts5)

    # Final mixed-punctuation cleanup after exclamation pairing
    text = re.sub(r'!\s*\.', '!', text)
    text = re.sub(r'!\s*\?', '!', text)

    # Capitalize sentence starts (handles leading punctuation/quotes)
    text = _es_capitalize_sentence_starts(text)

    # Do NOT capitalize after an ellipsis within a continuing clause.
    # Example: "De tener bancarrota... rota." should keep "rota" lowercase.
    # Handles both ASCII "..." and Unicode ellipsis "…".
    text = re.sub(r"((?:\.\.\.|…)\s+)([A-ZÁÉÍÓÚÑ])", lambda m: m.group(1) + m.group(2).lower(), text)

    # Cleanup spacing and duplicates
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"¿{2,}", "¿", text)
    text = re.sub(r"\?{2,}", "?", text)
    # Preserve ellipses '…' and '...':
    text = re.sub(r"\.{4,}", "...", text)          # 4+ dots -> ...
    text = re.sub(r"(?<!\.)\.\.(?!\.)", ".", text)  # reduce exactly two dots
    return text


# (Module has no __main__ block; import-only.)