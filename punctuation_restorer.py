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
from domain_utils import mask_domains, unmask_domains, apply_safe_text_processing, create_domain_aware_regex, SINGLE_TLDS, SINGLE_MASK

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


# SpaCy import for NLP-based capitalization (mandatory)
import spacy

# Try to import language detection for mixed-language content
try:
    from spacy_language_detection import LanguageDetector
    SPACY_LANG_DETECTION_AVAILABLE = True
except ImportError:
    SPACY_LANG_DETECTION_AVAILABLE = False
    logger.info("spacy-language-detection not available. Will use fallback heuristics for mixed-language content.")


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
        'de': r'aus|von|in'
    }
    
    lang_code = language.lower()
    prepositions = location_prepositions.get(lang_code, r'de|from|aus|von|in|du|des')
    
    # Pattern 1: comma + preposition + location + period + location
    # Convert the period to a comma for proper appositive punctuation
    # But exclude cases that start new sentences with subjects like "Y yo soy", "And I'm", etc.
    pattern1 = rf'(,\s*(?:{prepositions})\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑ-]*)\.\s+([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑ-]*)'
    
    # Check if the following text starts a new sentence with a subject
    def _safe_location_merge_preposition(match):
        prefix = match.group(1)
        following = match.group(2)
        
        # Don't merge if following text looks like start of new sentence with subject
        # Common patterns: "Y yo", "And I", "Et je", "Und ich", etc.
        new_sentence_patterns = [
            r'^(Y|And|Et|Und)\s+(yo|I|je|ich)',  # "Y yo", "And I", "Et je", "Und ich"
            r'^(Y|And|Et|Und)\s+\w+\s+(soy|am|suis|bin)',  # "Y alguien soy", "And someone am"
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
            r'^(Y|And|Et|Und)\s+(yo|I|je|ich)',  # "Y yo", "And I", "Et je", "Und ich"  
            r'^(Y|And|Et|Und)\s+\w+\s+(soy|am|suis|bin)',  # "Y alguien soy", "And someone am"
        ]
        
        # Check the full following context (might be more than one word)
        rest_of_text = text[match.end():]
        full_following = following_word + " " + rest_of_text.split('.')[0][:50]  # Check first 50 chars
        
        for pattern in new_sentence_patterns:
            if re.match(pattern, full_following, re.IGNORECASE):
                return match.group(0)  # Return unchanged - don't merge
        
        return f"{location_part} {following_word.lower()}"
    
    pattern2 = r'(\b[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑ-]*,\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑ-]*)\.\s+([A-ZÁÉÍÓÚÑa-záéíóúñ][\wÁÉÍÓÚÑ-]*)'
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
    # Normalize comma spacing using centralized function
    out = _normalize_comma_spacing(out)
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
                    merged.append(f"{base} {cont}".strip())
                    i += 2
                    continue
        merged.append(sentences[i])
        i += 1
    return merged
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

    # Remove leading inverted question before 'Hola' when an embedded question follows later in the sentence
    # e.g., "¿Hola para todos, ¿Cómo están?" -> "Hola para todos, ¿Cómo están?"
    text = re.sub(r"(^|[\n\.!?]\s*)¿\s*(Hola\b[^¿\n]*?),\s*(¿)", r"\1\2, \3", text, flags=re.IGNORECASE)

    # Greeting punctuation: if a greeting sentence ends with a period and followed by a question, turn period into comma
    text = re.sub(r"(\bHola[^.!?]*?)\.\s+(¿)", r"\1, \2", text, flags=re.IGNORECASE)
    # Also handle 'como' without accent
    text = re.sub(r"(\bHola[^.!?]*?)\.\s+(como\s+estan)\b", r"\1, ¿\2?", text, flags=re.IGNORECASE)
    # Remove comma right after "Hola" when followed by a prepositional phrase (more natural Spanish)
    # Also match when preceded by an inverted question/exclamation
    text = re.sub(r"(^|[\n\.\?¡!¿]\s*)Hola,\s+(a|para)\b", r"\1Hola \2", text, flags=re.IGNORECASE)
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
    #    Guard: do not add another comma if one already precedes the inverted mark
    text = re.sub(r"((^|(?<=\n)|(?<=[\.!?]\s))Hola[^.!?]*?)(?<![,，])\s+([¿¡])", r"\1, \3", text, flags=re.IGNORECASE)
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
            # Don't capitalize if this looks like the start of a domain name
            remaining_text = s[idx:]
            # Updated pattern to include accented characters for domains like sinónimosonline.com
            if not re.match(r'^[a-zA-Z0-9\u00C0-\u017F\-]+\.(com|net|org|co|es|io|edu|gov|uk|us|ar|mx)\b', remaining_text):
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
        # Gate: if the sentence starts with a greeting lead-in (Hola, Buenos días, etc.) and contains an embedded '¿',
        # do not force-add a leading '¿' at the very start; preserve embedded question only.
        greeting_start = bool(re.match(r'^(hola|buenos\s+d[ií]as|buenas\s+tardes|buenas\s+noches|bienvenidos)\b', s, flags=re.IGNORECASE))
        has_embedded_q = '¿' in s
        # Full-sentence question: add opening '¿' if missing and no existing embedded '¿'
        if p == '?' and '¿' not in s and not s.startswith('¿') and not greeting_start:
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
        return _should_add_terminal_punctuation(text, language, PunctuationContext.SENTENCE_END)


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
        return _should_add_terminal_punctuation(text, language, PunctuationContext.SENTENCE_END)

    # 1) Semantic split into sentences (token-based loop with _should_end_sentence_here)
    sentences = _semantic_split_into_sentences(text, language, model)
    # 2) Punctuate sentences and join
    result = _punctuate_semantic_sentences(sentences, model, language)
    
    # Apply Spanish-specific formatting
    if language == 'es':
        # COMPREHENSIVE DOMAIN PROTECTION: Mask domains before ALL Spanish processing
        result_with_protected_domains = mask_domains(result, use_exclusions=True, language=language)
        # Domain-preserving sentence split and formatting is now handled by centralized masking
        sentences_list, trailing_fragment = assemble_sentences_from_processed(result_with_protected_domains, 'es')
        if trailing_fragment:
            cleaned = re.sub(r'^[",\s]+', '', trailing_fragment)
            cleaned = _should_add_terminal_punctuation(cleaned, language, PunctuationContext.TRAILING)
            if cleaned:
                sentences_list.append(cleaned)

        formatted_sentences = []
        for s in sentences_list:
            sentence = (s or '').strip()
            if not sentence:
                continue
            # Capitalize first letter (but not for domains)
            if sentence and sentence[0].isalpha():
                # Don't capitalize if this looks like a domain name
                # Updated pattern to include accented characters for domains like sinónimosonline.com
                if not re.match(r'^[a-zA-Z0-9\u00C0-\u017F\-]+\.(com|net|org|co|es|io|edu|gov|uk|us|ar|mx)\b', sentence.lower()):
                    sentence = sentence[0].upper() + sentence[1:]

            # Ensure sentence ends with single terminal punctuation
            if not sentence.endswith(('.', '!', '?')):
                question_words = ES_QUESTION_WORDS_CORE + ES_QUESTION_STARTERS_EXTRA
                sentence_lower = sentence.lower()
                if any(word in sentence_lower for word in question_words):
                    sentence += '?'
                else:
                    # Use centralized punctuation logic for Spanish
                    sentence = _should_add_terminal_punctuation(sentence, language, PunctuationContext.SPANISH_SPECIFIC)
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
        result_masked_for_split = mask_domains(result, use_exclusions=True, language=language)
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
        
        # SpaCy capitalization pass (moved before comma insertion to avoid feedback loop)
        # This ensures spaCy sees the original text without artificial capitalization from comma insertion
        # Mask domains before spaCy to prevent it from capitalizing domain names as proper nouns
        result_masked_for_spacy = mask_domains(result, use_exclusions=True, language=language)
        result_capitalized = _apply_spacy_capitalization(result_masked_for_spacy, language)
        result = unmask_domains(result_capitalized)
        
        # Special handling for greetings and lead-ins (now after spaCy capitalization)
        result = _es_greeting_and_leadin_commas(result)
        
        # Remove the comma addition for location patterns - Spanish grammar doesn't require it
        # result = re.sub(r'\b(andrea|carlos|maría|juan|ana)\s+(de)\s+(colombia|santander|madrid|españa|méxico|argentina)\b', r'\1 \2, \3', result, flags=re.IGNORECASE)
        
        # Ensure proper sentence separation
        # CRITICAL: Mask domains before space insertion to prevent breaking them
        result_masked_for_separation = mask_domains(result, use_exclusions=True, language=language)
        result_masked_for_separation = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', result_masked_for_separation)
        result = unmask_domains(result_masked_for_separation)
        
        # Insert comma after common Spanish greeting starters, but avoid when followed by prepositional phrase
        result = re.sub(r'(^|[\n\.\!?¿¡]\s*)(Hola)\s+(?!a\b|para\b)', r"\1\2, ", result, flags=re.IGNORECASE)
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
        result_masked_for_punct = mask_domains(result, use_exclusions=True, language=language)
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
                        # Use centralized punctuation logic
                        sentence = _should_add_terminal_punctuation(sentence, language, PunctuationContext.SENTENCE_END)
                    sentences[i] = sentence
        result = ''.join(sentences)
        
        # Additional fix: ensure short phrases get proper punctuation
        # Handle cases like "También sí" that don't get punctuation
        result = re.sub(r'\b(también sí|sí|no|claro|exacto|perfecto|vale|bien)\s*$', r'\1.', result, flags=re.IGNORECASE)
        
        # Fix any remaining sentences without punctuation at the end
        # This catches any sentences that might have been missed
        # CRITICAL: Mask domains before splitting to prevent breaking them  
        result_masked_for_final = mask_domains(result, use_exclusions=True, language=language)
        sentences = _split_sentences_preserving_delims(result_masked_for_final)
        # Unmask domains in the split sentences
        sentences = [re.sub(r"__DOT__", ".", s) for s in sentences]
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i].strip()
                # Use centralized punctuation logic
                sentence = _should_add_terminal_punctuation(sentence, language, PunctuationContext.SENTENCE_END)
                sentences[i] = sentence
        result = ''.join(sentences)
        
        # Removed redundant short phrase patterns - already handled by line 1255
        
        # Additional comprehensive fix for any remaining sentences without punctuation
        # Split by sentences and ensure each one ends with punctuation
        # CRITICAL: Mask domains before splitting to prevent breaking them
        result_masked_for_comprehensive = mask_domains(result, use_exclusions=True, language=language)
        sentences = re.split(r'([.!?]+)', result_masked_for_comprehensive)
        # Unmask domains in the split sentences
        sentences = [re.sub(r"__DOT__", ".", s) for s in sentences]
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i].strip()
                if sentence and not sentence.endswith(('.', '!', '?')):
                    # Use centralized punctuation logic
                    sentence = _should_add_terminal_punctuation(sentence, language, PunctuationContext.SENTENCE_END)
                    sentences[i] = sentence
        result = ''.join(sentences)
        
        # Bug 2: Fix sentences ending in commas
        # Replace trailing commas with proper punctuation
        result = re.sub(r',\s*$', '.', result)  # End of text
        result = re.sub(r',\s*([.!?])', r'\1', result)  # Before other punctuation
        # Removed overly aggressive comma-to-period pattern that was breaking Spanish grammar
        # The pattern r',\s*([A-Z])', r'. \1' incorrectly assumed comma + capital = new sentence
        # This broke natural Spanish flow like "Sí, Andrea" → "Sí. Andrea" and grammar like 
        # "María, José y Ana" → "María. José y Ana". Spanish commonly uses commas before 
        # proper names in lists, greetings, and conversational transitions.
        
        # Removed overly broad regex that was breaking mid-sentence Spanish flow
        # The pattern r'\b(también sí|sí|no|claro|exacto|perfecto|vale|bien)\s*,' was incorrectly
        # converting natural transitions like "Sí, bueno" to "Sí. bueno" causing sentence splits
        
        # Bug 4: Fix question marks followed by inverted question marks in the middle
        # Remove the problematic pattern "?¿" in the middle of sentences
        result = re.sub(r'\?\s*¿(\w+)', r' \1', result)  # ?¿ followed by word
        result = re.sub(r'(\w+)\?\s*¿(\w+)', r'\1? \2', result)  # word?¿word
        
        # Additional cleanup for Spanish-specific patterns
        # Fix sentences that start with ¿ but don't end with ?
        # CRITICAL: Mask domains before splitting to prevent breaking them
        result_masked_for_patterns = mask_domains(result, use_exclusions=True, language=language)
        sentences = re.split(r'([.!?]+)', result_masked_for_patterns)
        # Unmask domains in the split sentences
        sentences = [re.sub(r"__DOT__", ".", s) for s in sentences]
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i].strip()
                if sentence.startswith('¿') and not sentence.endswith('?'):
                    # Remove the ¿ and add proper punctuation using centralized logic
                    sentence = sentence[1:]  # Remove ¿
                    sentence = _should_add_terminal_punctuation(sentence, language, PunctuationContext.SENTENCE_END)
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
        
        # Ensure all sentences end with proper punctuation using centralized logic
        result = _should_add_terminal_punctuation(result, language, PunctuationContext.SENTENCE_END)
        
        # Removed redundant short phrase patterns - already handled comprehensively by line 1255
        
        # FINAL STEP: Add inverted question marks for all questions
        # This runs after all punctuation has been added
        # Use semantic gating to decide whether to keep/add inverted question marks
        # CRITICAL: Mask domains before splitting to prevent breaking them
        result_masked_for_semantic = mask_domains(result, use_exclusions=True, language=language)
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

        # (Removed) final collapse of emphatic repeats
        # Normalize comma spacing using centralized function
        result = _normalize_comma_spacing(result)
        # Ensure a single space after terminal punctuation when followed by a non-space and not part of an ellipsis
        # CRITICAL: Mask domains before space insertion to prevent breaking them
        result_masked_for_spacing = mask_domains(result, use_exclusions=True, language=language)
        result_masked_for_spacing = re.sub(r'(?<!\.)\.([^\s.])', r'. \1', result_masked_for_spacing)
        result = unmask_domains(result_masked_for_spacing)
        result = re.sub(r'\?\s*(?=\S)', r'? ', result)
        result = re.sub(r'!\s*(?=\S)', r'! ', result)
        # Capitalize the first letter after sentence terminators when appropriate (Spanish sentences start capitalized)
        def _cap_after_terminator(m):
            punct = m.group(1)
            ch = m.group(2)
            return f"{punct} {ch.upper()}"
        result = re.sub(r'([.!?])\s+([a-záéíóúñ])', _cap_after_terminator, result)
        # Final domain TLD lowercasing safeguard (after all formatting/capitalization)
        # Updated pattern to include accented characters for domains like sinónimosonline.com
        result = re.sub(r"\b([a-zA-Z0-9\u00C0-\u017F\-]+)\.([A-Za-z]{2,24})\b", lambda m: f"{m.group(1)}.{m.group(2).lower()}", result, flags=re.IGNORECASE)
        
        # COMPREHENSIVE DOMAIN PROTECTION: Unmask domains at the very end of Spanish processing
        result = unmask_domains(result)
        
        # Final domain capitalization fix: correct any incorrectly capitalized domains
        # This handles cases where domains got capitalized during processing despite masking
        
        # Fix spaced domains: "www. Espanolistos.com" -> "www.espanolistos.com"
        # Updated pattern to include accented characters for domains like sinónimosonline.com
        result = re.sub(r'\b(www|ftp|mail|blog|shop|app|api|cdn|static|news|support|help|docs|admin|secure|login|mobile|store|sub|dev|test|staging|prod|beta|alpha)\.\s+([A-Z][a-zA-Z0-9\u00C0-\u017F\-]*\.[a-z]{2,24})\b', 
                       lambda m: f"{m.group(1).lower()}.{m.group(2).lower()}", result, flags=re.IGNORECASE)
        
        # Fix connected domains: "www.Espanolistos.com" -> "www.espanolistos.com"
        # Updated pattern to include accented characters for domains like sinónimosonline.com
        result = re.sub(r'\b(www|ftp|mail|blog|shop|app|api|cdn|static|news|support|help|docs|admin|secure|login|mobile|store|sub|dev|test|staging|prod|beta|alpha)\.([A-Z][a-zA-Z0-9\u00C0-\u017F\-]*\.[a-z]{2,24})\b', 
                       lambda m: f"{m.group(1).lower()}.{m.group(2).lower()}", result, flags=re.IGNORECASE)
        
        # Fix standalone domain names: "Espanolistos.com" -> "espanolistos.com"
        # Updated pattern to include accented characters for domains like sinónimosonline.com
        result = re.sub(r'\b([A-Z][a-zA-Z0-9\u00C0-\u017F\-]*\.(com|net|org|co|es|io|edu|gov|uk|us|ar|mx))\b', 
                       lambda m: m.group(1).lower(), result)
        
        # Additional fix for the specific "Espanolistos" case and similar patterns
        result = re.sub(r'\bwww\.([A-Z][a-z]+)\.(com|net|org|co|es|io|edu|gov|uk|us|ar|mx)\b', 
                       lambda m: f"www.{m.group(1).lower()}.{m.group(2)}", result)
    else:
        # Apply light, language-aware formatting for non-Spanish languages
        result = _format_non_spanish_text(result, language)

        # SpaCy capitalization pass (always applied)
        # Mask domains before spaCy to prevent it from capitalizing domain names as proper nouns
        result_masked_for_spacy = mask_domains(result, use_exclusions=True, language=language)
        result_capitalized = _apply_spacy_capitalization(result_masked_for_spacy, language)
        result = unmask_domains(result_capitalized)

    # Fix location appositive punctuation across languages
    result = _fix_location_appositive_punctuation(result, language)
    
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
    
    # CRITICAL: Never end a sentence on a coordinating conjunction
    # These words grammatically require something to follow them
    # Examples: "y" (es), "and" (en), "et" (fr), "und" (de), "but", "or", etc.
    # This prevents splits like "teníamos muchos errores y." | "Eco..."
    current_word_clean = current_word.lower().strip('.,;:!?')
    coordinating_conjunctions = {
        'y', 'e', 'o', 'u',  # Spanish: and, and (before i-), or, or (before o-)
        'pero', 'mas', 'sino',  # Spanish: but
        'and', 'but', 'or', 'nor', 'for', 'so', 'yet',  # English
        'et', 'ou', 'mais', 'donc', 'or', 'ni', 'car',  # French
        'und', 'oder', 'aber', 'denn', 'sondern',  # German
    }
    if current_word_clean in coordinating_conjunctions:
        return False
    
    # Spanish: Never split after or inside an unclosed inverted question mark
    # This prevents splits like "Pues, ¿Qué?" | "¿Pasó, Nate?" from "Pues, ¿qué pasó, Nate?"
    if language == 'es':
        # Don't split if current word contains/ends with '¿'
        if '¿' in current_word:
            return False
        # Don't split if we're inside an unclosed question (have '¿' but no '?')
        current_text = ' '.join(current_chunk)
        if '¿' in current_text and '?' not in current_text:
            return False
    
    # CRITICAL: Never split when current word is a noun that commonly precedes numbers
    # and next word is a standalone number (e.g., "episode 184", "épisode 184", "chapter 5")
    # OR when current word is a conjunction in a number list (e.g., "177 y 184", "3 and 4")
    # This prevents incorrect splits like "episode. 184" or "y. 184" in all languages
    # Strip punctuation from next_word to check if it's a number (handles "184." → "184")
    if next_word:
        next_word_clean = next_word.strip('.,;:!?')
        if next_word_clean.isdigit():
            current_word_clean = current_word.lower().strip('.,;:!?')
            
            # Check if current word is a conjunction (y/o/and/or/et/ou/und/oder) in a number list
            # Pattern: "number, number, ... conjunction number"
            conjunctions_before_numbers = {'y', 'o', 'and', 'or', 'et', 'ou', 'und', 'oder'}
            if current_word_clean in conjunctions_before_numbers:
                # Check if there's a number before the conjunction (indicating a list)
                # Look back a few words to see if we're in a number list context
                if len(current_chunk) >= 2:
                    # Check if any of the previous 3 words contains digits (number list context)
                    prev_words = current_chunk[-3:] if len(current_chunk) >= 3 else current_chunk
                    has_prev_number = any(any(c.isdigit() for c in w.strip('.,;:!?')) for w in prev_words)
                    if has_prev_number:
                        return False  # Don't split in a number list like "147, 151, 177 y 184"
            
            # Common nouns that precede numbers across languages
            number_preceding_nouns = {
                'episode', 'episodes', 'episodio', 'episodios', 'épisode', 'épisodes',  # episodes
                'chapter', 'chapters', 'capítulo', 'capítulos', 'chapitre', 'chapitres', 'kapitel',  # chapters
                'year', 'years', 'año', 'años', 'année', 'années', 'jahr', 'jahre',  # years
                'season', 'seasons', 'temporada', 'temporadas', 'saison', 'saisons', 'staffel',  # seasons
                'volume', 'volumes', 'volumen', 'volúmenes', 'band', 'bände',  # volumes
                'part', 'parts', 'parte', 'partes', 'partie', 'parties', 'teil', 'teile',  # parts
                'page', 'pages', 'página', 'páginas', 'seite', 'seiten',  # pages
                'number', 'numbers', 'número', 'números', 'numéro', 'numéros', 'nummer', 'nummern',  # numbers
                'serie', 'series', 'série', 'séries',  # series
                'track', 'tracks', 'pista', 'pistas',  # tracks
            }
            # Also handle contractions like "l'épisode" -> "épisode"
            if "'" in current_word_clean:
                current_word_clean = current_word_clean.split("'")[-1]
            
            if current_word_clean in number_preceding_nouns:
                return False
    
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
        # Also protect when the NEXT token starts a question clause (e.g., "Qué", "Cómo", or "por qué")
        next_low = next_word.lower() if next_word else ''
        next2_low = next_next_word.lower() if 'next_next_word' in locals() and next_next_word else ''
        if next_low in {'qué', 'como', 'cómo', 'dónde', 'cuando', 'cuándo', 'quién', 'cual', 'cuál'}:
            return False
        if next_low == 'por' and next2_low in {'qué'}:
            return False
        
        # Don't break in the middle of common Spanish phrases
        # Check if we're in the middle of a phrase that should stay together
        current_text = ' '.join(current_chunk).lower()
        
        # General rule: don't break after articles, prepositions, determiners, or common quantifiers
        # These words typically continue the sentence
        # CRITICAL: Include 'a' (to), 'ante' (before), 'bajo' (under) - common prepositions that should never end sentences
        if current_word.lower() in ['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'a', 'ante', 'bajo', 'de', 'del', 'al', 'en', 'con', 'por', 'para', 'sin', 'sobre', 'entre', 'tras', 'durante', 'mediante', 'según', 'hacia', 'hasta', 'desde', 'contra',
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
        # CRITICAL: Include 'a' (to), 'ante' (before), 'bajo' (under) - these are common Spanish prepositions
        if current_word.lower() in ['a', 'ante', 'bajo', 'de', 'del', 'al', 'en', 'con', 'por', 'para', 'sin', 'sobre', 'entre', 'tras', 'durante', 'mediante', 'según', 'hacia', 'hasta', 'desde', 'contra']:
            return False
    
    # English: Don't break after common prepositions that require continuation
    if language == 'en':
        if current_word.lower() in ['to', 'at', 'from', 'with', 'by', 'of', 'in', 'on', 'for', 'about', 'into', 'onto', 'upon', 'after', 'before', 'through', 'during', 'without', 'within', 'among', 'between', 'under', 'over', 'above', 'below', 'across', 'along', 'around', 'behind', 'beside', 'beyond', 'near', 'off', 'toward', 'towards', 'until', 'up', 'down', 'out', 'inside', 'outside', 'throughout', 'against']:
            return False
    
    # French: Don't break after common prepositions that require continuation
    if language == 'fr':
        if current_word.lower() in ['à', 'de', 'en', 'pour', 'avec', 'sans', 'sous', 'sur', 'dans', 'chez', 'par', 'vers', 'contre', 'entre', 'parmi', 'pendant', 'depuis', 'devant', 'derrière', 'avant', 'après', 'durant', 'selon', 'malgré', 'sauf', 'jusque', "jusqu'à"]:
            return False
    
    # German: Don't break after common prepositions that require continuation
    if language == 'de':
        if current_word.lower() in ['zu', 'an', 'auf', 'aus', 'bei', 'mit', 'nach', 'von', 'vor', 'in', 'für', 'durch', 'über', 'unter', 'hinter', 'neben', 'zwischen', 'um', 'ohne', 'gegen', 'seit', 'bis', 'während', 'wegen', 'trotz', 'innerhalb', 'außerhalb']:
            return False
        
    # Protect common Spanish question tail constructions: "qué" + infinitive (e.g., "qué decir")
    prev_low = re.sub(r"[\.,;:!?]+$", "", current_word.lower())
    if language == 'es' and prev_low in {'qué', 'que'}:
        nw_clean = re.sub(r"[\.,;:!?]+$", "", (next_word or '').lower())
        if nw_clean.endswith(('ar', 'er', 'ir')) or nw_clean in {'decir', 'hacer', 'ser', 'estar', 'poder'}:
            return False

    # Check for capital letter after reasonable length (be much more conservative for Spanish ASR capitalization)
    next_next_word = words[current_index + 2] if current_index + 2 < len(words) else ""
    # Guard: avoid breaking right after conjunction + pronoun (e.g., "Y yo", "Y él")
    if language == 'es' and len(current_chunk) >= 2:
        last_two = [w.lower() for w in current_chunk[-2:]]
        if last_two[0] in {'y', 'e'} and last_two[1] in {'yo','tú','tu','él','ella','nosotros','nosotras','ellos','ellas','usted','ustedes'}:
            return False
    if (next_word and next_word[0].isupper() and
        len(current_chunk) >= thresholds.get('min_chunk_capital_break', 28) and  # conservative
        not _is_continuation_word(current_word, language) and
        not _is_transitional_word(current_word, language)):
        # ES: avoid breaking after copular/aux verbs when followed by a proper name (e.g., "soy Nate,")
        if language == 'es':
            cur_low = current_word.lower().strip('.,;:!?')
            next_has_comma = next_word.endswith(',') if next_word else False
            if cur_low in {'soy','eres','es','somos','son','estoy','está','están','era','eras','éramos','eran','fui','fue','estuve','estaba'}:
                # If next token is a capitalized name and (has comma or next-next is 'de'), keep together
                if next_word[0].isupper():
                    if next_has_comma or (next_next_word and next_next_word.lower() == 'de'):
                        return False
            # Avoid splitting inside appositive location chains: 
            # pattern: ", de <Proper> (,)? <Proper>" (e.g., ", de Texas, Estados Unidos")
            # Look back for recent 'de' and require keeping following proper tokens together
            tail = [w for w in current_chunk[-4:]]
            if 'de' in [t.lower().strip('.,;:!?') for t in tail]:
                if next_word and next_word[:1].isupper():
                    if next_next_word and next_next_word[:1].isupper():
                        return False
        # Avoid breaking when followed by two capitalized tokens (likely multi-word proper noun: "Estados Unidos")
        if next_next_word and next_next_word[0].isupper():
            return False
        # Avoid breaking after comma or after preposition like "de"
        if current_word.endswith(',') or current_word.lower() in ['de', 'en']:
            return False
        # Avoid breaking when the next word is a determiner/preposition starting a continuation
        if next_word.lower() in ['los', 'las', 'el', 'la', 'de', 'del']:
            return False
        # Require semantic corroboration for a capital break
        if len(current_chunk) >= thresholds.get('min_chunk_semantic_break', 30):
            return _check_semantic_break(words, current_index, model)
        return False
    
    # Use semantic coherence to determine if chunks should be separated
    if len(current_chunk) >= thresholds.get('min_chunk_semantic_break', 30):  # conservative semantic split
        # If the next token is a lowercase continuation or preposition, do not break
        if next_word and next_word and next_word[0].islower():
            if next_word.lower() in ['y', 'o', 'pero', 'que', 'de', 'en', 'para', 'con', 'por', 'sin', 'sobre']:
                return False
        # Do not split immediately after a comma (likely appositive continuation)
        if current_word.endswith(','):
            return False
        # Avoid splitting when followed by two capitalized tokens (likely proper noun: "Estados Unidos")
        if next_word and next_word[:1].isupper() and next_next_word and next_next_word[:1].isupper():
            return False
        # Spanish-specific continuation guards (mirror capital-break protections)
        if language == 'es':
            # Avoid splitting right after conjunction + pronoun (e.g., "Y yo", "Y él")
            if len(current_chunk) >= 2:
                last_two = [w.lower() for w in current_chunk[-2:]]
                if last_two[0] in {'y', 'e'} and last_two[1] in {'yo','tú','tu','él','ella','nosotros','nosotras','ellos','ellas','usted','ustedes'}:
                    return False
            # Avoid splitting when the next token is a finite verb beginning a continuation (e.g., "soy", "estoy", "era", "fui")
            if next_word:
                next_low2 = next_word.lower()
                finite_verb_starts = {'soy','estoy','era','fui','seré','estaré','estuve','estaba','sería','estaría','eres','es','somos','son','fue'}
                if next_low2 in finite_verb_starts:
                    return False
            # Avoid splitting after copular verb when followed by a proper name and comma ("soy Nate,")
            cur_low2 = current_word.lower().strip('.,;:!?')
            next_has_comma2 = next_word.endswith(',') if next_word else False
            if cur_low2 in {'soy','eres','es','somos','son','estoy','está','están','era','eras','éramos','eran','fui','fue','estuve','estaba'}:
                if next_word and next_word[0].isupper() and (next_has_comma2 or (next_next_word and next_next_word.lower() == 'de')):
                    return False
            # Avoid splitting inside appositive location chains after 'de'
            tail2 = [w for w in current_chunk[-4:]]
            if 'de' in [t.lower().strip('.,;:!?') for t in tail2]:
                if next_word and next_word[:1].isupper() and next_next_word and next_next_word[:1].isupper():
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
        'es': ['y', 'o', 'pero', 'así', 'porque', 'si', 'cuando', 'mientras', 'desde', 'aunque', 'sin', 'embargo', 'por', 'tanto', 'entonces', 'también', 'además', 'furthermore', 'más', 'aún', 'a', 'al', 'hacia', 'hasta', 'de', 'del', 'en', 'con'],
        'de': ['und', 'oder', 'aber', 'also', 'weil', 'wenn', 'während', 'seit', 'obwohl', 'jedoch', 'daher', 'deshalb', 'dann', 'auch', 'außerdem', 'ferner', 'zudem'],
        'fr': ['et', 'ou', 'mais', 'donc', 'parce', 'si', 'quand', 'pendant', 'depuis', 'bien', 'que', 'cependant', 'donc', 'alors', 'aussi', 'de', 'plus', 'en', 'outre', 'par', 'ailleurs']
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


def _should_add_terminal_punctuation(text: str, language: str, context: str = None, model=None) -> str:
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
    if not text or text.endswith(('.', '!', '?')):
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
            return text + '?'
        if is_exclamation_semantic(text, model, language):
            return text + '!'
    
    # Language-specific question detection fallback
    if language == 'es':
        question_words = ES_QUESTION_WORDS_CORE + ES_QUESTION_STARTERS_EXTRA
        text_lower = text.lower()
        if any(word in text_lower for word in question_words):
            return text + '?'
    
    # Special handling for short Spanish phrases
    if language == 'es' and text.lower() in ['también sí', 'sí', 'no', 'claro', 'exacto', 'perfecto', 'vale', 'bien', 'pues tranquilo']:
        return text + '.'
    
    # Default to period for complete sentences
    return text + '.'


def _should_carry_forward_segment(text: str, language: str) -> bool:
    """
    Determine if a segment should be carried forward to the next segment.
    
    This helps handle cases where Whisper splits incomplete phrases like "Ve a"
    into separate segments that should be combined.
    
    Args:
        text: The text segment to analyze
        language: Language code
        
    Returns:
        True if the segment should be carried forward
    """
    if not text:
        return False
        
    words = text.split()
    last_word = words[-1] if words else ''
    
    # Phrases ending with continuation words should be carried forward
    return _is_continuation_word(last_word, language)


def restore_punctuation_segment(text: str, language: str = 'en') -> str:
    """
    Restore punctuation for a single Whisper segment.
    
    This is similar to restore_punctuation() but uses STANDALONE_SEGMENT context
    to avoid adding periods to incomplete phrases like "Ve a" that should be 
    carried forward to the next segment.
    
    Args:
        text: The transcribed text segment from Whisper
        language: Language code ('en', 'es', 'de', 'fr')
    
    Returns:
        str: Text with restored punctuation, using segment-aware context
    """
    if not text or not text.strip():
        return text
    
    # Clean up the text
    text = text.strip()
    
    # Use SentenceTransformers for better sentence boundary detection if available
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        return _transformer_based_restoration_segment(text, language)
    else:
        # Simple fallback: clean up whitespace and use segment-aware punctuation
        text = re.sub(r'\s+', ' ', text.strip())
        return _should_add_terminal_punctuation(text, language, PunctuationContext.STANDALONE_SEGMENT)


def _transformer_based_restoration_segment(text: str, language: str = 'en') -> str:
    """
    Segment-aware version of _transformer_based_restoration.
    
    This version uses STANDALONE_SEGMENT context to avoid adding periods
    to incomplete phrases that should be carried forward.
    """
    # Initialize the model once (use multilingual model for better language support)
    model = _load_sentence_transformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    if model is None:
        # Fallback path if sentence-transformers is unavailable
        text = re.sub(r'\s+', ' ', text.strip())
        return _should_add_terminal_punctuation(text, language, PunctuationContext.STANDALONE_SEGMENT)

    # For segment processing, check if this should be carried forward
    if _should_carry_forward_segment(text, language):
        # Don't add punctuation to segments that should be carried forward
        return _should_add_terminal_punctuation(text, language, PunctuationContext.STANDALONE_SEGMENT)
    
    # For complete segments, use normal sentence processing
    return _should_add_terminal_punctuation(text, language, PunctuationContext.SENTENCE_END, model)


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
    
    # Use centralized punctuation logic
    return _should_add_terminal_punctuation(sentence, language, PunctuationContext.SENTENCE_END)

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
        # Fallback to English for unsupported languages
        name = 'en_core_web_sm'
        logger.warning(f"Language '{language}' not supported for spaCy. Using English model as fallback.")
    try:
        nlp = spacy.load(name, disable=["lemmatizer"])  # speed
        _SPACY_PIPELINES[language] = nlp
        return nlp
    except Exception as e:
        # If we can't load the model, this is a critical error since SpaCy is now mandatory
        raise RuntimeError(f"Failed to load spaCy model '{name}': {e}")


def _detect_english_phrases_with_spacy(text: str, target_language: str) -> set:
    """Detect English phrases within text using spaCy language detection or heuristics.
    
    Returns a set of token indices that are likely English words.
    Prioritizes NER entities (locations, people, organizations) to avoid misclassification.
    """
    english_token_idxs = set()
    
        
    # Get both the target language pipeline and English pipeline
    target_nlp = _get_spacy_pipeline(target_language)
    english_nlp = _get_spacy_pipeline('en')
        
    try:
        # Process with target language pipeline first
        doc = target_nlp(text)
        
        # STEP 1: Identify protected entities (locations, people, orgs) that should NOT be treated as English
        protected_entities = set()
        protected_types = {"PERSON", "ORG", "GPE", "LOC", "NORP"}  # Include demonyms (NORP)
        for ent in doc.ents:
            if ent.label_ in protected_types:
                # Filter out obviously incorrect entity classifications for Spanish
                if target_language == 'es' and len(ent.text) == 1:
                    continue  # Single characters are rarely valid entities
                
                # Don't protect common Spanish words that spaCy incorrectly classifies as entities
                if target_language == 'es':
                    ent_text_lower = ent.text.lower()
                    # Skip entities that are obviously Spanish verbs/nouns being misclassified
                    if (re.match(r'^[a-z]+(ar|er|ir)(me|te|se|nos|os)?$', ent_text_lower) or  # infinitive + reflexive
                        re.match(r'^[a-z]+(me|te|se|nos|os)$', ent_text_lower) or  # reflexive verbs
                        # Common Spanish noun patterns that shouldn't be entities
                        re.match(r'^[a-z]+(a|o|as|os)$', ent_text_lower) and len(ent_text_lower) > 3):  # likely common nouns
                        continue
                
                for token in ent:
                    protected_entities.add(token.i)
            elif ent.label_ == "MISC":
                # For MISC entities, don't protect any tokens - let the English detection handle them
                # This prevents spaCy's confused mixed-language groupings from interfering
                pass
        
        # STEP 1b: Add heuristic protection for common location patterns that spaCy might miss in mixed content
        # Look for words in location-indicating contexts (case-insensitive to catch lowercase instances)
        location_context_patterns = [
            r'\bde\s+([a-záéíóúñA-ZÁÉÍÓÚÑ]+)',  # "de colombia", "de santander" 
            r'\ben\s+([a-záéíóúñA-ZÁÉÍÓÚÑ]+)',  # "en colombia"
            r'\ba\s+([a-záéíóúñA-ZÁÉÍÓÚÑ]+)',   # "a colombia"
            r'\bto\s+([a-záéíóúñA-ZÁÉÍÓÚÑ]+)',  # "to colombia" (English)
            r'\bin\s+([a-záéíóúñA-ZÁÉÍÓÚÑ]+)', # "in colombia" (English)
            r'\bgoing\s+to\s+([a-záéíóúñA-ZÁÉÍÓÚÑ]+)',  # "going to colombia"
            r'\bvivo\s+en\s+([a-záéíóúñA-ZÁÉÍÓÚÑ]+)',  # "vivo en colombia"
        ]
        
        for pattern in location_context_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                location_word = match.group(1)
                # Find the token index for this location word
                for token in doc:
                    if (token.text.lower() == location_word.lower() and 
                        token.idx >= match.start(1) and token.idx < match.end(1)):
                        protected_entities.add(token.i)
                        break
        
        # STEP 1c: Use cross-linguistic analysis - if a word appears to be PROPN in English pipeline but not in Spanish context,
        # it might be a location that Spanish spaCy missed due to mixed content
        if english_nlp and target_language == 'es':
            try:
                # Process same text with English pipeline to see if it catches locations
                en_doc = english_nlp(text)
                for en_ent in en_doc.ents:
                    if en_ent.label_ in {"GPE", "LOC"}:
                        # Find corresponding tokens in Spanish doc
                        for token in doc:
                            if (token.text.lower() == en_ent.text.lower() and 
                                token.idx >= en_ent.start_char and token.idx < en_ent.end_char):
                                protected_entities.add(token.i)
            except Exception:
                pass
        
        # STEP 2: Use spacy-language-detection if available
        if SPACY_LANG_DETECTION_AVAILABLE and target_language == 'es':
            try:
                # Add language detector to a temporary pipeline
                temp_nlp = spacy.load('es_core_news_sm')
                language_detector = LanguageDetector()
                temp_nlp.add_pipe(language_detector, name='language_detector', last=True)
                
                # Process text with language detection
                lang_doc = temp_nlp(text)
                
                # Check sentences for English content
                for sent in lang_doc.sents:
                    if hasattr(sent._, 'language') and sent._.language.get('language') == 'en':
                        # Mark tokens as English, but exclude protected entities
                        for token in sent:
                            if token.i not in protected_entities:
                                english_token_idxs.add(token.i)
            except Exception:
                # Fall through to heuristic method
                pass
        
        # STEP 3: Fallback heuristics for English detection (with entity protection)
        if not english_token_idxs:
            # Look for English patterns using linguistic features
            for token in doc:
                # Skip if this token is part of a protected entity
                if token.i in protected_entities:
                    continue
                    
                is_likely_english = False
                
                # Check for common English function words that don't exist in Spanish
                english_only_words = {
                    'i', 'am', 'going', 'to', 'the', 'a', 'an', 'this', 'that', 'these', 'those',
                    'you', 'we', 'they', 'he', 'she', 'it', 'me', 'him', 'her', 'us', 'them',
                    'my', 'your', 'his', 'her', 'its', 'our', 'their',
                    'do', 'does', 'did', 'don\'t', 'doesn\'t', 'didn\'t',
                    'have', 'has', 'had', 'haven\'t', 'hasn\'t', 'hadn\'t',
                    'will', 'would', 'won\'t', 'wouldn\'t', 'can', 'could', 'can\'t', 'couldn\'t',
                    'should', 'shouldn\'t', 'may', 'might', 'must', 'mustn\'t',
                    'is', 'are', 'was', 'were', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t',
                    'be', 'being', 'been', 'get', 'got', 'getting',
                    'think', 'thought', 'know', 'knew', 'see', 'saw', 'look', 'looked',
                    'good', 'bad', 'better', 'best', 'worse', 'worst',
                    'in', 'on', 'at', 'by', 'for', 'with', 'without', 'about', 'from',
                    'and', 'or', 'but', 'so', 'if', 'when', 'where', 'why', 'how', 'what', 'who',
                    'very', 'really', 'quite', 'just', 'only', 'also', 'too', 'still', 'already',
                    'test', 'skirt', 'shirt', 'dress', 'pants', 'clothes'
                }
                
                if token.text.lower() in english_only_words:
                    is_likely_english = True
                
                # Check for English morphological patterns
                elif re.match(r'^[a-z]+(ing|ed|ly|er|est)$', token.text.lower()):
                    is_likely_english = True
                
                # Check for English contractions
                elif re.match(r"^[a-z]+'[a-z]+$", token.text.lower()):
                    is_likely_english = True
                
                if is_likely_english:
                    english_token_idxs.add(token.i)
            
            # STEP 4: Extend detection to nearby tokens in likely English phrases
            # If we find English words, extend to adjacent non-Spanish tokens
            extended_idxs = set(english_token_idxs)
            for idx in english_token_idxs:
                # Check previous and next tokens
                for offset in [-2, -1, 1, 2]:
                    neighbor_idx = idx + offset
                    if 0 <= neighbor_idx < len(doc):
                        neighbor = doc[neighbor_idx]
                        # Skip if neighbor is a protected entity
                        if neighbor_idx in protected_entities:
                            continue
                        # Skip if clearly Spanish or punctuation
                        if (neighbor.is_punct or neighbor.is_space or
                            neighbor.text.lower() in {'de', 'la', 'el', 'en', 'y', 'que', 'a', 'con', 'por', 'para', 'es', 'son', 'soy'}):
                            continue
                        # Include if length > 1 and not clearly Spanish
                        if len(neighbor.text) > 1:
                            extended_idxs.add(neighbor_idx)
            
            english_token_idxs = extended_idxs
                
    except Exception as e:
        logger.debug(f"Error in English phrase detection: {e}")
        
    return english_token_idxs


def _apply_spacy_capitalization(text: str, language: str) -> str:
    """Capitalize named entities and proper nouns using spaCy, conservatively.

    - Capitalize tokens in entities PERSON/ORG/GPE/LOC
    - Capitalize tokens with POS=PROPN  
    - Preserve particles: de, del, la, las, los, y, o, da, di, do, du, van, von, du, des, le
    - Skip URLs/emails/handles
    - Detect and properly handle English phrases in mixed-language content
    - Fix overcapitalized English words while preserving proper sentence starts
    """
    nlp = _get_spacy_pipeline(language)
    if not text.strip():
        return text

    try:
        doc = nlp(text)
    except Exception:
        return text

    # For Spanish, be very conservative with spaCy's entity detection since it's unreliable with mixed content
    ent_token_idxs = set()
    if language == 'es':
        # Only trust clear, unambiguous entities from spaCy
        for ent in doc.ents:
            if ent.label_ in {"ORG"}:  # Organizations are usually reliable
                for t in ent:
                    ent_token_idxs.add(t.i)
            elif ent.label_ in {"GPE", "LOC"} and len(ent.text) > 1:
                # Only trust location entities that don't look like Spanish common words
                ent_lower = ent.text.lower()
                if not (re.match(r'^[a-z]+(ar|er|ir)(me|te|se|nos|os)?$', ent_lower) or
                        re.match(r'^[a-z]+(me|te|se|nos|os)$', ent_lower) or
                        re.match(r'^[a-z]+(a|o|as|os)$', ent_lower) and len(ent_lower) > 3):
                    for t in ent:
                        ent_token_idxs.add(t.i)
    else:
        # For other languages, trust spaCy more
        ent_types = {"PERSON", "ORG", "GPE", "LOC"}
        for ent in doc.ents:
            if ent.label_ in ent_types:
                for t in ent:
                    ent_token_idxs.add(t.i)

    # Never capitalize these connectors (common Spanish function words)
    cfg = _get_language_config(language)
    connectors = cfg.connectors
    
    # Detect English phrases for mixed-language content
    english_phrase_idxs = set()
    if language == 'es':
        english_phrase_idxs = _detect_english_phrases_with_spacy(text, language)

    def should_capitalize(tok) -> bool:
        txt = tok.text
        # Skip tokens in detected English phrases (but allow proper nouns/entities)
        if tok.i in english_phrase_idxs and tok.i not in ent_token_idxs:
            return False
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
            # For Spanish: be very conservative with PROPN since spaCy often misclassifies common words  
            if language == 'es':
                # Don't capitalize obvious Spanish morphological patterns
                if (re.match(r'^[a-z]+(ar|er|ir)(me|te|se|nos|os)?$', low) or  # verbs + reflexives
                    re.match(r'^[a-z]+(me|te|se|nos|os)$', low)):  # pure reflexives
                    return False
                # Only capitalize if already uppercase (preserving existing capitalization)
                if not tok.text[0].isupper():
                    return False
            else:
                # For other languages, be more permissive
                if tok.text[0].isupper():
                    return True
            return True
        
        # Additional heuristic: capitalize words that look like location names based on context
        # even if spaCy didn't detect them as entities (common in mixed-language content)
        # Be more restrictive and only capitalize when we have strong indicators it's a location
        if (language == 'es' and tok.is_alpha and len(tok.text) > 3 and 
            not low in connectors and tok.i > 0):
            
            prev_token = doc[tok.i - 1]
            
            # Only capitalize after specific contextual cues that strongly suggest locations
            location_context = False
            
            # Pattern 1: After specific verbs + prepositions that indicate location
            if (tok.i > 1):
                prev2_token = doc[tok.i - 2]
                verb_location_patterns = {
                    ('vivo', 'en'), ('trabajo', 'en'), ('nací', 'en'), ('estudié', 'en'),
                    ('voy', 'a'), ('fui', 'a'), ('viajé', 'a'), ('mudé', 'a'),
                    ('vengo', 'de'), ('soy', 'de'), ('llegué', 'de')
                }
                if (prev2_token.text.lower(), prev_token.text.lower()) in verb_location_patterns:
                    location_context = True
            
            # Pattern 2: After "en" when the word looks like a proper noun (starts with capital or uncommon ending)
            if (prev_token.text.lower() == 'en' and 
                (tok.text[0].isupper() or  # Already capitalized in input
                 low.endswith(('nia', 'dad', 'burg', 'land', 'shire', 'ford', 'ton')))):  # Place-like endings
                location_context = True
            
            # Pattern 3: After "de" only if the word was already capitalized (indicating proper noun)
            if (prev_token.text.lower() == 'de' and tok.text[0].isupper()):
                # Only capitalize if it was already a proper noun in the input
                location_context = True
            
            if location_context:
                return True
            # English patterns: "in Colombia", "to Colombia", "going to Colombia"  
            if (prev_token.text.lower() in {'in', 'to'} and 
                len(prev_token.text) <= 2):
                return True
            # Two tokens back: "vivo en Colombia", "going to Colombia"
            if (tok.i > 1):
                prev2_token = doc[tok.i - 2]
                if (prev2_token.text.lower() in {'vivo', 'trabajo', 'nací', 'estudié'} and
                    prev_token.text.lower() == 'en'):
                    return True
                if (prev2_token.text.lower() in {'going', 'traveling', 'moving'} and
                    prev_token.text.lower() == 'to'):
                    return True
        
        return False

    out = []
    for tok in doc:
        t = tok.text
        
        # Fix overcapitalized English words in detected phrases
        if tok.i in english_phrase_idxs and t and t[0].isupper() and len(t) > 1:
            # Lowercase English words in phrases, except sentence-initial "I"
            if t.lower() == 'i':
                # Keep "I" capitalized as it's always capitalized in English
                pass
            else:
                # Check if this is at the start of a sentence
                is_sentence_start = (tok.i == 0 or 
                                   (tok.i > 0 and doc[tok.i - 1].text in {'.', '!', '?', '¿', '¡'}))
                if not is_sentence_start:
                    t = t[0].lower() + t[1:]
        elif should_capitalize(tok):
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
    # Updated patterns to include accented characters for domains like sinónimosonline.com
    text = re.sub(r"\b([a-zA-Z0-9\u00C0-\u017F\-]+)\s*[.\-]\s*(com|net|org|co|es|io|edu|gov|uk|us|ar|mx)\b", lambda m: f"{m.group(1)}.{m.group(2).lower()}", text, flags=re.IGNORECASE)
    # Also handle 'www . domain . tld'
    text = re.sub(r"\b(www)\s*[.\-]\s*([a-zA-Z0-9\u00C0-\u017F\-]+)\s*[.\-]\s*(com|net|org|co|es|io|edu|gov|uk|us|ar|mx)\b", lambda m: f"{m.group(1)}.{m.group(2)}.{m.group(3).lower()}", text, flags=re.IGNORECASE)

    # Ensure TLDs are lowercase within domains: label.TLD -> label.tld
    def _lowercase_tld(m):
        return f"{m.group(1)}.{m.group(2).lower()}"
    # Updated pattern to include accented characters for domains like sinónimosonline.com
    text = re.sub(r"\b([a-zA-Z0-9\u00C0-\u017F\-]+)\.([A-Za-z]{2,24})\b", _lowercase_tld, text, flags=re.IGNORECASE)

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
    text_masked_for_coord = mask_domains(text, use_exclusions=True, language='es')
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

    # Merge location appositives split across sentences: 
    # "..., de Texas." + "Estados Unidos." -> "..., de Texas, Estados Unidos."
    try:
        parts6 = _split_sentences_preserving_delims(text)
        i = 0
        while i + 3 < len(parts6):
            s1 = (parts6[i] or '').strip()
            p1 = parts6[i+1]
            s2 = (parts6[i+2] or '').strip()
            p2 = parts6[i+3]
            if p1 == '.' and re.search(r"(?i),?\s*de\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑ-]+\s*$", s1):
                if s2 and s2[0].isupper():
                    # Insert comma if missing before appositive continuation
                    if not s1.endswith(','):
                        s1 = s1 + ','
                    merged = f"{s1} {s2}"
                    parts6[i] = merged
                    parts6[i+1] = p2
                    del parts6[i+2:i+4]
                    continue
            i += 2
        text = ''.join(parts6)
    except Exception:
        pass

    # Capitalize sentence starts (handles leading punctuation/quotes)
    # Mask domains before capitalization to prevent splitting on domain periods
    text_masked_for_caps = mask_domains(text, use_exclusions=True, language='es')
    text_capitalized = _es_capitalize_sentence_starts(text_masked_for_caps)
    text = unmask_domains(text_capitalized)

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