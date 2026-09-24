#!/usr/bin/env python3
"""
Domain Detection and Masking Utilities

Centralized logic for detecting, masking, and unmasking domain names
to prevent false positives with Spanish words and ensure consistent
domain handling across the codebase.
"""

import re
from typing import Callable


# Centralized TLD and exclusion patterns
SINGLE_TLDS = r"com|net|org|co|es|io|edu|gov|uk|us|ar|mx|de|fr|br|ca|au|pt"
COMPOUND_TLDS = r"co\.uk|com\.ar|com\.mx|com\.br|com\.au|co\.jp|co\.in|gov\.uk|org\.uk|ac\.uk"

# A deliberately narrower TLD list, historically copy-pasted into several call
# sites outside this module (sentence_formatter, punctuation_restorer). It omits
# the country TLDs that most often collide with ordinary words (de, fr, br, ca,
# au). Kept separate from SINGLE_TLDS on purpose: widening those call sites to
# the full list would change Spanish and English behavior.
SINGLE_TLDS_CONSERVATIVE = r"com|net|org|co|es|io|edu|gov|uk|us|ar|mx|pt"

# Uppercase/lowercase accented Latin characters used by the supported languages.
# Defined here (the leaf module) so punctuation_restorer and podscripter can share
# them without an import cycle. The Spanish-era classes ([A-ZÁÉÍÓÚÑ]) predate
# Portuguese support and omit the nasal/circumflex vowels and the cedilla.
UPPER_ACCENTED = "ÁÉÍÓÚÑÃÕÂÊÔÀÇÜ"
LOWER_ACCENTED = "áéíóúñãõâêôàçü"

# Spanish words that should NOT be treated as domain labels
# These are common words that might appear before TLD-like suffixes in normal Spanish text
SPANISH_EXCLUSIONS = r"uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|este|esta|ese|esa|aquel|aquella|el|la|lo|los|las|mi|tu|su|nuestro|vuestro|han|son|fue|era|muy|mas|pero|por|para|con|sin|como|cuando|donde|porque|aunque|mientras|durante|desde|hasta|entre|sobre|bajo|ante|tras|hacia|según|contra|mediante|salvo|excepto|incluso|menos|antes|después|luego|entonces|ahora|aquí|ahí|allí|allá|ayer|hoy|mañana|siempre|nunca|jamás|también|tampoco|solo|sólo|tanto|tan|más|menos|mejor|peor|mayor|menor|mismo|misma|otro|otra|cada|todo|toda|algún|alguna|ningún|ninguna|varios|varias|mucho|mucha|poco|poca|bastante|demasiado|algo|nada|alguien|nadie|cualquier|cualquiera"

# Portuguese words that should NOT be treated as domain labels. Applied in
# addition to SPANISH_EXCLUSIONS when language == 'pt' (many entries overlap:
# "também", "entre", "durante", "desde", "nada", "algo").
# Unaccented variants are included deliberately: ASR output frequently drops
# Portuguese diacritics, and an unaccented "voce"/"nao" must be excluded too.
PORTUGUESE_EXCLUSIONS = r"um|uma|dois|duas|três|tres|quatro|cinco|seis|sete|oito|nove|dez|este|esta|esse|essa|aquele|aquela|o|a|os|as|meu|minha|teu|tua|seu|sua|nosso|nossa|é|e|são|sao|foi|era|com|sem|muito|muita|mais|menos|mas|por|para|como|quando|onde|porque|embora|enquanto|durante|desde|até|ate|entre|sobre|sob|ante|perante|contra|mediante|salvo|exceto|inclusive|antes|depois|logo|então|entao|agora|aqui|ali|lá|la|ontem|hoje|amanhã|amanha|sempre|nunca|jamais|também|tambem|tampouco|só|so|somente|tanto|tão|tao|melhor|pior|maior|menor|mesmo|mesma|outro|outra|cada|todo|toda|algum|alguma|nenhum|nenhuma|vários|varias|pouco|pouca|bastante|demais|algo|nada|alguém|alguem|ninguém|ninguem|qualquer|tudo|você|voce|vocês|voces|nós|nos|não|nao|ainda|pois"

# TLDs suppressed per language because they collide with extremely common words.
# Two tables, because the two entry points carry different risk:
#   * mask_domains matches only a *contiguous* "label.tld", which in ASR output is
#     almost always a genuine domain.
#   * fix_spaced_domains actively rejoins "label. tld" across a sentence break,
#     so it needs to be stricter.
# Portuguese "com" (= "with") is the motivating case: ".com" must stay enabled for
# masking (it is the most common real TLD) but must never trigger a rejoin, or
# "acabou. Com ele" becomes "acabou.com ele".
_TLD_SUPPRESSIONS = {
    'es': {'de', 'es'},
    'pt': {'de'},
}
_TLD_SUPPRESSIONS_SPACED = {
    'es': {'de', 'es'},
    'pt': {'de', 'com'},
}

# Masking tokens
SINGLE_MASK = "__DOT__"
COMPOUND_MASK = "_DOT_"


def _tlds_for(language: str | None, spaced: bool = False) -> str:
    """Return the single-TLD alternation for `language`.

    Args:
        language: Language code, or None for the unrestricted list.
        spaced: True for the `fix_spaced_domains` (rejoin) path, which suppresses
            more TLDs than the masking path.
    """
    table = _TLD_SUPPRESSIONS_SPACED if spaced else _TLD_SUPPRESSIONS
    suppressed = table.get((language or '').lower())
    if not suppressed:
        return SINGLE_TLDS
    return "|".join(t for t in SINGLE_TLDS.split("|") if t not in suppressed)


def _is_excluded_label(label: str, language: str | None = None) -> bool:
    """Check if a label is a common word that should not be treated as a domain.

    The Spanish exclusions are applied for every language (long-standing
    behavior: they are a useful generic stopword guard). Portuguese adds its own
    list on top when language == 'pt'.
    """
    if re.match(rf"^({SPANISH_EXCLUSIONS})$", label, re.IGNORECASE):
        return True
    if language and language.lower() == 'pt':
        return bool(re.match(rf"^({PORTUGUESE_EXCLUSIONS})$", label, re.IGNORECASE))
    return False


def _is_spanish_word(label: str) -> bool:
    """Back-compat alias for `_is_excluded_label` with no language context."""
    return _is_excluded_label(label)


def mask_domains(text: str, use_exclusions: bool = True, language: str | None = None) -> str:
    """
    Mask domains in text to protect them from text processing.
    
    Args:
        text: Input text that may contain domains
        use_exclusions: Whether to apply Spanish word exclusions (default True)
        language: Language code (e.g., 'es', 'en') for language-specific exclusions
        
    Returns:
        Text with domains masked using __DOT__ and _DOT_ tokens
        
    Example:
        "Visit google.com and uno.de" -> "Visit google__DOT__com and uno.de" (with exclusions)
        "Visit www.google.com" -> "Visit www__DOT__google__DOT__com"
        "Necesita ser tratada.de hecho" -> "Necesita ser tratada.de hecho" (Spanish: .de/.es excluded)
    """
    # Suppress TLDs that collide with very common words in this language
    # (Spanish: .de/.es; Portuguese: .de). See _TLD_SUPPRESSIONS.
    single_tlds = _tlds_for(language)

    def _mask_single(m):
        label = m.group(1)
        tld = m.group(2)
        if use_exclusions and _is_excluded_label(label, language):
            return m.group(0)  # Return unchanged if it's a common word
        return f"{label}{SINGLE_MASK}{tld}"

    def _mask_compound(m):
        label = m.group(1)
        compound_tld = m.group(2)
        if use_exclusions and _is_excluded_label(label, language):
            return m.group(0)  # Return unchanged if it's a common word
        # Replace dots in compound TLD: "co.uk" -> "co_DOT_uk"
        masked_tld = compound_tld.replace('.', COMPOUND_MASK)
        return f"{label}{SINGLE_MASK}{masked_tld}"
    
    def _mask_subdomain(m):
        subdomain = m.group(1)  # e.g., "www."
        domain = m.group(2)     # e.g., "google"  
        tld = m.group(3)        # e.g., "com"
        if use_exclusions and _is_excluded_label(domain, language):
            return m.group(0)  # Return unchanged if the domain part is a common word
        # Replace dots with mask tokens: "www.domain.tld" -> "www__DOT__domain__DOT__tld"
        return f"{subdomain.replace('.', SINGLE_MASK)}{domain}{SINGLE_MASK}{tld}"
    
    def _mask_subdomain_compound(m):
        subdomain = m.group(1)     # e.g., "www."
        domain = m.group(2)        # e.g., "bbc"
        compound_tld = m.group(3)  # e.g., "co.uk"
        if use_exclusions and _is_excluded_label(domain, language):
            return m.group(0)  # Return unchanged if the domain part is a common word
        # Replace dots: "www.domain.co.uk" -> "www__DOT__domain__DOT__co_DOT_uk"
        masked_tld = compound_tld.replace('.', COMPOUND_MASK)
        return f"{subdomain.replace('.', SINGLE_MASK)}{domain}{SINGLE_MASK}{masked_tld}"
    
    # CRITICAL: Apply subdomain patterns FIRST to avoid conflicts with basic domain patterns
    
    # Mask subdomain compound TLDs: "www.domain.co.uk" -> "www__DOT__domain__DOT__co_DOT_uk"
    subdomain_compound_pattern = rf"\b((?:www|ftp|mail|blog|shop|app|api|cdn|static|news|support|help|docs|admin|secure|login|m|mobile|store|sub|dev|test|staging|prod|beta|alpha)\.)([a-zA-Z0-9\u00C0-\u017F\-]+)\.({COMPOUND_TLDS})\b"
    masked = re.sub(subdomain_compound_pattern, _mask_subdomain_compound, text, flags=re.IGNORECASE)
    
    # Mask subdomain single TLDs: "www.domain.com" -> "www__DOT__domain__DOT__com"
    subdomain_single_pattern = rf"\b((?:www|ftp|mail|blog|shop|app|api|cdn|static|news|support|help|docs|admin|secure|login|m|mobile|store|sub|dev|test|staging|prod|beta|alpha)\.)([a-zA-Z0-9\u00C0-\u017F\-]+)\.({single_tlds})\b"
    masked = re.sub(subdomain_single_pattern, _mask_subdomain, masked, flags=re.IGNORECASE)
    
    # Then apply compound TLD masking for remaining domains (non-subdomain)
    # Mask compound TLDs: "domain.co.uk" -> "domain__DOT__co_DOT_uk"
    # Updated pattern to include accented characters (Unicode \u00C0-\u017F covers Latin-1 Supplement and Latin Extended-A)
    masked = re.sub(rf"\b([a-zA-Z0-9\u00C0-\u017F\-]+)\.({COMPOUND_TLDS})\b", _mask_compound, masked, flags=re.IGNORECASE)
    
    # Finally mask single TLDs: "domain.com" -> "domain__DOT__com"  
    # Updated pattern to include accented characters for domains like sinónimosonline.com
    masked = re.sub(rf"\b([a-zA-Z0-9\u00C0-\u017F\-]+)\.({single_tlds})\b", _mask_single, masked, flags=re.IGNORECASE)
    
    return masked


def unmask_domains(text: str) -> str:
    """
    Unmask domains by replacing masking tokens with actual dots.
    
    Args:
        text: Text with masked domains
        
    Returns:
        Text with domains unmasked
        
    Example:
        "Visit google__DOT__com and bbc__DOT__co_DOT_uk" -> "Visit google.com and bbc.co.uk"
    """
    # Unmask single TLDs: "domain__DOT__com" -> "domain.com"
    unmasked = text.replace(SINGLE_MASK, ".")
    
    # Unmask compound TLDs: "domain.co_DOT_uk" -> "domain.co.uk"
    unmasked = unmasked.replace(COMPOUND_MASK, ".")
    
    return unmasked


def fix_spaced_domains(text: str, use_exclusions: bool = True, language: str | None = None) -> str:
    """
    Fix domains that have been broken with spaces: "domain. com" -> "domain.com"
    
    Args:
        text: Text that may contain broken domains with spaces
        use_exclusions: Whether to apply Spanish word exclusions (default True)
        language: Language code for language-specific exclusions
        
    Returns:
        Text with spaced domains fixed
        
    Example:
        "Visit google. com and uno. de" -> "Visit google.com and uno. de" (with exclusions)
        "Tratada. de hecho" -> "Tratada. de hecho" (Spanish: .de/.es excluded)
    """
    # Suppress TLDs that collide with very common words in this language. Stricter
    # than the masking path: Portuguese also suppresses ".com" here, because "com"
    # means "with" and this function rejoins across a sentence break
    # ("acabou. Com ele" must NOT become "acabou.com ele").
    single_tlds = _tlds_for(language, spaced=True)

    def _fix_single_tld(m):
        label = m.group(1)
        tld = m.group(2)
        if use_exclusions and _is_excluded_label(label, language):
            return m.group(0)  # Return unchanged if it's a common word
        return f"{label}.{tld.lower()}"

    def _fix_compound_tld(pattern_func):
        def _compound_replacer(m):
            label = m.group(1)
            if use_exclusions and _is_excluded_label(label, language):
                return m.group(0)  # Return unchanged if it's a common word
            return pattern_func(m)
        return _compound_replacer
    
    # Fix compound TLDs FIRST (before single TLDs to avoid conflicts)
    # Updated patterns to include accented characters for domains like sinónimosonline.com
    fixed = re.sub(rf"\b([a-zA-Z0-9\u00C0-\u017F\-]+)\.\s+(co)\.\s+(uk)\b", 
                   _fix_compound_tld(lambda m: f"{m.group(1)}.co.uk"), text, flags=re.IGNORECASE)
    fixed = re.sub(rf"\b([a-zA-Z0-9\u00C0-\u017F\-]+)\.\s+(com)\.\s+(ar|mx|br|au)\b", 
                   _fix_compound_tld(lambda m: f"{m.group(1)}.com.{m.group(3).lower()}"), fixed, flags=re.IGNORECASE)
    fixed = re.sub(rf"\b([a-zA-Z0-9\u00C0-\u017F\-]+)\.\s+(co)\.\s+(jp|in)\b", 
                   _fix_compound_tld(lambda m: f"{m.group(1)}.co.{m.group(3).lower()}"), fixed, flags=re.IGNORECASE)
    fixed = re.sub(rf"\b([a-zA-Z0-9\u00C0-\u017F\-]+)\.\s+(gov|org|ac)\.\s+(uk)\b", 
                   _fix_compound_tld(lambda m: f"{m.group(1)}.{m.group(2).lower()}.uk"), fixed, flags=re.IGNORECASE)
    
    # Fix single TLDs: "domain. com" -> "domain.com" (after compound TLDs)
    # Updated pattern to include accented characters for domains like sinónimosonline.com
    fixed = re.sub(rf"\b([a-zA-Z0-9\u00C0-\u017F\-]+)\.\s+({single_tlds})\b", _fix_single_tld, fixed, flags=re.IGNORECASE)
    
    return fixed


def _get_domain_safe_split_pattern() -> str:
    """
    Get a regex pattern for splitting text that won't break domains.
    
    Returns:
        Regex pattern that can be used with masked text to split sentences
        while preserving domain integrity
    """
    # This pattern splits on sentence terminators followed by whitespace and capital letters
    # It should be used on masked text where domains are protected
    return rf"(?<=[.!?])\s+(?=[A-Z{UPPER_ACCENTED}¿¡])"


def apply_safe_text_processing(text: str, processing_func: Callable[[str], str], use_exclusions: bool = True, language: str | None = None) -> str:
    r"""
    Apply text processing function while protecting domains from modification.
    
    Args:
        text: Input text
        processing_func: Function that processes text (e.g., adds spaces, changes case)
        use_exclusions: Whether to apply Spanish word exclusions (default True)
        language: Language code for language-specific exclusions
        
    Returns:
        Processed text with domains protected
        
    Example:
        def add_spaces(s): return re.sub(r'\.([A-Z])', r'. \\1', s)
        apply_safe_text_processing("Visit google.com.Then go home", add_spaces)
        -> "Visit google.com. Then go home"
    """
    masked = mask_domains(text, use_exclusions, language)
    processed = processing_func(masked)
    return unmask_domains(processed)


def create_domain_aware_regex(pattern: str, replacement: str, use_exclusions: bool = True, language: str | None = None) -> Callable[[str], str]:
    r"""
    Create a domain-aware regex function that masks domains before applying the regex.
    
    Args:
        pattern: Regex pattern to apply
        replacement: Replacement string
        use_exclusions: Whether to apply Spanish word exclusions (default True)
        language: Language code for language-specific exclusions
        
    Returns:
        Function that applies the regex while protecting domains
        
    Example:
        space_after_period = create_domain_aware_regex(r'\.([A-Z])', r'. \\1')
        result = space_after_period("Visit google.com.Then go home")
        # -> "Visit google.com. Then go home"
    """
    def _domain_aware_sub(text: str) -> str:
        return apply_safe_text_processing(
            text, 
            lambda s: re.sub(pattern, replacement, s, flags=re.IGNORECASE), 
            use_exclusions,
            language
        )
    return _domain_aware_sub
