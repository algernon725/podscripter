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
SINGLE_TLDS = r"com|net|org|co|es|io|edu|gov|uk|us|ar|mx|de|fr|it|nl|br|ca|au|jp|cn|in|ru"
COMPOUND_TLDS = r"co\.uk|com\.ar|com\.mx|com\.br|com\.au|co\.jp|co\.in|gov\.uk|org\.uk|ac\.uk"

# Spanish words that should NOT be treated as domain labels
# These are common words that might appear before TLD-like suffixes in normal Spanish text
SPANISH_EXCLUSIONS = r"uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|este|esta|ese|esa|aquel|aquella|el|la|lo|los|las|mi|tu|su|nuestro|vuestro|han|son|fue|era|muy|mas|pero|por|para|con|sin|como|cuando|donde|porque|aunque|mientras|durante|desde|hasta|entre|sobre|bajo|ante|tras|hacia|según|contra|mediante|salvo|excepto|incluso|menos|antes|después|luego|entonces|ahora|aquí|ahí|allí|allá|ayer|hoy|mañana|siempre|nunca|jamás|también|tampoco|solo|sólo|tanto|tan|más|menos|mejor|peor|mayor|menor|mismo|misma|otro|otra|cada|todo|toda|algún|alguna|ningún|ninguna|varios|varias|mucho|mucha|poco|poca|bastante|demasiado|algo|nada|alguien|nadie|cualquier|cualquiera"

# Masking tokens
SINGLE_MASK = "__DOT__"
COMPOUND_MASK = "_DOT_"


def _is_spanish_word(label: str) -> bool:
    """Check if a label is a common Spanish word that should not be treated as a domain."""
    return bool(re.match(rf"^({SPANISH_EXCLUSIONS})$", label, re.IGNORECASE))


def mask_domains(text: str, use_exclusions: bool = True, language: str = None) -> str:
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
        "Visit google.com and uno.de" -> "Visit google__DOT__com and uno__DOT__de" (without exclusions)
        "Necesita ser tratada.de hecho" -> "Necesita ser tratada.de hecho" (Spanish: .de excluded)
    """
    # Exclude .de TLD for Spanish text since "de" is an extremely common Spanish preposition
    single_tlds = SINGLE_TLDS
    if language and language.lower() == 'es':
        single_tlds = single_tlds.replace('de|', '').replace('|de', '')
    
    def _mask_single(m):
        label = m.group(1)
        tld = m.group(2)
        if use_exclusions and _is_spanish_word(label):
            return m.group(0)  # Return unchanged if it's a Spanish word
        return f"{label}{SINGLE_MASK}{tld}"
        
    def _mask_compound(m):
        label = m.group(1)
        compound_tld = m.group(2)
        if use_exclusions and _is_spanish_word(label):
            return m.group(0)  # Return unchanged if it's a Spanish word
        # Replace dots in compound TLD: "co.uk" -> "co_DOT_uk"
        masked_tld = compound_tld.replace('.', COMPOUND_MASK)
        return f"{label}{SINGLE_MASK}{masked_tld}"
    
    # CRITICAL: Apply compound TLD masking FIRST to avoid conflicts with single TLDs
    # Mask compound TLDs: "domain.co.uk" -> "domain__DOT__co_DOT_uk"
    masked = re.sub(rf"\b([a-z0-9\-]+)\.({COMPOUND_TLDS})\b", _mask_compound, text, flags=re.IGNORECASE)
    
    # Mask single TLDs: "domain.com" -> "domain__DOT__com"
    masked = re.sub(rf"\b([a-z0-9\-]+)\.({single_tlds})\b", _mask_single, masked, flags=re.IGNORECASE)
    
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


def fix_spaced_domains(text: str, use_exclusions: bool = True, language: str = None) -> str:
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
        "Tratada. de hecho" -> "Tratada. de hecho" (Spanish: .de excluded)
    """
    # Exclude .de TLD for Spanish text since "de" is an extremely common Spanish preposition
    single_tlds = SINGLE_TLDS
    if language and language.lower() == 'es':
        single_tlds = single_tlds.replace('de|', '').replace('|de', '')
    
    def _fix_single_tld(m):
        label = m.group(1)
        tld = m.group(2)
        if use_exclusions and _is_spanish_word(label):
            return m.group(0)  # Return unchanged if it's a Spanish word
        return f"{label}.{tld.lower()}"
    
    def _fix_compound_tld(pattern_func):
        def _compound_replacer(m):
            label = m.group(1)
            if use_exclusions and _is_spanish_word(label):
                return m.group(0)  # Return unchanged if it's a Spanish word
            return pattern_func(m)
        return _compound_replacer
    
    # Fix compound TLDs FIRST (before single TLDs to avoid conflicts)
    fixed = re.sub(rf"\b([a-z0-9\-]+)\.\s+(co)\.\s+(uk)\b", 
                   _fix_compound_tld(lambda m: f"{m.group(1)}.co.uk"), text, flags=re.IGNORECASE)
    fixed = re.sub(rf"\b([a-z0-9\-]+)\.\s+(com)\.\s+(ar|mx|br|au)\b", 
                   _fix_compound_tld(lambda m: f"{m.group(1)}.com.{m.group(3).lower()}"), fixed, flags=re.IGNORECASE)
    fixed = re.sub(rf"\b([a-z0-9\-]+)\.\s+(co)\.\s+(jp|in)\b", 
                   _fix_compound_tld(lambda m: f"{m.group(1)}.co.{m.group(3).lower()}"), fixed, flags=re.IGNORECASE)
    fixed = re.sub(rf"\b([a-z0-9\-]+)\.\s+(gov|org|ac)\.\s+(uk)\b", 
                   _fix_compound_tld(lambda m: f"{m.group(1)}.{m.group(2).lower()}.uk"), fixed, flags=re.IGNORECASE)
    
    # Fix single TLDs: "domain. com" -> "domain.com" (after compound TLDs)
    fixed = re.sub(rf"\b([a-z0-9\-]+)\.\s+({single_tlds})\b", _fix_single_tld, fixed, flags=re.IGNORECASE)
    
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
    return r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡])"


def apply_safe_text_processing(text: str, processing_func: Callable[[str], str], use_exclusions: bool = True, language: str = None) -> str:
    """
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


def create_domain_aware_regex(pattern: str, replacement: str, use_exclusions: bool = True, language: str = None) -> Callable[[str], str]:
    """
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


# Legacy compatibility - old function signatures for backward compatibility
def _mask_domains_legacy(text: str, tld_pattern: str = None) -> str:
    """Legacy function for backward compatibility."""
    return mask_domains(text, use_exclusions=True)


def _unmask_domains_legacy(text: str) -> str:
    """Legacy function for backward compatibility.""" 
    return unmask_domains(text)
