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

import re

# Try to import sentence transformers for better punctuation restoration
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: SentenceTransformers not available. Using basic punctuation restoration.")


def restore_punctuation(text, language='en'):
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
    
    # Use advanced punctuation restoration if available
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            return advanced_punctuation_restoration(text, language)
        except Exception as e:
            print(f"Warning: Advanced punctuation restoration failed: {e}")
            print("Falling back to basic punctuation restoration...")
    
    # Fallback: basic punctuation restoration
    return basic_punctuation_restoration(text, language)


def advanced_punctuation_restoration(text, language='en'):
    """
    Advanced punctuation restoration using sentence transformers and NLP techniques.
    
    Args:
        text (str): The transcribed text without proper punctuation
        language (str): Language code ('en', 'es', 'de', 'fr')
    
    Returns:
        str: Text with restored punctuation
    """
    # Language-specific sentence endings and patterns
    patterns = {
        'en': {
            'sentence_endings': [
                r'\b(thank you|thanks|goodbye|bye|see you|talk to you later)\b',
                r'\b(okay|ok|alright|all right)\b',
                r'\b(yes|no|yeah|nope|yep|nah)\b',
                r'\b(please|excuse me|sorry|pardon)\b'
            ],
            'question_words': [
                r'\b(what|where|when|why|how|who|which)\b',
                r'\b(can you|could you|would you|will you|do you|are you)\b'
            ]
        },
        'es': {
            'sentence_endings': [
                r'\b(gracias|adiós|hasta luego|nos vemos|chao)\b',
                r'\b(vale|ok|bien|está bien)\b',
                r'\b(sí|no|claro|por supuesto)\b',
                r'\b(por favor|perdón|disculpa|lo siento)\b'
            ],
            'question_words': [
                r'\b(qué|dónde|cuándo|por qué|cómo|quién|cuál)\b',
                r'\b(puedes|podrías|te gustaría|vas a|haces|eres)\b'
            ]
        },
        'de': {
            'sentence_endings': [
                r'\b(danke|tschüss|auf wiedersehen|bis später)\b',
                r'\b(okay|ok|gut|in ordnung)\b',
                r'\b(ja|nein|klar|natürlich)\b',
                r'\b(bitte|entschuldigung|sorry)\b'
            ],
            'question_words': [
                r'\b(was|wo|wann|warum|wie|wer|welche)\b',
                r'\b(kannst du|könntest du|würdest du|wirst du|machst du|bist du)\b'
            ]
        },
        'fr': {
            'sentence_endings': [
                r'\b(merci|au revoir|à bientôt|salut)\b',
                r'\b(okay|ok|d\'accord|ça va)\b',
                r'\b(oui|non|bien sûr|évidemment)\b',
                r'\b(s\'il vous plaît|pardon|désolé)\b'
            ],
            'question_words': [
                r'\b(quoi|où|quand|pourquoi|comment|qui|quel)\b',
                r'\b(peux-tu|pourrais-tu|voudrais-tu|vas-tu|fais-tu|es-tu)\b'
            ]
        }
    }
    
    lang_patterns = patterns.get(language, patterns['en'])
    
    # Step 1: Handle repeated words for emphasis in Spanish, French, and German
    if language == 'es':
        # Add commas between repeated "si" or "no" for emphasis
        text = re.sub(r'\b(sí)\s+\1\b', r'\1, \1', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(no)\s+\1\b', r'\1, \1', text, flags=re.IGNORECASE)
        # Handle more than 2 repetitions
        text = re.sub(r'\b(sí)\s+\1\s+\1\b', r'\1, \1, \1', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(no)\s+\1\s+\1\b', r'\1, \1, \1', text, flags=re.IGNORECASE)
    elif language == 'fr':
        # Add commas between repeated "oui" or "non" for emphasis
        text = re.sub(r'\b(oui)\s+\1\b', r'\1, \1', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(non)\s+\1\b', r'\1, \1', text, flags=re.IGNORECASE)
        # Handle more than 2 repetitions
        text = re.sub(r'\b(oui)\s+\1\s+\1\b', r'\1, \1, \1', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(non)\s+\1\s+\1\b', r'\1, \1, \1', text, flags=re.IGNORECASE)
    elif language == 'de':
        # Add commas between repeated "ja" or "nein" for emphasis
        text = re.sub(r'\b(ja)\s+\1\b', r'\1, \1', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(nein)\s+\1\b', r'\1, \1', text, flags=re.IGNORECASE)
        # Handle more than 2 repetitions
        text = re.sub(r'\b(ja)\s+\1\s+\1\b', r'\1, \1, \1', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(nein)\s+\1\s+\1\b', r'\1, \1, \1', text, flags=re.IGNORECASE)
    
    # Step 2: Add punctuation after sentence endings
    for pattern in lang_patterns['sentence_endings']:
        text = re.sub(f'({pattern})(?!\s*[.!?])', r'\1.', text, flags=re.IGNORECASE)
    
    # Step 3: Add question marks after question patterns
    for pattern in lang_patterns['question_words']:
        # Look for question patterns followed by text without punctuation
        text = re.sub(f'({pattern})\s+([^.!?]+?)(?=\s|$)', r'\1 \2?', text, flags=re.IGNORECASE)
    
    # Step 4: Smart sentence splitting based on length and content
    sentences = re.split(r'[.!?]+', text)
    processed_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Add period to sentences that don't end with punctuation
        if sentence and not sentence.endswith(('.', '!', '?')):
            # Don't add period if it ends with a conjunction
            if not sentence.lower().endswith(('and', 'or', 'but', 'so', 'because', 'if', 'when', 'while', 'since', 'although')):
                sentence += '.'
        
        processed_sentences.append(sentence)
    
    # Rejoin sentences
    result = ' '.join(processed_sentences)
    
    # Clean up
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\s+\.', '.', result)
    
    return result.strip()


def basic_punctuation_restoration(text, language='en'):
    """
    Basic punctuation restoration using regex patterns for multiple languages.
    
    Args:
        text (str): The transcribed text without proper punctuation
        language (str): Language code ('en', 'es', 'de', 'fr')
    
    Returns:
        str: Text with restored punctuation
    """
    # Language-specific sentence endings
    sentence_endings = {
        'en': [
            r'\b(thank you|thanks|goodbye|bye|see you|talk to you later)\b',
            r'\b(okay|ok|alright|all right)\b',
            r'\b(yes|no|yeah|nope|yep|nah)\b',
            r'\b(please|excuse me|sorry|pardon)\b'
        ],
        'es': [
            r'\b(gracias|adiós|hasta luego|nos vemos|chao)\b',
            r'\b(vale|ok|bien|está bien)\b',
            r'\b(sí|no|claro|por supuesto)\b',
            r'\b(por favor|perdón|disculpa|lo siento)\b'
        ],
        'de': [
            r'\b(danke|tschüss|auf wiedersehen|bis später)\b',
            r'\b(okay|ok|gut|in ordnung)\b',
            r'\b(ja|nein|klar|natürlich)\b',
            r'\b(bitte|entschuldigung|sorry)\b'
        ],
        'fr': [
            r'\b(merci|au revoir|à bientôt|salut)\b',
            r'\b(okay|ok|d\'accord|ça va)\b',
            r'\b(oui|non|bien sûr|évidemment)\b',
            r'\b(s\'il vous plaît|pardon|désolé)\b'
        ]
    }
    
    # Get sentence endings for the specified language
    endings = sentence_endings.get(language, sentence_endings['en'])
    
    # Add periods after common sentence endings
    for pattern in endings:
        text = re.sub(f'({pattern})(?!\s*[.!?])', r'\1.', text, flags=re.IGNORECASE)
    
    # Add periods after long phrases that don't end with punctuation
    if len(text) > 50 and not text.endswith(('.', '!', '?')):
        text += '.'
    
    return text.strip()


# For testing the module directly
if __name__ == "__main__":
    # Test cases
    test_cases = [
        ("hello how are you today I hope you are doing well thank you", "en"),
        ("hola como estas hoy espero que estes bien gracias", "es"),
        ("hallo wie geht es dir heute ich hoffe es geht dir gut danke", "de"),
        ("bonjour comment allez vous aujourd'hui j'espere que vous allez bien merci", "fr")
    ]
    
    print("Testing punctuation restoration module...")
    for text, lang in test_cases:
        result = restore_punctuation(text, lang)
        print(f"\n{lang.upper()}:")
        print(f"Input:  {text}")
        print(f"Output: {result}") 