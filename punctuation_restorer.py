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
import numpy as np
import warnings

# Suppress PyTorch FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Try to import sentence transformers for better punctuation restoration
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: SentenceTransformers not available. Advanced punctuation restoration may be limited.")


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
    
    # Use advanced punctuation restoration
    try:
        return advanced_punctuation_restoration(text, language, True)  # Enable custom patterns by default
    except Exception as e:
        print(f"Warning: Advanced punctuation restoration failed: {e}")
        print("Returning original text without punctuation restoration.")
        return text


def advanced_punctuation_restoration(text, language='en', use_custom_patterns=True):
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
        return transformer_based_restoration(text, language, use_custom_patterns)
    else:
        # Simple fallback: just clean up whitespace and add basic punctuation
        text = re.sub(r'\s+', ' ', text.strip())
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        return text


def transformer_based_restoration(text, language='en', use_custom_patterns=True):
    """
    Improved punctuation restoration using SentenceTransformers for semantic understanding.
    
    Args:
        text (str): The transcribed text without proper punctuation
        language (str): Language code ('en', 'es', 'de', 'fr')
        use_custom_patterns (bool): Whether to use custom patterns
    
    Returns:
        str: Text with restored punctuation
    """
    # Initialize the model (use multilingual model for better language support)
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # Split text into words
    words = text.split()
    if len(words) < 3:
        return text
    
    # Create potential sentence boundaries using semantic coherence
    sentences = []
    current_chunk = []
    
    for i, word in enumerate(words):
        current_chunk.append(word)
        
        # Check if we should end the sentence here
        if should_end_sentence_here(words, i, current_chunk, model, language):
            sentence_text = ' '.join(current_chunk)
            if sentence_text.strip():
                sentences.append(sentence_text)
            current_chunk = []
    
    # Add remaining words as the last sentence
    if current_chunk:
        sentence_text = ' '.join(current_chunk)
        if sentence_text.strip():
            sentences.append(sentence_text)
    
    # Process each sentence with appropriate punctuation
    processed_sentences = []
    
    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
            
        # Apply punctuation based on semantic analysis
        punctuated_sentence = apply_semantic_punctuation(sentence, model, language, i, len(sentences))
        processed_sentences.append(punctuated_sentence)
    
    result = ' '.join(processed_sentences)
    
    # Final cleanup
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\s+\.', '.', result)
    result = re.sub(r'\s+\?', '?', result)
    result = re.sub(r'\s+\!', '!', result)
    
    return result.strip()


def should_end_sentence_here(words, current_index, current_chunk, model, language):
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
    if len(current_chunk) < 5:  # Increased minimum length
        return False
    
    # Check for natural sentence endings
    current_word = words[current_index]
    next_word = words[current_index + 1] if current_index + 1 < len(words) else ""
    
    # Strong indicators for sentence end
    strong_end_indicators = get_strong_end_indicators(language)
    if any(indicator in current_word.lower() for indicator in strong_end_indicators):
        return True
    
    # Check for capital letter after reasonable length (but be more conservative)
    if (next_word and next_word[0].isupper() and 
        len(current_chunk) >= 12 and  # Increased minimum length
        not is_continuation_word(current_word, language) and
        not is_transitional_word(current_word, language)):
        return True
    
    # Use semantic coherence to determine if chunks should be separated
    if len(current_chunk) >= 20:  # Only check for longer chunks
        return check_semantic_break(words, current_index, model)
    
    return False


def get_strong_end_indicators(language):
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


def is_transitional_word(word, language):
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
        'es': ['entonces', 'después', 'antes', 'mientras', 'durante', 'desde', 'hasta', 'cuando', 'donde', 'si', 'aunque', 'a', 'pesar', 'de'],
        'de': ['dann', 'nächste', 'nach', 'vor', 'während', 'seit', 'bis', 'wenn', 'wo', 'falls', 'obwohl', 'trotz'],
        'fr': ['alors', 'après', 'avant', 'pendant', 'depuis', 'jusqu\'à', 'quand', 'où', 'si', 'bien', 'que', 'malgré']
    }
    
    words = transitional_words.get(language, transitional_words['en'])
    return word.lower() in words


def is_continuation_word(word, language):
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


def check_semantic_break(words, current_index, model):
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


def apply_semantic_punctuation(sentence, model, language, sentence_index, total_sentences):
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
        if not is_continuation_word(sentence.split()[-1] if sentence.split() else '', language):
            sentence += '.'
    
    return sentence


def is_question_semantic(sentence, model, language):
    """
    Determine if a sentence is a question using semantic similarity.
    
    Args:
        sentence (str): The sentence to analyze
        model: SentenceTransformer model
        language (str): Language code
    
    Returns:
        bool: True if sentence is a question
    """
    # First check for obvious question indicators
    if has_question_indicators(sentence, language):
        return True
    
    # For Spanish, be extra careful with introductions and statements
    if language == 'es':
        sentence_lower = sentence.lower()
        
        # Check for strong question indicators first
        strong_question_words = ['qué', 'dónde', 'cuándo', 'cómo', 'quién', 'cuál', 'por qué']
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
    
    question_patterns = get_question_patterns(language)
    if not question_patterns:
        return False
    
    try:
        # Encode sentence and question patterns
        texts_to_encode = [sentence] + question_patterns
        embeddings = model.encode(texts_to_encode)
        
        # Calculate similarities with question patterns
        sentence_embedding = embeddings[0]
        question_embeddings = embeddings[1:]
        
        similarities = []
        for q_emb in question_embeddings:
            similarity = cosine_similarity([sentence_embedding], [q_emb])[0][0]
            similarities.append(similarity)
        
        # Lower threshold for better question detection, but be more conservative for Spanish
        max_similarity = max(similarities)
        if language == 'es':
            # Be more conservative for Spanish to avoid false positives
            return max_similarity > 0.75
        else:
            return max_similarity > 0.6
        
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
    
    # Question words
    question_words = {
        'en': ['what', 'where', 'when', 'why', 'how', 'who', 'which', 'whose', 'whom'],
        'es': ['qué', 'dónde', 'cuándo', 'por', 'qué', 'cómo', 'quién', 'cuál', 'cuáles', 'de', 'quién'],
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
        spanish_question_starters = ['qué', 'dónde', 'cuándo', 'cómo', 'quién', 'cuál', 'cuáles', 'por qué']
        for starter in spanish_question_starters:
            if sentence_lower.startswith(starter + ' '):
                return True
    
    # Check for question words anywhere in the sentence (for embedded questions)
    # But be more conservative to avoid false positives
    for word in words:
        if ' ' + word + ' ' in sentence_lower:
            # Only consider it a question if it's not a common greeting or statement
            if language == 'es':
                # Avoid false positives for common greetings and statements
                if any(greeting in sentence_lower for greeting in ['hola', 'buenos días', 'buenas tardes', 'buenas noches']):
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
            has_strong_question = any(word in sentence_lower for word in ['qué', 'dónde', 'cuándo', 'cómo', 'quién', 'cuál'])
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
            has_strong_question = any(word in sentence_lower for word in ['qué', 'dónde', 'cuándo', 'cómo', 'quién', 'cuál'])
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
            strong_question_words = ['qué', 'dónde', 'cuándo', 'cómo', 'quién', 'cuál', 'por qué']
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


def is_exclamation_semantic(sentence, model, language):
    """
    Determine if a sentence is an exclamation using semantic similarity.
    
    Args:
        sentence (str): The sentence to analyze
        model: SentenceTransformer model
        language (str): Language code
    
    Returns:
        bool: True if sentence is an exclamation
    """
    exclamation_patterns = get_exclamation_patterns(language)
    if not exclamation_patterns:
        return False
    
    try:
        # Encode sentence and exclamation patterns
        texts_to_encode = [sentence] + exclamation_patterns
        embeddings = model.encode(texts_to_encode)
        
        # Calculate similarities with exclamation patterns
        sentence_embedding = embeddings[0]
        exclamation_embeddings = embeddings[1:]
        
        similarities = []
        for e_emb in exclamation_embeddings:
            similarity = cosine_similarity([sentence_embedding], [e_emb])[0][0]
            similarities.append(similarity)
        
        return max(similarities) > 0.7
        
    except Exception:
        return False


def get_exclamation_patterns(language):
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


def apply_basic_punctuation_rules(sentence, language, use_custom_patterns):
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


def get_question_patterns(language):
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