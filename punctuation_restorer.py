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
    Punctuation restoration using SentenceTransformers for semantic understanding.
    
    Args:
        text (str): The transcribed text without proper punctuation
        language (str): Language code ('en', 'es', 'de', 'fr')
        use_custom_patterns (bool): Whether to use custom patterns
    
    Returns:
        str: Text with restored punctuation
    """
    # Initialize the model (use multilingual model for better language support)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Split text into potential sentence chunks
    words = text.split()
    if len(words) < 3:
        return text
    
    # Create potential sentence boundaries based on semantic breaks
    potential_sentences = []
    current_sentence = []
    
    for i, word in enumerate(words):
        current_sentence.append(word)
        
        # Consider sentence boundary at natural breaks
        if (i < len(words) - 1 and 
            (len(current_sentence) > 15 or  # Long sentence break
             (i < len(words) - 1 and words[i+1][0].isupper() and len(current_sentence) > 5))):  # Capital letter after reasonable length
            
            potential_sentences.append(' '.join(current_sentence))
            current_sentence = []
    
    # Add remaining words
    if current_sentence:
        potential_sentences.append(' '.join(current_sentence))
    
    # Process each sentence chunk using semantic understanding
    processed_sentences = []
    
    for i, sentence in enumerate(potential_sentences):
        if not sentence.strip():
            continue
            
        # Apply basic punctuation rules first
        sentence = apply_basic_punctuation_rules(sentence, language, use_custom_patterns)
        
        # Use semantic similarity to determine if this should be a question
        question_patterns = get_question_patterns(language)
        if question_patterns:
            similarities = []
            for pattern in question_patterns:
                try:
                    # Encode sentences and calculate similarity
                    embeddings = model.encode([sentence, pattern])
                    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                    similarities.append(similarity)
                except:
                    continue
            
            # If high similarity to question patterns, add question mark
            if similarities and max(similarities) > 0.6:
                if not sentence.endswith('?'):
                    sentence = sentence.rstrip('.!') + '?'
        
        processed_sentences.append(sentence)
    
    result = ' '.join(processed_sentences)
    
    # Final cleanup
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\s+\.', '.', result)
    
    return result.strip()


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
            "¿Eres listo?"
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