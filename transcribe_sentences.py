import nltk
import sys
import os
import glob
import time
import re

#import whisper
from pydub import AudioSegment
from faster_whisper import WhisperModel

# Try to import sentence transformers for better punctuation restoration
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: SentenceTransformers not available. Using basic punctuation restoration.")

# Ensure NLTK punkt is available
#nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

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
    
    # Step 1: Add punctuation after sentence endings
    for pattern in lang_patterns['sentence_endings']:
        text = re.sub(f'({pattern})(?!\s*[.!?])', r'\1.', text, flags=re.IGNORECASE)
    
    # Step 2: Add question marks after question patterns
    for pattern in lang_patterns['question_words']:
        # Look for question patterns followed by text without punctuation
        text = re.sub(f'({pattern})\s+([^.!?]+?)(?=\s|$)', r'\1 \2?', text, flags=re.IGNORECASE)
    
    # Step 3: Smart sentence splitting based on length and content
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

def split_audio(audio_file, chunk_length_sec=300):
    audio = AudioSegment.from_file(audio_file)
    chunks = []
    for i in range(0, len(audio), chunk_length_sec * 1000):
        chunk = audio[i:i + chunk_length_sec * 1000]
        chunk_path = f"{audio_file}_chunk_{i // (chunk_length_sec * 1000)}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks

def write_txt(sentences, output_file):
    try:
        with open(output_file, "w") as f:
            for sentence in sentences:
                f.write(f"{sentence.strip()}\n\n")
    except Exception as e:
        print(f"Error writing TXT file: {e}")
        sys.exit(1)

def write_srt(segments, output_file):
    def format_timestamp(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    try:
        with open(output_file, "w") as f:
            for i, seg in enumerate(segments, 1):
                start = format_timestamp(seg['start'])
                end = format_timestamp(seg['end'])
                text = seg['text'].strip()
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    except Exception as e:
        print(f"Error writing SRT file: {e}")
        sys.exit(1)

def transcribe_with_sentences(audio_file, output_dir, language, model_size, output_format):
    os.environ["OMP_NUM_THREADS"] = "8"
    # Load faster-whisper model
    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")  # or "cuda" for GPU
        #model = WhisperModel("turbo", device="cpu", compute_type="int8")  # or "cuda" for GPU
    except Exception as e:
        print(f"Error loading faster-whisper model: {e}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    print("Splitting audio into chunks...")
    chunk_files = split_audio(audio_file, chunk_length_sec=300)  # 5 minute chunk size

    all_text = ""
    all_segments = []
    offset = 0.0

    print("Transcribing chunks...")
    for idx, chunk_file in enumerate(chunk_files, 1):
        print(f"Transcribing chunk {idx}/{len(chunk_files)}: {chunk_file}")
        try:
            segments, info = model.transcribe(chunk_file, language=language, beam_size=3)
            text = ""
            chunk_segments = []
            for seg in segments:
                seg_dict = {
                    "start": seg.start + offset,
                    "end": seg.end + offset,
                    "text": seg.text
                }
                chunk_segments.append(seg_dict)
                text += seg.text + " "
            all_segments.extend(chunk_segments)
            all_text += text.strip() + "\n"
            offset += AudioSegment.from_file(chunk_file).duration_seconds
        except Exception as e:
            print(f"Error during transcription: {e}")
            sys.exit(1)
        finally:
            if os.path.exists(chunk_file):
                os.remove(chunk_file)

    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    if output_format == "srt":
        output_file = os.path.join(output_dir, base_name + ".srt")
        write_srt(all_segments, output_file)
    else:
        # Restore punctuation before sentence tokenization
        print("Restoring punctuation...")
        punctuated_text = restore_punctuation(all_text, language)
        
        lang_map = {
            'en': 'english',
            'es': 'spanish',
            'de': 'german',
            'fr': 'french'
        }
        nltk_lang = lang_map.get(language, 'english')
        sentences = nltk.sent_tokenize(punctuated_text, language=nltk_lang)
        output_file = os.path.join(output_dir, base_name + ".txt")
        write_txt(sentences, output_file)

def cleanup_chunks(audio_file):
    audio_dir = os.path.dirname(os.path.abspath(audio_file))
    pattern = os.path.join(audio_dir, "*_chunk_*.wav")
    for chunk_path in glob.glob(pattern):
        try:
            os.remove(chunk_path)
            print(f"Removed leftover chunk file: {chunk_path}")
        except Exception as e:
            print(f"Error removing chunk file {chunk_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 6:
        print("Usage: python transcribe_sentences.py <audio_file> <output_dir> [language (default 'en')] [model_size (default 'medium')] [output_format (txt|srt, default 'txt')]")
        sys.exit(1)
    audio_file = sys.argv[1]
    output_dir = sys.argv[2]
    language = sys.argv[3] if len(sys.argv) >= 4 else "en"
    model_size = sys.argv[4] if len(sys.argv) >= 5 else "medium"
    output_format = sys.argv[5] if len(sys.argv) == 6 else "txt"
    if output_format not in ("txt", "srt"):
        print(f"Warning: Output format '{output_format}' not recognized. Defaulting to 'txt'.")
        output_format = "txt"    
    
    start_time = time.time()

    cleanup_chunks(audio_file)  # Cleanup any existing chunks before starting

    transcribe_with_sentences(audio_file, output_dir, language, model_size, output_format)

    end_time = time.time()
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"Script completed in {minutes} minutes and {seconds} seconds.")