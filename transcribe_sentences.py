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
Transcribe audio files into sentences and save as TXT or SRT files.
Primary language focus: English (en), Spanish (es), French (fr), German (de).
Other languages are considered experimental.
"""

import re
import sys
import os
import glob
import time
import re

#import whisper
from pydub import AudioSegment
from faster_whisper import WhisperModel

# Import punctuation restoration module
from punctuation_restorer import restore_punctuation

# Primary language focus
FOCUS_LANGS = {"en", "es", "fr", "de"}

# No longer need NLTK - using simple sentence splitting

def get_supported_languages():
    """Return a dictionary of commonly used language codes.

    Primary: en, es, fr, de
    Others: experimental (may have reduced accuracy)
    """
    return {
        'en': 'English',
        'es': 'Spanish', 
        'fr': 'French',
        'de': 'German',
        'ja': 'Japanese',
        'ru': 'Russian',
        'cs': 'Czech',
        'it': 'Italian',
        'pt': 'Portuguese',
        'nl': 'Dutch',
        'pl': 'Polish',
        'tr': 'Turkish',
        'ar': 'Arabic',
        'zh': 'Chinese',
        'ko': 'Korean',
        'hi': 'Hindi',
        'sv': 'Swedish',
        'da': 'Danish',
        'no': 'Norwegian',
        'fi': 'Finnish'
    }

def validate_language_code(language_code):
    """Validate and provide helpful information about language codes."""
    if language_code is None:
        return None
    
    supported = get_supported_languages()
    if language_code in supported:
        return language_code
    else:
        print(f"Warning: Language code '{language_code}' not in common list.")
        print("Primary language codes:")
        for code in ["en","es","fr","de"]:
            if code in supported:
                print(f"  {code}: {supported[code]}")
        print("Experimental language codes:")
        for code, name in supported.items():
            if code not in FOCUS_LANGS:
                print(f"  {code}: {name} (experimental)")
        print("Whisper supports many more languages. The code will still work if it's valid.")
        return language_code

def display_transcription_info(media_file, model_name, language, beam_size, compute_type, output_format):
    """
    Display transcription parameters and settings.
    """
    print("\n" + "="*60)
    print("TRANSCRIPTION PARAMETERS")
    print("="*60)
    print(f"File name:        {os.path.basename(media_file)}")
    print(f"Model:            {model_name}")
    print(f"Language:         {'Auto-detect' if language is None else language}")
    print(f"Beam size:        {beam_size}")
    print(f"Compute type:     {compute_type}")
    print(f"Output format:    {output_format.upper()}")
    print("="*60 + "\n")

def split_audio(media_file, chunk_length_sec=480):
    """
    Split audio/video file into chunks of a specified length.
    """
    audio = AudioSegment.from_file(media_file)
    chunks = []
    base_name = os.path.splitext(os.path.basename(media_file))[0]
    for i in range(0, len(audio), chunk_length_sec * 1000):
        chunk = audio[i:i + chunk_length_sec * 1000]
        chunk_path = f"{base_name}_chunk_{i // (chunk_length_sec * 1000)}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks

def write_txt(sentences, output_file):
    """
    Write sentences to a TXT file.
    """
    try:
        with open(output_file, "w") as f:
            for sentence in sentences:
                f.write(f"{sentence.strip()}\n\n")
    except Exception as e:
        print(f"Error writing TXT file: {e}")
        sys.exit(1)

def write_srt(segments, output_file):
    """
    Write SRT file with timestamps.
    """
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

def transcribe_with_sentences(media_file, output_dir, language, output_format):
    """
    Transcribe audio/video file into sentences and save as TXT or SRT file.
    """
    os.environ["OMP_NUM_THREADS"] = "8"
    
    # Model configuration
    model_name = "medium"
    beam_size = 3
    compute_type = "int8"
    device = "cpu"
    
    # Display transcription information
    display_transcription_info(media_file, model_name, language, beam_size, compute_type, output_format)
    
    # Load faster-whisper model
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)  # or "cuda" for GPU
        #model = WhisperModel("turbo", device="cpu", compute_type="int8")  # or "cuda" for GPU
    except Exception as e:
        print(f"Error loading faster-whisper model: {e}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    print("Splitting media into chunks...")
    chunk_files = split_audio(media_file, chunk_length_sec=480)  # 8 minute chunk size

    all_text = ""
    all_segments = []
    offset = 0.0

    print("Transcribing chunks...")
    detected_language = None
    for idx, chunk_file in enumerate(chunk_files, 1):
        print(f"Transcribing chunk {idx}/{len(chunk_files)}: {chunk_file}")
        try:
            segments, info = model.transcribe(chunk_file, language=language, beam_size=beam_size)
            # Store detected language from first chunk
            if idx == 1 and language is None:
                detected_language = info.language
                print(f"Auto-detected language: {detected_language} (confidence: {info.language_probability:.2f})")
            
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

    base_name = os.path.splitext(os.path.basename(media_file))[0]
    if output_format == "srt":
        output_file = os.path.join(output_dir, base_name + ".srt")
        write_srt(all_segments, output_file)
    else:
        # Restore punctuation before sentence tokenization
        print("Restoring punctuation...")
        # Use detected language if auto-detection was used, otherwise use provided language
        lang_for_punctuation = detected_language if language is None else language
        
        # First, split the text into individual segments (as they come from Whisper)
        # Split by double newlines (which separate sentences in the transcription)
        text_segments = [seg.strip() for seg in all_text.split('\n\n') if seg.strip()]
        
        sentences = []
        for segment in text_segments:
            # Process each segment individually for better punctuation
            processed_segment = restore_punctuation(segment, lang_for_punctuation)
            
            # Split the processed segment into sentences while preserving punctuation
            # Split by sentence-ending punctuation but capture the punctuation
            parts = re.split(r'([.!?]+)', processed_segment)
            
            for i in range(0, len(parts), 2):
                if i < len(parts):
                    sentence_text = parts[i].strip()
                    punctuation = parts[i + 1] if i + 1 < len(parts) else ""
                    
                    if sentence_text:
                        # Combine sentence text with its punctuation
                        full_sentence = sentence_text + punctuation
                        
                        # Remove leading punctuation and whitespace
                        cleaned = re.sub(r'^[",\s]+', '', full_sentence)
                        
                        # Capitalize first letter if it's a letter
                        if cleaned and cleaned[0].isalpha():
                            cleaned = cleaned[0].upper() + cleaned[1:]
                        
                        if cleaned:
                            # Ensure the sentence ends with punctuation
                            if not cleaned.endswith(('.', '!', '?')):
                                cleaned += '.'
                            sentences.append(cleaned)
        output_file = os.path.join(output_dir, base_name + ".txt")
        write_txt(sentences, output_file)

def cleanup_chunks(media_file):
    """
    Clean up any leftover chunk files.
    """
    media_dir = os.path.dirname(os.path.abspath(media_file))
    pattern = os.path.join(media_dir, "*_chunk_*.wav")
    for chunk_path in glob.glob(pattern):
        try:
            os.remove(chunk_path)
            print(f"Removed leftover chunk file: {chunk_path}")
        except Exception as e:
            print(f"Error removing chunk file {chunk_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print("Usage: python transcribe_sentences.py <media_file> <output_dir> [language (default auto-detect)] [output_format (txt|srt, default 'txt')]")
        print("\nSupported language codes:")
        supported = get_supported_languages()
        print("Primary:")
        for code in ["en","es","fr","de"]:
            if code in supported:
                print(f"  {code}: {supported[code]}")
        print("Experimental:")
        for code, name in supported.items():
            if code not in FOCUS_LANGS:
                print(f"  {code}: {name} (experimental)")
        print("\nNote: Whisper supports many more languages. Use 'auto' or omit language parameter for auto-detection.")
        sys.exit(1)
    media_file = sys.argv[1]
    output_dir = sys.argv[2]
    language = sys.argv[3] if len(sys.argv) >= 4 else None  # Default to None for auto-detection
    output_format = sys.argv[4] if len(sys.argv) == 5 else "txt"
    if output_format not in ("txt", "srt"):
        print(f"Warning: Output format '{output_format}' not recognized. Defaulting to 'txt'.")
        output_format = "txt"    
    
    # Validate language code if provided
    if language:
        language = validate_language_code(language)
        if language == "auto":
            language = None  # Convert "auto" to None for auto-detection
    
    start_time = time.time()

    cleanup_chunks(media_file)  # Cleanup any existing chunks before starting

    transcribe_with_sentences(media_file, output_dir, language, output_format)

    end_time = time.time()
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"Script completed in {minutes} minutes and {seconds} seconds.")