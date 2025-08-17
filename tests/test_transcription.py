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
Test script for transcribing audio and video files using Whisper.
This is for testing and experimentation purposes only.

Primary languages: English (en), Spanish (es), French (fr), German (de)
Other languages are considered experimental (auto-detection supported)
"""

import os
import sys
import time
import argparse
from pathlib import Path
from pydub import AudioSegment
from faster_whisper import WhisperModel
from punctuation_restorer import restore_punctuation
from podscripter import _normalize_srt_cues

def get_supported_languages():
    """Return a dictionary of commonly used language codes.

    Primary: en, es, fr, de; others experimental
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
        print("Primary languages:")
        for code in ["en","es","fr","de"]:
            if code in supported:
                print(f"  {code}: {supported[code]}")
        print("Experimental languages:")
        for code, name in supported.items():
            if code not in {"en","es","fr","de"}:
                print(f"  {code}: {name} (experimental)")
        print("Whisper supports many more languages. The code will still work if it's valid.")
        return language_code

def test_transcribe_file(media_file, model_size="medium", language=None, 
                        chunk_length_sec=None, device="cpu", compute_type="auto",
                        beam_size=3, output_format="txt", translate_to_english=False,
                        apply_restoration=False,
                        vad_filter=True, vad_speech_pad_ms=200,
                        dump_raw=False):
    """
    Test transcription with various Whisper settings.
    
    Args:
        media_file: Path to audio/video file
        model_size: Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)
        language: Language code (en, es, fr, de, etc.) or None for auto-detection
        chunk_length_sec: Chunk length in seconds (None for no chunking)
        device: Device to use (cpu, cuda)
        compute_type: Compute type (int8, float16, float32)
        beam_size: Beam size for decoding
        output_format: Output format (txt, srt, raw)
        translate_to_english: Whether to translate the output to English
    """
    
    print(f"Testing transcription with settings:")
    print(f"  File: {media_file}")
    print(f"  Model: {model_size}")
    print(f"  Language: {language or 'auto-detect'}")
    print(f"  Device: {device}")
    print(f"  Compute type: {compute_type}")
    print(f"  Beam size: {beam_size}")
    print(f"  Chunking: {chunk_length_sec} seconds" if chunk_length_sec else "  Chunking: disabled")
    print(f"  Output format: {output_format}")
    print(f"  VAD: {'on' if vad_filter else 'off'} (pad {vad_speech_pad_ms} ms)")
    print(f"  Translate to English: {translate_to_english}")
    print("-" * 50)
    
    # Load model
    start_time = time.time()
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Process file
    if chunk_length_sec:
        # Use chunking
        print("Processing with chunking...")
        result = transcribe_with_chunking(
            model, media_file, chunk_length_sec, language, beam_size, translate_to_english,
            vad_filter=vad_filter, vad_speech_pad_ms=vad_speech_pad_ms,
        )
    else:
        # Process entire file
        print("Processing entire file...")
        result = transcribe_full_file(
            model, media_file, language, beam_size, translate_to_english,
            vad_filter=vad_filter, vad_speech_pad_ms=vad_speech_pad_ms,
        )
    
    # Output results
    if output_format == "txt":
        output_txt(result, media_file, language=language, apply_restoration=apply_restoration)
    elif output_format == "srt":
        output_srt(result, media_file)
    elif output_format == "raw":
        output_raw(result, media_file)
    
    # Optionally also write raw output for debugging, regardless of main output format
    if dump_raw and output_format != "raw":
        output_raw(result, media_file)
    
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")

def transcribe_full_file(model, media_file, language, beam_size, translate_to_english=False,
                         vad_filter=True, vad_speech_pad_ms=200):
    """Transcribe entire file without chunking."""
    task = "translate" if translate_to_english else "transcribe"
    kwargs = {"language": language, "beam_size": beam_size, "task": task, "vad_filter": vad_filter}
    if vad_filter:
        kwargs["vad_parameters"] = {"speech_pad_ms": int(max(0, vad_speech_pad_ms))}
    segments, info = model.transcribe(media_file, **kwargs)
    
    all_segments = []
    all_text = ""
    
    for segment in segments:
        all_segments.append({
            'start': segment.start,
            'end': segment.end,
            'text': segment.text.strip()
        })
        all_text += segment.text + " "
    
    return {
        'segments': all_segments,
        'text': all_text.strip(),
        'language': info.language,
        'language_probability': info.language_probability,
        'task': task
    }

def transcribe_with_chunking(model, media_file, chunk_length_sec, language, beam_size, translate_to_english=False,
                             vad_filter=True, vad_speech_pad_ms=200):
    """Transcribe file with chunking."""
    task = "translate" if translate_to_english else "transcribe"
    audio = AudioSegment.from_file(media_file)
    chunks = []
    
    # Create chunks
    for i in range(0, len(audio), chunk_length_sec * 1000):
        chunk = audio[i:i + chunk_length_sec * 1000]
        chunk_path = f"temp_chunk_{i // (chunk_length_sec * 1000)}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    
    all_segments = []
    all_text = ""
    offset = 0.0
    
    # Process each chunk
    for i, chunk_file in enumerate(chunks):
        print(f"  Processing chunk {i+1}/{len(chunks)}")
        kwargs = {"language": language, "beam_size": beam_size, "task": task, "vad_filter": vad_filter}
        if vad_filter:
            kwargs["vad_parameters"] = {"speech_pad_ms": int(max(0, vad_speech_pad_ms))}
        segments, info = model.transcribe(chunk_file, **kwargs)
        
        for segment in segments:
            all_segments.append({
                'start': segment.start + offset,
                'end': segment.end + offset,
                'text': segment.text.strip()
            })
            all_text += segment.text + " "
        
        offset += AudioSegment.from_file(chunk_file).duration_seconds
        os.remove(chunk_file)  # Clean up
    
    return {
        'segments': all_segments,
        'text': all_text.strip(),
        'language': info.language,
        'language_probability': info.language_probability,
        'task': task
    }

def output_txt(result, media_file, language=None, apply_restoration=False):
    """Output as TXT file."""
    # Create output path in audio-files folder
    output_dir = "audio-files"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{Path(media_file).stem}_test.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        task_info = f"Task: {result['task']}" if 'task' in result else ""
        f.write(f"Language: {result['language']} (confidence: {result['language_probability']:.2f})\n")
        if task_info:
            f.write(f"{task_info}\n")
        f.write("=" * 50 + "\n\n")
        text = result['text']
        if apply_restoration and language:
            try:
                text = restore_punctuation(text, language)
            except Exception as e:
                f.write(f"[Warning] punctuation restoration failed: {e}\n\n")
        f.write(text)
    print(f"Output saved to: {output_file}")

def output_srt(result, media_file):
    """Output as SRT file."""
    # Create output path in audio-files folder
    output_dir = "audio-files"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{Path(media_file).stem}_test.srt")
    # Normalize cues to avoid lingering during silence
    original_ends = [seg['end'] for seg in result['segments']]
    segments = _normalize_srt_cues(
        list(result['segments']),
        max_duration=3.0,
        min_gap=0.2,
        min_duration=1.0,
        chars_per_second=17.0,
    )
    trimmed_count = 0
    total_trim = 0.0
    max_trim = 0.0
    for before, after in zip(original_ends, segments):
        try:
            delta = float(before) - float(after['end'])
        except Exception:
            delta = 0.0
        if delta > 0.0005:
            trimmed_count += 1
            total_trim += max(0.0, delta)
            max_trim = max(max_trim, delta)
    
    def format_timestamp(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, 1):
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            f.write(f"{i}\n{start} --> {end}\n{segment['text']}\n\n")
    
    print(f"Output saved to: {output_file}")
    print(
        f"SRT normalization: trimmed {trimmed_count}/{len(segments)} cues "
        f"(max trim {max_trim:.1f}s, total {total_trim:.1f}s)"
    )

def output_raw(result, media_file):
    """Output raw transcription data."""
    # Create output path in audio-files folder
    output_dir = "audio-files"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{Path(media_file).stem}_test_raw.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Language: {result['language']}\n")
        f.write(f"Language Probability: {result['language_probability']}\n")
        if 'task' in result:
            f.write(f"Task: {result['task']}\n")
        f.write(f"Number of segments: {len(result['segments'])}\n")
        f.write("=" * 50 + "\n\n")
        
        for i, segment in enumerate(result['segments']):
            f.write(f"Segment {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s\n")
            f.write(f"Text: {segment['text']}\n\n")
    
    print(f"Output saved to: {output_file}")

def main():
    """CLI entry point for ad-hoc transcription testing.

    Usage:
      python tests/test_transcription.py <media_file> [options]

    Example (write TXT plus a raw Whisper dump for debugging):
      python tests/test_transcription.py \
        audio-files/example.mp3 \
        --output-format txt \
        --dump-raw

    This script loads a Whisper model and transcribes either the full file
    (with --single) or in user-defined chunks (with --chunk-length). It can
    optionally apply punctuation restoration to the concatenated TXT output
    (--apply-restoration). Use --dump-raw to also emit a raw dump capturing the
    model’s detected language, confidence, and per-segment timings.
    """
    parser = argparse.ArgumentParser(description="Test Whisper transcription with various settings")
    parser.add_argument("media_file", help="Path to audio or video file")
    parser.add_argument("--list-languages", action="store_true", 
                       help="List supported language codes and exit")
    parser.add_argument("--model", default="medium", 
                       choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                       help="Whisper model size (default: medium)")
    parser.add_argument("--language", help="Language code (en, es, fr, de, ja, ru, cs, etc.) or auto-detect if not specified")
    parser.add_argument("--chunk-length", type=int, help="Chunk length in seconds (disable chunking if not specified)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to use (default: cpu)")
    parser.add_argument("--compute-type", default="auto", choices=["auto", "int8", "float16", "float32"], 
                       help="Compute type (default: auto)")
    parser.add_argument("--beam-size", type=int, default=3, help="Beam size for decoding (default: 3)")
    parser.add_argument("--output-format", default="txt", choices=["txt", "srt", "raw"], 
                       help="Output format (default: txt)")
    parser.add_argument("--translate", action="store_true", 
                       help="Translate the output to English")
    parser.add_argument("--apply-restoration", action="store_true",
                       help="Apply punctuation restoration to the concatenated text before writing TXT output")
    parser.add_argument("--dump-raw", action="store_true",
                        help="Also write raw Whisper output for debugging (in addition to chosen format)")
    parser.add_argument("--single", action="store_true",
                       help="Bypass manual chunking and transcribe the full file in a single call")
    parser.add_argument("--no-vad", dest="vad_filter", action="store_false",
                        help="Disable VAD filtering (may help avoid clipping the first words)")
    parser.add_argument("--vad-speech-pad-ms", type=int, default=200,
                        help="Padding (ms) added around detected speech when VAD is enabled (default: 200)")
    
    args = parser.parse_args()
    
    # Handle list-languages option
    if args.list_languages:
        print("Supported language codes:")
        supported = get_supported_languages()
        print("Primary:")
        for code in ["en","es","fr","de"]:
            if code in supported:
                print(f"  {code}: {supported[code]}")
        print("Experimental:")
        for code, name in supported.items():
            if code not in {"en","es","fr","de"}:
                print(f"  {code}: {name} (experimental)")
        print("\nNote: Whisper supports many more languages. These are the common ones.")
        sys.exit(0)
    
    if not os.path.exists(args.media_file):
        print(f"Error: File '{args.media_file}' not found")
        sys.exit(1)
    
    # Validate language code if provided
    validated_language = validate_language_code(args.language)
    
    # Decide chunking based on --single flag
    eff_chunk_len = None if args.single else args.chunk_length

    test_transcribe_file(
        media_file=args.media_file,
        model_size=args.model,
        language=validated_language,
        chunk_length_sec=eff_chunk_len,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        output_format=args.output_format,
        translate_to_english=args.translate,
        apply_restoration=args.apply_restoration,
        vad_filter=args.vad_filter,
        vad_speech_pad_ms=args.vad_speech_pad_ms,
        dump_raw=args.dump_raw,
    )

if __name__ == "__main__":
    main() 