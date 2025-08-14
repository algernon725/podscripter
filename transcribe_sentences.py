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
import argparse
from pathlib import Path

from pydub import AudioSegment
from faster_whisper import WhisperModel
from tempfile import TemporaryDirectory, NamedTemporaryFile

# Import punctuation restoration module
from punctuation_restorer import restore_punctuation

# Primary language focus
FOCUS_LANGS = {"en", "es", "fr", "de"}

# No longer need NLTK - using simple sentence splitting

# --- Helpers: language-specific pre-split normalization ---
def _collapse_dotted_acronyms_en(text: str) -> str:
    """Collapse dotted uppercase acronyms to avoid false sentence splits.

    Examples:
      - "U. S." → "US", "D. C." → "DC", "U.S." → "US"
      - Also matches when followed by hyphen or whitespace (e.g., "U.S.-Mexico").
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

# Defaults and tuning constants
DEFAULT_CHUNK_SEC = 480 # chunk length in seconds
DEFAULT_OVERLAP_SEC = 3 # overlap between chunks in seconds
DEFAULT_BEAM_SIZE = 3 # beam size to use for the model
DEFAULT_COMPUTE_TYPE = "auto" # compute type to use for the model
DEFAULT_DEVICE = "cpu" # device to use for the model
DEFAULT_MODEL_NAME = "medium" # Whisper model to use for the transcription
DEFAULT_OMP_THREADS = "8" # number of threads to use for the model
DEDUPE_EPSILON_SEC = 0.05 # number of seconds of overlap between chunks to deduplicate
PROMPT_TAIL_CHARS = 200 # number of characters of the previous chunk to use as a prompt for the next chunk
DEFAULT_VAD_FILTER = True # whether to use VAD to filter out non-speech segments
DEFAULT_VAD_SPEECH_PAD_MS = 200 # padding in milliseconds to add around detected speech when VAD is enabled

def get_supported_languages() -> dict[str, str]:
    """Return supported language codes and names.

    Parameters:
      - None

    Returns:
      - dict[str, str]: map of language code to human-readable name

    Notes:
      - Primary: en, es, fr, de
      - Others: experimental (may have reduced accuracy)
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

def validate_language_code(language_code: str | None) -> str | None:
    """Validate a language code and print guidance if unknown.

    Parameters:
      - language_code: str | None — the requested language code (or None for auto)

    Returns:
      - str | None: the same code if recognized, or the original value if unrecognized.
                    None indicates auto-detect.
    """
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

def _display_transcription_info(media_file, model_name, language, beam_size, compute_type, output_format):
    """
    Display transcription parameters and settings.
    """
    print("\n" + "="*60)
    print("TRANSCRIPTION PARAMETERS")
    print("="*60)
    print(f"File name:        {Path(media_file).name}")
    print(f"Model:            {model_name}")
    print(f"Language:         {'Auto-detect' if language is None else language}")
    print(f"Beam size:        {beam_size}")
    print(f"Compute type:     {compute_type}")
    print(f"Output format:    {output_format}")
    print("="*60 + "\n")

def _split_audio_with_overlap(media_file: str, chunk_length_sec: int = DEFAULT_CHUNK_SEC, overlap_sec: int = DEFAULT_OVERLAP_SEC, chunk_dir: Path | None = None):
    """Split audio into overlapping chunks.

    Returns list of dicts: { 'path': str, 'start_sec': float, 'duration_sec': float }
    """
    audio = AudioSegment.from_file(media_file)
    media_path = Path(media_file)
    out_dir = chunk_dir or media_path.parent
    base_name = media_path.stem
    chunk_infos = []
    chunk_ms = chunk_length_sec * 1000
    overlap_ms = max(0, overlap_sec * 1000)
    step_ms = max(1, chunk_ms - overlap_ms)
    idx = 0
    for start_ms in range(0, len(audio), step_ms):
        end_ms = min(start_ms + chunk_ms, len(audio))
        chunk = audio[start_ms:end_ms]
        chunk_path = out_dir / f"{base_name}_chunk_{idx}.wav"
        chunk.export(str(chunk_path), format="wav")
        chunk_infos.append({
            'path': str(chunk_path),
            'start_sec': start_ms / 1000.0,
            'duration_sec': (end_ms - start_ms) / 1000.0,
        })
        idx += 1
        if end_ms >= len(audio):
            break
    return chunk_infos

def _write_txt(sentences, output_file):
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

def _write_srt(segments, output_file):
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

def _validate_paths(media_file: str, output_dir: str) -> tuple[Path, Path]:
    """Validate that input exists and output dir is writable.

    Returns normalized (media_path, output_dir_path) on success, exits on failure.
    """
    media_path = Path(media_file)
    if not media_path.exists() or not media_path.is_file():
        print(f"Error: Input file does not exist or is not a file: {media_file}")
        sys.exit(1)
    if not os.access(media_path, os.R_OK):
        print(f"Error: Input file is not readable: {media_file}")
        sys.exit(1)

    out_dir = Path(output_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error: Cannot create output directory '{output_dir}': {e}")
        sys.exit(1)
    # Writability probe
    try:
        with NamedTemporaryFile(dir=str(out_dir), delete=True) as _:
            pass
    except Exception as e:
        print(f"Error: Output directory is not writable '{output_dir}': {e}")
        sys.exit(1)
    return media_path, out_dir

def _transcribe_file(
    model,
    audio_path: str,
    language,
    beam_size: int,
    prev_prompt: str | None = None,
    *,
    vad_filter: bool = DEFAULT_VAD_FILTER,
    vad_speech_pad_ms: int = DEFAULT_VAD_SPEECH_PAD_MS,
):
    """Run faster-whisper transcription on a single file and return (segments, info).

    prev_prompt: optional initial prompt (used for chunk stitching)
    """
    kwargs = {
        "language": language,
        "beam_size": beam_size,
        "vad_filter": vad_filter,
        "condition_on_previous_text": True,
    }
    if vad_filter:
        kwargs["vad_parameters"] = {"speech_pad_ms": int(max(0, vad_speech_pad_ms))}
    if prev_prompt:
        kwargs["initial_prompt"] = prev_prompt
    segments, info = model.transcribe(audio_path, **kwargs)
    return segments, info

def _dedupe_segments(global_segments, last_end: float, epsilon: float = DEDUPE_EPSILON_SEC):
    """Deduplicate segments by end time.

    global_segments: iterable of (global_start: float, global_end: float, text: str)
    Returns (deduped: list[dict], new_last_end: float)
    """
    out = []
    new_last = last_end
    for g_start, g_end, text in global_segments:
        if g_end <= new_last + epsilon:
            continue
        out.append({"start": g_start, "end": g_end, "text": text})
        new_last = max(new_last, g_end)
    return out, new_last

def _accumulate_segments(model_segments, chunk_start: float, last_end: float, epsilon: float = DEDUPE_EPSILON_SEC):
    """Convert model segments to global segments, dedupe, and concat text.

    Returns (segment_dicts, text_concat, new_last_end)
    """
    global_segs = []
    for seg in model_segments:
        g_start = chunk_start + float(seg.start)
        g_end = chunk_start + float(seg.end)
        global_segs.append((g_start, g_end, seg.text))
    deduped, new_last = _dedupe_segments(global_segs, last_end, epsilon)
    text = " ".join(d["text"] for d in deduped)
    return deduped, text, new_last

def transcribe_with_sentences(
    media_file: str,
    output_dir: str,
    language: str | None,
    output_format: str,
    single_call: bool = False,
    *,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    quiet: bool = False,
    vad_filter: bool = DEFAULT_VAD_FILTER,
    vad_speech_pad_ms: int = DEFAULT_VAD_SPEECH_PAD_MS,
) -> dict:
    """Transcribe an audio/video file and write TXT or SRT output.

    Parameters:
      - media_file: str — path to input media file
      - output_dir: str — directory to write output
      - language: str | None — language code or None for auto-detect
      - output_format: str — 'txt' or 'srt'
      - single_call: bool — if True, run full-file transcription; otherwise chunked mode

    Returns:
      - dict: {
          'segments': list[dict],
          'sentences': list[str],
          'detected_language': str | None,
          'output_path': str,
        }
    """
    os.environ["OMP_NUM_THREADS"] = DEFAULT_OMP_THREADS
    _t0 = time.time()
    media_path, out_dir = _validate_paths(media_file, output_dir)
    
    # Model configuration
    model_name = DEFAULT_MODEL_NAME
    beam_size = DEFAULT_BEAM_SIZE
    device = DEFAULT_DEVICE
    
    # Display transcription information
    if not quiet:
        _display_transcription_info(media_file, model_name, language, beam_size, compute_type, output_format)
    
    # Load faster-whisper model
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)  # or "cuda" for GPU
        #model = WhisperModel("turbo", device="cpu", compute_type="int8")  # or "cuda" for GPU
    except Exception as e:
        print(f"Error loading faster-whisper model: {e}")
        sys.exit(1)

    # Output dir already validated/created by _validate_paths
    detected_language = None
    if single_call:
        if not quiet:
            print("Transcribing full file in a single call (no manual chunking)...")
        all_text = ""
        all_segments = []
        try:
            segments, info = _transcribe_file(
                model,
                media_file,
                language,
                beam_size,
                vad_filter=vad_filter,
                vad_speech_pad_ms=vad_speech_pad_ms,
            )
            if language is None:
                detected_language = info.language
                if not quiet:
                    print(f"Auto-detected language: {detected_language} (confidence: {info.language_probability:.2f})")
            accum, text, _ = _accumulate_segments(segments, 0.0, last_end=0.0)
            all_segments.extend(accum)
            all_text += (text + " ")
        except Exception as e:
            print(f"Error during transcription: {e}")
            sys.exit(1)
    else:
        if not quiet:
            print("Splitting media into chunks with overlap...")
        overlap_sec = DEFAULT_OVERLAP_SEC
        all_text = ""
        all_segments = []
        with TemporaryDirectory() as tmp_dir:
            chunk_infos = _split_audio_with_overlap(
                media_file,
                chunk_length_sec=DEFAULT_CHUNK_SEC,
                overlap_sec=overlap_sec,
                chunk_dir=Path(tmp_dir),
            )
            if not quiet:
                print("Transcribing chunks...")
            last_global_end = 0.0
            prev_prompt = None
            for idx, info in enumerate(chunk_infos, 1):
                chunk_file = info['path']
                chunk_start = info['start_sec']
                if not quiet:
                    print(f"Transcribing chunk {idx}/{len(chunk_infos)}: {chunk_file} (start={chunk_start:.2f}s)")
                try:
                    segments, info = _transcribe_file(
                        model,
                        chunk_file,
                        language,
                        beam_size,
                        prev_prompt=prev_prompt,
                        vad_filter=vad_filter,
                        vad_speech_pad_ms=vad_speech_pad_ms,
                    )
                    # Store detected language from first chunk
                    if idx == 1 and language is None:
                        detected_language = info.language
                        if not quiet:
                            print(f"Auto-detected language: {detected_language} (confidence: {info.language_probability:.2f})")
                    chunk_segs, text, last_global_end = _accumulate_segments(segments, chunk_start, last_global_end)
                    all_segments.extend(chunk_segs)
                    all_text += text.strip() + "\n"
                    # Update prompt with last tail chars of accumulated text
                    prev_prompt = (all_text[-PROMPT_TAIL_CHARS:]).strip() if all_text else None
                except Exception as e:
                    print(f"Error during transcription: {e}")
                    sys.exit(1)
                finally:
                    cf = Path(chunk_file)
                    if cf.exists():
                        cf.unlink()

    base_name = Path(media_file).stem
    if output_format == "srt":
        # Ensure segments are sorted by start time before writing
        all_segments = sorted(all_segments, key=lambda d: d["start"]) if all_segments else []
        output_file = Path(output_dir) / f"{base_name}.srt"
        _write_srt(all_segments, str(output_file))
        return {
            "segments": all_segments,
            "sentences": [],
            "detected_language": detected_language,
            "output_path": str(output_file),
            "num_segments": len(all_segments),
            "elapsed_secs": round(time.time() - _t0, 3),
        }
    else:
        # Restore punctuation before sentence tokenization
        if not quiet:
            print("Restoring punctuation...")
        # Use detected language if auto-detection was used, otherwise use provided language
        lang_for_punctuation = detected_language if language is None else language
        
        # First, optionally normalize dotted acronyms for English to avoid false splits
        if (lang_for_punctuation or '').lower() == 'en':
            all_text = _collapse_dotted_acronyms_en(all_text)

        # Then, split the text into individual segments (as they come from Whisper)
        # Split by double newlines (which separate sentences in the transcription)
        text_segments = [seg.strip() for seg in all_text.split('\n\n') if seg.strip()]
        
        sentences = []
        for segment in text_segments:
            # Process each segment individually for better punctuation
            processed_segment = restore_punctuation(segment, lang_for_punctuation)
            
            # Split the processed segment into sentences while preserving punctuation.
            # Capture unicode ellipsis too and treat it as non-terminal (continuation) punctuation.
            parts = re.split(r'(…|[.!?]+)', processed_segment)

            # Accumulate across ellipses so we don't split at "..." or "…"
            buffer = ""
            idx = 0
            while idx < len(parts):
                chunk = parts[idx].strip() if idx < len(parts) else ""
                punct = parts[idx + 1] if idx + 1 < len(parts) else ""

                if chunk:
                    # Append current chunk to buffer (punctuation handled below)
                    buffer = (buffer + " " + chunk).strip()

                # If punctuation is an ellipsis, append it and continue accumulating; not a boundary
                if punct in ('...', '…'):
                    buffer += punct
                    idx += 2
                    continue

                # If punctuation is a period inside a domain name (e.g., "espanolistos.com"),
                # merge the dot and the following TLD into the buffer and continue accumulating.
                if punct == '.':
                    next_chunk = parts[idx + 2] if idx + 2 < len(parts) else ""
                    # previous label: last token of current chunk
                    prev_label_match = re.search(r"([A-Za-z0-9-]+)$", chunk)
                    # next TLD: leading letters (2-24) of next chunk
                    next_tld_match = re.match(r"^([A-Za-z]{2,24})(\b|\W)(.*)$", next_chunk)
                    if prev_label_match and next_tld_match:
                        tld = next_tld_match.group(1)
                        boundary = next_tld_match.group(2) or ""
                        remainder = next_tld_match.group(3)
                        # Append dot + TLD without inserting a space
                        buffer += '.' + tld
                        # Replace the next chunk with boundary + remainder to preserve punctuation/space after TLD
                        parts[idx + 2] = boundary + remainder
                        idx += 2
                        continue

                # Default: if we have punctuation, append and flush as a sentence
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

                # No punctuation: if at end, flush buffer; otherwise continue accumulating
                if idx + 1 >= len(parts):
                    cleaned = re.sub(r'^[",\s]+', '', buffer)
                    if cleaned:
                        if not cleaned.endswith(('.', '!', '?')):
                            cleaned += '.'
                        sentences.append(cleaned)
                    buffer = ""
                idx += 2
        # Ensure segments are sorted by start time (for consistency in results)
        all_segments = sorted(all_segments, key=lambda d: d["start"]) if all_segments else []
        output_file = Path(output_dir) / f"{base_name}.txt"
        _write_txt(sentences, str(output_file))
        return {
            "segments": all_segments,
            "sentences": sentences,
            "detected_language": detected_language,
            "output_path": str(output_file),
            "num_segments": len(all_segments),
            "elapsed_secs": round(time.time() - _t0, 3),
        }

def _cleanup_chunks(media_file):
    """
    Clean up any leftover chunk files.
    """
    media_dir = Path(media_file).resolve().parent
    for p in media_dir.glob("*_chunk_*.wav"):
        try:
            p.unlink()
            print(f"Removed leftover chunk file: {p}")
        except Exception as e:
            print(f"Error removing chunk file {p}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio/video to sentences (TXT) or subtitles (SRT).")
    parser.add_argument("media_file", help="Path to the media file to transcribe")
    parser.add_argument("--output_dir", required=True, help="Directory where output will be written")
    parser.add_argument("--language", default="auto", help="Language code (e.g., en, es, fr, de). Use 'auto' for auto-detect")
    parser.add_argument("--output_format", choices=["txt", "srt"], default="txt", help="Output format (txt or srt)")
    parser.add_argument("--single", action="store_true", help="Transcribe the entire file in a single call (no manual chunking)")
    parser.add_argument(
        "--compute-type",
        dest="compute_type",
        default=DEFAULT_COMPUTE_TYPE,
        choices=["auto", "int8", "int8_float16", "int8_float32", "float16", "float32"],
        help="faster-whisper compute type",
    )
    # VAD settings are constants; no CLI flags for VAD to keep interface simple
    vg = parser.add_mutually_exclusive_group()
    vg.add_argument("--quiet", action="store_true", help="Reduce log output")
    vg.add_argument("--verbose", action="store_true", help="Verbose log output (default)")
    parser.set_defaults(verbose=True)

    args = parser.parse_args()

    # Determine verbosity
    quiet = args.quiet or (not args.verbose)

    # Language handling
    language_arg = args.language.strip().lower() if args.language else "auto"
    language: str | None
    if language_arg in ("auto", ""):
        language = None
    else:
        language = validate_language_code(language_arg)

    start_time = time.time()

    _cleanup_chunks(args.media_file)

    result = transcribe_with_sentences(
        args.media_file,
        args.output_dir,
        language,
        args.output_format,
        single_call=args.single,
        compute_type=args.compute_type,
        quiet=quiet,
    )

    if not quiet and result.get("detected_language"):
        print(f"Detected language: {result['detected_language']}")
    # Always print output path unless fully silent mode is introduced
    print(f"Wrote: {result['output_path']}")

    if not quiet:
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        print(f"Script completed in {minutes} minutes and {seconds} seconds.")