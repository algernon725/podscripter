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
import time
import argparse
import logging
from pathlib import Path

from pydub import AudioSegment
from faster_whisper import WhisperModel
from tempfile import TemporaryDirectory, NamedTemporaryFile

from punctuation_restorer import (
    restore_punctuation,
    normalize_dotted_acronyms_en,
    split_processed_segment,
    fr_merge_short_connector_breaks,
)

FOCUS_LANGS = {"en", "es", "fr", "de"}

DEFAULT_CHUNK_SEC = 480
DEFAULT_OVERLAP_SEC = 3
DEFAULT_BEAM_SIZE = 3
DEFAULT_COMPUTE_TYPE = "auto"
DEFAULT_DEVICE = "cpu"
DEFAULT_MODEL_NAME = "medium"
DEFAULT_OMP_THREADS = "8"
DEDUPE_EPSILON_SEC = 0.05
PROMPT_TAIL_CHARS = 200
DEFAULT_VAD_FILTER = True
DEFAULT_VAD_SPEECH_PAD_MS = 200

logger = logging.getLogger("podscripter")

class InvalidInputError(Exception):
    pass

class ModelLoadError(Exception):
    pass

class TranscriptionError(Exception):
    pass

class OutputWriteError(Exception):
    pass

 
def get_supported_languages() -> dict[str, str]:
    return {
        'en': 'English','es': 'Spanish','fr': 'French','de': 'German','ja': 'Japanese','ru': 'Russian','cs': 'Czech',
        'it': 'Italian','pt': 'Portuguese','nl': 'Dutch','pl': 'Polish','tr': 'Turkish','ar': 'Arabic','zh': 'Chinese',
        'ko': 'Korean','hi': 'Hindi','sv': 'Swedish','da': 'Danish','no': 'Norwegian','fi': 'Finnish'
    }

def validate_language_code(language_code: str | None) -> str | None:
    if language_code is None:
        return None
    supported = get_supported_languages()
    if language_code in supported:
        return language_code
    logger.warning(f"Language code '{language_code}' not in common list.")
    logger.info("Primary language codes:")
    for code in ["en","es","fr","de"]:
        if code in supported:
            logger.info(f"  {code}: {supported[code]}")
    logger.info("Experimental language codes:")
    for code, name in supported.items():
        if code not in FOCUS_LANGS:
            logger.info(f"  {code}: {name} (experimental)")
    logger.info("Whisper supports many more languages. The code will still work if it's valid.")
    return language_code

def _display_transcription_info(media_file, model_name, language, beam_size, compute_type, output_format):
    logger.info("\n" + "="*60)
    logger.info("TRANSCRIPTION PARAMETERS")
    logger.info("="*60)
    logger.info(f"File name:        {Path(media_file).name}")
    logger.info(f"Model:            {model_name}")
    logger.info(f"Language:         {'Auto-detect' if language is None else language}")
    logger.info(f"Beam size:        {beam_size}")
    logger.info(f"Compute type:     {compute_type}")
    logger.info(f"Output format:    {output_format}")
    logger.info("="*60 + "\n")

def _split_audio_with_overlap(media_file: str, chunk_length_sec: int = DEFAULT_CHUNK_SEC, overlap_sec: int = DEFAULT_OVERLAP_SEC, chunk_dir: Path | None = None):
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
        chunk_infos.append({'path': str(chunk_path),'start_sec': start_ms / 1000.0,'duration_sec': (end_ms - start_ms) / 1000.0})
        idx += 1
        if end_ms >= len(audio):
            break
    return chunk_infos

def _write_txt(sentences, output_file):
    with open(output_file, "w") as f:
        for sentence in sentences:
            f.write(f"{sentence.strip()}\n\n")

def _write_srt(segments, output_file):
    def format_timestamp(seconds):
        h = int(seconds // 3600); m = int((seconds % 3600) // 60); s = int(seconds % 60); ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    with open(output_file, "w") as f:
        for i, seg in enumerate(segments, 1):
            start = format_timestamp(seg['start']); end = format_timestamp(seg['end']); text = seg['text'].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

def _validate_paths(media_file: str, output_dir: str) -> tuple[Path, Path]:
    media_path = Path(media_file)
    if not media_path.exists() or not media_path.is_file():
        raise InvalidInputError(f"Input file does not exist or is not a file: {media_file}")
    if not os.access(media_path, os.R_OK):
        raise InvalidInputError(f"Input file is not readable: {media_file}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=str(out_dir), delete=True):
        pass
    return media_path, out_dir

def _transcribe_file(model, audio_path: str, language, beam_size: int, prev_prompt: str | None = None, *, vad_filter: bool = DEFAULT_VAD_FILTER, vad_speech_pad_ms: int = DEFAULT_VAD_SPEECH_PAD_MS):
    kwargs = {"language": language,"beam_size": beam_size,"vad_filter": vad_filter,"condition_on_previous_text": True}
    if vad_filter:
        kwargs["vad_parameters"] = {"speech_pad_ms": int(max(0, vad_speech_pad_ms))}
    if prev_prompt:
        kwargs["initial_prompt"] = prev_prompt
    segments, info = model.transcribe(audio_path, **kwargs)
    return segments, info

def _dedupe_segments(global_segments, last_end: float, epsilon: float = DEDUPE_EPSILON_SEC):
    out = []; new_last = last_end
    for g_start, g_end, text in global_segments:
        if g_end <= new_last + epsilon:
            continue
        out.append({"start": g_start, "end": g_end, "text": text}); new_last = max(new_last, g_end)
    return out, new_last

def _accumulate_segments(model_segments, chunk_start: float, last_end: float, epsilon: float = DEDUPE_EPSILON_SEC):
    global_segs = []
    for seg in model_segments:
        g_start = chunk_start + float(seg.start); g_end = chunk_start + float(seg.end)
        global_segs.append((g_start, g_end, seg.text))
    deduped, new_last = _dedupe_segments(global_segs, last_end, epsilon)
    text = " ".join(d["text"] for d in deduped)
    return deduped, text, new_last

def _load_model(model_name: str, device: str, compute_type: str) -> WhisperModel:
    return WhisperModel(model_name, device=device, compute_type=compute_type)

def _transcribe_single_call(model, media_file: str, language, beam_size: int, *, vad_filter: bool, vad_speech_pad_ms: int, quiet: bool) -> tuple[list[dict], str, str | None]:
    if not quiet:
        logger.info("Transcribing full file in a single call (no manual chunking)...")
    all_text = ""
    all_segments: list[dict] = []
    detected_language = None
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
            logger.info(f"Auto-detected language: {detected_language} (confidence: {info.language_probability:.2f})")
    accum, text, _ = _accumulate_segments(segments, 0.0, last_end=0.0)
    all_segments.extend(accum)
    all_text += (text + " ")
    return all_segments, all_text, detected_language

def _transcribe_chunked(model, media_file: str, language, beam_size: int, *, vad_filter: bool, vad_speech_pad_ms: int, overlap_sec: int, quiet: bool) -> tuple[list[dict], str, str | None]:
    if not quiet:
        logger.info("Splitting media into chunks with overlap...")
    all_text = ""
    all_segments: list[dict] = []
    detected_language = None
    with TemporaryDirectory() as tmp_dir:
        chunk_infos = _split_audio_with_overlap(
            media_file,
            chunk_length_sec=DEFAULT_CHUNK_SEC,
            overlap_sec=overlap_sec,
            chunk_dir=Path(tmp_dir),
        )
        if not quiet:
            logger.info("Transcribing chunks...")
        last_global_end = 0.0
        prev_prompt = None
        for idx, info in enumerate(chunk_infos, 1):
            chunk_file = info['path']
            chunk_start = info['start_sec']
            if not quiet:
                logger.info(f"Transcribing chunk {idx}/{len(chunk_infos)}: {chunk_file} (start={chunk_start:.2f}s)")
            segments, info = _transcribe_file(
                model,
                chunk_file,
                language,
                beam_size,
                prev_prompt=prev_prompt,
                vad_filter=vad_filter,
                vad_speech_pad_ms=vad_speech_pad_ms,
            )
            if idx == 1 and language is None:
                detected_language = info.language
                if not quiet:
                    logger.info(f"Auto-detected language: {detected_language} (confidence: {info.language_probability:.2f})")
            chunk_segs, text, last_global_end = _accumulate_segments(segments, chunk_start, last_global_end)
            all_segments.extend(chunk_segs)
            all_text += text.strip() + "\n"
            prev_prompt = (all_text[-PROMPT_TAIL_CHARS:]).strip() if all_text else None
            cf = Path(chunk_file)
            if cf.exists():
                cf.unlink()
    return all_segments, all_text, detected_language

def _assemble_sentences(all_text: str, lang_for_punctuation: str | None, quiet: bool) -> list[str]:
    if (lang_for_punctuation or '').lower() == 'en':
        all_text = normalize_dotted_acronyms_en(all_text)
    text_segments = [seg.strip() for seg in all_text.split('\n\n') if seg.strip()]
    sentences: list[str] = []
    carry_fragment = "" if (lang_for_punctuation or '').lower() == 'fr' else None
    for segment in text_segments:
        processed_segment = restore_punctuation(segment, lang_for_punctuation)
        if carry_fragment is not None and carry_fragment:
            processed_segment = (carry_fragment + ' ' + processed_segment).strip()
            carry_fragment = ""
        seg_sentences, trailing = split_processed_segment(processed_segment, (lang_for_punctuation or '').lower())
        sentences.extend(seg_sentences)
        if trailing:
            if (lang_for_punctuation or '').lower() == 'fr':
                carry_fragment = trailing
            else:
                cleaned = re.sub(r'^[",\s]+', '', trailing)
                if cleaned:
                    if not cleaned.endswith(('.', '!', '?')):
                        cleaned += '.'
                    sentences.append(cleaned)
    if carry_fragment:
        cleaned = re.sub(r'^[",\s]+', '', carry_fragment)
        if cleaned:
            if not cleaned.endswith(('.', '!', '?')):
                cleaned += '.'
            sentences.append(cleaned)
    if (lang_for_punctuation or '').lower() == 'fr' and sentences:
        sentences = fr_merge_short_connector_breaks(sentences)
    return sentences
def transcribe_with_sentences(media_file: str, output_dir: str, language: str | None, output_format: str, single_call: bool = False, *, compute_type: str = DEFAULT_COMPUTE_TYPE, quiet: bool = False, vad_filter: bool = DEFAULT_VAD_FILTER, vad_speech_pad_ms: int = DEFAULT_VAD_SPEECH_PAD_MS) -> dict:
    os.environ["OMP_NUM_THREADS"] = DEFAULT_OMP_THREADS
    _t0 = time.time()
    media_path, out_dir = _validate_paths(media_file, output_dir)
    model_name = DEFAULT_MODEL_NAME; beam_size = DEFAULT_BEAM_SIZE; device = DEFAULT_DEVICE
    if not quiet:
        _display_transcription_info(media_file, model_name, language, beam_size, compute_type, output_format)
    try:
        model = _load_model(model_name, device, compute_type)
    except Exception as e:
        raise ModelLoadError(f"Error loading faster-whisper model: {e}")
    detected_language = None
    if single_call:
        if not quiet:
            logger.info("Transcribing full file in a single call (no manual chunking)...")
        all_text = ""; all_segments = []
        try:
            segments, info = _transcribe_file(model, media_file, language, beam_size, vad_filter=vad_filter, vad_speech_pad_ms=vad_speech_pad_ms)
            if language is None:
                detected_language = info.language
                if not quiet:
                    logger.info(f"Auto-detected language: {detected_language} (confidence: {info.language_probability:.2f})")
            accum, text, _ = _accumulate_segments(segments, 0.0, last_end=0.0)
            all_segments.extend(accum); all_text += (text + " ")
        except Exception as e:
            raise TranscriptionError(f"Error during transcription: {e}")
    else:
        if not quiet:
            logger.info("Splitting media into chunks with overlap...")
        overlap_sec = DEFAULT_OVERLAP_SEC; all_text = ""; all_segments = []
        with TemporaryDirectory() as tmp_dir:
            chunk_infos = _split_audio_with_overlap(media_file, chunk_length_sec=DEFAULT_CHUNK_SEC, overlap_sec=overlap_sec, chunk_dir=Path(tmp_dir))
            if not quiet:
                logger.info("Transcribing chunks...")
            last_global_end = 0.0; prev_prompt = None
            for idx, info in enumerate(chunk_infos, 1):
                chunk_file = info['path']; chunk_start = info['start_sec']
                if not quiet:
                    logger.info(f"Transcribing chunk {idx}/{len(chunk_infos)}: {chunk_file} (start={chunk_start:.2f}s)")
                try:
                    segments, info = _transcribe_file(model, chunk_file, language, beam_size, prev_prompt=prev_prompt, vad_filter=vad_filter, vad_speech_pad_ms=vad_speech_pad_ms)
                    if idx == 1 and language is None:
                        detected_language = info.language
                        if not quiet:
                            logger.info(f"Auto-detected language: {detected_language} (confidence: {info.language_probability:.2f})")
                    chunk_segs, text, last_global_end = _accumulate_segments(segments, chunk_start, last_global_end)
                    all_segments.extend(chunk_segs); all_text += text.strip() + "\n"
                    prev_prompt = (all_text[-PROMPT_TAIL_CHARS:]).strip() if all_text else None
                except Exception as e:
                    raise TranscriptionError(f"Error during transcription: {e}")
                finally:
                    cf = Path(chunk_file)
                    if cf.exists():
                        cf.unlink()
    base_name = Path(media_file).stem
    if output_format == "srt":
        all_segments = sorted(all_segments, key=lambda d: d["start"]) if all_segments else []
        output_file = Path(output_dir) / f"{base_name}.srt"
        try:
            _write_srt(all_segments, str(output_file))
        except Exception as e:
            raise OutputWriteError(f"Failed to write SRT: {e}")
        return {"segments": all_segments, "sentences": [], "detected_language": detected_language, "output_path": str(output_file), "num_segments": len(all_segments), "elapsed_secs": round(time.time() - _t0, 3)}
    else:
        if not quiet:
            logger.info("Restoring punctuation...")
        lang_for_punctuation = detected_language if language is None else language
        sentences = _assemble_sentences(all_text, lang_for_punctuation, quiet)
        all_segments = sorted(all_segments, key=lambda d: d["start"]) if all_segments else []

        output_file = Path(output_dir) / f"{base_name}.txt"
        try:
            _write_txt(sentences, str(output_file))
        except Exception as e:
            raise OutputWriteError(f"Failed to write TXT: {e}")
        return {"segments": all_segments, "sentences": sentences, "detected_language": detected_language, "output_path": str(output_file), "num_segments": len(all_segments), "elapsed_secs": round(time.time() - _t0, 3)}

def _cleanup_chunks(media_file):
    media_dir = Path(media_file).resolve().parent
    for p in media_dir.glob("*_chunk_*.wav"):
        try:
            p.unlink(); logger.info(f"Removed leftover chunk file: {p}")
        except Exception as e:
            logger.warning(f"Error removing chunk file {p}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio/video to sentences (TXT) or subtitles (SRT).")
    parser.add_argument("media_file", help="Path to the media file to transcribe")
    parser.add_argument("--output_dir", required=True, help="Directory where output will be written")
    parser.add_argument("--language", default="auto", help="Language code (e.g., en, es, fr, de). Use 'auto' for auto-detect")
    parser.add_argument("--output_format", choices=["txt", "srt"], default="txt", help="Output format (txt or srt)")
    parser.add_argument("--single", action="store_true", help="Transcribe the entire file in a single call (no manual chunking)")
    parser.add_argument("--compute-type", dest="compute_type", default=DEFAULT_COMPUTE_TYPE, choices=["auto", "int8", "int8_float16", "int8_float32", "float16", "float32"], help="faster-whisper compute type")
    vg = parser.add_mutually_exclusive_group(); vg.add_argument("--quiet", action="store_true", help="Reduce log output"); vg.add_argument("--verbose", action="store_true", help="Verbose log output (default)"); parser.set_defaults(verbose=True)
    args = parser.parse_args()
    quiet = args.quiet or (not args.verbose)
    # Configure logging
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR if quiet else logging.INFO)
    language_arg = args.language.strip().lower() if args.language else "auto"
    language: str | None = None if language_arg in ("auto", "") else validate_language_code(language_arg)
    start_time = time.time(); _cleanup_chunks(args.media_file)
    try:
        result = transcribe_with_sentences(args.media_file, args.output_dir, language, args.output_format, single_call=args.single, compute_type=args.compute_type, quiet=quiet)
        if not quiet and result.get("detected_language"):
            logger.info(f"Detected language: {result['detected_language']}")
        logger.info(f"Wrote: {result['output_path']}")
        if not quiet:
            elapsed = time.time() - start_time; minutes = int(elapsed // 60); seconds = int(elapsed % 60)
            logger.info(f"Script completed in {minutes} minutes and {seconds} seconds.")
    except InvalidInputError as e:
        logger.error(str(e)); sys.exit(2)
    except ModelLoadError as e:
        logger.error(str(e)); sys.exit(3)
    except TranscriptionError as e:
        logger.error(str(e)); sys.exit(4)
    except OutputWriteError as e:
        logger.error(str(e)); sys.exit(5)
    except Exception as e:
        if not quiet:
            logger.error(f"Unexpected error: {e}")
        else:
            logger.error("Unexpected error. Run with --verbose for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()

