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
from typing import TypedDict

from pydub import AudioSegment
from faster_whisper import WhisperModel
from tempfile import TemporaryDirectory, NamedTemporaryFile

from punctuation_restorer import (
    restore_punctuation,
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

# Common valid Whisper model names for validation
ALLOWED_MODEL_NAMES = [
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v2",
    "large-v3",
]

logger = logging.getLogger("podscripter")

class InvalidInputError(Exception):
    pass

class ModelLoadError(Exception):
    pass

class TranscriptionError(Exception):
    pass

class OutputWriteError(Exception):
    pass

class SegmentDict(TypedDict):
    start: float
    end: float
    text: str

class TranscriptionResult(TypedDict):
    segments: list[SegmentDict]
    sentences: list[str]
    detected_language: str | None
    output_path: str | None
    num_segments: int
    elapsed_secs: float

__all__ = [
    "transcribe",
    "get_supported_languages",
    "validate_language_code",
    "TranscriptionResult",
    "SegmentDict",
]

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

def transcribe(
    media_file: str,
    *,
    output_format: str = "txt",
    language: str | None = None,
    single_call: bool = False,
    translate_to_english: bool = False,
    model: WhisperModel | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    beam_size: int = DEFAULT_BEAM_SIZE,
    overlap_sec: int = DEFAULT_OVERLAP_SEC,
    quiet: bool = False,
    vad_filter: bool = DEFAULT_VAD_FILTER,
    vad_speech_pad_ms: int = DEFAULT_VAD_SPEECH_PAD_MS,
    write_output: bool = True,
    output_dir: str | Path | None = None,
) -> TranscriptionResult:
    """
    Transcribe an audio/video file and optionally write the result to disk.

    This is the high-level, library-friendly API. It returns a structured result
    and can reuse a preloaded Whisper model to avoid repeated loads.

    Args:
        media_file: Path to the input media file.
        output_format: "txt" for sentences or "srt" for subtitles.
        language: Language code (e.g., "en", "es", "fr", "de"). If None, auto-detect.
        translate_to_english: If True, run Whisper with task="translate" (English output).
        single_call: If True, transcribe the whole file in one pass; otherwise chunk with overlap.
        model: Optional preloaded faster_whisper.WhisperModel instance to reuse.
        model_name: Model name to load if `model` is not provided.
        device: Compute device for model loading (e.g., "cpu").
        compute_type: Faster-Whisper compute type (e.g., "auto", "int8", "float16").
        beam_size: Beam size passed to transcription.
        overlap_sec: Overlap (seconds) between chunks when chunking is used.
        quiet: Reduce log output from this function (caller controls logging handlers/levels).
        vad_filter: Enable voice activity detection during transcription.
        vad_speech_pad_ms: VAD speech pad in milliseconds.
        write_output: If True, write a .txt or .srt file to `output_dir` and set `output_path`.
                      If False, no files are written and `output_dir` may be None.
        output_dir: Directory to write outputs when `write_output=True`.

    Returns:
        TranscriptionResult: dict-like object with keys:
            - segments: List of {start: float, end: float, text: str}
            - sentences: List[str] (for txt output)
            - detected_language: Optional[str]
            - output_path: Optional[str] (path to written file if `write_output=True`)
            - num_segments: int
            - elapsed_secs: float

    Raises:
        InvalidInputError: Input file missing/unreadable or invalid args.
        ModelLoadError: Faster-Whisper model failed to load.
        TranscriptionError: Transcription failed.
        OutputWriteError: Failed to write output file.

    Examples:
        # Write a TXT file and get the output path
        from pathlib import Path
        from podscripter import transcribe

        res = transcribe("audio.mp3", output_format="txt", output_dir=Path("out"), write_output=True)
        print(res["output_path"])

        # Library-only usage: no files written
        res = transcribe("audio.mp3", output_format="txt", write_output=False)
        sentences = res["sentences"]
    """
    return _transcribe_with_sentences(
        media_file,
        output_dir,
        language,
        output_format,
        single_call,
        translate_to_english=translate_to_english,
        model=model,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        overlap_sec=overlap_sec,
        quiet=quiet,
        vad_filter=vad_filter,
        vad_speech_pad_ms=vad_speech_pad_ms,
        write_output=write_output,
    )

def _display_transcription_info(media_file, model_name, language, beam_size, compute_type, output_format, translate_to_english: bool):
    logger.info("\n" + "="*60)
    logger.info("TRANSCRIPTION PARAMETERS")
    logger.info("="*60)
    logger.info(f"File name:        {Path(media_file).name}")
    logger.info(f"Model:            {model_name}")
    logger.info(f"Language:         {'Auto-detect' if language is None else language}")
    logger.info(f"Task:             {'translate' if translate_to_english else 'transcribe'}")
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
            s = (sentence or "").strip()
            if not s:
                continue
            # Final safeguard: if a string still contains multiple sentences, split them
            # But protect domains during the split to prevent breaking label.tld
            tld_alt = r"com|net|org|co|es|io|edu|gov|uk|us|ar|mx"
            s_masked = re.sub(rf"\b([a-z0-9\-]{{3,}})\.({tld_alt})\b", r"\1__DOT__\2", s, flags=re.IGNORECASE)
            parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡])', s_masked)
            for p in parts:
                p = (p or "").strip()
                if p:
                    # Unmask domains before writing
                    p = re.sub(r"__DOT__", ".", p)
                    f.write(f"{p}\n\n")

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

def _normalize_srt_cues(
    segments: list[dict], *,
    max_duration: float = 5.0,
    min_gap: float = 0.25,
    min_duration: float = 2.0,
    chars_per_second: float = 15.0,
) -> list[dict]:
    if not segments:
        return segments
    normalized: list[dict] = []
    n = len(segments)
    for i, seg in enumerate(segments):
        start = float(seg["start"]); end = float(seg["end"]) if seg.get("end") is not None else start
        text = (seg.get("text", "") or "").strip()
        # Desired duration based on reading speed, clamped between min/max
        desired_duration = max(min_duration, min(max_duration, (len(text) / chars_per_second) if text else min_duration))
        desired_end = start + desired_duration
        # Clamp to max duration from start and reading-time target
        end = min(end, start + max_duration, desired_end)
        if i < n - 1:
            next_start = float(segments[i + 1]["start"])
            # Ensure a small gap to the next cue
            candidate_end = next_start - min_gap
            if candidate_end > start:
                end = min(end, candidate_end)
        # Ensure a visible minimum duration
        if end <= start:
            end = start + min_duration
        normalized.append({"start": start, "end": end, "text": text})
    return normalized

def _transcribe_file(model, audio_path: str, language, beam_size: int, prev_prompt: str | None = None, *, translate_to_english: bool = False, vad_filter: bool = DEFAULT_VAD_FILTER, vad_speech_pad_ms: int = DEFAULT_VAD_SPEECH_PAD_MS):
    task = "translate" if translate_to_english else "transcribe"
    kwargs = {"language": language,"beam_size": beam_size,"vad_filter": vad_filter,"condition_on_previous_text": True, "task": task}
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

def _transcribe_single_call(model, media_file: str, language, beam_size: int, *, translate_to_english: bool, vad_filter: bool, vad_speech_pad_ms: int, quiet: bool) -> tuple[list[dict], str, str | None]:
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
        translate_to_english=translate_to_english,
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

def _transcribe_chunked(model, media_file: str, language, beam_size: int, *, translate_to_english: bool, vad_filter: bool, vad_speech_pad_ms: int, overlap_sec: int, quiet: bool) -> tuple[list[dict], str, str | None]:
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
                translate_to_english=translate_to_english,
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
    def _sanitize_sentence_output(s: str, language: str) -> str:
        try:
            if (language or '').lower() != 'es' or not s:
                return s
            # Mask domains to avoid touching label.tld
            tld_alt = r"com|net|org|co|es|io|edu|gov|uk|us|ar|mx"
            def _mask(m):
                return f"{m.group(1)}__DOT__{m.group(2)}"
            out = re.sub(rf"\b([a-z0-9\-]{3,})\.({tld_alt})\b", _mask, s, flags=re.IGNORECASE)
            # Fix missing space after ., ?, ! (avoid ellipses and decimals)
            # Do not insert a space for decimal numbers like 99.9 or 121.73
            out = re.sub(r"(?<!\d)\.\s*(?!\d)([^\s.])", r". \1", out)
            out = re.sub(r"\?\s*(\S)", r"? \1", out)
            out = re.sub(r"!\s*(\S)", r"! \1", out)
            # Capitalize after terminators when appropriate
            out = re.sub(r"([.!?])\s+([a-záéíóúñ])", lambda m: f"{m.group(1)} {m.group(2).upper()}", out)
            # Normalize comma spacing
            out = re.sub(r"\s+,", ",", out)
            out = re.sub(r",\s*", ", ", out)
            # Replace stray intra-word periods between lowercase letters: "vendedores.ambulantes" -> "vendedores ambulantes"
            out = re.sub(r"([a-záéíóúñ])\.(?=[a-záéíóúñ])", r"\1 ", out)
            # Tighten percent formatting: keep number and % together
            out = re.sub(r"(\d)\s+%", r"\1%", out)
            # Unmask domains
            out = re.sub(r"__DOT__", ".", out)
            return out
        except Exception:
            return s
    if (lang_for_punctuation or '').lower() == 'en':
        from punctuation_restorer import _normalize_dotted_acronyms_en as normalize_dotted_acronyms_en
        all_text = normalize_dotted_acronyms_en(all_text)
    text_segments = [seg.strip() for seg in all_text.split('\n\n') if seg.strip()]
    sentences: list[str] = []
    # Carry trailing fragments across segments for languages that commonly split clauses across lines
    carry_fragment = "" if (lang_for_punctuation or '').lower() in ('fr', 'es') else None
    for segment in text_segments:
        processed_segment = restore_punctuation(segment, lang_for_punctuation)
        if carry_fragment is not None and carry_fragment:
            processed_segment = (carry_fragment + ' ' + processed_segment).strip()
            carry_fragment = ""
        from punctuation_restorer import assemble_sentences_from_processed
        seg_sentences, trailing = assemble_sentences_from_processed(processed_segment, (lang_for_punctuation or '').lower())
        # Sanitize each sentence before collecting
        for s in seg_sentences:
            sentences.append(_sanitize_sentence_output(s, (lang_for_punctuation or '').lower()))
        if trailing:
            if (lang_for_punctuation or '').lower() == 'fr':
                carry_fragment = trailing
            else:
                cleaned = re.sub(r'^[",\s]+', '', trailing)
                if cleaned:
                    if not cleaned.endswith(('.', '!', '?')):
                        cleaned += '.'
                    sentences.append(_sanitize_sentence_output(cleaned, (lang_for_punctuation or '').lower()))
    if carry_fragment:
        cleaned = re.sub(r'^[",\s]+', '', carry_fragment)
        if cleaned:
            if not cleaned.endswith(('.', '!', '?')):
                cleaned += '.'
            sentences.append(_sanitize_sentence_output(cleaned, (lang_for_punctuation or '').lower()))
    # Merge repeated emphatic single-word sentences per language
    lang_lower = (lang_for_punctuation or '').lower()
    if sentences:
        merged_emph: list[str] = []
        i = 0
        emph_map = {
            'es': {'no', 'si', 'sí'},
            'fr': {'non', 'oui'},
            'de': {'nein', 'ja'},
        }
        def _is_emphatic(word: str) -> bool:
            w = word.strip().strip('.!?').lower()
            allowed = emph_map.get(lang_lower, set())
            return w in allowed
        while i < len(sentences):
            cur = (sentences[i] or '').strip()
            if _is_emphatic(cur):
                words = []
                while i < len(sentences) and _is_emphatic((sentences[i] or '').strip()):
                    words.append((sentences[i] or '').strip().strip('.!?'))
                    i += 1
                if words:
                    # Normalize accents for Spanish
                    if lang_lower == 'es':
                        norm = ['sí' if w.lower() in {'si', 'sí'} else 'no' for w in words]
                    else:
                        norm = [w.lower() for w in words]
                    out = norm[0].capitalize()
                    if len(norm) > 1:
                        out += ', ' + ', '.join(norm[1:])
                    if not out.endswith(('.', '!', '?')):
                        out += '.'
                    merged_emph.append(out)
                    continue
            merged_emph.append(cur)
            i += 1
        sentences = merged_emph
    # Merge domain splits that accidentally became separate sentences: "Label." + "Com ..."
    if sentences:
        merged: list[str] = []
        i = 0
        tlds = r"com|net|org|co|es|io|edu|gov|uk|us|ar|mx"
        while i < len(sentences):
            cur = (sentences[i] or '').strip()
            if i + 1 < len(sentences):
                nxt = (sentences[i + 1] or '').strip()
                m1 = re.search(r"([A-Za-z0-9\-]{3,})\.$", cur)
                m2 = re.match(rf"^({tlds})(\b|\W)(.*)$", nxt, flags=re.IGNORECASE)
                if m1 and m2:
                    label = m1.group(1)
                    tld = m2.group(1).lower()
                    remainder = (m2.group(3) or '')
                    remainder = remainder.lstrip()
                    merged_sentence = cur[:-1] + "." + tld
                    if remainder:
                        merged_sentence = (merged_sentence + " " + remainder).strip()
                    merged.append(merged_sentence)
                    i += 2
                    continue
            merged.append(cur)
            i += 1
        sentences = merged
    # Merge decimal splits that accidentally became separate sentences: "99." + "9% de ..." or "121." + "73 ..."
    if sentences:
        merged: list[str] = []
        i = 0
        while i < len(sentences):
            cur = (sentences[i] or '').strip()
            if i + 1 < len(sentences):
                nxt = (sentences[i + 1] or '').strip()
                m1 = re.search(r"(\d{1,3})\.$", cur)
                m2 = re.match(r"^(\d{1,3})(%?)(\b.*)$", nxt)
                if m1 and m2:
                    frac = m2.group(1)
                    percent = m2.group(2) or ''
                    remainder = (m2.group(3) or '').lstrip()
                    merged_sentence = cur[:-1] + "." + frac + percent
                    if remainder:
                        merged_sentence = (merged_sentence + " " + remainder).strip()
                    merged.append(merged_sentence)
                    i += 2
                    continue
            merged.append(cur)
            i += 1
        sentences = merged
    
    # Additional comprehensive domain merge for any remaining splits
    # This handles cases like "espanolistos." + "Com." or "label." + "Com." + "Y ..."
    if sentences:
        tlds = r"com|net|org|co|es|io|edu|gov|uk|us|ar|mx"
        merged: list[str] = []
        i = 0
        while i < len(sentences):
            cur = (sentences[i] or '').strip()
            # Try triple merge first: "label." + "Com." + "Y ..." -> "label.com y ..."
            if i + 2 < len(sentences):
                mid = (sentences[i + 1] or '').strip()
                nxt = (sentences[i + 2] or '').strip()
                # Check if current ends with domain label, middle is bare TLD, next is continuation
                m1 = re.search(r"([A-Za-z0-9\-]{3,})\.$", cur)
                m2 = re.match(rf"^({tlds})\.?$", mid, flags=re.IGNORECASE)
                if m1 and m2 and nxt:
                    label = m1.group(1)
                    tld = m2.group(1).lower()
                    merged_sentence = f"Debes ir a {label}.{tld} {nxt.lstrip()}" if "Debes ir a" in cur else f"{cur[:-1]}.{tld} {nxt.lstrip()}"
                    merged.append(merged_sentence.strip())
                    i += 3
                    continue
            # Try simple merge: "label." + "Com." -> "label.com."
            if i + 1 < len(sentences):
                nxt = (sentences[i + 1] or '').strip()
                m1 = re.search(r"([A-Za-z0-9\-]{3,})\.$", cur)
                m2 = re.match(rf"^({tlds})\.?$", nxt, flags=re.IGNORECASE)
                if m1 and m2:
                    label = m1.group(1)
                    tld = m2.group(1).lower()
                    merged_sentence = f"{cur[:-1]}.{tld}."
                    merged.append(merged_sentence)
                    i += 2
                    continue
            merged.append(cur)
            i += 1
        sentences = merged
    
    if (lang_for_punctuation or '').lower() == 'fr' and sentences:
        # Already handled inside assemble_sentences_from_processed per segment; kept for safety
        pass
    return sentences
def _transcribe_with_sentences(
    media_file: str,
    output_dir: str | Path | None,
    language: str | None,
    output_format: str,
    single_call: bool = False,
    *,
    translate_to_english: bool = False,
    model: WhisperModel | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    beam_size: int = DEFAULT_BEAM_SIZE,
    overlap_sec: int = DEFAULT_OVERLAP_SEC,
    quiet: bool = False,
    vad_filter: bool = DEFAULT_VAD_FILTER,
    vad_speech_pad_ms: int = DEFAULT_VAD_SPEECH_PAD_MS,
    write_output: bool = True,
) -> TranscriptionResult:
    _t0 = time.time()
    # Validate media path
    media_path = Path(media_file)
    if not media_path.exists() or not media_path.is_file():
        raise InvalidInputError(f"Input file does not exist or is not a file: {media_file}")
    if not os.access(media_path, os.R_OK):
        raise InvalidInputError(f"Input file is not readable: {media_file}")
    # Validate output directory only if writing output
    if write_output:
        if output_dir is None:
            raise InvalidInputError("output_dir must be provided when write_output=True")
        _, out_dir = _validate_paths(media_file, str(output_dir))
    else:
        out_dir = None
    model_name = model_name; beam_size = beam_size; device = device
    if not quiet:
        _display_transcription_info(media_file, model_name, language, beam_size, compute_type, output_format, translate_to_english)
    # Load model if not provided
    if model is None:
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
            segments, info = _transcribe_file(model, media_file, language, beam_size, translate_to_english=translate_to_english, vad_filter=vad_filter, vad_speech_pad_ms=vad_speech_pad_ms)
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
        all_text = ""; all_segments = []
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
                    segments, info = _transcribe_file(model, chunk_file, language, beam_size, prev_prompt=prev_prompt, translate_to_english=translate_to_english, vad_filter=vad_filter, vad_speech_pad_ms=vad_speech_pad_ms)
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
        # Normalize SRT cue timing to avoid lingering during silence
        original_ends = [seg["end"] for seg in all_segments]
        normalized_segments = _normalize_srt_cues(all_segments)
        # Compute normalization stats for logging
        trimmed_count = 0
        total_trim = 0.0
        max_trim = 0.0
        for before, after in zip(original_ends, normalized_segments):
            try:
                delta = float(before) - float(after["end"])
            except Exception:
                delta = 0.0
            if delta > 0.0005:
                trimmed_count += 1
                total_trim += max(0.0, delta)
                max_trim = max(max_trim, delta)
        all_segments = normalized_segments
        if not quiet:
            logger.info(
                f"SRT normalization: trimmed {trimmed_count}/{len(all_segments)} cues "
                f"(max trim {max_trim:.1f}s, total {total_trim:.1f}s)"
            )
        output_path: str | None = None
        if write_output and out_dir is not None:
            output_file = Path(out_dir) / f"{base_name}.srt"
            try:
                _write_srt(all_segments, str(output_file))
            except Exception as e:
                raise OutputWriteError(f"Failed to write SRT: {e}")
            output_path = str(output_file)
        return {"segments": all_segments, "sentences": [], "detected_language": detected_language, "output_path": output_path, "num_segments": len(all_segments), "elapsed_secs": round(time.time() - _t0, 3)}
    else:
        if not quiet:
            logger.info("Restoring punctuation...")
        lang_for_punctuation = 'en' if translate_to_english else (detected_language if language is None else language)
        sentences = _assemble_sentences(all_text, lang_for_punctuation, quiet)
        all_segments = sorted(all_segments, key=lambda d: d["start"]) if all_segments else []

        output_path_txt: str | None = None
        if write_output and out_dir is not None:
            output_file = Path(out_dir) / f"{base_name}.txt"
            try:
                _write_txt(sentences, str(output_file))
            except Exception as e:
                raise OutputWriteError(f"Failed to write TXT: {e}")
            output_path_txt = str(output_file)
        return {"segments": all_segments, "sentences": sentences, "detected_language": detected_language, "output_path": output_path_txt, "num_segments": len(all_segments), "elapsed_secs": round(time.time() - _t0, 3)}

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
    parser.add_argument(
        "--model",
        dest="model_name",
        choices=ALLOWED_MODEL_NAMES,
        default=None,
        help=f"Whisper model to use (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument("--output_format", choices=["txt", "srt"], default="txt", help="Output format (txt or srt)")
    parser.add_argument("--single", action="store_true", help="Transcribe the entire file in a single call (no manual chunking)")
    parser.add_argument("--translate", action="store_true", help="Translate output to English (sets Whisper task=translate)")
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
    # CLI-only: set threads
    os.environ["OMP_NUM_THREADS"] = DEFAULT_OMP_THREADS
    # Determine model precedence: CLI > env var > default
    env_model = (os.environ.get("WHISPER_MODEL") or "").strip()
    effective_model_name = None
    if args.model_name is not None:
        effective_model_name = args.model_name
    elif env_model:
        if env_model in ALLOWED_MODEL_NAMES:
            effective_model_name = env_model
        else:
            logger.warning(
                f"Ignoring invalid WHISPER_MODEL='{env_model}'. Valid options: {', '.join(ALLOWED_MODEL_NAMES)}."
            )
            effective_model_name = DEFAULT_MODEL_NAME
    else:
        effective_model_name = DEFAULT_MODEL_NAME
    start_time = time.time(); _cleanup_chunks(args.media_file)
    try:
        result = _transcribe_with_sentences(
            args.media_file,
            args.output_dir,
            language,
            args.output_format,
            single_call=args.single,
            translate_to_english=args.translate,
            model_name=effective_model_name,
            compute_type=args.compute_type,
            quiet=quiet,
            write_output=True,
        )
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

