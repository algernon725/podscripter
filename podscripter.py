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
    _normalize_comma_spacing,
)
from domain_utils import fix_spaced_domains, mask_domains, unmask_domains
from sentence_splitter import Sentence, Utterance

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
    # Diarization debugging info (None if diarization not enabled)
    diarization_result: dict | None
    whisper_boundaries: list[float] | None
    merged_boundaries: list[float] | None

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
    enable_diarization: bool = False,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    hf_token: str | None = None,
    dump_merge_metadata: bool = False,
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
        enable_diarization: If True, perform speaker diarization for improved sentence boundaries.
        min_speakers: Minimum number of speakers (None for auto-detect).
        max_speakers: Maximum number of speakers (None for auto-detect).
        hf_token: Hugging Face token for pyannote model access (required for first-time download).

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
        enable_diarization=enable_diarization,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        hf_token=hf_token,
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

def _write_merge_metadata_dump(merge_metadata, output_file: str):
    """
    Write merge metadata debug dump to file.
    
    Args:
        merge_metadata: List of MergeMetadata objects
        output_file: Path to output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Sentence Merge Metadata Report\n")
        f.write("=" * 60 + "\n\n")
        
        # Count merges by type
        merge_counts = {}
        skipped_counts = {}
        for m in merge_metadata:
            if m.reason.startswith('skipped:'):
                skipped_counts[m.merge_type] = skipped_counts.get(m.merge_type, 0) + 1
            else:
                merge_counts[m.merge_type] = merge_counts.get(m.merge_type, 0) + 1
        
        # Summary
        total_merges = sum(merge_counts.values())
        total_skipped = sum(skipped_counts.values())
        f.write(f"Total merges applied: {total_merges}\n")
        f.write(f"Total merges skipped: {total_skipped}\n\n")
        
        if merge_counts:
            f.write("Merges by type:\n")
            for mtype, count in sorted(merge_counts.items()):
                f.write(f"  - {mtype}: {count}\n")
            f.write("\n")
        
        if skipped_counts:
            f.write("Skipped merges by type:\n")
            for mtype, count in sorted(skipped_counts.items()):
                f.write(f"  - {mtype}: {count}\n")
            f.write("\n")
        
        # Detailed merge information
        f.write("=" * 60 + "\n")
        f.write("Detailed Merge Information\n")
        f.write("=" * 60 + "\n\n")
        
        merge_num = 1
        skip_num = 1
        
        for m in merge_metadata:
            if m.reason.startswith('skipped:'):
                f.write(f"Skipped Merge #{skip_num}: {m.merge_type}\n")
                f.write(f"  Sentences: {m.sentence1_idx} + {m.sentence2_idx}\n")
                f.write(f"  Reason: {m.reason}\n")
                f.write(f"  Before (sent1): \"{m.before_text1}\"\n")
                f.write(f"  Before (sent2): \"{m.before_text2}\"\n")
                if m.speaker1 or m.speaker2:
                    f.write(f"  Speakers: {m.speaker1} vs {m.speaker2}\n")
                f.write("\n")
                skip_num += 1
            else:
                f.write(f"Merge #{merge_num}: {m.merge_type}\n")
                f.write(f"  Sentences: {m.sentence1_idx} + {m.sentence2_idx}\n")
                f.write(f"  Reason: {m.reason}\n")
                f.write(f"  Before (sent1): \"{m.before_text1}\"\n")
                f.write(f"  Before (sent2): \"{m.before_text2}\"\n")
                f.write(f"  After: \"{m.after_text}\"\n")
                if m.speaker1 or m.speaker2:
                    speaker_info = f"{m.speaker1}" if m.speaker1 == m.speaker2 else f"{m.speaker1} + {m.speaker2}"
                    f.write(f"  Speakers: {speaker_info}\n")
                f.write("\n")
                merge_num += 1


def _write_txt(sentences, output_file, language: str | None = None):
    """
    Write sentences to TXT file with speaker-aware paragraph breaks.
    
    v0.6.0: Splits utterances from different speakers within the same sentence
            into separate paragraphs. Ensures proper capitalization.
    v0.4.0: Simplified - no longer splits sentences since SentenceSplitter handles all boundaries.
    
    Args:
        sentences: List of Sentence objects (or strings for backward compat)
        output_file: Path to output file
        language: Language code for domain fixing
    """
    def _capitalize_first_letter(text: str) -> str:
        """Capitalize the first letter of a sentence if it's lowercase, handling special chars."""
        if not text:
            return text
        # Find the first alphabetic character
        for i, char in enumerate(text):
            if char.isalpha():
                # Only capitalize if it's currently lowercase
                if char.islower():
                    return text[:i] + char.upper() + text[i+1:]
                else:
                    # Already capitalized, return as-is
                    return text
        return text
    
    def _fix_mid_sentence_capitals(text: str) -> str:
        """
        Fix incorrectly capitalized words mid-sentence.
        The punctuation restorer sometimes capitalizes common words (connectors, articles) mid-sentence.
        Examples: 
          - "best Y aquí" -> "best y aquí"
          - "best. Y aquí" -> "best. y aquí"
        
        Strategy: Lowercase common Spanish words when preceded by space/punctuation (not hyphens).
        _capitalize_first_letter() will re-capitalize the first word if it's truly at sentence start.
        """
        import re
        # Common Spanish connectors and articles that are usually lowercase unless starting a sentence
        common_words = ['Y', 'E', 'O', 'U', 'A', 'De', 'En', 'Por', 'Para', 'Con', 'Sin', 
                       'Sobre', 'Entre', 'Pero', 'Ni', 'Mas', 'Sino', 'Desde', 'Hasta', 'Hacia',
                       'La', 'El', 'Los', 'Las', 'Un', 'Una', 'Unos', 'Unas', 
                       'Aquí', 'Ahí', 'Allí', 'También', 'Todo', 'Todos', 'Toda', 'Todas']
        
        # Lowercase these words only when they appear MID-SENTENCE (not at sentence starts)
        # After sentence-ending punctuation (.!?), words should stay capitalized as they start new sentences
        # This prevents lowercasing letters in acronyms like "B-E-S-T"
        for word in common_words:
            # Pattern 1: After mid-sentence punctuation (,;:) - always lowercase
            text = re.sub(r'([,;:])(\s*)' + word + r'\b', r'\1\2' + word.lower(), text)
            # Pattern 2: After space NOT preceded by sentence-ending punctuation (.!?)
            # This handles mid-sentence cases like "café Y el agua" → "café y el agua"
            # But preserves "dos. En vez" (En stays capitalized as sentence start)
            text = re.sub(r'(?<![.!?])(\s)' + word + r'\b', r'\1' + word.lower(), text)
        
        return text
    
    with open(output_file, "w") as f:
        prev_speaker = None
        sentences_with_speaker_changes = 0
        
        for sentence_obj in sentences:
            # Handle backward compatibility with string sentences
            if not isinstance(sentence_obj, Sentence):
                s = (sentence_obj or "").strip()
                if s:
                    # SAFETY NET: Ensure space after sentence-ending punctuation
                    # (This may incorrectly add spaces to domains, but fix_spaced_domains() will fix them)
                    s = re.sub(r'([.!?])([A-ZÁÉÍÓÚÑa-záéíóúñ¿¡])', r'\1 \2', s)
                    # Fix domains AFTER safety net (removes incorrectly added spaces from domains)
                    s = fix_spaced_domains(s, use_exclusions=True, language=language)
                    s = _fix_mid_sentence_capitals(s)
                    s = _capitalize_first_letter(s)
                    f.write(f"{s}\n\n")
                continue
            
            # Check if this sentence contains multiple speakers
            if sentence_obj.has_speaker_changes():
                sentences_with_speaker_changes += 1
                
                # Merge consecutive utterances from the same speaker, then split by speaker changes
                merged_utterances = []
                current_utterance = None
                
                for utterance in sentence_obj.utterances:
                    if not utterance.text or not utterance.text.strip():
                        continue
                    
                    if current_utterance is None:
                        current_utterance = {
                            'text': utterance.text,
                            'speaker': utterance.speaker
                        }
                    elif current_utterance['speaker'] == utterance.speaker:
                        # Same speaker - merge
                        current_utterance['text'] += ' ' + utterance.text
                    else:
                        # Different speaker - save current and start new
                        merged_utterances.append(current_utterance)
                        current_utterance = {
                            'text': utterance.text,
                            'speaker': utterance.speaker
                        }
                
                # Don't forget the last one
                if current_utterance:
                    merged_utterances.append(current_utterance)
                
                # FIX (v0.6.0): Only split if all utterances are substantial (≥3 words)
                # This prevents excessive fragmentation from short interjections
                MIN_UTTERANCE_WORDS = 3
                all_substantial = all(
                    len(u['text'].split()) >= MIN_UTTERANCE_WORDS 
                    for u in merged_utterances
                )
                
                if all_substantial and len(merged_utterances) > 1:
                    # All utterances are substantial - split them
                    for idx, merged in enumerate(merged_utterances):
                        text = merged['text'].strip()
                        if text:
                            # SAFETY NET: Ensure space after sentence-ending punctuation
                            text = re.sub(r'([.!?])([A-ZÁÉÍÓÚÑa-záéíóúñ¿¡])', r'\1 \2', text)
                            # Fix domains AFTER safety net (removes incorrectly added spaces)
                            text = fix_spaced_domains(text, use_exclusions=True, language=language)
                            text = _fix_mid_sentence_capitals(text)
                            # Capitalize each utterance since they become separate paragraphs
                            text = _capitalize_first_letter(text)
                            f.write(f"{text}\n\n")
                            prev_speaker = merged['speaker']
                else:
                    # Some utterances are too short - keep sentence together
                    full_text = sentence_obj.text.strip()
                    if full_text:
                        # SAFETY NET: Ensure space after sentence-ending punctuation
                        full_text = re.sub(r'([.!?])([A-ZÁÉÍÓÚÑa-záéíóúñ¿¡])', r'\1 \2', full_text)
                        # Fix domains AFTER safety net (removes incorrectly added spaces)
                        full_text = fix_spaced_domains(full_text, use_exclusions=True, language=language)
                        full_text = _fix_mid_sentence_capitals(full_text)
                        full_text = _capitalize_first_letter(full_text)
                        f.write(f"{full_text}\n\n")
                        prev_speaker = sentence_obj.get_first_speaker()
            else:
                # Single speaker sentence - write as one paragraph
                s = (sentence_obj.text or "").strip()
                if not s:
                    continue
                
                # SAFETY NET: Ensure space after sentence-ending punctuation
                # This catches any concatenations that slipped through earlier stages
                s = re.sub(r'([.!?])([A-ZÁÉÍÓÚÑa-záéíóúñ¿¡])', r'\1 \2', s)
                
                # Fix domains AFTER safety net (removes incorrectly added spaces from domains)
                s = fix_spaced_domains(s, use_exclusions=True, language=language)
                s = _fix_mid_sentence_capitals(s)
                s = _capitalize_first_letter(s)
                f.write(f"{s}\n\n")
                prev_speaker = sentence_obj.get_first_speaker()
        
        logger.debug(f"Found {sentences_with_speaker_changes} sentences with speaker changes")

def _write_srt(segments, output_file):
    def format_timestamp(seconds):
        h = int(seconds // 3600); m = int((seconds % 3600) // 60); s = int(seconds % 60); ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    with open(output_file, "w") as f:
        for i, seg in enumerate(segments, 1):
            start = format_timestamp(seg['start']); end = format_timestamp(seg['end']); text = seg['text'].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

def _write_raw(segments, output_file, detected_language: str | None = None, task: str = "transcribe"):
    """Write raw transcription data for debugging purposes."""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Language: {detected_language or 'unknown'}\n")
        f.write(f"Task: {task}\n")
        f.write(f"Number of segments: {len(segments)}\n")
        f.write("=" * 50 + "\n\n")
        
        for i, segment in enumerate(segments, 1):
            start = segment.get('start', 0.0)
            end = segment.get('end', 0.0)
            text = segment.get('text', '').strip()
            f.write(f"Segment {i}: {start:.2f}s - {end:.2f}s\n")
            f.write(f"Text: {text}\n\n")

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
    # Join segments with single newlines to preserve boundaries while allowing sentence assembly
    # Single newlines allow the punctuation logic to see segment boundaries and merge intelligently
    text = "\n".join(d["text"].strip() for d in deduped)
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


def _convert_speaker_timestamps_to_char_positions(
    speaker_boundaries: list[float],
    whisper_segments: list[dict],
    text: str
) -> list[int]:
    """
    Convert speaker boundary timestamps (seconds) to character positions in the text.
    
    Speaker diarization returns timestamps when speakers change. To use these for
    sentence splitting, we need to map them to character positions in the concatenated
    text, which is what the punctuation restorer uses.
    
    For each speaker boundary timestamp:
    1. Find the Whisper segment whose time range contains or is closest to the boundary
    2. Map that to the character position where that segment ends in the text
    
    Args:
        speaker_boundaries: List of timestamps (seconds) where speakers change
        whisper_segments: List of Whisper segment dicts with 'start', 'end', 'text' fields
        text: The full concatenated text from all segments
        
    Returns:
        List of character positions corresponding to speaker boundaries
    """
    if not speaker_boundaries or not whisper_segments:
        return []
    
    # Build a mapping of segment index to character end position
    # This mirrors the logic in _extract_segment_boundaries from punctuation_restorer.py
    segment_char_positions = []
    position = 0
    for seg in whisper_segments:
        seg_text = seg.get('text', '').strip()
        if not seg_text:
            continue
        position += len(seg_text)
        segment_char_positions.append({
            'start': seg.get('start', 0),
            'end': seg.get('end', 0),
            'char_end': position,
            'text': seg_text
        })
        position += 1  # Account for separator (space or newline)
    
    if not segment_char_positions:
        return []
    
    # For each speaker boundary, find the best matching segment
    char_positions = []
    for boundary_time in speaker_boundaries:
        best_segment = None
        min_distance = float('inf')
        segment_idx = -1
        
        for idx, seg_info in enumerate(segment_char_positions):
            seg_start = seg_info['start']
            seg_end = seg_info['end']
            
            # Check if boundary falls within this segment
            if seg_start <= boundary_time <= seg_end:
                best_segment = seg_info
                segment_idx = idx
                break
            
            # Otherwise, find the segment whose end is closest to (but not after) the boundary
            # This handles cases where the speaker change happens in a gap between segments
            if seg_end <= boundary_time:
                distance = boundary_time - seg_end
                if distance < min_distance:
                    min_distance = distance
                    best_segment = seg_info
                    segment_idx = idx
        
        # If no segment ends before the boundary, use the first segment after it
        if best_segment is None:
            for idx, seg_info in enumerate(segment_char_positions):
                if seg_info['start'] >= boundary_time:
                    best_segment = seg_info
                    segment_idx = idx
                    break
        
        if best_segment:
            char_positions.append(best_segment['char_end'])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_positions = []
    for pos in char_positions:
        if pos not in seen:
            seen.add(pos)
            unique_positions.append(pos)
    
    return unique_positions


def _convert_speaker_segments_to_char_ranges(
    speaker_segments: list[dict],
    whisper_segments: list[dict],
    text: str
) -> list[dict]:
    """
    Convert speaker segments (with start/end times and speaker labels) to character ranges.
    
    Speaker diarization returns segments with time ranges and speaker labels.
    To use this for sentence splitting, we need to map these to character positions
    in the concatenated text.
    
    Args:
        speaker_segments: List of speaker segment dicts with 'start', 'end', 'speaker' fields
        whisper_segments: List of Whisper segment dicts with 'start', 'end', 'text' fields
        text: The full concatenated text from all segments
        
    Returns:
        List of dicts with:
        - start_char: character position where speaker segment starts
        - end_char: character position where speaker segment ends
        - speaker: speaker label
    """
    if not speaker_segments or not whisper_segments:
        return []
    
    # Build a mapping of Whisper segments to character positions
    # This mirrors the logic in _convert_speaker_timestamps_to_char_positions
    # CRITICAL FIX (v0.5.1): Must normalize segment text the same way all_text was normalized
    # Otherwise character positions won't align with the normalized text
    from punctuation_restorer import _normalize_initials_and_acronyms
    
    whisper_char_positions = []
    position = 0
    for idx, seg in enumerate(whisper_segments):
        seg_text = seg.get('text', '').strip()
        if not seg_text:
            continue
        
        # CRITICAL: Apply EXACT same normalizations that were applied to all_text
        # 1. Normalize initials and acronyms (e.g., "C. S. Lewis" -> "C.S. Lewis")
        seg_text = _normalize_initials_and_acronyms(seg_text)
        # 2. Normalize whitespace (multiple spaces/tabs/newlines -> single space)
        import re
        seg_text = re.sub(r'\s+', ' ', seg_text.strip())
        
        char_start = position
        position += len(seg_text)
        whisper_char_positions.append({
            'start': seg.get('start', 0),
            'end': seg.get('end', 0),
            'char_start': char_start,
            'char_end': position,
            'text': seg_text,
            'whisper_idx': idx
        })
        position += 1  # Account for separator (space or newline)
    
    if not whisper_char_positions:
        return []
    
    logger.debug(f"Built {len(whisper_char_positions)} Whisper char positions, total text length: {position}")
    logger.debug(f"Provided text length: {len(text)}, calculated position: {position}")
    if abs(len(text) - position) > 10:
        logger.warning(f"Text length mismatch! Provided: {len(text)}, calculated: {position}, diff: {len(text) - position}")
    
    # FIX (v0.5.2): Split Whisper segments when they contain multiple speakers
    # Previous approach assigned each Whisper segment to ONE speaker (most overlap),
    # losing boundaries that fell in the middle of segments.
    # New approach: detect multi-speaker segments and split them proportionally.
    
    # FIX (v0.6.0): Filter out very short speaker segments (likely diarization errors)
    # Very short segments are often artifacts and create spurious speaker changes
    # v0.6.0.1: Increased from 0.5s to 1.3s to filter out rapid speaker flipping (e.g., 0.56s, 0.93s, 1.10s, 1.28s artifacts)
    MIN_SEGMENT_DURATION = 1.3
    filtered_speaker_segments = []
    for spk_seg in speaker_segments:
        duration = spk_seg.get('end', 0) - spk_seg.get('start', 0)
        if duration >= MIN_SEGMENT_DURATION:
            filtered_speaker_segments.append(spk_seg)
        else:
            logger.debug(f"Filtered out short speaker segment: {duration:.2f}s < {MIN_SEGMENT_DURATION}s")
    
    logger.debug(f"Filtered speaker segments: {len(speaker_segments)} -> {len(filtered_speaker_segments)}")
    speaker_segments = filtered_speaker_segments
    
    speaker_char_ranges = []
    
    for whisp_info in whisper_char_positions:
        whisp_idx = whisp_info['whisper_idx']
        whisp_start = whisp_info['start']
        whisp_end = whisp_info['end']
        whisp_duration = whisp_end - whisp_start
        whisp_text = whisp_info['text']
        whisp_char_start = whisp_info['char_start']
        whisp_char_end = whisp_info['char_end']
        whisp_char_len = whisp_char_end - whisp_char_start
        
        # Find ALL speaker segments that overlap this Whisper segment
        overlapping_speakers = []
        for spk_seg in speaker_segments:
            spk_start = spk_seg.get('start', 0)
            spk_end = spk_seg.get('end', 0)
            spk_label = spk_seg.get('speaker', 'UNKNOWN')
            
            # Calculate overlap
            overlap_start = max(whisp_start, spk_start)
            overlap_end = min(whisp_end, spk_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            # Skip if no overlap or overlap is too short (likely noise/gap)
            # FIX (v0.5.2.1): Check OVERLAP duration, not total segment duration
            # A 0.46s segment is valid if it overlaps 0.4s with this Whisper segment
            if overlap_duration < 0.3:
                continue
            
            overlapping_speakers.append({
                'speaker': spk_label,
                'spk_start': spk_start,
                'spk_end': spk_end,
                'overlap_start': overlap_start,
                'overlap_end': overlap_end,
                'overlap_duration': overlap_duration
            })
        
        if not overlapping_speakers:
            # No speaker assigned - skip this segment
            continue
        
        # Sort by overlap start time to process in chronological order
        overlapping_speakers.sort(key=lambda s: s['overlap_start'])
        
        if len(overlapping_speakers) == 1:
            # Simple case: entire Whisper segment belongs to one speaker
            speaker_char_ranges.append({
                'start_char': whisp_char_start,
                'end_char': whisp_char_end,
                'speaker': overlapping_speakers[0]['speaker']
            })
            logger.debug(f"Whisper seg {whisp_idx} [{whisp_start:.2f}s-{whisp_end:.2f}s]: single speaker {overlapping_speakers[0]['speaker']}")
        else:
            # Complex case: Whisper segment contains MULTIPLE speakers
            # Check if one speaker dominates (>80%) AND the minor speaker is at the EDGE
            # This handles cases where pyannote misattributes a few words at segment boundaries
            # But we preserve short utterances in the MIDDLE (e.g., "Ok." between two longer segments)
            dominant_speaker = None
            max_duration = 0
            total_duration = sum(s['overlap_duration'] for s in overlapping_speakers)
            
            for spk_info in overlapping_speakers:
                if spk_info['overlap_duration'] > max_duration:
                    max_duration = spk_info['overlap_duration']
                    dominant_speaker = spk_info['speaker']
            
            # Check if dominant speaker threshold applies
            # Only filter if one speaker dominates AND minor speaker is at the edge (not middle)
            dominant_ratio = max_duration / total_duration
            is_edge_only = False
            
            if dominant_ratio > 0.8 and len(overlapping_speakers) == 2:
                # Find the minor speaker
                minor_speakers = [s for s in overlapping_speakers if s['speaker'] != dominant_speaker]
                if len(minor_speakers) > 0:
                    minor_speaker = minor_speakers[0]
                    
                    # Check if minor speaker is only at the edge (beginning or end, not middle)
                    # Minor at beginning: overlap_start ≈ whisp_start
                    # Minor at end: overlap_end ≈ whisp_end
                    edge_threshold = 0.1 * whisp_duration  # Within first/last 10% of segment
                    
                    at_beginning = abs(minor_speaker['overlap_start'] - whisp_start) < edge_threshold
                    at_end = abs(minor_speaker['overlap_end'] - whisp_end) < edge_threshold
                    
                    is_edge_only = at_beginning or at_end
            
            if dominant_ratio > 0.8 and is_edge_only:
                # One speaker dominates and minor is at edge - likely diarization error
                speaker_char_ranges.append({
                    'start_char': whisp_char_start,
                    'end_char': whisp_char_end,
                    'speaker': dominant_speaker
                })
                logger.debug(f"Whisper seg {whisp_idx} [{whisp_start:.2f}s-{whisp_end:.2f}s]: dominant speaker {dominant_speaker} ({dominant_ratio*100:.1f}%), ignoring edge overlap")
            else:
                # Multiple significant speakers OR minor speaker in middle - split the text proportionally
                logger.debug(f"Whisper seg {whisp_idx} [{whisp_start:.2f}s-{whisp_end:.2f}s]: SPLIT across {len(overlapping_speakers)} speakers")
                
                # Calculate character split points based on time ratios
                char_position = whisp_char_start
                for i, spk_info in enumerate(overlapping_speakers):
                    # Calculate what fraction of the Whisper segment this speaker occupies
                    time_ratio = spk_info['overlap_duration'] / whisp_duration
                    
                    # Calculate character range for this speaker
                    if i == len(overlapping_speakers) - 1:
                        # Last speaker gets remaining characters to avoid rounding errors
                        char_end = whisp_char_end
                    else:
                        # Proportional split based on time
                        char_end = whisp_char_start + int(whisp_char_len * 
                                                          (spk_info['overlap_end'] - whisp_start) / whisp_duration)
                        
                        # Try to split at word boundary for cleaner splits
                        # Look for space near the calculated split point (within ±20% of segment)
                        search_window = max(5, int(whisp_char_len * 0.2))
                        search_start = max(whisp_char_start, char_end - search_window)
                        search_end = min(whisp_char_end, char_end + search_window)
                        
                        # Find nearest word boundary (space) within the window
                        best_split = char_end
                        min_distance = float('inf')
                        for pos in range(search_start, search_end):
                            if pos < len(text) and text[pos] == ' ':
                                distance = abs(pos - char_end)
                                if distance < min_distance:
                                    min_distance = distance
                                    best_split = pos
                        
                        char_end = best_split
                    
                    # Only create range if it has content
                    if char_end > char_position:
                        speaker_char_ranges.append({
                            'start_char': char_position,
                            'end_char': char_end,
                            'speaker': spk_info['speaker']
                        })
                        logger.debug(f"  → {spk_info['speaker']}: chars {char_position}-{char_end} (time ratio: {time_ratio:.2f})")
                        char_position = char_end
    
    # Sort by character position
    speaker_char_ranges.sort(key=lambda r: r['start_char'])
    
    # Merge consecutive ranges from the same speaker
    # This consolidates ranges that may have been split across Whisper segments
    if speaker_char_ranges:
        merged_ranges = []
        current_range = speaker_char_ranges[0].copy()
        
        for next_range in speaker_char_ranges[1:]:
            if (next_range['speaker'] == current_range['speaker'] and 
                next_range['start_char'] <= current_range['end_char'] + 1):
                # Same speaker and adjacent/overlapping - merge
                current_range['end_char'] = max(current_range['end_char'], next_range['end_char'])
            else:
                # Different speaker or gap - save current and start new
                merged_ranges.append(current_range)
                current_range = next_range.copy()
        
        # Don't forget the last range
        merged_ranges.append(current_range)
        speaker_char_ranges = merged_ranges
    
    logger.info(f"Created {len(speaker_char_ranges)} speaker character ranges from {len(speaker_segments)} diarization segments")
    
    return speaker_char_ranges


def _assemble_sentences(all_text: str, all_segments: list[dict], lang_for_punctuation: str | None, quiet: bool, speaker_boundaries: list[float] | None = None, speaker_segments: list[dict] | None = None) -> tuple[list[str], list]:
    def _sanitize_sentence_output(s: str, language: str) -> str:
        try:
            if (language or '').lower() != 'es' or not s:
                return s
            
            
            # Use centralized domain masking with Spanish exclusions
            out = mask_domains(s, use_exclusions=True, language=language)
            
            # Fix missing space after ., ?, ! (avoid ellipses, decimals, and already-masked domains)
            # Avoid breaking domains by not adding spaces after periods in domain-like patterns
            # Don't add space after periods that are part of domain patterns (www., ftp., etc.)
            out = re.sub(r"(?<!www)(?<!ftp)(?<!mail)(?<!blog)(?<!shop)(?<!app)(?<!api)(?<!cdn)(?<!static)(?<!news)(?<!support)(?<!help)(?<!docs)(?<!admin)(?<!secure)(?<!login)(?<!mobile)(?<!store)(?<!sub)(?<!dev)(?<!test)(?<!staging)(?<!prod)(?<!beta)(?<!alpha)\.([A-ZÁÉÍÓÚÑ¿¡])", r". \1", out)  # Add space before capital letters
            # Only add space after periods that are likely sentence terminators (not domain patterns)
            out = re.sub(r"(?<!www)(?<!ftp)(?<!mail)(?<!blog)(?<!shop)(?<!app)(?<!api)(?<!cdn)(?<!static)(?<!news)(?<!support)(?<!help)(?<!docs)(?<!admin)(?<!secure)(?<!login)(?<!mobile)(?<!store)(?<!sub)(?<!dev)(?<!test)(?<!staging)(?<!prod)(?<!beta)(?<!alpha)\.([a-záéíóúñ])", r". \1", out)
            out = re.sub(r"\?\s*(\S)", r"? \1", out)
            out = re.sub(r"!\s*(\S)", r"! \1", out)
            # Capitalize after terminators when appropriate
            out = re.sub(r"([.!?])\s+([a-záéíóúñ])", lambda m: f"{m.group(1)} {m.group(2).upper()}", out)
            # Normalize comma spacing using centralized function
            out = _normalize_comma_spacing(out)
            # Replace stray intra-word periods between lowercase letters: "vendedores.ambulantes" -> "vendedores ambulantes"
            out = re.sub(r"([a-záéíóúñ])\.(?=[a-záéíóúñ])", r"\1 ", out)
            # Tighten percent formatting: keep number and % together
            out = re.sub(r"(\d)\s+%", r"\1%", out)
            
            # Fix capitalization for words that were capitalized at segment boundaries
            # but are now mid-sentence after punctuation restoration (e.g., "independencia, Ojalá" -> "independencia, ojalá")
            # Use algorithmic approach to determine when capitalized words should be lowercased
            def _should_lowercase_mid_sentence_word(word: str, context_before: str, context_after: str) -> bool:
                """Determine if a capitalized word should be lowercased based on linguistic patterns."""
                word_lower = word.lower()
                
                
                # Never lowercase single letters (could be initials) unless specific patterns
                if len(word) == 1:
                    # Special case for "A veces" pattern
                    return word_lower == 'a' and context_after.strip().startswith('veces')
                
                # Strong indicators this is a proper noun (should stay capitalized)
                proper_noun_indicators = [
                    # Followed by "de" (suggesting location: "Madrid de...")
                    context_after.strip().startswith('de '),
                    # Preceded by "en" or "a" (suggesting location: "en Madrid", "a París")
                    context_before.strip().endswith(' en') or context_before.strip().endswith(' a'),
                    # Two consecutive capitalized words (proper noun phrase)
                    re.match(r'^\s*[A-Z][a-z]+', context_after),
                    # Looks like a common proper name pattern and isn't in common words list
                    (re.match(r'^[A-Z][a-z]{3,}$', word) and 
                     word_lower not in {
                         'ojalá', 'entonces', 'pero', 'también', 'además', 'ahora', 'después', 
                         'antes', 'luego', 'finalmente', 'mientras', 'cuando', 'donde', 'aunque', 
                         'porque', 'algunos', 'algunas', 'otro', 'otra', 'episodio', 'capítulo',
                         'temporada', 'parte', 'sección', 'tema', 'momento', 'tiempo'
                     }),
                ]
                
                if any(proper_noun_indicators):
                    return False
                
                # Strong indicators this is a common word (should be lowercased)
                common_word_indicators = [
                    # Known Spanish adverbs/conjunctions that are commonly capitalized incorrectly
                    word_lower in {
                        'ojalá', 'entonces', 'pero', 'también', 'además', 'sin embargo',
                        'por ejemplo', 'es decir', 'por tanto', 'aunque', 'mientras',
                        'cuando', 'donde', 'como', 'porque', 'para que', 'si', 'que',
                        'ahora', 'después', 'antes', 'luego', 'finalmente', 'primero',
                        'segundo', 'tercero', 'último', 'otro', 'otra', 'algunos', 'algunas',
                        'además', 'incluso', 'sobre todo', 'en realidad', 'de hecho',
                        'episodio', 'capítulo', 'temporada', 'parte', 'sección', 'tema', 'momento', 'tiempo'
                    },
                    # Spanish verb forms (unlikely to be proper nouns)
                    re.match(r'^[a-z]+(ar|er|ir)(me|te|se|nos|os)?$', word_lower),  # infinitives
                    re.match(r'^[a-z]+(ando|iendo)$', word_lower),  # gerunds
                    re.match(r'^[a-z]+(ado|ido)$', word_lower),  # past participles
                    re.match(r'^[a-z]+(aba|ía|ará|ería)s?$', word_lower),  # conjugated forms
                    # Spanish adjective/noun patterns
                    word_lower.endswith(('mente', 'ción', 'sión', 'dad', 'tad', 'eza', 'anza')),
                    # Starts with lowercase article/preposition pattern (wrong split)
                    word_lower.startswith(('de', 'el', 'la', 'los', 'las', 'un', 'una')),
                ]
                
                if any(common_word_indicators):
                    return True
                
                # Additional check for common short words that could be ambiguous
                ambiguous_short_words = {'veces', 'forma', 'parte', 'manera', 'tiempo', 'caso', 'lugar', 'momento'}
                if word_lower in ambiguous_short_words:
                    return True
                
                # Default: if uncertain and it's a short word (≤6 chars), lowercase it
                # This catches common words while preserving longer proper nouns
                return len(word) <= 6
            
            def _replace_mid_sentence_caps(match):
                punctuation = match.group(1)  # comma or semicolon
                space = match.group(2)        # space(s)
                word = match.group(3)         # capitalized word
                
                # Get context for decision making
                start_pos = match.start()
                end_pos = match.end()
                context_before = out[:start_pos + len(punctuation)]
                context_after = out[end_pos:]
                
                if _should_lowercase_mid_sentence_word(word, context_before, context_after):
                    return punctuation + space + word[0].lower() + word[1:]
                else:
                    return match.group(0)  # no change
            
            # Apply the pattern: comma/semicolon + space + capitalized word
            out = re.sub(r'([,;])(\s+)([A-ZÁÉÍÓÚÑ][a-záéíóúñ]*)', _replace_mid_sentence_caps, out)
            
            # Unmask domains using centralized function
            out = unmask_domains(out)
            return out
        except Exception:
            return s
    
    # Normalize person initials and organizational acronyms for ALL languages
    # This prevents false sentence breaks when names like "C.S. Lewis" or "J.K. Rowling"
    # appear in non-English transcriptions (e.g., Spanish podcasts discussing English authors)
    from punctuation_restorer import _normalize_initials_and_acronyms
    all_text = _normalize_initials_and_acronyms(all_text)
    
    # CRITICAL FIX (v0.4.3): Normalize whitespace BEFORE calculating speaker_word_ranges
    # This ensures word indices in speaker_word_ranges match the normalized text
    # that will be used by restore_punctuation(). Without this, speaker boundaries
    # are calculated on text with inconsistent spacing, then the text is normalized
    # inside restore_punctuation(), causing word index misalignment.
    # Bug: Speaker boundaries not triggering splits (e.g., Andrea/Nate in Episodio212)
    all_text = re.sub(r'\s+', ' ', all_text.strip())
    
    # v0.4.0: Convert speaker segments to word ranges for SentenceSplitter
    # SentenceSplitter now handles all boundary logic using full segment information
    speaker_word_ranges = None
    if speaker_segments and all_segments:
        # First convert to character ranges
        speaker_char_ranges = _convert_speaker_segments_to_char_ranges(
            speaker_segments, all_segments, all_text
        )
        if speaker_char_ranges and not quiet:
            logger.debug(f"Converted {len(speaker_segments)} speaker segments to {len(speaker_char_ranges)} char ranges")
        
        # Then convert character ranges to word indices
        if speaker_char_ranges:
            from punctuation_restorer import _convert_char_ranges_to_word_ranges
            speaker_word_ranges = _convert_char_ranges_to_word_ranges(all_text, speaker_char_ranges)
            
            if speaker_word_ranges and not quiet:
                logger.debug(f"Converted {len(speaker_char_ranges)} char ranges to {len(speaker_word_ranges)} word ranges")
                if len(speaker_word_ranges) > 0:
                    logger.debug(f"First word range: {speaker_word_ranges[0]}")
                    logger.debug(f"Last word range: {speaker_word_ranges[-1]}")
                    # Count speaker changes
                    speaker_changes = 0
                    for i in range(len(speaker_word_ranges) - 1):
                        if speaker_word_ranges[i]['speaker'] != speaker_word_ranges[i+1]['speaker']:
                            speaker_changes += 1
                    logger.info(f"Speaker word ranges contain {speaker_changes} speaker changes")
    
    # Process the entire text using unified SentenceSplitter (v0.4.0+)
    # Pass full Whisper segments and speaker segments (not just boundaries)
    # SentenceSplitter handles all boundary evaluation and punctuation tracking
    from punctuation_restorer import restore_punctuation
    
    if not quiet and speaker_word_ranges:
        logger.debug(f"Calling restore_punctuation with {len(speaker_word_ranges)} speaker word ranges")
        logger.debug(f"Total words in all_text: {len(all_text.split())}")
    
    processed_text, pre_split_sentences = restore_punctuation(
        all_text, 
        lang_for_punctuation, 
        whisper_segments=all_segments,  # Pass full segments, not boundaries
        speaker_segments=speaker_word_ranges  # Word-indexed segments for speaker tracking
    )
    
    # v0.4.0: restore_punctuation() ALWAYS returns sentences (never None)
    # SentenceSplitter has already handled all boundary decisions
    sentences = pre_split_sentences if pre_split_sentences else [processed_text]
    
    if not quiet:
        logger.debug(f"restore_punctuation returned {len(sentences)} sentences")
    
    # v0.5.1 FIX: Apply formatting BEFORE sanitization
    # The sanitization process can change word counts (e.g., splitting "vendedores.ambulantes" into two words),
    # which would cause word position misalignment with speaker_word_ranges.
    # So we need to do speaker-aware formatting first, then sanitize after.
    from sentence_formatter import SentenceFormatter
    
    if not quiet:
        logger.debug(f"About to call SentenceFormatter with {len(sentences)} sentences")
        logger.debug(f"Speaker word ranges: {speaker_word_ranges is not None}")
        if speaker_word_ranges:
            logger.debug(f"Number of speaker word ranges: {len(speaker_word_ranges)}")
    
    formatter = SentenceFormatter(
        language=lang_for_punctuation or 'en',
        speaker_segments=speaker_word_ranges  # Use original word ranges before sanitization
    )
    sentences, merge_metadata = formatter.format(sentences)
    
    if not quiet:
        logger.debug(f"SentenceFormatter returned {len(sentences)} sentences and {len(merge_metadata)} merge operations")
    
    # Sanitize all sentences AFTER formatting
    sentences = [_sanitize_sentence_output(s, (lang_for_punctuation or '').lower()) for s in sentences]
    
    # Log merge summary
    if merge_metadata and not quiet:
        merge_counts = {}
        skipped_counts = {}
        for m in merge_metadata:
            if m.reason.startswith('skipped:'):
                skipped_counts[m.merge_type] = skipped_counts.get(m.merge_type, 0) + 1
            else:
                merge_counts[m.merge_type] = merge_counts.get(m.merge_type, 0) + 1
        
        if merge_counts:
            summary = ', '.join(f"{count} {mtype}" for mtype, count in merge_counts.items())
            logger.info(f"Applied merges: {summary}")
        if skipped_counts:
            summary = ', '.join(f"{count} {mtype}" for mtype, count in skipped_counts.items())
            logger.debug(f"Skipped merges (speaker boundaries): {summary}")
    
    # v0.5.0: Removed additional comprehensive domain merge loops
    # All domain merging is now handled by SentenceFormatter above
    
    if (lang_for_punctuation or '').lower() == 'fr' and sentences:
        # Already handled inside assemble_sentences_from_processed per segment; kept for safety
        pass
    
    # Final pass: Apply capitalization correction to all sentences
    # This catches any capitalization issues that arise from segment merging
    if sentences and (lang_for_punctuation or '').lower() == 'es':
        final_sentences = []
        for sentence in sentences:
            final_sentences.append(_sanitize_sentence_output(sentence, (lang_for_punctuation or '').lower()))
        sentences = final_sentences
    
    # v0.5.0: Return both sentences and merge metadata
    return sentences, merge_metadata
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
    enable_diarization: bool = False,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    hf_token: str | None = None,
    dump_merge_metadata: bool = False,
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
    
    # Perform speaker diarization if enabled
    speaker_boundaries = None
    diarization_result = None  # Store for potential dump
    if enable_diarization:
        if not quiet:
            logger.info("Performing speaker diarization...")
        try:
            from speaker_diarization import diarize_audio
            # Get HF token from environment if not provided
            token = hf_token or os.environ.get("HF_TOKEN")
            diarization_result = diarize_audio(
                media_file,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                token=token,
                device=device,
            )
            speaker_boundaries = diarization_result['speaker_boundaries']
        except Exception as e:
            logger.warning(f"Speaker diarization failed: {e}. Continuing without speaker boundaries.")
            if not quiet:
                logger.info("Tip: For first-time use, provide --hf-token or set HF_TOKEN environment variable")
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
        return {
            "segments": all_segments,
            "sentences": [],
            "detected_language": detected_language,
            "output_path": output_path,
            "num_segments": len(all_segments),
            "elapsed_secs": round(time.time() - _t0, 3),
            "diarization_result": diarization_result,
            "whisper_boundaries": None,
            "merged_boundaries": None,
        }
    else:
        if not quiet:
            logger.info("Restoring punctuation...")
        lang_for_punctuation = 'en' if translate_to_english else (detected_language if language is None else language)
        
        # Extract Whisper boundaries for debugging
        from punctuation_restorer import _extract_segment_boundaries
        whisper_boundaries_for_dump = _extract_segment_boundaries(all_text, all_segments) if all_segments else None

        # Compute merged boundaries for debugging
        # Note: Both whisper_boundaries and speaker boundaries are converted to character positions
        # before merging (same as in _assemble_sentences)
        merged_boundaries_for_dump = whisper_boundaries_for_dump
        if speaker_boundaries and all_segments:
            speaker_char_positions = _convert_speaker_timestamps_to_char_positions(
                speaker_boundaries, all_segments, all_text
            )
            if speaker_char_positions:
                merged_boundaries_for_dump = sorted(set((whisper_boundaries_for_dump or []) + speaker_char_positions))
            elif whisper_boundaries_for_dump:
                merged_boundaries_for_dump = whisper_boundaries_for_dump
        
        # Extract speaker segments if diarization was performed
        speaker_segments_list = None
        if diarization_result:
            speaker_segments_list = diarization_result['segments']
        
        sentences, merge_metadata = _assemble_sentences(all_text, all_segments, lang_for_punctuation, quiet, speaker_boundaries=speaker_boundaries, speaker_segments=speaker_segments_list)
        all_segments = sorted(all_segments, key=lambda d: d["start"]) if all_segments else []

        output_path_txt: str | None = None
        if write_output and out_dir is not None:
            output_file = Path(out_dir) / f"{base_name}.txt"
            try:
                # v0.4.0: No longer need skip_resplit - SentenceSplitter handles all boundaries
                _write_txt(sentences, str(output_file), language=lang_for_punctuation)
            except Exception as e:
                raise OutputWriteError(f"Failed to write TXT: {e}")
            output_path_txt = str(output_file)
            
            # v0.5.0: Write merge metadata dump if requested
            if dump_merge_metadata and merge_metadata:
                merge_dump_file = Path(out_dir) / f"{base_name}_merges.txt"
                try:
                    _write_merge_metadata_dump(merge_metadata, str(merge_dump_file))
                    if not quiet:
                        logger.info(f"Merge metadata dump written to: {merge_dump_file}")
                except Exception as e:
                    logger.warning(f"Failed to write merge metadata dump: {e}")
        return {
            "segments": all_segments,
            "sentences": sentences,
            "detected_language": detected_language,
            "output_path": output_path_txt,
            "num_segments": len(all_segments),
            "elapsed_secs": round(time.time() - _t0, 3),
            "diarization_result": diarization_result,
            "whisper_boundaries": whisper_boundaries_for_dump,
            "merged_boundaries": merged_boundaries_for_dump,
        }

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
    parser.add_argument("--beam-size", dest="beam_size", type=int, default=DEFAULT_BEAM_SIZE, help="Beam size for decoding (default: 3)")
    # VAD controls
    parser.add_argument("--no-vad", dest="vad_filter", action="store_false", help="Disable VAD filtering (default: enabled)")
    parser.add_argument("--vad-speech-pad-ms", dest="vad_speech_pad_ms", type=int, default=DEFAULT_VAD_SPEECH_PAD_MS, help="Padding (ms) around detected speech when VAD is enabled")
    parser.add_argument("--dump-raw", dest="dump_raw", action="store_true", help="Also write raw Whisper output for debugging (filename_raw.txt)")
    parser.add_argument("--dump-diarization", dest="dump_diarization", action="store_true", help="Write speaker diarization debug dump (filename_diarization.txt)")
    parser.add_argument("--dump-merge-metadata", dest="dump_merge_metadata", action="store_true", help="Write sentence merge metadata for debugging (filename_merges.txt)")
    # Speaker diarization controls
    parser.add_argument("--enable-diarization", dest="enable_diarization", action="store_true", help="Enable speaker diarization for improved sentence boundaries")
    parser.add_argument("--min-speakers", dest="min_speakers", type=int, default=None, help="Minimum number of speakers (optional, auto-detect if not specified)")
    parser.add_argument("--max-speakers", dest="max_speakers", type=int, default=None, help="Maximum number of speakers (optional, auto-detect if not specified)")
    parser.add_argument("--hf-token", dest="hf_token", default=None, help="Hugging Face token for pyannote models (required for first-time download)")
    vg = parser.add_mutually_exclusive_group(); vg.add_argument("--quiet", action="store_true", help="Reduce log output"); vg.add_argument("--verbose", action="store_true", help="Verbose log output (default)"); vg.add_argument("--debug", action="store_true", help="Debug log output (shows detailed sentence splitting decisions)"); parser.set_defaults(verbose=True)
    # Defaults for VAD
    parser.set_defaults(vad_filter=DEFAULT_VAD_FILTER)
    args = parser.parse_args()
    quiet = args.quiet or (not args.verbose and not args.debug)
    # Configure logging
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(handler)
    
    # Set level for all podscripter loggers
    if args.debug:
        log_level = logging.DEBUG
    elif quiet:
        log_level = logging.ERROR
    else:
        log_level = logging.INFO
    
    logger.setLevel(log_level)
    # Also set level on child loggers
    for logger_name in ['podscripter.splitter', 'podscripter.formatter']:
        logging.getLogger(logger_name).setLevel(log_level)
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
            beam_size=args.beam_size,
            quiet=quiet,
            vad_filter=args.vad_filter,
            vad_speech_pad_ms=args.vad_speech_pad_ms,
            write_output=True,
            enable_diarization=args.enable_diarization,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            hf_token=args.hf_token,
            dump_merge_metadata=args.dump_merge_metadata,
        )
        if not quiet and result.get("detected_language"):
            logger.info(f"Detected language: {result['detected_language']}")
        logger.info(f"Wrote: {result['output_path']}")
        
        # Write raw output if requested
        if args.dump_raw:
            base_name = Path(args.media_file).stem
            raw_output_file = Path(args.output_dir) / f"{base_name}_raw.txt"
            task = "translate" if args.translate else "transcribe"
            try:
                _write_raw(result['segments'], str(raw_output_file), result.get('detected_language'), task)
                logger.info(f"Raw output written to: {raw_output_file}")
            except Exception as e:
                logger.error(f"Failed to write raw output: {e}")
        
        # Write diarization dump if requested
        if args.dump_diarization:
            if result.get('diarization_result'):
                base_name = Path(args.media_file).stem
                diarization_output_file = Path(args.output_dir) / f"{base_name}_diarization.txt"
                try:
                    from speaker_diarization import write_diarization_dump
                    write_diarization_dump(
                        result['diarization_result'],
                        str(diarization_output_file),
                        merged_boundaries=result.get('merged_boundaries'),
                        whisper_boundaries=result.get('whisper_boundaries'),
                    )
                    logger.info(f"Diarization dump written to: {diarization_output_file}")
                except Exception as e:
                    logger.error(f"Failed to write diarization dump: {e}")
            else:
                logger.warning("--dump-diarization requested but diarization was not enabled. Use --enable-diarization to enable.")
        
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

