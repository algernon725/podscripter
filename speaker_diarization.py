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
Speaker diarization utilities for podscripter.

Detects speaker changes in audio and provides speaker boundaries
as hints for sentence splitting.
"""

from typing import TypedDict, Optional
import logging
import warnings
from pathlib import Path

# Suppress torchaudio warning about MPEG_LAYER_III subtype (harmless MP3 metadata issue)
warnings.filterwarnings(
    "ignore",
    message="The MPEG_LAYER_III subtype is unknown to TorchAudio",
    category=UserWarning,
    module="torchaudio"
)

# Suppress torchaudio 2.8 deprecation warning about switching to torchcodec in 2.9.
# We intentionally use torchaudio.load() with the soundfile backend to bypass
# torchcodec's MP3 chunk decoding issues. Pinned to torchaudio==2.8.0.
warnings.filterwarnings(
    "ignore",
    message="In 2.9, this function's implementation will be changed",
    category=UserWarning,
    module="torchaudio"
)

# Suppress pyannote pooling warning when processing very short audio segments
# This occurs when std() is called on tensors with only 1 sample (degrees of freedom = 0)
# It's a harmless numerical edge case that doesn't affect output quality
warnings.filterwarnings(
    "ignore",
    message=r"std\(\): degrees of freedom is <= 0",
    category=UserWarning,
)

logger = logging.getLogger("podscripter")

# Default settings
DEFAULT_DIARIZATION_DEVICE = "cpu"
SPEAKER_BOUNDARY_EPSILON_SEC = 1.0  # Merge boundaries within 1 second
MIN_SPEAKER_SEGMENT_SEC = 0.5  # Ignore very short speaker segments (< 0.5s likely noise)

# Priority weights for boundary merging
SPEAKER_BOUNDARY_PRIORITY = 10
WHISPER_BOUNDARY_PRIORITY = 5


class SpeakerSegment(TypedDict):
    """A segment of audio attributed to a single speaker."""
    start: float
    end: float
    speaker: str


class DiarizationResult(TypedDict):
    """Result of speaker diarization."""
    segments: list[SpeakerSegment]
    num_speakers: int
    speaker_boundaries: list[float]  # Speaker change timestamps
    filtered_boundaries: list[dict]  # Detailed info about filtered boundaries


class DiarizationError(Exception):
    """Raised when speaker diarization fails."""
    pass


def diarize_audio(
    media_file: str,
    *,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    token: Optional[str] = None,
    device: str = DEFAULT_DIARIZATION_DEVICE,
) -> DiarizationResult:
    """
    Perform speaker diarization on an audio file.
    
    Args:
        media_file: Path to the audio/video file
        min_speakers: Minimum number of speakers (None for auto-detect)
        max_speakers: Maximum number of speakers (None for auto-detect)
        token: Hugging Face token for model access
        device: Device to run on ("cpu" or "cuda")
    
    Returns:
        DiarizationResult with segments, speaker count, and boundaries
        
    Raises:
        DiarizationError: If diarization fails
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise DiarizationError(
            "pyannote.audio not installed. Install with: pip install pyannote.audio"
        ) from e
    
    media_path = Path(media_file)
    if not media_path.exists() or not media_path.is_file():
        raise DiarizationError(f"Media file not found: {media_file}")
    
    try:
        # Load the pre-trained pipeline (pyannote.audio 4.x, community-1)
        logger.info("Loading speaker diarization model...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=token
        )
        
        # Move to specified device
        if device == "cuda":
            import torch
            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
            else:
                logger.warning("CUDA requested but not available, using CPU")
                device = "cpu"
        
        # Configure speaker count if specified
        params = {}
        if min_speakers is not None:
            params["min_speakers"] = min_speakers
        if max_speakers is not None:
            params["max_speakers"] = max_speakers
        
        # Pre-load audio with torchaudio to bypass torchcodec's MP3 chunk
        # decoding issues (sample count mismatches with lossy formats).
        import torchaudio
        waveform, sample_rate = torchaudio.load(str(media_file))
        audio_input = {"waveform": waveform, "sample_rate": sample_rate}
        
        # Run diarization
        logger.info(f"Running speaker diarization on {media_path.name}...")
        output = pipeline(audio_input, **params)
        
        # pyannote 4.x returns an object with .speaker_diarization attribute
        diarization = output.speaker_diarization
        
        # Extract segments
        segments: list[SpeakerSegment] = []
        speakers = set()
        
        for turn, speaker in diarization:
            segments.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker)
            })
            speakers.add(str(speaker))
        
        if not segments:
            logger.warning("No speakers detected in diarization")
            return {
                "segments": [],
                "num_speakers": 0,
                "speaker_boundaries": [],
                "filtered_boundaries": []
            }
        
        # Extract speaker boundaries (where speakers change)
        boundaries, boundary_details = _extract_speaker_boundaries(segments)
        
        logger.info(f"Detected {len(speakers)} unique speakers with {len(boundaries)} speaker changes")
        
        return {
            "segments": segments,
            "num_speakers": len(speakers),
            "speaker_boundaries": boundaries,
            "filtered_boundaries": boundary_details
        }
        
    except Exception as e:
        raise DiarizationError(f"Speaker diarization failed: {e}") from e


class BoundaryInfo(TypedDict):
    """Detailed info about a potential speaker boundary."""
    timestamp: float
    from_speaker: str
    to_speaker: str
    segment_duration: float
    included: bool
    reason: str


def _extract_speaker_boundaries(segments: list[SpeakerSegment]) -> tuple[list[float], list[BoundaryInfo]]:
    """
    Extract speaker change timestamps from speaker segments.
    
    A speaker boundary occurs at the end of a segment when the next
    segment has a different speaker.
    
    Args:
        segments: List of speaker segments
        
    Returns:
        Tuple of:
        - List of timestamps where speakers change (filtered)
        - List of BoundaryInfo with details about all potential boundaries
    """
    if not segments:
        return [], []
    
    boundaries = []
    boundary_details: list[BoundaryInfo] = []
    
    # Sort segments by start time to ensure proper ordering
    sorted_segments = sorted(segments, key=lambda s: s["start"])
    
    for i in range(len(sorted_segments) - 1):
        current = sorted_segments[i]
        next_seg = sorted_segments[i + 1]
        
        # Check if speaker changes
        if current["speaker"] != next_seg["speaker"]:
            # Use the end of the current segment as the boundary
            boundary_time = current["end"]
            
            # Only add if segment is long enough (filter out very short segments)
            segment_duration = current["end"] - current["start"]
            included = segment_duration >= MIN_SPEAKER_SEGMENT_SEC
            
            reason = "included" if included else f"segment too short ({segment_duration:.2f}s < {MIN_SPEAKER_SEGMENT_SEC}s)"
            
            boundary_details.append({
                "timestamp": boundary_time,
                "from_speaker": current["speaker"],
                "to_speaker": next_seg["speaker"],
                "segment_duration": segment_duration,
                "included": included,
                "reason": reason,
            })
            
            if included:
                boundaries.append(boundary_time)
    
    return boundaries, boundary_details


def _merge_boundaries(
    whisper_boundaries: Optional[list[float]],
    speaker_boundaries: Optional[list[float]],
    epsilon: float = SPEAKER_BOUNDARY_EPSILON_SEC,
) -> list[float]:
    """
    Merge Whisper segment boundaries with speaker boundaries.
    
    Speaker boundaries get higher priority. Boundaries within epsilon
    seconds of each other are deduplicated.
    
    Args:
        whisper_boundaries: Timestamps from Whisper segment boundaries
        speaker_boundaries: Timestamps from speaker changes
        epsilon: Maximum distance (seconds) to consider boundaries as duplicates
        
    Returns:
        Merged and deduplicated list of boundary timestamps
    """
    # Handle None cases
    if whisper_boundaries is None and speaker_boundaries is None:
        return []
    if whisper_boundaries is None:
        return sorted(speaker_boundaries) if speaker_boundaries else []
    if speaker_boundaries is None:
        return sorted(whisper_boundaries)
    
    # Combine all boundaries
    all_boundaries = list(speaker_boundaries) + list(whisper_boundaries)
    
    if not all_boundaries:
        return []
    
    # Sort boundaries
    all_boundaries.sort()
    
    # Deduplicate boundaries that are very close together
    # Keep the first one in each cluster (which will be from speaker if present)
    merged = []
    i = 0
    
    while i < len(all_boundaries):
        current = all_boundaries[i]
        merged.append(current)
        
        # Skip any boundaries within epsilon of current
        j = i + 1
        while j < len(all_boundaries) and all_boundaries[j] - current <= epsilon:
            j += 1
        
        i = j
    
    return merged


def write_diarization_dump(
    diarization_result: DiarizationResult,
    output_file: str,
    merged_boundaries: list[float] | None = None,
    whisper_boundaries: list[float] | None = None,
) -> None:
    """
    Write diarization data to a file for debugging purposes.
    
    Args:
        diarization_result: Result from diarize_audio()
        output_file: Path to write the dump file
        merged_boundaries: Optional merged boundaries (speaker + Whisper)
        whisper_boundaries: Optional Whisper segment boundaries
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("SPEAKER DIARIZATION DEBUG DUMP\n")
        f.write("=" * 60 + "\n\n")
        
        # Summary
        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Number of speakers detected: {diarization_result['num_speakers']}\n")
        f.write(f"Total speaker segments: {len(diarization_result['segments'])}\n")
        f.write(f"Speaker boundaries used: {len(diarization_result['speaker_boundaries'])}\n")
        f.write(f"Min segment duration for boundary: {MIN_SPEAKER_SEGMENT_SEC}s\n")
        f.write(f"Boundary merge epsilon: {SPEAKER_BOUNDARY_EPSILON_SEC}s\n")
        f.write("\n")
        
        # All raw speaker segments
        f.write("RAW SPEAKER SEGMENTS (from pyannote)\n")
        f.write("-" * 40 + "\n")
        for i, seg in enumerate(diarization_result['segments'], 1):
            duration = seg['end'] - seg['start']
            f.write(f"{i:3d}. [{seg['start']:7.2f}s - {seg['end']:7.2f}s] "
                   f"({duration:5.2f}s) {seg['speaker']}\n")
        f.write("\n")
        
        # Boundary analysis
        f.write("SPEAKER BOUNDARY ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write("(Speaker changes and whether they were used for sentence splitting)\n\n")
        
        filtered = diarization_result.get('filtered_boundaries', [])
        if filtered:
            for i, b in enumerate(filtered, 1):
                status = "✓ INCLUDED" if b['included'] else "✗ FILTERED"
                f.write(f"{i:3d}. {b['timestamp']:7.2f}s: {b['from_speaker']} → {b['to_speaker']}\n")
                f.write(f"     Segment duration: {b['segment_duration']:.2f}s\n")
                f.write(f"     Status: {status}\n")
                if not b['included']:
                    f.write(f"     Reason: {b['reason']}\n")
                f.write("\n")
        else:
            f.write("No speaker changes detected.\n\n")
        
        # Final speaker boundaries used
        f.write("FINAL SPEAKER BOUNDARIES (used for sentence splitting)\n")
        f.write("-" * 40 + "\n")
        if diarization_result['speaker_boundaries']:
            for i, ts in enumerate(diarization_result['speaker_boundaries'], 1):
                f.write(f"{i:3d}. {ts:7.2f}s\n")
        else:
            f.write("No speaker boundaries passed filtering.\n")
        f.write("\n")
        
        # Whisper boundaries if provided
        if whisper_boundaries is not None:
            f.write("WHISPER SEGMENT BOUNDARIES\n")
            f.write("-" * 40 + "\n")
            for i, ts in enumerate(whisper_boundaries, 1):
                f.write(f"{i:3d}. {ts:7.2f}s\n")
            f.write("\n")
        
        # Merged boundaries if provided
        if merged_boundaries is not None:
            f.write("MERGED BOUNDARIES (speaker + Whisper, deduplicated)\n")
            f.write("-" * 40 + "\n")
            for i, ts in enumerate(merged_boundaries, 1):
                # Indicate source
                is_speaker = any(abs(ts - sb) < 0.01 for sb in diarization_result['speaker_boundaries'])
                is_whisper = whisper_boundaries and any(abs(ts - wb) < 0.01 for wb in whisper_boundaries)
                source = []
                if is_speaker:
                    source.append("speaker")
                if is_whisper:
                    source.append("whisper")
                source_str = " (" + "+".join(source) + ")" if source else ""
                f.write(f"{i:3d}. {ts:7.2f}s{source_str}\n")
            f.write("\n")
        
        f.write("=" * 60 + "\n")
        f.write("END OF DIARIZATION DUMP\n")
        f.write("=" * 60 + "\n")


def _convert_boundaries_to_word_indices(
    boundaries: list[float],
    segments: list[dict],
    text: str
) -> set[int]:
    """
    Convert timestamp boundaries to word indices in the text.
    
    This helper is used to map speaker/Whisper boundaries (in seconds)
    to word positions in the concatenated text.
    
    Args:
        boundaries: List of timestamp boundaries in seconds
        segments: Whisper segments with timing info
        text: The full concatenated text
        
    Returns:
        Set of word indices where boundaries occur
    """
    if not boundaries or not segments:
        return set()
    
    word_indices = set()
    words = text.split()
    
    # Build a mapping from character position to word index
    char_to_word = {}
    char_pos = 0
    for word_idx, word in enumerate(words):
        word_len = len(word)
        for i in range(word_len):
            char_to_word[char_pos + i] = word_idx
        char_pos += word_len + 1  # +1 for space
    
    # For each boundary, find the closest segment end and map to word index
    for boundary in boundaries:
        # Find the segment that ends closest to this boundary
        closest_segment = None
        min_distance = float('inf')
        
        for seg in segments:
            seg_end = seg.get('end', 0)
            distance = abs(seg_end - boundary)
            if distance < min_distance:
                min_distance = distance
                closest_segment = seg
        
        if closest_segment:
            # Find where this segment's text ends in the full text
            seg_text = closest_segment.get('text', '').strip()
            if seg_text:
                # Find occurrences of segment text in full text
                # This is approximate - we take the last word of the segment
                seg_words = seg_text.split()
                if seg_words:
                    last_word = seg_words[-1]
                    # Find this word in the full word list
                    for word_idx, word in enumerate(words):
                        if word == last_word:
                            word_indices.add(word_idx)
    
    return word_indices

