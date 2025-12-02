#!/usr/bin/env python3
"""
Speaker diarization utilities for podscripter.

Detects speaker changes in audio and provides speaker boundaries
as hints for sentence splitting.
"""

from typing import TypedDict, Optional
import logging
from pathlib import Path

logger = logging.getLogger("podscripter")

# Default settings
DEFAULT_DIARIZATION_DEVICE = "cpu"
SPEAKER_BOUNDARY_EPSILON_SEC = 1.0  # Merge boundaries within 1 second
MIN_SPEAKER_SEGMENT_SEC = 2.0  # Ignore very short speaker segments

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


class DiarizationError(Exception):
    """Raised when speaker diarization fails."""
    pass


def diarize_audio(
    media_file: str,
    *,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    use_auth_token: Optional[str] = None,
    device: str = DEFAULT_DIARIZATION_DEVICE,
) -> DiarizationResult:
    """
    Perform speaker diarization on an audio file.
    
    Args:
        media_file: Path to the audio/video file
        min_speakers: Minimum number of speakers (None for auto-detect)
        max_speakers: Maximum number of speakers (None for auto-detect)
        use_auth_token: Hugging Face token for model access
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
        # Load the pre-trained pipeline
        logger.info("Loading speaker diarization model...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=use_auth_token
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
        
        # Run diarization
        logger.info(f"Running speaker diarization on {media_path.name}...")
        diarization = pipeline(str(media_file), **params)
        
        # Extract segments
        segments: list[SpeakerSegment] = []
        speakers = set()
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
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
                "speaker_boundaries": []
            }
        
        # Extract speaker boundaries (where speakers change)
        boundaries = _extract_speaker_boundaries(segments)
        
        logger.info(f"Detected {len(speakers)} unique speakers with {len(boundaries)} speaker changes")
        
        return {
            "segments": segments,
            "num_speakers": len(speakers),
            "speaker_boundaries": boundaries
        }
        
    except Exception as e:
        raise DiarizationError(f"Speaker diarization failed: {e}") from e


def _extract_speaker_boundaries(segments: list[SpeakerSegment]) -> list[float]:
    """
    Extract speaker change timestamps from speaker segments.
    
    A speaker boundary occurs at the end of a segment when the next
    segment has a different speaker.
    
    Args:
        segments: List of speaker segments
        
    Returns:
        List of timestamps where speakers change
    """
    if not segments:
        return []
    
    boundaries = []
    
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
            if segment_duration >= MIN_SPEAKER_SEGMENT_SEC:
                boundaries.append(boundary_time)
    
    return boundaries


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

