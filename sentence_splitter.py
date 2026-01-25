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
Unified sentence splitting module with punctuation provenance tracking.

This module consolidates all sentence splitting logic that was previously
scattered across multiple locations:
- _semantic_split_into_sentences()
- Spanish post-processing splits
- assemble_sentences_from_processed() splits
- _write_txt() final splits

Benefits:
- Single source of truth for all splitting decisions
- Tracks punctuation provenance (Whisper vs our logic)
- Coordinates speaker context with punctuation decisions
- Enables comprehensive debugging
- Solves period-before-same-speaker-connector bug
"""

import re
import logging
from typing import List, Optional, Set, Dict, Tuple
from dataclasses import dataclass

from domain_utils import mask_domains, unmask_domains

logger = logging.getLogger("podscripter.splitter")


@dataclass
class Utterance:
    """A single speaker's utterance (may be part of a sentence)."""
    text: str
    speaker: Optional[str]  # e.g., 'SPEAKER_01', or None if no diarization
    start_word: int
    end_word: int


@dataclass
class Sentence:
    """A sentence that may contain utterances from multiple speakers."""
    text: str  # Full sentence text
    utterances: List['Utterance']  # Sub-utterances by speaker
    speaker: Optional[str]  # Primary/dominant speaker (for simple cases)
    
    def has_speaker_changes(self) -> bool:
        """Check if this sentence contains multiple speakers."""
        if not self.utterances:
            return False
        speakers = {u.speaker for u in self.utterances if u.speaker}
        return len(speakers) > 1
    
    def get_first_speaker(self) -> Optional[str]:
        """Get the first speaker in this sentence."""
        if self.utterances:
            return self.utterances[0].speaker
        return self.speaker


@dataclass
class SentenceMetadata:
    """Metadata about how a sentence was split."""
    split_reason: str  # 'speaker_change', 'whisper_boundary', 'semantic', 'grammatical'
    confidence: float  # For semantic splits
    start_word: int
    end_word: int
    speaker: Optional[str]
    whisper_period_removed: bool  # Track if we removed Whisper period
    whisper_period_preserved: bool  # Track if we kept Whisper period


class SentenceSplitter:
    """Unified sentence splitting with punctuation provenance tracking."""
    
    # Connector words that should not start a new sentence when same speaker continues
    CONNECTOR_WORDS = {
        'y', 'e', 'o', 'u',  # Spanish: and, and (before i-), or, or (before o-)
        'pero', 'mas', 'sino',  # Spanish: but
        'and', 'but', 'or',  # English: and, but, or
        'et', 'ou', 'mais',  # French: and, or, but
        'und', 'oder', 'aber',  # German: and, or, but
    }
    
    # Coordinating conjunctions (should never end sentences)
    COORDINATING_CONJUNCTIONS = {
        'y', 'e', 'o', 'u', 'pero', 'mas', 'sino',  # Spanish
        'and', 'but', 'or', 'nor', 'for', 'so', 'yet',  # English
        'et', 'ou', 'mais', 'donc', 'or', 'ni', 'car',  # French
        'und', 'oder', 'aber', 'denn', 'sondern',  # German
    }
    
    # Continuative/auxiliary verbs (should never end sentences)
    CONTINUATIVE_AUXILIARY_VERBS = {
        # Spanish: Infinitive forms (critical for phrases like "sin ser parte")
        'ser', 'estar', 'haber', 'ir', 'tener', 'hacer', 'poder', 'deber',
        'querer', 'saber', 'venir', 'decir', 'ver', 'dar', 'poner', 'salir',
        
        # Spanish: Preterite/past tense (critical for phrases like "fueron dirigidos")
        'fue', 'fueron', 'fui', 'fuimos', 'fuiste', 'fuisteis',
        'estuvo', 'estuvieron', 'estuve', 'estuvimos', 'estuviste', 'estuvisteis',
        'hubo',  # impersonal preterite of haber
        
        # Spanish: Imperfect tense and auxiliary verbs (existing)
        'estaba', 'estaban', 'estabas', 'estábamos', 'estabais',
        'era', 'eran', 'eras', 'éramos', 'erais',
        'había', 'habían', 'habías', 'habíamos', 'habíais',
        'tenía', 'tenían', 'tenías', 'teníamos', 'teníais',
        'iba', 'iban', 'ibas', 'íbamos', 'ibais',
        'hacía', 'hacían', 'hacías', 'hacíamos', 'hacíais',
        'podía', 'podían', 'podías', 'podíamos', 'podíais',
        'debía', 'debían', 'debías', 'debíamos', 'debíais',
        'quería', 'querían', 'querías', 'queríamos', 'queríais',
        'sabía', 'sabían', 'sabías', 'sabíamos', 'sabíais',
        'venía', 'venían', 'venías', 'veníamos', 'veníais',
        'decía', 'decían', 'decías', 'decíamos', 'decíais',
        'he', 'has', 'ha', 'hemos', 'habéis', 'han',
        
        # Spanish: Present tense of ser/estar (often followed by adjectives/participles)
        'es', 'son', 'soy', 'somos', 'eres', 'sois',
        'está', 'están', 'estoy', 'estamos', 'estás', 'estáis',
        
        # English: Infinitive forms
        'be', 'have', 'do', 'go', 'get', 'make',
        
        # English: Past continuous and auxiliary verbs (existing + additions)
        'was', 'were', 'had', 'been', 'have', 'has', 'being',
        'is', 'are', 'am',  # Present tense often followed by participles
        
        # French: Infinitive forms
        'être', 'avoir', 'aller', 'faire', 'pouvoir', 'devoir', 'vouloir', 'savoir',
        
        # French: Preterite/passé simple forms
        'fut', 'furent', 'fus', 'fûmes', 'fûtes',
        'eut', 'eurent', 'eus', 'eûmes', 'eûtes',
        
        # French: Present tense of être/avoir
        'est', 'sont', 'suis', 'sommes', 'êtes',
        'a', 'ont', 'ai', 'avons', 'avez',
        
        # French: Imperfect tense and auxiliary verbs (existing)
        'étais', 'était', 'étions', 'étiez', 'étaient',
        'avais', 'avait', 'avions', 'aviez', 'avaient',
        'allais', 'allait', 'allions', 'alliez', 'allaient',
        'faisais', 'faisait', 'faisions', 'faisiez', 'faisaient',
        
        # German: Infinitive forms
        'sein', 'haben', 'werden', 'gehen', 'machen', 'können', 'müssen',
        'wollen', 'sollen', 'dürfen',
        
        # German: Present tense of sein/haben/werden
        'ist', 'sind', 'bin', 'bist', 'seid',
        'hat', 'habe', 'hast', 'habt',
        'wird', 'werden', 'wirst', 'werdet',
        
        # German: Imperfect tense and auxiliary verbs (existing)
        'war', 'warst', 'waren', 'wart',
        'hatte', 'hattest', 'hatten', 'hattet',
        'ging', 'gingst', 'gingen', 'gingt',
        'machte', 'machtest', 'machten', 'machtet',
        'konnte', 'konntest', 'konnten', 'konntet',
        'wollte', 'wolltest', 'wollten', 'wolltet',
        'musste', 'musstest', 'mussten', 'musstet',
        'sollte', 'solltest', 'sollten', 'solltet',
        'wurde', 'wurdest', 'wurden', 'wurdet',
    }
    
    def __init__(self, language: str, model, config: dict):
        """
        Initialize sentence splitter.
        
        Args:
            language: Language code (en, es, fr, de)
            model: SentenceTransformer model for semantic analysis
            config: LanguageConfig with thresholds and language-specific settings
        """
        self.language = language
        self.model = model
        self.config = config
        self.logger = logging.getLogger("podscripter.splitter")
        
        # Metadata tracking for debugging
        self.split_metadata: List[SentenceMetadata] = []
        self.removed_periods: List[Dict] = []
        self.added_periods: List[Dict] = []
        self.skipped_whisper_boundaries: Set[int] = set()  # Track skipped Whisper word boundaries
        self.speaker_word_segments: Optional[List[Dict]] = None  # Speaker segments in word ranges
    
    def split(
        self,
        text: str,
        whisper_segments: Optional[List[Dict]] = None,
        speaker_segments: Optional[List[Dict]] = None,
        mode: str = 'semantic'
    ) -> Tuple[List[Sentence], Dict]:
        """
        Single entry point for all sentence splitting.
        
        Args:
            text: Raw text with Whisper-added punctuation
            whisper_segments: Whisper segment boundaries with metadata
            speaker_segments: Speaker change segments with labels
            mode: 'semantic', 'punctuation', 'hybrid', or 'preserve'
        
        Returns:
            (sentences, metadata) where metadata includes:
              - split_reasons: list of why each split was made
              - removed_periods: list of Whisper periods that were removed
              - added_periods: list of periods we added
        """
        # Reset metadata
        self.split_metadata = []
        self.removed_periods = []
        self.added_periods = []
        self.skipped_whisper_boundaries = set()
        
        # 1. Parse and track Whisper punctuation
        whisper_periods = self._extract_whisper_punctuation(text, whisper_segments)
        
        # 2. Convert boundaries to word indices
        whisper_word_boundaries = self._convert_segments_to_word_boundaries(whisper_segments, text)
        speaker_word_boundaries = self._convert_segments_to_word_boundaries(speaker_segments, text, is_speaker=True)
        speaker_word_segments = self._create_speaker_word_ranges(speaker_segments, text)
        
        # Store as instance variable for _detect_speaker_changes_in_sentence() to access
        self.speaker_word_segments = speaker_word_segments
        
        # Debug logging
        if speaker_word_boundaries:
            self.logger.debug(f"SentenceSplitter detected {len(speaker_word_boundaries)} speaker word boundaries")
            if len(speaker_word_boundaries) > 0:
                boundaries_list = sorted(list(speaker_word_boundaries))[:10]
                self.logger.debug(f"First 10 speaker boundaries at words: {boundaries_list}")
        else:
            self.logger.warning("SentenceSplitter received NO speaker word boundaries!")
        
        # Debug logging for speaker boundaries
        if speaker_word_boundaries:
            self.logger.debug(f"SentenceSplitter detected {len(speaker_word_boundaries)} speaker boundaries")
            if len(speaker_word_boundaries) > 0:
                sorted_boundaries = sorted(speaker_word_boundaries)
                self.logger.debug(f"First 5 speaker boundaries: {sorted_boundaries[:5]}")
                self.logger.debug(f"Last 5 speaker boundaries: {sorted_boundaries[-5:]}")
        else:
            self.logger.warning("NO speaker boundaries detected despite speaker_segments being provided!")
        
        # 3. Evaluate boundaries with full context
        sentences = self._evaluate_boundaries(
            text, 
            whisper_word_boundaries,
            speaker_word_boundaries,
            speaker_word_segments,
            whisper_periods,
            mode
        )
        
        # 4. Manage Whisper punctuation intelligently
        # This SOLVES the period-before-same-speaker-connector bug
        sentences = self._process_whisper_punctuation(
            sentences,
            whisper_periods,
            speaker_word_segments,
            text
        )
        
        # 5. Generate metadata
        metadata = self._build_metadata()
        
        return sentences, metadata
    
    def _extract_whisper_punctuation(
        self,
        text: str,
        whisper_segments: Optional[List[Dict]]
    ) -> Dict[int, str]:
        """
        Track which periods/punctuation came from Whisper segment ends.
        
        Args:
            text: The full text
            whisper_segments: List of Whisper segments with 'text', 'start', 'end' fields
        
        Returns:
            Dict mapping character positions to punctuation type
        """
        whisper_periods = {}
        
        if not whisper_segments:
            return whisper_periods
        
        # Build character position mapping
        char_pos = 0
        for seg in whisper_segments:
            seg_text = seg.get('text', '').strip()
            if not seg_text:
                continue
            
            # Calculate where this segment ends in the full text
            seg_end_char = char_pos + len(seg_text)
            
            # Check if segment ends with terminal punctuation
            if seg_text.rstrip().endswith(('.', '!', '?')):
                punctuation = seg_text.rstrip()[-1]
                whisper_periods[seg_end_char - 1] = punctuation
                self.logger.debug(f"Whisper period tracked: char {seg_end_char - 1} = '{punctuation}'")
            
            # Move position forward (segment text + space)
            char_pos += len(seg_text) + 1
        
        return whisper_periods
    
    def _convert_segments_to_word_boundaries(
        self,
        segments: Optional[List[Dict]],
        text: str,
        is_speaker: bool = False
    ) -> Optional[Set[int]]:
        """
        Convert segment boundaries to word indices.
        
        Args:
            segments: List of segments with time-based or word-based boundaries
            text: The full text
            is_speaker: Whether these are speaker segments
        
        Returns:
            Set of word indices where boundaries occur, or None
        """
        if not segments:
            return None
        
        boundaries = set()
        words = text.split()
        
        # For speaker segments, extract boundaries where speaker CHANGES
        if is_speaker:
            for i in range(len(segments) - 1):
                current_seg = segments[i]
                next_seg = segments[i + 1]
                
                # Only add boundary if speaker changes
                if current_seg.get('speaker') != next_seg.get('speaker'):
                    if 'end_word' in current_seg:
                        # end_word is EXCLUSIVE, so the last word of current speaker is end_word - 1
                        # We want to split AFTER the last word of the current speaker,
                        # so the boundary should be at end_word - 1 (the last word of current speaker)
                        boundary_word = current_seg['end_word'] - 1
                        if boundary_word >= 0:
                            boundaries.add(boundary_word)
        else:
            # For Whisper segments, we need to calculate word positions
            # This is done by tracking cumulative text length
            char_to_word = self._build_char_to_word_map(text)
            
            char_pos = 0
            for seg in segments:
                seg_text = seg.get('text', '').strip()
                if not seg_text:
                    continue
                
                seg_end_char = char_pos + len(seg_text)
                
                # Find the word index at this character position
                word_idx = char_to_word.get(seg_end_char - 1)
                if word_idx is not None:
                    boundaries.add(word_idx)
                
                char_pos += len(seg_text) + 1
        
        return boundaries if boundaries else None
    
    def _build_char_to_word_map(self, text: str) -> Dict[int, int]:
        """Build a mapping from character positions to word indices."""
        char_to_word = {}
        words = text.split()
        char_pos = 0
        
        for word_idx, word in enumerate(words):
            # Map all characters in this word to its index
            for i in range(len(word)):
                char_to_word[char_pos + i] = word_idx
            char_pos += len(word) + 1  # word length + space
        
        return char_to_word
    
    def _create_speaker_word_ranges(
        self,
        speaker_segments: Optional[List[Dict]],
        text: str
    ) -> Optional[List[Dict]]:
        """
        Convert speaker segments to word-range format if needed.
        
        Args:
            speaker_segments: Speaker segments (may already be in word format)
            text: The full text
        
        Returns:
            List of dicts with 'start_word', 'end_word', 'speaker' fields
        """
        if not speaker_segments:
            return None
        
        # Check if already in word format
        if speaker_segments and 'start_word' in speaker_segments[0]:
            return speaker_segments
        
        # If in time format, we'd need to convert (not implemented in current version)
        # For now, assume they're already in word format from podscripter.py conversion
        return speaker_segments
    
    def _detect_speaker_changes_in_sentence(
        self,
        sentence_text: str,
        sentence_start_word: int,
        sentence_end_word: int
    ) -> List[Utterance]:
        """
        Detect if a sentence contains speaker changes and split into utterances.
        
        Args:
            sentence_text: The text of the sentence
            sentence_start_word: Word index where sentence starts
            sentence_end_word: Word index where sentence ends
        
        Returns:
            List of Utterance objects (one per speaker segment within the sentence)
        """
        if not self.speaker_word_segments:
            # No diarization - return single utterance with no speaker
            return [Utterance(
                text=sentence_text,
                speaker=None,
                start_word=sentence_start_word,
                end_word=sentence_end_word
            )]
        
        utterances = []
        words = sentence_text.split()
        
        # Find all speaker segments that overlap this sentence
        overlapping_segments = []
        for seg in self.speaker_word_segments:
            seg_start = seg['start_word']
            seg_end = seg['end_word']
            # Check for overlap
            if seg_start < sentence_end_word and seg_end > sentence_start_word:
                overlapping_segments.append(seg)
        
        if not overlapping_segments:
            # No speaker info for this range - return single utterance
            return [Utterance(
                text=sentence_text,
                speaker=None,
                start_word=sentence_start_word,
                end_word=sentence_end_word
            )]
        
        # Sort by start position
        overlapping_segments.sort(key=lambda s: s['start_word'])
        
        # Merge overlapping segments and resolve conflicts by assigning each word to one speaker
        # When segments overlap, the segment with more overlap on that word wins
        word_to_speaker = {}
        for word_idx in range(sentence_start_word, sentence_end_word):
            # Find which speaker segment this word belongs to (most overlap)
            best_speaker = None
            best_overlap = 0
            
            for seg in overlapping_segments:
                if seg['start_word'] <= word_idx < seg['end_word']:
                    # Calculate how much this segment "owns" this word
                    # If multiple segments claim this word, prefer the one with longer total duration
                    seg_duration = seg['end_word'] - seg['start_word']
                    if seg_duration > best_overlap:
                        best_overlap = seg_duration
                        best_speaker = seg['speaker']
            
            if best_speaker:
                word_to_speaker[word_idx] = best_speaker
        
        # Now create utterances by grouping consecutive words from the same speaker
        current_speaker = None
        current_start = None
        current_words = []
        
        for word_idx in range(sentence_start_word, sentence_end_word):
            speaker = word_to_speaker.get(word_idx)
            word = words[word_idx - sentence_start_word]
            
            if speaker != current_speaker:
                # Speaker changed - save current utterance if any
                if current_words:
                    utterances.append(Utterance(
                        text=' '.join(current_words),
                        speaker=current_speaker,
                        start_word=current_start,
                        end_word=word_idx
                    ))
                
                # Start new utterance
                current_speaker = speaker
                current_start = word_idx
                current_words = [word] if speaker else []
            else:
                # Same speaker - continue utterance
                if speaker:
                    current_words.append(word)
        
        # Don't forget the last utterance
        if current_words:
            utterances.append(Utterance(
                text=' '.join(current_words),
                speaker=current_speaker,
                start_word=current_start,
                end_word=sentence_end_word
            ))
        
        if not utterances:
            # Fallback - return single utterance
            return [Utterance(
                text=sentence_text,
                speaker=None,
                start_word=sentence_start_word,
                end_word=sentence_end_word
            )]
        
        return utterances
    
    def _evaluate_boundaries(
        self,
        text: str,
        whisper_word_boundaries: Optional[Set[int]],
        speaker_word_boundaries: Optional[Set[int]],
        speaker_word_segments: Optional[List[Dict]],
        whisper_periods: Dict[int, str],
        mode: str
    ) -> List[str]:
        """
        Unified boundary evaluation (consolidates _should_end_sentence_here logic).
        
        This is the core splitting logic migrated from _semantic_split_into_sentences
        and _should_end_sentence_here.
        
        Args:
            text: The text to split
            whisper_word_boundaries: Set of word indices where Whisper segments end
            speaker_word_boundaries: Set of word indices where speakers change
            speaker_word_segments: List of speaker segments with word ranges
            whisper_periods: Dict of Whisper-added punctuation positions
            mode: Splitting mode
        
        Returns:
            List of sentences
        """
        # Mask domains before splitting to prevent breaking them
        text_masked = mask_domains(text, use_exclusions=True, language=self.language)
        
        words = text_masked.split()
        if len(words) < 3:
            # Too short to split
            unmasked = unmask_domains(text_masked)
            # Create a single Sentence object for the entire text
            utterances = self._detect_speaker_changes_in_sentence(unmasked, 0, len(words))
            primary_speaker = utterances[0].speaker if utterances else None
            return [Sentence(text=unmasked, utterances=utterances, speaker=primary_speaker)]
        
        sentences: List[Sentence] = []
        current_chunk: List[str] = []
        sentence_start_word: int = 0  # Track the starting word index of current sentence
        
        for i, word in enumerate(words):
            current_chunk.append(word)
            
            should_end = self._should_end_sentence_here(
                words, i, current_chunk,
                whisper_word_boundaries,
                speaker_word_boundaries,
                speaker_word_segments
            )
            
            # NOTE: We no longer remove Whisper periods at skipped boundaries unconditionally.
            # The skipped_whisper_boundaries set tracks positions where we skipped a Whisper boundary
            # because a speaker boundary was nearby. In this case, the period should STAY because
            # it marks the end of the current speaker's sentence before a different speaker starts.
            # Period removal only happens in the "same-speaker connector" logic below.

            
            # DEBUG: Log sentence endings around connectors
            if i + 1 < len(words):
                next_word_clean = words[i + 1].lower().strip('.,;:!?¿¡')
                if next_word_clean in self.CONNECTOR_WORDS:
                    self.logger.debug(
                        f"SENTENCE END CHECK: word {i} ('{word}'), "
                        f"next='{words[i + 1]}', connector='{next_word_clean}', "
                        f"should_end={should_end}, chunk_len={len(current_chunk)}"
                    )
            
            # CRITICAL FIX: Remove Whisper period before same-speaker connectors
            # If we decided NOT to split here, but the word has a Whisper period,
            # and next word is a connector, remove the period AND lowercase the connector
            # BUT ONLY if the same speaker continues - different speakers should keep the period
            if not should_end and i + 1 < len(words):
                next_word_clean = words[i + 1].lower().strip('.,;:!?¿¡')
                if next_word_clean in self.CONNECTOR_WORDS and word.rstrip().endswith(('.', '!', '?')):
                    # Check if same speaker continues
                    speaker_at_current = None
                    speaker_at_next = None
                    if speaker_word_segments:
                        speaker_at_current = self._get_speaker_at_word(i, speaker_word_segments)
                        speaker_at_next = self._get_speaker_at_word(i + 1, speaker_word_segments)
                    
                    # IMPORTANT: Only remove period if SAME speaker continues
                    # If speakers are different, keep the period - it marks end of current speaker's sentence
                    # Also require speaker info to be available for both positions
                    if speaker_at_current is not None and speaker_at_next is not None and speaker_at_current == speaker_at_next:
                        # Remove the period from the current word in the chunk
                        current_chunk[-1] = current_chunk[-1].rstrip('.!?')
                        # Also lowercase the connector in the next iteration
                        # We need to modify the words list directly
                        words[i + 1] = next_word_clean
                        self.logger.debug(
                            f"REMOVED Whisper period before same-speaker connector: "
                            f"word={i} ('{word}'), connector='{next_word_clean}' (lowercased), "
                            f"speaker={speaker_at_current}"
                        )
                        self.removed_periods.append({
                            'position': i,
                            'reason': 'same_speaker_connector_inline',
                            'speaker': speaker_at_current,
                            'connector': next_word_clean
                        })
                    elif speaker_at_current != speaker_at_next:
                        self.logger.debug(
                            f"KEPT Whisper period before different-speaker connector: "
                            f"word={i} ('{word}'), connector='{next_word_clean}', "
                            f"curr_spk={speaker_at_current}, next_spk={speaker_at_next}"
                        )
            
            if should_end:
                sentence_text = ' '.join(current_chunk).strip()
                if sentence_text:
                    # Unmask domains before adding to sentences
                    sentence_text = unmask_domains(sentence_text)
                    
                    # Calculate sentence word range
                    sentence_end_word = i + 1  # End is exclusive
                    
                    # Detect speaker changes within this sentence and create utterances
                    utterances = self._detect_speaker_changes_in_sentence(
                        sentence_text,
                        sentence_start_word,
                        sentence_end_word
                    )
                    
                    # Determine primary speaker (first speaker in sentence)
                    primary_speaker = utterances[0].speaker if utterances else None
                    
                    # Create Sentence object
                    sentence_obj = Sentence(
                        text=sentence_text,
                        utterances=utterances,
                        speaker=primary_speaker
                    )
                    sentences.append(sentence_obj)
                    
                    # Update start word for next sentence
                    sentence_start_word = i + 1
                current_chunk = []
        
        # Add remaining chunk
        if current_chunk:
            sentence_text = ' '.join(current_chunk).strip()
            if sentence_text:
                sentence_text = unmask_domains(sentence_text)
                
                # Calculate sentence word range for final sentence
                sentence_end_word = len(words)
                
                # Detect speaker changes within this sentence and create utterances
                utterances = self._detect_speaker_changes_in_sentence(
                    sentence_text,
                    sentence_start_word,
                    sentence_end_word
                )
                
                # Determine primary speaker (first speaker in sentence)
                primary_speaker = utterances[0].speaker if utterances else None
                
                # Create Sentence object
                sentence_obj = Sentence(
                    text=sentence_text,
                    utterances=utterances,
                    speaker=primary_speaker
                )
                sentences.append(sentence_obj)
        
        return sentences
    
    def _should_end_sentence_here(
        self,
        words: List[str],
        current_index: int,
        current_chunk: List[str],
        whisper_word_boundaries: Optional[Set[int]],
        speaker_word_boundaries: Optional[Set[int]],
        speaker_word_segments: Optional[List[Dict]]
    ) -> bool:
        """
        Determine if a sentence should end at the current position.
        
        This consolidates the logic from the old _should_end_sentence_here function.
        
        Priority hierarchy:
        1. Grammatical guards (NEVER break)
        2. Speaker continuity at connectors (don't break if same speaker continues)
        3. Speaker boundaries (highest priority, min 1 word - v0.6.1)
        4. Whisper boundaries (medium priority, min 10 words, skip if speaker nearby)
        5. General semantic splitting (fallback, min 20 words Spanish, 15 others)
        
        Args:
            words: List of all words
            current_index: Current word index
            current_chunk: Current sentence chunk
            whisper_word_boundaries: Set of Whisper boundary indices
            speaker_word_boundaries: Set of speaker boundary indices
            speaker_word_segments: List of speaker segments
        
        Returns:
            True if sentence should end here
        """
        # Get thresholds from config
        thresholds = self.config.thresholds
        
        current_word = words[current_index]
        next_word = words[current_index + 1] if current_index + 1 < len(words) else ""
        
        # PRIORITY 1: Speaker boundaries (HIGHEST PRIORITY - check first!)
        # Speaker changes ALWAYS indicate sentence breaks, even if next word is a connector
        # Different speakers should NEVER have their sentences merged, even with connectors
        # Per v0.4.3: "Speaker boundaries now ALWAYS create splits, regardless of whether 
        # the next word is a connector"
        if speaker_word_boundaries and current_index in speaker_word_boundaries:
            min_words_speaker = 1  # Minimal threshold - speaker changes are definitive signals
            
            if len(current_chunk) >= min_words_speaker:
                # Speaker change - ALWAYS split here (per v0.4.3 fix)
                # Note: We do NOT check for connectors here. If a new speaker starts with
                # "Y" or "y", that's valid - they're a different speaker and should be separated.
                self.logger.debug(
                    f"✓ SPLIT at speaker boundary: word {current_index} '{current_word}', "
                    f"next='{next_word}', chunk_len={len(current_chunk)}"
                )
                return True
            else:
                self.logger.debug(
                    f"✗ SKIP speaker boundary: word {current_index} '{current_word}', chunk too short "
                    f"({len(current_chunk)} < {min_words_speaker})"
                )
        
        # Don't end sentence if we're at the very beginning or very end
        # (Allow index 1 for speaker boundaries with 2-word minimum)
        if current_index < 1 or current_index >= len(words) - 1:
            return False
        
        # PRIORITY 2: Grammatical guards (NEVER break)
        if self._violates_grammatical_rules(current_word, next_word):
            return False
        
        # PRIORITY 3: Check for Whisper boundaries that should be skipped (TRACK EARLY)
        # Do this BEFORE min_total_words check so we track skips even in short texts
        if whisper_word_boundaries and current_index in whisper_word_boundaries:
            self.logger.debug(
                f"At Whisper boundary: word {current_index}, "
                f"has_speaker_boundaries={speaker_word_boundaries is not None}"
            )
            
            # Skip this Whisper boundary if there's a speaker boundary coming VERY soon
            # AND the next word suggests continuation (connector or lowercase)
            # This prevents splitting when speaker boundaries are misaligned by 1-3 words
            if speaker_word_boundaries and current_index + 1 < len(words):
                # Find closest upcoming speaker boundary
                closest_boundary = None
                for boundary_idx in speaker_word_boundaries:
                    if current_index < boundary_idx <= current_index + 3:
                        if closest_boundary is None or boundary_idx < closest_boundary:
                            closest_boundary = boundary_idx
                
                if closest_boundary is not None:
                    # Check if next word suggests continuation (not a new sentence)
                    next_word = words[current_index + 1]
                    next_word_clean = next_word.strip('.,;:!?¿¡').lower()
                    
                    # Skip only if next word is a connector OR starts lowercase (continuation)
                    is_connector = next_word_clean in self.CONNECTOR_WORDS
                    starts_lowercase = next_word[0].islower() if next_word else False
                    
                    if is_connector or starts_lowercase:
                        # Skip and let speaker boundary handle the split
                        # TRACK THIS: We skipped a Whisper boundary, so we should remove its period
                        self.skipped_whisper_boundaries.add(current_index)
                        self.logger.debug(
                            f"SKIPPED Whisper boundary at word {current_index} "
                            f"(speaker boundary {closest_boundary - current_index} words away, "
                            f"next_word='{next_word}', is_connector={is_connector}, starts_lowercase={starts_lowercase})"
                        )
                        return False
                    else:
                        self.logger.debug(
                            f"NOT SKIPPING Whisper boundary at word {current_index} "
                            f"(speaker boundary close but next word '{next_word}' suggests new sentence)"
                        )
        
        # If the entire input is very short, don't split (UNLESS speaker boundary above)
        if len(words) <= thresholds.get('min_total_words_no_split', 25):
            return False
        
        # PRIORITY 4: Whisper boundaries - process split decision (MEDIUM-HIGH PRIORITY)
        # Whisper boundaries represent acoustic pauses
        if whisper_word_boundaries and current_index in whisper_word_boundaries:
            
            min_words_whisper = thresholds.get('min_words_whisper_break', 10)
            
            if len(current_chunk) >= min_words_whisper:
                # Check if next word is a connector and same speaker continues
                next_word_clean = next_word.lower().strip('.,;:!?¿¡')
                
                if next_word_clean in self.CONNECTOR_WORDS:
                    # Check speaker continuity
                    current_speaker = self._get_speaker_at_word(current_index, speaker_word_segments)
                    next_speaker = self._get_speaker_at_word(current_index + 1, speaker_word_segments)
                    
                    self.logger.debug(
                        f"CONNECTOR: Whisper boundary at word {current_index} "
                        f"('{words[current_index]}'), next='{next_word}', "
                        f"connector='{next_word_clean}', curr_spk={current_speaker}, "
                        f"next_spk={next_speaker}"
                    )
                    
                    if current_speaker == next_speaker and current_speaker is not None:
                        # Same speaker continues with connector - don't break
                        self.logger.debug(
                            f"  → PREVENTING BREAK: same speaker {current_speaker} "
                            f"continues with connector '{next_word_clean}'"
                        )
                        return False
                    else:
                        # Different speakers or no info - allow break
                        # Per user requirement: speaker changes should still split even inside questions
                        self.logger.debug(
                            f"  → ALLOWING BREAK: speakers differ or no info "
                            f"(curr={current_speaker}, next={next_speaker})"
                        )
                        return True
                else:
                    # Not a connector - check for unclosed Spanish question before allowing break
                    # This prevents splitting "¿qué cambios ha habido" | "desde la pandemia?"
                    if self._is_inside_unclosed_question(current_chunk, words[current_index]):
                        self.logger.debug(
                            f"  → PREVENTING BREAK: inside unclosed Spanish question at word {current_index}"
                        )
                        return False
                    # Allow break
                    return True
        
        # PRIORITY 4: General minimum chunk length for semantic splitting
        if len(current_chunk) < thresholds.get('min_chunk_before_split', 18):
            return False
        
        # Additional language-specific checks
        if not self._passes_language_specific_checks(words, current_index, current_chunk, next_word):
            return False
        
        # PRIORITY 5: Semantic coherence check (if we have the model)
        if self.model is not None:
            if len(current_chunk) >= thresholds.get('min_chunk_semantic_break', 30):
                return self._check_semantic_break(words, current_index)
        
        return False
    
    def _violates_grammatical_rules(self, current_word: str, next_word: str) -> bool:
        """
        Check if breaking at current position would violate basic grammatical rules.
        
        Args:
            current_word: The word at the current position
            next_word: The following word
        
        Returns:
            True if breaking here would violate grammar rules
        """
        current_clean = current_word.lower().strip('.,;:!?¿¡')
        
        # Never end on coordinating conjunctions
        if current_clean in self.COORDINATING_CONJUNCTIONS:
            return True
        
        # Never end on continuative/auxiliary verbs
        if current_clean in self.CONTINUATIVE_AUXILIARY_VERBS:
            return True
        
        # Language-specific prepositions and articles
        if self.language == 'es':
            spanish_forbidden = {
                'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
                'a', 'ante', 'bajo', 'de', 'del', 'al', 'en', 'con', 'por', 
                'para', 'sin', 'sobre', 'entre', 'tras', 'durante', 'mediante',
                'según', 'hacia', 'hasta', 'desde', 'contra',
                'todo', 'toda', 'todos', 'todas', 'alguno', 'alguna', 'algunos',
                'algunas', 'cualquier', 'cualquiera', 'ningún', 'ninguna', 'ninguno',
                'otro', 'otra', 'otros', 'otras'
            }
            if current_clean in spanish_forbidden:
                return True
        
        elif self.language == 'en':
            english_forbidden = {
                'the', 'a', 'an',
                'to', 'at', 'from', 'with', 'by', 'of', 'in', 'on', 'for', 'about',
                'this', 'that', 'these', 'those',
                'some', 'any', 'many', 'much', 'few', 'several',
            }
            if current_clean in english_forbidden:
                return True
        
        elif self.language == 'fr':
            french_forbidden = {
                'le', 'la', 'les', 'un', 'une', 'des',
                'à', 'de', 'en', 'pour', 'avec', 'sans', 'sous', 'sur', 'dans', 'chez',
                'ce', 'cet', 'cette', 'ces',
                'du', 'au', 'aux',
            }
            if current_clean in french_forbidden:
                return True
        
        elif self.language == 'de':
            german_forbidden = {
                'der', 'die', 'das', 'den', 'dem', 'des',
                'ein', 'eine', 'einen', 'einem', 'einer', 'eines',
                'zu', 'an', 'auf', 'aus', 'bei', 'mit', 'nach', 'von', 'vor', 'in', 'für',
                'dieser', 'diese', 'dieses', 'diesen',
            }
            if current_clean in german_forbidden:
                return True
        
        return False
    
    def _is_inside_unclosed_question(
        self,
        current_chunk: List[str],
        current_word: str = None
    ) -> bool:
        """
        Check if we're inside an unclosed Spanish inverted question.
        
        Spanish questions use inverted punctuation: ¿pregunta?
        We should never split a sentence while inside such a construct
        (i.e., after seeing ¿ but before seeing the closing ?).
        
        This also applies to exclamations: ¡exclamación!
        
        Args:
            current_chunk: Current sentence chunk being built
            current_word: Optional current word to check for ¿ or ¡
        
        Returns:
            True if inside an unclosed Spanish question/exclamation
        """
        if self.language != 'es':
            return False
        
        # Check if current word contains opening inverted punctuation
        if current_word and ('¿' in current_word or '¡' in current_word):
            return True
        
        # Check if chunk contains unclosed inverted question
        current_text = ' '.join(current_chunk)
        if '¿' in current_text and '?' not in current_text:
            return True
        
        # Check if chunk contains unclosed inverted exclamation
        if '¡' in current_text and '!' not in current_text:
            return True
        
        return False
    
    def _passes_language_specific_checks(
        self,
        words: List[str],
        current_index: int,
        current_chunk: List[str],
        next_word: str
    ) -> bool:
        """
        Language-specific checks for sentence boundaries.
        
        Args:
            words: All words
            current_index: Current word index
            current_chunk: Current chunk
            next_word: Next word
        
        Returns:
            True if checks pass, False if should not break
        """
        current_word = words[current_index]
        current_word_clean = current_word.lower().strip('.,;:!?¿¡')
        
        # Spanish: Never split after or inside unclosed inverted question/exclamation mark
        if self._is_inside_unclosed_question(current_chunk, current_word):
            return False
        
        if next_word:
            next_word_clean = next_word.lower().strip('.,;:!?¿¡')
            
            # Never split when current word precedes a number
            if next_word.strip('.,;:!?').isdigit():
                # Conjunction before number in a list
                conjunctions_before_numbers = {'y', 'o', 'and', 'or', 'et', 'ou', 'und', 'oder'}
                if current_word_clean in conjunctions_before_numbers:
                    if len(current_chunk) >= 2:
                        prev_words = current_chunk[-3:] if len(current_chunk) >= 3 else current_chunk
                        has_prev_number = any(any(c.isdigit() for c in w.strip('.,;:!?')) for w in prev_words)
                        if has_prev_number:
                            return False
                
                # Nouns that precede numbers
                number_preceding_nouns = {
                    'episode', 'episodes', 'episodio', 'episodios', 'épisode', 'épisodes',
                    'chapter', 'chapters', 'capítulo', 'capítulos', 'chapitre', 'chapitres', 'kapitel',
                    'year', 'years', 'año', 'años', 'année', 'années', 'jahr', 'jahre',
                }
                if current_word_clean in number_preceding_nouns:
                    return False
            
            # CRITICAL FIX: Never split after numbers when followed by time/measurement units
            # This prevents splits like "a los 18. Años" (should be "a los 18 años")
            if any(c.isdigit() for c in current_word_clean):
                time_measurement_units = {
                    # Spanish
                    'años', 'año', 'meses', 'mes', 'días', 'día', 'horas', 'hora',
                    'minutos', 'minuto', 'segundos', 'segundo', 'semanas', 'semana',
                    'siglos', 'siglo', 'décadas', 'década', 'veces', 'vez',
                    'personas', 'persona', 'dólares', 'dólar', 'euros', 'euro',
                    'pesos', 'peso', 'metros', 'metro', 'kilómetros', 'kilómetro',
                    'por', 'ciento', 'mil', 'millones', 'millón',
                    # English
                    'years', 'year', 'months', 'month', 'days', 'day', 'hours', 'hour',
                    'minutes', 'minute', 'seconds', 'second', 'weeks', 'week',
                    'centuries', 'century', 'decades', 'decade', 'times', 'time',
                    'people', 'person', 'dollars', 'dollar', 'percent', 'thousand',
                    'million', 'billion', 'meters', 'meter', 'kilometers', 'kilometer',
                    # French
                    'ans', 'an', 'mois', 'jours', 'jour', 'heures', 'heure',
                    'minutes', 'minute', 'secondes', 'seconde', 'semaines', 'semaine',
                    'siècles', 'siècle', 'décennies', 'décennie', 'fois',
                    'personnes', 'personne', 'euros', 'euro', 'pour', 'cent',
                    'mille', 'millions', 'million', 'mètres', 'mètre',
                    # German
                    'jahre', 'jahr', 'monate', 'monat', 'tage', 'tag', 'stunden', 'stunde',
                    'minuten', 'minute', 'sekunden', 'sekunde', 'wochen', 'woche',
                    'jahrhunderte', 'jahrhundert', 'jahrzehnte', 'jahrzehnt', 'mal',
                    'personen', 'person', 'euro', 'prozent', 'tausend',
                    'millionen', 'million', 'meter', 'kilometer',
                }
                if next_word_clean in time_measurement_units:
                    return False
            
            # CRITICAL FIX: Never split auxiliary verbs from following past participles
            # This prevents splits like "fueron. Dirigidos" (should be "fueron dirigidos")
            if current_word_clean in self.CONTINUATIVE_AUXILIARY_VERBS:
                if self._is_past_participle(next_word_clean):
                    return False
        
        return True
    
    def _is_past_participle(self, word: str) -> bool:
        """
        Check if a word is likely a past participle based on common endings.
        
        Args:
            word: The word to check (should be lowercase and stripped of punctuation)
        
        Returns:
            True if the word appears to be a past participle
        """
        word = word.lower().strip('.,;:!?¿¡')
        
        # Spanish past participles: -ado, -ido (and some common irregulars)
        spanish_participle_endings = ('ado', 'ados', 'ada', 'adas', 'ido', 'idos', 'ida', 'idas')
        spanish_irregular_participles = {
            'hecho', 'hechos', 'hecha', 'hechas',
            'dicho', 'dichos', 'dicha', 'dichas',
            'escrito', 'escritos', 'escrita', 'escritas',
            'visto', 'vistos', 'vista', 'vistas',
            'puesto', 'puestos', 'puesta', 'puestas',
            'abierto', 'abiertos', 'abierta', 'abiertas',
            'cubierto', 'cubiertos', 'cubierta', 'cubiertas',
            'muerto', 'muertos', 'muerta', 'muertas',
            'roto', 'rotos', 'rota', 'rotas',
            'vuelto', 'vueltos', 'vuelta', 'vueltas',
            'resuelto', 'resueltos', 'resuelta', 'resueltas',
            'satisfecho', 'satisfechos', 'satisfecha', 'satisfechas',
            'dirigido', 'dirigidos', 'dirigida', 'dirigidas',  # specifically for the bug case
        }
        
        # English past participles: -ed, -en, -t (and common irregulars)
        english_participle_endings = ('ed', 'en', 'wn', 'ne', 'nt')
        english_irregular_participles = {
            'been', 'done', 'gone', 'seen', 'taken', 'given', 'known', 'shown',
            'grown', 'drawn', 'thrown', 'blown', 'flown', 'chosen', 'frozen',
            'spoken', 'broken', 'stolen', 'written', 'driven', 'ridden', 'risen',
            'bitten', 'hidden', 'forbidden', 'forgiven', 'forgotten', 'gotten',
            'eaten', 'beaten', 'fallen', 'shaken', 'taken', 'woken', 'worn',
            'torn', 'born', 'sworn', 'bought', 'brought', 'caught', 'fought',
            'thought', 'taught', 'sought', 'left', 'lost', 'sent', 'spent',
            'built', 'burnt', 'dealt', 'felt', 'kept', 'slept', 'meant', 'met',
            'paid', 'said', 'sold', 'told', 'heard', 'held', 'led', 'read',
            'fed', 'fled', 'bled', 'bred', 'sped', 'shed', 'spread', 'wed',
            'set', 'cut', 'hit', 'hurt', 'put', 'shut', 'split', 'quit',
            'cast', 'cost', 'burst', 'thrust', 'knit', 'spit',
            'made', 'had', 'laid', 'stood', 'understood', 'sat', 'shot', 'got',
        }
        
        # French past participles: -é, -i, -u, -s, -t (and irregulars)
        french_participle_endings = ('é', 'ée', 'és', 'ées', 'i', 'ie', 'is', 'ies',
                                      'u', 'ue', 'us', 'ues', 'it', 'ite', 'its', 'ites')
        french_irregular_participles = {
            'fait', 'faite', 'faits', 'faites',
            'dit', 'dite', 'dits', 'dites',
            'écrit', 'écrite', 'écrits', 'écrites',
            'pris', 'prise', 'prises',
            'mis', 'mise', 'mises',
            'ouvert', 'ouverte', 'ouverts', 'ouvertes',
            'offert', 'offerte', 'offerts', 'offertes',
            'mort', 'morte', 'morts', 'mortes',
            'né', 'née', 'nés', 'nées',
            'été',  # être
            'eu', 'eue', 'eus', 'eues',  # avoir
            'su', 'sue', 'sus', 'sues',  # savoir
            'pu',  # pouvoir
            'voulu', 'voulue', 'voulus', 'voulues',
            'dû', 'due', 'dus', 'dues',  # devoir
            'vu', 'vue', 'vus', 'vues',  # voir
        }
        
        # German past participles: typically start with ge- and end with -t or -en
        german_participle_patterns = ('ge',)  # Start pattern
        german_participle_endings = ('t', 'en')
        german_irregular_participles = {
            'gewesen', 'gehabt', 'geworden', 'gegangen', 'gekommen', 'gesehen',
            'gegeben', 'genommen', 'gefunden', 'gewusst', 'gedacht', 'gebracht',
            'genannt', 'gekannt', 'gelassen', 'getan', 'gestanden', 'verstanden',
            'begonnen', 'gewonnen', 'gesprochen', 'getroffen', 'geschrieben',
            'gelesen', 'gehalten', 'gefallen', 'geschlafen', 'getragen',
            'gefahren', 'gezogen', 'geflogen', 'gewachsen', 'geboren',
        }
        
        # Check Spanish
        if word.endswith(spanish_participle_endings) or word in spanish_irregular_participles:
            return True
        
        # Check English
        if word.endswith(english_participle_endings) or word in english_irregular_participles:
            return True
        
        # Check French
        if word.endswith(french_participle_endings) or word in french_irregular_participles:
            return True
        
        # Check German (starts with ge- and ends with -t or -en)
        if word.startswith(german_participle_patterns):
            if word.endswith(german_participle_endings):
                return True
        if word in german_irregular_participles:
            return True
        
        return False
    
    def _check_semantic_break(self, words: List[str], current_index: int) -> bool:
        """
        Check if there's a semantic break at the current position.
        
        Uses SentenceTransformer model to compute similarity between
        sentences before and after the potential break.
        
        Args:
            words: All words
            current_index: Current word index
        
        Returns:
            True if semantic break detected
        """
        if self.model is None:
            return False
        
        try:
            # Get text before and after
            before = ' '.join(words[max(0, current_index - 10):current_index + 1])
            after = ' '.join(words[current_index + 1:min(len(words), current_index + 11)])
            
            if not before or not after:
                return False
            
            # Compute embeddings
            embeddings = self.model.encode([before, after])
            
            # Compute cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            # Lower similarity = more likely to be a break
            # Threshold can be tuned
            threshold = 0.75
            return similarity < threshold
        except Exception as e:
            self.logger.debug(f"Semantic check failed: {e}")
            return False
    
    def _get_speaker_at_word(
        self,
        word_index: int,
        speaker_segments: Optional[List[Dict]]
    ) -> Optional[str]:
        """
        Get the speaker label at a given word position.
        
        Args:
            word_index: The word index to check
            speaker_segments: List of speaker segments (end_word is EXCLUSIVE, like Python slices)
        
        Returns:
            Speaker label if found, None otherwise
        """
        if not speaker_segments:
            return None
        
        for segment in speaker_segments:
            # Note: end_word is EXCLUSIVE (per _convert_char_ranges_to_word_ranges)
            if segment['start_word'] <= word_index < segment['end_word']:
                return segment['speaker']
        
        return None
    
    def _process_whisper_punctuation(
        self,
        sentences: List[Sentence],  # Now accepts Sentence objects
        whisper_periods: Dict[int, str],
        speaker_word_segments: Optional[List[Dict]],
        original_text: str
    ) -> List[str]:
        """
        Remove Whisper periods before same-speaker connectors.
        
        This SOLVES the "trabajo. Y este meta" bug.
        
        Args:
            sentences: List of sentences
            whisper_periods: Dict of Whisper punctuation positions
            speaker_word_segments: Speaker segments for continuity checks
            original_text: The original text for word mapping
        
        Returns:
            Processed sentences with appropriate periods removed/merged
        """
        if not speaker_word_segments:
            # Without speaker info, we can't safely remove periods
            return sentences
        
        self.logger.debug(f"_process_whisper_punctuation: Processing {len(sentences)} sentences")
        
        processed = []
        words = original_text.split()
        current_word_idx = 0
        skip_next = False
        
        for i, sentence_obj in enumerate(sentences):
            if skip_next:
                skip_next = False
                continue
            
            # Extract text from Sentence object if needed
            if isinstance(sentence_obj, Sentence):
                sentence = sentence_obj.text
            else:
                sentence = sentence_obj
                
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)
            
            self.logger.debug(f"  Sentence {i}: words={sentence_word_count}, start_idx={current_word_idx}, text='{sentence[:60]}...')")
            
            if i + 1 < len(sentences):
                next_sentence_obj = sentences[i + 1]
                # Extract text from next Sentence object if needed
                if isinstance(next_sentence_obj, Sentence):
                    next_sentence = next_sentence_obj.text
                else:
                    next_sentence = next_sentence_obj
                next_words = next_sentence.split()
                
                if not next_words:
                    processed.append(sentence_obj)  # Append original Sentence object
                    current_word_idx += sentence_word_count
                    continue
                
                # Check if next sentence starts with connector
                next_first_word = next_words[0]
                next_word_clean = next_first_word.lower().strip('.,;:!?¿¡')
                
                self.logger.debug(f"    Next word: '{next_first_word}', clean: '{next_word_clean}', is_connector: {next_word_clean in self.CONNECTOR_WORDS}")
                
                if next_word_clean in self.CONNECTOR_WORDS:
                    # Check if same speaker continues
                    sentence_end_word_idx = current_word_idx + sentence_word_count - 1
                    next_start_word_idx = current_word_idx + sentence_word_count
                    
                    speaker_at_end = self._get_speaker_at_word(sentence_end_word_idx, speaker_word_segments)
                    speaker_at_next = self._get_speaker_at_word(next_start_word_idx, speaker_word_segments)
                    
                    self.logger.debug(
                        f"    Connector detected: '{next_word_clean}', "
                        f"speaker_at_end={speaker_at_end} (word {sentence_end_word_idx}), "
                        f"speaker_at_next={speaker_at_next} (word {next_start_word_idx})"
                    )
                    
                    if speaker_at_end == speaker_at_next and speaker_at_end is not None:
                        # Same speaker continues - remove period and merge with lowercased connector
                        sentence = sentence.rstrip('.!?')
                        
                        # Merge: current sentence + connector (lowercased) + rest of next sentence
                        connector_lowercased = next_word_clean
                        rest_of_next = ' '.join(next_words[1:]) if len(next_words) > 1 else ''
                        
                        if rest_of_next:
                            merged_text = f"{sentence} {connector_lowercased} {rest_of_next}"
                        else:
                            merged_text = f"{sentence} {connector_lowercased}"
                        
                        # Create merged Sentence object
                        if isinstance(sentence_obj, Sentence) and isinstance(next_sentence_obj, Sentence):
                            # Combine utterances from both sentences
                            merged_utterances = sentence_obj.utterances + next_sentence_obj.utterances
                            merged_obj = Sentence(
                                text=merged_text,
                                utterances=merged_utterances,
                                speaker=sentence_obj.speaker
                            )
                            processed.append(merged_obj)
                        else:
                            # Fallback for backward compatibility
                            processed.append(merged_text)
                        
                        # Track that we removed a period
                        self.removed_periods.append({
                            'position': sentence_end_word_idx,
                            'reason': 'same_speaker_connector',
                            'speaker': speaker_at_end,
                            'connector': next_word_clean
                        })
                        
                        self.logger.debug(
                            f"MERGED: Removed period before same-speaker connector '{next_word_clean}' "
                            f"(speaker={speaker_at_end})"
                        )
                        
                        # Skip the next sentence since we merged it
                        current_word_idx += sentence_word_count + len(next_words)
                        skip_next = True
                        continue
            
            # Normal case: no merging needed
            if sentence:  # Only add non-empty sentences
                processed.append(sentence_obj)  # Append original Sentence object
            current_word_idx += sentence_word_count
        
        self.logger.debug(f"_process_whisper_punctuation: Returned {len(processed)} sentences (merged {len(sentences) - len(processed)})")
        return processed
    
    def _build_metadata(self) -> Dict:
        """
        Build metadata about the splitting process.
        
        Returns:
            Dict with split_metadata, removed_periods, added_periods
        """
        return {
            'split_metadata': self.split_metadata,
            'removed_periods': self.removed_periods,
            'added_periods': self.added_periods,
        }

