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
Sentence formatting and post-processing consolidation module.

This module consolidates all post-processing merge operations that were
previously scattered across podscripter.py:
- Domain merges: "example." + "com" → "example.com"
- Decimal merges: "99." + "9%" → "99.9%"
- Spanish appositive merges: ", de Texas. Estados Unidos" → ", de Texas, Estados Unidos"
- Emphatic word merges: "No. No. No." → "No, no, no."

Benefits:
- Single location for all merge logic
- Speaker-aware merges (never merge different speakers)
- Merge provenance tracking for debugging
- Clear separation from splitting logic
"""

import re
import logging
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

# Import Sentence and Utterance from sentence_splitter
from sentence_splitter import Sentence, Utterance

logger = logging.getLogger("podscripter.formatter")


@dataclass
class MergeMetadata:
    """Metadata about a merge operation."""
    merge_type: str  # 'domain', 'decimal', 'appositive', 'emphatic'
    sentence1_idx: int
    sentence2_idx: int
    reason: str
    speaker1: Optional[str]
    speaker2: Optional[str]
    before_text1: str
    before_text2: str
    after_text: str


class SentenceFormatter:
    """Unified sentence formatting with speaker-aware merge operations."""
    
    def __init__(self, language: str, speaker_segments: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize the SentenceFormatter.
        
        Args:
            language: Language code (e.g., 'es', 'en', 'fr', 'de')
            speaker_segments: Optional speaker segment data with word ranges
                             Format: [{'speaker': 'SPEAKER_00', 'start_word': 0, 'end_word': 10}, ...]
                             When None, speaker boundary checks are bypassed (backward compatible)
        """
        self.language = language.lower() if language else 'en'
        self.speaker_segments = speaker_segments
        self.merge_history: List[MergeMetadata] = []
        
        # Build sentence-to-speaker mapping if speaker data is available
        self._sentence_speakers: Dict[int, str] = {}
        
        # Will be populated in format() to map sentence_idx -> (start_word, end_word)
        self._sentence_word_ranges: Dict[int, Tuple[int, int]] = {}
    
    @staticmethod
    def _lowercase_first_letter(text: str) -> str:
        """
        Lowercase the first alphabetic character in text.
        Used when merging sentences to fix mid-sentence capitalization from punctuation restorer.
        
        Example: "Y aquí puedes..." -> "y aquí puedes..."
        """
        if not text:
            return text
        for i, char in enumerate(text):
            if char.isalpha():
                return text[:i] + char.lower() + text[i+1:]
        return text
    
    def format(self, sentences: List[Sentence]) -> Tuple[List[Sentence], List[MergeMetadata]]:
        """
        Apply all formatting operations in order.
        
        Args:
            sentences: List of Sentence objects to format
            
        Returns:
            Tuple of (formatted_sentences, merge_metadata)
        """
        logger.debug(f"SentenceFormatter.format() called with {len(sentences)} sentences")
        # Extract text for logging
        sentence_texts = [s.text if isinstance(s, Sentence) else s for s in sentences]
        logger.debug(f"First 5 sentences: {sentence_texts[:5]}")
        logger.debug(f"Speaker segments available: {self.speaker_segments is not None}")
        if self.speaker_segments:
            logger.debug(f"Number of speaker segments: {len(self.speaker_segments)}")
        
        # Build word-range mapping for speaker lookups
        self._build_sentence_word_ranges(sentences)
        
        # Order matters: patterns (domains/decimals) first, then language-specific formatting
        # NOTE: After each merge, we rebuild word ranges since sentence indices change
        initial_count = len(sentences)
        sentences = self._merge_domains(sentences)
        self._build_sentence_word_ranges(sentences)
        logger.debug(f"After domain merge: {len(sentences)} sentences (was {initial_count})")
        
        initial_count = len(sentences)
        sentences = self._merge_decimals(sentences)
        self._build_sentence_word_ranges(sentences)
        logger.debug(f"After decimal merge: {len(sentences)} sentences (was {initial_count})")
        
        if self.language == 'es':
            initial_count = len(sentences)
            sentences = self._merge_spanish_appositives(sentences)
            self._build_sentence_word_ranges(sentences)
            logger.debug(f"After appositive merge: {len(sentences)} sentences (was {initial_count})")
        
        initial_count = len(sentences)
        sentences = self._merge_emphatic_words(sentences)
        logger.debug(f"After emphatic merge: {len(sentences)} sentences (was {initial_count})")
        # No need to rebuild after last merge
        
        logger.debug(f"SentenceFormatter.format() complete: {len(self.merge_history)} merge operations recorded")
        
        return sentences, self.merge_history
    
    def _build_sentence_word_ranges(self, sentences: List[Sentence]) -> None:
        """
        Build mapping of sentence indices to word ranges.
        
        This allows us to look up which speaker(s) correspond to each sentence.
        
        Args:
            sentences: List of Sentence objects
        """
        current_word_idx = 0
        
        for sent_idx, sentence in enumerate(sentences):
            # Extract text from Sentence object
            sentence_text = sentence.text if isinstance(sentence, Sentence) else sentence
            # Count words in this sentence
            word_count = len(sentence_text.split())
            
            # Store the range
            if word_count > 0:
                start_word = current_word_idx
                end_word = current_word_idx + word_count - 1
                self._sentence_word_ranges[sent_idx] = (start_word, end_word)
                current_word_idx += word_count
                
                # Debug logging for first few sentences
                if sent_idx < 5 and self.speaker_segments:
                    logger.debug(f"Sentence {sent_idx} words [{start_word}:{end_word}]: {sentence_text[:60]}...")
            else:
                # Empty sentence (shouldn't happen but handle gracefully)
                self._sentence_word_ranges[sent_idx] = (current_word_idx, current_word_idx)
        
        # Log speaker segments info
        if self.speaker_segments and len(sentences) > 0:
            logger.debug(f"Total sentences: {len(sentences)}, Total word positions: {current_word_idx}")
            logger.debug(f"Speaker segments available: {len(self.speaker_segments)}")
            if len(self.speaker_segments) > 0:
                logger.debug(f"First speaker segment: {self.speaker_segments[0]}")
                logger.debug(f"Last speaker segment: {self.speaker_segments[-1]}")
    
    def _get_speaker_for_sentence(self, sentence: Sentence, sentence_idx: int = None) -> Optional[str]:
        """
        Get the speaker for a given sentence.
        
        Args:
            sentence: Sentence object
            sentence_idx: Optional index for caching
            
        Returns:
            Speaker label or None if no speaker data available
        """
        # If sentence has utterances, use first utterance's speaker
        if isinstance(sentence, Sentence) and sentence.utterances:
            return sentence.utterances[0].speaker if sentence.utterances else None
        
        # Fallback to speaker field
        if isinstance(sentence, Sentence):
            return sentence.speaker
        
        # Legacy: use word range lookup if needed
        if not self.speaker_segments or sentence_idx is None:
            return None
        
        # Return cached result if available
        if sentence_idx in self._sentence_speakers:
            return self._sentence_speakers[sentence_idx]
        
        # Get word range for this sentence
        if sentence_idx not in self._sentence_word_ranges:
            logger.warning(f"Sentence {sentence_idx} not in word ranges!")
            return None
        
        start_word, end_word = self._sentence_word_ranges[sentence_idx]
        
        # Find which speaker segment(s) overlap with this word range
        # speaker_segments format: [{'start_word': 0, 'end_word': 10, 'speaker': 'SPEAKER_01'}, ...]
        speakers_in_sentence = set()
        
        for seg in self.speaker_segments:
            seg_start = seg['start_word']
            seg_end = seg['end_word']
            
            # Check if this speaker segment overlaps with the sentence's word range
            if seg_start <= end_word and seg_end >= start_word:
                speakers_in_sentence.add(seg['speaker'])
        
        # Determine the dominant speaker
        if len(speakers_in_sentence) == 0:
            # No speaker data for this sentence
            logger.debug(f"Sentence {sentence_idx} words [{start_word}:{end_word}]: NO SPEAKER FOUND")
            speaker = None
        elif len(speakers_in_sentence) == 1:
            # Single speaker - common case
            speaker = list(speakers_in_sentence)[0]
            logger.debug(f"Sentence {sentence_idx} words [{start_word}:{end_word}]: speaker={speaker}")
        else:
            # Multiple speakers in one sentence - use first speaker we found
            # This can happen at sentence boundaries where diarization is uncertain
            logger.debug(f"Sentence {sentence_idx} words [{start_word}:{end_word}]: multiple speakers {speakers_in_sentence}")
            speaker = list(speakers_in_sentence)[0]
        
        # Cache and return
        self._sentence_speakers[sentence_idx] = speaker
        return speaker
    
    def _should_merge(self, sent1: Sentence, sent2: Sentence, idx1: int, idx2: int, merge_type: str) -> Tuple[bool, str]:
        """
        Determine if two sentences should be merged.
        
        CRITICAL: Never merge different speakers.
        
        Args:
            sent1: First Sentence object
            sent2: Second Sentence object
            idx1: Index of first sentence
            idx2: Index of second sentence
            merge_type: Type of merge being considered
            
        Returns:
            Tuple of (should_merge, reason)
        """
        # Get speakers for both sentences
        speaker1 = self._get_speaker_for_sentence(sent1, idx1)
        speaker2 = self._get_speaker_for_sentence(sent2, idx2)
        
        # CRITICAL: If both speakers are known and they differ, never merge
        if speaker1 and speaker2 and speaker1 != speaker2:
            reason = f"speaker_boundary_conflict: {speaker1} != {speaker2}"
            logger.debug(f"Skipping {merge_type} merge of sentences {idx1}+{idx2}: {reason}")
            return False, reason
        
        # Allow merge - either no speaker data, or speakers match, or one/both unknown
        return True, "allowed"
    
    def _merge_domains(self, sentences: List[Sentence]) -> List[Sentence]:
        """
        Merge split domains: 'example.' + 'com' → 'example.com'
        
        Includes natural language guards to prevent false positives like:
        "jugar." + "Es que..." → "jugar.es" (incorrect)
        
        Args:
            sentences: List of Sentence objects
            
        Returns:
            List with domain splits merged
        """
        if not sentences:
            return sentences
        
        merged: List[Sentence] = []
        i = 0
        tlds = r"com|net|org|co|es|io|edu|gov|uk|us|ar|mx"
        
        while i < len(sentences):
            cur_obj = sentences[i]
            cur = (cur_obj.text if isinstance(cur_obj, Sentence) else cur_obj).strip()
            
            if i + 1 < len(sentences):
                nxt_obj = sentences[i + 1]
                nxt = (nxt_obj.text if isinstance(nxt_obj, Sentence) else nxt_obj).strip()
                m1 = re.search(r"([A-Za-z0-9\-]+)\.$", cur)
                m2 = re.match(rf"^({tlds})(\b|\W)(.*)$", nxt, flags=re.IGNORECASE)
                
                if m1 and m2:
                    label = m1.group(1)
                    tld = m2.group(1).lower()
                    remainder = (m2.group(3) or '').lstrip()
                    
                    # Natural language guards (v0.4.4)
                    # Only merge if sentence is short (< 50 chars) OR label is capitalized
                    # This prevents "jugar." + "Es que..." from being merged as "jugar.es"
                    is_short_sentence = len(cur) < 50
                    is_capitalized_label = label[0].isupper() if label else False
                    
                    if not (is_short_sentence or is_capitalized_label):
                        # Skip this merge - likely natural language, not a domain
                        logger.debug(f"Skipping domain merge (natural language guard): '{cur}' + '{nxt}'")
                        merged.append(cur_obj)
                        i += 1
                        continue
                    
                    # Check speaker boundaries
                    should_merge, reason = self._should_merge(cur_obj, nxt_obj, i, i + 1, 'domain')
                    if not should_merge:
                        # Record skipped merge
                        self.merge_history.append(MergeMetadata(
                            merge_type='domain',
                            sentence1_idx=i,
                            sentence2_idx=i + 1,
                            reason=f'skipped: {reason}',
                            speaker1=self._get_speaker_for_sentence(cur_obj, i),
                            speaker2=self._get_speaker_for_sentence(nxt_obj, i + 1),
                            before_text1=cur,
                            before_text2=nxt,
                            after_text=''
                        ))
                        merged.append(cur_obj)
                        i += 1
                        continue
                    
                    # Perform merge
                    merged_text = cur[:-1] + "." + tld
                    
                    # Check for triple merge (domain split across 3 sentences)
                    if (not remainder or remainder in ('.', '!', '?')) and i + 2 < len(sentences):
                        third_obj = sentences[i + 2]
                        third = (third_obj.text if isinstance(third_obj, Sentence) else third_obj).strip()
                        if third:
                            # Lowercase first letter when merging mid-sentence
                            third_lowercase = self._lowercase_first_letter(third)
                            merged_text = merged_text + " " + third_lowercase
                            
                            # Combine utterances from all three sentences
                            merged_utterances = []
                            if isinstance(cur_obj, Sentence):
                                merged_utterances.extend(cur_obj.utterances)
                            if isinstance(nxt_obj, Sentence):
                                merged_utterances.extend(nxt_obj.utterances)
                            if isinstance(third_obj, Sentence):
                                merged_utterances.extend(third_obj.utterances)
                            
                            # Create merged Sentence object
                            merged_sentence_obj = Sentence(
                                text=merged_text,
                                utterances=merged_utterances,
                                speaker=cur_obj.speaker if isinstance(cur_obj, Sentence) else None
                            )
                            
                            # Record triple merge
                            self.merge_history.append(MergeMetadata(
                                merge_type='domain',
                                sentence1_idx=i,
                                sentence2_idx=i + 2,
                                reason='triple_merge',
                                speaker1=self._get_speaker_for_sentence(cur_obj, i),
                                speaker2=self._get_speaker_for_sentence(third_obj, i + 2),
                                before_text1=f"{cur} | {nxt} | {third}",
                                before_text2='',
                                after_text=merged_text
                            ))
                            
                            merged.append(merged_sentence_obj)
                            i += 3
                            continue
                    
                    # Regular merge (2 sentences)
                    if remainder:
                        if remainder.startswith(('.', '!', '?')):
                            merged_text = merged_text + remainder
                        else:
                            # Lowercase first letter when merging mid-sentence
                            remainder_lowercase = self._lowercase_first_letter(remainder)
                            merged_text = (merged_text + " " + remainder_lowercase).strip()
                    
                    # Combine utterances from both sentences
                    merged_utterances = []
                    if isinstance(cur_obj, Sentence):
                        merged_utterances.extend(cur_obj.utterances)
                    if isinstance(nxt_obj, Sentence):
                        merged_utterances.extend(nxt_obj.utterances)
                    
                    # Create merged Sentence object
                    merged_sentence_obj = Sentence(
                        text=merged_text,
                        utterances=merged_utterances,
                        speaker=cur_obj.speaker if isinstance(cur_obj, Sentence) else None
                    )
                    
                    # Record merge
                    self.merge_history.append(MergeMetadata(
                        merge_type='domain',
                        sentence1_idx=i,
                        sentence2_idx=i + 1,
                        reason='domain_pattern_match',
                        speaker1=self._get_speaker_for_sentence(cur_obj, i),
                        speaker2=self._get_speaker_for_sentence(nxt_obj, i + 1),
                        before_text1=cur,
                        before_text2=nxt,
                        after_text=merged_text
                    ))
                    
                    merged.append(merged_sentence_obj)
                    i += 2
                    continue
            
            merged.append(cur_obj)
            i += 1
        
        return merged
    
    def _merge_decimals(self, sentences: List[Sentence]) -> List[Sentence]:
        """
        Merge split decimals: '99.' + '9%' → '99.9%'
        
        Handles patterns like:
        - "99." + "9% de los casos"
        - "121." + "73 meters"
        
        Args:
            sentences: List of Sentence objects
            
        Returns:
            List with decimal splits merged
        """
        if not sentences:
            return sentences
        
        merged: List[Sentence] = []
        i = 0
        
        while i < len(sentences):
            cur_obj = sentences[i]
            cur = (cur_obj.text if isinstance(cur_obj, Sentence) else cur_obj).strip()
            
            if i + 1 < len(sentences):
                nxt_obj = sentences[i + 1]
                nxt = (nxt_obj.text if isinstance(nxt_obj, Sentence) else nxt_obj).strip()
                m1 = re.search(r"(\d{1,3})\.$", cur)
                m2 = re.match(r"^(\d{1,3})(%?)\s*(.*)$", nxt)
                
                if m1 and m2:
                    # Check speaker boundaries
                    should_merge, reason = self._should_merge(cur_obj, nxt_obj, i, i + 1, 'decimal')
                    if not should_merge:
                        # Record skipped merge
                        self.merge_history.append(MergeMetadata(
                            merge_type='decimal',
                            sentence1_idx=i,
                            sentence2_idx=i + 1,
                            reason=f'skipped: {reason}',
                            speaker1=self._get_speaker_for_sentence(cur_obj, i),
                            speaker2=self._get_speaker_for_sentence(nxt_obj, i + 1),
                            before_text1=cur,
                            before_text2=nxt,
                            after_text=''
                        ))
                        merged.append(cur_obj)
                        i += 1
                        continue
                    
                    # Perform merge
                    frac = m2.group(1)
                    percent = m2.group(2) or ''
                    remainder = (m2.group(3) or '').lstrip()
                    # Build merged decimal: "99." + "9%" → "99.9%"
                    merged_text = cur[:-1] + "." + frac + percent
                    
                    if remainder:
                        # Add space before remainder text, lowercasing first letter
                        remainder_lowercase = self._lowercase_first_letter(remainder)
                        merged_text = merged_text + " " + remainder_lowercase
                    
                    # Combine utterances from both sentences
                    merged_utterances = []
                    if isinstance(cur_obj, Sentence):
                        merged_utterances.extend(cur_obj.utterances)
                    if isinstance(nxt_obj, Sentence):
                        merged_utterances.extend(nxt_obj.utterances)
                    
                    # Create merged Sentence object
                    merged_sentence_obj = Sentence(
                        text=merged_text,
                        utterances=merged_utterances,
                        speaker=cur_obj.speaker if isinstance(cur_obj, Sentence) else None
                    )
                    
                    # Record merge
                    self.merge_history.append(MergeMetadata(
                        merge_type='decimal',
                        sentence1_idx=i,
                        sentence2_idx=i + 1,
                        reason='decimal_pattern_match',
                        speaker1=self._get_speaker_for_sentence(cur_obj, i),
                        speaker2=self._get_speaker_for_sentence(nxt_obj, i + 1),
                        before_text1=cur,
                        before_text2=nxt,
                        after_text=merged_text
                    ))
                    
                    merged.append(merged_sentence_obj)
                    i += 2
                    continue
            
            merged.append(cur_obj)
            i += 1
        
        return merged
    
    def _merge_spanish_appositives(self, sentences: List[Sentence]) -> List[Sentence]:
        """
        Merge Spanish location appositives (ES only).
        
        Reformats: ", de Texas. Estados Unidos" → ", de Texas, Estados Unidos"
        
        Args:
            sentences: List of Sentence objects
            
        Returns:
            List with Spanish appositives merged
        """
        if not sentences:
            return sentences
        
        try:
            from punctuation_restorer import _es_merge_appositive_location_breaks as es_merge_appos
            
            # Extract text for processing
            sentence_texts = [s.text if isinstance(s, Sentence) else s for s in sentences]
            
            # Call the existing helper function
            result_texts = es_merge_appos(sentence_texts)
            
            # Reconstruct Sentence objects
            # Note: This is a simplified approach - we're assuming the merge function
            # preserves the order and we can map back to original Sentence objects
            # For now, we'll just wrap the text results in Sentence objects
            result = []
            for text in result_texts:
                # Try to find matching original sentence
                found = False
                for orig_sent in sentences:
                    if isinstance(orig_sent, Sentence) and orig_sent.text == text:
                        result.append(orig_sent)
                        found = True
                        break
                if not found:
                    # This is a merged sentence - create new Sentence object without utterances
                    result.append(Sentence(text=text, utterances=[], speaker=None))
            
            return result
        except Exception as e:
            logger.warning(f"Failed to merge Spanish appositives: {e}")
            return sentences
    
    def _merge_emphatic_words(self, sentences: List[Sentence]) -> List[Sentence]:
        """
        Merge emphatic word repeats: 'No. No. No.' → 'No, no, no.'
        
        Handles language-specific emphatic words:
        - Spanish: no, si/sí
        - French: non, oui
        - German: nein, ja
        
        Args:
            sentences: List of Sentence objects
            
        Returns:
            List with emphatic repeats merged
        """
        if not sentences:
            return sentences
        
        emph_map = {
            'es': {'no', 'si', 'sí'},
            'fr': {'non', 'oui'},
            'de': {'nein', 'ja'},
        }
        
        allowed = emph_map.get(self.language, set())
        if not allowed:
            return sentences  # No emphatic words defined for this language
        
        def _is_emphatic(word: str) -> bool:
            w = word.strip().strip('.!?').lower()
            return w in allowed
        
        merged: List[Sentence] = []
        i = 0
        
        while i < len(sentences):
            cur_obj = sentences[i]
            cur = (cur_obj.text if isinstance(cur_obj, Sentence) else cur_obj).strip()
            
            if _is_emphatic(cur):
                # Collect consecutive emphatic words
                words = []
                sentence_objs = []
                start_idx = i
                
                while i < len(sentences):
                    sent_obj = sentences[i]
                    sent_text = (sent_obj.text if isinstance(sent_obj, Sentence) else sent_obj).strip()
                    if _is_emphatic(sent_text):
                        words.append(sent_text.strip('.!?'))
                        sentence_objs.append(sent_obj)
                        i += 1
                    else:
                        break
                
                if words:
                    # Check speaker boundaries across all collected emphatic words
                    # For simplicity, check first vs last
                    should_merge = True
                    if len(words) > 1:
                        should_merge, reason = self._should_merge(
                            sentence_objs[0], 
                            sentence_objs[-1], 
                            start_idx, 
                            i - 1, 
                            'emphatic'
                        )
                    
                    if not should_merge:
                        # Don't merge - add them separately
                        logger.debug(f"Skipping emphatic merge of {len(words)} words: {reason}")
                        for sent_obj in sentence_objs:
                            merged.append(sent_obj)
                        continue
                    
                    # Normalize accents for Spanish
                    if self.language == 'es':
                        norm = ['sí' if w.lower() in {'si', 'sí'} else 'no' for w in words]
                    else:
                        norm = [w.lower() for w in words]
                    
                    out = norm[0].capitalize()
                    if len(norm) > 1:
                        out += ', ' + ', '.join(norm[1:])
                    if not out.endswith(('.', '!', '?')):
                        out += '.'
                    
                    # Combine utterances from all merged sentences
                    merged_utterances = []
                    for sent_obj in sentence_objs:
                        if isinstance(sent_obj, Sentence):
                            merged_utterances.extend(sent_obj.utterances)
                    
                    # Create merged Sentence object
                    merged_sentence_obj = Sentence(
                        text=out,
                        utterances=merged_utterances,
                        speaker=sentence_objs[0].speaker if isinstance(sentence_objs[0], Sentence) else None
                    )
                    
                    # Record merge
                    self.merge_history.append(MergeMetadata(
                        merge_type='emphatic',
                        sentence1_idx=start_idx,
                        sentence2_idx=i - 1,
                        reason=f'emphatic_repeat_{len(words)}_words',
                        speaker1=self._get_speaker_for_sentence(sentence_objs[0], start_idx),
                        speaker2=self._get_speaker_for_sentence(sentence_objs[-1], i - 1),
                        before_text1=' | '.join(words),
                        before_text2='',
                        after_text=out
                    ))
                    
                    merged.append(merged_sentence_obj)
                    continue
            
            merged.append(cur_obj)
            i += 1
        
        return merged
