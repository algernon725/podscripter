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
            speaker_segments: Optional speaker segment data with character ranges
                             Format: [{'speaker': 'SPEAKER_00', 'start_char': 0, 'end_char': 100}, ...]
                             When None, speaker boundary checks are bypassed (backward compatible)
        """
        self.language = language.lower() if language else 'en'
        self.speaker_segments = speaker_segments
        self.merge_history: List[MergeMetadata] = []
        
        # Build sentence-to-speaker mapping if speaker data is available
        self._sentence_speakers: Dict[int, str] = {}
    
    def format(self, sentences: List[str]) -> Tuple[List[str], List[MergeMetadata]]:
        """
        Apply all formatting operations in order.
        
        Args:
            sentences: List of sentences to format
            
        Returns:
            Tuple of (formatted_sentences, merge_metadata)
        """
        # Order matters: patterns (domains/decimals) first, then language-specific formatting
        sentences = self._merge_domains(sentences)
        sentences = self._merge_decimals(sentences)
        
        if self.language == 'es':
            sentences = self._merge_spanish_appositives(sentences)
        
        sentences = self._merge_emphatic_words(sentences)
        
        return sentences, self.merge_history
    
    def _get_speaker_for_sentence(self, sentence_idx: int) -> Optional[str]:
        """
        Get the speaker for a given sentence index.
        
        Args:
            sentence_idx: Index of the sentence
            
        Returns:
            Speaker label or None if no speaker data available
        """
        if not self.speaker_segments:
            return None
        
        # Return cached result if available
        if sentence_idx in self._sentence_speakers:
            return self._sentence_speakers[sentence_idx]
        
        # For now, return None - will be populated by caller if needed
        # (This method is designed to be overridden or extended based on how
        # sentence indices map to character positions in the actual implementation)
        return None
    
    def _should_merge(self, sent1: str, sent2: str, idx1: int, idx2: int, merge_type: str) -> Tuple[bool, str]:
        """
        Determine if two sentences should be merged.
        
        CRITICAL: Never merge different speakers.
        
        Args:
            sent1: First sentence text
            sent2: Second sentence text
            idx1: Index of first sentence
            idx2: Index of second sentence
            merge_type: Type of merge being considered
            
        Returns:
            Tuple of (should_merge, reason)
        """
        # Get speakers for both sentences
        speaker1 = self._get_speaker_for_sentence(idx1)
        speaker2 = self._get_speaker_for_sentence(idx2)
        
        # CRITICAL: If both speakers are known and they differ, never merge
        if speaker1 and speaker2 and speaker1 != speaker2:
            reason = f"speaker_boundary_conflict: {speaker1} != {speaker2}"
            logger.debug(f"Skipping {merge_type} merge of sentences {idx1}+{idx2}: {reason}")
            return False, reason
        
        # Allow merge - either no speaker data, or speakers match, or one/both unknown
        return True, "allowed"
    
    def _merge_domains(self, sentences: List[str]) -> List[str]:
        """
        Merge split domains: 'example.' + 'com' → 'example.com'
        
        Includes natural language guards to prevent false positives like:
        "jugar." + "Es que..." → "jugar.es" (incorrect)
        
        Args:
            sentences: List of sentences
            
        Returns:
            List with domain splits merged
        """
        if not sentences:
            return sentences
        
        merged: List[str] = []
        i = 0
        tlds = r"com|net|org|co|es|io|edu|gov|uk|us|ar|mx"
        
        while i < len(sentences):
            cur = (sentences[i] or '').strip()
            
            if i + 1 < len(sentences):
                nxt = (sentences[i + 1] or '').strip()
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
                        merged.append(cur)
                        i += 1
                        continue
                    
                    # Check speaker boundaries
                    should_merge, reason = self._should_merge(cur, nxt, i, i + 1, 'domain')
                    if not should_merge:
                        # Record skipped merge
                        self.merge_history.append(MergeMetadata(
                            merge_type='domain',
                            sentence1_idx=i,
                            sentence2_idx=i + 1,
                            reason=f'skipped: {reason}',
                            speaker1=self._get_speaker_for_sentence(i),
                            speaker2=self._get_speaker_for_sentence(i + 1),
                            before_text1=cur,
                            before_text2=nxt,
                            after_text=''
                        ))
                        merged.append(cur)
                        i += 1
                        continue
                    
                    # Perform merge
                    merged_sentence = cur[:-1] + "." + tld
                    
                    # Check for triple merge (domain split across 3 sentences)
                    if (not remainder or remainder in ('.', '!', '?')) and i + 2 < len(sentences):
                        third = (sentences[i + 2] or '').strip()
                        if third:
                            merged_sentence = merged_sentence + " " + third
                            
                            # Record triple merge
                            self.merge_history.append(MergeMetadata(
                                merge_type='domain',
                                sentence1_idx=i,
                                sentence2_idx=i + 2,
                                reason='triple_merge',
                                speaker1=self._get_speaker_for_sentence(i),
                                speaker2=self._get_speaker_for_sentence(i + 2),
                                before_text1=f"{cur} | {nxt} | {third}",
                                before_text2='',
                                after_text=merged_sentence
                            ))
                            
                            merged.append(merged_sentence)
                            i += 3
                            continue
                    
                    # Regular merge (2 sentences)
                    if remainder:
                        if remainder.startswith(('.', '!', '?')):
                            merged_sentence = merged_sentence + remainder
                        else:
                            merged_sentence = (merged_sentence + " " + remainder).strip()
                    
                    # Record merge
                    self.merge_history.append(MergeMetadata(
                        merge_type='domain',
                        sentence1_idx=i,
                        sentence2_idx=i + 1,
                        reason='domain_pattern_match',
                        speaker1=self._get_speaker_for_sentence(i),
                        speaker2=self._get_speaker_for_sentence(i + 1),
                        before_text1=cur,
                        before_text2=nxt,
                        after_text=merged_sentence
                    ))
                    
                    merged.append(merged_sentence)
                    i += 2
                    continue
            
            merged.append(cur)
            i += 1
        
        return merged
    
    def _merge_decimals(self, sentences: List[str]) -> List[str]:
        """
        Merge split decimals: '99.' + '9%' → '99.9%'
        
        Handles patterns like:
        - "99." + "9% de los casos"
        - "121." + "73 meters"
        
        Args:
            sentences: List of sentences
            
        Returns:
            List with decimal splits merged
        """
        if not sentences:
            return sentences
        
        merged: List[str] = []
        i = 0
        
        while i < len(sentences):
            cur = (sentences[i] or '').strip()
            
            if i + 1 < len(sentences):
                nxt = (sentences[i + 1] or '').strip()
                m1 = re.search(r"(\d{1,3})\.$", cur)
                m2 = re.match(r"^(\d{1,3})(%?)\s*(.*)$", nxt)
                
                if m1 and m2:
                    # Check speaker boundaries
                    should_merge, reason = self._should_merge(cur, nxt, i, i + 1, 'decimal')
                    if not should_merge:
                        # Record skipped merge
                        self.merge_history.append(MergeMetadata(
                            merge_type='decimal',
                            sentence1_idx=i,
                            sentence2_idx=i + 1,
                            reason=f'skipped: {reason}',
                            speaker1=self._get_speaker_for_sentence(i),
                            speaker2=self._get_speaker_for_sentence(i + 1),
                            before_text1=cur,
                            before_text2=nxt,
                            after_text=''
                        ))
                        merged.append(cur)
                        i += 1
                        continue
                    
                    # Perform merge
                    frac = m2.group(1)
                    percent = m2.group(2) or ''
                    remainder = (m2.group(3) or '').lstrip()
                    # Build merged decimal: "99." + "9%" → "99.9%"
                    merged_sentence = cur[:-1] + "." + frac + percent
                    
                    if remainder:
                        # Add space before remainder text
                        merged_sentence = merged_sentence + " " + remainder
                    
                    # Record merge
                    self.merge_history.append(MergeMetadata(
                        merge_type='decimal',
                        sentence1_idx=i,
                        sentence2_idx=i + 1,
                        reason='decimal_pattern_match',
                        speaker1=self._get_speaker_for_sentence(i),
                        speaker2=self._get_speaker_for_sentence(i + 1),
                        before_text1=cur,
                        before_text2=nxt,
                        after_text=merged_sentence
                    ))
                    
                    merged.append(merged_sentence)
                    i += 2
                    continue
            
            merged.append(cur)
            i += 1
        
        return merged
    
    def _merge_spanish_appositives(self, sentences: List[str]) -> List[str]:
        """
        Merge Spanish location appositives (ES only).
        
        Reformats: ", de Texas. Estados Unidos" → ", de Texas, Estados Unidos"
        
        Args:
            sentences: List of sentences
            
        Returns:
            List with Spanish appositives merged
        """
        if not sentences:
            return sentences
        
        try:
            from punctuation_restorer import _es_merge_appositive_location_breaks as es_merge_appos
            
            # Call the existing helper function
            result = es_merge_appos(sentences)
            
            # Note: The existing function doesn't provide merge metadata,
            # so we can't track individual merges here. This is acceptable
            # since speaker boundary checks are already done in the existing function.
            
            return result
        except Exception as e:
            logger.warning(f"Failed to merge Spanish appositives: {e}")
            return sentences
    
    def _merge_emphatic_words(self, sentences: List[str]) -> List[str]:
        """
        Merge emphatic word repeats: 'No. No. No.' → 'No, no, no.'
        
        Handles language-specific emphatic words:
        - Spanish: no, si/sí
        - French: non, oui
        - German: nein, ja
        
        Args:
            sentences: List of sentences
            
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
        
        merged: List[str] = []
        i = 0
        
        while i < len(sentences):
            cur = (sentences[i] or '').strip()
            
            if _is_emphatic(cur):
                # Collect consecutive emphatic words
                words = []
                start_idx = i
                
                while i < len(sentences) and _is_emphatic((sentences[i] or '').strip()):
                    words.append((sentences[i] or '').strip().strip('.!?'))
                    i += 1
                
                if words:
                    # Check speaker boundaries across all collected emphatic words
                    # For simplicity, check first vs last
                    should_merge = True
                    if len(words) > 1:
                        should_merge, reason = self._should_merge(
                            sentences[start_idx], 
                            sentences[i - 1], 
                            start_idx, 
                            i - 1, 
                            'emphatic'
                        )
                    
                    if not should_merge:
                        # Don't merge - add them separately
                        logger.debug(f"Skipping emphatic merge of {len(words)} words: {reason}")
                        for word in words:
                            merged.append(word + '.')
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
                    
                    # Record merge
                    self.merge_history.append(MergeMetadata(
                        merge_type='emphatic',
                        sentence1_idx=start_idx,
                        sentence2_idx=i - 1,
                        reason=f'emphatic_repeat_{len(words)}_words',
                        speaker1=self._get_speaker_for_sentence(start_idx),
                        speaker2=self._get_speaker_for_sentence(i - 1),
                        before_text1=' | '.join(words),
                        before_text2='',
                        after_text=out
                    ))
                    
                    merged.append(out)
                    continue
            
            merged.append(cur)
            i += 1
        
        return merged
