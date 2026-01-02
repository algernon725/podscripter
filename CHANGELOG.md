# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.2] - 2025-01-01

### Fixed
- **Speaker change separation (Critical)**: Fixed sentences from different speakers not being separated by blank lines
  - **Issue**: When speaker boundaries fell within Whisper segments (not at boundaries), the entire segment was assigned to one speaker
  - **Example**: "Estoy mejorando cada día con tu instrucción." (Nate) followed by "¡Nate! Este año..." (Andrea) appeared in same paragraph
  - **Root causes**:
    1. `SentenceSplitter._convert_segments_to_word_boundaries()` extracted boundaries from ALL speaker segments, not just where speaker changes
    2. `MIN_SPEAKER_SEGMENT_SEC` threshold in `speaker_diarization.py` was 2.0s, filtering out short utterances like "¡Uy, Nate!"
    3. `_convert_speaker_segments_to_char_ranges()` in `podscripter.py` used duration-based sorting that failed when speaker boundaries fell within Whisper segments
  - **Solution**: 
    - Modified `SentenceSplitter._convert_segments_to_word_boundaries()` to only extract boundaries where `speaker` label changes between consecutive segments
    - Lowered `MIN_SPEAKER_SEGMENT_SEC` from 2.0s to 0.5s to capture brief speaker changes
    - Rewrote speaker-to-Whisper assignment algorithm to assign each Whisper segment to the speaker with most **temporal overlap**, then group consecutive segments with same speaker
  - **Implementation**:
    - `sentence_splitter.py` (line 269-276): Loop through segments pairwise, only add `end_word` boundary if speakers differ
    - `speaker_diarization.py` (line 60): Changed threshold to 0.5s (filters only noise/artifacts)
    - `podscripter.py` (lines 613-703): New algorithm calculates overlap duration for each Whisper-speaker pair, assigns to best match, groups into ranges
  - **Impact**: All diarization-enabled transcriptions now correctly separate different speakers' utterances, while preserving same-speaker multi-sentence grouping
  - **Debug**: Added logging for speaker boundary split/skip decisions
- **Whisper periods at skipped boundaries (Known Limitation Resolved)**: Fixed Whisper-added periods remaining when Whisper boundaries are skipped
  - **Issue**: When a Whisper segment boundary was skipped (because a speaker boundary was nearby), the Whisper period remained
  - **Example**: `"ustedes."` + `"Mateo 712"` → `"ustedes. Mateo 712"` instead of `"ustedes Mateo 712"`
  - **Root cause**: Whisper adds periods to segment ends. Even though we skipped the boundary (no split), we didn't remove the period
  - **Solution**: Track skipped Whisper boundaries in `skipped_whisper_boundaries` set, then remove periods at those positions
  - **Implementation**: 
    - Moved Whisper boundary skip detection BEFORE `min_total_words_no_split` check (works in short texts now)
    - After `_should_end_sentence_here` returns, check if current word is at a skipped boundary
    - Remove trailing `.!?` from words at skipped boundary positions
    - Track removal with reason `'skipped_whisper_boundary'` for debugging
  - **Impact**: Resolves known limitation documented in AGENT.md lines 442-470. Affects short segments (< 3 words) preceding speaker changes
  - **Tests**: `test_whisper_skipped_boundary_detailed.py`, `test_whisper_boundary_debug.py`
- **Period-before-connector inline removal (Critical)**: Fixed bug where v0.4.0 refactor didn't remove periods in all code paths
  - **Issue**: `_evaluate_boundaries()` correctly decided NOT to split before connectors, but period remained in text
  - **Example**: `"Ama a tu prójimo como a ti mismo. Y también..."` still had unwanted period before "Y"
  - **Solution**: Added inline period removal when deciding not to split + connector lowercasing
  - **Impact**: Completes the period-before-same-speaker-connector fix introduced in v0.4.0
- **Trailing comma before terminal punctuation (Spanish)**: Fixed `", ?"` appearing at sentence ends
  - **Root cause**: Whisper's trailing commas weren't stripped before adding terminal punctuation
  - **Solution**: Added `.rstrip(',;: ')` before applying terminal punctuation in `_should_add_terminal_punctuation()`
  - **Example**: `"Sin importar si crees en Dios o no, ?"` → `"Sin importar si crees en Dios o no?"`
- **False question marks in Spanish**: Fixed sentences incorrectly ending with `?` without `¿`
  - **Root cause**: Aggressive word-based fallback overrode accurate semantic question detection
  - **Solution**: Prioritize semantic analysis, only use word-based heuristics when sentence-transformers unavailable
  - **Example**: `"...tal y como quieren que ellos los traten a ustedes?"` → `"...tal y como quieren que ellos los traten a ustedes."`
- **Speaker boundary priority in short texts**: Fixed speaker changes not being respected in texts < 25 words
  - **Solution**: Check speaker boundaries BEFORE min_total_words_no_split guard
  - **Impact**: Ensures speaker changes create sentence breaks even in short transcriptions
- **Metadata logging crash**: Fixed KeyError when logging removed periods that don't have 'connector' key
  - **Root cause**: Some removed period entries (e.g., 'skipped_whisper_boundary') don't include connector information
  - **Solution**: Made connector logging conditional in `punctuation_restorer.py` line 1449

### Changed
- `SentenceSplitter._convert_segments_to_word_boundaries()`: Now only extracts boundaries where speaker actually changes (not from all segments)
- `SentenceSplitter._should_end_sentence_here()`: Added debug logging for speaker boundary split/skip decisions
- `speaker_diarization.py`: Lowered `MIN_SPEAKER_SEGMENT_SEC` from 2.0s to 0.5s to capture brief speaker changes
- `podscripter.py._convert_speaker_segments_to_char_ranges()`: Completely rewritten to use overlap-based assignment instead of duration-based sorting
- `SentenceSplitter._evaluate_boundaries()`: Now removes Whisper periods inline when deciding not to split at connectors
- `_should_add_terminal_punctuation()`: Semantic question detection now takes priority over word-based fallback
- `punctuation_restorer.py`: Made metadata logging safer by checking for 'connector' key presence

### Added
- `tests/test_trailing_comma_bug.py`: Test suite to prevent regression of trailing comma bug
- Enhanced debug logging in `SentenceSplitter` to track period removal decisions

## [0.4.0] - 2025-12-30

### Added
- **Unified `SentenceSplitter` class**: All sentence splitting logic consolidated into single `sentence_splitter.py` module
  - Tracks punctuation provenance (Whisper vs our logic)
  - Coordinates speaker context with punctuation decisions
  - Supports multiple splitting modes (semantic, punctuation, hybrid, preserve)
  - Enables comprehensive debugging with split metadata
- **Whisper punctuation tracking**: System now tracks which periods came from Whisper segment ends
- **Intelligent period removal**: Automatically removes Whisper-added periods before same-speaker connectors
- **Split provenance metadata**: Each split now includes reason, confidence, speaker info, and punctuation tracking

### Fixed
- **Period-before-same-speaker-connector bug (Critical)**: Whisper-added periods are now removed when the same speaker continues with a connector word
  - **Impact**: Affects all diarization-enabled transcriptions in all supported languages (ES/EN/FR/DE)
  - **Example fix**: `"trabajo. Y este meta"` → `"trabajo y este meta"` (same speaker continues)
  - **Root cause**: Periods came from Whisper transcription, but speaker continuity decisions happened later in scattered pipeline
  - **Solution**: Unified `SentenceSplitter` tracks punctuation sources and removes periods intelligently based on speaker context

### Changed
- **Breaking**: `restore_punctuation()` signature changed - now accepts `whisper_segments` and `speaker_segments` instead of `whisper_boundaries` and `speaker_boundaries`
- **Breaking**: `restore_punctuation()` now ALWAYS returns tuple `(text, sentences_list)` - sentences_list is never None
- **Breaking**: Internal functions `_semantic_split_into_sentences()` and `_should_end_sentence_here()` moved to `SentenceSplitter` class
- **Simplified**: `_write_txt()` no longer performs sentence splitting - `SentenceSplitter` handles all boundaries
- **Simplified**: `_assemble_sentences()` passes full Whisper segments instead of just boundaries
- **Removed**: `skip_resplit` parameter from `_write_txt()` - no longer needed with unified splitting

### Removed
- Sentence splitting logic from Spanish post-processing (no more `_split_sentences_preserving_delims` call)
- Sentence splitting logic from `assemble_sentences_from_processed()` (keeps only ellipsis/domain logic)
- Final re-splitting from `_write_txt()` (no more location appositive protection needed)

### Technical Details
- **Files created**:
  - `sentence_splitter.py`: New module (~900 lines) with unified splitting logic
  - `tests/test_sentence_splitter_unit.py`: Comprehensive unit tests for `SentenceSplitter`
- **Files modified**:
  - `punctuation_restorer.py`: Updated to use `SentenceSplitter`, removed old splitting functions
  - `podscripter.py`: Updated `_assemble_sentences()` and simplified `_write_txt()`
- **Architecture**: Sentence splitting now happens in ONE place instead of 5+ scattered locations
- **Maintainability**: Future splitting features only require changes to `SentenceSplitter` class
- **Debugging**: Split provenance enables understanding exactly why each sentence was split

### Migration Notes
- Old `whisper_boundaries` and `speaker_boundaries` parameters are deprecated but still accepted for backward compatibility
- Tests should be updated to use new `whisper_segments` and `speaker_segments` parameters
- Any code extending the splitting logic should now extend `SentenceSplitter` class

## [0.3.1] - 2025-12-27

### Added
- **`--debug` flag**: New CLI flag to show detailed sentence splitting decisions
  - Shows speaker segment conversions (boundaries, char ranges, word ranges)
  - Logs every connector word evaluation with speaker continuity checks
  - Displays sentence ending decisions at Whisper boundaries
  - Part of mutually exclusive logging group: `--quiet` / `--verbose` (default) / `--debug`
- **Speaker segment tracking for connector word handling**: Full speaker segment information (with speaker labels and ranges) is now threaded through the entire punctuation pipeline
- New helper functions for speaker segment conversion:
  - `_convert_speaker_segments_to_char_ranges()`: Converts time-based speaker segments to character positions
  - `_convert_char_ranges_to_word_ranges()`: Converts character-based ranges to word-based ranges
  - `_get_speaker_at_word()`: Retrieves speaker label at a specific word index

### Fixed
- **Connector word sentence splitting bug (Critical)**: Sentences no longer incorrectly start with coordinating conjunctions ("Y", "O", "and", "et", "und") when the same speaker is speaking continuously
  - **Impact**: Affects all diarization-enabled transcriptions in all supported languages (ES/EN/FR/DE)
  - **Example fix**: `"...Colombia."` | `"Y yo soy Nate..."` → `"...Colombia. Y yo soy Nate..."` (same speaker continues)
  - **Root cause**: Sentence splitting logic checked for connector words but didn't verify if the same speaker was continuing vs. a new speaker starting
  - **Solution**: Enhanced `_should_end_sentence_here()` to check speaker continuity at connector words:
    - When same speaker continues with a connector: merge into same sentence
    - When different speakers: allow the break (new speaker starting with a connector is valid)
  - **Secondary fixes**: Eliminated three separate re-splitting steps that were undoing the speaker-aware logic:
    1. Spanish post-processing in `_transformer_based_restoration` now returns pre-split sentences
    2. `restore_punctuation()` returns tuple `(text, sentences_list)` to bypass re-splitting
    3. `_write_txt()` accepts `skip_resplit` flag to preserve speaker-aware boundaries
  - **Debug**: Added comprehensive logging to track sentence boundary decisions at connector words

### Changed
- **Debug messages moved to debug level**: Detailed sentence splitting logs (speaker conversions, connector checks, boundary decisions) now only appear with `--debug` flag instead of always showing
- `restore_punctuation()` now returns a tuple `(processed_text, sentences_list)` instead of just a string
  - `sentences_list` is `None` for non-diarization cases
  - When speaker segments are provided, returns pre-split sentences to preserve speaker-aware boundaries
- `_write_txt()` now accepts optional `skip_resplit` parameter to prevent re-splitting of carefully constructed sentences
- Spanish formatting in `_transformer_based_restoration` now bypasses re-splitting when speaker segments are used

### Technical Details
- **Files modified**:
  - `podscripter.py`: Added segment conversion functions, updated `_assemble_sentences()`, modified `_write_txt()`
  - `punctuation_restorer.py`: Enhanced speaker segment handling, modified return signatures, added word-level speaker tracking
- **API changes**: 
  - `restore_punctuation()` signature changed (backward compatible for non-diarization use)
  - Internal functions now accept `speaker_word_segments` parameter
- **Testing**: Verified with Episodio212.mp3 (33-minute Spanish podcast) - zero sentences starting with connector words

## [0.3.0] - 2025-12-07

### Added
- Speaker diarization integration with pyannote.audio
- Whisper segment boundary integration for improved sentence splitting
- Centralized punctuation system with context-aware processing

(Previous versions not documented - this is the first CHANGELOG)

---

## Version History Notes

- **0.3.x**: Focus on speaker diarization and sentence boundary accuracy
- **0.2.x**: Multilingual support and domain handling
- **0.1.x**: Initial release with basic transcription
