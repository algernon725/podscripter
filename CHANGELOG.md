# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
