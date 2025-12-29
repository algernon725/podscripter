# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
