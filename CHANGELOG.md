# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0.4] - 2026-01-11

### Fixed
- **Sentence-start words incorrectly lowercased (Bug #4)**: Fixed words like "En" and "Sin" being lowercased after sentence-ending punctuation
  - **Example**: `"método número dos. En vez de"` → incorrectly became `"método número dos. en vez de"`
  - **Root Cause**: The `_fix_mid_sentence_capitals()` regex pattern matched words after ALL punctuation including `.!?`, but words after sentence-ending punctuation should remain capitalized
  - **Fix**: Split into two patterns:
    1. After mid-sentence punctuation (`,;:`) → always lowercase
    2. After space NOT preceded by `.!?` → lowercase (uses negative lookbehind `(?<![.!?])`)
  - Words after `.!?` now correctly stay capitalized as they start new sentences
  - **Testing**: All 35 unit tests pass

## [0.6.0.3] - 2026-01-11

### Fixed
- **Missing space after periods (Bug #3 - Root Cause Found)**: Fixed instances where words were concatenated without spaces after sentence-ending punctuation (e.g., "yo.el" → "yo. el", "dos.en" → "dos. en")
  - **Root Cause**: The `_fix_mid_sentence_capitals()` function in `_write_txt()` had a regex bug that consumed whitespace without preserving it
    - Old pattern: `r'([\s.,;:!?¿¡])\s*' + word + r'\b'` with replacement `r'\1' + word.lower()`
    - The `\s*` matched whitespace after punctuation but didn't capture it, so the replacement discarded it
    - Example: `"yo. El método"` → matched `. El` (group 1 = `.`, the space was consumed by `\s*`) → replaced with `.el`
  - **Fix**: Updated regex to capture whitespace in group 2 and preserve it in the replacement
    - New pattern: `r'([\s.,;:!?¿¡])(\s*)' + word + r'\b'` with replacement `r'\1\2' + word.lower()`
    - Now `"yo. El método"` → matched `. El` (group 1 = `.`, group 2 = ` `) → replaced with `. el`
  - **Testing**: All 35 unit tests pass
  - **Note**: The safety net regex added in v0.6.0.2 was working correctly, but `_fix_mid_sentence_capitals()` was undoing it by removing the space
  - Removed debug logging that was added during investigation

## [0.6.0.2] - 2026-01-10

### Fixed
- **Missing space after periods (Bug #3)**: Added safety net in `_write_txt()` to add spaces after sentence-ending punctuation
  - Added regex pattern `r'([.!?])([A-ZÁÉÍÓÚÑa-záéíóúñ¿¡])` → `r'\1 \2'` before writing sentences
  - Followed by `fix_spaced_domains()` to prevent incorrectly adding spaces to domains
  - **Note**: This was a workaround; the root cause was found and fixed in v0.6.0.3

## [0.6.0.1] - 2026-01-10

### Fixed
- **Split utterance capitalization**: Fixed issue where split utterances from different speakers were not all capitalized. Now ALL utterances are capitalized when splitting by speaker, not just the first one (e.g., "y yo soy Nate" → "Y yo soy Nate" when it's a separate paragraph)
  - Modified `_write_txt()` in `podscripter.py` to capitalize each utterance when splitting sentences by speaker changes
  - Each utterance becomes its own paragraph, so each should start with a capital letter
- **Rapid speaker change filtering (Bugs #1 and #2)**: Increased `MIN_SEGMENT_DURATION` from 0.5s to 1.3s to filter out diarization artifacts
  - **Bug #1 (Missing space)**: Fixed "Métodos.pero" → "métodos pero" - Short segments (0.56s, 0.93s, 1.10s) were causing rapid speaker flipping, preventing proper sentence merging
  - **Bug #2 (Aggressive splitting)**: Fixed "Errores en español. Entonces, yo siempre les digo," being split across speakers - 1.28s segment misattributed to wrong speaker in middle of utterance
  - **Root Cause**: 0.5s threshold was too low; many short misattributed segments between 0.5s-1.3s were creating false speaker boundaries
  - **Impact**: Reduces speaker changes (e.g., Episodio213-trim: 10→5 changes, 20→13 segments filtered) while preserving legitimate speaker transitions
  - **Trade-off**: Very brief interjections (<1.3s) may be merged with adjacent speaker, but eliminates sentence fragmentation from diarization noise
  - **Threshold choice**: 1.3s chosen to filter highest artifact (1.28s) with small safety margin while preserving potential legitimate brief segments (>1.3s)

### Changed
- **Logging cleanup**: Removed duplicate log messages to reduce console clutter
  - Removed duplicate "Detected X speakers with Y speaker changes" message from `podscripter.py` (kept the one in `speaker_diarization.py` which says "unique speakers")
  - Removed duplicate "Speaker word ranges contain X speaker changes" message from `punctuation_restorer.py` (kept the one in `podscripter.py`)

## [0.6.0] - 2025-01-07

### Added
- **Speaker-Aware Output Formatting**: Utterances from different speakers within the same sentence are now split into separate paragraphs
  - Introduced `Utterance` and `Sentence` dataclasses to track speaker information throughout the pipeline
  - Speaker metadata now flows through entire transcription pipeline without loss
  - Addresses cases where short utterances from different speakers were appearing on the same line
  - **Architecture**: Introduced `Utterance` and `Sentence` dataclasses to track speaker information throughout the pipeline
  - **Implementation**:
    - `Sentence` dataclass: Contains text, list of `Utterance` objects, and primary speaker
    - `Utterance` dataclass: Represents a single speaker's contribution within a sentence (text, speaker, word range)
    - `SentenceSplitter.split()`: Now returns `List[Sentence]` instead of `List[str]`
    - `SentenceSplitter._detect_speaker_changes_in_sentence()`: Creates non-overlapping utterances, ensuring each word belongs to only one speaker
    - `SentenceSplitter._should_end_sentence_at_index()`: Prevents splitting after connector words AND before connector words (sentences shouldn't end or start with connectors/prepositions); minimum 4 words for speaker boundaries
    - `SentenceFormatter`: Updated to preserve `Utterance` lists when merging sentences
    - `_write_txt()`: When a sentence contains utterances from multiple speakers, merges consecutive utterances, then splits only if all utterances are ≥3 words; ensures all sentences start with capital letter
    - `_convert_speaker_segments_to_char_ranges()`: Filters out very short diarization segments (<0.5s) that are likely artifacts
    - `_convert_char_ranges_to_word_ranges()`: Merges consecutive word ranges from the same speaker
  - **Backward Compatibility**: Non-diarization mode works identically (empty utterances list, no splits)
  - **Testing**: Added `test_speaker_aware_output.py` with 5 test cases covering speaker changes, same-speaker continuation, and backward compatibility

### Fixed
- **Overlapping utterances**: Fixed issue where words appeared in multiple utterances due to overlapping speaker segments
- **Duplicate words in output**: Fixed issue where overlapping utterances caused words to be written multiple times
- **Spurious speaker changes**: Filter out very short diarization segments (<0.5s) that create artificial speaker boundaries
- **Connector words at sentence ends**: Conjunctions ("y", "o", "pero") and prepositions ("de", "a", "en", etc.) now stay with following text instead of being orphaned
- **Excessive fragmentation**: Increased minimum chunk length for speaker boundaries from 2 to 4 words to reduce 1-2 word sentence fragments
- **Short utterance handling**: Added minimum utterance length check (≥3 words) when splitting sentences by speaker
- **Sentence capitalization**: Sentences that start with lowercase letters (e.g., after connector word preservation) are now capitalized; preserves existing capitalization from punctuation restorer
- **Connector word split prevention**: Prevents splitting before connector words (don't start sentences with "y", "o", "pero", etc.)
- **Mid-sentence capitalization**: Fixed issue where words in the middle of a sentence were incorrectly capitalized (e.g., "best. Y aquí" → "best. y aquí") by:
  - Lowercasing the first letter when merging sentences in `SentenceFormatter`
  - Only capitalizing the first utterance when splitting sentences by speaker in `_write_txt()`
  - Added `_fix_mid_sentence_capitals()` in `_write_txt()` to lowercase common Spanish connectors/articles after periods mid-sentence (fixes punctuation restorer over-capitalization)

### Changed
- **Logging**: Changed speaker boundary decision messages (SKIP/SPLIT) and formatter messages from INFO to DEBUG level; now only shown with `--debug` flag
- **`sentence_splitter.py`**:
  - Added `Utterance` and `Sentence` dataclasses (lines 54-84)
  - Updated `SentenceSplitter.split()` return type to `List[Sentence]` (line 179)
  - Added `_detect_speaker_changes_in_sentence()` method (lines 393-484)
  - Updated sentence assembly loop to create `Sentence` objects with utterances (lines 517-647)
- **`sentence_formatter.py`**:
  - Updated `format()` to work with `List[Sentence]` (line 88)
  - Updated all merge methods (`_merge_domains`, `_merge_decimals`, `_merge_spanish_appositives`, `_merge_emphatic_words`) to preserve utterances when merging
  - Updated `_get_speaker_for_sentence()` to extract speaker from `Sentence.utterances` (line 178)
  - Added `_lowercase_first_letter()` static method to fix mid-sentence capitalization when merging sentences
  - Applied lowercase fix in `_merge_domains()` and `_merge_decimals()` when appending text
- **`punctuation_restorer.py`**:
  - Updated `_transformer_based_restoration()` to handle `Sentence` objects from `SentenceSplitter`
  - Reconstructs `Sentence` objects with formatted text while preserving utterances and speaker info
- **`podscripter.py`**:
  - Updated `_write_txt()` to detect speaker changes and add extra paragraph breaks (lines 349-387)
  - Maintains backward compatibility with string-based sentences
  - Modified `_write_txt()` to only capitalize the first utterance when splitting sentences by speaker (prevents mid-sentence capitalization)
  - Added `_fix_mid_sentence_capitals()` helper function to lowercase common Spanish words (connectors, articles) that appear capitalized after periods mid-sentence

### Notes
- This feature addresses the "Future Refactoring Opportunities" item documented in v0.5.2.3
- Output formatting: Single blank line between all paragraphs; utterances from different speakers are split into separate paragraphs
- Speaker separation quality depends on diarization accuracy; very short segments (<0.5s) are filtered out as likely artifacts
- In cases where diarization produces many short, alternating speaker segments, the output may contain short utterances on separate lines
- Speaker information is preserved for potential future features (labels, coloring, etc.)

## [0.5.2.3] - 2025-01-05

### Fixed
- **Refined dominant speaker threshold to preserve middle utterances (Bug #3b)**
  - **Problem**: v0.5.2.2's dominant speaker threshold (>80%) was too aggressive, filtering out legitimate short utterances in the MIDDLE of segments
  - **Impact**: Went from 84 speaker boundaries preserved to only 68 (16 legitimate boundaries lost), including "En espanolistos.com slash best. Ok." where "Ok." was incorrectly merged
  - **Root Cause**: The 80% threshold couldn't distinguish between edge misattributions and legitimate middle utterances
  - **Fix**: Refined logic to only apply dominant speaker threshold when minor speaker is at the EDGE (first/last 10% of segment)
    - Edge misattributions (like "Y yo" at START): filtered
    - Legitimate middle utterances (like "Ok." in MIDDLE): preserved
  - **Implementation**: Check if minor speaker's time range is within first/last 10% of Whisper segment before applying 80% threshold
  - **Impact**: Preserves all legitimate speaker changes while filtering only edge misattributions

## [0.5.2.2] - 2025-01-05

### Fixed
- **Dominant speaker threshold for edge misattributions (Bug #3)**
  - **Problem**: Pyannote occasionally misattributes a few words at segment boundaries to the wrong speaker
  - **Impact**: "Yo soy Andrea de Santander, Colombia y yo." instead of "Y yo soy Nate..." - the "Y yo" was incorrectly merged with Andrea's sentence
  - **Root Cause**: Pyannote diarization error at segment edge (assigned first 0.55s/4.0s = 14% of segment to wrong speaker), and our code was faithfully splitting based on that error
  - **Fix**: Added dominant speaker threshold - if one speaker accounts for >80% of a Whisper segment's duration, assign the entire segment to them
  - **Rationale**: Small misattributions at edges (<20%) are more likely diarization errors than actual speaker changes
  - **Example**: Segment with 0.55s SPEAKER_02 + 2.84s SPEAKER_01 (83.8% dominant) → assigned entirely to SPEAKER_01
  - **Note**: This fix was too aggressive and refined in v0.5.2.3 to only apply at edges

## [0.5.2.1] - 2025-01-05

### Fixed
- **CRITICAL: Overlap duration threshold filtering valid speech (Bug #2)**
  - **Problem**: The 0.3s overlap threshold was incorrectly checking TOTAL segment duration instead of overlap duration in some cases
  - **Impact**: Short speaker segments (e.g., 0.46s "Bueno!") that had substantial overlap (0.41s) with Whisper segments were being filtered out
  - **Root Cause**: Overlap duration check at line 724 was filtering based on total speaker segment duration (`< 0.5s`), not the actual overlap with the Whisper segment
  - **Fix**: Changed filter to check `overlap_duration < 0.3s` instead of `spk_duration < 0.5s`
  - **Example**: "Está bien." (SPEAKER_01) / "Bueno!" (SPEAKER_00) now properly split across lines

## [0.5.2] - 2025-01-05

### Fixed
- **CRITICAL: Speaker boundaries lost during conversion (Bug #1)**
  - **Problem**: Diarization detected 84 speaker changes, but only 48 made it to sentence splitting (36 boundaries lost!)
  - **Root Cause**: `_convert_speaker_segments_to_char_ranges()` assigned each Whisper segment to ONE speaker based on "most overlap"
    - When a Whisper segment contained text from multiple speakers, it was assigned entirely to the majority speaker
    - Example: "Aquí. Listo, eso es todo..." (Andrea 0.5s + Nate 2.0s) → assigned entirely to Nate → boundary lost
  - **Fix**: Rewrote segment assignment logic to SPLIT Whisper segments when they contain multiple speakers
    - Now detects all overlapping speakers per Whisper segment
    - Splits segments proportionally based on time overlaps
    - Attempts to split at word boundaries for cleaner results
    - Merges consecutive ranges from the same speaker
  - **Impact**: All speaker boundaries are now preserved (84 → 84 instead of 84 → 48)

### Changed
- **`_convert_speaker_segments_to_char_ranges()` completely rewritten** (lines 703-813)
  - Old approach: Assign each Whisper segment to single speaker (loses boundaries)
  - New approach: Split Whisper segments when they contain multiple speakers (preserves boundaries)
  - Added detailed debug logging for multi-speaker segments
  - Added character position logging for tracking splits

### Investigation History
- **v0.5.1 Investigation** (not released):
  - Fixed text normalization alignment in `_convert_speaker_segments_to_char_ranges()`
  - Applied `_normalize_initials_and_acronyms()` to segment text before calculating positions
  - Moved `SentenceFormatter.format()` to run BEFORE `_sanitize_sentence_output()`
  - Implemented proper speaker lookup in `SentenceFormatter._get_speaker_for_sentence()`
  - **Result**: These fixes were necessary but insufficient - deeper problem remained
- **v0.5.2 Root Cause Analysis**:
  - Debug output revealed: 84 boundaries detected, only 48 preserved
  - Identified "most overlap" assignment as the culprit
  - Implemented segment splitting solution

## [0.5.1] - 2025-01-04

### Fixed
- **Text normalization alignment in speaker boundary detection**
  - Applied `_normalize_initials_and_acronyms()` to Whisper segment text in `_convert_speaker_segments_to_char_ranges()`
  - Applied whitespace normalization to match `all_text` processing
  - Ensures character positions align correctly with normalized text

### Changed
- **Speaker lookup in `SentenceFormatter`**:
  - Added `_build_sentence_word_ranges()` to map sentence indices to word positions
  - Implemented proper `_get_speaker_for_sentence()` using word range overlaps
  - Moved `SentenceFormatter.format()` to run BEFORE `_sanitize_sentence_output()` to prevent word count misalignment

### Notes
- These fixes were necessary but insufficient to fully resolve the speaker boundary bug
- See v0.5.2 for the complete fix

## [0.5.0] - 2025-01-04

### Added
- **Unified `SentenceFormatter` class**: Consolidated all post-processing merge operations (domains, decimals, Spanish appositives, emphatic words) into single `sentence_formatter.py` module
  - All merge logic now in ONE location for easier maintenance and debugging
  - Speaker-aware merge decisions: NEVER merges different speakers (prevents bugs like "jugar. Es que..." cross-speaker merge)
  - Merge provenance tracking for debugging
  - Comprehensive unit tests (`tests/test_sentence_formatter.py`)
- **`--dump-merge-metadata` CLI flag**: Writes merge provenance to `<basename>_merges.txt` for debugging
  - Shows which sentences were merged and why
  - Shows which merges were skipped due to speaker boundaries
  - Includes detailed before/after text and speaker information

### Changed
- **Breaking (Internal)**: Post-processing merge operations moved from `podscripter.py` to `sentence_formatter.py`
  - Public API unchanged (input/output identical)
  - `_assemble_sentences()` now returns tuple `(sentences, merge_metadata)`
- **Improved**: Natural language guards prevent false domain merges (already in v0.4.4, now consolidated)
- **Improved**: All merge types (domain, decimal, appositive, emphatic) now respect speaker boundaries

### Fixed
- **Speaker boundary enforcement for merges**: Different speakers' sentences are never merged, even when patterns match
  - Example: "jugar." (Speaker A) + "Es que vamos..." (Speaker B) no longer incorrectly merges as "jugar.es"
  - Applies to all merge types: domains, decimals, appositives, emphatic words

### Benefits
- **Maintainability**: All post-processing in ONE place (`sentence_formatter.py`)
- **Correctness**: Speaker boundaries enforced for ALL merges
- **Debuggability**: Merge provenance tracks WHY each merge happened
- **Testing**: Isolated unit tests for each merge type
- **Architecture**: Clear separation - `SentenceSplitter` handles splitting, `SentenceFormatter` handles formatting

### Backward Compatibility
- **100% backward compatible for non-diarization mode**: When `--enable-diarization` is not used, all merges work identically to v0.4.4
- Speaker boundary checks are an opt-in safety feature that only activates when speaker data is available
- All existing tests pass with same output (or better due to speaker boundary fixes)

## [0.4.4] - 2025-01-03

### Fixed
- **False domain merge in natural language (Critical)**: Fixed domain merge logic incorrectly merging sentences when word before period matched a TLD
  - **Issue**: When a sentence ended with a word that matched a TLD in the domain list (e.g., "jugar." followed by "Es que vamos..."), the two sentences were incorrectly merged as if they were a broken domain name (e.g., "jugar.es")
  - **Example**: "Eso lo seguiremos haciendo, pero jugar tenis juntos, bueno, si es que yo aprendo y soy capaz de jugar. Es que vamos a tratar de tener lecciones de tenis en Colombia." (Andrea's sentence + Nate's sentence merged despite being different speakers)
  - **Root cause**: `podscripter.py` lines 949-988 domain merge logic used regex `([A-Za-z0-9\-]+)\.$` to match any word ending with period, then checked if next sentence starts with a TLD pattern. The TLD list includes "es" (Spanish TLD), so "jugar." + "Es que..." matched as "jugar.es" domain
  - **Solution**: Added natural language guards (lines 962-972) to prevent false domain merges:
    - Only merge if current sentence is short (< 50 characters) OR the label before the period is capitalized
    - Long natural language sentences like "pero jugar tenis juntos..." are now excluded from domain merging
    - Capitalized labels like "Google." + "Com ..." still merge correctly as "Google.Com"
  - **Impact**: Affects all transcriptions where a sentence ends with a common word that happens to match a TLD in the list (e.g., "es", "de", "co", "io"). Particularly important for diarization-enabled transcriptions where different speakers' sentences were being incorrectly merged
  - **Rationale**: Domain names in transcriptions are typically short standalone mentions (e.g., "Visit example.com") or capitalized (e.g., "Check out Google.Com"). Long natural language sentences ending with lowercase words are almost never domain names
  - **Tests**: Verified with Episodio212.mp3 - Andrea's "jugar." sentence and Nate's "Es que vamos..." sentence now correctly separated

## [0.4.3] - 2025-01-02

### Fixed
- **Missing periods after Whisper segments (Critical)**: Fixed periods being incorrectly removed when speaker changes occurred many words later
  - **Issue**: Legitimate sentence-ending periods were removed if a speaker change occurred within the next 15 words
  - **Example**: "Dile adiós a todos esos momentos incómodos Entonces, empecemos." (missing period after "incómodos")
  - **Root cause**: `sentence_splitter.py` line 558 used a 15-word lookahead window to skip Whisper boundaries when speaker changes were nearby. This was too aggressive and removed periods from legitimate sentence endings
  - **Solution**: 
    - Reduced lookahead window from 15 words to 3 words (only skip for true misalignment)
    - Added checks: only skip if next word is a connector OR starts lowercase (indicates continuation)
    - If next word is capitalized and not a connector, preserve the Whisper boundary and period
  - **Impact**: Affects all diarization-enabled transcriptions where Whisper segment boundaries don't align perfectly with speaker boundaries
  - **Tests**: All 35 tests pass

- **Speaker changes with connector words not separated (Critical)**: Fixed different speakers' sentences being merged when next sentence starts with connector
  - **Issue**: When a speaker change occurred and the next speaker's sentence started with a connector word ("Y", "and", "et", "und"), the two sentences were merged into one paragraph
  - **Example**: "Yo soy Andrea de Santander, Colombia. Y yo soy Nate de Texas, Estados Unidos." (Andrea and Nate on same line despite being different speakers)
  - **Root cause**: `sentence_splitter.py` line 519-531 skipped speaker boundaries when the next word was a connector, assuming same-speaker continuation
  - **Solution**: Speaker boundaries now ALWAYS create splits, regardless of whether next word is a connector. Connector merging only applies when the SAME speaker continues
  - **Impact**: All diarization-enabled transcriptions now correctly separate different speakers even when one starts with a connector word
  - **Tests**: All 35 tests pass

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
