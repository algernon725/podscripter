# PodScripter Architecture

PodScripter transcribes audio/video into punctuated, readable text and SRT subtitles. It uses Faster-Whisper for ASR and a hybrid semantic + rule approach for punctuation and formatting. The system is designed to run entirely in Docker with model caches mounted for reproducibility and offline operation.

## System context

- **Inputs**: Audio/video files (e.g., MP3, MP4, WAV)
- **Outputs**: Plain text (`.txt`) or subtitles (`.srt`)
- **Primary dependencies**:
  - Faster-Whisper (ASR)
  - Sentence-Transformers (semantic cues for punctuation)
  - spaCy (mandatory; automatic capitalization; models baked into Docker)
  - pydub (chunking)
- **Runtime**: Docker container; model caches bound via volumes

## Goals and non-goals

- **Goals**
  - Accurate, readable transcripts across EN/ES/FR/DE
  - Deterministic runs via containerization and caching
  - CPU-friendly defaults with support for long audio
  - Generalizable punctuation/formatting improvements over one-off fixes
  - Comprehensive domain protection (single and compound TLDs) to preserve URLs/websites in transcriptions
  - Spanish false domain prevention through centralized exclusion lists
  - Spanish-only `.de` and `.es` exclusion: `.de` and `.es` TLDs are ignored in Spanish to avoid false positives with common words "de" (preposition) and "es" (verb "is") (e.g., `tratada.de` → `tratada. de`, `naturales.es` → `naturales. es`)
  - Popular-only TLD list: only commonly-used TLDs are included (`com|net|org|co|es|io|edu|gov|uk|us|ar|mx|de|fr|br|ca|au`); obscure TLDs removed to prevent false positives (e.g., `.it` conflicting with English "it" in "Escucha.it", `.in` conflicting with English "in")
- **Non-goals**
  - End-to-end cloud service or hosted UI
  - Perfect linguistic analysis for all languages
  - Arbitrary output formats beyond TXT/SRT

## High-level architecture

```mermaid
flowchart TD
  subgraph infra["Runtime & Infra"]
    Dock["Docker Container"]
    ModelCache["Local Model Cache"]
    Storage["Output Dir / Temp Storage"]
  end

  subgraph ingestion["Ingestion"]
    Upload["User Upload: mp3/mp4/..."]
  end

  Upload --> Pre["Preprocessing"]

  Pre --> Chunking["Chunking"]
  Chunking --> Chunks["Chunks with overlap"]

  subgraph asr["ASR Layer"]
    FW["WhisperModel (faster-whisper)"]
    LangDetect["Language Detection (from Whisper info)"]
    ASRraw["Raw ASR Segments"]
  end
  Chunks --> FW
  FW --> ASRraw
  FW --> LangDetect

  subgraph post["Postprocessing"]
    Dedup["Dedupe + Global Timestamps"]
    Punct["Punctuation + Sentence Assembly (helpers in punctuation_restorer.py)"]
    Format["Formatting: txt / srt"]
  end

  ASRraw --> Dedup --> Punct --> Format --> Storage

  subgraph agent["Agent / Orchestration"]
    CLI["CLI Entrypoint"]
    Orch["Orchestrator / Agent"]
    Config["Config & Flags"]
    Tests["Unit + Integration Tests"]
  end

  Dock --> FW
  Dock --> Orch
  Dock --> Tests
  ModelCache --> FW
  Config --> Orch
  CLI --> Orch
  Orch --> Chunking
  Orch --> FW
  Orch --> post
  Tests --> Orch

  Storage --> User["User Download"]
```

## Data flow

1. Pre-validate input/output paths and clean leftover chunk files.
2. If not `--single`, split media into ~480s chunks with ~3s overlap.
3. Transcribe (Faster-Whisper) with optional VAD and `initial_prompt` continuity; obtain language (if auto).
4. Convert per-chunk timestamps to global, dedupe overlap, accumulate raw text.
5. Restore punctuation using segment-aware processing (`restore_punctuation_segment()`) and assemble sentences using helper utilities (ellipsis/domain-aware; automatic spaCy capitalization).
6. Write TXT or SRT.

## Components and responsibilities

- **CLI and Orchestrator** (`podscripter.py`)
  - Parse args; validate paths
  - Choose single-call vs chunked mode
  - Manage model load, model selection, translation task, VAD settings, continuity prompts
  - Write outputs (TXT/SRT)
  - Optional raw debug dump when `--dump-raw` is set: writes `<basename>_raw.txt` containing detected language (if auto), task, segment count, and per-segment timings/text
  - Optional merge metadata dump when `--dump-merge-metadata` is set: writes `<basename>_merges.txt` with merge provenance (v0.5.0+)

- **Chunking**
  - `_split_audio_with_overlap(media_file, chunk_length_sec=480, overlap_sec=3)` using pydub
  - Generates chunk files and metadata for alignment

- **ASR (Faster-Whisper)**
  - `_transcribe_file(...)` → `WhisperModel.transcribe`
  - Returns segments and info (e.g., detected language)
  - Sets `task=translate` when `--translate` is provided; info logs include the selected task in CLI

- **Alignment and deduplication**
  - `_accumulate_segments(...)` and `_dedupe_segments(...)`
  - Globalize timestamps and remove overlap duplicates

- **Speaker Diarization** (`speaker_diarization.py`) - Optional
  - `diarize_audio(...)` performs speaker detection using pyannote.audio
  - `_extract_speaker_boundaries(...)` extracts speaker change timestamps; returns both filtered boundaries and detailed `BoundaryInfo` for debugging
  - `_merge_boundaries(...)` merges speaker and Whisper boundaries with deduplication
  - `write_diarization_dump(...)` writes comprehensive debug file with raw segments, boundary analysis, and merge details
  - Boundaries represent high-confidence sentence break hints
  - Debug dump (`--dump-diarization`) aids troubleshooting when speaker changes don't trigger sentence breaks

- **Sentence Splitting** (`sentence_splitter.py`) - NEW in v0.4.0
  - **Unified `SentenceSplitter` class**: Consolidates all sentence boundary logic
  - Tracks punctuation provenance (Whisper vs our logic)
  - Coordinates speaker context with punctuation decisions
  - Supports multiple splitting modes (semantic, punctuation, hybrid, preserve)
  - **Intelligent Whisper period removal**: Removes periods before same-speaker connectors
  - Priority hierarchy: Grammatical guards → Speaker continuity → Speaker boundaries → Whisper boundaries → Semantic splitting
  - Enables comprehensive debugging with split metadata

- **Sentence Formatting** (`sentence_formatter.py`) - NEW in v0.5.0
  - **Unified `SentenceFormatter` class**: Consolidates all post-processing merge operations
  - Handles domain, decimal, appositive, and emphatic word merges in one location
  - **Speaker-aware merge decisions**: NEVER merges different speakers (prevents cross-speaker bugs)
  - Merge provenance tracking with `MergeMetadata` dataclass
  - Natural language guards prevent false domain merges (e.g., "jugar. Es que..." stays separate)
  - 100% backward compatible when speaker data not available (non-diarization mode)
  - Debug support via `--dump-merge-metadata` flag

- **Punctuation and formatting** (`punctuation_restorer.py`)
  - `restore_punctuation(...)` → Uses `SentenceSplitter` for boundary decisions (v0.4.0+)
  - `restore_punctuation_segment(...)` → segment-aware processing for individual Whisper segments
  - **Centralized punctuation system**: `_should_add_terminal_punctuation()` with context-aware processing
  - **Context types**: `STANDALONE_SEGMENT`, `SENTENCE_END`, `FRAGMENT`, `TRAILING`, `SPANISH_SPECIFIC`
  - Sentence-Transformers semantic cues + curated regex rules
  - Language-specific formatting (ES/EN/FR/DE)
  - Comma spacing normalization is centralized in `_normalize_comma_spacing(text)` and must not be duplicated inline at call sites. This helper removes spaces before commas, deduplicates multiple commas, and ensures a single space after commas. Trade-off: thousands separators like `1,000` will appear as `1, 000` to guarantee correct spacing for common number lists.
  - Comprehensive domain protection: preserves single TLDs (`github.io`, `harvard.edu`) and compound TLDs (`bbc.co.uk`, `amazon.com.br`) across all processing stages
  - Domain assembly logic: uses a general multi-pass merge loop to recover split domains across sentence boundaries and intermediate transformations
  - Spanish false domain prevention: centralized exclusion logic prevents Spanish words (e.g., `uno.de`, `este.es`, `naturales.es`) from being incorrectly treated as domains
  - Natural language domain merge guards (v0.4.4): prevents false domain merges when sentences end with words matching TLDs (e.g., "jugar." + "Es que..." should NOT merge as "jugar.es"). Only merges if sentence is short (< 50 chars) OR label is capitalized, ensuring domain detection targets actual URL mentions rather than natural language coincidences
  - Spanish processing wraps all transformations with domain masking/unmasking so that URLs (including subdomains like `www.example.com`) remain intact through punctuation and capitalization stages
  - Pipeline order correction (Spanish): automatic spaCy capitalization now runs before greeting/comma insertion to avoid capitalization feedback loops and over-capitalization of common words
  - Location appositive normalization (EN/ES/FR/DE): punctuation-restoration converts ", <preposition> <Location>. <Location>" to ", <preposition> <Location>, <Location>" using language-specific prepositions (ES: de; EN: from/in; FR: de/du/des; DE: aus/von/in). Also normalizes direct comma-separated forms like "City, Region. and/pero/y …" to keep the location intact and continue the clause. Includes a new-sentence guard to avoid merging when the following fragment starts a new sentence with a subject (e.g., "Y yo …", "And I …", "Et je …", "Und ich …").
  - TXT writer multilingual location protection: during final TXT splitting, protects appositive location patterns like ", <preposition> <Location>. <Location>" to avoid breaking location descriptions. Applies across EN/ES/FR/DE using language-specific prepositions (ES: de; EN: from/in; FR: de/du/des; DE: aus/von/in); restores protected periods after splitting.
  - Spanish greeting and inverted-question guards:
    - Guard comma insertion after greetings: avoid comma after `Hola` when followed by `a`/`para` (e.g., "Hola para todos")
    - When a greeting precedes an inverted mark (`¿`/`¡`), add a comma only if absent to prevent duplicate commas
    - If a sentence begins with a greeting and contains an embedded Spanish question, do not add a leading `¿` at the start; keep only the embedded `¿ … ?`
  - Spanish appositive merge guard: only merge `..., de <Proper>. <Proper> ...` when the second fragment is just a location continuation (minimal trailing content). Prevents accidental merging when the next sentence contains additional clauses.
  - Sentence assembly public helper:
    - `assemble_sentences_from_processed(processed, language)` which performs ellipsis continuation, domain-aware splitting, and French short-connector merging
  - Cross-segment carry of trailing fragments for French and Spanish
  - Automatic spaCy capitalization (always enabled) with mixed-language support:
    - English phrase detection in Spanish transcriptions using spaCy language detection or linguistic heuristics
    - Multi-layered location handling: spaCy NER + cross-linguistic analysis + conservative contextual patterns
    - Prevents English phrase over‑capitalization (e.g., "I am Going To Test" → "I am going to test")
    - Preserves location capitalization when already present in the input (e.g., "de Santander, Colombia")
    - Conservative location capitalization to avoid false positives:
      - Capitalize after strong cues (e.g., `vivo en`, `trabajo en`, `soy de`, `vengo de`)
      - After `de`, capitalize only when the next token was already capitalized in the input
      - After `en`, capitalize when the next token was already capitalized; avoid common nouns
  - SRT normalization in CLI: reading-speed-based cue timing; INFO log summarizes trimmed cues

## Configuration

- **Environment variables**
  - `HF_HOME`: Hugging Face cache root
  - `WHISPER_MODEL`: default Whisper model name (overridden by CLI `--model`)
  - Optionally set `HF_HUB_OFFLINE=1` when caches are warm
- **Runtime flags** (CLI)
  - `--output_dir <dir>` (required)
  - `--language <code>|auto` (default auto)
  - `--output_format {txt|srt}` (default txt)
  - `--single` (disable manual chunking)
  - `--model {tiny,base,small,medium,large,large-v2,large-v3}` (default `medium`; precedence: CLI > `WHISPER_MODEL` env > default)
  - `--translate` (Whisper `task=translate`; punctuation uses English rules)
  - `--compute-type {auto,int8,int8_float16,int8_float32,float16,float32}` (default auto)
  - `--beam-size <int>` (beam size for decoding; default 3)
  - `--no-vad` (disable VAD filtering; default is enabled)
  - `--vad-speech-pad-ms <int>` (padding in ms when VAD is enabled; default 200)
  - `--dump-raw` (also write raw Whisper output for debugging to `<basename>_raw.txt` in `--output_dir`)
  - `--enable-diarization` (enable speaker diarization for improved sentence boundaries; default disabled)
  - `--min-speakers <int>` (minimum number of speakers for diarization; default auto-detect)
  - `--max-speakers <int>` (maximum number of speakers for diarization; default auto-detect)
  - `--hf-token <str>` (Hugging Face token for pyannote model access; required for first download)
  - `--dump-diarization` (write diarization debug dump to `<basename>_diarization.txt`; requires `--enable-diarization`)
  - `--quiet`/`--verbose`/`--debug` (mutually exclusive; default verbose)
    - `--debug` shows detailed sentence splitting decisions and speaker segment tracking

## Operations

- **Modes**
  - Single-call: transcribe entire file; best context, higher resource usage
  - Chunked: default for long files; overlap + prompt tail for continuity
- **Caching**
  - Mount volumes to persist models between runs:
    - Faster-Whisper via Hugging Face Hub: `models/huggingface/` → `/root/.cache/huggingface`
    - Sentence-Transformers: `models/sentence-transformers/` → `/root/.cache/torch/sentence_transformers`
    - Media: `audio-files/` → `/app/audio-files`
  - Build efficiency:
    - Combine pip installs into a single `RUN` with `--no-cache-dir`
    - Keep `COPY . .` last to maximize layer caching
    - Use `.dockerignore` to exclude large local media and caches (`audio-files/`, `models/`)
- **Error handling**
  - Early exits for invalid input or unwritable output
  - Conservative fallbacks when ST/spaCy unavailable
  - Typed exceptions surfaced from the orchestrator and mapped to exit codes:
    - `InvalidInputError` (2), `ModelLoadError` (3), `TranscriptionError` (4), `OutputWriteError` (5), unexpected (1)
  - Logging via a single `podscripter` logger; three levels:
    - `--quiet` → ERROR (minimal)
    - `--verbose` → INFO (default; lifecycle logs)
    - `--debug` → DEBUG (detailed sentence splitting, speaker tracking, connector evaluations)

## Performance characteristics

- CPU-friendly defaults (`compute_type=auto`, modest beam size)
- Overlap + dedupe minimizes boundary artifacts
- Prompt tail (~200 chars) improves chunk continuity

## Testing and quality gates

- All tests run in Docker with caches mounted
- Language-specific and cross-language tests for punctuation, splitting, and domain protection (including compound TLD support)
- Ad-hoc `tests/test_transcription.py` for manual experiments
- Focused unit tests included by default:
  - `tests/test_sentence_assembly_unit.py` protects Spanish ellipsis/domain handling and French connector merges
  - `tests/test_chunk_merge_helpers.py` validates dedupe and accumulation correctness for segments
  - `tests/test_srt_normalization.py` validates SRT cue trimming behavior
  - `tests/test_initials_normalization.py` (WIP) documents expected behavior for person initial normalization

## Extensibility

- Add languages via `LanguageConfig` and per-language helpers
- Tune thresholds centrally without rewriting logic
- Additional output formats can be added in the writer layer

## Known limitations

- Non EN/ES/FR/DE languages are experimental
- spaCy capitalization requires language models; disabled if unavailable
- Perfect punctuation restoration is not guaranteed; favors robust heuristics
- Thousands separators include a space after commas (e.g., `1, 000`) due to centralized comma spacing. This trade-off was chosen to reliably fix number-list spacing in transcripts.
- **WIP**: Person initials (e.g., "C.S. Lewis", "J.K. Rowling") in non-English transcriptions may still split into separate sentences. Infrastructure is in place (`_normalize_initials_and_acronyms()`) but requires additional masking to protect initials through spaCy processing. See AGENT.md "Known Open Issues" for details.

## Recent architectural improvements

### Unified Sentence Splitting (2025 - v0.4.0)
- **Problem solved**: Sentence splitting logic was scattered across 5+ locations in the codebase, making debugging and maintenance extremely difficult. The period-before-same-speaker-connector bug couldn't be solved without architectural change.
- **Solution**: Created unified `SentenceSplitter` class in `sentence_splitter.py` that consolidates ALL splitting logic
- **Benefits**:
  - Single source of truth for all sentence boundary decisions
  - Tracks punctuation provenance (Whisper vs our logic)
  - Coordinates speaker context with punctuation decisions
  - Intelligently removes Whisper periods before same-speaker connectors
  - Enables comprehensive debugging with split metadata
  - Future splitting features only require changes to one class
- **Implementation**: `SentenceSplitter` class with priority hierarchy: Grammatical guards → Speaker continuity → Speaker boundaries → Whisper boundaries → Semantic splitting
- **Impact**: Solved the period-before-same-speaker-connector bug across all languages (ES/EN/FR/DE)

### Unified Sentence Formatting (2025 - v0.5.0)
- **Problem solved**: Post-processing merge operations were scattered across `podscripter.py` with no coordination with speaker boundaries, causing cross-speaker merge bugs (e.g., "jugar. Es que..." incorrectly merging as "jugar.es" across different speakers).
- **Solution**: Created unified `SentenceFormatter` class in `sentence_formatter.py` that consolidates ALL post-processing merges
- **Benefits**:
  - Single source of truth for all merge operations (domain, decimal, appositive, emphatic)
  - Speaker-aware merge decisions: NEVER merges different speakers
  - Merge provenance tracking for debugging
  - Natural language guards prevent false domain merges
  - 100% backward compatible for non-diarization mode
  - Future merge patterns only require extending `SentenceFormatter`
- **Implementation**: `SentenceFormatter` class with speaker boundary checks before every merge operation
- **Impact**: Solved cross-speaker merge bugs across all merge types; improved debuggability with `--dump-merge-metadata` flag

### Centralized Punctuation System (2025 - v0.3.0)
- **Problem solved**: Previously, period insertion logic was scattered across 14+ locations in the codebase, making bugs like "Ve a" → "Ve a." difficult to fix and maintain
- **Solution**: Centralized all punctuation decisions into `_should_add_terminal_punctuation()` with context-aware processing
- **Benefits**: 
  - Single source of truth for punctuation logic
  - Context-aware processing prevents incomplete phrase punctuation
  - Easier debugging and maintenance
  - Prevents similar bugs in the future
- **Implementation**: Uses `PunctuationContext` enum to apply different rules for different scenarios (Whisper segments vs complete sentences)

### Whisper segment boundary integration

- The orchestrator now passes `all_segments` into sentence assembly. `_assemble_sentences(all_text, all_segments, ...)` extracts Whisper segment end positions and forwards them to `restore_punctuation(text, language, whisper_boundaries=...)`.
- The punctuation pipeline converts character boundaries to word indices and uses them as prioritized hints inside `_semantic_split_into_sentences(..., whisper_word_boundaries=...)` and `_should_end_sentence_here(...)`.
- Hints are gated by grammatical rules and minimum chunk size; they are ignored for invalid break positions (e.g., conjunctions/prepositions/continuative verbs).
- Thresholds (language-agnostic; in `_get_language_thresholds(language)`):
  - `min_words_whisper_break` (default 10)
  - `max_words_force_split` (default 100)
- Backward compatible: behavior is unchanged when boundaries are not provided.

Tests: `tests/test_whisper_boundary_integration.py` covers extraction, gating, and multi-language behavior.

### Speaker Diarization Integration (2025)

Podscripter optionally uses speaker diarization to detect when speakers change, providing high-priority hints for sentence boundaries.

**Library**: pyannote.audio 3.3.2

**Integration**: Full speaker segment information (with labels and ranges) is threaded through the entire punctuation pipeline to enable speaker-aware sentence splitting.

**Speaker Segment Tracking** (v0.3.1, enhanced in v0.4.2, v0.5.2, v0.6.0):
- **Full speaker context**: Instead of just boundary timestamps, entire speaker segments (with labels and ranges) are tracked
- **Conversion pipeline**:
  1. `_convert_speaker_segments_to_char_ranges()`: Time-based segments → character positions in text
     - **v0.4.2**: Used overlap-based assignment algorithm (assigned each Whisper segment to ONE speaker with most overlap)
     - **v0.5.2**: SPLITS Whisper segments when they contain multiple speakers instead of assigning to single speaker
       - Detects all overlapping speakers per Whisper segment
       - Splits segments proportionally based on time overlaps
       - Attempts to split at word boundaries for cleaner results
       - Merges consecutive ranges from the same speaker
       - **v0.5.2.1**: Filters on overlap duration (not total duration) to preserve short utterances
       - **v0.5.2.2**: Applies dominant speaker threshold (>80%) to filter edge misattributions
       - **v0.5.2.3**: Refined threshold to only apply at segment edges (first/last 10%), preserving middle utterances
     - **Preserves boundaries**: All speaker boundaries preserved (84 → 84 instead of 84 → 48 in v0.4.2)
  2. `_convert_char_ranges_to_word_ranges()`: Character positions → word indices
  3. `_get_speaker_at_word()`: Query speaker label at any word position
- **Connector word handling**: When a sentence would start with a connector word ("Y", "O", "and", "et", "und"), check if the same speaker is continuing. If yes, merge into previous sentence. If no (different speaker), allow the break.
- **Example**: `"...Colombia. Y yo soy..."` stays together if same speaker, splits if different speaker starting with "Y"
- **v0.6.0 Speaker-Aware Output**:
  - **Utterance and Sentence dataclasses**: Speaker information flows through entire pipeline
    - `Utterance`: Represents a single speaker's contribution (text, speaker, word range)
    - `Sentence`: Contains text, list of utterances, and primary speaker
  - **Speaker tracking in pipeline**:
    - `SentenceSplitter._detect_speaker_changes_in_sentence()`: Splits sentences into utterances based on speaker segments
    - `SentenceFormatter`: Preserves utterances when merging sentences
    - `_write_txt()`: Adds extra paragraph break (`\n\n\n`) when speaker changes between sentences
  - **Impact**: Visual separation of different speakers in output while preserving same-speaker grouping

**Priority hierarchy** (in `_should_end_sentence_here`):
1. **Speaker boundaries** (HIGHEST priority - min 1 word, v0.6.1)
   - **v0.4.2**: `SentenceSplitter` only extracts boundaries where speaker changes (not from all segments)
   - **v0.4.3**: Speaker boundaries ALWAYS create splits, even if next word is a connector (different speakers must be separated)
   - **v0.6.1**: Removed connector word checks that were incorrectly blocking speaker splits; reduced threshold from 4 to 1 word
2. Grammatical guards (never break on conjunctions/prepositions/auxiliary verbs)
   - **v0.6.3**: Expanded `CONTINUATIVE_AUXILIARY_VERBS` to include infinitives (`ser`, `estar`, `be`, `have`, `être`, `avoir`, `sein`, `haben`, etc.) and preterite forms (`fue`, `fueron`, `was`, `were`, `fut`, `furent`, `wurde`, etc.)
   - **v0.6.3**: Added guard for numbers followed by time/measurement units (prevents "a los 18. Años" → keeps "a los 18 años")
   - **v0.6.3**: Added `_is_past_participle()` helper and guard to prevent splitting auxiliary verbs from past participles (prevents "fueron. Dirigidos" → keeps "fueron dirigidos")
3. Whisper boundaries (medium priority - min 10 words)
   - **Whisper boundary skipping**: If a speaker boundary is within the next 3 words AND next word is connector/lowercase, skip the Whisper boundary to avoid splitting when the same speaker continues across Whisper segments
   - **v0.4.2**: Periods at skipped boundaries are now automatically removed
   - **v0.4.3**: Reduced lookahead window from 15 words to 3 words, added continuation checks to prevent over-aggressive skipping
   - **v0.6.2**: Spanish inverted question/exclamation protection - before allowing splits at Whisper boundaries, check if inside unclosed `¿...?` or `¡...!` via `_is_inside_unclosed_question()` helper
4. General min_chunk_before_split check (20 words for Spanish, 15 for others)
5. Semantic coherence (fallback)

**Critical implementation details**: 
- Speaker/Whisper boundary checks happen BEFORE the general `min_chunk_before_split` threshold check to ensure short phrases like "Mateo 712" can break at speaker changes
- Whisper boundaries are skipped only when a speaker boundary is very close (3 words) AND next word suggests continuation (connector or lowercase)
- **v0.4.2**: Whisper-added periods at skipped boundaries are now tracked and removed automatically
- **v0.4.3**: Speaker boundaries are never skipped, ensuring different speakers are always separated even with connector words
- **v0.6.1**: Period removal before connectors only happens when SAME speaker continues (verified via `_get_speaker_at_word()`); `end_word` in speaker segments is treated as EXCLUSIVE (like Python slices)
- **v0.6.2**: Spanish unclosed inverted question/exclamation protection prevents splits inside `¿...?` and `¡...!` constructs at Whisper boundaries

**Opt-in**: Disabled by default, enabled via `--enable-diarization` flag.

Benefits:
- More accurate sentence breaks at dialogue transitions
- Especially useful for interviews, conversations, and multi-speaker content
- Complements existing grammatical guards and semantic analysis
- Natural speaker changes almost always align with sentence boundaries

Caching:
- Models cached under `HF_HOME` (`/root/.cache/huggingface`, mounted from `models/huggingface`)
- Requires Hugging Face token for first-time model download (`--hf-token` or `HF_TOKEN` env var)
- Subsequent runs use cached models for fast startup

Threading through pipeline:
- Diarization runs on media file before transcription (can be parallelized in future)
- Speaker boundaries extracted via `_extract_speaker_boundaries(...)` as timestamps (seconds)
- Timestamps converted to character positions via `_convert_speaker_timestamps_to_char_positions(...)` in `_assemble_sentences(...)`
- Passed SEPARATELY to `restore_punctuation(..., whisper_boundaries=..., speaker_boundaries=...)`
- Converted to word indices in `_transformer_based_restoration()`
- Full speaker segments converted to word ranges and passed as `speaker_word_segments` parameter
- Checked with low threshold (2 words) in `_should_end_sentence_here(..., speaker_word_boundaries=..., speaker_word_segments=...)`
- Used for connector word continuity checks: "Is the same speaker continuing with 'Y'?"

Configuration:
- `DEFAULT_DIARIZATION_DEVICE = "cpu"` (can use GPU if available)
- `SPEAKER_BOUNDARY_EPSILON_SEC = 1.0` (merge boundaries within 1s)
- `MIN_SPEAKER_SEGMENT_SEC = 0.5` (v0.4.2: lowered from 2.0s to capture brief speaker changes while filtering noise)

Debugging:
- Use `--dump-diarization` to write `<basename>_diarization.txt` with:
  - Summary (speaker count, segment count, filtering parameters)
  - Raw speaker segments from pyannote (timestamps and speaker labels)
  - Speaker boundary analysis (each change with included/filtered status and reason)
  - Final speaker boundaries used for sentence splitting
  - Whisper segment boundaries (for comparison)
  - Merged boundaries with source labels (speaker/whisper)
- Helps troubleshoot why specific speaker changes did not cause sentence breaks

**Sentence Re-splitting Protection** (v0.3.1):
- **Problem**: Sentence splitting was happening in three separate places, causing speaker-aware boundaries to be overridden
- **Solution**: When speaker segments are used:
  1. `restore_punctuation()` returns tuple `(text, sentences_list)` with pre-split sentences
  2. Spanish post-processing bypasses `_split_sentences_preserving_delims()` and returns pre-split list
  3. `podscripter.py` uses pre-split sentences, skipping `assemble_sentences_from_processed()`
  4. `_write_txt()` accepts `skip_resplit=True` to write sentences without final splitting
- **Result**: Speaker-aware sentence boundaries are preserved through entire pipeline to file output

## Key files

- `podscripter.py`: orchestration, chunking, ASR, output
- `sentence_splitter.py`: unified sentence splitting with punctuation provenance tracking (v0.4.0+)
- `sentence_formatter.py`: unified post-processing merge operations with speaker-aware decisions (v0.5.0+)
- `punctuation_restorer.py`: punctuation restoration, language formatting, capitalization
- `domain_utils.py`: centralized domain detection, masking, and Spanish false domain prevention
- `speaker_diarization.py`: speaker change detection and boundary extraction (optional feature)
- `Dockerfile`: runtime and dependency setup
