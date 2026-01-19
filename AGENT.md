# AI Agent Guidelines for podscripter

## Project Overview

**podscripter** is a multilingual audio transcription tool that generates accurate, punctuated transcriptions for language learning platforms like LingQ. The project uses Docker containerization and leverages state-of-the-art NLP models for punctuation restoration.

### Core Technologies
- **Whisper**: OpenAI's speech-to-text model for transcription
- **Sentence-Transformers**: For semantic understanding and punctuation restoration
- **Hugging Face Hub (caches)**: Used by `sentence-transformers`; managed via `HF_HOME` and optional offline mode
- **spaCy (mandatory)**: Lightweight NLP capitalization and entity awareness; models are baked into the Docker image (`en_core_web_sm`, `es_core_news_sm`, `fr_core_news_sm`, `de_core_news_sm`)
- **Docker**: Containerization for reproducible environments
- **Python 3.10+**: Primary development language

### Supported Languages
- Primary focus: English (en), Spanish (es), French (fr), German (de)
- Other languages may work via Whisper auto-detect but are considered experimental

## Architectural Principles

### 1. Container-First Design
- All development and testing must be done inside Docker containers
- Dependencies are managed through the Dockerfile
- Model caching is handled via Docker volumes
- Always mount caches when running containers:
  - `-v $(pwd)/models/huggingface:/root/.cache/huggingface`
  - `-v $(pwd)/models/sentence-transformers:/root/.cache/torch/sentence_transformers`
  - `-v $(pwd)/audio-files:/app/audio-files`

### 2. Model Caching Strategy
- Faster-Whisper (Whisper) models cached via Hugging Face Hub under `/root/.cache/huggingface` (mounted from `models/huggingface`)
- Sentence-Transformers cached in `/root/.cache/torch/sentence_transformers`
- Hugging Face models cached in `/root/.cache/huggingface`
- Pyannote speaker diarization models cached in `/root/.cache/pyannote` (mounted from `models/pyannote`)
- Use `HF_HOME` environment variable (avoid deprecated `TRANSFORMERS_CACHE`)
- Use `PYANNOTE_CACHE` environment variable for pyannote.audio 3.x (version 4.0+ uses `HF_HOME` instead)
- Prefer offline use when cache exists: set `HF_HUB_OFFLINE=1` for tests/runs to avoid 429 rate limits
- Use a singleton model loader to avoid repeated model instantiation within a process
- SpaCy capitalization is always enabled, with models baked into the image (see Docker Best Practices)
- Sentence-Transformers loader: only load from a direct cache path if `modules.json` or `config_sentence_transformers.json` exists in that folder; otherwise load by name with `cache_folder` to avoid the "Creating a new one with mean pooling" message while still using caches. Also sets `HF_HOME` and may set `HF_HUB_OFFLINE=1` when a local model directory is used.

### 2a. Post-Processing Formatting (SentenceFormatter)
- **`SentenceFormatter`**: Unified class for all post-processing merge operations (v0.5.0+)
- Consolidates domain, decimal, appositive, and emphatic word merges in one location
- Speaker-aware merge decisions: NEVER merges different speakers
- Merge provenance tracking for debugging
- Usage:
  ```python
  from sentence_formatter import SentenceFormatter
  
  formatter = SentenceFormatter(
      language='es',
      speaker_segments=speaker_word_ranges  # None when diarization disabled
  )
  formatted_sentences, merge_metadata = formatter.format(sentences)
  ```
- Merge types:
  - **Domain merges**: "example." + "com" → "example.com" (with natural language guards)
  - **Decimal merges**: "99." + "9%" → "99.9%"
  - **Spanish appositive merges**: ", de Texas. Estados Unidos" → ", de Texas, Estados Unidos"
  - **Emphatic word merges**: "No. No. No." → "No, no, no." (ES/FR/DE)
- Debug flag: `--dump-merge-metadata` writes merge provenance to `<basename>_merges.txt`
- Backward compatible: When `speaker_segments=None`, all merges work identically to pre-v0.5.0 behavior

### 2b. Transcription Orchestration (Whisper usage)
- Two supported modes:
  - Single-call transcription (recommended when resources allow): process full file in one call so Whisper maintains context. Enabled via `--single` flag in `podscripter.py`.
  - Overlapped-chunk transcription (fallback): default path with 480s chunks and 3s overlap.
- Common settings:
  - `vad_filter=True` for both modes by default; `speech_pad_ms=200`.
  - CLI exposes VAD controls: `--no-vad` to disable VAD, and `--vad-speech-pad-ms <int>` to adjust padding when VAD is enabled.
  - `condition_on_previous_text=True` to keep context continuity
  - For chunked mode, pass `initial_prompt` using the last ~200 characters of accumulated text
  - Deduplicate overlap during merge using global timestamps (skip segments that finish before the prior chunk’s end)
  - Keep `beam_size` modest (1–3) for long files; `compute_type` default is `auto` on CPU
  - `PROMPT_TAIL_CHARS=200` controls how much trailing text is used for `initial_prompt`
  - Optional raw dump for debugging: `--dump-raw` writes a raw Whisper dump (`<basename>_raw.txt`) alongside the chosen output format

### 3. Modular Processing Pipeline
```
Audio Input → Chunking (overlap) → Whisper Transcription (with language detection, optional VAD) → Dedup/Globalize Segments → Punctuation Restoration → Sentence Splitting → Output (TXT/SRT)

- CLI flags (argparse in `podscripter.py`):
  - `--output_dir <dir>` (required)
  - `--language <code>|auto` (default `auto`)
  - `--output_format {txt|srt}` (default `txt`)
  - `--single` (bypass manual chunking)
  - `--model {tiny,base,small,medium,large,large-v2,large-v3}` (default `medium`; precedence: CLI > `WHISPER_MODEL` env > default)
  - `--translate` (Whisper `task=translate`; punctuation uses English rules)
  - `--compute-type {auto,int8,int8_float16,int8_float32,float16,float32}` (default `auto`)
  - `--beam-size <int>` (beam size for decoding; default 3)
  - `--no-vad` (disable VAD filtering; default is enabled)
  - `--vad-speech-pad-ms <int>` (padding in ms when VAD is enabled; default 200)
  - `--dump-raw` (also write raw Whisper output for debugging to `<basename>_raw.txt` in `--output_dir`)
  - `--enable-diarization` (enable speaker diarization; default disabled)
  - `--min-speakers <int>` (minimum speakers for diarization; optional)
  - `--max-speakers <int>` (maximum speakers for diarization; optional)
  - `--hf-token <str>` (Hugging Face token for first-time model download)
  - `--dump-diarization` (write diarization debug dump to `<basename>_diarization.txt`)
  - `--dump-merge-metadata` (write merge provenance to `<basename>_merges.txt`)
  - `--quiet`/`--verbose`/`--debug` (mutually exclusive; default `--verbose`)
    - `--debug` shows detailed sentence splitting decisions including speaker segment tracking
```

## Coding Style & Standards

### Python Code Style
- Follow PEP 8 conventions
- Use descriptive variable names
- Include comprehensive docstrings for functions
- Prefer explicit imports over wildcard imports

### File Organization
- Core logic in `punctuation_restorer.py`
- Transcription orchestration in `podscripter.py`
- Domain detection and masking utilities in `domain_utils.py`
- Tests in `tests/` directory with descriptive names
- Documentation in markdown files

### Error Handling
- Use try-catch blocks for model loading and API calls
- Provide meaningful error messages
- Handle rate limiting gracefully (especially for HuggingFace API)
- Raise typed exceptions at the source and handle them centrally in the CLI:
  - `InvalidInputError`, `ModelLoadError`, `TranscriptionError`, `OutputWriteError`
  - Exit codes: 2=input, 3=model load, 4=transcription, 5=write, 1=unexpected

### Logging
- Use a single logger named `podscripter` configured in `podscripter.py`
- Levels are controlled by CLI flags (mutually exclusive):
  - `--quiet` → ERROR (minimal output)
  - `--verbose` → INFO (default; informative lifecycle logs)
  - `--debug` → DEBUG (detailed sentence splitting decisions, speaker tracking, connector word evaluations)
- Debug messages use `logger.debug()` and include:
  - Speaker segment conversions (boundaries → chars → words)
  - Connector word evaluations with speaker continuity checks
  - Sentence ending decisions at every potential boundary
- Replace ad-hoc prints with `logger.info/warning/error/debug`
- SRT path logs a normalization summary (trimmed cues, max/total seconds) when writing subtitles

## Project-Specific Rules

### 1. Punctuation Restoration Guidelines

#### General Approach
- Use `re.sub()` for text cleanup (preferred over other regex methods)
- Apply fixes broadly rather than one-off hacks for individual sentences
- Accept less than 100% accuracy for difficult edge cases
- Focus on semantic understanding over rule-based patterns
- Do not emit optional debug output for punctuated text

#### Centralized Punctuation System (NEW)
- **`_should_add_terminal_punctuation()`**: Single centralized function for all period/punctuation insertion decisions
- **`PunctuationContext`**: Context-aware processing with different rules for different scenarios:
  - `STANDALONE_SEGMENT`: For individual Whisper segments (avoids periods on incomplete phrases like "Ve a")
  - `SENTENCE_END`: For complete sentences that should get terminal punctuation
  - `FRAGMENT`: For sentence fragments during processing
  - `TRAILING`: For trailing fragments that may be carried forward
  - `SPANISH_SPECIFIC`: For Spanish-specific formatting contexts
- **`restore_punctuation_segment()`**: Segment-aware API for processing individual Whisper segments
- **Benefits**: Single source of truth, easier debugging, context-aware rules, prevents bugs like "Ve a" → "Ve a."

#### Formatting Responsibilities
- Perform punctuation, language-specific formatting, capitalization, comma insertion, hyphenation, and sentence-assembly utilities in `punctuation_restorer.py`.
- Keep `podscripter.py` focused on orchestration (I/O, model selection, mode, calling helpers) and output writing. Assembly behavior is invoked via helpers in `punctuation_restorer.py`:
  - Ellipsis continuation and domain-aware splitting are exposed via `assemble_sentences_from_processed(processed, language)` (public API)
  - Comma spacing is centralized in `_normalize_comma_spacing(text)` and MUST NOT be re-implemented inline. This helper:
    - removes spaces before commas,
    - deduplicates multiple commas,
    - ensures a single space after commas.
    - Trade-off: thousands like "1,000" become "1, 000". This is acceptable given the priority to correctly space number lists (e.g., episode numbers: "147,151,156" → "147, 151, 156").
  - The detailed helpers for initial/acronym normalization, French connector merges, etc., are private (`_normalize_initials_and_acronyms`, `_normalize_dotted_acronyms_en` (legacy alias), `_fr_merge_short_connector_breaks`, and others)
  - Segment carry-over: when a segment ends without terminal punctuation, carry the trailing fragment into the next segment for French and Spanish
  - SRT normalization: reading-speed-based cue timing to prevent lingering in silences (defaults: cps=15.0, min=2.0s, max=5.0s, gap=0.25s)

#### Language-Specific Heuristics (recent)
- Spanish:
  - Preserve embedded questions mid-sentence: keep and properly pair `¿ … ?` inside larger sentences; do not strip mid-sentence `¿`
  - Coordinated yes/no questions: detect verb-initial starts with later `o <verbo>` (e.g., "¿Quieren … o prefieren …?")
  - Greeting/lead-in formatting (guarded):
    - Add comma after greeting phrases ("Hola …,") and set-phrases ("Como siempre,")
    - Do not add a comma after `Hola` when followed by prepositional phrases like `a`/`para` (e.g., "Hola para todos")
    - When a greeting is followed by an inverted mark (`¿`/`¡`), insert the comma only if there isn't one already immediately before the mark (prevents duplicate commas)
    - If a sentence starts with a greeting and contains an embedded Spanish question later, do not inject a leading `¿` at the very start; keep only the embedded `¿ … ?`
    - The "remove comma after Hola" rule also applies when the greeting is preceded by `¿` or `¡`
  - Diacritic-insensitive detection for gating where appropriate (e.g., `como` ~ `cómo`)
  - Appositive introductions: format as "Yo soy <Nombre>, de <Ciudad>, <País>"
  - Soft sentence splitting: avoid breaking inside entities; merge `auxiliar + gerundio` and possessive splits ("tu español")
  - Automatic spaCy capitalization: capitalize entities/PROPN; keep connectors (de, del, y, en, a, con, por, para, etc.) lowercase
  - Pipeline order correction (Spanish): run spaCy capitalization before greeting/lead‑in comma insertion to avoid capitalization feedback loops and misclassification
  - Do not split on ellipses mid-clause; keep continuation after `...`/`…` within the same sentence
  - Comprehensive domain protection: preserve single TLDs (`espanolistos.com`, `github.io`, etc.) and compound TLDs (`bbc.co.uk`, `amazon.com.br`, `cambridge.ac.uk`, etc.) as single tokens
  - Supported TLDs: Single (`com|net|org|co|es|io|edu|gov|uk|us|ar|mx|de|fr|it|nl|br|ca|au|jp|cn|in|ru`), Compound (`co.uk|com.ar|com.mx|com.br|com.au|co.jp|co.in|gov.uk|org.uk|ac.uk`)
  - Domain assembly logic: use a general multi-pass merge loop to restore split domains across sentence boundaries and intermediate transformations
  - Domain masking protects domains during space insertion, capitalization, and sentence splitting to prevent formatting issues; Spanish processing wraps all transformations with masking/unmasking so `www.example.com` and compound TLDs remain intact
  - Spanish false domain prevention: excludes common Spanish words (e.g., `uno.de` → `uno. de`, `naturales.es` → `naturales. es`) from being treated as domains through centralized exclusion lists
  - Spanish-only `.de` and `.es` exclusion: when language is Spanish, the `.de` and `.es` TLDs are not treated as domains to avoid false positives with common words "de" (preposition) and "es" (verb "is") (e.g., `tratada.de` → `tratada. de`, `noche.de` → `noche. de`, `naturales.es` → `naturales. es`, `no.es` → `no. es`)
  - Normalize mismatched inverted punctuation: leading `¡` with trailing `?` becomes a proper question `¿...?`
  - TXT writer ensures one sentence per paragraph; domains preserved and properly formatted with lowercase TLDs
  - TXT writer multilingual location protection: during final sentence splitting, protects appositive location patterns like ", <preposition> <Location>. <Location>" to avoid breaking location descriptions. Applies across EN/ES/FR/DE using language-specific prepositions (ES: de; EN: from/in; FR: de/du/des; DE: aus/von/in). Examples: ", de Texas. Estados Unidos", ", from Texas. United States", ", de Paris. France", ", aus Berlin. Deutschland".
  - Location appositive punctuation normalization (EN/ES/FR/DE): during punctuation restoration, convert erroneous period in ", <preposition> <Location>. <Location>" into a comma → ", <preposition> <Location>, <Location>". Uses language-specific prepositions (ES: de; EN: from/in; FR: de/du/des; DE: aus/von/in).
  - Location normalization new-sentence guard: do not merge when the following fragment starts a new sentence with a subject (e.g., "Y yo …", "And I …", "Et je …", "Und ich …").
  - Direct comma-separated location normalization: during punctuation restoration, fix patterns like "City, Region. and/pero/y …" to keep the location intact and continue the clause (e.g., "Austin, Texas. Y allá" → "Austin, Texas y allá").
- French: apply clitic hyphenation for inversion (e.g., `allez-vous`, `est-ce que`, `qu'est-ce que`, `y a-t-il`, `va-t-il`)
- German: insert commas before common subordinating conjunctions (`dass|weil|ob|wenn`) when safe; expand question starters/modals; capitalize `Ich` after punctuation; capitalize `Herr/Frau + Name`; minimal noun capitalization after determiners; maintain a small whitelist of proper nouns
- English/French/German: add greeting commas (`Hello, ...`, `Bonjour, ...`, `Hallo, ...`) and capitalize sentence starts

#### Cross-language: Location appositive normalization
- Normalize location appositives by replacing an erroneous period before the second location label with a comma.
- Applies across EN/ES/FR/DE using language-specific preposition cues to avoid false positives.

#### Spanish appositive merge guard (safety)
- Only merge appositive location breaks when the second sentence is just a location continuation with minimal trailing content.
- Pattern handled: `"..., de <Proper>. <Proper> ..." -> "..., de <Proper>, <Proper> ..."`.
- Guard prevents accidental merging when the next sentence contains additional clauses (e.g., "Hola para todos ...").

### 2. Testing Requirements
- All tests must run inside Docker container
- Create focused test files for specific bugs/issues
- Use descriptive test names that explain the scenario
- Test both individual functions and full transcription pipeline
- Ensure model caches are mounted for reliable, fast tests and to avoid 429s
- Unit guardrails included by default in the suite:
  - `test_sentence_assembly_unit.py`: ES ellipsis continuation and domain handling; FR short-connector merge
  - `test_chunk_merge_helpers.py`: verifies `_dedupe_segments` and `_accumulate_segments` integrity
- Spanish-only tests to be included by default:
  - `test_spanish_embedded_questions.py` (embedded `¿ … ?` clauses)
  - `test_human_vs_program_intro.py` (human-vs-program similarity on intro + extended lines; token-level F1 thresholds)
    - Intro average F1 threshold: ≥ 0.80
    - Overall average F1 threshold: ≥ 0.70
  - `test_spanish_helpers.py` (unit tests for `_es_*` helpers: tags, collocations, merges, pairing, greetings)
    - Includes greeting guards to prevent duplicate commas and ensure natural "Hola para/a ..." behavior
    - Includes embedded Spanish samples and a human-reference excerpt; computes SequenceMatcher ratio, token F1, and sentence-level alignment metrics (no external media required)
  - `test_spanish_domains_and_ellipses.py` (comprehensive domain handling including single/compound TLDs and ellipsis continuation; tests triple merge functionality)
  - `test_spanish_false_domains.py` (tests prevention of Spanish words being treated as domains; e.g., `uno.de` → `uno. de`)
  - `test_domain_utils.py` (comprehensive tests for centralized domain detection, masking, and exclusion utilities)
  - `test_initials_normalization.py` (WIP: tests for person initial normalization like "C.S. Lewis"; currently documents expected behavior)
  - Run selection controlled by env flags in `tests/run_all_tests.py`: `RUN_ALL`, `RUN_MULTILINGUAL`, `RUN_TRANSCRIPTION`, `RUN_DEBUG`
 - The ad-hoc script `tests/test_transcription.py` is for manual experiments:
  - Defaults: model `medium`, device `cpu`, compute type `auto`
  - Toggles: `--single`, `--chunk-length N`, `--apply-restoration`, `--dump-raw` (writes raw Whisper output for debugging)
  - It also exposes VAD toggles (`--no-vad`, `--vad-speech-pad-ms`) strictly for debugging; the main CLI uses constants

### 3. Docker Best Practices
- Mount volumes for model caching and media:
  - `-v $(pwd)/models/huggingface:/root/.cache/huggingface`
  - `-v $(pwd)/models/sentence-transformers:/root/.cache/torch/sentence_transformers`
  - `-v $(pwd)/models/pyannote:/root/.cache/pyannote` (if using speaker diarization)
  - `-v $(pwd)/audio-files:/app/audio-files`
- Include all necessary environment variables in Dockerfile
- Avoid deprecated environment variables (e.g., `TRANSFORMERS_CACHE`)
- SpaCy capitalization is always enabled for all runs.
- For performance on long files (> 1 hour): prefer single-call mode if resources allow; otherwise use overlapped chunking with 3s overlap and deduplication on merge.
- Build efficiency tips:
  - Combine pip installs in a single `RUN` and use `--no-cache-dir`
  - Keep `COPY . .` as the last step to maximize layer caching
  - Use a `.dockerignore` to exclude large local media (e.g., `audio-files/`, `models/`) from the build context

### 4. Model Management
- Cache models to avoid repeated downloads
 - `WHISPER_MODEL` env var may be set; overridden by the CLI `--model` flag
- Handle model loading errors gracefully
- Use appropriate model sizes for the task
- Consider memory usage when selecting models

## Bug Fixing Guidelines

### 1. Problem Analysis
- Create focused test files to reproduce the issue
- Use step-by-step debugging to trace the problem
- Test fixes across multiple languages when applicable

### 2. Solution Approach
- Prefer general solutions over specific hacks
- Document the reasoning behind fixes
- Test edge cases and similar patterns
- Verify fixes don't break existing functionality

### 3. Testing Strategy
- Create `test_[specific_issue].py` files for focused testing
- Use `test_[component]_debug.py` for debugging complex issues
- Test both positive and negative cases
- Verify fixes work across all supported languages

### 4. Known Resolved Issues

#### Coordinating Conjunction Split Bug (Fixed)
**Problem**: Sentences were incorrectly split after coordinating conjunctions (e.g., "y", "and", "et", "und") when processing long texts, violating grammar rules.
- Example: `"errores y."` | `"Eco..."` instead of `"errores y eco..."`
- Only occurred when semantic splitting thresholds were triggered (long texts >219 words)

**Root Cause**: `_should_end_sentence_here()` lacked a general guard against ending sentences on coordinating conjunctions.

**Solution**: Added early guard (lines ~1549-1562 in `punctuation_restorer.py`) that prevents splits when current word is a coordinating conjunction:
- Spanish: y, e, o, u, pero, mas, sino
- English: and, but, or, nor, for, so, yet
- French: et, ou, mais, donc, or, ni, car
- German: und, oder, aber, denn, sondern

**Tests**: `test_conjunction_split_bug.py`, `test_episodio190_y_eco_bug.py`

#### Spanish Question Split Bug (Fixed)
**Problem**: Spanish questions starting mid-sentence with inverted question marks were incorrectly split into separate sentences.
- Example: `"Pues, ¿qué pasó, Nate?"` → `"Pues, ¿Qué?"` | `"¿Pasó, Nate?"` (incorrect)
- Occurred when semantic splitting thresholds triggered (long texts) and question word was capitalized

**Root Cause**: `_should_end_sentence_here()` lacked guards against splitting when:
1. Current word contains `¿` (Spanish inverted question mark)
2. Inside an unclosed Spanish question (seen `¿` but not closing `?`)

**Solution**: Added Spanish-specific guard (lines 1601-1610 in `punctuation_restorer.py`) that prevents splits:
- When current word contains `¿`
- When current chunk has unclosed question (`¿` present, `?` absent)

**Tests**: `test_pues_que_split_bug.py`

#### Preposition Split Bug (Fixed)
**Problem**: Sentences were incorrectly ending with prepositions across all supported languages, which violates basic grammar rules.
- Spanish example: `"entonces yo conocí a."` | `"Un amigo que trabajaba con cámaras"` (incorrect)
- Should be: `"entonces yo conocí a un amigo que trabajaba con cámaras"` (correct)
- English example: Ending with "to", "at", "from", "with", etc.
- French example: Ending with "à", "de", "avec", "pour", etc.
- German example: Ending with "zu", "bei", "mit", "von", etc.
- Occurred when semantic splitting thresholds were triggered in long texts

**Root Cause**: `_should_end_sentence_here()` was missing critical preposition guards:
1. Spanish preposition `'a'` (to) was not in the check lists at lines 1700-1702 and 1727-1728
2. Other Spanish prepositions `'ante'` (before), `'bajo'` (under) were also missing
3. No preposition guards existed for English, French, or German

**Solution**: Added comprehensive preposition guards (lines ~1700-1745 in `punctuation_restorer.py`):
- Spanish: Added `'a'`, `'ante'`, `'bajo'` to existing preposition lists
- English: Added guard for common prepositions (to, at, from, with, by, of, in, on, for, about, etc.)
- French: Added guard for common prepositions (à, de, en, pour, avec, sans, sous, sur, dans, chez, etc.)
- German: Added guard for common prepositions (zu, an, auf, aus, bei, mit, nach, von, vor, in, für, etc.)

**Tests**: `test_preposition_split_bug.py`, `test_preposition_split_long_text.py`

#### Continuative/Auxiliary Verb Split Bug (Fixed)
**Problem**: Sentences were incorrectly breaking at continuative/auxiliary verbs (e.g., "estaba", "era", "was", "were", "était") when processing long texts, violating grammar rules across all supported languages.
- Spanish example: `"...y yo estaba en Colombia y estaba."` | `"Continuando con la universidad..."` (incorrect)
- Should be: `"...y yo estaba en Colombia y estaba continuando con la universidad..."` (correct)
- English example: Ending with "was", "were", "had" (e.g., "I was." | "Working...")
- French example: Ending with "était", "avait" (e.g., "il était." | "En train de...")
- German example: Ending with "war", "hatte" (e.g., "er war." | "Gerade...")
- Only occurred when semantic splitting thresholds were triggered in long texts (>200 words)

**Root Cause**: `_should_end_sentence_here()` lacked a guard against ending sentences on continuative/auxiliary verbs. These verbs are grammatically incomplete without their complement.

**Solution**: Added comprehensive guard (lines ~1601-1645 in `punctuation_restorer.py`) that prevents splits when current word is a continuative/auxiliary verb:
- Spanish: Imperfect tense (estaba, era, tenía, había, iba, hacía, podía, debía, quería, sabía, venía, decía) and perfect auxiliaries (he, has, ha, hemos, habéis, han)
- English: Past continuous (was, were), perfect auxiliaries (had, been, have, has)
- French: Imperfect tense (étais, était, étions, étiez, étaient, avais, avait, allais, allait, faisais, faisait) and auxiliary (avait)
- German: Imperfect tense (war, hatte, ging, machte) and modal verbs (konnte, wollte, musste, sollte)

**Tests**: `test_continuative_verb_split_bug.py`

#### Speaker Diarization Short Segment Bug (Fixed)
**Problem**: Speaker changes were not triggering sentence breaks for short segments (e.g., "Mateo 712") because the general `min_chunk_before_split` check (20 words for Spanish) happened BEFORE the speaker/Whisper boundary checks.
- Example: Speaker change after "Mateo 712" (2 words) should break, but didn't because 2 < 20
- Result: `"Mateo 712 Bueno, Andrea, creo que..."` instead of `"Mateo 712." | "Bueno, Andrea, creo que..."`

**Root Cause**: In `_should_end_sentence_here()`, the check `if len(current_chunk) < min_chunk_before_split: return False` was an early return that happened BEFORE checking speaker/Whisper boundaries. Short segments would return early without ever checking if there was a speaker change.

**Solution**: 
1. Moved speaker/Whisper boundary checks to happen BEFORE the general `min_chunk_before_split` check
2. Speaker boundaries now use a minimal threshold (1 word, v0.6.1) since speaker changes are definitive
3. Pass speaker boundaries separately to `restore_punctuation()` (not merged with Whisper boundaries) so they can use different thresholds
4. Added `_convert_speaker_timestamps_to_char_positions()` to properly convert speaker timestamps to character positions

**Tests**: `test_speaker_boundary_conversion.py`

#### Connector Word with Diarization Bug (Fixed - v0.3.1)
**Problem**: When using speaker diarization (`--enable-diarization`), sentences were incorrectly starting with coordinating conjunctions ("Y", "O", "and", "et", "und") even when the same speaker was continuing their speech.
- Spanish example: `"Yo soy Andrea de Santander, Colombia."` | `"Y yo soy Nate de Texas, Estados Unidos."` (incorrect - same speaker continues)
- Should be: `"Yo soy Andrea de Santander, Colombia. Y yo soy Nate de Texas, Estados Unidos."` (correct)
- English example: `"I live in Texas."` | `"And I work remotely."` → should be one sentence when same speaker
- Only occurred when `--enable-diarization` was enabled; non-diarization mode was unaffected
- Affected ALL supported languages (ES/EN/FR/DE) when diarization was enabled

**Root Causes**: Multiple compounding issues were discovered during debugging:
1. **Primary**: `_should_end_sentence_here()` checked if next word was a connector but didn't verify speaker continuity. It correctly prevented breaks at connectors in general, but didn't account for the special case where the same speaker continues with "Y" after a Whisper segment boundary.
2. **Secondary - Re-splitting bug #1**: Spanish post-processing in `_transformer_based_restoration()` called `_split_sentences_preserving_delims()` to add inverted question marks, which re-split text by punctuation marks and undid the speaker-aware semantic boundaries.
3. **Secondary - Re-splitting bug #2**: `podscripter.py` called `assemble_sentences_from_processed()` after `restore_punctuation()` returned, which re-split the carefully merged text by punctuation marks.
4. **Secondary - Re-splitting bug #3**: `_write_txt()` function re-split each sentence by punctuation marks before writing to file, undoing all previous work.

**Solution**: Multi-layered fix addressing all issues:
1. **Enhanced speaker tracking**: 
   - Added `_convert_speaker_segments_to_char_ranges()` to convert time-based speaker data to character positions
   - Added `_convert_char_ranges_to_word_ranges()` to convert character positions to word indices
   - Added `_get_speaker_at_word()` to query speaker at any word position
   - Modified `_should_end_sentence_here()` to check: when next word is a connector AND current_speaker == next_speaker, return `False` to prevent break
2. **Fixed re-splitting #1 (Spanish post-processing)**: 
   - When speaker segments are provided, format each sentence individually for question marks
   - Skip the `_split_sentences_preserving_delims()` call that was re-splitting the text
   - Return pre-split sentences list directly
3. **Fixed re-splitting #2 (restore_punctuation)**: 
   - Changed `restore_punctuation()` return signature to tuple: `(processed_text, sentences_list)`
   - When speaker segments used: return pre-split sentences that bypass `assemble_sentences_from_processed()`
   - Updated `podscripter.py` to use pre-split sentences when available
4. **Fixed re-splitting #3 (_write_txt)**: 
   - Added `skip_resplit` parameter to `_write_txt()`
   - When True, write sentences as-is without final punctuation-based splitting
   - Set `skip_resplit=True` when speaker segments were used

**Key insight**: The bug revealed that sentence splitting was happening in **three separate places** in the pipeline, making it extremely difficult to debug and maintain speaker-aware boundaries. Each re-splitting step was undoing the careful work of the semantic splitter.

**Tests**: Verified with Episodio212.mp3 (33-minute Spanish podcast with 3 speakers, 86 speaker changes). Result: Zero sentences starting with connector words. All 34+ connector word boundaries correctly merged when same speaker continues.

### 5. Known Resolved Issues (Recent)

#### Spanish Inverted Question Split at Whisper Boundaries (RESOLVED - v0.6.2)
**Problem**: Spanish questions starting with `¿` were being incorrectly split in the middle when a Whisper segment boundary occurred before the closing `?`.
- Example: `"¿qué cambios ha habido desde la pandemia?"` → `"¿qué cambios ha habido."` | `"Desde la pandemia?"` (incorrect)
- Also affected: `"¿cómo fueron esos meses donde era extremadamente estricto?"` similarly split mid-question
- The first part ended with a period (added by `_should_add_terminal_punctuation`) and the second part lost its opening `¿`

**Root Cause**: In `sentence_splitter.py`, the Whisper boundary handling in `_should_end_sentence_here()` returned `True` (allow split) when the next word was not a connector, WITHOUT checking if we were inside an unclosed Spanish inverted question (`¿` present but no closing `?` yet). The Spanish inverted question guard in `_passes_language_specific_checks()` was never reached because the Whisper boundary handling returned early.

**Solution Implemented (v0.6.2)**:
1. **New helper method `_is_inside_unclosed_question()`**: Detects when current sentence chunk contains unclosed `¿...?` or `¡...!`
2. **Updated Whisper boundary handling**: Calls `_is_inside_unclosed_question()` before allowing splits at Whisper boundaries
3. **Updated `_passes_language_specific_checks()`**: Now uses the helper method (DRY principle)
4. **Exclamation support**: Also protects unclosed `¡...!` constructs

**Key Insight**: Spanish inverted punctuation (`¿...?` and `¡...!`) creates grammatical constructs that must not be split mid-sentence. Unlike regular sentences where Whisper boundaries are reasonable split points, these constructs have explicit opening and closing markers that must be preserved together.

**Impact**: All Spanish transcriptions with questions or exclamations that span multiple Whisper segments. The fix respects speaker changes - if diarization indicates a different speaker mid-question, the split is still allowed per user requirements.

**Tests**: Verified with Episodio221.mp3 examples. New tests confirm questions are preserved as single sentences.

#### Double Inverted Question Mark in Embedded Questions (RESOLVED - v0.6.2)
**Problem**: Spanish sentences with embedded questions incorrectly received a second `¿` at the start.
- Example: `"Valentina, cuéntenos, ¿usted ahorita está estudiando en persona o virtual y cómo son las clases?"` became `"¿Valentina, cuéntenos, ¿usted ahorita..."` (two `¿` but one `?`)
- Grammatically incorrect: Spanish requires each `¿` to pair with a `?`

**Root Cause**: Three places in `punctuation_restorer.py` checked if a sentence **started** with `¿`, but not if one already **existed** mid-sentence (embedded question):
1. Line 1538: Question patterns check
2. Line 1770: Semantic question gate
3. Line 4219: Coordinated questions pattern

**Solution Implemented (v0.6.2)**:
Added `'¿' not in sentence` guard to all three locations, preventing `¿` from being added at the start when the sentence already contains an embedded `¿`.

**Key Insight**: Embedded questions in Spanish (e.g., "Valentina, cuéntenos, ¿usted...?") are grammatically valid. The question portion starts mid-sentence with `¿` and ends with `?`. Adding another `¿` at the sentence start creates mismatched punctuation.

**Tests**: Verified with Episodio221.mp3 examples.

#### Speaker Boundary Splits Blocked by Connector Checks (RESOLVED - v0.6.1)
**Problem**: Speaker boundaries were not creating sentence breaks when followed by connector words, despite v0.4.3 establishing that "speaker boundaries ALWAYS create splits."
- Example: `"Malala. Sí. Bueno, Malala nació el 12 de julio de 1997 y es reconocida por? Es."` kept as one paragraph
- "Malala." was Andrea, "Sí. Bueno, Malala nació..." was Nate, "y es reconocida por?" was Andrea, "Es." was Nate
- Despite 4 speaker changes, all utterances appeared in the same paragraph

**Root Causes** (multiple compounding issues):
1. **Connector word checks blocking speaker splits** (v0.6.0 regression): Added checks that skipped speaker boundaries when current or next word was a connector (y, o, pero, de, a, etc.), violating the v0.4.3 principle
2. **Minimum chunk threshold too high**: `min_words_speaker` was 4, preventing single-word utterances like "Malala." from triggering splits
3. **Period removal at different-speaker boundaries**: The period removal logic (`speaker_at_current == speaker_at_next or both None`) would remove periods even when speakers were different
4. **Off-by-one in boundary calculation**: `boundary_word = current_seg['end_word']` placed splits AFTER the first word of the new speaker instead of BEFORE it (since `end_word` is exclusive)
5. **Inclusive end_word comparison**: `_get_speaker_at_word()` used `<= end_word` but `end_word` is exclusive per `_convert_char_ranges_to_word_ranges()`

**Solution Implemented (v0.6.1)**:
1. **Removed connector checks**: Speaker boundaries now unconditionally create splits (restored v0.4.3 behavior)
2. **Reduced threshold to 1**: Single-word utterances like "Malala." now properly split
3. **Stricter period removal**: Only remove periods when `speaker_at_current is not None AND speaker_at_next is not None AND speaker_at_current == speaker_at_next`
4. **Fixed boundary calculation**: Changed to `boundary_word = current_seg['end_word'] - 1`
5. **Fixed speaker lookup**: Changed from `<= end_word` to `< end_word`

**Key Insight**: The v0.6.0 connector checks were well-intentioned (prevent orphaned connectors) but violated a more fundamental principle: speaker boundaries are definitive signals that should never be skipped. The period removal and connector merging logic already has its own speaker continuity checks, so blocking speaker splits was redundant and harmful.

**Impact**: All diarization-enabled transcriptions now correctly separate speakers even when utterances are single words or start with connector words.

**Tests**: All 35 tests pass. Verified with Episodio218-trim.mp3.

#### False Domain Merge in Natural Language (RESOLVED - v0.4.4)
**Problem**: Domain merge logic was incorrectly merging sentences when the word before a period matched a TLD in the domain list.
- Example: "...pero jugar tenis juntos, bueno, si es que yo aprendo y soy capaz de jugar. Es que vamos a tratar de tener lecciones de tenis en Colombia."
- Andrea's sentence ends with "jugar." (Spanish for "to play")
- Nate's sentence starts with "Es que vamos..." ("Es" starts sentence)
- Domain merge logic saw "jugar." + "Es..." and treated it as "jugar.es" (Spanish TLD)
- Result: Two different speakers' sentences merged into one paragraph

**Root Cause**: In `podscripter.py` lines 949-988, the domain merge logic uses regex `([A-Za-z0-9\-]+)\.$` to match any word ending with a period, then checks if the next sentence starts with a TLD pattern (com|net|org|co|**es**|io|...). Since "es" is the Spanish TLD and also a common Spanish word meaning "is", natural language like "jugar." + "Es que..." was matching as a broken domain.

**Solution Implemented (v0.4.4)**:
1. **Added natural language guards** (lines 962-972): Only merge if EITHER condition is true:
   - Current sentence is short (< 50 characters) - indicates standalone domain mention
   - Label before period is capitalized - indicates proper noun/brand name (e.g., "Google.Com")
2. **Preserves legitimate domain merges**: Short mentions like "Visit example.com" or capitalized "Check out Google.Com" still merge correctly
3. **Prevents false positives**: Long natural language sentences ending with lowercase words are excluded from domain merging

**Key Insight**: Domain names in transcriptions have distinct characteristics:
- **Short**: Standalone mentions like "Check out example.com"
- **Capitalized**: Brand names like "Google.Com" or "GitHub.Io"
- **URL context**: Part of explicit URL mentions

Long natural language sentences (> 50 chars) ending with lowercase words that happen to match TLDs are almost never actual domains. The 50-character threshold provides a generous buffer while catching nearly all false positives.

**Impact**: All transcriptions where sentences end with words matching TLD patterns (especially "es", "de", "co", "io" in Spanish/German/Portuguese contexts). Critical for diarization-enabled transcriptions where this bug could merge different speakers.

**Tests**: Verified with Episodio212.mp3 (Andrea and Nate sentences now properly separated).

#### Whisper Boundary Skipping Too Aggressive (RESOLVED - v0.4.3)
**Problem**: Legitimate sentence-ending periods were being removed when speaker changes occurred many words later in the text.
- Example: "Dile adiós a todos esos momentos incómodos Entonces, empecemos." (missing period after "incómodos")
- Whisper correctly added period after "incómodos" (end of segment)
- Speaker change occurred 11 words later at "Y yo soy Nate..."
- The 15-word lookahead window incorrectly marked the "incómodos" boundary as "skipped" and removed its period

**Root Cause**: In `sentence_splitter.py` line 558, the logic checked if ANY speaker boundary existed within the next 15 words and would skip the Whisper boundary. This was too aggressive - it assumed that if a speaker change was coming "soon", the current Whisper boundary must be a false split. But 15 words is far enough that there can be multiple legitimate sentences in between.

**Solution Implemented (v0.4.3)**:
1. **Reduced lookahead window**: Changed from 15 words to 3 words (only skip for true misalignment, not distant speaker changes)
2. **Added continuation checks**: Only skip Whisper boundary if BOTH conditions are met:
   - Speaker boundary is within 3 words
   - Next word is a connector (y, and, et, und) OR starts lowercase (indicates continuation)
3. **Preserve legitimate endings**: If next word is capitalized and not a connector, keep the Whisper boundary and period

**Key Insight**: The 15-word window was designed to handle cases where speaker boundaries are misaligned by 1-2 words from Whisper boundaries. But 15 words is too large - it can span multiple complete sentences. A 3-word window is sufficient for misalignment while preserving legitimate sentence endings.

**Impact**: All diarization-enabled transcriptions now correctly preserve periods at Whisper segment boundaries when they represent legitimate sentence endings.

**Tests**: All 35 tests pass. Verified with Episodio212-trim.mp3.

#### Speaker Changes with Connector Words (RESOLVED - v0.4.3)
**Problem**: When a speaker change occurred and the next speaker's sentence started with a connector word, the two different speakers' sentences were merged into one paragraph.
- Example: "Yo soy Andrea de Santander, Colombia. Y yo soy Nate de Texas, Estados Unidos." (Andrea and Nate on same line)
- Andrea's sentence ends with "Colombia."
- Nate's sentence starts with "Y yo soy Nate..."
- Despite being different speakers, they appeared in the same paragraph

**Root Cause**: In `sentence_splitter.py` lines 519-531, when a speaker boundary was detected, the code checked if the next word was a connector. If it was, it would SKIP the speaker boundary and fall through to other checks. This logic assumed that connectors always indicate same-speaker continuation, but it didn't verify that the speakers were actually the same.

**Solution Implemented (v0.4.3)**:
- **Simplified speaker boundary logic**: Speaker boundaries now ALWAYS create splits, regardless of whether the next word is a connector
- **Removed connector check**: Deleted the conditional that skipped speaker boundaries when next word was a connector
- **Rationale**: If there's a speaker change, the sentences should ALWAYS be separated, even if one happens to start with "and", "y", "et", or "und". The connector merging logic in `_process_whisper_punctuation()` already checks speaker continuity before merging, so it won't incorrectly merge different speakers.

**Key Insight**: Speaker boundaries are definitive - they represent actual speaker changes detected by the diarization model. They should never be skipped based on the next word. Connector words should only prevent splits when the SAME speaker continues, not when speakers change.

**Impact**: All diarization-enabled transcriptions now correctly separate different speakers' sentences with blank lines, even when one speaker starts with a connector word.

**Tests**: All 35 tests pass. Verified with Episodio212-trim.mp3 (Andrea and Nate now on separate lines).

#### Whisper-Added Periods at Skipped Boundaries (RESOLVED - v0.4.2)
**Problem**: When a Whisper segment boundary was skipped (because a speaker boundary was nearby), Whisper's period at that segment end remained in the text and caused an unwanted sentence split.
- Example: Whisper segment 10 ends with "ustedes." and segment 11 is "Mateo 712"
- Speaker boundary at 69.47s falls WITHIN segment 11 (67.89s-69.89s)
- We correctly skipped the Whisper boundary at "ustedes" (since speaker boundary was 2 words away)
- But the period after "ustedes." remained, causing: "...ustedes." | "Mateo 712." instead of "...ustedes Mateo 712."
- The actual speaker change happened AFTER "Mateo 712", so both should be in the same sentence

**Root Cause**: Whisper adds terminal punctuation to segments in its raw transcription. Even though we skipped Whisper boundaries during semantic splitting, the punctuation remained in the concatenated text.

**Solution Implemented (v0.4.2, refined in v0.4.3)**:
1. **Track skipped boundaries**: When a Whisper boundary is skipped (speaker boundary within 3 words AND next word is connector/lowercase), record the word index in `skipped_whisper_boundaries` set
2. **Remove periods at skipped positions**: In `_evaluate_boundaries`, after calling `_should_end_sentence_here` (which populates the skipped set), immediately check if the current word is at a skipped boundary and remove any trailing period
3. **Priority fix**: Moved Whisper boundary skip detection BEFORE `min_total_words_no_split` check so skips are tracked even in short texts
4. **Metadata tracking**: Record all removed periods with reason 'skipped_whisper_boundary' for debugging

**Impact**: Edge case affecting very short segments (< 3 words) that precede speaker changes. Now correctly handles these cases.

**Tests**: `test_whisper_skipped_boundary_detailed.py`, `test_whisper_boundary_debug.py`

#### Speaker Change Separation (RESOLVED - v0.4.2)
**Problem**: When speaker boundaries fell within Whisper segments (not at segment boundaries), sentences from different speakers were not separated by blank lines in the output.
- Example: "Estoy mejorando cada día con tu instrucción." (Nate speaking) followed by "¡Nate! Este año..." (Andrea speaking) appeared in the same paragraph
- The user's preference: Keep multiple short sentences from the SAME speaker together (no blank line), but ALWAYS separate different speakers with blank lines

**Root Causes**: Three compounding issues:
1. **Boundary extraction bug**: `SentenceSplitter._convert_segments_to_word_boundaries()` extracted `end_word` from ALL speaker segments, not just where speakers changed. This created false boundaries within a single speaker's utterance.
2. **Segment filtering**: `MIN_SPEAKER_SEGMENT_SEC` threshold in `speaker_diarization.py` was 2.0s, filtering out short but legitimate utterances like "¡Uy, Nate!" (Andrea's brief interjection).
3. **Misalignment handling**: `_convert_speaker_segments_to_char_ranges()` in `podscripter.py` used duration-based sorting that assigned entire Whisper segments to speakers. When a speaker boundary fell WITHIN a Whisper segment (e.g., speaker changes at 118.78s within Whisper segment 115.58s-119.08s), the entire segment was incorrectly assigned to one speaker.

**Solution Implemented (v0.4.2 - Partial, Superseded by v0.5.2)**:
1. **Fixed boundary extraction** (`sentence_splitter.py` lines 269-276): Loop through speaker segments pairwise, only add `end_word` boundary when `current_seg['speaker'] != next_seg['speaker']`
2. **Lowered segment threshold** (`speaker_diarization.py` line 60): Changed `MIN_SPEAKER_SEGMENT_SEC` from 2.0s to 0.5s to capture brief speaker changes while still filtering noise
3. **Rewrote speaker assignment** (`podscripter.py` lines 613-703): New algorithm:
   - For each Whisper segment, calculate temporal overlap with all speaker segments
   - Assign Whisper segment to speaker with **most overlap duration**
   - Group consecutive Whisper segments with same speaker into character ranges
   - Accurately handles cases where diarization boundaries don't align with Whisper boundaries

**Key Insight**: Speaker boundaries don't always align with Whisper segment boundaries. The diarization model detects speaker changes at arbitrary time points (e.g., 118.78s), while Whisper segments have their own boundaries (e.g., 115.58s-119.08s, 119.08s-127.08s). The old algorithm couldn't handle this misalignment.

**Impact**: Improved speaker separation significantly, but the "most overlap" assignment still lost boundaries when multiple speakers were in a single Whisper segment.

**NOTE**: The v0.4.2 "most overlap duration" approach was fully replaced in **v0.5.2** (2025-01-05) with a segment splitting approach that preserves ALL speaker boundaries instead of losing 36 out of 84. See "Speaker Segment Splitting Fix (v0.5.2 & v0.5.2.1)" section above for the complete solution.

**Tests**: Verified with Episodio212.mp3 (33-minute Spanish podcast, 3 speakers, 117 speaker changes after fix). Partial improvement achieved, full fix delivered in v0.5.2.

#### Period Before Same-Speaker Connectors (RESOLVED - v0.4.0)
**Problem**: When the same speaker continues speaking with a connector word ("Y", "and", "et", "und"), Whisper-added periods remained in the text even though our logic prevented starting a new sentence with the connector.
- Spanish example: `"...es importante tener una estructura como un trabajo. Y este meta es tu trabajo cada día."` (incorrect - same speaker, period shouldn't be there)
- Should be: `"...es importante tener una estructura como un trabajo y este meta es tu trabajo cada día."` (correct)
- English example: `"I work from home. And I enjoy it."` when same speaker → should be `"I work from home and I enjoy it."`
- Occurred across ALL supported languages (ES/EN/FR/DE) when diarization was enabled

**Root Cause**: 
1. **Whisper** adds terminal punctuation to its raw segment outputs (e.g., "trabajo." at end of segment)
2. Our v0.3.1 fix prevented sentences from **starting** with connectors when same speaker continues
3. BUT the fix didn't **remove** the Whisper-added periods that were already in the text before those connectors
4. Result: Period stayed, but connector didn't start new sentence → unnatural mid-sentence period

**Why It Was Complex**:
The punctuation restoration pipeline was scattered across 5+ stages (Whisper output, semantic splitting, punctuation application, Spanish post-processing, TXT writing). Whisper added periods BEFORE our pipeline ran, but speaker continuity decisions happened DURING the pipeline. No single stage had visibility into both.

**Solution Implemented (v0.4.0)**: 
Implemented the sentence splitting consolidation refactor. The new `SentenceSplitter` class:
- ✅ Tracks which periods came from Whisper segments
- ✅ Evaluates speaker continuity during boundary decisions
- ✅ Removes Whisper periods before same-speaker connectors
- ✅ All in one unified location (`sentence_splitter.py`)

**Status**: RESOLVED. The unified `SentenceSplitter` class now intelligently manages Whisper punctuation based on speaker context. Periods are automatically removed when the same speaker continues with a connector word.

#### Person Initials Normalization (WIP - Partial)
**Problem**: Names with initials like "C.S. Lewis" or "J.K. Rowling" in non-English transcriptions (especially Spanish) are incorrectly split into separate sentences.
- Example: `"es a C. S. Lewis porque"` → `"Es a c."` | `"S."` | `"Lewis porque..."` (3 sentences - incorrect)
- Should be: One sentence containing "C.S. Lewis"
- Occurs across all languages when English names with initials appear in transcriptions

**Root Cause**: spaCy's tokenizer/detokenizer reconstructs text with spaces during `_apply_spacy_capitalization()`, undoing the initial normalization. The normalized "C.S. Lewis" becomes "c. S. Lewis" which then splits into separate sentences.

**Current Progress** (WIP):
- ✅ Created `_normalize_initials_and_acronyms()` function (language-agnostic)
  - Removes spaces from person initials: "C. S. Lewis" → "C.S. Lewis"
  - Collapses organizational acronyms: "U. S. A." → "USA"
  - Distinguishes between person names and acronyms using context
- ✅ Applied globally before punctuation restoration in `podscripter.py`
- ✅ Added spaCy protection for initial patterns in `should_capitalize()`
- ✅ Added negative lookbehind to prevent space insertion after initials
- ✅ Comprehensive test suite: `tests/test_initials_normalization.py`

**What's Working**:
- English organizational acronyms normalize correctly (U.S. → US, USA, FBI, D.C. → DC)
- The normalization function itself works as designed

**What's Not Working Yet**:
- Spanish (and other non-English) transcriptions with English names still split incorrectly
- The issue is that spaCy processing re-adds spaces during text reconstruction

**Next Steps**:
- Implement initial-masking system (similar to domain masking in `domain_utils.py`)
- Mask initials before spaCy processing
- Unmask after all transformations
- This will protect initials throughout the entire processing pipeline

**Tests**: `test_initials_normalization.py` (comprehensive coverage for EN/ES/FR/DE)

### 6. Known Limitations and Open Issues

#### Diarization Misalignment Causing Sentence Fragments (OPEN - v0.6.1)
**Problem**: When pyannote diarization incorrectly attributes a brief interjection to the previous speaker, the period removal logic can create sentence fragments.

**Example**:
- Raw Whisper output: Segment 69: "Ajá." (343.18s-344.18s), Segment 70: "Y ella nació, como dije, en Pakistán." (344.18s-348.18s)
- Pyannote diarization: SPEAKER_00 (Nate) until 344.67s, then SPEAKER_01 (Andrea) starts
- **Problem**: Andrea says "Ajá" but pyannote assigns it to Nate (whose segment extends to 344.67s)
- **Result**: "Ajá y ella." appears as a fragment, split from "Nació, como dije, en Pakistán."
- **Expected**: "Ajá y ella nació, como dije, en Pakistán." as a single sentence (all Andrea)

**Root Cause Analysis**:
1. **Diarization misattribution**: Pyannote's SPEAKER_00 segment extends to 344.67s, but Andrea's "Ajá" occurs at 343.18s-344.18s. The brief interjection is incorrectly attributed to the previous speaker (Nate).
2. **Proportional character splitting**: `_convert_speaker_segments_to_char_ranges()` splits Whisper Segment 70 proportionally. Since the speaker boundary (344.67s) falls within the segment (344.18s-348.18s), the first few characters ("Y ella") are attributed to SPEAKER_00.
3. **Period removal logic**: The system sees "Ajá." (SPEAKER_00) followed by "Y" (also attributed to SPEAKER_00 due to proportional split). Since they're "same speaker", the period is removed and "Y" is lowercased → "Ajá y".
4. **Speaker boundary split**: A speaker boundary is correctly detected at word "ella" (where SPEAKER_01 starts), triggering a split → "Ajá y ella." becomes a fragment.

**Why Code Behaves Correctly**:
The code is functioning as designed given the diarization input:
- Period removal only applies when same-speaker continues (correct logic per v0.6.1)
- Speaker boundaries correctly trigger splits (restored v0.4.3 behavior)
- The issue is that the **input** (diarization segments) is misaligned with reality

**Why Increasing `min_words_speaker` Would Cause Regression**:
The v0.6.1 fix reduced `min_words_speaker` from 4 to 1 specifically to allow single-word utterances like "Malala." to trigger speaker boundary splits. Increasing this threshold would:
- ❌ Re-introduce the "Malala." bug where Andrea's single-word prompt was merged into Nate's sentence
- ❌ Violate the principle that "speaker boundaries are definitive signals"
- ❌ Trade one bug (fragments from misalignment) for another (missed splits from threshold)

**Potential Future Fixes** (not implemented):

**Option 1: Accept as a limitation**
- **Pros**: No code changes, no risk of regressions
- **Cons**: Users may see occasional fragments in transcriptions
- **When appropriate**: When diarization quality is generally good and fragments are rare

**Option 2: More cautious period removal near imminent speaker boundaries**
- **Approach**: Before removing a period, check if a speaker boundary is within the next N words. If so, preserve the period even if "same speaker" continues.
- **Pros**: Would preserve "Ajá." as a standalone sentence rather than creating "Ajá y ella."
- **Cons**: 
  - May preserve unnecessary periods in other cases
  - Adds complexity to period removal logic
  - Doesn't fix the root cause (diarization misalignment)
- **Implementation**: Modify `_process_whisper_punctuation()` to check for upcoming speaker boundaries before period removal

**Option 3: Post-processing fragment detection and merge**
- **Approach**: After sentence splitting, detect very short fragments (e.g., <4 words) that end with a period and are followed by a sentence from the same actual speaker. Merge them.
- **Pros**: Addresses symptom directly at output stage
- **Cons**:
  - Heuristic-based, may have false positives
  - "Same actual speaker" is hard to determine after splits (would need to preserve speaker info)
  - Violates "fix root cause, not symptoms" principle
  - Adds post-processing that could interact poorly with existing merge logic

**Current Status**: Documented as known limitation. The issue is fundamentally caused by diarization model inaccuracies at speaker transitions, not by our sentence splitting logic. The code correctly processes the (flawed) diarization input it receives.

**Files Affected**: N/A (no code changes)

**Tests**: N/A (manual verification with Episodio218.mp3)

#### Short Speaker Segment Filtering Causing Missed Splits (OPEN - v0.6.1)
**Problem**: The `MIN_SEGMENT_DURATION = 1.3s` filter (introduced in v0.6.0.1 to prevent rapid speaker flipping artifacts) can filter out legitimate short speaker segments at transitions, causing speaker boundaries to not trigger splits.

**Example**:
- Whisper segment 125 (507.81s - 509.81s): `"¿Cuántos habitantes tiene?"` (Andrea's question)
- Whisper segment 126 (510.81s - 511.81s): `"Guinea."` (Nate's answer starts)
- Pyannote diarization: SPEAKER_01 segment 107 (509.08s - 510.33s) duration = **1.25s**
- Speaker boundary at 510.33s: SPEAKER_01 → SPEAKER_00 (marked as ✓ INCLUDED)
- **Problem**: SPEAKER_01's segment (1.25s) is filtered because 1.25s < 1.3s threshold
- **Result**: `"¿Cuántos habitantes tiene? Guinea."` merged into one paragraph (both speakers)
- **Expected**: Andrea's question on separate line from Nate's answer

**Root Cause Analysis**:
1. **MIN_SEGMENT_DURATION threshold**: The 1.3s filter in `_convert_speaker_segments_to_char_ranges()` (line 851 of `podscripter.py`) removes all speaker segments shorter than 1.3 seconds
2. **Legitimate segment filtered**: SPEAKER_01's segment 107 (1.25s) that covers Andrea's question is filtered out
3. **Lost speaker attribution**: When processing Whisper segment 125, the system can't find any overlapping SPEAKER_01 segment (it was filtered)
4. **Missed split**: Without proper speaker attribution, the boundary at 510.33s doesn't trigger a paragraph split

**Why 1.3s Threshold Was Added (v0.6.0.1)**:
The threshold was increased from 0.5s to 1.3s to filter out rapid speaker flipping artifacts. From CHANGELOG.md:
> Fixed `"Métodos.pero"` → `"métodos pero"` - Short segments (0.56s, 0.93s, 1.10s, 1.28s) were causing rapid speaker flipping, preventing proper sentence merging

The artifacts ranged from 0.56s to 1.28s, so 1.3s was chosen to filter them all with a small margin.

**Why Lowering Threshold Would Risk Regression**:
- ❌ Setting threshold < 1.28s would re-introduce the rapid speaker flipping bugs from v0.6.0.1
- ❌ Those artifacts caused sentence fragmentation like `"Métodos.pero"` instead of `"métodos pero"`
- ❌ Trade-off: filtering artifacts (< 1.3s) vs. preserving legitimate short segments (~1.0-1.3s)

**Comparison to Diarization Misalignment Issue**:

| Aspect | Misalignment Issue | This Issue |
|--------|-------------------|------------|
| **Cause** | Pyannote attributes words to wrong speaker | Our filter removes valid speaker segment |
| **Manifestation** | Sentence fragments created | Speaker split not triggered |
| **Root** | External (pyannote accuracy) | Internal (our threshold too aggressive) |
| **Example** | "Ajá y ella." fragment | "¿Cuántos habitantes tiene? Guinea." merged |

**Potential Future Fixes** (not implemented):

**Option 1: Lower threshold to ~1.0s**
- **Pros**: Would preserve segments like 1.25s, fixing this bug
- **Cons**: May re-introduce some rapid speaker flipping artifacts (1.10s-1.28s range)
- **Risk**: Medium - artifacts in that range were real problems in v0.6.0.1

**Option 2: Context-aware filtering**
- **Approach**: Only filter short segments that are "sandwiched" between segments of the same speaker (likely artifacts). Preserve short segments that represent genuine speaker transitions (different speakers before/after).
- **Pros**: More accurate, preserves legitimate transitions while filtering artifacts
- **Cons**: More complex logic, potential edge cases

**Option 3: Two-tier threshold**
- **Approach**: Use a lower threshold (e.g., 0.8s) for segments at speaker transitions, higher threshold (1.3s) for mid-speaker segments
- **Pros**: Targeted filtering based on context
- **Cons**: Added complexity, need to define "at transition" precisely

**Current Status**: Documented as known limitation. The 1.3s threshold is a trade-off between filtering artifacts and preserving short legitimate segments. Segments between ~1.0-1.3s at speaker transitions may be filtered, causing missed splits.

**Files Affected**: `podscripter.py` line 851 (`MIN_SEGMENT_DURATION = 1.3`)

**Tests**: N/A (manual verification with Episodio220.mp3)

## Recent Refactors

### Sentence Splitting Consolidation (Completed - v0.4.0)

**Problem Identified**: Working on the connector word bug revealed that sentence splitting logic was scattered across multiple locations in the codebase:

1. **`_semantic_split_into_sentences()`** in `punctuation_restorer.py` (~line 1775)
   - Primary semantic splitter using transformer embeddings
   - Calls `_should_end_sentence_here()` for boundary decisions
   - Respects Whisper boundaries, speaker boundaries, grammatical guards

2. **Spanish post-processing in `_transformer_based_restoration()`** (~line 1464-1595)
   - Re-splits for inverted question mark insertion
   - Was calling `_split_sentences_preserving_delims()` 
   - Now bypassed when speaker segments used, but still runs for non-diarization

3. **`assemble_sentences_from_processed()`** in `punctuation_restorer.py` (~line 656)
   - Splits by punctuation marks: `[.!?]+`
   - Handles ellipsis continuation for ES/FR
   - Performs domain-aware splitting
   - Called by `podscripter.py` after `restore_punctuation()` returns

4. **`_write_txt()` in `podscripter.py`** (~line 276)
   - Final splitting before file output
   - Protects location appositives and number lists
   - Re-splits by: `(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡])`
   - Now accepts `skip_resplit` flag but still runs for non-diarization

5. **Whisper transcription** (external, before our pipeline)
   - Adds terminal punctuation to segment ends
   - These periods become part of the input text
   - Cannot be controlled by our code

**Impact**: 
- **Maintainability**: Bug fixes require changes in multiple places
- **Debuggability**: Difficult to trace why a sentence was split or where a period came from
- **Feature additions**: Adding new split logic requires coordinating across 5+ locations (including Whisper output handling)
- **Test complexity**: Must test splitting behavior at multiple pipeline stages
- **Speaker-aware issues**: Cannot easily coordinate punctuation decisions with speaker continuity across stages

**Specific Open Issue** (Related to Scattered Splitting):
- **Period Before Same-Speaker Connectors**: When the same speaker continues speaking with a connector word ("Y", "and", "et", "und"), Whisper-added periods remain in the text even though the sentence shouldn't split. 
  - Example: "...trabajo. Y este meta..." should be "...trabajo y este meta..."
  - Current fix prevents NEW SENTENCES from starting with connectors, but doesn't REMOVE existing periods
  - Root cause: Periods come from Whisper transcription, but speaker continuity decisions happen later in pipeline
  - Workaround needed: Post-processing to strip periods before same-speaker connectors
  - **With refactor**: This becomes a simple rule in `SentenceSplitter` - "Don't preserve Whisper periods before same-speaker connectors"

**Solution Implemented (v0.4.0)**:

**Phase 1: Consolidate Splitting Logic** (Completed - Breaking change)
1. Create single `SentenceSplitter` class in new file `sentence_splitter.py`:
   ```python
   class SentenceSplitter:
       def __init__(self, language, model, config):
           self.language = language
           self.model = model
           self.config = config
       
       def split(self, text, whisper_segments, speaker_segments, mode='semantic'):
           """Single entry point for all sentence splitting and punctuation management"""
           # Input: raw text with Whisper-added punctuation
           # Process: 
           #   1. Parse Whisper punctuation (where did periods come from?)
           #   2. Evaluate boundaries with speaker/semantic context
           #   3. Decide which Whisper periods to keep/remove
           #   4. Add new punctuation where needed
           # Output: sentences with correct punctuation
           pass
   ```

2. Modes to support:
   - `semantic`: Use transformer embeddings (current `_semantic_split_into_sentences`)
   - `punctuation`: Split by punctuation marks (current `assemble_sentences_from_processed`)
   - `hybrid`: Semantic + punctuation validation
   - `preserve`: Use pre-split sentences (current speaker segment path)

3. Move all splitting logic into `SentenceSplitter`:
   - `_should_end_sentence_here()` → `SentenceSplitter._evaluate_boundary()`
   - Grammatical guards → `SentenceSplitter._check_grammatical_validity()`
   - Speaker/Whisper boundary handling → `SentenceSplitter._process_boundaries()`
   - Language-specific rules → `SentenceSplitter._apply_language_rules()`
   - **Whisper punctuation handling** → `SentenceSplitter._process_whisper_punctuation()`
     - Tracks which periods came from Whisper segments
     - Removes periods before same-speaker connectors
     - Preserves periods at legitimate sentence boundaries

4. Replace all call sites with single `splitter.split()` call:
   - `restore_punctuation()` calls `splitter.split()` once
   - Remove splitting from Spanish post-processing
   - Remove `assemble_sentences_from_processed()` splitting (keep only domain/ellipsis logic)
   - Remove `_write_txt()` splitting (keep only domain protection)

5. **Solve period-before-connector issue**: 
   - `SentenceSplitter` tracks Whisper segment boundaries and their punctuation
   - When same speaker continues with connector, removes Whisper period
   - No post-processing patches needed

**Phase 2: Unify Return Values** (v0.4.1)
1. Have `restore_punctuation()` always return `(text, sentences_list)` tuple
2. Make `sentences_list` always populated (never None)
3. Remove conditional re-splitting paths in `podscripter.py`
4. Simplify `_write_txt()` to just format and write pre-split sentences

**Phase 3: Add Split Provenance** (v0.4.2)
1. Add metadata to track why each sentence was split:
   ```python
   @dataclass
   class Sentence:
       text: str
       split_reason: str  # 'speaker_change', 'whisper_boundary', 'semantic', 'punctuation'
       confidence: float
       start_word: int
       end_word: int
       speaker: Optional[str]
   ```

2. Enable debugging: `--dump-sentences` flag to write split provenance
3. Improve testing: Assert on split reasons, not just final output

**Benefits**:
- **Easier debugging**: Single place to add logging and debug; know exactly where each period came from
- **Better testing**: Test splitting logic in isolation
- **Cleaner code**: Remove duplicated split patterns
- **Future-proof**: Easy to add new splitting strategies
- **Provenance**: Understand why each split was made and which periods came from Whisper vs. our logic
- **Speaker-aware punctuation**: Coordinate speaker continuity with punctuation decisions in one place
- **Solves period-before-connector issue**: Handles Whisper punctuation intelligently based on speaker context

**Migration Strategy**:
- v0.4.0: Breaking change (major refactor)
- Provide migration guide for anyone extending the code
- Keep comprehensive tests to verify behavior unchanged
- Add deprecation warnings in v0.3.x for functions that will move

**Estimated Effort**: 
- Phase 1: ~40 hours (major refactor, testing across languages, Whisper punctuation handling)
- Phase 2: ~8 hours (cleanup and simplification)
- Phase 3: ~12 hours (metadata and debugging features)
- **Total**: ~60 hours for complete refactor

**Status**: COMPLETED (v0.4.0)

**Results Achieved**:
- ✅ All sentence splitting logic consolidated into single `SentenceSplitter` class
- ✅ Period-before-same-speaker-connector bug RESOLVED
- ✅ Punctuation provenance tracking implemented
- ✅ Comprehensive unit tests created (`tests/test_sentence_splitter_unit.py`)
- ✅ Maintainability significantly improved
- ✅ Future features (custom split rules, ML-based boundary detection) now much easier to implement

**Immediate Benefit Delivered**: Solved the period-before-same-speaker-connector issue. Now: "trabajo y este" (correct) instead of "trabajo. Y este" (wrong).

**Breaking Changes**:
- `restore_punctuation()` signature changed (accepts `whisper_segments` instead of `whisper_boundaries`)
- `restore_punctuation()` now ALWAYS returns tuple `(text, sentences_list)`
- Internal functions moved to `SentenceSplitter` class

**Migration**: Old parameters still accepted for backward compatibility but are deprecated.

## Documentation Standards

### README Updates
- Keep setup instructions current with Docker commands
- Document model caching benefits and setup
- Include troubleshooting sections for common issues
- Mention core technologies in Features section

### Code Comments
- Explain complex regex patterns
- Document language-specific logic
- Include examples for non-obvious fixes
- Reference bug numbers or issues when applicable

## Development Workflow

### 1. Feature Development
- Start with a focused test to define the requirement
- Implement the feature with proper error handling
- Test across all supported languages
- Update documentation as needed

### 2. Bug Fixes
- Reproduce the issue with a minimal test case
- Implement a general solution rather than specific hack
- Test the fix thoroughly across languages
- Document the fix and reasoning
- **Use centralized punctuation system**: When fixing punctuation bugs, use `_should_add_terminal_punctuation()` with appropriate context rather than adding scattered `text += '.'` logic

### 3. Testing
- Run tests inside Docker container
- Use `python3` command (not `python`)
- Create focused test files for specific issues
- Test both individual components and full pipeline

## Common Pitfalls to Avoid

### 1. Environment Issues
- Don't run tests outside Docker container
- Don't use deprecated environment variables
- Don't forget to mount model cache volumes

### 2. Code Quality
- Don't create one-off fixes for specific sentences
- Don't hardcode language-specific rules unnecessarily
- Don't ignore edge cases in punctuation patterns
- **Don't add scattered period insertion**: Use the centralized `_should_add_terminal_punctuation()` function instead of adding `text += '.'` in multiple locations

### 3. Testing
- Don't skip testing across multiple languages
- Don't create overly broad test files
- Don't ignore Docker container requirements

## Project Goals

### Primary Objectives
1. **Accurate Transcription**: Generate precise text from audio
2. **Proper Punctuation**: Restore correct punctuation across languages
3. **Language Learning Support**: Optimize for platforms like LingQ
4. **Ease of Use**: Simple setup and operation via Docker

### Quality Standards
- Maintain high accuracy across all supported languages
- Ensure consistent punctuation restoration
- Provide reliable, reproducible results
- Keep setup and usage simple for end users

## Debugging Guidelines

### When Creating Debug Tests
- Use descriptive file names: `test_[component]_debug.py`
- Include step-by-step trace of the processing pipeline
- Test individual functions and full pipeline
- Focus on specific problematic patterns or sentences

### Common Debug Patterns
- Test punctuation preservation after restoration
- Verify question detection logic
- Check sentence splitting behavior
- Validate multi-language support

## Checklist for AI Agents

Before submitting any changes, ensure:

- [ ] Code runs inside Docker container
- [ ] Tests pass across all supported languages
- [ ] No deprecated environment variables used
- [ ] Model caching properly configured
- [ ] Documentation updated if needed
- [ ] Fixes are general, not specific hacks
- [ ] Error handling included for edge cases
- [ ] Code follows project's architectural patterns

---

**Remember**: This project prioritizes accuracy, maintainability, and ease of use. Always consider the impact of changes across all supported languages and the overall user experience.

## Recent Refactors and Tuning Points

- Centralized thresholds and configs
  - `LanguageConfig` via `get_language_config(language)` and `_get_language_thresholds(language)` now control:
    - Spanish semantic thresholds: `semantic_question_threshold_with_indicator`, `semantic_question_threshold_default`
    - Splitting thresholds: `min_total_words_no_split`, `min_chunk_before_split`, `min_chunk_inside_question`, `min_chunk_capital_break`, `min_chunk_semantic_break`
  - Also provides per-language greetings and question starter lists for en/fr/de/es (used by non-Spanish formatter)
  - Used in `should_end_sentence_here`, `is_question_semantic`, and non-Spanish formatter to avoid inline constants.

- Centralized per-language constants in `punctuation_restorer.py`
  - Spanish: `ES_QUESTION_WORDS_CORE`, `ES_QUESTION_STARTERS_EXTRA`, `ES_GREETINGS`, `ES_CONNECTORS`, `ES_POSSESSIVES`
  - French/German: `FR_GREETINGS`, `DE_GREETINGS`, `FR_QUESTION_STARTERS`, `DE_QUESTION_STARTERS`
  - English: `EN_QUESTION_STARTERS`

- Spanish helper functions (pure, testable sub-steps)
  - `_es_greeting_and_leadin_commas`, `_es_wrap_imperative_exclamations`
  - `_es_normalize_tag_questions`, `_es_fix_collocations`
  - `_es_pair_inverted_questions`
  - `_es_merge_possessive_splits`, `_es_merge_aux_gerund`, `_es_merge_capitalized_one_word_sentences`, `_es_intro_location_appositive_commas`
  - Emphatic one‑word repeat collapsing (e.g., "No. No. No." → "No, no, no.") has been removed for maintainability

- Shared utilities
  - `_split_sentences_preserving_delims(text)` ensures consistent splitting everywhere
  - `_normalize_mixed_terminal_punctuation(text)` removes patterns like `!.`, `?.`, `!?`, compresses repeats
  - Final universal cleanup `_finalize_text_common(text)` normalizes punctuation/whitespace and spacing after sentence punctuation
  - Public sentence assembly helper: `assemble_sentences_from_processed(processed, language)`

- Public API hygiene
  - Public functions are type-annotated (e.g., `restore_punctuation`, `transformer_based_restoration`, `apply_semantic_punctuation`, `is_question_semantic`, `is_exclamation_semantic`, `format_non_spanish_text`)
  - `punctuation_restorer.py` is import-only (no `__main__` block)
  - Legacy `format_spanish_text` was removed; Spanish formatting is performed by the unified helpers in the main pipeline

- Capitalization (spaCy mode)
  - Uses `LanguageConfig` connectors/possessives for Spanish to avoid mid-sentence mis-capitalization (e.g., `tu español`)
  - SpaCy capitalization is always enabled (models included in the Docker image)
  - Mixed-language content handling: detects English phrases in Spanish transcriptions to prevent over‑capitalization while preserving true proper nouns
  - Multi-layered entity protection: combines spaCy NER, cross‑linguistic analysis, and contextual patterns
  - Conservative location capitalization to avoid false positives:
    - Only capitalize after strong cues (e.g., `vivo en`, `trabajo en`, `soy de`, `vengo de`)
    - After `de`, capitalize only when the next word was already capitalized in the input (treat as proper noun)
    - After `en`, capitalize when the next word was already capitalized; avoid capitalizing common nouns
    - Avoids over‑capitalizing common words like `semana`, `ellos`, `julio` unless they are proper nouns in context

- Tuning guidance
  - Prefer editing constants and thresholds over changing logic
  - After any change, run `tests/run_all_tests.py` inside Docker with model caches mounted
  - Avoid adding one-off hacks; extend constants or helper behavior instead

- Centralized comma spacing (2025)
  - Problem: Comma spacing logic (e.g., for number lists) was duplicated across multiple locations, causing inconsistency and harder maintenance.
  - Solution: Introduced `_normalize_comma_spacing(text)` in `punctuation_restorer.py` and replaced all inline regex variants.
  - Behavior: Removes spaces before commas, deduplicates commas, ensures a single space after commas.
  - Trade-off: Thousands separators will include a space (e.g., `1,000` → `1, 000`) to reliably fix number-list spacing; acceptable for transcript readability.

### Transcription pipeline refinements

- Chunking and merge helpers (private)
  - `_split_audio_with_overlap(media_file, chunk_length_sec, overlap_sec, chunk_dir)` to generate overlapped chunks (defaults: 480s chunks, 3s overlap)
  - `_dedupe_segments(global_segments, last_end, epsilon)` to drop overlap-duplicate segments based on end-time
  - `_accumulate_segments(local_segments, chunk_start, last_end)` to offset per-chunk timestamps to global time and build text
- Robustness and hygiene
  - Chunks are written into a `TemporaryDirectory` for automatic cleanup
  - SRT export sorts segments by start time
  - Public API: `transcribe(...)` returns a structured result (`segments`, `sentences`, `detected_language`, `output_path`, `num_segments`, `elapsed_secs`)
  - Early input/output validation via `_validate_paths`
  - Constants hoisted (e.g., `DEFAULT_CHUNK_SEC`, `DEFAULT_OVERLAP_SEC`, `DEDUPE_EPSILON_SEC`, `PROMPT_TAIL_CHARS`)
  - Prefer `pathlib.Path` over `os.path`/`glob`; keep orchestration helpers private

## Recent Refactors (Continued)

### Post-Processing Merge Consolidation (Completed - v0.5.0)

**Problem Identified (v0.4.4)**: During troubleshooting of the false domain merge bug, we discovered that sentences are split by the `SentenceSplitter` and then merged back together in multiple post-processing steps. This "split then merge" pattern created several issues:

**Current Architecture Issues**:

1. **Debugging Difficulty**: 
   - When a bug occurs in the final output, it's hard to trace whether the issue is in the splitting phase or one of the merge phases
   - Example: The "jugar. Es que vamos..." bug required adding debug logging at multiple points to track where sentences were being merged
   - No visibility into which merge operation affected which sentences

2. **Multiple Post-Processing Merge Operations** (all in `podscripter.py` lines 879-998):
   - **Spanish appositive location merge** (lines 879-884): Merges `"..., de <Proper>. <Proper> ..."` → `"..., de <Proper>, <Proper> ..."`
   - **Emphatic word merge** (lines 886-922): Merges repeated single-word emphatics like `"No. No. No."` → `"No, no, no."`
   - **Domain merge** (lines 923-975): Merges split domains like `"example." + "com"` → `"example.com"`
   - **Decimal merge** (lines 976-998): Merges split decimals like `"99." + "9%"` → `"99.9%"`

3. **Conflicting Logic**:
   - The `SentenceSplitter` correctly identifies a speaker boundary and creates a split
   - But a post-processing merge (e.g., domain merge) might undo this split
   - No coordination between splitting decisions and merging decisions
   - Speaker context is lost by the time post-processing merges run

4. **Performance**: 
   - Split then immediately merge is wasteful
   - Text is tokenized multiple times (once for splitting, then for each merge operation)

5. **Maintenance**:
   - Logic is scattered across multiple locations (`punctuation_restorer.py`, `podscripter.py`)
   - Each merge operation has its own loop and regex patterns
   - Hard to ensure all merge operations respect speaker boundaries

**Proposed Solution**: Consolidate merge logic into the `SentenceSplitter` (v0.4.0+)

The v0.4.0 refactor successfully consolidated all splitting logic into the `SentenceSplitter` class. However, post-processing merges were left in `podscripter.py` to maintain backward compatibility. Now that we have a mature `SentenceSplitter`, we should complete the consolidation.

**Approach**:

1. **Move merge patterns into `SentenceSplitter`**:
   - Add domain pattern awareness to avoid splitting domains in the first place
   - Add decimal pattern awareness to avoid splitting numbers
   - Add location appositive awareness for Spanish
   - Add emphatic word detection for merge decisions

2. **Add "should_merge_sentences()" method**:
   ```python
   def should_merge_sentences(self, sentence1: str, sentence2: str, 
                             speaker1: str | None, speaker2: str | None) -> bool:
       """
       Determine if two sentences should be merged.
       
       CRITICAL: Never merge if speakers differ.
       Then check for:
       - Domain patterns (example. + com)
       - Decimal patterns (99. + 9%)
       - Location appositives (ES: ", de <Proper>. <Proper>")
       - Emphatic word repeats (No. No. No.)
       """
       # Rule 0: NEVER merge different speakers
       if speaker1 and speaker2 and speaker1 != speaker2:
           return False
       
       # Rule 1: Check for domain patterns
       if self._is_domain_split(sentence1, sentence2):
           return self._should_merge_domain(sentence1, sentence2)
       
       # Rule 2: Check for decimal patterns
       if self._is_decimal_split(sentence1, sentence2):
           return True
       
       # Rule 3: Check for location appositives (Spanish)
       if self.language == 'es' and self._is_location_appositive(sentence1, sentence2):
           return True
       
       # Rule 4: Check for emphatic repeats
       if self._is_emphatic_repeat(sentence1, sentence2):
           return True
       
       return False
   ```

3. **Add merge provenance tracking** (similar to split provenance):
   - Track WHY sentences were merged
   - Include in debug output and metadata
   - Helps troubleshooting and understanding decisions

4. **Benefits**:
   - ✅ All split/merge decisions in one place (`sentence_splitter.py`)
   - ✅ Speaker context available for all merge decisions
   - ✅ Easier debugging with single decision point
   - ✅ Better performance (one pass instead of multiple)
   - ✅ Merge provenance tracking for transparency
   - ✅ Easier to add new merge patterns (just extend `should_merge_sentences()`)
   - ✅ Consistent with v0.4.0 architecture vision

**Implementation Plan**:

**Phase 1: Add merge detection to SentenceSplitter**
1. Add helper methods for pattern detection:
   - `_is_domain_split(s1, s2)` - detect domain patterns
   - `_is_decimal_split(s1, s2)` - detect decimal patterns
   - `_is_location_appositive(s1, s2)` - detect location patterns (ES)
   - `_is_emphatic_repeat(s1, s2)` - detect emphatic word repeats
2. Add `should_merge_sentences()` method with speaker-aware logic
3. Add tests for merge detection

**Phase 2: Integrate merge logic into splitting**
1. After splitting, run merge pass within `SentenceSplitter`
2. Track merge provenance similar to split provenance
3. Remove post-processing merge loops from `podscripter.py`
4. Add comprehensive tests for split-then-merge scenarios
5. Verify all existing tests still pass

**Phase 3: Add merge provenance and debugging**
1. Track which sentences were merged and why
2. Add to debug output and metadata
3. Update `--debug` flag output to show merge decisions
4. Document new architecture in `ARCHITECTURE.md`

**Breaking Changes**: Minimal
- API remains the same (input/output unchanged)
- Merge logic moves location but behavior stays identical
- Metadata structure enhanced (additive change)

**Backward Compatibility**: High
- All existing tests should pass
- Output should be identical (just produced differently)
- If discrepancies found, they're likely bug fixes

**Priority**: Medium-High
- Not urgent (current system works)
- But significantly improves maintainability
- Makes future debugging much easier
- Completes the v0.4.0 consolidation vision

**Status**: COMPLETED (v0.5.0)

**Results Achieved**:
- ✅ All merge logic consolidated into single `SentenceFormatter` class in `sentence_formatter.py`
- ✅ Speaker boundary checks enforce "never merge different speakers" across ALL merge types
- ✅ Merge provenance tracking implemented with `MergeMetadata` dataclass
- ✅ Natural language guards prevent false domain merges (e.g., "jugar. Es que..." stays separate)
- ✅ Comprehensive unit tests created (`tests/test_sentence_formatter.py` - 15/15 passing)
- ✅ `--dump-merge-metadata` CLI flag for debugging merge decisions
- ✅ 100% backward compatible for non-diarization mode
- ✅ Clear separation: `SentenceSplitter` = boundaries, `SentenceFormatter` = formatting

**Immediate Benefit Delivered**: Solved cross-speaker merge bugs. Different speakers' sentences are never merged, even when patterns match (e.g., domain, decimal, emphatic patterns).

### Speaker Segment Splitting Fix (v0.5.2 & v0.5.2.1 - 2025-01-05)

**Problem Identified**: Despite the v0.5.0 `SentenceFormatter` correctly preventing cross-speaker merges, sentences from different speakers were still being incorrectly combined. Debug investigation revealed that the root cause was BEFORE the formatter—sentences were already wrong when entering the formatting phase.

**Root Cause (Bug #1 - v0.5.2)**: The `_convert_speaker_segments_to_char_ranges()` function (lines 703-783) was assigning each Whisper segment to ONE speaker based on "most time overlap." When a Whisper segment contained text from multiple speakers, it was assigned entirely to the majority speaker, **losing the boundary** between them.

**Example**:
- Whisper segment: "Aquí. Listo, eso es todo..." (178.38-185.00s)
- Speaker segments: 
  - Andrea: 178.50-179.43s (0.93s, says "Aquí")
  - Nate: 179.97-185.00s (5.03s, says "Listo, eso es todo...")
- **Old behavior**: Entire Whisper segment assigned to Nate (most overlap) → boundary lost
- **Impact**: 84 speaker boundaries detected by diarization → only 48 preserved for sentence splitting (36 boundaries lost!)

**Solution Implemented (v0.5.2)**:
Completely rewrote `_convert_speaker_segments_to_char_ranges()` (lines 703-835) to **split Whisper segments** when they contain multiple speakers:

1. **Detect ALL overlapping speakers** per Whisper segment (not just the one with most overlap)
2. **Split segments proportionally** based on time overlaps when multiple speakers detected
3. **Word boundary splitting**: Attempts to split at spaces (within ±20% window) for cleaner results
4. **Merge consecutive ranges**: Consolidates adjacent ranges from the same speaker for efficiency

**Code pattern**:
```python
for whisp_segment in whisper_segments:
    overlapping_speakers = find_all_overlaps(whisp_segment, speaker_segments)
    
    if len(overlapping_speakers) == 1:
        # Simple case: entire segment is one speaker
        create_range(whisp_segment, speaker)
    else:
        # Complex case: SPLIT the segment
        for speaker in overlapping_speakers:
            time_ratio = speaker_overlap_duration / whisp_duration
            char_end = calculate_proportional_split(time_ratio)
            char_end = find_nearest_word_boundary(char_end)  # Try to split at space
            create_range(char_start, char_end, speaker)
```

**Impact**: All 84 speaker boundaries now preserved (84 → 84 instead of 84 → 48)

---

**Root Cause (Bug #2 - v0.5.2.1)**: The v0.5.2 fix introduced a subtle bug in the overlap filtering logic. Line 729 checked `if spk_duration < 0.5: continue` which filtered out speaker segments shorter than 0.5s **regardless of overlap duration**.

**Example**:
- Whisper segment 35: "Bueno." (179.38-180.38s, 1.0s duration)
- Speaker segment: SPEAKER_00 at 179.97-180.42s (0.46s total duration, 0.41s overlap with segment 35)
- **Bug**: 0.46s < 0.5s → filtered out, even though 0.41s overlap is substantial
- **Impact**: "Está bien." (SPEAKER_01) and "Bueno!" (SPEAKER_00) incorrectly merged into same paragraph

**Solution Implemented (v0.5.2.1)**:
Changed filter to check **overlap duration** instead of total segment duration:
- **Before**: `if spk_duration < 0.5: continue`
- **After**: `if overlap_duration < 0.3: continue`

**Rationale**: A 0.46s speaker segment with 0.41s overlap is valid speech, not noise. The filter should check the actual overlap with the Whisper segment, not the total speaker segment duration from diarization.

**Impact**: Short but legitimate utterances like "Bueno!" (0.46s) are now correctly preserved as separate speaker segments.

---

**Root Cause (Bug #3 - v0.5.2.2)**: Pyannote diarization occasionally misattributes small portions at segment edges to the wrong speaker, and v0.5.2's faithful segment splitting propagated these errors.

**Example**:
- Whisper segment 11: "Y yo soy Nate de Texas, Estados Unidos." (44.18-48.18s, 4.0s duration)
- Pyannote diarization:
  - 44.18-44.73s (0.55s, 14%): SPEAKER_02 (Andrea) - **incorrect!**
  - 44.73-48.18s (3.45s, 86%): SPEAKER_01 (Nate) - correct
- **Bug**: v0.5.2 split "Y yo" (14%) to Andrea, "soy Nate..." (86%) to Nate
- **Impact**: "Yo soy Andrea de Santander, Colombia y yo." instead of two separate sentences

**Solution Implemented (v0.5.2.2)**:
Added **dominant speaker threshold** - if one speaker accounts for >80% of a Whisper segment, assign the entire segment to them.

**Rationale**: Small misattributions at edges (<20%) are more likely diarization errors than actual speaker changes. Pyannote can confuse speakers during overlaps, transitions, or similar voices.

**Impact**: 
- Segment with 14% SPEAKER_02 + 86% SPEAKER_01 → assigned entirely to SPEAKER_01
- Correct output: "Yo soy Andrea de Santander, Colombia." / "Y yo soy Nate de Texas, Estados Unidos."

**Issue with v0.5.2.2**: Too aggressive - filtered out 16 legitimate speaker boundaries (84 → 68), including short utterances like "Ok." in the middle of segments.

---

**Root Cause (Bug #3b - v0.5.2.3)**: The 80% dominant speaker threshold couldn't distinguish between **edge misattributions** and **legitimate middle utterances**.

**Example**:
- Segment: "Listo, eso es todo... en espanolistos.com slash best. Ok. Esto fue todo..."
- Pyannote correctly identifies "Ok." as different speaker in the MIDDLE
- v0.5.2.2 incorrectly filtered it due to 80% threshold → merged "best. Ok." together

**Solution Implemented (v0.5.2.3)**:
Refined dominant speaker logic to only apply at **edges** (first/last 10% of segment):

```python
if dominant_ratio > 0.8 and len(overlapping_speakers) == 2:
    minor_speaker = find_minor_speaker()
    edge_threshold = 0.1 * segment_duration  # 10% of segment
    
    # Check if minor speaker is at beginning or end (not middle)
    at_beginning = abs(minor_start - segment_start) < edge_threshold
    at_end = abs(minor_end - segment_end) < edge_threshold
    is_edge_only = at_beginning or at_end
    
    if is_edge_only:
        # Edge misattribution - assign entire segment to dominant speaker
    else:
        # Middle utterance - split proportionally (preserve boundary)
```

**Impact**: 
- Edge misattributions: filtered (e.g., "Y yo" at START of segment)
- Middle utterances: preserved (e.g., "Ok." in MIDDLE of segment)
- All legitimate speaker boundaries preserved while filtering only edge errors

---

**Key Lessons Learned**:

1. **Text Normalization Alignment**: Character positions MUST be calculated from the same normalized text that downstream processing uses. Applied `_normalize_initials_and_acronyms()` and whitespace normalization to Whisper segment text in `_convert_speaker_segments_to_char_ranges()` before calculating positions (v0.5.1).

2. **Word Position Tracking**: `SentenceFormatter` operates on word indices, so sentence positions must align with the same word count. Moved `SentenceFormatter.format()` to run BEFORE `_sanitize_sentence_output()` to prevent word count misalignment (v0.5.1).

3. **Segment Assignment vs. Splitting**: Assigning entire segments to ONE speaker (even with "best overlap") loses boundaries. Must detect multi-speaker segments and split them proportionally (v0.5.2).

4. **Filter What Matters**: When filtering based on duration, check the relevant duration. For overlap-based logic, filter on overlap duration, not total segment duration (v0.5.2.1).

5. **Trust the Dominant Signal (with context)**: ML models like pyannote can make errors at edges/transitions. When one signal dominates (>80%), trust it over minor conflicting signals - BUT only for edge cases. Middle utterances are usually legitimate even if short (v0.5.2.2, refined in v0.5.2.3).

6. **Edge vs. Middle distinction**: Filtering/thresholding logic should consider **position** not just magnitude. Edge anomalies are likely errors; middle anomalies are likely real signals (v0.5.2.3).

7. **Debug Methodology**: 
   - Check boundary preservation at each pipeline stage (diarization → char ranges → word ranges → sentence splits)
   - Use targeted audio clips (e.g., `Episodio213-trim.mp3`) for faster iteration during debugging
   - Add temporary debug logging to track data transformations
   - Verify fixes with both short test files and full production files

**Tests**: 
- Verified with `Episodio213-trim.mp3` (short test file, 3 minutes)
- Verified with `Episodio213.mp3` (full file, 30+ minutes, 84 speaker changes)
- All four problematic examples now correctly split:
  1. "Aquí." / "Listo, eso es todo..." (v0.5.2)
  2. "Está bien." / "Bueno!" (v0.5.2.1)
  3. "Yo soy Andrea de Santander, Colombia." / "Y yo soy Nate de Texas, Estados Unidos." (v0.5.2.2)
  4. "...en espanolistos.com slash best." / "Ok." (v0.5.2.3)

**Files Modified**:
- `podscripter.py`: 
  - Rewrote `_convert_speaker_segments_to_char_ranges()` (v0.5.2)
  - Fixed overlap filter (v0.5.2.1)
  - Added dominant speaker threshold (v0.5.2.2)
  - Refined to edge-only threshold (v0.5.2.3)
- `CHANGELOG.md`: Documented v0.5.2, v0.5.2.1, v0.5.2.2, and v0.5.2.3
- `AGENT.md`: Updated Speaker Segment Splitting Fix section with all iterations
- `SPEAKER_BOUNDARY_BUG_INVESTIGATION.md`: Complete investigation documentation

### Whisper Segment Boundary Integration (2025)

Podscripter now uses a 4-signal hybrid approach for sentence breaking (language‑agnostic across EN/ES/FR/DE):

- **1. Grammatical guards**: Avoid ending sentences on coordinating conjunctions, prepositions, and continuative/auxiliary verbs.
- **2. Semantic coherence**: Use Sentence-Transformers similarity to confirm low-coherence boundaries.
- **3. Configurable thresholds**: Minimum chunk length, overall length, and capital/semantic break limits.
- **4. Whisper segment boundaries (new)**: Treat Whisper `all_segments` boundaries as prioritized hints for sentence breaks.

Rules for Whisper boundaries:
- Boundaries are prioritized hints, still gated by grammatical guards and minimum chunk size.
- Boundaries are ignored at grammatically invalid positions (e.g., after conjunctions/prepositions/continuative verbs).
- Backward compatible: when boundaries are absent, behavior remains unchanged.

Threading through the pipeline:
- Orchestrator passes `all_segments` to `_assemble_sentences(...)`.
- `_assemble_sentences` extracts character boundary positions and calls `restore_punctuation(text, language, whisper_boundaries=...)`.
- Punctuation converts character boundaries to word indices and uses them in `_semantic_split_into_sentences(..., whisper_word_boundaries=...)` and `_should_end_sentence_here(...)`.

Thresholds (set via `_get_language_thresholds(language)`):
- `min_words_whisper_break` (default 10): minimum words in the current chunk before honoring a Whisper boundary.
- `max_words_force_split` (default 100): safety for very long run-ons (reserved for future tuning/usage).

Tests:
- `tests/test_whisper_boundary_integration.py` validates boundary extraction, grammatical gating, and multi-language behavior.

### Speaker Diarization Integration (2025)

Podscripter includes optional speaker diarization to improve sentence boundaries at speaker changes.

**Purpose:**
- Detects when speakers change in audio
- Provides high-priority hints for sentence boundaries
- Especially useful for multi-speaker content (interviews, conversations, podcasts with guests)
- Speaker changes naturally align with sentence breaks in dialogue

**Implementation Guidelines:**
- **Opt-in feature** (disabled by default to avoid dependency bloat)
- Uses pyannote.audio 3.3.2 with Hugging Face model caching
- Speaker boundaries passed SEPARATELY to `restore_punctuation()` (not merged with Whisper boundaries)
- Speaker timestamps converted to character positions via `_convert_speaker_timestamps_to_char_positions()`
- Priority: Speaker boundaries > Whisper boundaries > Semantic coherence
- Still respects grammatical guards (no breaks on prepositions/conjunctions/auxiliary verbs)
- **Critical**: Speaker boundary checks happen BEFORE general `min_chunk_before_split` threshold to allow breaking on short phrases

**Minimum word thresholds** (in `_should_end_sentence_here`):
- Speaker boundaries: **2 words** (very low since speaker changes are definitive signals)
- Whisper boundaries: **10 words** (acoustic pause hints)
- General semantic splitting: **20 words for Spanish, 15 for other languages** (only if no boundary hint)

**Module structure:**
- `speaker_diarization.py`: Separate module following project's modularity pattern
- `diarize_audio(...)`: Main entry point, returns `DiarizationResult`
- `_extract_speaker_boundaries(...)`: Extracts timestamps where speakers change; returns both filtered boundaries and detailed `BoundaryInfo` for debugging
- `_convert_speaker_timestamps_to_char_positions(...)`: Converts speaker timestamps (seconds) to character positions in text (in `podscripter.py`)
- `write_diarization_dump(...)`: Writes comprehensive debug dump file (raw segments, boundary analysis, merge details)
- `DiarizationError`: Typed exception for error handling
- `BoundaryInfo`: TypedDict with detailed info about each potential boundary (timestamp, speakers, duration, included/filtered status, reason)

**CLI Flags:**
- `--enable-diarization`: Enable speaker diarization (default: disabled)
- `--min-speakers <int>`: Minimum number of speakers (optional, auto-detect by default)
- `--max-speakers <int>`: Maximum number of speakers (optional, auto-detect by default)
- `--hf-token <str>`: Hugging Face token (required for first download)
- `--dump-diarization`: Write diarization debug dump to `<basename>_diarization.txt` (requires `--enable-diarization`)

**Environment Variables:**
- `HF_TOKEN`: Alternative to `--hf-token` flag
- Precedence: CLI flag > environment variable

**Model Caching:**
- Cache directory: `models/pyannote/` → `/root/.cache/pyannote`
- First run requires HF token (same pattern as sentence-transformers)
- Subsequent runs use cached models
- Mount in Docker: `-v $(pwd)/models/pyannote:/root/.cache/pyannote`

**Testing:**
- Unit tests: `tests/test_speaker_diarization_unit.py` (boundary extraction, merging, deduplication)
- Conversion tests: `tests/test_speaker_boundary_conversion.py` (timestamp-to-char-position conversion, realistic scenarios)
- Integration tests: `tests/test_speaker_diarization_integration.py` (realistic scenarios, edge cases)
- Controlled by `RUN_DIARIZATION=1` environment flag in test suite
- Tests do not require actual audio or diarization models (mock data)

**Error Handling:**
- `DiarizationError` raised for diarization failures
- Gracefully degrades: logs warning and continues without speaker boundaries (no hard exit)
- Tip message suggests providing `--hf-token` or `HF_TOKEN` for first-time use

**Performance:**
- Adds ~10-30% overhead depending on audio length
- Runs on CPU by default (same device as Whisper)
- Can use GPU if available (`device="cuda"`)
- Model caching makes subsequent runs much faster

---

## Future Refactoring Opportunities

### ~~Speaker-Aware Output Formatting~~ (✅ COMPLETED in v0.6.0)

**Problem Identified (v0.5.2.3)**: During speaker diarization debugging, we discovered that while speaker boundaries are correctly detected and preserved through the pipeline (84 boundaries → 74 preserved through filtering → 71 final output), some short utterances from different speakers still appear on the same output line (paragraph) in TXT files.

**Solution Implemented (v0.6.0)**: Refactored sentence objects to preserve internal speaker structure throughout the pipeline:

**Example**: 
- Input: `"...en espanolistos.com slash best. Ok. Esto fue todo..."`
- Speakers: "best." (Nate) + "Ok." (Andrea) + "Esto fue todo..." (Andrea)
- Current output (line 479): `"...en espanolistos.com slash best. Ok."` (both speakers combined on one line)
- Desired behavior: Line 1: `"...en espanolistos.com slash best."` / Line 2: `"Ok. Esto fue todo..."`

**Current State**:
- ✅ Speaker boundaries correctly detected at all pipeline stages (diarization → char ranges → word ranges → sentence splits)
- ✅ Word-level speaker tracking works (debug shows `✓ SPLIT at speaker boundary: word 3861 'Ok,'`)
- ✅ Main use cases work correctly (e.g., "Yo soy Andrea..." / "Y yo soy Nate..." properly separated)
- ❌ Output formatting sometimes groups consecutive utterances from different speakers on same line
- **Success rate**: ~88% (74 out of 84 boundaries preserved in final output)

**Why It's Not a Critical Bug**:
- The system is working as architecturally designed
- Speaker boundaries ARE preserved for sentence splitting decisions
- Main dialogue transitions work correctly (e.g., speaker introductions)
- The grouping happens at the output formatting stage for readability
- Edge cases typically involve very short utterances (1-2 words like "Ok.", "Sí.")

**Root Cause (Architectural)**:
The issue is that "sentence" objects flow through the pipeline as simple strings, losing their internal speaker structure:

1. `SentenceSplitter` correctly identifies speaker boundaries and creates split points
2. Sentence assembly joins with spaces for readability (`' '.join(processed)`)
3. `_write_txt()` treats each sentence object as a single paragraph
4. By the time we reach output writing, we've lost information about which periods represent speaker changes vs. natural sentence endings

**Non-Conforming Solutions** (rejected per AGENT.md guidelines):

❌ **Option 1**: Accept current 88% success rate
- Most cases work correctly
- Edge cases are minor (short utterances)
- No code changes needed

❌ **Option 2**: Add post-processing in `_write_txt()` to split at periods with speaker changes
- Violates "implement general solutions not specific hacks" guideline
- Treats symptom (output formatting) rather than root cause (architecture)
- Adds workaround at output stage instead of fixing sentence assembly
- Would need to re-lookup speaker information that was already known earlier in pipeline

**Implementation (v0.6.0)**: ✅ COMPLETED

**Phase 1: Enhanced Sentence Objects** (✅ Completed)
- Added `Utterance` dataclass in `sentence_splitter.py` (lines 54-59)
- Added `Sentence` dataclass in `sentence_splitter.py` (lines 62-84)
- Implemented `has_speaker_changes()` and `get_first_speaker()` methods

**Phase 2: Speaker-Aware Assembly** (✅ Completed)
- Updated `SentenceSplitter.split()` to return `List[Sentence]` instead of `List[str]`
- Implemented `_detect_speaker_changes_in_sentence()` method to populate utterances
- Updated sentence assembly loop to create `Sentence` objects with utterances
- Updated `SentenceFormatter` to preserve utterances when merging sentences

**Phase 3: Speaker-Aware Output Writing** (✅ Completed)
- Updated `_write_txt()` to detect speaker changes between sentences
- Adds extra paragraph break (`\n\n\n`) when speaker changes
- Maintains backward compatibility (no extra breaks when speaker info unavailable)
- Created comprehensive test suite in `test_speaker_aware_output.py`

**Results**:
- ✅ All 5 reported speaker boundary bugs fixed
- ✅ Visual separation of different speakers in output
- ✅ Same-speaker sentences remain grouped
- ✅ Backward compatible with non-diarization mode
- ✅ Speaker information preserved throughout entire pipeline
- ✅ No architectural workarounds or hacks

**Actual Effort**: ~20 hours (as estimated)
- Phase 1: Dataclasses and interfaces (~2 hours)
- Phase 2: SentenceSplitter and SentenceFormatter updates (~10 hours)
- Phase 3: Output writers and punctuation_restorer updates (~5 hours)
- Testing and documentation (~3 hours)

**Status**: ✅ Completed in v0.6.0
