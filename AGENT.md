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
- Use `HF_HOME` environment variable (avoid deprecated `TRANSFORMERS_CACHE`)
- Prefer offline use when cache exists: set `HF_HUB_OFFLINE=1` for tests/runs to avoid 429 rate limits
- Use a singleton model loader to avoid repeated model instantiation within a process
- SpaCy capitalization is always enabled, with models baked into the image (see Docker Best Practices)
- Sentence-Transformers loader: only load from a direct cache path if `modules.json` or `config_sentence_transformers.json` exists in that folder; otherwise load by name with `cache_folder` to avoid the "Creating a new one with mean pooling" message while still using caches. Also sets `HF_HOME` and may set `HF_HUB_OFFLINE=1` when a local model directory is used.

### 2a. Transcription Orchestration (Whisper usage)
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
  - `--quiet`/`--verbose` (default `--verbose`)
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
  - Exit codes: 2=input, 3=model load, 4=transcription, 5=write, 6=diarization, 1=unexpected

### Logging
- Use a single logger named `podscripter` configured in `podscripter.py`
- Levels are controlled by CLI flags:
  - `--quiet` → ERROR
  - default (no flag) → INFO
  - `--verbose` → INFO plus selective detail; avoid debug noise in punctuation helpers
- Replace ad-hoc prints with `logger.info/warning/error`
- Keep `punctuation_restorer.py` free of optional debug output
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

### 5. Known Open Issues / Work In Progress

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
- Uses pyannote.audio 3.1.1 with Hugging Face model caching
- Speaker boundaries merged with Whisper boundaries via `_merge_boundaries(...)`
- Priority: Speaker boundaries > Whisper boundaries > Semantic coherence
- Still respects grammatical guards (no breaks on prepositions/conjunctions/auxiliary verbs)
- Boundaries within 1.0s are deduplicated to avoid redundancy

**Module structure:**
- `speaker_diarization.py`: Separate module following project's modularity pattern
- `diarize_audio(...)`: Main entry point, returns `DiarizationResult`
- `_extract_speaker_boundaries(...)`: Extracts timestamps where speakers change
- `_merge_boundaries(...)`: Merges and deduplicates Whisper + speaker boundaries
- `DiarizationError`: Typed exception for error handling

**CLI Flags:**
- `--enable-diarization`: Enable speaker diarization (default: disabled)
- `--min-speakers <int>`: Minimum number of speakers (optional, auto-detect by default)
- `--max-speakers <int>`: Maximum number of speakers (optional, auto-detect by default)
- `--hf-token <str>`: Hugging Face token (required for first download)

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
- Integration tests: `tests/test_speaker_diarization_integration.py` (realistic scenarios, edge cases)
- Controlled by `RUN_DIARIZATION=1` environment flag in test suite
- Tests do not require actual audio or diarization models (mock data)

**Error Handling:**
- `DiarizationError` raised for diarization failures
- Gracefully degrades: logs warning and continues without speaker boundaries
- Exit code 6 for diarization errors in CLI (new code added to exit code list)

**Performance:**
- Adds ~10-30% overhead depending on audio length
- Runs on CPU by default (same device as Whisper)
- Can use GPU if available (`device="cuda"`)
- Model caching makes subsequent runs much faster
