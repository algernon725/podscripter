# AI Agent Guidelines for PodScripter

## Project Overview

**PodScripter** is a multilingual audio transcription tool that generates accurate, punctuated transcriptions for language learning platforms like LingQ. The project uses Docker containerization and leverages state-of-the-art NLP models for punctuation restoration.

### Core Technologies
- **Whisper**: OpenAI's speech-to-text model for transcription
- **Sentence-Transformers**: For semantic understanding and punctuation restoration
- **HuggingFace Transformers**: NLP model integration
- **spaCy (optional)**: Lightweight NLP capitalization and entity awareness (models: `en_core_web_sm`, `es_core_news_sm`, `fr_core_news_sm`, `de_core_news_sm`)
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
  - `-v $(pwd)/models/whisper:/app/models`
  - `-v $(pwd)/models/sentence-transformers:/root/.cache/torch/sentence_transformers`
  - `-v $(pwd)/models/huggingface:/root/.cache/huggingface`
  - `-v $(pwd)/audio-files:/app/audio-files`

### 2. Model Caching Strategy
- Whisper models cached in `/app/models`
- Sentence-Transformers cached in `/root/.cache/torch/sentence_transformers`
- HuggingFace models cached in `/root/.cache/huggingface`
- Use `HF_HOME` environment variable (avoid deprecated `TRANSFORMERS_CACHE`)
- Prefer offline use when cache exists: set `HF_HUB_OFFLINE=1` for tests/runs to avoid 429 rate limits
- Use a singleton model loader to avoid repeated model instantiation within a process
- For spaCy capitalization mode, models are baked into the image; enable/disable via `NLP_CAPITALIZATION` (see Docker Best Practices)
 - Sentence-Transformers loader: only load from a direct cache path if `modules.json` or `config_sentence_transformers.json` exists in that folder; otherwise load by name with `cache_folder` to avoid the "Creating a new one with mean pooling" message while still using caches.

### 2a. Transcription Orchestration (Whisper usage)
- Two supported modes:
  - Single-call transcription (recommended when resources allow): process full file in one call so Whisper maintains context. Enabled via `--single` flag in `transcribe_sentences.py`.
  - Overlapped-chunk transcription (fallback): default path with 480s chunks and 3s overlap.
- Common settings:
  - `vad_filter=True` for both modes; default `speech_pad_ms=200` (constant). No VAD CLI flags in the main program.
  - `condition_on_previous_text=True` to keep context continuity
  - For chunked mode, pass `initial_prompt` using the last ~200 characters of accumulated text
  - Deduplicate overlap during merge using global timestamps (skip segments that finish before the prior chunk’s end)
  - Keep `beam_size` modest (1–3) for long files; `compute_type` default is `auto` on CPU
  - `PROMPT_TAIL_CHARS=200` controls how much trailing text is used for `initial_prompt`

### 3. Modular Processing Pipeline
```
Audio Input → Whisper Transcription → Punctuation Restoration → Sentence Splitting → Output

- CLI flags (argparse in `transcribe_sentences.py`):
  - `--output_dir <dir>` (required)
  - `--language <code>|auto` (default `auto`)
  - `--output_format {txt|srt}` (default `txt`)
  - `--single` (bypass manual chunking)
  - `--compute-type {auto,int8,int8_float16,int8_float32,float16,float32}` (default `auto`)
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
- Transcription orchestration in `transcribe_sentences.py`
- Tests in `tests/` directory with descriptive names
- Documentation in markdown files

### Error Handling
- Use try-catch blocks for model loading and API calls
- Provide meaningful error messages
- Handle rate limiting gracefully (especially for HuggingFace API)

## Project-Specific Rules

### 1. Punctuation Restoration Guidelines

#### General Approach
- Use `re.sub()` for text cleanup (preferred over other regex methods)
- Apply fixes broadly rather than one-off hacks for individual sentences
- Accept less than 100% accuracy for difficult edge cases
- Focus on semantic understanding over rule-based patterns
- Do not emit optional debug output for punctuated text

#### Formatting Responsibilities
- Perform capitalization, comma insertion, hyphenation, and all punctuation fixes in `punctuation_restorer.py`
- Do not implement formatting logic in `transcribe_sentences.py`

#### Language-Specific Heuristics (recent)
- Spanish:
  - Preserve embedded questions mid-sentence: keep and properly pair `¿ … ?` inside larger sentences; do not strip mid-sentence `¿`
  - Coordinated yes/no questions: detect verb-initial starts with later `o <verbo>` (e.g., "¿Quieren … o prefieren …?")
  - Greeting/lead-in formatting: add comma after greeting phrases ("Hola …,") and set-phrases ("Como siempre,")
  - Diacritic-insensitive detection for gating where appropriate (e.g., `como` ~ `cómo`)
  - Appositive introductions: format as "Yo soy <Nombre>, de <Ciudad>, <País>"
  - Soft sentence splitting: avoid breaking inside entities; merge `auxiliar + gerundio` and possessive splits ("tu español")
  - Optional spaCy capitalization: capitalize entities/PROPN; keep connectors (de, del, y, en, a, con, por, para, etc.) lowercase
- French: apply clitic hyphenation for inversion (e.g., `allez-vous`, `est-ce que`, `qu'est-ce que`, `y a-t-il`, `va-t-il`)
- German: insert commas before common subordinating conjunctions (`dass|weil|ob|wenn`) when safe; expand question starters/modals; capitalize `Ich` after punctuation; capitalize `Herr/Frau + Name`; minimal noun capitalization after determiners; maintain a small whitelist of proper nouns
- English/French/German: add greeting commas (`Hello, ...`, `Bonjour, ...`, `Hallo, ...`) and capitalize sentence starts

### 2. Testing Requirements
- All tests must run inside Docker container
- Create focused test files for specific bugs/issues
- Use descriptive test names that explain the scenario
- Test both individual functions and full transcription pipeline
- Ensure model caches are mounted for reliable, fast tests and to avoid 429s
- Spanish-only tests to be included by default:
  - `test_spanish_embedded_questions.py` (embedded `¿ … ?` clauses)
  - `test_human_vs_program_intro.py` (human-vs-program similarity on intro + extended lines; token-level F1 thresholds)
    - Intro average F1 threshold: ≥ 0.80
    - Overall average F1 threshold: ≥ 0.70
  - `test_spanish_helpers.py` (unit tests for `_es_*` helpers: tags, collocations, merges, pairing, greetings)
  - Run selection controlled by env flags in `tests/run_all_tests.py`: `RUN_ALL`, `RUN_MULTILINGUAL`, `RUN_TRANSCRIPTION`, `RUN_DEBUG`
 - The ad-hoc script `tests/test_transcription.py` is for manual experiments:
   - Defaults: model `medium`, device `cpu`, compute type `auto`
   - Toggles: `--single`, `--chunk-length N`, `--apply-restoration`, `--dump-raw` (writes raw Whisper output for debugging)
   - It also exposes VAD toggles (`--no-vad`, `--vad-speech-pad-ms`) strictly for debugging; the main CLI uses constants

### 3. Docker Best Practices
- Use `--platform linux/arm64` for M-series Mac compatibility
- Mount volumes for model caching: `-v $(pwd)/models/whisper:/app/models`
- Include all necessary environment variables in Dockerfile
- Avoid deprecated environment variables (e.g., `TRANSFORMERS_CACHE`)
- NLP capitalization mode: Dockerfile sets `ENV NLP_CAPITALIZATION=1` (on by default). Override per run with `-e NLP_CAPITALIZATION=0` to disable.
- For performance on long files (> 1 hour): prefer single-call mode if resources allow; otherwise use overlapped chunking with 3s overlap and deduplication on merge.

### 4. Model Management
- Cache models to avoid repeated downloads
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

- Shared utilities
  - `_split_sentences_preserving_delims(text)` ensures consistent splitting everywhere
  - `_normalize_mixed_terminal_punctuation(text)` removes patterns like `!.`, `?.`, `!?`, compresses repeats
  - Final universal cleanup `_finalize_text_common(text)` normalizes punctuation/whitespace and spacing after sentence punctuation

- Public API hygiene
  - Public functions are type-annotated (e.g., `restore_punctuation`, `transformer_based_restoration`, `apply_semantic_punctuation`, `is_question_semantic`, `is_exclamation_semantic`, `format_non_spanish_text`)
  - `punctuation_restorer.py` is import-only (no `__main__` block)
  - Legacy `format_spanish_text` was removed; Spanish formatting is performed by the unified helpers in the main pipeline

- Capitalization (spaCy mode)
  - Uses `LanguageConfig` connectors/possessives for Spanish to avoid mid-sentence mis-capitalization (e.g., `tu español`)
  - Keep `NLP_CAPITALIZATION` opt-in/out via env (`1` default in Dockerfile)

- Tuning guidance
  - Prefer editing constants and thresholds over changing logic
  - After any change, run `tests/run_all_tests.py` inside Docker with model caches mounted
  - Avoid adding one-off hacks; extend constants or helper behavior instead

### Transcription pipeline refinements

- Chunking and merge helpers (private)
  - `_split_audio_with_overlap(media_file, chunk_length_sec, overlap_sec, chunk_dir)` to generate overlapped chunks (defaults: 480s chunks, 3s overlap)
  - `_dedupe_segments(global_segments, last_end, epsilon)` to drop overlap-duplicate segments based on end-time
  - `_accumulate_segments(local_segments, chunk_start, last_end)` to offset per-chunk timestamps to global time and build text
- Robustness and hygiene
  - Chunks are written into a `TemporaryDirectory` for automatic cleanup
  - SRT export sorts segments by start time
  - `transcribe_with_sentences(...)` returns a structured dict: `segments`, `sentences`, `detected_language`, `output_path`, `num_segments`, `elapsed_secs`
  - Early input/output validation via `_validate_paths`
  - Constants hoisted (e.g., `DEFAULT_CHUNK_SEC`, `DEFAULT_OVERLAP_SEC`, `DEDUPE_EPSILON_SEC`, `PROMPT_TAIL_CHARS`)
  - Prefer `pathlib.Path` over `os.path`/`glob`; keep orchestration helpers private
