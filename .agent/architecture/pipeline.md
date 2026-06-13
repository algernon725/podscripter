# Pipeline Architecture

Processing pipeline, model caching, post-processing formatting, transcription orchestration, and speaker diarization for podscripter.

Related: [CLI spec](cli-spec.md) | [Troubleshooting history](../troubleshooting/history.md) | [AGENT.md hub](../../AGENT.md) | [ARCHITECTURE.md](../../ARCHITECTURE.md)

## Data flow

```
Audio Input
  -> Chunking (overlap)            # or single-call when --single
  -> Whisper Transcription         # language detection, optional VAD
  -> Dedup/Globalize Segments
  -> Punctuation Restoration
  -> Sentence Splitting            # SentenceSplitter
  -> Post-processing Merge         # SentenceFormatter
  -> Output (TXT/SRT)
```

Core files:
- `punctuation_restorer.py` — punctuation, language-specific formatting, capitalization, comma insertion, hyphenation, sentence-assembly utilities.
- `podscripter.py` — orchestration only (I/O, model selection, mode, calling helpers, output writing).
- `sentence_splitter.py` — all sentence-boundary decisions (`SentenceSplitter`).
- `sentence_formatter.py` — all post-processing merges (`SentenceFormatter`).
- `domain_utils.py` — domain detection and masking utilities.
- `speaker_diarization.py` — optional speaker diarization.

## Model caching strategy

- Use `HF_HOME` for Hugging Face caches; avoid deprecated `TRANSFORMERS_CACHE`.
- Prefer offline use when cache exists: set `HF_HUB_OFFLINE=1` for tests/runs to avoid 429 rate limits.
- Cache locations:
  - Faster-Whisper models: Hugging Face Hub under `/root/.cache/huggingface` (mounted from `models/huggingface`).
  - Sentence-Transformers: `/root/.cache/torch/sentence_transformers`.
  - Hugging Face models generally: `/root/.cache/huggingface`.
  - Pyannote speaker diarization models: under `HF_HOME` (`/root/.cache/huggingface`). pyannote.audio 4.x uses `HF_HOME`; `PYANNOTE_CACHE` is no longer used.
- Use a singleton model loader to avoid repeated model instantiation within a process.
- spaCy capitalization is always enabled; models are baked into the Docker image: `en_core_web_sm`, `es_core_news_sm`, `fr_core_news_sm`, `de_core_news_sm`.
- Sentence-Transformers loader rule: only load from a direct cache path if `modules.json` or `config_sentence_transformers.json` exists in that folder; otherwise load by name with `cache_folder` (avoids the "Creating a new one with mean pooling" message while still using caches). Also sets `HF_HOME` and may set `HF_HUB_OFFLINE=1` when a local model directory is used.

## Volume mounts

Always mount caches and media when running containers:

```
-v $(pwd)/models/huggingface:/root/.cache/huggingface
-v $(pwd)/models/sentence-transformers:/root/.cache/torch/sentence_transformers
-v $(pwd)/audio-files:/app/audio-files
```

Build efficiency: combine pip installs in a single `RUN` with `--no-cache-dir`; keep `COPY . .` last for layer caching; use `.dockerignore` to exclude `audio-files/` and `models/` from build context.

## Transcription orchestration (Whisper usage)

Two supported modes:
- Single-call transcription (recommended when resources allow): process the full file in one call so Whisper maintains context. Enabled via `--single` in `podscripter.py`.
- Overlapped-chunk transcription (fallback): default path with 480s chunks and 3s overlap.

Common settings:
- `vad_filter=True` by default for both modes; `speech_pad_ms=200`. CLI exposes `--no-vad` and `--vad-speech-pad-ms <int>`.
- `condition_on_previous_text=True` to keep context continuity.
- For chunked mode, pass `initial_prompt` using the last ~200 characters of accumulated text (`PROMPT_TAIL_CHARS=200`).
- Deduplicate overlap during merge using global timestamps (skip segments that finish before the prior chunk's end).
- Keep `beam_size` modest (1-3) for long files; `compute_type` default is `auto` on CPU.
- Optional raw dump for debugging: `--dump-raw` writes `<basename>_raw.txt`.
- For files > 1 hour: prefer single-call mode if resources allow; otherwise overlapped chunking with 3s overlap and dedup on merge.

Chunking and merge helpers (private, in `podscripter.py`):
- `_split_audio_with_overlap(media_file, chunk_length_sec, overlap_sec, chunk_dir)` — generate overlapped chunks (defaults: 480s chunks, 3s overlap).
- `_dedupe_segments(global_segments, last_end, epsilon)` — drop overlap-duplicate segments based on end-time.
- `_accumulate_segments(local_segments, chunk_start, last_end)` — offset per-chunk timestamps to global time and build text.

Robustness/hygiene:
- Chunks written into a `TemporaryDirectory` for automatic cleanup.
- SRT export sorts segments by start time.
- Early input/output validation via `_validate_paths`.
- Constants hoisted: `DEFAULT_CHUNK_SEC`, `DEFAULT_OVERLAP_SEC`, `DEDUPE_EPSILON_SEC`, `PROMPT_TAIL_CHARS`.
- Prefer `pathlib.Path` over `os.path`/`glob`; keep orchestration helpers private.

Public API: `transcribe(...)` returns a structured result with keys `segments`, `sentences`, `detected_language`, `output_path`, `num_segments`, `elapsed_secs`. Since v0.6.0 `result["sentences"]` is `list[Sentence]` (not `list[str]`).

## Post-processing formatting (SentenceFormatter)

`SentenceFormatter` (v0.5.0+) is the unified class for all post-processing merge operations, consolidating domain, decimal, appositive, and emphatic word merges in one location.

- Speaker-aware merge decisions: NEVER merges different speakers.
- Merge provenance tracking via the `MergeMetadata` dataclass for debugging.
- Backward compatible: when `speaker_segments=None`, all merges work identically to pre-v0.5.0 behavior.
- Debug flag: `--dump-merge-metadata` writes merge provenance to `<basename>_merges.txt`.

Usage:

```python
from sentence_formatter import SentenceFormatter

formatter = SentenceFormatter(
    language='es',
    speaker_segments=speaker_word_ranges  # None when diarization disabled
)
formatted_sentences, merge_metadata = formatter.format(sentences)
```

Merge types:
- Domain merges: "example." + "com" -> "example.com" (with natural-language guards).
- Decimal merges: "99." + "9%" -> "99.9%".
- Spanish appositive merges: ", de Texas. Estados Unidos" -> ", de Texas, Estados Unidos".
- Emphatic word merges: "No. No. No." -> "No, no, no." (ES/FR/DE).

Architectural separation: `SentenceSplitter` = boundaries; `SentenceFormatter` = formatting. `SentenceFormatter.format()` runs BEFORE `_sanitize_sentence_output()` (in `podscripter.py`) to keep word-count alignment.

## Formatting responsibilities (punctuation_restorer.py)

- Ellipsis continuation and domain-aware splitting are exposed via `assemble_sentences_from_processed(processed, language)` (public API).
- Comma spacing is centralized in `_normalize_comma_spacing(text)` and MUST NOT be re-implemented inline. It removes spaces before commas, deduplicates multiple commas, and ensures a single space after commas. Trade-off: thousands like "1,000" become "1, 000" (acceptable, prioritizes number-list spacing like "147,151,156" -> "147, 151, 156").
- Private helpers: `_normalize_initials_and_acronyms`, `_normalize_dotted_acronyms_en` (legacy alias), `_fr_merge_short_connector_breaks`, and others.
- Segment carry-over: when a segment ends without terminal punctuation, carry the trailing fragment into the next segment for French and Spanish.
- SRT normalization: reading-speed-based cue timing to prevent lingering in silences (defaults: cps=15.0, min=2.0s, max=5.0s, gap=0.25s).

### Centralized punctuation system

- `_should_add_terminal_punctuation()` — single centralized function for all period/punctuation insertion decisions (strips trailing `,;: ` before adding terminal marks).
- `PunctuationContext` — context-aware rules: `STANDALONE_SEGMENT`, `SENTENCE_END`, `FRAGMENT`, `TRAILING`, `SPANISH_SPECIFIC`.
- `restore_punctuation_segment()` — segment-aware API for processing individual Whisper segments.
- Benefit: single source of truth; prevents bugs like "Ve a" -> "Ve a.".

### Formatting internals (centralized constants and helpers)

- Centralized thresholds/configs: `LanguageConfig` via `get_language_config(language)` and `_get_language_thresholds(language)` control Spanish semantic thresholds (`semantic_question_threshold_with_indicator`, `semantic_question_threshold_default`) and splitting thresholds (`min_total_words_no_split`, `min_chunk_before_split`, `min_chunk_inside_question`, `min_chunk_capital_break`, `min_chunk_semantic_break`, `semantic_whisper_lookahead`). Also provides per-language greetings and question-starter lists for en/fr/de/es.
- Per-language constants in `punctuation_restorer.py`:
  - Spanish: `ES_QUESTION_WORDS_CORE`, `ES_QUESTION_STARTERS_EXTRA`, `ES_GREETINGS`, `ES_CONNECTORS`, `ES_POSSESSIVES`.
  - French/German: `FR_GREETINGS`, `DE_GREETINGS`, `FR_QUESTION_STARTERS`, `DE_QUESTION_STARTERS`.
  - English: `EN_QUESTION_STARTERS`.
- Spanish helper functions (pure, testable): `_es_greeting_and_leadin_commas`, `_es_wrap_imperative_exclamations`, `_es_normalize_tag_questions`, `_es_fix_collocations`, `_es_pair_inverted_questions`, `_es_merge_possessive_splits`, `_es_merge_aux_gerund`, `_es_merge_capitalized_one_word_sentences`, `_es_intro_location_appositive_commas`.
- Shared utilities: `_split_sentences_preserving_delims(text)`, `_normalize_mixed_terminal_punctuation(text)` (removes `!.`, `?.`, `!?`, compresses repeats), `_finalize_text_common(text)`, `assemble_sentences_from_processed(processed, language)`.
- Public API hygiene: public functions are type-annotated (`restore_punctuation`, `transformer_based_restoration`, `apply_semantic_punctuation`, `is_question_semantic`, `is_exclamation_semantic`, `format_non_spanish_text`). `punctuation_restorer.py` is import-only (no `__main__`). Legacy `format_spanish_text` was removed.
- Capitalization (spaCy mode): uses `LanguageConfig` connectors/possessives for Spanish to avoid mid-sentence mis-capitalization (e.g., `tu español`); multi-layered entity protection (spaCy NER + cross-linguistic analysis + contextual patterns); conservative location capitalization (only after strong cues like `vivo en`, `trabajo en`, `soy de`, `vengo de`).
- Tuning guidance: prefer editing constants/thresholds over changing logic; avoid one-off hacks; after any change run `pytest` inside Docker with model caches mounted.

## Sentence splitting (4-signal hybrid)

Language-agnostic across EN/ES/FR/DE:
1. Grammatical guards — avoid ending sentences on coordinating conjunctions, prepositions, and continuative/auxiliary verbs.
2. Semantic coherence — Sentence-Transformers similarity confirms low-coherence boundaries.
3. Configurable thresholds — minimum chunk length, overall length, capital/semantic break limits.
4. Whisper segment boundaries — `all_segments` boundaries are prioritized hints, still gated by grammatical guards and minimum chunk size; ignored at grammatically invalid positions; backward compatible when absent.

Whisper boundary threading (production path):
- Orchestrator passes `all_segments` to `_assemble_sentences(...)` (in `podscripter.py`).
- `_assemble_sentences` extracts character boundary positions and calls `restore_punctuation(text, language, whisper_segments=...)` (in `punctuation_restorer.py`).
- `restore_punctuation` -> `_transformer_based_restoration` (in `punctuation_restorer.py`) instantiates `SentenceSplitter` and calls `SentenceSplitter.split(...)` (in `sentence_splitter.py`); boundary decisions live in `SentenceSplitter._should_end_sentence_here(...)`.
- All splitting lives in `SentenceSplitter` (consolidated in v0.4.0). The legacy module-level `_semantic_split_into_sentences()` / `_should_end_sentence_here()` duplicates in `punctuation_restorer.py` were removed in v0.10.2.

Whisper boundary thresholds (via `_get_language_thresholds(language)`):
- `min_words_whisper_break` (default 10): minimum words in current chunk before honoring a Whisper boundary.
- `max_words_force_split` (default 100): safety for very long run-ons (reserved).

## Speaker diarization integration

Optional, opt-in feature (disabled by default to avoid dependency bloat). Uses pyannote.audio 4.0.4 (community-1 pipeline) with Hugging Face model caching.

Priority: Speaker boundaries > Whisper boundaries > Semantic coherence. Speaker boundaries are passed SEPARATELY to `restore_punctuation()` (not merged with Whisper boundaries) and converted via `_convert_speaker_timestamps_to_char_positions()`. Speaker boundary checks happen BEFORE the general `min_chunk_before_split` threshold so short phrases can break.

Minimum word thresholds (in `SentenceSplitter._should_end_sentence_here`, `sentence_splitter.py`):
- Speaker boundaries: 1 word (definitive signal; `min_words_speaker = 1`, per history v0.6.1).
- Whisper boundaries: 10 words (acoustic pause hints; `min_words_whisper_break`).
- General semantic splitting: 20 words for Spanish, 15 for other languages.

Module structure (`speaker_diarization.py`):
- `diarize_audio(...)` — main entry point, returns `DiarizationResult`.
- `_extract_speaker_boundaries(...)` — extracts timestamps where speakers change; returns filtered boundaries and detailed `BoundaryInfo`.
- `_convert_speaker_timestamps_to_char_positions(...)` — seconds -> character positions (in `podscripter.py`).
- `write_diarization_dump(...)` — debug dump (raw segments, boundary analysis, merge details).
- `DiarizationError` — typed exception; `BoundaryInfo` — TypedDict per potential boundary.

Model caching: pyannote.audio 4.x uses `HF_HOME`. First run requires an HF token (accept the agreement at hf.co/pyannote/speaker-diarization-community-1); subsequent runs use cached models under `/root/.cache/huggingface`.

Env vars: `HF_TOKEN` (alternative to `--hf-token`); precedence CLI flag > environment variable.

Error handling: `DiarizationError` raised on failure; gracefully degrades (logs warning, continues without speaker boundaries); tip suggests `--hf-token`/`HF_TOKEN` for first-time use.

Performance: ~10-30% overhead depending on audio length; CPU by default (same device as Whisper); GPU via `device="cuda"`; caching speeds subsequent runs.

CLI flags: see [CLI spec](cli-spec.md) (`--enable-diarization`, `--min-speakers`, `--max-speakers`, `--hf-token`, `--dump-diarization`).
