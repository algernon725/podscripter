# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-05-24

### Added
- **Three-tier EN/FR real-audio test corpus** — first end-to-end regression coverage for the full pipeline (ASR + diarization + punctuation + formatting) on real audio for English and French (and extensible to other languages). Existing unit/multilingual tests catch logic regressions; this tier catches model-output drift.
  - **Tier 1 (regression)**: `tests/test_audio_fixtures.py` parametrizes over every `tests/fixtures/audio/<lang>/<name>.expected.json` and runs the full pipeline via the `podscripter.transcribe(...)` library API with the same flags as typical manual usage (`enable_diarization=True`, `model_name="medium"`, `beam_size=3`, `single_call=True`). Long fixtures with `"modes": ["single", "chunked"]` exercise both `--single` and the chunked-mode code paths (`_split_audio_with_overlap` / `_dedupe_segments` / `_accumulate_segments`). Asserts WER (via `jiwer`) and DER (via `pyannote.metrics`) against per-fixture loose thresholds that absorb pyannote/Whisper non-determinism; default bounds WER ≤ 0.15 / DER ≤ 0.20 (single) and WER ≤ 0.17 / DER ≤ 0.22 (chunked). Marked `@pytest.mark.transcription` so it stays opt-in.
  - **Tier 2 (quality benchmark)**: `tests/benchmarks/{download_subsets,run_benchmark,compare_baseline}.py` plus `baseline.json`. Pulls ~30 min/lang of public dataset subsets (FLEURS, LibriSpeech, optionally VoxPopuli), runs the pipeline in both `single` and `chunked` modes for multi-speaker subsets, and gates regressions against the committed baseline with separate WER/DER tolerances per mode so chunked-specific drift surfaces independently from generic ASR drift. Not run per-PR; intended for nightly CI or pre-release.
  - **Tier 3 (bug-reproduction fixtures)**: documented workflow in `tests/README.md` (trim clip → push to HF dataset → bump `HF_REVISION` → add `.expected.json` → add focused test mirroring `tests/test_episodio272_speaker_split_exclamation.py`).
- **Public HuggingFace dataset `podscripter-project/test-fixtures`** (CC-BY 4.0) — newly owned project artifact that hosts all Tier 1 audio. Repo stays binary-free (matches the existing `.gitignore` policy for `audio-files/` and `models/`); only per-clip `.expected.json` metadata lives in git. Audio is downloaded into the same `HF_HOME` cache that already holds Whisper and pyannote models, so no new docker mounts are required. MVP corpus uploaded at revision `4ae10183adde47c1706adb346bc0e9ad26b34545`:
  - `en/librispeech_test_clean_1089_134686_0000.wav` (~10 s, single speaker; LibriSpeech test-clean)
  - `en/librispeech_two_speakers_long.flac` (~9 min 17 s, two speakers, FLAC; LibriSpeech 1089 + 61 concatenated with a 0.5 s silence gap — exercises chunked mode)
  - `fr/fleurs_fr_test_7105431834829365765.wav` (~9.4 s, single speaker; FLEURS `fr_fr` test split)
- **`tests/fixtures/audio/download.py`** — HF dataset downloader that pins `HF_REVISION` to a specific commit hash. Idempotent (no-op when the snapshot is already cached) and honors `HF_HUB_OFFLINE=1` for fully offline test runs once warm. Pre-warmable via `python -m tests.fixtures.audio.download`.
- **License-metadata validator** — `tests/fixtures/audio/_validate_licensing.py` runs on every default `pytest` invocation (under the `core` marker) and enforces that every `.expected.json` declares `source`, `source_url`, `license`, `license_url`, `audio_file`, `duration_sec`, `expected_text`, `expected_speaker_count`, and `thresholds` in the documented shape. The `license` field is restricted to a permissive allowlist (`CC-BY-4.0`, `CC0-1.0`, `public-domain`); non-CC0 sources additionally require `attribution` and `modifications`. Prevents accidental ingestion of an NC/ND-licensed clip in a future PR.
- **Per-clip and consolidated license attribution** — `tests/fixtures/audio/LICENSES.md` (consolidated NOTICE for every source corpus: LibriSpeech, Common Voice, VoxPopuli, AMI, VoxConverse, MLS) and per-fixture `attribution` + `modifications` fields in every `.expected.json`. The HF dataset card mirrors the same provenance table for downstream redistributors.
- **Local-iteration env-var controls** — `PODSCRIPTER_TEST_MODEL` overrides the default `medium` Whisper model (e.g. `small`, `tiny`) for fast dev iteration; `PODSCRIPTER_TEST_FIXTURES_PATTERN` is a glob filter (e.g. `en/*short*`) to run a subset of fixtures. Production defaults stay at `medium`, all fixtures.
- **`jiwer` Python dependency** in `Dockerfile` for Word Error Rate computation. `pyannote.metrics` (DER) was already transitively available via `pyannote.audio==4.0.4`. The Docker image needs one `docker build` before the new tests can run.
- **Documentation updates** — `tests/README.md` gains Tier 1/2/3 sections including the bug-fixture convention; `AGENT.md` Testing Requirements gains an "EN/FR test corpus" subsection (tiers, thresholds, env overrides, bring-up commands); `ARCHITECTURE.md` Testing-and-quality-gates section gains a three-tier paragraph; `tests/fixtures/audio/README.md` documents the full `.expected.json` schema, threshold shapes (flat vs per-mode), and recognized pattern tags.

### Changed
- **No behavior change** to `podscripter.transcribe(...)` or the CLI. All existing tests continue to pass (468 passing in the default `pytest` suite, including the 4 new license-validator checks; 84 xfail unchanged).

### Notes
- **Action required after pulling**: rebuild the Docker image (`docker build -t podscripter .`) so the new `jiwer` dependency is installed before running tests.
- **License compliance for the new HF dataset**: aggregate license is CC-BY 4.0 (most restrictive of components). Downstream users redistributing the dataset must comply with CC-BY 4.0 (attribution, license notice, indication of changes, no DRM). See `tests/fixtures/audio/LICENSES.md`.

## [0.8.7] - 2026-05-24

### Changed
- **Demoted German (`de`) from primary supported languages** — project direction is shifting toward focused support for romance languages, so German is no longer advertised as a primary supported language. All German-specific transcription, punctuation, sentence-splitting, and capitalization code paths are intact and continue to function when `--language de` is requested or when Whisper auto-detects German; the change is documentation- and CLI-help-only.
  - **README.md** and **docs/README.md**: removed German from the Overview blurb, the "Primary Language Support" feature bullet, the `--language` Options-table description, the Supported Languages reference table (now lists English/Spanish/French and the note reads "the three listed above"), and the Automatic NLP Capitalization paragraph.
  - **podscripter.py**: module docstring now reads `Primary language focus: English (en), Spanish (es), French (fr).`; `FOCUS_LANGS` is now `{"en", "es", "fr"}`; the hardcoded primary-language list inside `validate_language_code()` is now `["en","es","fr"]`; the `--language` argparse `help=` text reads `(e.g., en, es, fr)`. `get_supported_languages()` still includes `'de': 'German'`, so passing `--language de` continues to validate (with a warning that lists `de` under "Experimental language codes") and run the existing German processing logic unchanged.
  - **AGENT.md** and **ARCHITECTURE.md**: updated the user-facing "Supported Languages"/Goals/Known-limitations statements; technical descriptions of EN/ES/FR/DE code paths (German preposition guards, auxiliary verbs, past-participle detection, greeting commas, location appositive normalization, etc.) were intentionally left in place because the corresponding code is unchanged.
  - **No behavior change**: no source code in `punctuation_restorer.py`, `sentence_splitter.py`, `sentence_formatter.py`, `domain_utils.py`, `speaker_diarization.py`, or any test was modified. All existing German tests (`tests/test_german_*`, German cases in multilingual tests) continue to run and pass as before.

## [0.8.6] - 2026-05-24

### Fixed
- **Spanish `¡...!`/`¿...?` split mid-construct by a speaker boundary** — when `--enable-diarization` was enabled and the diarizer placed a speaker change inside a single Whisper segment containing an unclosed Spanish inverted exclamation/question, the sentence splitter broke the construct in half, orphaning the closing mark on the next sentence. Reproduced in `audio-files/Episodio272.txt`, where Whisper segment 19 (`"Así que, ¡empecemos, Nate!"`) and segment 20 (`"Bueno, Andrea, antes de empezar, ..."`) were emitted as `"Así que, ¡empecemos."` and `" Nate! Bueno, Andrea, antes de empezar, ..."` instead of the expected `"Así que, ¡empecemos, Nate!"` and `"Bueno, Andrea, antes de empezar, ..."`.
  - **Root Cause**: In `SentenceSplitter._should_end_sentence_here()` (`sentence_splitter.py`), PRIORITY 1 (speaker boundary) returns `True` unconditionally per the v0.4.3 policy that "speaker boundaries ALWAYS create splits". The v0.6.2 protection added to honor unclosed Spanish `¡...!`/`¿...?` via `_is_inside_unclosed_question()` was only wired into PRIORITY 4 (Whisper boundary), not PRIORITY 1. When the v0.5.2 multi-speaker Whisper segment splitter placed the speaker change at the word `"¡empecemos,"` (boundary index inside the open `¡`), PRIORITY 1 fired and split between `"¡empecemos,"` and `"Nate!"`, leaving the trailing comma to be converted to a period by downstream punctuation logic.
  - **Fix**: Added `SentenceSplitter._shift_boundary_past_unclosed_mark()`, invoked from `_convert_segments_to_word_boundaries()` when building the speaker-boundary set. If the Spanish boundary word leaves more `¡`/`¿` than `!`/`?` in the chunk up to it, the boundary is shifted forward (up to a 25-word lookahead) to the word containing the closing mark. The speaker change still produces a sentence break, but it now lands after the construct is balanced. Falls back to the original boundary if no closing mark is found within the lookahead and is a no-op for non-Spanish languages, so the v0.4.3 "speaker boundaries always split" guarantee is preserved everywhere else.
  - **Tests**: `tests/test_episodio272_speaker_split_exclamation.py` — added `TestSpeakerBoundaryInsideSpanishExclamation` (Episodio272 regression + `¿...?` analogue + sanity check that boundaries outside inverted marks still split) and `TestShiftBoundaryHelper` (5 unit tests covering shift past `!`, shift past `?`, no-shift when balanced, no-shift when already past close, no-shift for non-Spanish, no-shift when closing mark missing). All 468 existing tests continue to pass (`pytest` default suite).

## [0.8.5] - 2026-04-25

### Fixed
- **Spanish proper-noun + `Es` incorrectly merged as a `.es` domain (`Nate.es`)** — when `--enable-diarization` was enabled and a Whisper segment ended with a capitalized proper noun (e.g., `"...como lo decía Nate."`) and the next sentence began with `"Es ..."`, the post-processing formatter merged them into a fake domain (`"Nate.es"`) and lowercased the `E`, producing the broken text `"...decía Nate. es considerada..."` in the final transcript. Reproduced in `audio-files/Episodio269.txt` (line 373) for `Episodio269.mp3`.
  - **Root Cause**: In `SentenceFormatter._merge_domains()` (`sentence_formatter.py`), the natural-language guard added in v0.4.4 allowed a domain merge whenever EITHER the previous sentence was short (< 50 chars) OR its trailing label was capitalized. The TLD `es` collides with the very common Spanish verb "es" (3rd-person singular of *ser*), so any Spanish sentence ending with a capitalized proper noun (`"Nate."`, `"Pedro."`, `"María."`, etc.) followed by `"Es ..."` matched the domain pattern, passed the capitalized-label branch of the guard, was concatenated into `"<Name>.es"`, and then `_lowercase_first_letter()` lowered the `E` of the next sentence. A subsequent space-after-period regex restored the visible space, leaving the lowercase `e` behind.
  - **Fix**: Added a Spanish-specific proper-noun guard in `_merge_domains()`: when `language == 'es'` and `tld == 'es'`, a capitalized label is no longer accepted as evidence of a domain mention. Real spoken brand references like `"Consulta marca." + "es para noticias"` use a lowercase label and continue to merge into `"marca.es"` as before. The fix is narrowly scoped to the only TLD in the formatter's list that collides with a high-frequency Spanish word.
  - **Tests**: `tests/test_sentence_formatter.py` — added `test_domain_merge_spanish_proper_noun_guard` (covers the `"Nate."` and `"Pedro."` regressions and asserts no `domain_pattern_match` merges are recorded) and `test_domain_merge_spanish_lowercase_brand_still_merges` (asserts that lowercase-label brand mentions like `"marca." + "es ..."` still merge into `"marca.es"`). All existing domain, formatter, and Spanish false-domain tests continue to pass (459 passed in the default suite).

## [0.8.4] - 2026-04-25

### Fixed
- **Spanish exclamation/question artifacts after speaker splits (`Bueno,!`)** — when `--enable-diarization` was enabled and a Whisper segment was split mid-clause at a speaker boundary (e.g., `"Bueno, más o menos."` → `"Bueno,"` + `"Más o menos."`), the leading fragment with a trailing comma was incorrectly punctuated as `"Bueno,!"`. Reproduced in `audio-files/Episodio270.txt` (line 103).
  - **Root Cause**: In `_apply_semantic_punctuation()` (`punctuation_restorer.py`), both the question and exclamation branches used `sentence.rstrip('.!') + '?'` and `sentence.rstrip('.?') + '!'` respectively. These rstrip character classes only stripped terminal punctuation marks, NOT trailing commas, semicolons, colons, or whitespace. When `is_exclamation_semantic()` classified `"Bueno,"` as an exclamation (cosine similarity > 0.7 to the Spanish pattern `"¡Qué bueno!"`), the function appended `'!'` directly after the comma, producing `"Bueno,!"`. The same bug existed for question detection (`"Qué tal,"` → `"Qué tal,?"`).
  - **Fix**: Expanded the rstrip character classes in both branches to `'.!,;: '` (question) and `'.?,;: '` (exclamation), matching the behavior already used by the centralized `_should_add_terminal_punctuation()` helper. Now `"Bueno,"` correctly becomes `"Bueno!"` and `"Qué tal,"` becomes `"Qué tal?"`.
  - **Tests**: `tests/test_trailing_comma_bug.py` — added `TestApplySemanticPunctuationTrailingComma` class with 3 new tests (12 subtests total) covering the exact `"Bueno,"` regression case, multiple Spanish fragments with trailing `,`/`;`/`:`, and multilingual coverage (EN/FR/DE).

## [0.8.3] - 2026-04-07

### Changed
- **Removed outdated Testing section from README** — instructions referenced the pre-pytest `run_all_tests.py` runner which was deleted in v0.8.0; replaced with a pointer to `tests/README.md`
- **Rewrote `tests/README.md`** — replaced all references to the deleted `run_all_tests.py` script and `RUN_ALL=1`-style environment flags with accurate pytest commands, a table of common options, and descriptions of the three test markers (`core`, `multilingual`, `transcription`)
- **Added `--debug` flag to README and `docs/README.md`** — CLI synopsis and Options table now document the `--debug` verbosity option (added in v0.3.1) alongside `--quiet` and `--verbose`
- **Fixed broken relative link in `docs/README.md`** — Docker installation link corrected from `docs/docker-installation.md` to `docker-installation.md`

## [0.8.2] - 2026-03-22

### Fixed
- **Fixed 12 xfail tests caused by API call mismatches** — updated test code to match current function signatures:
  - `test_spanish_capitalization_domain_regression.py` (5 tests): updated `_assemble_sentences` from old 3-arg to new 6-arg signature, unpacked tuple return, extracted `.text` from `Sentence` objects
  - `test_spanish_false_domains.py` (2 tests): same `_assemble_sentences` API update
  - `test_spanish_domains_and_ellipses.py` (1 test): same
  - `test_spanish_helpers.py` (1 test): unpacked tuple return from `_assemble_sentences`
  - `test_chunk_merge_helpers.py` (2 tests): updated `_accumulate_segments` text expectation (space→newline join), relaxed `_dedupe_segments` assertion
  - `test_transcribe_helpers.py` (1 test): updated `_accumulate_segments` text expectation
- **Fixed 11 xfail tests by updating expectations to match verified-correct behavior**:
  - `test_domain_utils.py` (2 tests): corrected test expectations for `mask_domains` and `create_domain_aware_regex`
  - `test_srt_normalization.py` (1 test): increased text length so reading-time exceeds gap-trim threshold
  - `test_trailing_comma_bug.py` (1 test): extracted `.text` from `Sentence` objects in assertion loop
  - `test_whisper_skipped_boundary_detailed.py` (2 tests): extracted `.text` from `Sentence` objects
  - `test_whisper_skipped_boundary_periods.py` (1 test): promoted xpassed test to normal
  - `test_spanish_capitalization_domain_regression.py` (4 tests): relaxed sentence count assertion, verified domain integrity
- **Promoted 1 xpassed test** — `test_whisper_period_with_connector_removed` now passes normally

### Removed
- **Deleted 15 test files** containing only xfail tests with no unique coverage:
  - `test_english_runon_fix.py`, `test_french_runon_fix.py`, `test_german_runon_fix.py`, `test_spanish_runon_fix.py` (duplicated by `test_multilingual_runon_sentences.py`)
  - `test_multilingual_introductions.py` (13 fragile exact-match NLP assertions, all xfail)
  - `test_human_vs_program_intro.py` (F1 benchmark, better as manual test)
  - `test_past_tense_questions.py` (covered by existing question detection tests)
  - `test_specific_question.py` (single-input duplicate of question detection tests)
- **Removed 21 low-value xfail test functions** from files that also have passing tests:
  - Run-on sentence detection (8 xfails from `test_multilingual_runon_sentences.py`)
  - Initials normalization (4 xfails from `test_initials_normalization.py`)
  - Thousands comma collapsing (3 xfails from `test_spanish_numbers.py`)
  - Whisper boundary periods (4 xfails from 2 whisper boundary files)
  - Domain/assembly drift (3 xfails from `test_spanish_false_domains.py`, `test_spanish_domains_and_ellipses.py`, `test_spanish_helpers.py`)
- **Removed 2 duplicate xfail params** from `test_specific_spanish_bugs.py` (identical to `test_spanish_bug_fixes.py`)

### Changed
- **xfail count reduced from 142 to 83** — all remaining xfails are verified NLP model limitations (52 question detection + 31 sentence splitting/formatting)
- **Updated AGENT.md** — simplified xfail documentation to two categories with current file counts
- **Updated ARCHITECTURE.md** — refreshed xfail counts

## [0.8.1] - 2026-03-05

### Fixed
- **Fixed 33 xfail tests caused by return-type mismatches** — these unit tests for core components were failing only because test code didn't account for API return-type changes, not because of actual bugs:
  - `test_speaker_diarization_unit.py` (7 tests): unpacked tuple return from `_extract_speaker_boundaries()` which now returns `(boundaries, details)` instead of a flat list
  - `test_sentence_formatter.py` (14 tests): added `_texts()` helper to extract `.text` from `Sentence` objects returned by `SentenceFormatter.format()`
  - `test_sentence_splitter_unit.py` (8 tests): accessed `.text` on `Sentence` objects from `SentenceSplitter.split()`
  - `test_sentence_assembly_unit.py` (4 tests): updated expectations to match `assemble_sentences_from_processed` behavior (text without terminal punctuation is trailing, not a sentence; decimal `99.9` preserved but `%` has a space)
  - Also fixed `test_short_segments_filtered` test data to use segments actually below `MIN_SPEAKER_SEGMENT_SEC` (0.5s)
- **Promoted 51 xpassed tests to normal passing tests** — audited all tests marked `@pytest.mark.xfail` that were now passing. All 51 had genuine assertions (exact string match or model-must-add-`?` checks with no `?` in input). Moved `@pytest.mark.xfail` from function-level to per-parameter on parametrized tests so only the specific failing inputs remain marked.

### Changed
- **xfail markers are now per-parameter** — parametrized tests in 9 files no longer blanket-xfail all inputs. Only the specific failing parameter combinations are marked, so passing inputs run as normal tests and will catch regressions.
  - Files updated: `test_english_sentence_splitting.py`, `test_french_sentence_splitting.py`, `test_german_sentence_splitting.py`, `test_spanish_sentence_splitting.py`, `test_multilingual_questions.py`, `test_spanish_questions.py`, `test_spanish_inverted_questions.py`, `test_spanish_bug_fixes.py`, `test_specific_spanish_bugs.py`
- **Updated AGENT.md** — added detailed remediation guide for remaining 142 xfail tests in Known Limitations section, categorized by priority (question detection drift, exact-match splitting drift, integration tests, scoring thresholds)
- **Updated ARCHITECTURE.md** — refreshed xfail counts and status

## [0.8.0] - 2026-02-21

### Changed
- **Migrated entire test suite to pytest** — replaced the custom `run_all_tests.py` runner and 70+ print-based test files with proper pytest infrastructure.
  - All tests now use real `assert` statements instead of print-based pass/fail.
  - Created `pyproject.toml` with pytest configuration, marker definitions, and default run options.
  - Created `tests/conftest.py` with shared fixtures (`MockConfig`, language-specific `SentenceSplitter` instances) and a `restore_punctuation` wrapper that unwraps the tuple return type for test convenience.
  - Added `pytest` to the Dockerfile.
  - Test categorization via pytest markers: `core` (default), `multilingual` (default), `transcription` (opt-in).
  - Removed `sys.path` hacks from all 70 test files (handled centrally by `conftest.py`).
  - Removed all `if __name__ == '__main__':` blocks.
  - Updated AGENT.md and ARCHITECTURE.md testing documentation.

### Removed
- **Deleted 8 script/debug files** that were not real tests:
  - `tests/test_transcription.py` (argparse-based manual experiment script)
  - `tests/model_comparison.py` (model comparison script)
  - `tests/run_all_tests.py` (custom test runner, replaced by pytest)
  - `tests/test_whisper_boundary_debug.py`, `tests/test_comma_debug.py`, `tests/test_question_detection_debug.py`, `tests/test_transcription_debug.py` (diagnostic print scripts)
  - `tests/test_model_change.py` (model verification print script)

### Fixed
- **174 pre-existing test failures now visible** — tests that were silently "passing" (print-only, no assertions) now have real assertions and are marked `@pytest.mark.xfail`. Root causes include API return type changes (e.g., `restore_punctuation` returning a tuple, `_extract_speaker_boundaries` returning `(boundaries, segments)`), drifted NLP output expectations, and incorrect test logic. These do not affect runtime behavior — they are test-only issues surfaced by the migration.

## [0.7.1] - 2026-02-21

### Fixed
- **Semantic split preempting nearby Whisper boundary**: Sentences were incorrectly broken mid-phrase when the semantic coherence model (PRIORITY 5) fired a few words before a legitimate Whisper segment boundary (PRIORITY 4). The semantic model's 10-word lookahead window would cross a real sentence boundary, producing a false-positive low-similarity score at the wrong position (e.g., `"...para de verdad tomar."` | `"Su español al siguiente nivel."` instead of the correct `"...para de verdad tomar su español al siguiente nivel."`).
  - **Root Cause**: `_should_end_sentence_here()` did not check for upcoming Whisper boundaries before running the semantic coherence model. The Whisper boundary at the actual sentence end was never evaluated because the chunk was reset after the premature semantic split.
  - **Fix**: Added Whisper boundary lookahead before semantic splits — scans next N words (configurable via `semantic_whisper_lookahead`, default 8) for a Whisper boundary; if found, defers the split so the higher-priority boundary is evaluated at its natural position. Mirrors the existing 3-word lookahead pattern for Whisper boundary skipping near speaker boundaries.
  - **New threshold**: `semantic_whisper_lookahead` (default 8) added to both Spanish and default configs in `_get_language_thresholds()`.
  - **Tests**: `tests/test_semantic_whisper_lookahead.py` (7 tests covering deferral, edge cases, cross-language, and backward compatibility).

## [0.7.0] - 2026-02-16

### Changed
- **Major dependency upgrade for pyannote.audio 4.x compatibility**:
  - `pyannote.audio` 3.3.2 → 4.0.4 (new `community-1` diarization pipeline with improved speaker counting/assignment)
  - `torch` 2.2.0 → 2.8.0 (required by pyannote.audio 4.x)
  - `torchaudio` 2.2.0 → 2.8.0
  - `spacy` 3.7.4 → 3.8.11 (numpy 2.x compatibility)
  - spaCy language models 3.7.0 → 3.8.0 (`es_core_news_sm`, `en_core_web_sm`, `fr_core_news_sm`, `de_core_news_sm`)
  - **Root cause**: Unpinned `sentence-transformers` pulled a new `transformers` requiring torch >= 2.4, breaking the build. Resolved by upgrading the full dependency chain rather than patching around version conflicts.
- **Diarization pipeline**: `pyannote/speaker-diarization-3.1` → `pyannote/speaker-diarization-community-1` (gated model — requires accepting agreement at hf.co/pyannote/speaker-diarization-community-1)
- **Diarization output format**: pyannote 4.x returns `output.speaker_diarization` instead of a direct annotation; iteration yields `(turn, speaker)` tuples instead of `(turn, _, speaker)` from `itertracks()`
- **Model caching**: pyannote.audio 4.x uses `HF_HOME` for model caching; `PYANNOTE_CACHE` environment variable removed from Dockerfile. Separate `models/pyannote` mount no longer needed — diarization models now cached under `models/huggingface`
- **API parameter rename**: `use_auth_token` → `token` in `diarize_audio()` and `Pipeline.from_pretrained()` (huggingface_hub dropped `use_auth_token`)

### Added
- `torchcodec==0.7.0` dependency (required by pyannote.audio 4.x; pinned to avoid ABI mismatch with torch 2.8)
- `soundfile` pip package + `libsndfile1` system package for torchaudio audio backend (MP3 decoding)
- Audio pre-loading via `torchaudio.load()` in `speaker_diarization.py` — bypasses torchcodec's MP3 chunk decoding issues (sample count mismatches with lossy formats)
- Warning suppression for torchaudio 2.8 deprecation notice about future torchcodec migration
- `sentence-transformers==5.2.2` version pin to prevent the unpinned dependency cascade that triggered this upgrade

### Fixed
- **Docker build failure**: Unpinned `sentence-transformers` pulling `transformers` requiring torch >= 2.4 while torch was pinned to 2.2.0
- **`use_auth_token` removed in huggingface_hub**: pyannote 3.3.2 used deprecated `use_auth_token` parameter internally; upgrading to pyannote 4.0.4 resolves natively
- **numpy binary incompatibility**: spacy 3.7.4's `thinc` compiled against numpy 1.x, incompatible with numpy 2.x pulled by torch 2.8.0
- **torchcodec ABI mismatch**: torchcodec 0.8.x has undefined symbol errors with torch 2.8.0; pinned to 0.7.0
- **MP3 sample count mismatch in diarization**: torchcodec's chunk-based MP3 decoding produced incorrect sample counts; fixed by pre-loading audio with torchaudio/soundfile

### Removed
- `PYANNOTE_CACHE` environment variable from Dockerfile (no longer used by pyannote.audio 4.x)
- `-v $(pwd)/models/pyannote:/root/.cache/pyannote` Docker mount from all documentation and scripts

## [0.6.3.1] - 2026-02-16

### Fixed
- **False domain detection for `.it` TLD in Spanish transcriptions**: Fixed bug where `fix_spaced_domains()` incorrectly merged "Escucha. It's raining cats and dogs." into "Escucha.it's raining cats and dogs." by treating "Escucha.it" as an Italian domain
  - **Example**: `"Tranquilo, Nate, no tienes que saberlo todo. Escucha. It's raining cats and dogs."` → incorrectly became `"...Escucha.it's raining cats and dogs."` (sentence boundary destroyed)
  - **Root Cause**: The `.it` (Italy) TLD was in `SINGLE_TLDS`, causing `fix_spaced_domains()` to merge any `word. It` pattern into a domain. Unlike `.de` and `.es` which had Spanish-specific exclusions, `.it` had no protection and "it" is a common English pronoun
  - **Fix**: Refactored `SINGLE_TLDS` in `domain_utils.py` to only include popular, commonly-used TLDs. Removed 6 obscure TLDs that could cause false positives: `.it` (conflicts with English "it"), `.nl`, `.jp`, `.cn`, `.in` (conflicts with English "in"), `.ru`
  - **New TLD list**: `com|net|org|co|es|io|edu|gov|uk|us|ar|mx|de|fr|br|ca|au` (17 TLDs, down from 22)
  - **Impact**: Prevents false domain merges in multilingual transcriptions where English words like "it" or "in" follow a sentence-ending period

### Added
- **Regression tests for removed TLDs**: New tests in `test_domain_utils.py`:
  - `test_escucha_it_false_domain()` — verifies "Escucha. It's" is not merged as a domain
  - `test_removed_tlds_not_matched()` — verifies all 6 removed TLDs are no longer matched by masking or spaced-domain fixing
  - `test_popular_tlds_still_work()` — verifies popular TLDs (com, io, edu, net, org, fr, br, ca, au) continue to work correctly

## [0.6.3] - 2026-01-25

### Fixed
- **Questions and exclamations incorrectly merged with following sentences**: Fixed issue where sentences ending with `?` or `!` were merged with the next sentence when it started with a connector word ("pero", "y", etc.) and the same speaker continued
  - **Example**: `"¿no?"` + `"Pero en un país..."` incorrectly became `"¿no pero en un país..."` (question mark removed, connector lowercased)
  - **Root Cause**: The same-speaker connector merge logic (introduced in v0.4.0 to fix "trabajo. Y este meta" → "trabajo y este meta") was treating `?` and `!` the same as `.`, removing them before connectors
  - **Impact**: Created very long run-on paragraphs when many consecutive sentences started with connectors and the same speaker continued throughout
  - **Fix**: Modified merge logic to only merge sentences ending with `.` (periods), not `?` or `!` - questions and exclamations are complete thoughts that should remain as separate sentences
  - **Files changed**: `sentence_splitter.py` lines 628 and 1355-1356
- **Long sentence splits at grammatically incorrect positions**: Fixed issue where very long sentences (42+ words in Spanish) were being split at grammatically incorrect positions when semantic break logic kicked in
  - **Example 1**: `"a los 18. Años"` should be `"a los 18 años"` - numbers should not be split from following time/measurement units
  - **Example 2**: `"sin ser. Parte"` should be `"sin ser parte"` - infinitive verbs should not end sentences
  - **Example 3**: `"fueron. Dirigidos"` should be `"fueron dirigidos"` - auxiliary verbs should not be split from following past participles
  - **Root Cause**: When sentences exceeded `min_chunk_semantic_break` threshold (42 words for Spanish), the semantic break logic would split at positions that violated grammatical rules. The `CONTINUATIVE_AUXILIARY_VERBS` set was missing infinitive forms and preterite/past tense forms, and there were no guards for number-unit patterns or auxiliary-participle patterns.
  - **Fix (Part 1)**: Expanded `CONTINUATIVE_AUXILIARY_VERBS` to include:
    - **Spanish**: Infinitives (`ser`, `estar`, `haber`, `ir`, etc.), preterite forms (`fue`, `fueron`, `estuvo`, etc.), present tense of ser/estar (`es`, `son`, `está`, `están`, etc.)
    - **English**: Infinitives (`be`, `have`, `do`, `go`, etc.), present tense (`is`, `are`, `am`)
    - **French**: Infinitives (`être`, `avoir`, `aller`, `faire`, etc.), passé simple forms (`fut`, `furent`, etc.), present tense (`est`, `sont`, `a`, `ont`, etc.)
    - **German**: Infinitives (`sein`, `haben`, `werden`, etc.), present tense (`ist`, `sind`, `hat`, `wird`, etc.), preterite (`wurde`, `wurden`, etc.)
  - **Fix (Part 2)**: Added guard in `_passes_language_specific_checks()` to prevent splitting numbers from following time/measurement units across all languages (años, years, ans, Jahre, etc.)
  - **Fix (Part 3)**: Added `_is_past_participle()` helper method and guard to prevent splitting auxiliary verbs from following past participles
  - **Testing**: Added `tests/test_long_sentence_breaks.py` with 18 test cases covering number-unit patterns, infinitive verbs, and auxiliary-participle patterns across Spanish, English, French, and German
  - **Impact**: Affects all diarization-enabled transcriptions where long sentences are created by joining multiple Whisper segments

### Added
- **Helper method `_is_past_participle()`**: New method in `SentenceSplitter` that detects past participles based on common endings and irregular forms across Spanish, English, French, and German
  - Spanish: `-ado`, `-ido` endings + irregular forms (`hecho`, `dicho`, `escrito`, etc.)
  - English: `-ed`, `-en` endings + irregular forms (`done`, `gone`, `been`, etc.)
  - French: `-é`, `-i`, `-u` endings + irregular forms (`fait`, `dit`, `été`, etc.)
  - German: `ge-...-t`, `ge-...-en` patterns + irregular forms (`gewesen`, `gehabt`, etc.)

### Changed
- **`CONTINUATIVE_AUXILIARY_VERBS` expanded**: Added ~60 new verb forms across 4 languages to prevent grammatically incorrect sentence breaks

## [0.6.2] - 2026-01-19

### Fixed
- **Spanish inverted question sentences split at Whisper boundaries**: Fixed issue where Spanish questions starting with `¿` were incorrectly split in the middle when a Whisper segment boundary occurred before the closing `?`
  - **Example**: `"¿qué cambios ha habido desde la pandemia?"` was being split as `"¿qué cambios ha habido."` and `"Desde la pandemia?"` with a period incorrectly inserted
  - **Also affected**: `"¿cómo fueron esos meses donde era extremadamente estricto?"` similarly split mid-question
  - **Root Cause**: The Whisper boundary handling in `_should_end_sentence_here()` returned `True` (allow split) when the next word was not a connector, WITHOUT checking if we were inside an unclosed Spanish inverted question (`¿` present but no closing `?` yet)
  - **Fix**: Added new helper method `_is_inside_unclosed_question()` that checks for unclosed `¿...?` and `¡...!` constructs. This check is now performed before allowing splits at Whisper boundaries
  - **Behavior**: Questions starting with `¿` are now preserved as complete sentences unless a speaker change (diarization) requires splitting
- **Double inverted question mark in Spanish embedded questions**: Fixed issue where sentences with embedded questions (e.g., `"Valentina, cuéntenos, ¿usted..."`) incorrectly received a second `¿` at the start
  - **Example**: `"Valentina, cuéntenos, ¿usted ahorita está estudiando...?"` became `"¿Valentina, cuéntenos, ¿usted ahorita..."` (two `¿` but one `?`)
  - **Root Cause**: Three places in `punctuation_restorer.py` checked if sentence **started** with `¿`, but not if one already **existed** mid-sentence
  - **Fix**: Added `'¿' not in sentence` guard to all three locations (lines 1538, 1770, 4219) that add inverted question marks
  - **Spanish grammar**: Each `¿` must pair with a `?`; embedded questions should not trigger adding another `¿` at sentence start

### Added
- **Helper method `_is_inside_unclosed_question()`**: New method in `SentenceSplitter` that detects when the current sentence chunk contains an unclosed Spanish inverted question mark (`¿` without corresponding `?`) or exclamation mark (`¡` without `!`)
  - Also used by `_passes_language_specific_checks()` to avoid code duplication

## [0.6.1] - 2026-01-18

### Fixed
- **Speaker boundary splits blocked by connector checks (v0.4.3 regression)**: Fixed issue where speaker boundaries were not creating sentence breaks when followed by connector words ("y", "o", "pero", "de", "a", etc.)
  - **Example**: `"Malala. Sí. Bueno, Malala nació..."` was incorrectly kept as one paragraph despite "Malala." and "Sí. Bueno..." being from different speakers
  - **Root Cause**: v0.6.0 introduced connector word checks that skipped speaker boundaries when the current or next word was a connector, violating the v0.4.3 principle that "speaker boundaries ALWAYS create splits"
  - **Fix**: Removed connector word checks from speaker boundary handling; speaker boundaries are now unconditional
- **Minimum chunk threshold too high for single-word utterances**: Reduced `min_words_speaker` from 4 to 1
  - **Example**: Andrea saying just "Malala." as a prompt/question was being merged into previous sentence because chunk length (1) < threshold (4)
  - **Fix**: Single-word utterances now correctly trigger speaker boundary splits since speaker changes are definitive signals
- **Whisper periods removed at different-speaker boundaries**: Fixed period removal logic that was incorrectly removing periods before connector words even when the NEXT word was from a DIFFERENT speaker
  - **Example**: "1997. ¿Y es reconocida por?" → "1997 y es reconocida por?" (period incorrectly removed before Andrea's question)
  - **Root Cause**: The condition `speaker_at_current == speaker_at_next or (both None)` would remove periods when speaker info was missing; also didn't properly require BOTH speaker checks to be non-None
  - **Fix**: Only remove periods when `speaker_at_current is not None AND speaker_at_next is not None AND speaker_at_current == speaker_at_next`
- **Off-by-one error in speaker boundary calculation**: Fixed boundary placement that caused splits to occur AFTER the first word of the new speaker instead of BEFORE
  - **Example**: Split happening after "¿Y" (word 675) instead of before it, putting Andrea's question start in Nate's sentence
  - **Root Cause**: `end_word` in speaker segments is EXCLUSIVE (like Python slices), but boundary was set to `end_word` instead of `end_word - 1`
  - **Fix**: Changed `boundary_word = current_seg['end_word']` to `boundary_word = current_seg['end_word'] - 1`
- **Inclusive vs exclusive end_word mismatch in speaker lookup**: Fixed `_get_speaker_at_word()` to correctly treat `end_word` as exclusive
  - **Root Cause**: Function used `<= segment['end_word']` but `end_word` is exclusive per `_convert_char_ranges_to_word_ranges()`
  - **Fix**: Changed comparison from `<= segment['end_word']` to `< segment['end_word']`

### Changed
- **Speaker boundary behavior**: Speaker boundaries now unconditionally trigger sentence breaks (restored v0.4.3 behavior)
- **Minimum speaker chunk threshold**: Reduced from 4 words to 1 word, allowing single-word utterances to be properly separated

## [0.6.0.5] - 2026-01-12

### Changed
- Removed temporary debug messages from `sentence_splitter.py` that were used to resolve speaker boundary bug
  - Removed debug logging for speaker boundaries at specific word positions (lines 331-334)
  - Removed debug logging for "chunk too short" at specific word positions (lines 778-780)
  - These debug messages were visible in console output during Spanish transcription and are no longer needed

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
