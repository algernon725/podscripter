# Troubleshooting History

Bug history, language edge cases, open limitations, and testing quirks for podscripter. Condensed; version tags and test file names preserved for traceability.

Related: [Pipeline architecture](../architecture/pipeline.md) | [CLI spec](../architecture/cli-spec.md) | [AGENT.md hub](../../AGENT.md) | [CHANGELOG.md](../../CHANGELOG.md)

## Language edge cases

### German (experimental; code paths and tests retained)
German was previously a primary language, now experimental. German-specific code paths and tests remain so `--language de` and Whisper auto-detect keep working:
- Preposition guards (no sentence end on `zu, an, auf, aus, bei, mit, nach, von, vor, in, für, ...`).
- Auxiliary verbs / continuative verbs (imperfect `war, hatte, ging, machte`; modals `konnte, wollte, musste, sollte`; infinitives `sein, haben, werden, gehen, machen`; present `ist, sind, hat, wird`; preterite `wurde, wurden`).
- Past-participle detection (`ge-...-t` / `ge-...-en`) for "auxiliary + past participle" merges.
- Insert commas before subordinating conjunctions (`dass|weil|ob|wenn`) when safe.
- Greeting commas (`Hallo, ...`); capitalize `Ich` after punctuation; capitalize `Herr/Frau + Name`; minimal noun capitalization after determiners; small proper-noun whitelist.

### Spanish
- Preserve embedded questions mid-sentence: keep/pair `¿ … ?` inside larger sentences; do not strip mid-sentence `¿`.
- Coordinated yes/no questions: verb-initial start with later `o <verbo>`.
- Greeting/lead-in commas (guarded): add comma after greetings ("Hola …,") and set-phrases ("Como siempre,"); do NOT add comma after `Hola` when followed by prepositions like `a`/`para` ("Hola para todos"); avoid duplicate commas before an existing `¿`/`¡`; the remove-comma-after-Hola rule also applies when preceded by `¿`/`¡`.
- Diacritic-insensitive gating where appropriate (`como` ~ `cómo`).
- Appositive introductions: "Yo soy <Nombre>, de <Ciudad>, <País>".
- Soft sentence splitting: avoid breaking inside entities; merge `auxiliar + gerundio` and possessive splits ("tu español").
- spaCy capitalization runs BEFORE greeting/lead-in comma insertion (pipeline-order correction to avoid feedback loops).
- Do not split on ellipses mid-clause.
- Domain protection (single + compound TLDs). Supported: single `com|net|org|co|es|io|edu|gov|uk|us|ar|mx|de|fr|br|ca|au`; compound `co.uk|com.ar|com.mx|com.br|com.au|co.jp|co.in|gov.uk|org.uk|ac.uk`. Removed for false-positive prevention: `.it` (English "it"), `.nl`, `.jp`, `.cn`, `.in` (English "in"), `.ru`.
- Spanish-only `.de`/`.es` exclusion: not treated as domains in Spanish (collide with "de" preposition and "es" verb), e.g. `tratada.de` -> `tratada. de`, `naturales.es` -> `naturales. es`. Centralized exclusion lists.
- Normalize mismatched inverted punctuation: leading `¡` with trailing `?` -> proper `¿...?`.
- Cross-language location-appositive normalization (EN/ES/FR/DE): convert erroneous period in ", <preposition> <Location>. <Location>" into a comma; uses language-specific prepositions (ES: de; EN: from/in; FR: de/du/des; DE: aus/von/in). New-sentence guard: don't merge when the next fragment starts a new subject ("Y yo …", "And I …", "Et je …", "Und ich …"). Appositive-merge safety guard: only merge when the second sentence is a minimal location continuation.

### French
- Clitic hyphenation for inversion: `allez-vous`, `est-ce que`, `qu'est-ce que`, `y a-t-il`, `va-t-il`.
- `_fr_merge_short_connector_breaks` handles short-connector merges.

### English / French / German
- Greeting commas (`Hello, ...`, `Bonjour, ...`, `Hallo, ...`); capitalize sentence starts.

## Resolved bugs

- Coordinating-conjunction split (Fixed): early guard in `_should_end_sentence_here()` prevents ending on conjunctions (ES y/e/o/u/pero/mas/sino; EN and/but/or/nor/for/so/yet; FR et/ou/mais/donc/or/ni/car; DE und/oder/aber/denn/sondern). Tests: `test_conjunction_split_bug.py`, `test_episodio190_y_eco_bug.py`.
- Spanish question split (Fixed): guard against splitting when current word contains `¿` or chunk has unclosed `¿`…`?`. Tests: `test_pues_que_split_bug.py`.
- Preposition split (Fixed): added preposition guards across ES/EN/FR/DE (ES added `a`, `ante`, `bajo`). Tests: `test_preposition_split_bug.py`, `test_preposition_split_long_text.py`.
- Continuative/auxiliary verb split (Fixed): `CONTINUATIVE_AUXILIARY_VERBS` set in `sentence_splitter.py` prevents ending on incomplete verbs across ES/EN/FR/DE. v0.6.3 additions: number + time/measure unit ("a los 18. Años" -> "a los 18 años"); auxiliary + past participle ("fueron. Dirigidos" -> "fueron dirigidos") via `_is_past_participle()` (ES -ado/-ido, EN -ed/-en, FR -é/-i/-u, DE ge-...-t/ge-...-en). Tests: `test_continuative_verb_split_bug.py`.
- Speaker diarization short-segment (Fixed): moved speaker/Whisper boundary checks before `min_chunk_before_split`; speaker boundaries use 1-word threshold (v0.6.1); added `_convert_speaker_timestamps_to_char_positions()`. Tests: `test_speaker_boundary_conversion.py`.
- Connector word with diarization (Fixed v0.3.1): sentences no longer start with connectors when the same speaker continues. Added `_convert_speaker_segments_to_char_ranges()`, `_convert_char_ranges_to_word_ranges()`, `_get_speaker_at_word()`. Fixed three re-splitting sites (Spanish post-processing, `restore_punctuation()` return tuple, `_write_txt(skip_resplit=True)`). Verified with Episodio212.mp3.
- False domain merge in natural language (Fixed v0.4.4): natural-language guards (merge only if previous sentence < 50 chars OR capitalized label). Verified with Episodio212.mp3.
- Whisper boundary skipping too aggressive (Fixed v0.4.3): lookahead reduced 15 -> 3 words; only skip if next word is connector or lowercase. Verified Episodio212-trim.mp3.
- Speaker changes with connector words (Fixed v0.4.3): speaker boundaries ALWAYS split, regardless of next connector. Verified Episodio212-trim.mp3.
- Whisper-added periods at skipped boundaries (Fixed v0.4.2, refined v0.4.3): track `skipped_whisper_boundaries`, remove trailing periods at skipped positions. Tests: `test_whisper_skipped_boundary_detailed.py`, `test_whisper_boundary_debug.py`.
- Speaker change separation (Fixed v0.4.2, superseded by v0.5.2): fixed boundary extraction, lowered `MIN_SPEAKER_SEGMENT_SEC` 2.0s -> 0.5s, rewrote speaker assignment by overlap. Verified Episodio212.mp3.
- Period before same-speaker connectors (Fixed v0.4.0): solved by the `SentenceSplitter` consolidation (tracks Whisper periods, removes before same-speaker connectors). "trabajo y este" instead of "trabajo. Y este".
- Speaker boundary splits blocked by connector checks (Fixed v0.6.1): removed connector checks blocking speaker splits; `min_words_speaker` -> 1; stricter period removal; fixed `boundary_word = end_word - 1`; `_get_speaker_at_word()` uses `< end_word`. Verified Episodio218-trim.mp3.
- Spanish inverted question split at Whisper boundaries (Fixed v0.6.2): added `_is_inside_unclosed_question()` (also protects `¡…!`). Verified Episodio221.mp3.
- Double inverted question mark in embedded questions (Fixed v0.6.2): added `'¿' not in sentence` guard at three sites. Verified Episodio221.mp3.
- Questions/exclamations merged with connectors (Fixed v0.6.3): only periods trigger connector merging (`endswith('.')`); `?`/`!` preserved. Verified Episodio225.mp3.
- Semantic split preempting nearby Whisper boundary (Fixed v0.7.1): Whisper-boundary lookahead before semantic splits; new threshold `semantic_whisper_lookahead` (default 8). Tests: `test_semantic_whisper_lookahead.py`.
- Spanish exclamation/question artifacts after speaker splits — `Bueno,!` (Fixed v0.8.4): expanded rstrip classes in `_apply_semantic_punctuation()` to `'.!,;: '` / `'.?,;: '`. Tests: `test_trailing_comma_bug.py`.
- Spanish proper-noun + `Es` falsely merged as `.es` domain — `Nate.es` (Fixed v0.8.5): in `SentenceFormatter._merge_domains()`, when `language=='es'` AND `tld=='es'` AND `is_capitalized_label`, skip the merge (logged). Lowercase brand labels like "marca.es" still merge. Tests in `tests/test_sentence_formatter.py`.
- Person initials normalization (WIP/partial): `_normalize_initials_and_acronyms()` collapses "C. S. Lewis" -> "C.S. Lewis", "U. S. A." -> "USA". EN acronyms work; non-English splits still occur because spaCy re-adds spaces. Next step: initial-masking system (like domain masking). Tests: `test_initials_normalization.py`.
- Tier 1 audio fixture test assumed `sentences` is `list[str]` (Fixed, ES MVP May 2026): `tests/test_audio_fixtures.py` raised `AttributeError: 'Sentence' object has no attribute 'strip'` after v0.6.0 made `result["sentences"]` a `list[Sentence]`. Fixed with `hasattr(s, "text")` extraction.
- Inert Spanish post-sanitization removed (Fixed v0.10.3): `_assemble_sentences()` (`podscripter.py`) applied the `str`-typed `_sanitize_sentence_output()` to the `Sentence` objects from `SentenceFormatter.format()`; `mask_domains()` raised `TypeError` and was swallowed by `except Exception: return s`, so the Spanish-only cleanups were a no-op for every language. Verified that enabling it (feeding `.text`) would REGRESS Spanish — the proper-noun guard uses ASCII `[a-z]`, lowercasing accented/`ñ` proper nouns after a comma (`México`→`méxico`, `José`→`josé`, `España`→`españa`, `Perú`→`perú`, `Bogotá`→`bogotá`), while non-accented names (`Andrea`, `Nate`) are preserved, and the intra-word-period fix mis-orders into a false split (`vendedores.ambulantes`→`vendedores. Ambulantes`). Deleted the helper + its two call sites (and the adjacent `fr` `pass` no-op) rather than repair. Concurrently corrected the sentences-flow types to honest `list[Sentence]` end-to-end (`restore_punctuation`/`_advanced_punctuation_restoration`/`_transformer_based_restoration`/`_assemble_sentences`, `TranscriptionResult.sentences`) and removed 11 redundant in-function `from sentence_splitter import Sentence` imports. Tests: full suite green incl. `tests/test_spanish_capitalization_domain_regression.py`.

### Speaker segment splitting saga (v0.5.2 - v0.5.2.3, 2025-01-05)
Rewrote `_convert_speaker_segments_to_char_ranges()` in `podscripter.py`:
- v0.5.2: split Whisper segments that contain multiple speakers (detect ALL overlapping speakers, split proportionally at word boundaries, merge consecutive same-speaker ranges). Restored 84 -> 84 boundaries (was 84 -> 48).
- v0.5.2.1: filter on overlap duration (`overlap_duration < 0.3`) instead of total segment duration (`spk_duration < 0.5`). Preserves short legit utterances like "Bueno!" (0.46s).
- v0.5.2.2: dominant-speaker threshold — if one speaker > 80% of a segment, assign the whole segment (fixes 14%/86% edge misattributions). Too aggressive: lost 16 boundaries (84 -> 68).
- v0.5.2.3: apply dominant-speaker only at edges (first/last 10%); middle utterances ("Ok.") preserved.
Key lessons: align char positions with the normalized text downstream uses; `SentenceFormatter` operates on word indices; filter the relevant duration; trust dominant signal only at edges; edge anomalies are likely errors, middle anomalies likely real. Verified Episodio213-trim.mp3 / Episodio213.mp3.

## Refactor history (consolidation milestones)

- Sentence splitting consolidation (v0.4.0): all splitting logic moved into a single `SentenceSplitter` class (`sentence_splitter.py`); replaced scattered logic across `_semantic_split_into_sentences()`, Spanish post-processing, `assemble_sentences_from_processed()`, and `_write_txt()`. `restore_punctuation()` accepts `whisper_segments` and ALWAYS returns `(text, sentences_list)`. Provenance via `Sentence` dataclass (`split_reason`, `confidence`, `start_word`, `end_word`, `speaker`). Tests: `tests/test_sentence_splitter_unit.py`.
- Post-processing merge consolidation (v0.5.0): all merge logic moved into `SentenceFormatter` (`sentence_formatter.py`); enforces "never merge different speakers"; `MergeMetadata` provenance; `--dump-merge-metadata`. Tests: `tests/test_sentence_formatter.py`. Clean separation: `SentenceSplitter` = boundaries, `SentenceFormatter` = formatting.
- Speaker-aware output (v0.6.0): `Utterance` + `Sentence` dataclasses (`has_speaker_changes()`, `get_first_speaker()`); `SentenceSplitter.split()` returns `List[Sentence]`; `_write_txt()` adds an extra paragraph break (`\n\n\n`) on speaker change. Tests: `test_speaker_aware_output.py`.
- Whisper segment boundary integration (2025): see [4-signal hybrid splitting](../architecture/pipeline.md#sentence-splitting-4-signal-hybrid). Tests: `tests/test_whisper_boundary_integration.py`.
- Centralized comma spacing (2025): `_normalize_comma_spacing(text)` replaced inline variants (trade-off: "1,000" -> "1, 000").

## Open limitations

- Remaining xfail tests — NLP output drift (OPEN, v0.8.2): ~33 xfail invocations across 5 files; per-parameter xfails. Sentence splitting/formatting (31, MEDIUM): `test_english_sentence_splitting.py` (9), `test_german_sentence_splitting.py` (8), `test_spanish_sentence_splitting.py` (7), `test_french_sentence_splitting.py` (7). Inverted-question-mark fusion cleanup (2, LOW): `test_spanish_bug_fixes.py` bug3.
- Question detection — verb-first/implicit questions (CLOSED — accepted limitation, v0.10.1): 50 unrealistic/mislabeled xfails retired in v0.10.1; no detection code changed. Production passes Whisper output with native `¿`/`?` preserved, so verb-first questions usually arrive punctuated; `restore_punctuation()` only normalizes/pairs (`_es_pair_inverted_questions`). Text-only disambiguation is inherently ambiguous; starter lists (`ES_QUESTION_STARTERS_EXTRA`, `has_question_indicators()`) intentionally not used as hard triggers (would regress negatives like "necesito más información"). Only meaningful path is acoustic/prosodic signal.
- Diarization misalignment causing sentence fragments (OPEN, v0.6.1): pyannote misattributes brief interjections to the previous speaker; proportional char-splitting + period removal create fragments ("Ajá y ella."). Code behaves correctly given flawed input; increasing `min_words_speaker` would regress the "Malala." fix. Verified Episodio218.mp3. No code change.
- Short speaker-segment filtering causing missed splits (OPEN, v0.6.1): `MIN_SEGMENT_DURATION = 1.3s` (podscripter.py line ~851) filters legit ~1.0-1.3s segments at transitions (e.g., "Tauromaquia." at 1.28s misses by 0.02s). Lowering risks re-introducing rapid-flip artifacts (v0.6.0.1). Best fix is context-aware filtering (preserve shorts flanked by different speakers). Verified Episodio220.mp3 / Episodio282.
- torchaudio.load() deprecation (KNOWN — dependency upgrade): pyannote.audio 4.x uses torchcodec, which mishandles MP3 frame boundaries. Workaround pre-loads audio with `torchaudio.load()` (soundfile backend via `libsndfile1` + `soundfile` pip) and passes a `{"waveform", "sample_rate"}` dict. Pinned `torchaudio==2.8.0`; warning suppressed in `speaker_diarization.py`. On upgrade past 2.8.0, revisit audio loading. Files: `speaker_diarization.py`, `Dockerfile`.

## Testing quirks

- All tests use pytest and must run inside Docker with model caches mounted (avoids 429s). Use `python3` (not `python`).
- Shared infra in `tests/conftest.py` (handles `sys.path`; provides `MockConfig` and fixtures `mock_config`, `es_splitter`, `en_splitter`, `de_splitter`, `fr_splitter`). pytest config in `pyproject.toml`.
- Markers: `@pytest.mark.core` (primary-language, bug-fix, unit — default), `@pytest.mark.multilingual` (cross-language aggregate — default), `@pytest.mark.transcription` (integration needing models/media — opt-in).
- Running: default `pytest`; all `pytest -m ''`; multilingual `pytest -m multilingual`; transcription `pytest -m transcription`; single file `pytest tests/test_spanish_bug_fixes.py`; keyword `pytest -k "question"`; stop-on-first `pytest -x`; rerun failures `pytest --lf`.
- xfail semantics: xfails show as `xfail` (not `FAILED`); a previously-xfail test passing without code change reports `XPASS` (informational). When fixing, remove the `@pytest.mark.xfail` decorator / `marks=...` from the `pytest.param`.
- Key test files: `test_sentence_assembly_unit.py`, `test_chunk_merge_helpers.py` (`_dedupe_segments`/`_accumulate_segments`), `test_spanish_embedded_questions.py`, `test_human_vs_program_intro.py` (F1 intro ≥ 0.80, overall ≥ 0.70), `test_spanish_helpers.py`, `test_spanish_domains_and_ellipses.py`, `test_spanish_false_domains.py`, `test_domain_utils.py`, `test_initials_normalization.py`, `test_audio_fixtures.py`.

### Three-tier EN/ES/FR audio corpus
Documented in `tests/README.md` and `tests/fixtures/audio/README.md`.
- Tier 1 — regression (`tests/test_audio_fixtures.py`, marker `transcription`): per-clip `.expected.json` under `tests/fixtures/audio/<lang>/` (`en/`, `es/`, `fr/`), audio in public HF dataset `podscripter-project/test-fixtures`, pinned via `HF_REVISION` in `tests/fixtures/audio/download.py`. Fixtures run with `enable_diarization=True`, `model_name="medium"`, `beam_size=3`, `single_call=True`; long fixtures (`"modes": ["single","chunked"]`) also run chunked. Default short-fixture thresholds WER ≤ 0.15, DER ≤ 0.20. The ES/FR long MLS concats (`mls_es_two_speakers_long.flac`, `mls_fr_two_speakers_long.flac`) use looser bounds (WER ≤ 0.17/0.19, DER ≤ 0.22/0.25 single/chunked). Spanish fixtures also assert the `spanish-questions` pattern (both `¿` and `?` present). Expect ~80-90 min on CPU at `medium` for the full sweep.
- Tier 2 — quality benchmark (`tests/benchmarks/`): pipeline over bounded public subsets (FLEURS en_us/es_419/fr_fr + MLS Spanish/French streamed) vs committed `tests/benchmarks/baseline.json` (75 clips, `medium`, 15/dataset). WER-only by design (all Tier 2 datasets single-speaker). Manual ad-hoc drift gate (not per-PR, not CI); deterministic decoding -> bit-identical WER across clean runs. `compare_baseline.py` flags regressions; tolerance via `--wer-tolerance`.
- Tier 3 — bug fixtures: trim offending audio, push to the HF dataset, bump `HF_REVISION`, add `.expected.json` + focused test (mirrors `tests/test_episodio272_speaker_split_exclamation.py`).
- License compliance: only `CC-BY-4.0`, `CC0-1.0`, `public-domain` clips. `tests/fixtures/audio/_validate_licensing.py` (marker `core`, no download) enforces metadata + allowlist every PR. Aggregate license CC-BY 4.0; see `tests/fixtures/audio/LICENSES.md`.
- Bring-up: rebuild image (`docker build -t podscripter .` — required when Dockerfile changes, e.g. v0.9.0 `jiwer` add); prime cache (`python -m tests.fixtures.audio.download`); run `pytest`; opt into `pytest -m transcription tests/test_audio_fixtures.py` (needs `HF_TOKEN`). Local overrides: `PODSCRIPTER_TEST_MODEL=small`, `PODSCRIPTER_TEST_FIXTURES_PATTERN`, `HF_HUB_OFFLINE=1`.

### Corpus milestones (HF dataset revisions)
- v0.9.3: MLS ES + FR long two-speaker concats (16 kHz mono FLAC, 14+14 utts, 0.5s inter-speaker silence). HF commit `d007be782d831a1471dc51ef67e4c681dabe1a94`. License validator 23 -> 25 parametrizations.
- v0.9.4: EN AMI conversational short (`en/ami_en2001b_1993_2020.wav`, ~25.85s, EN2001b Mix-Headset, thresholds wer 0.20/der 0.25) + FR MLS two-speaker short (`fr/mls_fr_two_speakers_short.wav`, ~27.02s, thresholds wer 0.15/der 0.25). VoxPopuli FR slot replaced by MLS concat; url-pattern fixture attempted and dropped (WER ~0.44 > 0.25 cap). HF revision `6e1c1eced6e68bb35b1f1d89c56478229201a2f7` (26 fixtures).
- v0.10.0: Tier 2 WER baseline populated (75 clips) + MLS ES/FR streamed via `datasets` (`streaming=True`, `Audio(decode=False)`); required adding `datasets` to Dockerfile; `download_subsets.py` uses `os._exit(rc)` to avoid torchcodec/aiohttp teardown segfault (exit 139). Per-dataset mean WER: FLEURS en 0.077 / es 0.000 / fr 0.059; MLS es 0.028 / fr 0.099.

### De-scoped / deferred corpus axes
- Common Voice CC0 shorts (DE-SCOPED/optional): CC0 splits are tens of GB; clip selection benefits from listening. Selection from `validated.tsv` (`up_votes >= 2 AND down_votes == 0`, 5-15s); re-encode `ffmpeg -i <in.mp3> -ac 1 -ar 16000 -c:a pcm_s16le <out.wav>`; thresholds wer 0.17/der 0.22.
- Tier 2 optional axes (DEFERRED): DER metric (`DiarizationErrorRate(collar=0.25, skip_overlap=False)`), multi-speaker chunked mode, real VoxPopuli FR download, LibriSpeech EN enumeration, `sentence-F1` metric, nightly CI (deliberately not pursued — maintainer runs Tier 2 ad-hoc ~1-2×/month).
