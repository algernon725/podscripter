## 🧪 Running the Test Suite

All tests must be run inside Docker with model caches mounted.

### Build the image

```bash
docker build --platform linux/arm64 -t podscripter .
```

### Run all default tests

```bash
docker run --rm --platform linux/arm64 \
  -v $(pwd):/app \
  -v $(pwd)/models/sentence-transformers:/root/.cache/torch/sentence_transformers \
  -v $(pwd)/models/huggingface:/root/.cache/huggingface \
  -v $(pwd)/audio-files:/app/audio-files \
  podscripter python3 /app/tests/run_all_tests.py
```

The default selection runs core English/Spanish/French/German punctuation and splitting tests, plus key checks like environment variables and deprecation warnings.

### Optional groups via environment flags

- `RUN_ALL=1`: run the entire suite
- `RUN_MULTILINGUAL=1`: include extra multilingual tests
- `RUN_TRANSCRIPTION=1`: include longer end-to-end transcription tests
- `RUN_DEBUG=1`: include any debug-focused tests

Example:

```bash
docker run --rm --platform linux/arm64 \
  -e RUN_ALL=1 \
  -v $(pwd):/app \
  -v $(pwd)/models/sentence-transformers:/root/.cache/torch/sentence_transformers \
  -v $(pwd)/models/huggingface:/root/.cache/huggingface \
  -v $(pwd)/audio-files:/app/audio-files \
  podscripter python3 /app/tests/run_all_tests.py
```

### Run a single test file

```bash
docker run --rm --platform linux/arm64 \
  -v $(pwd):/app \
  -v $(pwd)/models/sentence-transformers:/root/.cache/torch/sentence_transformers \
  -v $(pwd)/models/huggingface:/root/.cache/huggingface \
  -v $(pwd)/audio-files:/app/audio-files \
  podscripter python3 /app/tests/test_spanish_questions.py
```

### Notable tests (Spanish-focused by default)

- `tests/test_spanish_embedded_questions.py`: preserves embedded `¿ … ?` mid-sentence
- `tests/test_human_vs_program_intro.py`: human vs program intro + extended lines; token-level F1 thresholds
- `tests/test_spanish_helpers.py`: unit tests for `_es_*` helpers (tags, collocations, merges, pairing, greetings)
- `tests/test_transcribe_helpers.py`: unit tests for transcription helpers (`_split_audio_with_overlap`, `_dedupe_segments`, `_accumulate_segments`)

#### Domain handling and masking
- `tests/test_domain_utils.py`: centralized domain detection/masking; language-aware behavior (e.g., Spanish-only `.de` exclusion), spaced-domain repair, mask→process→unmask roundtrip.
- `tests/test_spanish_false_domains.py`: prevents common Spanish words from being treated as domains (e.g., `uno.de` → `uno. de`, `tratada.de` → `tratada. de`).
- `tests/test_spanish_domains_and_ellipses.py`: preserves single/compound TLDs (`.com`, `.co.uk`, `.com.ar`), ellipsis continuation, and triple merge across sentence boundaries.

#### Sentence assembly edge cases
- `tests/test_sentence_assembly_unit.py`: domain triple-merge (`label.` + `Com.` + `Y …`) and decimal merges (`99.` + `9%`, `121.` + `73`).
- `tests/test_spanish_runon_fix.py`, `tests/test_french_runon_fix.py`, `tests/test_german_runon_fix.py`: run‑on sentence regressions.

#### Output normalization
- `tests/test_srt_normalization.py`: trims lingering SRT cues and enforces timing constraints.
- `tests/test_punctuation_preservation.py`: general punctuation integrity checks.

#### Health and environment
- `tests/test_environment_variables.py`: validates required env vars and cache usage.
- `tests/test_no_deprecation_warning.py`: guards against noisy deprecation warnings.

### Caching and rate limiting

- Always mount model cache volumes to avoid repeated downloads and HTTP 429s
- The suite is designed to operate offline when caches are present (HuggingFace/Whisper/Sentence-Transformers)
