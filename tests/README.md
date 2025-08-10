## 🧪 Running the Test Suite

All tests must be run inside Docker with model caches mounted.

### Build the image

```bash
docker build --platform linux/arm64 -t podscripter .
```

### Run all default tests

```bash
docker run --rm --platform linux/arm64 \
  -e NLP_CAPITALIZATION=1 \
  -v $(pwd):/app \
  -v $(pwd)/models/whisper:/app/models \
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
  -v $(pwd)/models/whisper:/app/models \
  -v $(pwd)/models/sentence-transformers:/root/.cache/torch/sentence_transformers \
  -v $(pwd)/models/huggingface:/root/.cache/huggingface \
  -v $(pwd)/audio-files:/app/audio-files \
  podscripter python3 /app/tests/run_all_tests.py
```

### Run a single test file

```bash
docker run --rm --platform linux/arm64 \
  -v $(pwd):/app \
  -v $(pwd)/models/whisper:/app/models \
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

### Caching and rate limiting

- Always mount model cache volumes to avoid repeated downloads and HTTP 429s
- The suite is designed to operate offline when caches are present (HuggingFace/Whisper/Sentence-Transformers)
