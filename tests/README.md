# Punctuation Restoration Tests

This directory contains tests for punctuation restoration across the primary languages (English, Spanish, French, German) plus a set of general and experimental tests.

## How to Run

Always run tests inside Docker with model caches mounted.

Run primary tests (default selection):
```bash
docker run --rm --platform linux/arm64 \
  -v $(pwd)/models/whisper:/app/models \
  -v $(pwd)/models/sentence-transformers:/root/.cache/torch/sentence_transformers \
  -v $(pwd)/models/huggingface:/root/.cache/huggingface \
  -v $(pwd):/app \
  podscripter python3 /app/tests/run_all_tests.py
```

Include multilingual aggregate tests:
```bash
docker run --rm --platform linux/arm64 \
  -e RUN_MULTILINGUAL=1 \
  -v $(pwd)/models/whisper:/app/models \
  -v $(pwd)/models/sentence-transformers:/root/.cache/torch/sentence_transformers \
  -v $(pwd)/models/huggingface:/root/.cache/huggingface \
  -v $(pwd):/app \
  podscripter python3 /app/tests/run_all_tests.py
```

Include transcription integration tests:
```bash
docker run --rm --platform linux/arm64 \
  -e RUN_TRANSCRIPTION=1 \
  -v $(pwd)/models/whisper:/app/models \
  -v $(pwd)/models/sentence-transformers:/root/.cache/torch/sentence_transformers \
  -v $(pwd)/models/huggingface:/root/.cache/huggingface \
  -v $(pwd):/app \
  podscripter python3 /app/tests/run_all_tests.py
```

Run everything (debug/bench included):
```bash
docker run --rm --platform linux/arm64 \
  -e RUN_ALL=1 \
  -v $(pwd)/models/whisper:/app/models \
  -v $(pwd)/models/sentence-transformers:/root/.cache/torch/sentence_transformers \
  -v $(pwd)/models/huggingface:/root/.cache/huggingface \
  -v $(pwd):/app \
  podscripter python3 /app/tests/run_all_tests.py
```

## What gets selected by default

- Primary language suites:
  - `test_english_*`, `test_french_*`, `test_german_*`, `test_spanish_*`
- Core/general checks:
  - `test_improved_punctuation.py`, `test_punctuation.py`, `test_environment_variables.py`,
    `test_no_deprecation_warning.py`, `test_past_tense_questions.py`, `test_punctuation_preservation.py`
  - Spanish human-vs-program similarity: `test_human_vs_program_intro.py` (Spanish only)
  - Spanish embedded questions: `test_spanish_embedded_questions.py` (Spanish only)

## Optional groups

- Multilingual aggregates: `test_multilingual_*` (enable with `RUN_MULTILINGUAL=1`)
- Transcription integration: `test_transcription.py`, `test_transcription_logic.py` (enable with `RUN_TRANSCRIPTION=1`)
  - Human-vs-program intro/extended similarity (Spanish): `test_human_vs_program_intro.py` (included by default)
- Debug/bench/experimental: `test_question_detection_debug.py`, `test_transcription_debug.py`,
  `test_transcription_specific.py`, `model_comparison.py`, `test_model_change.py` (enable with `RUN_DEBUG=1` or `RUN_ALL=1`)

## Maintenance notes

- We focus on English, Spanish, French, and German. Other language tests may be combined into the multilingual group for lighter maintenance.
- Prefer broad, reusable assertions over one-off cases. If two tests overlap heavily, consolidate into a single parameterized test file.