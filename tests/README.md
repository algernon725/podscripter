## Running the Test Suite

All tests must be run inside Docker with model caches mounted.

### Build the image

```bash
docker build -t podscripter .
```

### Run all default tests (core + multilingual)

```bash
docker run --rm \
  -v $(pwd):/app \
  -v $(pwd)/models/sentence-transformers:/root/.cache/torch/sentence_transformers \
  -v $(pwd)/models/huggingface:/root/.cache/huggingface \
  -v $(pwd)/audio-files:/app/audio-files \
  podscripter pytest
```

Runs tests marked `core` and `multilingual` by default (configured in `pyproject.toml`).

### Run all tests including transcription (opt-in)

```bash
docker run --rm \
  -v $(pwd):/app \
  -v $(pwd)/models/sentence-transformers:/root/.cache/torch/sentence_transformers \
  -v $(pwd)/models/huggingface:/root/.cache/huggingface \
  -v $(pwd)/audio-files:/app/audio-files \
  podscripter pytest -m ''
```

### Useful pytest options

| Command | Description |
|---------|-------------|
| `pytest` | Run core + multilingual tests (default) |
| `pytest -m ''` | Run all tests including transcription |
| `pytest -m multilingual` | Only multilingual tests |
| `pytest -m transcription` | Only transcription integration tests |
| `pytest tests/test_spanish_bug_fixes.py` | Run a single test file |
| `pytest -k "question"` | Filter by keyword |
| `pytest -x` | Stop on first failure |
| `pytest --lf` | Re-run only previously failed tests |

### Test markers

Tests are categorized using pytest markers (defined in `pyproject.toml`):

- `@pytest.mark.core` — primary language tests, bug-fix regressions, unit tests (run by default)
- `@pytest.mark.multilingual` — cross-language aggregate tests (run by default)
- `@pytest.mark.transcription` — integration tests requiring models/media files (opt-in)

### Shared test infrastructure

- `tests/conftest.py` — shared fixtures (`MockConfig`, language-specific `SentenceSplitter` instances, `restore_punctuation` wrapper)
- `pyproject.toml` — pytest configuration, marker definitions, and default run options

### Caching and rate limiting

- Always mount model cache volumes to avoid repeated downloads and HTTP 429 errors
- The suite is designed to operate offline when caches are present (HuggingFace / Whisper / Sentence-Transformers)
- To run fully offline with warm caches: pass `-e HF_HUB_OFFLINE=1` to `docker run`
