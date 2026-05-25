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

## EN/ES/FR audio fixtures (Tier 1 regression)

`tests/test_audio_fixtures.py` runs the full pipeline (ASR + diarization + punctuation +
formatting) on a small corpus of real audio for English, Spanish, and French. Each fixture
is described by a JSON file under `tests/fixtures/audio/<lang>/<name>.expected.json` and the
matching audio lives in the public HuggingFace dataset
[`podscripter-project/test-fixtures`](https://huggingface.co/datasets/podscripter-project/test-fixtures),
pinned to a specific revision in `tests/fixtures/audio/download.py`. See
[`tests/fixtures/audio/README.md`](fixtures/audio/README.md) for the schema and
[`tests/fixtures/audio/LICENSES.md`](fixtures/audio/LICENSES.md) for source attribution.

These tests are gated by `@pytest.mark.transcription` and use the same flags as a typical
manual podscripter run:

- `enable_diarization=True`
- `model_name="medium"` (override with `PODSCRIPTER_TEST_MODEL=small` for faster local dev)
- `beam_size=3`
- `single_call=True` for short fixtures; long fixtures (with `"modes": ["single","chunked"]`)
  exercise both code paths

```bash
docker run --rm \
  -v $(pwd):/app \
  -v $(pwd)/models/sentence-transformers:/root/.cache/torch/sentence_transformers \
  -v $(pwd)/models/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN \
  podscripter pytest -m transcription tests/test_audio_fixtures.py
```

The same HuggingFace cache mount that already holds Whisper and pyannote models also holds
the test audio (downloaded by `tests/fixtures/audio/download.py` on first run). The
pinned revision keeps audio + tests in lockstep across `git checkout`s.

Useful environment overrides:

| Env var | Effect |
|---|---|
| `PODSCRIPTER_TEST_MODEL` | Override `medium` (e.g. `small`, `tiny`) for faster local iteration |
| `PODSCRIPTER_TEST_FIXTURES_PATTERN` | Glob filter, e.g. `en/*short*` to run a subset |
| `HF_HUB_OFFLINE=1` | Reuse cached audio + models without contacting HF |

### License validator (always runs)

`tests/fixtures/audio/_validate_licensing.py` is a `core`-marker test that enforces every
`.expected.json` declares CC-BY 4.0 / CC0 / public-domain metadata in the documented
shape, even on a normal `pytest` run with no audio downloaded. This prevents accidental
ingestion of an NC/ND-licensed clip in a future PR.

## Tier 2 quality benchmarks (not run per-PR)

`tests/benchmarks/` runs larger public-dataset subsets and tracks WER / DER over time
against a committed baseline. See [`tests/benchmarks/README.md`](benchmarks/README.md)
for the dataset list, runner, and `compare_baseline.py` regression gate. Intended for
nightly CI or pre-release validation, not per-PR.

## Tier 3 bug-reproduction fixtures

When you hit a specific EN/ES/FR (or other-language) bug worth pinning regression coverage
for, follow this convention so the bug fixture lives in the same corpus as Tier 1:

1. Trim the offending audio to the shortest clip that reproduces the bug:
   ```bash
   ffmpeg -ss <start> -to <end> -ac 1 -ar 16000 \
       -i source.wav tests/fixtures/audio/<lang>/bug_<short_name>.wav
   ```
   (For clips > 8 min, prefer `.flac`.)
2. Push the clip to the
   [`podscripter-project/test-fixtures`](https://huggingface.co/datasets/podscripter-project/test-fixtures)
   HF dataset under `<lang>/bug_<short_name>.<ext>`. Update the dataset's `README.md`
   provenance table with the source URL, license, attribution, and modifications.
3. Bump `HF_REVISION` in `tests/fixtures/audio/download.py` to the new commit SHA.
4. Add `tests/fixtures/audio/<lang>/bug_<short_name>.expected.json` documenting the
   *correct* expected behavior and the failure being prevented.
5. Add a focused test file `tests/test_<lang>_bug_<short_name>.py` mirroring the existing
   Spanish bug-fix pattern in
   [`tests/test_episodio272_speaker_split_exclamation.py`](test_episodio272_speaker_split_exclamation.py).

The `HF_REVISION` bump must land in the same PR as the new `.expected.json` and test
file, so a `git checkout` of any commit fetches exactly the audio its tests expect.

**Provenance for bug fixtures**: clips trimmed from your own production audio (e.g.
Españolistos episodes) still need a documented license. If you're the copyright holder,
mark them `license: CC-BY-4.0` (or any permissive license you choose) in the fixture
metadata and add yourself to `tests/fixtures/audio/LICENSES.md`. If clipped from
third-party podcasts, only include them if the source is CC-BY/CC0 — otherwise reproduce
the bug from a CC-licensed source instead.
