# Audio fixtures corpus

Real-audio test fixtures for the podscripter transcription pipeline (ASR + diarization + punctuation + formatting). Used by `tests/test_audio_fixtures.py` to catch regressions on real audio across EN, ES, and FR (and future languages).

## Where the audio lives

Audio files are **not** stored in this git repo. They live in the public HuggingFace dataset
[`podscripter-project/test-fixtures`](https://huggingface.co/datasets/podscripter-project/test-fixtures),
pinned to a specific commit revision in [`download.py`](download.py). Only the per-fixture
`.expected.json` metadata files live in git.

When a test first runs, [`download.py`](download.py) downloads (or reuses) the audio in the
existing `HF_HOME` cache (`/root/.cache/huggingface` in the Docker container, mounted from
`models/huggingface/` on the host). This is the same cache that already holds the Whisper and
pyannote models, so no extra mounts are needed.

## Directory layout

```
tests/fixtures/audio/
  README.md                       # this file
  LICENSES.md                     # consolidated CC-BY/CC0 attribution NOTICE
  download.py                     # HF dataset downloader, pinned by revision
  _validate_licensing.py          # core-marker test enforcing license metadata
  en/
    <name>.expected.json          # one metadata file per audio fixture
  es/
    <name>.expected.json
  fr/
    <name>.expected.json
```

Each `<name>.expected.json` references an audio file `<lang>/<name>.<ext>` inside the HF
dataset. The audio file is resolved by `download.py` at test time.

## `.expected.json` schema

Every fixture is described by a single JSON file. The schema below is **enforced** by
[`_validate_licensing.py`](_validate_licensing.py) (license fields) and by
[`tests/test_audio_fixtures.py`](../../test_audio_fixtures.py) (everything else).

### Full example

```json
{
  "language": "en",
  "audio_file": "en/librispeech_61-70968-0000.wav",

  "source": "LibriSpeech test-clean",
  "source_url": "https://www.openslr.org/12",
  "license": "CC-BY-4.0",
  "license_url": "https://creativecommons.org/licenses/by/4.0/",
  "attribution": "LibriSpeech ASR corpus, Panayotov et al., 2015",
  "modifications": "Extracted single FLAC; converted to 16 kHz mono WAV",

  "duration_sec": 10.4,
  "modes": ["single"],

  "expected_text": "Mister Quilter is the apostle of the middle classes and we are glad to welcome his gospel.",
  "speaker_turns": [
    {"start": 0.0, "end": 10.4, "speaker": "A"}
  ],
  "expected_speaker_count": 1,

  "patterns": ["single-speaker", "short"],
  "thresholds": {"wer_max": 0.15, "der_max": 0.20}
}
```

### Field reference

| Field | Required | Type | Notes |
|-------|----------|------|-------|
| `language` | yes | string | ISO 639-1 code (`en`, `fr`, `es`, `de`, ...). Drives `podscripter --language`. |
| `audio_file` | yes | string | Path inside the HF dataset, e.g. `en/foo.wav`. Must exist post-download. |
| `source` | yes | string | Human-readable corpus name. |
| `source_url` | yes | string | Canonical URL of the source corpus. |
| `license` | yes | enum | One of `CC-BY-4.0`, `CC0-1.0`, `public-domain`. Enforced by validator. |
| `license_url` | yes | string | Canonical URL of the license text. |
| `attribution` | yes (CC-BY) | string | Citation-style attribution. Required for any CC-BY source. |
| `modifications` | yes (CC-BY) | string | Description of trimming/resampling/concat applied. Required for any CC-BY source. |
| `duration_sec` | yes | number | Clip duration in seconds (post-trim). |
| `modes` | no | array<string> | Subset of `["single", "chunked"]`. Default `["single"]`. Long fixtures (>~8 min) should include `"chunked"` to exercise `_split_audio_with_overlap` / `_dedupe_segments` / `_accumulate_segments`. |
| `expected_text` | yes | string | Verbatim reference transcript. WER is computed against this. |
| `speaker_turns` | no | array<obj> | List of `{start, end, speaker}` for diarization DER. Required when `expected_speaker_count > 1`. |
| `expected_speaker_count` | yes | int | How many distinct speakers diarization should find. |
| `patterns` | no | array<string> | Tags driving pattern-specific assertions (e.g. `"questions"` asserts `?` present, `"url"` asserts no broken URL split). |
| `thresholds` | yes | obj | WER/DER thresholds. See "Threshold shape" below. |

### Threshold shape

Two equivalent forms are supported:

**Flat (same threshold for every mode):**

```json
"thresholds": {"wer_max": 0.15, "der_max": 0.20}
```

**Per-mode (lets chunked mode use a looser bound for boundary artifacts):**

```json
"thresholds": {
  "single":  {"wer_max": 0.15, "der_max": 0.20},
  "chunked": {"wer_max": 0.17, "der_max": 0.22}
}
```

The test reads per-mode if present and falls back to the flat shape otherwise.

### Pattern tags

Recognized values for `patterns` (each triggers an extra assertion in the test):

| Tag | Assertion |
|-----|-----------|
| `single-speaker` | `expected_speaker_count == 1` |
| `multi-speaker` | `expected_speaker_count >= 2` |
| `questions` | Output contains at least one `?` |
| `spanish-questions` | Output contains at least one `¿ ... ?` pair |
| `names` | (informational; no automated assertion yet) |
| `url` | Output does not split a URL across whitespace (no ` . ` followed by a known TLD) |
| `numbers` | (informational; no automated assertion yet) |
| `short` | (informational; clip < 30s) |
| `long` | (informational; clip > 8 min, eligible for `"chunked"` mode) |

Unknown tags are ignored without error so new tags can be added incrementally.

## Adding a new fixture

1. Pick a CC-BY/CC0 source (see [LICENSES.md](LICENSES.md) for the allowlist).
2. Trim the clip with `ffmpeg -ss <start> -to <end> -ac 1 -ar 16000 ...`. For long clips (>~8 min) prefer FLAC over WAV (~half the size).
3. Push the clip to the HF dataset under `<lang>/<name>.<ext>`. Update the dataset's `README.md` provenance table.
4. Bump `HF_REVISION` in [`download.py`](download.py) to the new dataset commit hash.
5. Add `tests/fixtures/audio/<lang>/<name>.expected.json` following the schema above. Include all required license fields.
6. If you're adding a long fixture to exercise chunked mode, set `"modes": ["single", "chunked"]`.
7. Run `pytest tests/fixtures/audio/_validate_licensing.py` (core marker) to verify license metadata is well-formed.
8. Run `pytest -m transcription -k <name>` to verify the pipeline produces output within thresholds.

See [LICENSES.md](LICENSES.md) for the consolidated NOTICE file and current source allowlist.
