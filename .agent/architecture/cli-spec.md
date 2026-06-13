# CLI Specification

argparse arguments, exit codes, logging configuration, and environment variables for `podscripter.py`.

Related: [Pipeline architecture](pipeline.md) | [Troubleshooting history](../troubleshooting/history.md) | [AGENT.md hub](../../AGENT.md)

## CLI flags (argparse in `podscripter.py`)

- `--output_dir <dir>` — required.
- `--language <code>|auto` — default `auto`.
- `--output_format {txt|srt}` — default `txt`.
- `--single` — bypass manual chunking (single-call transcription).
- `--model {tiny,base,small,medium,large,large-v2,large-v3}` — default `medium`. Precedence: CLI > `WHISPER_MODEL` env > default.
- `--translate` — Whisper `task=translate`; punctuation uses English rules.
- `--compute-type {auto,int8,int8_float16,int8_float32,float16,float32}` — default `auto`.
- `--beam-size <int>` — beam size for decoding; default 3.
- `--no-vad` — disable VAD filtering (default: enabled).
- `--vad-speech-pad-ms <int>` — padding in ms when VAD is enabled; default 200.
- `--dump-raw` — also write raw Whisper output to `<basename>_raw.txt` in `--output_dir`.
- `--enable-diarization` — enable speaker diarization (default: disabled).
- `--min-speakers <int>` — minimum speakers for diarization (optional; auto-detect by default).
- `--max-speakers <int>` — maximum speakers for diarization (optional; auto-detect by default).
- `--hf-token <str>` — Hugging Face token for first-time model download.
- `--dump-diarization` — write diarization debug dump to `<basename>_diarization.txt` (requires `--enable-diarization`).
- `--dump-merge-metadata` — write merge provenance to `<basename>_merges.txt`.
- `--quiet` / `--verbose` / `--debug` — mutually exclusive; default `--verbose`. `--debug` shows detailed sentence-splitting decisions including speaker segment tracking.

## Exit codes and typed exceptions

Raise typed exceptions at the source and handle them centrally in the CLI:
- `InvalidInputError`
- `ModelLoadError`
- `TranscriptionError`
- `OutputWriteError`

Exit codes:
- `2` = input error
- `3` = model load error
- `4` = transcription error
- `5` = output write error
- `1` = unexpected error

## Logging

- Use a single logger named `podscripter`, configured in `podscripter.py`.
- Levels controlled by the mutually-exclusive CLI flags:
  - `--quiet` -> ERROR (minimal output)
  - `--verbose` -> INFO (default; informative lifecycle logs)
  - `--debug` -> DEBUG (detailed sentence-splitting decisions, speaker tracking, connector-word evaluations)
- Debug messages use `logger.debug()` and include:
  - Speaker segment conversions (boundaries -> chars -> words)
  - Connector-word evaluations with speaker-continuity checks
  - Sentence-ending decisions at every potential boundary
- Replace ad-hoc prints with `logger.info/warning/error/debug`.
- The SRT path logs a normalization summary (trimmed cues, max/total seconds) when writing subtitles.

## Environment variables

- `HF_HOME` — Hugging Face cache root (avoid deprecated `TRANSFORMERS_CACHE`). See [pipeline caching](pipeline.md#model-caching-strategy).
- `HF_HUB_OFFLINE=1` — prefer offline use with warm caches (avoids 429 rate limits) for tests/runs.
- `HF_TOKEN` — alternative to `--hf-token` for diarization model download. Precedence: CLI flag > environment variable.
- `WHISPER_MODEL` — default model; overridden by the `--model` flag. Precedence: CLI > `WHISPER_MODEL` env > default.
- Test overrides for local iteration:
  - `PODSCRIPTER_TEST_MODEL=small` — speed up development runs.
  - `PODSCRIPTER_TEST_FIXTURES_PATTERN="en/*short*"` — filter audio fixtures.
  - `RUN_DIARIZATION=1` — enable diarization-path tests in the suite.
