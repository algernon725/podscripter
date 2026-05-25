# podscripter quality benchmarks (Tier 2)

Larger benchmark suites that quantify pipeline quality over time. Unlike the Tier 1 audio
fixtures (see [`tests/fixtures/audio/`](../fixtures/audio/) and
[`tests/test_audio_fixtures.py`](../test_audio_fixtures.py)), these are **not** run on every
PR. They:

- Pull tens of minutes of audio per language from public datasets (~30 min/lang typical)
- Compute WER (ASR), DER (diarization), and sentence-F1 (against human transcripts)
- Run multi-speaker subsets in **both** `--single` and chunked modes to surface
  chunked-mode-specific regressions separately from generic ASR drift
- Compare against a committed [`baseline.json`](baseline.json) so historical drift is
  visible in PR diffs

## Layout

```
tests/benchmarks/
  README.md                this file
  download_subsets.py      pulls public dataset subsets into a configurable cache
  run_benchmark.py         executes the pipeline; writes results/<date>-<version>.json
  compare_baseline.py      diffs results vs baseline.json; non-zero exit on regression
  baseline.json            committed baseline metrics (bump deliberately)
  results/                 per-run JSON outputs (gitignored)
```

## Usage

Run inside the podscripter Docker container, with model caches mounted (same as
[`tests/README.md`](../README.md)):

```bash
docker run --rm \
  -v $(pwd):/app \
  -v $(pwd)/models/sentence-transformers:/root/.cache/torch/sentence_transformers \
  -v $(pwd)/models/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN \
  podscripter bash -c "
    python tests/benchmarks/download_subsets.py --langs en,fr &&
    python tests/benchmarks/run_benchmark.py --langs en,fr --output tests/benchmarks/results/$(date +%Y-%m-%d).json &&
    python tests/benchmarks/compare_baseline.py tests/benchmarks/results/$(date +%Y-%m-%d).json
  "
```

The cache directory defaults to `~/.cache/podscripter-bench/` and is reused across runs.

## Datasets used

Tier 2 may use any of the Tier 1 source corpora plus additional permissive datasets:

- **VoxConverse dev** (CC-BY 4.0) — EN multi-speaker
- **AMI Headset Mix** (CC-BY 4.0) — EN multi-speaker, longer recordings
- **MLS French** (CC-BY 4.0) — FR single-speaker audiobooks
- **VoxPopuli FR** (CC0) — FR multi-speaker parliament debates
- **Common Voice EN/FR** (CC0) — short utterances

`download_subsets.py` downloads each dataset into the local cache; it does **not** push to
HuggingFace. The Tier 1 corpus (`podscripter-project/test-fixtures`) is the only thing
podscripter republishes.

## Modes per dataset

- Multi-speaker datasets (VoxConverse, AMI, VoxPopuli) — run in both `single` and `chunked` modes.
- Single-speaker datasets (LibriSpeech, MLS, Common Voice) — `single` only.

## Baseline & regression detection

`compare_baseline.py` reads two JSON files (current vs baseline) and reports per-mode deltas.
A regression is flagged when:

- WER worsens by more than `--wer-tolerance` (default `0.02`, i.e., 2 percentage points), or
- DER worsens by more than `--der-tolerance` (default `0.03`), or
- A previously-passing fixture is now missing from the results.

The exit code is non-zero when any regression is found, so this script is CI-friendly.

When intentionally accepting drift (e.g., after a model upgrade), update `baseline.json` in
the same PR with the metrics that justify the change.
