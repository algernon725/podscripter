# podscripter quality benchmarks (Tier 2)

A broad, aggregate WER benchmark that detects transcription **drift** over time. Unlike the
Tier 1 audio fixtures (see [`tests/fixtures/audio/`](../fixtures/audio/) and
[`tests/test_audio_fixtures.py`](../test_audio_fixtures.py)), which are per-clip pass/fail
regression gates run on every PR, Tier 2 is **not** part of the pytest suite and is **not**
wired into CI. It is a **manual, ad-hoc** check you run deliberately after a change that could
move transcription quality (a pipeline change, a dependency bump, or a Whisper model upgrade).

What it does:

- Pulls a bounded subset of public, permissively-licensed audio per language.
- Runs the full podscripter pipeline (production `medium` model by default) on each clip.
- Computes **WER** against the dataset's reference transcripts.
- Compares the per-clip WER against a committed [`baseline.json`](baseline.json) and flags any
  clip that regresses beyond tolerance.

> **WER-only by design.** All Tier 2 datasets below are single-speaker read/prepared speech, so
> the runner emits WER (no DER). Short-span diarization (DER) is covered by the Tier 1
> multi-speaker fixtures instead. `compare_baseline.py` will compare a `der` field if both sides
> have one, so DER datasets can be added later without changing the tooling.

## Layout

```
tests/benchmarks/
  README.md                this file
  download_subsets.py      pulls/streams public dataset subsets into a cache dir
  run_benchmark.py         runs the pipeline over the cache; writes a results JSON
  compare_baseline.py      diffs a results JSON vs baseline.json; non-zero exit on regression
  baseline.json            committed WER baseline (bump deliberately — see below)
```

## Datasets used

The active Tier 2 set (what `download_subsets.py` actually fetches):

| Language | Datasets | License | Notes |
| --- | --- | --- | --- |
| EN | FLEURS `en_us` | CC-BY 4.0 | short single-speaker utterances |
| ES | FLEURS `es_419`, MLS Spanish | CC-BY 4.0 | FLEURS utterances + audiobook clips |
| FR | FLEURS `fr_fr`, MLS French | CC-BY 4.0 | FLEURS utterances + audiobook clips |

- **FLEURS** is pulled as its per-language test archive via the HuggingFace Hub and extracted
  into the cache on first use.
- **MLS** (`facebook/multilingual_librispeech`) is **streamed** clip-by-clip from the HF test
  split (raw `.opus` + a `transcripts.tsv` index), bounded by `--mls-max-items`, so its
  footprint is tens of MB rather than the multi-GB monolithic archive.

`download_subsets.py` only downloads into the local cache; it does **not** push anywhere. The
Tier 1 corpus (`podscripter-project/test-fixtures`) is the only artifact podscripter
republishes.

## Usage

Run inside the podscripter Docker container with model caches mounted (same mounts as
[`tests/README.md`](../README.md)). `HF_TOKEN` is required for the dataset pulls. Use a cache
dir on the host (e.g. a bind-mounted `/cache`) so the download is reused across runs:

```bash
docker run --rm \
  -e HF_TOKEN \
  -v $(pwd):/app -w /app \
  -v $(pwd)/.bench-cache:/cache \
  -v $(pwd)/models/huggingface:/root/.cache/huggingface \
  -v $(pwd)/models/sentence-transformers:/root/.cache/torch/sentence_transformers \
  podscripter bash -c "
    python tests/benchmarks/download_subsets.py --langs en,es,fr --cache-dir /cache --mls-max-items 30 &&
    python tests/benchmarks/run_benchmark.py    --langs en,es,fr --cache-dir /cache --max-items-per-dataset 15 --output /cache/run.json &&
    python tests/benchmarks/compare_baseline.py /cache/run.json
  "
```

Key flags:

- `download_subsets.py`: `--langs`, `--cache-dir`, `--max-minutes` (FLEURS clip budget),
  `--mls-max-items` (MLS clips streamed per language; keep `>=` the runner's
  `--max-items-per-dataset`).
- `run_benchmark.py`: `--langs`, `--cache-dir`, `--model` (default `medium`),
  `--max-items-per-dataset` (default `20`), `--output` (required).
- `compare_baseline.py`: positional `current` results JSON, `--baseline`, `--wer-tolerance`
  (default `0.02`), `--der-tolerance` (default `0.03`).

## Baseline & regression detection

`compare_baseline.py` reads two JSON files (current vs baseline) and reports per-clip deltas. A
**regression** (non-zero exit) is flagged when:

- A clip's WER worsens by more than `--wer-tolerance` (default `0.02`, i.e. 2 points), or
- a clip's DER worsens by more than `--der-tolerance` (default `0.03`) when present, or
- a clip in the baseline is **missing** from the current run.

A clip that *improves* beyond tolerance is reported as a (non-failing) hint to refresh the
baseline.

### How the committed baseline was produced

Decoding is greedy and the model is fixed, so the pipeline's WER is **deterministic** — running
the benchmark twice on a clean image yields bit-identical per-clip WER. The committed
[`baseline.json`](baseline.json) is the second of two such runs (`medium` model, `--langs
en,es,fr`, `--max-items-per-dataset 15` → 75 clips: 15 each of FLEURS en/es/fr + MLS es/fr),
confirmed identical to the first run.

### Refreshing the baseline (accepting drift)

When a change intentionally moves WER (e.g. a model upgrade), regenerate and re-commit
`baseline.json` in the **same** change once the new numbers are reviewed and accepted: run the
benchmark, sanity-check the deltas with `compare_baseline.py` against the old baseline, then
overwrite `baseline.json` with the new results (preserving `_comment`). Because there is no
nightly job, the baseline only ever moves by deliberate, reviewed commits.
