"""Tier 2 benchmark dataset downloader.

Pulls small subsets of public, permissively-licensed corpora into the local benchmark
cache. Idempotent: re-running skips already-cached subsets.

Datasets, licenses, and approximate download footprint at the defaults:

  EN
    librispeech-test-clean  CC-BY 4.0  https://www.openslr.org/12/                  (~346 MB whole archive)
    fleurs-en-test          CC-BY 4.0  https://huggingface.co/datasets/google/fleurs (~1.3 GB test archive + tsv)

  ES
    fleurs-es-test          CC-BY 4.0  https://huggingface.co/datasets/google/fleurs (~1.3 GB test archive + tsv)
    mls-spanish-test        CC-BY 4.0  facebook/multilingual_librispeech (streamed)  (~tens of MB; `--mls-max-items` clips)

  FR
    fleurs-fr-test          CC-BY 4.0  https://huggingface.co/datasets/google/fleurs (~1.3 GB test archive + tsv)
    mls-french-test         CC-BY 4.0  facebook/multilingual_librispeech (streamed)  (~tens of MB; `--mls-max-items` clips)
    (voxpopuli-fr removed: raw archives are 5 GB+ per year and the multi-speaker FR slot
     is covered by Tier 1; MLS French gives the FR Tier 2 WER signal at a tiny footprint.)

MLS is *streamed* from the HuggingFace `facebook/multilingual_librispeech` test split
(`--mls-max-items` clips per language, written as raw `.opus` + a `transcripts.tsv` index),
NOT downloaded as the old ~2 GB+ monolithic `mls_<lang>_opus.tar.gz` archive. FLEURS still
pulls its single per-language test archive via HF Hub. So the dominant footprint is FLEURS
(~1.3 GB per language) and LibriSpeech (~346 MB), not MLS.

This script ONLY downloads. It does not push anywhere. Tier 1 fixtures are the only
artifacts podscripter republishes (see tests/fixtures/audio/).

Usage:

    python tests/benchmarks/download_subsets.py --langs en,es,fr [--cache-dir <path>] [--mls-max-items 30]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_CACHE_DIR = Path(os.environ.get("PODSCRIPTER_BENCH_CACHE", "~/.cache/podscripter-bench")).expanduser()

# Registry of dataset subsets. Each entry describes how to fetch a small slice.
# Implementations live below; this registry is the single source of truth.
DATASETS = {
    "en": [
        "fleurs-en-test",
        "librispeech-test-clean",
    ],
    "es": [
        "fleurs-es-test",
        "mls-spanish-test",
    ],
    "fr": [
        "fleurs-fr-test",
        "mls-french-test",
    ],
}

# MLS HF config name per podscripter language code.
MLS_CONFIGS = {"es": "spanish", "fr": "french"}


def _download_fleurs(lang_code: str, dest: Path, max_minutes: int) -> None:
    """Download a FLEURS test subset via HF Hub (single ~350 MB archive per language)."""
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    archive = hf_hub_download(
        repo_id="google/fleurs",
        filename=f"data/{lang_code}/audio/test.tar.gz",
        repo_type="dataset",
        local_dir=str(dest),
    )
    tsv = hf_hub_download(
        repo_id="google/fleurs",
        filename=f"data/{lang_code}/test.tsv",
        repo_type="dataset",
        local_dir=str(dest),
    )
    print(f"  fleurs {lang_code}: archive={archive}, tsv={tsv}")
    _ = max_minutes  # respected at run-time, not download-time


def _download_librispeech(dest: Path) -> None:
    """Download LibriSpeech test-clean (~346 MB)."""
    import urllib.request

    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "test-clean.tar.gz"
    if target.exists() and target.stat().st_size > 100_000_000:
        print(f"  librispeech: cached at {target}")
        return
    url = "https://www.openslr.org/resources/12/test-clean.tar.gz"
    print(f"  librispeech: downloading {url}")
    urllib.request.urlretrieve(url, target)
    print(f"  librispeech: saved to {target}")


def _stream_mls(lang: str, dest: Path, max_items: int) -> None:
    """Stream the first `max_items` clips of the MLS test split for `lang` into `dest`.

    Replaces the old ~2 GB+ monolithic `mls_<lang>_opus.tar.gz` download. Streams
    `facebook/multilingual_librispeech` (CC-BY 4.0, https://www.openslr.org/94/) test split
    with `datasets` (bounded by `islice(max_items)`), writing each clip's raw `.opus` bytes
    to `<dest>/audio/<utt_id>.opus` and an index line to `<dest>/transcripts.tsv`
    (`<utt_id>\t<reference text>`). Footprint is tens of MB, not gigabytes, and scales flat
    as more languages are added (just register them in `MLS_CONFIGS` + `DATASETS`).

    Idempotent: skips if `transcripts.tsv` already has >= `max_items` rows. Audio decoding
    is deferred to run-time (the runner feeds `.opus` straight to the pipeline), so this
    streams with `Audio(decode=False)` to avoid pulling in torchcodec here.
    """
    from itertools import islice

    config = MLS_CONFIGS.get(lang)
    if not config:
        raise SystemExit(f"no MLS config registered for lang={lang!r}")

    transcripts = dest / "transcripts.tsv"
    if transcripts.exists() and len(transcripts.read_text(encoding="utf-8").splitlines()) >= max_items:
        print(f"  mls-{lang}: cached ({max_items} clips) at {dest}")
        return

    from datasets import Audio, load_dataset

    audio_dir = dest / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    print(f"  mls-{lang}: streaming first {max_items} clips of facebook/multilingual_librispeech[{config}] test ...")
    ds = load_dataset("facebook/multilingual_librispeech", config, split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    lines: list[str] = []
    for ex in islice(ds, max_items):
        audio = ex["audio"]
        name = Path(audio["path"]).name  # e.g. 11266_10604_000000.opus
        utt_id = Path(name).stem
        (audio_dir / f"{utt_id}.opus").write_bytes(audio["bytes"])
        text = " ".join(str(ex["transcript"]).split())
        lines.append(f"{utt_id}\t{text}")

    transcripts.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  mls-{lang}: wrote {len(lines)} clips + {transcripts}")


def _dispatch(subset: str, cache_dir: Path, max_minutes: int, mls_max_items: int) -> None:
    if subset == "fleurs-en-test":
        _download_fleurs("en_us", cache_dir / "fleurs-en-test", max_minutes)
    elif subset == "fleurs-es-test":
        _download_fleurs("es_419", cache_dir / "fleurs-es-test", max_minutes)
    elif subset == "fleurs-fr-test":
        _download_fleurs("fr_fr", cache_dir / "fleurs-fr-test", max_minutes)
    elif subset == "librispeech-test-clean":
        _download_librispeech(cache_dir / "librispeech-test-clean")
    elif subset == "mls-spanish-test":
        _stream_mls("es", cache_dir / "mls-spanish-test", mls_max_items)
    elif subset == "mls-french-test":
        _stream_mls("fr", cache_dir / "mls-french-test", mls_max_items)
    else:
        raise SystemExit(f"unknown subset: {subset}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--langs", default="en,es,fr", help="Comma-separated languages (default: en,es,fr)")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help=f"Cache directory (default: {DEFAULT_CACHE_DIR})")
    parser.add_argument("--max-minutes", type=int, default=30, help="Approximate per-dataset budget in minutes (FLEURS run-time clip)")
    parser.add_argument("--mls-max-items", type=int, default=30, help="MLS clips to stream per language (>= run_benchmark --max-items-per-dataset)")
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"cache dir: {cache_dir}")

    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    for lang in langs:
        subsets = DATASETS.get(lang, [])
        if not subsets:
            print(f"WARNING: no benchmark subsets registered for lang={lang!r}; skipping")
            continue
        print(f"== {lang} ==")
        for subset in subsets:
            _dispatch(subset, cache_dir, args.max_minutes, args.mls_max_items)

    return 0


if __name__ == "__main__":  # pragma: no cover
    rc = main()
    # `datasets` streaming + torchcodec/aiohttp can segfault at interpreter teardown
    # (PyGILState_Release) even after all work + file writes are complete. Flush and use
    # os._exit so a successful download reports a clean exit code instead of 139.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
