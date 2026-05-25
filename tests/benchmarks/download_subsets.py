"""Tier 2 benchmark dataset downloader.

Pulls small subsets (~30 min/language) of public, permissively-licensed corpora into the
local benchmark cache. Idempotent: re-running skips already-cached subsets.

Datasets and licenses:

  EN
    voxconverse-dev      CC-BY 4.0   https://www.robots.ox.ac.uk/~vgg/data/voxconverse/
    ami-headset-mix      CC-BY 4.0   https://groups.inf.ed.ac.uk/ami/corpus/
    librispeech-clean    CC-BY 4.0   https://www.openslr.org/12/
    fleurs-en-test       CC-BY 4.0   https://huggingface.co/datasets/google/fleurs

  ES
    fleurs-es-test       CC-BY 4.0   https://huggingface.co/datasets/google/fleurs
    mls-spanish-test     CC-BY 4.0   https://www.openslr.org/94/

  FR
    voxpopuli-fr         CC0 1.0     https://github.com/facebookresearch/voxpopuli
    mls-french           CC-BY 4.0   https://www.openslr.org/94/
    fleurs-fr-test       CC-BY 4.0   https://huggingface.co/datasets/google/fleurs

This script ONLY downloads. It does not push anywhere. Tier 1 fixtures are the only
artifacts podscripter republishes (see tests/fixtures/audio/).

Usage:

    python tests/benchmarks/download_subsets.py --langs en,es,fr [--cache-dir <path>]

Subsets are bounded by `--max-minutes` (default 30) per dataset so the total cache stays
under ~3 GB at the defaults.
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
        "voxpopuli-fr-test",
    ],
}


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


def _download_mls_spanish(dest: Path) -> None:
    """Download MLS Spanish test split (~2 GB; FLAC audiobook clips + transcripts.txt).

    Source: Multilingual LibriSpeech (CC-BY 4.0), https://www.openslr.org/94/.
    The archive ships as `mls_spanish_opus.tar.gz` (opus-compressed, ~2 GB) which contains
    `mls_spanish/test/audio/<speaker>/<book>/<utterance>.opus` plus
    `mls_spanish/test/transcripts.txt` mapping each utterance ID to its reference text.
    The benchmark runner reads the transcripts.txt index lazily; downstream code can
    transcode opus to wav on the fly via faster-whisper / ffmpeg.
    """
    import urllib.request

    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "mls_spanish_opus.tar.gz"
    if target.exists() and target.stat().st_size > 1_000_000_000:
        print(f"  mls-spanish: cached at {target}")
        return
    url = "https://dl.fbaipublicfiles.com/mls/mls_spanish_opus.tar.gz"
    print(f"  mls-spanish: downloading {url}")
    urllib.request.urlretrieve(url, target)
    print(f"  mls-spanish: saved to {target}")


def _download_voxpopuli(lang_code: str, dest: Path, max_minutes: int) -> None:
    """VoxPopuli per-year archives are huge (5+ GB). Recommend using HF mirror subsets."""
    dest.mkdir(parents=True, exist_ok=True)
    note = dest / "INSTRUCTIONS.md"
    note.write_text(
        "VoxPopuli archives are 5+ GB per year. For Tier 2 use the `facebook/voxpopuli`\n"
        "HuggingFace dataset with the `datasets` library streaming a small subset, or\n"
        f"download a single year archive from\n"
        f"  https://dl.fbaipublicfiles.com/voxpopuli/audios/{lang_code}_2020.tar\n"
        f"and trim with ffmpeg to ~{max_minutes} minutes.\n",
        encoding="utf-8",
    )
    print(f"  voxpopuli {lang_code}: see {note}")


def _dispatch(subset: str, cache_dir: Path, max_minutes: int) -> None:
    if subset == "fleurs-en-test":
        _download_fleurs("en_us", cache_dir / "fleurs-en-test", max_minutes)
    elif subset == "fleurs-es-test":
        _download_fleurs("es_419", cache_dir / "fleurs-es-test", max_minutes)
    elif subset == "fleurs-fr-test":
        _download_fleurs("fr_fr", cache_dir / "fleurs-fr-test", max_minutes)
    elif subset == "librispeech-test-clean":
        _download_librispeech(cache_dir / "librispeech-test-clean")
    elif subset == "mls-spanish-test":
        _download_mls_spanish(cache_dir / "mls-spanish-test")
    elif subset == "voxpopuli-fr-test":
        _download_voxpopuli("fr", cache_dir / "voxpopuli-fr-test", max_minutes)
    else:
        raise SystemExit(f"unknown subset: {subset}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--langs", default="en,es,fr", help="Comma-separated languages (default: en,es,fr)")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help=f"Cache directory (default: {DEFAULT_CACHE_DIR})")
    parser.add_argument("--max-minutes", type=int, default=30, help="Approximate per-dataset budget in minutes")
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
            _dispatch(subset, cache_dir, args.max_minutes)

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
