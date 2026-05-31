"""Tier 2 benchmark runner.

Executes the podscripter pipeline over the cached benchmark subsets and writes a JSON
results file keyed by `(dataset, file, mode)`. Multi-speaker subsets run in BOTH `--single`
and chunked modes so chunked-specific regressions surface separately in
`compare_baseline.py`.

Metrics emitted per (dataset, file, mode):
    wer            jiwer-computed Word Error Rate vs ground truth
    der            pyannote.metrics Diarization Error Rate vs ground-truth turns (if any)
    sentence_f1    sentence-boundary F1 vs ground-truth sentence list (if any)
    elapsed_secs   wall-clock pipeline time

Usage:

    python tests/benchmarks/run_benchmark.py --langs en,es,fr \
        --output tests/benchmarks/results/2026-05-25.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_CACHE_DIR = Path(
    os.environ.get("PODSCRIPTER_BENCH_CACHE", "~/.cache/podscripter-bench")
).expanduser()


def _enumerate_items(cache_dir: Path, langs: list[str]) -> Iterable[dict[str, Any]]:
    """Yield benchmark items as dicts: dataset, lang, audio_path, expected_text, ...

    Recognizes:
      - FLEURS layout (`<dataset>/data/<lang>/test.tsv` + `test.tar.gz`) for en, es, fr.
      - MLS test-split streamed cache (`mls-<lang>-test/transcripts.tsv` +
        `mls-<lang>-test/audio/<utt_id>.opus`) for es + fr, produced by
        `download_subsets.py` streaming `facebook/multilingual_librispeech` (no monolithic
        tarball download).

    Extend with additional sources as they're integrated.
    """
    import tarfile

    fleurs_lang_codes = {"en": "en_us", "es": "es_419", "fr": "fr_fr"}
    for lang in langs:
        fleurs_lang_code = fleurs_lang_codes.get(lang)
        if not fleurs_lang_code:
            continue
        fleurs_root = cache_dir / f"fleurs-{lang}-test"
        tsv = fleurs_root / "data" / fleurs_lang_code / "test.tsv"
        tar = fleurs_root / "data" / fleurs_lang_code / "audio" / "test.tar.gz"
        if not (tsv.exists() and tar.exists()):
            continue

        extract_dir = fleurs_root / "data" / fleurs_lang_code / "audio" / "test"
        if not extract_dir.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar, "r:gz") as tf:
                tf.extractall(extract_dir)

        for line in tsv.read_text(encoding="utf-8").splitlines():
            cols = line.split("\t")
            if len(cols) < 3:
                continue
            audio_file = extract_dir / "test" / cols[1]
            if not audio_file.exists():
                continue
            yield {
                "dataset": f"fleurs-{lang}",
                "lang": lang,
                "audio_path": str(audio_file),
                "expected_text": cols[2],
                "expected_speakers": 1,
                "modes": ["single"],
            }

    if "es" in langs:
        yield from _enumerate_mls(cache_dir / "mls-spanish-test", lang="es", tag="mls-spanish")
    if "fr" in langs:
        yield from _enumerate_mls(cache_dir / "mls-french-test", lang="fr", tag="mls-french")


def _enumerate_mls(mls_root: Path, *, lang: str, tag: str) -> Iterable[dict[str, Any]]:
    """Yield MLS test items from the streamed local cache produced by `download_subsets.py`.

    Layout (written by `download_subsets._stream_mls`, which streams the test split of
    `facebook/multilingual_librispeech` bounded by `--mls-max-items` — no monolithic
    tarball, footprint is tens of MB per language):
        <mls_root>/transcripts.tsv     tab-separated: <utt_id>\t<reference text>
        <mls_root>/audio/<utt_id>.opus

    `<utt_id>` is `<speaker>_<book>_<utt>`. The runner reads this index lazily and feeds
    each `.opus` directly to the pipeline (faster-whisper decodes via ffmpeg/av).
    Single-speaker audiobook clips → modes=["single"], expected_speakers=1.

    Identical for every MLS language: pass `lang`/`tag` and the per-language cache dir. To
    add a language, register it in `download_subsets.DATASETS` and call this with the new
    cache dir — no per-language enumeration code is needed.
    """
    transcripts = mls_root / "transcripts.tsv"
    audio_root = mls_root / "audio"
    if not transcripts.exists():
        return
    for line in transcripts.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        utt_id, text = parts
        audio_file = audio_root / f"{utt_id}.opus"
        if not audio_file.exists():
            continue
        yield {
            "dataset": tag,
            "lang": lang,
            "audio_path": str(audio_file),
            "expected_text": text,
            "expected_speakers": 1,
            "modes": ["single"],
        }


def _normalize(text: str) -> str:
    import re

    text = text.lower()
    text = re.sub(r"[^\w\s'’\-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _run_one(item: dict[str, Any], mode: str, model_name: str) -> dict[str, Any]:
    import jiwer  # type: ignore

    from podscripter import transcribe

    t0 = time.time()
    result = transcribe(
        item["audio_path"],
        output_format="txt",
        language=item["lang"],
        single_call=(mode == "single"),
        model_name=model_name,
        beam_size=3,
        vad_filter=True,
        write_output=False,
        enable_diarization=(item.get("expected_speakers", 1) > 1),
        hf_token=os.environ.get("HF_TOKEN"),
        quiet=True,
    )
    elapsed = time.time() - t0

    # result["sentences"] is list[Sentence] (since the v0.6.0 Speaker-Aware Output refactor),
    # not list[str]; extract .text (str() fallback keeps old list[str] callers working).
    sentences = result.get("sentences") or []
    actual = " ".join(
        text
        for text in ((s.text if hasattr(s, "text") else str(s)).strip() for s in sentences if s)
        if text
    )
    wer = float(jiwer.wer(_normalize(item["expected_text"]), _normalize(actual)))
    return {
        "wer": round(wer, 4),
        "elapsed_secs": round(elapsed, 2),
        "actual_chars": len(actual),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--langs", default="en,es,fr", help="Comma-separated languages (default: en,es,fr)")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--model", default=os.environ.get("PODSCRIPTER_TEST_MODEL", "medium"))
    parser.add_argument("--max-items-per-dataset", type=int, default=20, help="Limit per dataset for speed")
    parser.add_argument("--output", required=True, help="Path for results JSON")
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache_dir).expanduser()
    langs = [s.strip() for s in args.langs.split(",") if s.strip()]

    results: dict[str, Any] = {
        "model": args.model,
        "results": {},
    }

    seen_per_dataset: dict[str, int] = {}
    for item in _enumerate_items(cache_dir, langs):
        seen_per_dataset[item["dataset"]] = seen_per_dataset.get(item["dataset"], 0) + 1
        if seen_per_dataset[item["dataset"]] > args.max_items_per_dataset:
            continue
        for mode in item["modes"]:
            key = f"{item['dataset']}|{Path(item['audio_path']).name}|{mode}"
            print(f"running {key} ...", flush=True)
            try:
                metrics = _run_one(item, mode, args.model)
            except Exception as exc:  # pragma: no cover
                metrics = {"error": str(exc)}
            results["results"][key] = metrics

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
