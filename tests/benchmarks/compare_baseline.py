"""Compare a Tier 2 benchmark run against the committed baseline.

Reports per-mode WER/DER deltas and exits non-zero on any regression beyond tolerance,
so this script is safe to use as a CI gate.

Usage:

    python tests/benchmarks/compare_baseline.py results/<date>.json \
        [--baseline tests/benchmarks/baseline.json] \
        [--wer-tolerance 0.02] [--der-tolerance 0.03]

When intentionally accepting drift (e.g., after a model upgrade), update `baseline.json`
in the same PR.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _compare(
    current: dict[str, Any],
    baseline: dict[str, Any],
    wer_tol: float,
    der_tol: float,
) -> tuple[list[str], list[str]]:
    """Return (warnings, regressions). Non-empty regressions cause non-zero exit."""
    warnings: list[str] = []
    regressions: list[str] = []

    cur_results = current.get("results", {})
    base_results = baseline.get("results", {})

    base_keys = set(base_results.keys())
    cur_keys = set(cur_results.keys())

    missing = sorted(base_keys - cur_keys)
    for key in missing:
        regressions.append(f"MISSING: baseline had {key} but current run does not")

    extra = sorted(cur_keys - base_keys)
    for key in extra:
        warnings.append(f"NEW: {key} not in baseline (consider updating baseline)")

    for key in sorted(base_keys & cur_keys):
        cur = cur_results[key]
        base = base_results[key]

        if "error" in cur and "error" not in base:
            regressions.append(f"ERROR: {key} raised: {cur['error']}")
            continue

        for metric, tol in (("wer", wer_tol), ("der", der_tol)):
            if metric in base and metric in cur:
                delta = float(cur[metric]) - float(base[metric])
                if delta > tol:
                    regressions.append(
                        f"REGRESSION: {key} {metric.upper()} "
                        f"{base[metric]:.3f} -> {cur[metric]:.3f} (delta {delta:+.3f} > tolerance {tol:.3f})"
                    )
                elif delta < -tol:
                    warnings.append(
                        f"IMPROVED: {key} {metric.upper()} "
                        f"{base[metric]:.3f} -> {cur[metric]:.3f} (delta {delta:+.3f}; consider updating baseline)"
                    )

    return warnings, regressions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("current", help="Path to the new results JSON")
    parser.add_argument(
        "--baseline",
        default=str(Path(__file__).parent / "baseline.json"),
        help="Path to the committed baseline JSON",
    )
    parser.add_argument("--wer-tolerance", type=float, default=0.02)
    parser.add_argument("--der-tolerance", type=float, default=0.03)
    args = parser.parse_args(argv)

    current = _load(Path(args.current))
    baseline = _load(Path(args.baseline))

    warnings, regressions = _compare(current, baseline, args.wer_tolerance, args.der_tolerance)

    print(f"compared {len(current.get('results', {}))} current vs {len(baseline.get('results', {}))} baseline entries")
    print(f"  model (current):  {current.get('model', '?')}")
    print(f"  model (baseline): {baseline.get('model', '?')}")

    for line in warnings:
        print(line)
    for line in regressions:
        print(line)

    return 1 if regressions else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
