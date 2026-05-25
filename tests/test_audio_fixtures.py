"""End-to-end audio fixture tests for the podscripter pipeline (Tier 1 regression).

Discovers every `tests/fixtures/audio/<lang>/<name>.expected.json`, downloads the matching
audio from the public HuggingFace dataset `podscripter-project/test-fixtures` (pinned by
revision in `tests/fixtures/audio/download.py`), runs the full pipeline via
`podscripter.transcribe(...)`, and asserts WER + DER thresholds plus per-pattern checks.

Each fixture is parametrized over its declared `modes` field (default `["single"]`).
Long fixtures that include both `"single"` and `"chunked"` generate two test invocations.

Pipeline configuration mirrors real-usage flags:
    enable_diarization=True, model_name="medium", beam_size=3, compute_type="auto"

These tests are gated by `@pytest.mark.transcription` and are NOT run by `pytest` by
default. Use `pytest -m transcription` (or `pytest -m ''`) to opt in.

Environment overrides for faster local iteration:
    PODSCRIPTER_TEST_MODEL          override "medium" (e.g. "small", "tiny") for dev speed
    PODSCRIPTER_TEST_FIXTURES_PATTERN  glob filter (e.g. "en/*short*") to run a subset

Determinism: seeds torch+numpy before each pipeline call. WER/DER thresholds are loose
(per-fixture defaults around 0.15/0.20) to absorb pyannote/Whisper non-determinism while
still catching meaningful regressions.
"""

from __future__ import annotations

import fnmatch
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

# tests/conftest.py already prepends the repo root to sys.path. Keep an extra guard for
# direct invocations (e.g. `python -m pytest tests/test_audio_fixtures.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytestmark = pytest.mark.transcription


FIXTURES_AUDIO_DIR = Path(__file__).parent / "fixtures" / "audio"

DEFAULT_MODEL = "medium"
DEFAULT_BEAM_SIZE = 3
DEFAULT_COMPUTE_TYPE = "auto"

# Loose deterministic seeds used before every pipeline call.
_DETERMINISTIC_SEED = 1729


# ---------------------------------------------------------------------------
# Fixture discovery and parametrization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixtureCase:
    path: Path
    mode: str

    @property
    def id(self) -> str:
        return f"{self.path.parent.name}/{self.path.stem}[{self.mode}]"


def _expected_json_files() -> list[Path]:
    return sorted(FIXTURES_AUDIO_DIR.glob("*/*.expected.json"))


def _matches_pattern_filter(path: Path) -> bool:
    pattern = os.environ.get("PODSCRIPTER_TEST_FIXTURES_PATTERN", "").strip()
    if not pattern:
        return True
    relative = f"{path.parent.name}/{path.stem.removesuffix('.expected')}"
    return fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(path.name, pattern)


def _discover_cases() -> list[FixtureCase]:
    cases: list[FixtureCase] = []
    for fixture_path in _expected_json_files():
        if not _matches_pattern_filter(fixture_path):
            continue
        try:
            data = json.loads(fixture_path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - validator catches malformed JSON
            continue
        modes = data.get("modes") or ["single"]
        for mode in modes:
            if mode in ("single", "chunked"):
                cases.append(FixtureCase(path=fixture_path, mode=mode))
    return cases


# ---------------------------------------------------------------------------
# Helpers (audio resolution, thresholds, determinism, metrics)
# ---------------------------------------------------------------------------


def _resolve_audio(audio_file: str) -> Path:
    """Download (or reuse cached) audio from the HF fixtures dataset."""
    from tests.fixtures.audio.download import resolve_audio_path

    return resolve_audio_path(audio_file)


def _thresholds_for_mode(data: dict[str, Any], mode: str) -> dict[str, float]:
    """Return effective thresholds for `mode`. Supports flat or per-mode shape."""
    thresholds = data["thresholds"]
    if "wer_max" in thresholds or "der_max" in thresholds:
        return thresholds
    if mode in thresholds:
        return thresholds[mode]
    raise KeyError(f"thresholds entry for mode={mode!r} not found in fixture")


def _seed_determinism() -> None:
    """Seed pyannote / torch / numpy before each pipeline invocation."""
    import random

    random.seed(_DETERMINISTIC_SEED)
    try:
        import numpy as np  # type: ignore

        np.random.seed(_DETERMINISTIC_SEED)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(_DETERMINISTIC_SEED)
        if torch.cuda.is_available():  # pragma: no cover - CPU test env
            torch.cuda.manual_seed_all(_DETERMINISTIC_SEED)
    except ImportError:  # pragma: no cover
        pass


def _effective_model() -> str:
    return os.environ.get("PODSCRIPTER_TEST_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _normalize_text_for_wer(text: str) -> str:
    """Lowercase and strip punctuation for WER comparison, preserving word boundaries."""
    import re

    text = text.lower()
    text = re.sub(r"[^\w\s'’\-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _compute_wer(reference: str, hypothesis: str) -> float:
    import jiwer  # type: ignore

    ref = _normalize_text_for_wer(reference)
    hyp = _normalize_text_for_wer(hypothesis)
    if not ref:
        return 1.0 if hyp else 0.0
    return float(jiwer.wer(ref, hyp))


def _compute_der(
    reference_turns: list[dict[str, Any]],
    diarization_result: dict[str, Any] | None,
    duration_sec: float,
) -> float | None:
    """Compute DER from speaker_turns (reference) vs pipeline diarization output.

    Returns None if diarization didn't run or pyannote.metrics is unavailable.
    """
    if not diarization_result:
        return None
    try:
        from pyannote.core import Annotation, Segment  # type: ignore
        from pyannote.metrics.diarization import DiarizationErrorRate  # type: ignore
    except ImportError:  # pragma: no cover
        return None

    reference = Annotation()
    for turn in reference_turns:
        reference[Segment(float(turn["start"]), float(turn["end"]))] = str(turn["speaker"])

    hypothesis = Annotation()
    raw_segments = diarization_result.get("raw_segments") or []
    if not raw_segments:
        return None
    for i, seg in enumerate(raw_segments):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if end <= start:
            continue
        speaker = str(seg.get("speaker", f"S{i}"))
        hypothesis[Segment(start, end)] = speaker

    metric = DiarizationErrorRate(collar=0.25, skip_overlap=True)
    total = Segment(0.0, float(duration_sec))
    return float(metric(reference, hypothesis, uem=total))


def _assert_pattern_checks(patterns: list[str], output_text: str) -> None:
    """Apply per-pattern assertions on the produced text."""
    import re

    pattern_set = set(patterns or [])
    if "questions" in pattern_set:
        assert "?" in output_text, "Pattern 'questions' requires '?' in output"
    if "spanish-questions" in pattern_set:
        assert "¿" in output_text and "?" in output_text, (
            "Pattern 'spanish-questions' requires both '¿' and '?' in output"
        )
    if "url" in pattern_set:
        # Crude check: no " . tld" split (e.g. "github . com").
        bad = re.search(
            r"\.\s+(com|net|org|co|io|edu|gov|fr|de|es|uk|us)\b",
            output_text,
            flags=re.IGNORECASE,
        )
        assert not bad, f"Pattern 'url' violated: found split TLD at {bad.group(0)!r}"


# ---------------------------------------------------------------------------
# The parametrized test itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _discover_cases(), ids=lambda c: c.id)
def test_audio_fixture(case: FixtureCase) -> None:
    """Run the full pipeline on one fixture+mode and assert WER/DER/pattern thresholds."""
    if not _discover_cases():  # pragma: no cover
        pytest.skip("No audio fixtures available")

    pytest.importorskip("jiwer", reason="jiwer is required for audio fixture WER assertions")

    data = json.loads(case.path.read_text(encoding="utf-8"))
    language = data["language"]
    expected_text = data["expected_text"]
    expected_speaker_count = int(data["expected_speaker_count"])
    duration_sec = float(data["duration_sec"])
    patterns = data.get("patterns") or []
    thresholds = _thresholds_for_mode(data, case.mode)

    audio_path = _resolve_audio(data["audio_file"])

    _seed_determinism()

    from podscripter import transcribe

    with tempfile.TemporaryDirectory() as tmp:
        result = transcribe(
            str(audio_path),
            output_format="txt",
            language=language,
            single_call=(case.mode == "single"),
            model_name=_effective_model(),
            compute_type=DEFAULT_COMPUTE_TYPE,
            beam_size=DEFAULT_BEAM_SIZE,
            vad_filter=True,
            write_output=False,
            output_dir=tmp,
            enable_diarization=True,
            hf_token=os.environ.get("HF_TOKEN"),
            quiet=True,
        )

    sentences = result.get("sentences") or []
    actual_text = " ".join(s.strip() for s in sentences if s and s.strip())

    wer_max = float(thresholds["wer_max"])
    wer = _compute_wer(expected_text, actual_text)
    assert wer <= wer_max, (
        f"{case.id}: WER={wer:.3f} exceeds threshold {wer_max:.3f}. "
        f"Expected (start): {expected_text[:80]!r}; Actual (start): {actual_text[:80]!r}"
    )

    if expected_speaker_count > 1 and "der_max" in thresholds:
        der = _compute_der(
            reference_turns=data.get("speaker_turns") or [],
            diarization_result=result.get("diarization_result"),
            duration_sec=duration_sec,
        )
        if der is not None:
            der_max = float(thresholds["der_max"])
            assert der <= der_max, (
                f"{case.id}: DER={der:.3f} exceeds threshold {der_max:.3f}"
            )

    _assert_pattern_checks(patterns, actual_text)
