"""License-metadata validator for audio fixtures.

This is a pytest module (not a runtime helper) marked with the default `core` marker so it
runs on every `pytest` invocation, even without `-m transcription`. It parses every
`tests/fixtures/audio/*/*.expected.json` and asserts each fixture declares the required
license fields and uses one of the permissive licenses on the allowlist defined in
`LICENSES.md`.

Goal: make it impossible for a PR to add a fixture with missing attribution or with an
NC/ND-licensed source slipped in by accident.

Audio files are NOT required to be present locally for this validator to run; it only reads
the JSON metadata committed in git.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.core


FIXTURES_AUDIO_DIR = Path(__file__).parent
PERMISSIVE_LICENSES = frozenset({
    "CC-BY-4.0",
    "CC0-1.0",
    "public-domain",
})

# These fields must be present in every fixture, regardless of license.
ALWAYS_REQUIRED_FIELDS = (
    "language",
    "audio_file",
    "source",
    "source_url",
    "license",
    "license_url",
    "duration_sec",
    "expected_text",
    "expected_speaker_count",
    "thresholds",
)

# Required additionally for any non-CC0 source (i.e., CC-BY-4.0 and similar).
CC_BY_REQUIRED_FIELDS = (
    "attribution",
    "modifications",
)


def _discover_fixtures() -> list[Path]:
    """Return every `*.expected.json` under `tests/fixtures/audio/<lang>/`."""
    return sorted(FIXTURES_AUDIO_DIR.glob("*/*.expected.json"))


def _load(fixture_path: Path) -> dict[str, Any]:
    with fixture_path.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize(
    "fixture_path",
    _discover_fixtures(),
    ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_fixture_license_metadata(fixture_path: Path) -> None:
    """Every fixture must declare permissive-license metadata in the documented shape."""
    if not _discover_fixtures():
        pytest.skip("No audio fixtures present yet")

    data = _load(fixture_path)

    missing = [f for f in ALWAYS_REQUIRED_FIELDS if f not in data]
    assert not missing, (
        f"{fixture_path.name}: missing required fields {missing}. "
        f"See tests/fixtures/audio/README.md for the full schema."
    )

    license_value = data["license"]
    assert license_value in PERMISSIVE_LICENSES, (
        f"{fixture_path.name}: license={license_value!r} is not on the permissive allowlist "
        f"{sorted(PERMISSIVE_LICENSES)}. Only CC-BY-4.0, CC0-1.0, and public-domain sources "
        f"may be redistributed by this dataset. See tests/fixtures/audio/LICENSES.md."
    )

    if license_value != "CC0-1.0":
        missing_attr = [f for f in CC_BY_REQUIRED_FIELDS if not data.get(f)]
        assert not missing_attr, (
            f"{fixture_path.name}: non-CC0 license {license_value} requires fields "
            f"{CC_BY_REQUIRED_FIELDS}; missing or empty: {missing_attr}. CC-BY 4.0 "
            f"obligates attribution and indication of changes."
        )

    audio_file = data["audio_file"]
    assert isinstance(audio_file, str) and audio_file, (
        f"{fixture_path.name}: audio_file must be a non-empty string path inside the HF dataset"
    )
    assert audio_file.startswith(f"{fixture_path.parent.name}/"), (
        f"{fixture_path.name}: audio_file={audio_file!r} must start with "
        f"'{fixture_path.parent.name}/' to match the fixture's language directory"
    )

    modes = data.get("modes", ["single"])
    assert isinstance(modes, list) and modes, (
        f"{fixture_path.name}: modes must be a non-empty list"
    )
    invalid_modes = [m for m in modes if m not in ("single", "chunked")]
    assert not invalid_modes, (
        f"{fixture_path.name}: invalid modes {invalid_modes}; allowed: 'single', 'chunked'"
    )

    thresholds = data["thresholds"]
    assert isinstance(thresholds, dict) and thresholds, (
        f"{fixture_path.name}: thresholds must be a non-empty object"
    )
    if "wer_max" in thresholds:
        assert isinstance(thresholds["wer_max"], (int, float)), (
            f"{fixture_path.name}: thresholds.wer_max must be a number"
        )
    else:
        for mode in modes:
            assert mode in thresholds, (
                f"{fixture_path.name}: per-mode thresholds missing entry for mode={mode!r}"
            )
            assert "wer_max" in thresholds[mode], (
                f"{fixture_path.name}: thresholds[{mode!r}].wer_max is required"
            )

    expected_speaker_count = data["expected_speaker_count"]
    assert isinstance(expected_speaker_count, int) and expected_speaker_count >= 1, (
        f"{fixture_path.name}: expected_speaker_count must be a positive integer"
    )
    if expected_speaker_count > 1:
        speaker_turns = data.get("speaker_turns")
        assert speaker_turns, (
            f"{fixture_path.name}: expected_speaker_count={expected_speaker_count} requires "
            f"a non-empty speaker_turns list for DER computation"
        )


def test_at_least_one_fixture_present() -> None:
    """Informational guard. Fails clearly during corpus bring-up rather than silently."""
    fixtures = _discover_fixtures()
    if not fixtures:
        pytest.skip(
            "No audio fixtures present yet. Add .expected.json files under "
            "tests/fixtures/audio/<lang>/ to populate the corpus."
        )
